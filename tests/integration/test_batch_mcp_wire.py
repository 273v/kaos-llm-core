"""Phase 15.3 batch MCP wire tests against the real kaos-llm-core-serve.

Drives the four batch tools (``batch-create``, ``batch-run``,
``batch-status``, ``batch-results``) through real stdio + the real
``mcp.ClientSession`` + a real Anthropic Haiku call.

The whole point of this file: prove that the workspace SQLite
metadata table actually persists batch state across MCP requests
(which is the entire reason the workspace exists). Each tool call is a
separate MCP request — they share no in-memory state. If the workspace
plumbing works, the batch_id minted by ``batch-create`` is recognized
by ``batch-run`` and ``batch-status`` and ``batch-results`` in
subsequent calls.

Hard ``cost cap < $0.10`` enforced by tiny inputs.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

pytestmark = pytest.mark.integration


def _has_key(*env_vars: str) -> bool:
    return any(os.getenv(v) for v in env_vars)


requires_anthropic = pytest.mark.skipif(
    not _has_key("KAOS_LLM_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    reason="No Anthropic API key",
)


HAIKU = "claude-haiku-4-5"


def _classify_envelope() -> dict:
    return {
        "kaos_program": "1",
        "name": "batch-wire-classify",
        "inputs": {"text": {"type": "string"}},
        "clients": {"default": {"provider": "anthropic", "model": HAIKU}},
        "steps": [
            {
                "id": "classify",
                "kind": "call",
                "client": "default",
                "instruction": (
                    "Classify the sentiment of the input as positive, negative, or neutral."
                ),
                "inputs": {"text": "$.inputs.text"},
                "output_fields": {
                    "sentiment": {
                        "description": "positive | negative | neutral",
                        "type": {"type": "string"},
                    }
                },
            }
        ],
        "output": {"sentiment": "$.steps.classify.output.sentiment"},
        "capabilities": ["call", "jsonpointer_refs"],
    }


SAMPLE_INPUTS = [
    {"text": "I love this product, it's wonderful!"},
    {"text": "Worst experience of my life. Terrible service."},
    {"text": "It is a chair. It does what a chair does."},
    {"text": "Absolutely fantastic, exceeded all expectations."},
    {"text": "I'm not sure how I feel about this."},
]


def _server_params(vfs_root: Path) -> StdioServerParameters:
    """Build the stdio command for kaos-llm-core-serve, rooted in vfs_root.

    The kaos-core VFS uses ``Path(".kaos-vfs")`` as its disk_base_path
    by default, resolved relative to the process cwd. By launching the
    server with ``cwd=vfs_root``, every disk write (workspace SQLite,
    JSONL log, persisted envelope) lands inside the test's tmp dir,
    isolated from any real ~/.kaos-vfs on the dev machine.
    """
    return StdioServerParameters(
        command="uv",
        args=[
            "run",
            "--directory",
            str(Path(__file__).parent.parent.parent),
            "kaos-llm-core-serve",
        ],
        env=dict(os.environ),
        cwd=str(vfs_root),
    )


def _structured(call_result: object) -> dict:
    sc = getattr(call_result, "structuredContent", None)
    assert isinstance(sc, dict), (
        f"Expected structuredContent dict on tool result; got: {type(sc).__name__} / {sc!r}"
    )
    return sc


def _text(call_result: object) -> str:
    chunks: list[str] = []
    for item in getattr(call_result, "content", []):
        text = getattr(item, "text", None)
        if text:
            chunks.append(text)
    return "\n".join(chunks)


@requires_anthropic
class TestBatchMCPWire:
    """End-to-end MCP wire test for the four Phase 15.3 batch tools."""

    async def test_batch_lifecycle_against_haiku(self, tmp_path: Path) -> None:
        """create → run → status → results across separate MCP calls."""
        async with (
            stdio_client(_server_params(tmp_path)) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()

            # 0. Discover the four batch tools.
            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
            for required in (
                "kaos-llm-core-batch-create",
                "kaos-llm-core-batch-run",
                "kaos-llm-core-batch-status",
                "kaos-llm-core-batch-results",
            ):
                assert required in names, f"{required} missing. All tools: {sorted(names)}"

            # 1. CREATE — pass the envelope inline. batch-create
            #    persists it to ${output_dir}/envelope.json automatically
            #    so subsequent MCP requests (batch-run) can re-load it.
            create = await session.call_tool(
                "kaos-llm-core-batch-create",
                arguments={
                    "envelope": _classify_envelope(),
                    "inputs_source": {
                        "type": "list",
                        "items": SAMPLE_INPUTS,
                    },
                    "output_dir": "runs/wire-batch",
                    "name": "wire-batch",
                },
            )
            assert not create.isError, _text(create)
            create_out = _structured(create)
            batch_id = create_out["batch_id"]
            assert create_out["status"] == "pending"
            assert create_out["inputs_count_hint"] == len(SAMPLE_INPUTS)
            print(f"\n  [batch_wire] created {batch_id}")

            # 3. STATUS (pending)
            status_pending = await session.call_tool(
                "kaos-llm-core-batch-status",
                arguments={"batch_id": batch_id},
            )
            assert not status_pending.isError, _text(status_pending)
            sp = _structured(status_pending)
            assert sp["status"] == "pending"
            assert sp["batch_id"] == batch_id

            # 4. RUN — synchronous, hits real Haiku.
            run = await session.call_tool(
                "kaos-llm-core-batch-run",
                arguments={"batch_id": batch_id},
            )
            assert not run.isError, _text(run)
            run_out = _structured(run)
            assert run_out["status"] == "completed"
            assert run_out["n_succeeded"] == len(SAMPLE_INPUTS)
            assert run_out["n_errored"] == 0
            assert run_out["cost_usd"] > 0
            assert run_out["cost_usd"] < 0.10, f"Cost cap exceeded: ${run_out['cost_usd']:.6f}"
            print(
                f"  [batch_wire] ran: {run_out['n_succeeded']} succeeded, "
                f"${run_out['cost_usd']:.6f}, "
                f"{run_out['duration_s']:.2f}s"
            )

            # 5. STATUS (completed) — must reflect the same SQLite row.
            status_done = await session.call_tool(
                "kaos-llm-core-batch-status",
                arguments={"batch_id": batch_id},
            )
            assert not status_done.isError, _text(status_done)
            sd = _structured(status_done)
            assert sd["status"] == "completed"
            assert sd["n_succeeded"] == len(SAMPLE_INPUTS)
            assert sd["cost_usd_so_far"] > 0
            assert sd["tokens_so_far"]["total"] > 0

            # 6. RESULTS (manifest)
            results = await session.call_tool(
                "kaos-llm-core-batch-results",
                arguments={"batch_id": batch_id, "format": "manifest"},
            )
            assert not results.isError, _text(results)
            ro = _structured(results)
            assert ro["status"] == "completed"
            assert ro["n_succeeded"] == len(SAMPLE_INPUTS)
            assert "manifest" in ro
            assert ro["manifest"]["n_total"] == len(SAMPLE_INPUTS)
            assert ro["manifest"]["status"] == "completed"
            print(
                f"  [batch_wire] manifest: cost=${ro['cost_usd']:.6f}, "
                f"tokens={ro['tokens']['total']}"
            )

    async def test_batch_results_unknown_id(self, tmp_path: Path) -> None:
        """A bogus batch_id returns a clean error, not a stack trace."""
        async with (
            stdio_client(_server_params(tmp_path)) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            result = await session.call_tool(
                "kaos-llm-core-batch-status",
                arguments={"batch_id": "batch-does-not-exist"},
            )
            assert result.isError
            assert "no batch" in _text(result).lower()
