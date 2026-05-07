"""Phase 15.3 batch MCP tool handler tests against the FunctionClient.

End-to-end exercises the four tools through their ``execute()`` API:

  - ``batch-create`` defines a batch (workspace row in pending state)
  - ``batch-run`` actually runs it via the same FunctionClient stub
  - ``batch-status`` reads back the row
  - ``batch-results`` returns the manifest

The whole tool surface is wired up against a per-test
``KaosRuntime`` whose VFS is rooted at ``tmp_path`` so the workspace
SQLite, the saved envelope, and the JSONL log all live in the test's
temp dir.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from kaos_core import KaosContext, KaosRuntime
from kaos_core.vfs.core import VirtualFileSystem
from kaos_core.vfs.models import VFSConfig
from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart

from kaos_llm_core.integrations.mcp.batch_create import KaosLLMCoreBatchCreateTool
from kaos_llm_core.integrations.mcp.batch_results import KaosLLMCoreBatchResultsTool
from kaos_llm_core.integrations.mcp.batch_run import KaosLLMCoreBatchRunTool
from kaos_llm_core.integrations.mcp.batch_status import KaosLLMCoreBatchStatusTool
from kaos_llm_core.observability.cost import PRICING, ModelPricing


@pytest.fixture
def runtime(tmp_path: Path) -> KaosRuntime:
    """KaosRuntime with VFS rooted at tmp_path so all writes are isolated."""
    rt = KaosRuntime()
    rt.vfs = VirtualFileSystem(VFSConfig(disk_base_path=tmp_path))
    return rt


@pytest.fixture
def context(runtime: KaosRuntime) -> KaosContext:
    return KaosContext(session_id="test-batch", runtime=runtime)


@pytest.fixture(autouse=True)
def _pricing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub pricing so batch_run reports a non-zero cost in tests."""
    pricing = ModelPricing(input_per_mtok=1.0, output_per_mtok=2.0)
    monkeypatch.setitem(PRICING, "function-test", pricing)
    # The envelope's client spec produces a "function:function-test" model
    # string, which is what trace.model carries through Call._run_pipeline.
    monkeypatch.setitem(PRICING, "function:function-test", pricing)


def _stub_function_client(answer_template: str = "label-{i}") -> FunctionClient:
    """A deterministic provider stub for the envelope's classify step."""
    counter = {"n": 0}

    def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
        i = counter["n"]
        counter["n"] += 1
        payload = {"answer": answer_template.format(i=i)}
        return ProviderResponse.model_construct(
            provider="function",
            model="function-test",
            raw={},
            parts=[ContentPart.model_construct(type="text", text=json.dumps(payload))],
            usage=UsageInfo.model_construct(input_tokens=10, output_tokens=5, total_tokens=15),
            stop_reason="end_turn",
            status_code=200,
            response_headers={},
        )

    return FunctionClient(function=fn)


def _envelope() -> dict[str, Any]:
    """A minimal one-step Program v3 envelope using the function-test client."""
    return {
        "kaos_program": "1",
        "name": "test-classify",
        "inputs": {"text": {"type": "string"}},
        "clients": {"default": {"provider": "function", "model": "function-test"}},
        "steps": [
            {
                "id": "classify",
                "kind": "call",
                "client": "default",
                "instruction": "Echo the answer field.",
                "inputs": {"text": "$.inputs.text"},
                "output_fields": {
                    "answer": {
                        "description": "The answer",
                        "type": {"type": "string"},
                    }
                },
            }
        ],
        "output": {"answer": "$.steps.classify.output.answer"},
        "capabilities": ["call", "jsonpointer_refs"],
    }


async def _save_envelope(runtime: KaosRuntime, context: KaosContext, env: dict) -> str:
    """Write an envelope into the VFS using the pinned batch context_id.

    The Phase 15.3 batch tools resolve all VFS lookups via the
    ``__kaos_llm_core_batch__`` sentinel context_id (so batches are
    shared across MCP sessions). Tests must write to the same scope.
    """
    from kaos_llm_core.integrations.mcp._batch_helpers import _BATCH_CONTEXT_ID

    _ = context  # signature symmetry; the pinned id is used instead
    path = "programs/test-classify.json"
    data = json.dumps(env).encode("utf-8")
    await runtime.vfs.write(path, data, context_id=_BATCH_CONTEXT_ID)
    return path


def _patch_function_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force every Call built by from_envelope to use a deterministic stub.

    The envelope's ``clients`` block names ``provider="function"``, so
    Call.invoke ends up calling ``create_client("function:function-test")``
    which fails because the real ``function`` provider needs a function
    arg. We monkey-patch ``create_client`` everywhere it is imported to
    short-circuit and return our stub for the function-test model.
    """
    from kaos_llm_client import providers as _providers_pkg

    real_create = _providers_pkg.create_client
    stub = _stub_function_client()

    def _patched(model: str, *args: Any, **kwargs: Any):
        if "function" in model:
            return stub
        return real_create(model, *args, **kwargs)

    # Patch every binding (the providers package + every module that
    # already did `from kaos_llm_client import create_client`).
    monkeypatch.setattr("kaos_llm_client.providers.create_client", _patched)
    monkeypatch.setattr("kaos_llm_client.create_client", _patched)
    monkeypatch.setattr("kaos_llm_core.programs.call.create_client", _patched)


# ---------------------------------------------------------------------------
# Tool instances
# ---------------------------------------------------------------------------


@pytest.fixture
def create_tool() -> KaosLLMCoreBatchCreateTool:
    return KaosLLMCoreBatchCreateTool()


@pytest.fixture
def run_tool() -> KaosLLMCoreBatchRunTool:
    return KaosLLMCoreBatchRunTool()


@pytest.fixture
def status_tool() -> KaosLLMCoreBatchStatusTool:
    return KaosLLMCoreBatchStatusTool()


@pytest.fixture
def results_tool() -> KaosLLMCoreBatchResultsTool:
    return KaosLLMCoreBatchResultsTool()


# ---------------------------------------------------------------------------
# batch-create
# ---------------------------------------------------------------------------


class TestBatchCreate:
    async def test_create_with_inline_envelope_pending(
        self,
        create_tool: KaosLLMCoreBatchCreateTool,
        context: KaosContext,
    ) -> None:
        result = await create_tool.execute(
            inputs={
                "envelope": _envelope(),
                "inputs_source": {
                    "type": "list",
                    "items": [{"text": "a"}, {"text": "b"}, {"text": "c"}],
                },
                "output_dir": "runs/inline-create",
                "name": "inline test",
            },
            context=context,
        )
        assert not result.isError, result.text
        out = result.require_structured()
        assert out["status"] == "pending"
        assert out["batch_id"].startswith("batch-")
        assert out["inputs_count_hint"] == 3
        assert out["program_name"] == "test-classify"

    async def test_create_with_envelope_path(
        self,
        create_tool: KaosLLMCoreBatchCreateTool,
        runtime: KaosRuntime,
        context: KaosContext,
    ) -> None:
        env_path = await _save_envelope(runtime, context, _envelope())
        result = await create_tool.execute(
            inputs={
                "envelope_path": env_path,
                "inputs_source": {
                    "type": "list",
                    "items": [{"text": "x"}, {"text": "y"}],
                },
                "output_dir": "runs/path-create",
            },
            context=context,
        )
        assert not result.isError, result.text
        out = result.require_structured()
        assert out["envelope_path"] == env_path
        assert out["inputs_count_hint"] == 2

    async def test_create_rejects_missing_envelope(
        self,
        create_tool: KaosLLMCoreBatchCreateTool,
        context: KaosContext,
    ) -> None:
        result = await create_tool.execute(
            inputs={
                "inputs_source": {"type": "list", "items": [{"text": "a"}]},
                "output_dir": "runs/missing",
            },
            context=context,
        )
        assert result.isError
        assert "envelope" in (result.text or "").lower()

    async def test_create_rejects_stop_after_n_without_max_errors(
        self,
        create_tool: KaosLLMCoreBatchCreateTool,
        context: KaosContext,
    ) -> None:
        result = await create_tool.execute(
            inputs={
                "envelope": _envelope(),
                "inputs_source": {"type": "list", "items": [{"text": "a"}]},
                "output_dir": "runs/bad-policy",
                "error_policy": "stop_after_n",
            },
            context=context,
        )
        assert result.isError
        assert "max_errors" in (result.text or "")

    async def test_create_rejects_unknown_inputs_source_type(
        self,
        create_tool: KaosLLMCoreBatchCreateTool,
        context: KaosContext,
    ) -> None:
        result = await create_tool.execute(
            inputs={
                "envelope": _envelope(),
                "inputs_source": {"type": "wat"},
                "output_dir": "runs/bad-source",
            },
            context=context,
        )
        assert result.isError


# ---------------------------------------------------------------------------
# Full batch-create → batch-run → batch-status → batch-results
# ---------------------------------------------------------------------------


class TestBatchLifecycle:
    async def test_full_lifecycle_against_function_client(
        self,
        create_tool: KaosLLMCoreBatchCreateTool,
        run_tool: KaosLLMCoreBatchRunTool,
        status_tool: KaosLLMCoreBatchStatusTool,
        results_tool: KaosLLMCoreBatchResultsTool,
        runtime: KaosRuntime,
        context: KaosContext,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_function_client(monkeypatch)
        env_path = await _save_envelope(runtime, context, _envelope())

        # 1. CREATE
        create_result = await create_tool.execute(
            inputs={
                "envelope_path": env_path,
                "inputs_source": {
                    "type": "list",
                    "items": [{"text": f"item-{i}"} for i in range(5)],
                },
                "output_dir": "runs/lifecycle",
                "name": "lifecycle",
            },
            context=context,
        )
        assert not create_result.isError, create_result.text
        batch_id = create_result.require_structured()["batch_id"]

        # 2. STATUS (pending)
        status_pending = await status_tool.execute(inputs={"batch_id": batch_id}, context=context)
        assert not status_pending.isError
        assert status_pending.require_structured()["status"] == "pending"

        # 3. RUN
        run_result = await run_tool.execute(inputs={"batch_id": batch_id}, context=context)
        assert not run_result.isError, run_result.text
        run_out = run_result.require_structured()
        assert run_out["status"] == "completed"
        assert run_out["n_succeeded"] == 5
        assert run_out["n_errored"] == 0
        assert run_out["cost_usd"] > 0

        # 4. STATUS (completed)
        status_done = await status_tool.execute(inputs={"batch_id": batch_id}, context=context)
        assert not status_done.isError
        sd = status_done.require_structured()
        assert sd["status"] == "completed"
        assert sd["n_succeeded"] == 5
        assert sd["cost_usd_so_far"] > 0
        assert sd["tokens_so_far"]["total"] > 0

        # 5. RESULTS (manifest)
        results = await results_tool.execute(
            inputs={"batch_id": batch_id, "format": "manifest"},
            context=context,
        )
        assert not results.isError
        ro = results.require_structured()
        assert ro["status"] == "completed"
        assert ro["n_succeeded"] == 5
        assert "manifest" in ro
        assert ro["manifest"]["n_total"] == 5

        # 6. RESULTS (jsonl) — small enough to fit inline
        jsonl = await results_tool.execute(
            inputs={"batch_id": batch_id, "format": "jsonl"},
            context=context,
        )
        assert not jsonl.isError
        rows = jsonl.require_structured()["rows"]
        # 1 header + 5 items
        assert len(rows) == 6
        assert rows[0]["kind"] == "header"
        assert all(r["status"] == "success" for r in rows[1:])

    async def test_run_unknown_batch_errors(
        self,
        run_tool: KaosLLMCoreBatchRunTool,
        context: KaosContext,
    ) -> None:
        result = await run_tool.execute(
            inputs={"batch_id": "batch-does-not-exist"},
            context=context,
        )
        assert result.isError
        assert "No batch" in (result.text or "")

    async def test_run_inline_envelope_persists_to_output_dir(
        self,
        create_tool: KaosLLMCoreBatchCreateTool,
        run_tool: KaosLLMCoreBatchRunTool,
        context: KaosContext,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Inline envelopes are persisted to ${output_dir}/envelope.json on
        create so batch-run can re-load them on a separate MCP call."""
        _patch_function_client(monkeypatch)
        create_result = await create_tool.execute(
            inputs={
                "envelope": _envelope(),
                "inputs_source": {"type": "list", "items": [{"text": "a"}]},
                "output_dir": "runs/inline-persisted",
            },
            context=context,
        )
        assert not create_result.isError, create_result.text
        co = create_result.require_structured()
        batch_id = co["batch_id"]
        # envelope_path was rewritten to point at the persisted file.
        assert co["envelope_path"].endswith("envelope.json")
        # Run successfully against the persisted envelope.
        run_result = await run_tool.execute(inputs={"batch_id": batch_id}, context=context)
        assert not run_result.isError, run_result.text
        assert run_result.require_structured()["status"] == "completed"

    async def test_run_completed_without_resume_errors(
        self,
        create_tool: KaosLLMCoreBatchCreateTool,
        run_tool: KaosLLMCoreBatchRunTool,
        runtime: KaosRuntime,
        context: KaosContext,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_function_client(monkeypatch)
        env_path = await _save_envelope(runtime, context, _envelope())
        create_result = await create_tool.execute(
            inputs={
                "envelope_path": env_path,
                "inputs_source": {"type": "list", "items": [{"text": "a"}]},
                "output_dir": "runs/completed",
            },
            context=context,
        )
        batch_id = create_result.require_structured()["batch_id"]
        first = await run_tool.execute(inputs={"batch_id": batch_id}, context=context)
        assert not first.isError
        # Re-run without resume → error
        second = await run_tool.execute(
            inputs={"batch_id": batch_id, "resume": False},
            context=context,
        )
        assert second.isError
        assert "completed" in (second.text or "").lower()
