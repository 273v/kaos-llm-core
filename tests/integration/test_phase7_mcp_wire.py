"""End-to-end MCP wire tests for Phase 7 tools.

Drives the 2 new Phase 7 MCP tools (``kaos-llm-core-metric``,
``kaos-llm-core-analyze-trial``) through the actual ``kaos-llm-core-serve``
stdio subprocess via FastMCP ``ClientSession``. Same discipline as
``test_phase6_mcp_wire.py``: real wire, real server process, real provider
calls when the metric is ``llm_judge``.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

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


def _server_params() -> StdioServerParameters:
    return StdioServerParameters(
        command="uv",
        args=["run", "kaos-llm-core-serve"],
        env=dict(os.environ),
    )


def _extract_text(call_result: object) -> str:
    content = getattr(call_result, "content", [])
    chunks: list[str] = []
    for item in content:
        text = getattr(item, "text", None)
        if text:
            chunks.append(text)
    return "\n".join(chunks)


def _extract_structured(call_result: object) -> dict[str, object] | None:
    """Pull the structuredContent dict from a CallToolResult.

    KAOS tool responses carry both ``content`` (text rendering for human-ish
    consumers) and ``structuredContent`` (the canonical JSON payload). The
    text is a one-line summary; the structuredContent has the data.
    """
    sc = getattr(call_result, "structuredContent", None)
    if isinstance(sc, dict):
        return sc
    return None


def _seed_log(path: Path) -> None:
    """Write a small JSONL fixture for the analyze-trial tool."""
    records = [
        {
            "mutation_id": "a" * 32,
            "run_id": "r" * 32,
            "trial_id": 0,
            "parent_mutation_id": None,
            "strategy": "bootstrap",
            "mutation_type": "add_examples",
            "call_name": "DemoCall",
            "before": {"n_examples": 0},
            "after": {"n_examples": 4},
            "rationale": "seeded for MCP wire test",
            "metric_before": 0.5,
            "metric_after": 0.85,
            "tokens_used": 220,
            "cost_usd": 0.00345,
            "duration_ms": 250.0,
            "accepted": True,
            "error": None,
            "timestamp": datetime.now(UTC).isoformat(),
        },
        {
            "mutation_id": "b" * 32,
            "run_id": "r" * 32,
            "trial_id": 1,
            "parent_mutation_id": "a" * 32,
            "strategy": "instruction_tuning",
            "mutation_type": "change_instructions",
            "call_name": "DemoCall",
            "before": {"instructions": "old"},
            "after": {"instructions": "new and improved"},
            "rationale": "proposer suggested",
            "metric_before": 0.85,
            "metric_after": 0.95,
            "tokens_used": 480,
            "cost_usd": 0.00892,
            "duration_ms": 510.0,
            "accepted": True,
            "error": None,
            "timestamp": datetime.now(UTC).isoformat(),
        },
    ]
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


class TestPhase7McpWire:
    @requires_anthropic
    async def test_phase7_tools_listed(self) -> None:
        async with (
            stdio_client(_server_params()) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            tools = await session.list_tools()
            names = [t.name for t in tools.tools]

        assert "kaos-llm-core-metric" in names
        assert "kaos-llm-core-analyze-trial" in names

    @requires_anthropic
    async def test_metric_deterministic_via_mcp(self) -> None:
        """Deterministic metric path — no provider call needed."""
        async with (
            stdio_client(_server_params()) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            result = await session.call_tool(
                "kaos-llm-core-metric",
                arguments={
                    "metric_name": "normalized_match",
                    "prediction": "Termination.",
                    "gold": "termination",
                },
            )

        assert not result.isError, (
            f"kaos-llm-core-metric returned isError; payload: {_extract_text(result)}"
        )
        structured = _extract_structured(result)
        text = _extract_text(result)
        print(f"\n  [mcp_metric_normalized] text={text[:120]} structured={structured}")
        # The normalized_match should produce a 1.0 score (F2 fix: strip
        # trailing punctuation and lowercase).
        assert structured is not None, "Expected structuredContent on metric response"
        assert structured.get("score") == 1.0, (
            f"Expected normalized_match('Termination.', 'termination')==1.0, "
            f"got {structured.get('score')}"
        )

    @requires_anthropic
    async def test_metric_llm_judge_via_mcp(self) -> None:
        """LLMJudge path — opens a provider call through the MCP wire."""
        async with (
            stdio_client(_server_params()) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            result = await session.call_tool(
                "kaos-llm-core-metric",
                arguments={
                    "metric_name": "llm_judge",
                    "prediction": "The capital of France is Paris.",
                    "gold": "What is the capital of France?",
                    "judge_model": "anthropic:claude-haiku-4-5",
                    "rubric": "helpfulness",
                },
            )

        assert not result.isError, (
            f"kaos-llm-core-metric (llm_judge) returned isError; payload: {_extract_text(result)}"
        )
        structured = _extract_structured(result)
        text = _extract_text(result)
        print(f"\n  [mcp_metric_llm_judge] text={text[:120]} structured={structured}")
        assert structured is not None, "Expected structuredContent on metric response"
        score = structured.get("score")
        assert isinstance(score, (int, float))
        assert 0.0 <= float(score) <= 1.0
        # A clearly correct factual answer should score above 0.5 from a
        # competent judge model.
        assert float(score) >= 0.5

    @requires_anthropic
    async def test_analyze_trial_via_mcp(self) -> None:
        """Analyze a real fixture log through the MCP wire."""
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "phase7_wire.jsonl"
            _seed_log(log_path)

            async with (
                stdio_client(_server_params()) as (read, write),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                result = await session.call_tool(
                    "kaos-llm-core-analyze-trial",
                    arguments={"mutation_log_path": str(log_path)},
                )

        assert not result.isError, (
            f"kaos-llm-core-analyze-trial returned isError; payload: {_extract_text(result)}"
        )
        structured = _extract_structured(result)
        text = _extract_text(result)
        print(f"\n  [mcp_analyze_trial] text={text[:200]}")
        assert structured is not None, "Expected structuredContent on analyze-trial response"
        # The summary should reflect 2 mutations across 2 strategies.
        summary_raw = structured.get("summary") or {}
        summary = cast("dict[str, Any]", summary_raw)
        assert summary.get("total_trials") == 2
        # Per-strategy contributions present.
        contribs_raw = (
            structured.get("strategy_contributions") or structured.get("by_strategy") or []
        )
        contribs = cast("list[Any]", contribs_raw)
        strategies = {c.get("strategy") for c in contribs if isinstance(c, dict)}
        assert "bootstrap" in strategies
        assert "instruction_tuning" in strategies
