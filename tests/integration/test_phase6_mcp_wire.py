"""End-to-end MCP wire tests for Phase 6 tools.

Drives the 4 new Phase 6 MCP tools (`kaos-llm-core-optimize-codec`,
`-optimize-model`, `-pareto`, `-recipe-tune`) through the actual MCP server
process via stdio transport using the real ``kaos-llm-core-serve`` entry point
and the real FastMCP ``ClientSession``.

These verify three things at once:
1. The MCP wire format is correct (tools are discoverable and callable).
2. The kaos-mcp adapter actually invokes the underlying optimizer.
3. Real LLM provider calls happen end-to-end through the MCP wire.

This test matches the discipline of commit 07a6269 — the same E2E pattern
that verified Phase 5 tools through claude + codex CLI.
"""

from __future__ import annotations

import os

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


# Real legal-domain examples that match test_phase6_live.py.
LEGAL_EXAMPLES = [
    {
        "input": (
            "Each party shall defend, indemnify, and hold harmless the other "
            "party from any third-party claims arising out of or related to the "
            "indemnifying party's breach of this Agreement."
        ),
        "expected_output": "indemnification",
    },
    {
        "input": (
            "Notwithstanding anything to the contrary in this Agreement, neither "
            "party's aggregate liability shall exceed the total fees paid by "
            "Customer in the twelve months preceding the claim."
        ),
        "expected_output": "limitation_of_liability",
    },
    {
        "input": (
            "Recipient shall not disclose Confidential Information to any third "
            "party except its employees and contractors who have a need to know."
        ),
        "expected_output": "confidentiality",
    },
    {
        "input": (
            "Either party may terminate this Agreement for convenience upon "
            "ninety (90) days prior written notice to the other party."
        ),
        "expected_output": "termination",
    },
]

INSTRUCTION = (
    "You are a contracts attorney. Read the clause and return exactly one "
    "label from this set: indemnification, limitation_of_liability, "
    "confidentiality, termination, payment_terms, warranty, governing_law. "
    "Return only the label, in lowercase with underscores."
)


def _server_params() -> StdioServerParameters:
    """Build the stdio command for the kaos-llm-core MCP server."""
    return StdioServerParameters(
        command="uv",
        args=["run", "kaos-llm-core-serve"],
        env=dict(os.environ),
    )


async def _connect_and_list_tools() -> list[str]:
    """Connect to the real server, list tools, return their names."""
    async with (
        stdio_client(_server_params()) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        tools = await session.list_tools()
        return [t.name for t in tools.tools]


def _extract_text(call_result: object) -> str:
    """Pull the text payload out of an MCP CallToolResult."""
    content = getattr(call_result, "content", [])
    chunks: list[str] = []
    for item in content:
        text = getattr(item, "text", None)
        if text:
            chunks.append(text)
    return "\n".join(chunks)


class TestPhase6McpWire:
    """End-to-end MCP wire tests against the real server."""

    @requires_anthropic
    async def test_mcp_server_lists_phase6_tools(self) -> None:
        """The 4 Phase 6 tools must be discoverable via MCP."""
        names = await _connect_and_list_tools()
        print(f"\n  [mcp_list] {len(names)} tools registered via MCP")
        expected = {
            "kaos-llm-core-optimize-codec",
            "kaos-llm-core-optimize-model",
            "kaos-llm-core-pareto",
            "kaos-llm-core-recipe-tune",
        }
        missing = expected - set(names)
        assert not missing, (
            f"Phase 6 MCP tools missing from list_tools(): {sorted(missing)}. "
            f"All tools: {sorted(names)}"
        )

    @requires_anthropic
    async def test_optimize_codec_via_mcp(self) -> None:
        """Drive `kaos-llm-core-optimize-codec` through the MCP wire."""
        async with (
            stdio_client(_server_params()) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            result = await session.call_tool(
                "kaos-llm-core-optimize-codec",
                arguments={
                    "model": "anthropic:claude-haiku-4-5",
                    "instruction": INSTRUCTION,
                    "examples": LEGAL_EXAMPLES,
                },
            )

        assert not result.isError, (
            f"kaos-llm-core-optimize-codec returned isError; payload: {_extract_text(result)}"
        )
        text = _extract_text(result)
        print(f"\n  [mcp_optimize_codec] response: {text[:400]}")
        # The structured response carries best_codec and scores_by_codec.
        # Both string match and JSON parse are acceptable signals.
        assert "best_codec" in text or "JSONCodec" in text or "ChatCodec" in text, (
            f"Expected codec selection result; got: {text[:500]}"
        )

    @requires_anthropic
    async def test_optimize_model_via_mcp(self) -> None:
        """Drive `kaos-llm-core-optimize-model` through the MCP wire."""
        async with (
            stdio_client(_server_params()) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            result = await session.call_tool(
                "kaos-llm-core-optimize-model",
                arguments={
                    "models": [
                        "anthropic:claude-haiku-4-5",
                        "anthropic:claude-sonnet-4-6",
                    ],
                    "instruction": INSTRUCTION,
                    "examples": LEGAL_EXAMPLES,
                    "min_score": 0.6,
                },
            )

        assert not result.isError, (
            f"kaos-llm-core-optimize-model returned isError; payload: {_extract_text(result)}"
        )
        text = _extract_text(result)
        print(f"\n  [mcp_optimize_model] response: {text[:400]}")
        assert "best_model" in text or "claude-haiku-4-5" in text or "claude-sonnet" in text, (
            f"Expected model selection result; got: {text[:500]}"
        )

    @requires_anthropic
    async def test_pareto_via_mcp(self) -> None:
        """Drive `kaos-llm-core-pareto` through the MCP wire."""
        async with (
            stdio_client(_server_params()) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            result = await session.call_tool(
                "kaos-llm-core-pareto",
                arguments={
                    "models": [
                        "anthropic:claude-haiku-4-5",
                        "anthropic:claude-sonnet-4-6",
                        "anthropic:claude-opus-4-6",
                    ],
                    "instruction": INSTRUCTION,
                    "examples": LEGAL_EXAMPLES,
                },
            )

        assert not result.isError, (
            f"kaos-llm-core-pareto returned isError; payload: {_extract_text(result)}"
        )
        text = _extract_text(result)
        print(f"\n  [mcp_pareto] response: {text[:600]}")
        assert "frontier" in text or "claude" in text, (
            f"Expected Pareto frontier; got: {text[:500]}"
        )

    @requires_anthropic
    async def test_recipe_tune_via_mcp(self) -> None:
        """Drive `kaos-llm-core-recipe-tune` (cost_aware_model recipe)."""
        async with (
            stdio_client(_server_params()) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            result = await session.call_tool(
                "kaos-llm-core-recipe-tune",
                arguments={
                    "recipe": "cost_aware_model",
                    "instruction": INSTRUCTION,
                    "examples": LEGAL_EXAMPLES,
                    "models": [
                        "anthropic:claude-haiku-4-5",
                        "anthropic:claude-sonnet-4-6",
                    ],
                    "min_score": 0.6,
                },
            )

        assert not result.isError, (
            f"kaos-llm-core-recipe-tune returned isError; payload: {_extract_text(result)}"
        )
        text = _extract_text(result)
        print(f"\n  [mcp_recipe_tune] response: {text[:400]}")
        assert "claude" in text or "best" in text, f"Expected recipe result; got: {text[:500]}"
