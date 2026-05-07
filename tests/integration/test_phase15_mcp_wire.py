"""End-to-end MCP wire tests for Phase 15 tools.

Drives the new Phase 15 MCP tools through the actual `kaos-llm-core-serve`
process via stdio + the real `mcp.ClientSession`. Same pattern as
`test_phase6_mcp_wire.py` and `test_phase7_mcp_wire.py`.

These verify three things at once:
1. The MCP wire format is correct (tools are discoverable and callable).
2. The kaos-mcp adapter actually invokes the underlying Phase 15 code.
3. Real LLM provider calls happen end-to-end through the MCP wire.

Per `docs/guides/mcp-testing.md`, every Phase 15 commit gets at least one
wire test in this file before merge.
"""

from __future__ import annotations

import os

import pytest
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


# A minimal v3 envelope: one call step that classifies legal risk.
# Reused across multiple wire tests so the failure modes are pinned to
# a single canonical envelope.
LEGAL_RISK_ENVELOPE: dict[str, object] = {
    "kaos_program": "1",
    "name": "legal-risk-classify",
    "description": "Classify the legal risk of a scenario as low/medium/high.",
    "inputs": {
        "text": {
            "type": "string",
            "description": "Legal scenario text",
            "required": True,
        }
    },
    "clients": {"default": {"provider": "anthropic", "model": HAIKU}},
    "steps": [
        {
            "id": "classify",
            "kind": "call",
            "client": "default",
            "instruction": (
                "Classify the legal risk of the scenario. Output the level "
                "(low/medium/high) and a one-sentence rationale."
            ),
            "inputs": {"text": "$.inputs.text"},
            "output_fields": {
                "level": {
                    "description": "Risk level: low, medium, or high",
                    "type": {"type": "string"},
                },
                "reason": {
                    "description": "One-sentence rationale",
                    "type": {"type": "string"},
                },
            },
        }
    ],
    "output": {
        "level": "$.steps.classify.output.level",
        "reason": "$.steps.classify.output.reason",
    },
    "capabilities": ["call", "jsonpointer_refs"],
}


def _server_params() -> StdioServerParameters:
    """Build the stdio command for the kaos-llm-core MCP server."""
    return StdioServerParameters(
        command="uv",
        args=["run", "kaos-llm-core-serve"],
        env=dict(os.environ),
    )


def _extract_text(call_result: object) -> str:
    """Pull the text payload out of an MCP CallToolResult."""
    chunks: list[str] = []
    for item in getattr(call_result, "content", []):
        text = getattr(item, "text", None)
        if text:
            chunks.append(text)
    return "\n".join(chunks)


def _extract_structured(call_result: object) -> dict | None:
    """Pull the structuredContent dict from a CallToolResult.

    KAOS tool responses carry both ``content`` (one-line text summary) and
    ``structuredContent`` (the canonical JSON payload). Assertions on the
    structured payload go through this helper.
    """
    sc = getattr(call_result, "structuredContent", None)
    if isinstance(sc, dict):
        return sc
    return None


class TestPhase15McpWire:
    """End-to-end MCP wire tests against the real server."""

    @requires_anthropic
    async def test_program_execute_listed(self) -> None:
        """The new Phase 15 tool must be discoverable via MCP."""
        async with (
            stdio_client(_server_params()) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
            assert "kaos-llm-core-program-execute" in names, (
                f"kaos-llm-core-program-execute missing. All tools: {sorted(names)}"
            )
            print(f"\n  [phase15_list] {len(names)} tools registered via MCP")

    @requires_anthropic
    async def test_program_execute_inline_envelope(self) -> None:
        """Drive the program-execute tool with an inline envelope against real Haiku.

        End-to-end: agent → MCP wire → kaos-llm-core-serve →
        from_envelope → program.invoke → real Anthropic Haiku → response →
        unwound back through the wire to the agent.
        """
        async with (
            stdio_client(_server_params()) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            result = await session.call_tool(
                "kaos-llm-core-program-execute",
                arguments={
                    "envelope": LEGAL_RISK_ENVELOPE,
                    "inputs": {
                        "text": (
                            "A pharma company hid adverse trial data from the FDA "
                            "for 18 months while continuing to market the drug."
                        )
                    },
                },
            )

        assert not result.isError, (
            f"program-execute returned isError; payload: {_extract_text(result)}"
        )
        structured = _extract_structured(result)
        assert structured is not None, (
            f"Expected structuredContent on program-execute response; got text: "
            f"{_extract_text(result)[:200]}"
        )
        outputs = structured.get("outputs", {})
        print(f"\n  [phase15_execute] outputs: {outputs}")
        assert "level" in outputs, f"Expected 'level' in outputs; got {outputs}"
        assert "reason" in outputs, f"Expected 'reason' in outputs; got {outputs}"
        level_value = str(outputs.get("level", "")).lower()
        assert "high" in level_value, (
            f"Expected 'high' risk for the FDA scenario; got level={outputs.get('level')!r}"
        )
        assert structured.get("cost_usd", 0) > 0, (
            f"Expected positive cost_usd; got {structured.get('cost_usd')}"
        )
        assert structured.get("tokens", {}).get("total", 0) > 0

    @requires_anthropic
    async def test_program_execute_two_step_chained(self) -> None:
        """Drive a two-step envelope where step 2 references step 1's output.

        Verifies the JSON-pointer chaining ($.steps.<id>.output.<field>)
        works through the wire.
        """
        envelope = {
            "kaos_program": "1",
            "name": "two-step-chain",
            "inputs": {"text": {"type": "string"}},
            "clients": {"default": {"provider": "anthropic", "model": HAIKU}},
            "steps": [
                {
                    "id": "extract",
                    "kind": "call",
                    "client": "default",
                    "instruction": (
                        "Extract the company name and the alleged misconduct "
                        "from this short scenario."
                    ),
                    "inputs": {"text": "$.inputs.text"},
                    "output_fields": {
                        "company": {
                            "description": "Company name",
                            "type": {"type": "string"},
                        },
                        "misconduct": {
                            "description": "Alleged misconduct (1 sentence)",
                            "type": {"type": "string"},
                        },
                    },
                },
                {
                    "id": "classify",
                    "kind": "call",
                    "client": "default",
                    "instruction": (
                        "Given the company and the alleged misconduct, classify "
                        "severity as low/medium/high."
                    ),
                    "inputs": {
                        "company": "$.steps.extract.output.company",
                        "misconduct": "$.steps.extract.output.misconduct",
                    },
                    "output_fields": {
                        "severity": {
                            "description": "low/medium/high",
                            "type": {"type": "string"},
                        }
                    },
                },
            ],
            "output": {
                "company": "$.steps.extract.output.company",
                "severity": "$.steps.classify.output.severity",
            },
            "capabilities": ["call", "jsonpointer_refs"],
        }

        async with (
            stdio_client(_server_params()) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            result = await session.call_tool(
                "kaos-llm-core-program-execute",
                arguments={
                    "envelope": envelope,
                    "inputs": {
                        "text": (
                            "Acme Corp allegedly forged signatures on 200 vendor "
                            "invoices over the past year."
                        )
                    },
                },
            )

        assert not result.isError, (
            f"two-step program-execute returned isError; payload: {_extract_text(result)}"
        )
        structured = _extract_structured(result)
        assert structured is not None, "Expected structuredContent on two-step response"
        outputs = structured.get("outputs", {})
        print(f"\n  [phase15_two_step] outputs: {outputs}")
        assert "company" in outputs, f"Expected 'company' in outputs; got {outputs}"
        assert "severity" in outputs, f"Expected 'severity' in outputs; got {outputs}"
        company_value = str(outputs.get("company", "")).lower()
        assert "acme" in company_value, (
            f"Expected 'Acme' in company; got {outputs.get('company')!r}"
        )
        severity_value = str(outputs.get("severity", "")).lower()
        assert any(level in severity_value for level in ("low", "medium", "high")), (
            f"Expected low/medium/high severity; got {outputs.get('severity')!r}"
        )

    @requires_anthropic
    async def test_program_execute_rejects_invalid_envelope(self) -> None:
        """An invalid envelope must surface a clean schema error, not crash."""
        async with (
            stdio_client(_server_params()) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            result = await session.call_tool(
                "kaos-llm-core-program-execute",
                arguments={
                    "envelope": {
                        "kaos_program": "1",
                        "name": "BAD-NAME-WITH-CAPS",  # rejected by name validator
                        "inputs": {},
                        "clients": {},
                        "steps": [],
                        "output": {},
                        "capabilities": [],
                    },
                    "inputs": {},
                },
            )
        # Tool returns isError=True with a structured error envelope
        assert result.isError
        text = _extract_text(result)
        assert "envelope" in text.lower() or "name" in text.lower(), (
            f"Expected error message about envelope/name; got: {text[:500]}"
        )

    @requires_anthropic
    async def test_program_execute_rejects_missing_envelope(self) -> None:
        """Calling without envelope or envelope_path returns a clean error."""
        async with (
            stdio_client(_server_params()) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            result = await session.call_tool(
                "kaos-llm-core-program-execute",
                arguments={"inputs": {"text": "x"}},
            )
        assert result.isError
        text = _extract_text(result)
        assert "envelope" in text.lower(), (
            f"Expected error about missing envelope; got: {text[:500]}"
        )
