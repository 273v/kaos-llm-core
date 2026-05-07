"""Unit tests for the 6 alpha-extractor MCP tools — WS-TR.PR-6f.7.

Covers: tool metadata + annotations, happy-path execution, empty-text
handling, missing-parameter error path, registration in the tool
registry, and the ``ALPHA_MAX_RESULTS`` truncation.
"""

from __future__ import annotations

from typing import Any

import pytest
from kaos_core import ToolResult

from kaos_llm_core.integrations.mcp._alpha_common import ALPHA_MAX_RESULTS
from kaos_llm_core.integrations.mcp.alpha_date import KaosLLMCoreAlphaDateTool
from kaos_llm_core.integrations.mcp.alpha_duration import KaosLLMCoreAlphaDurationTool
from kaos_llm_core.integrations.mcp.alpha_entity import KaosLLMCoreAlphaEntityTool
from kaos_llm_core.integrations.mcp.alpha_money import KaosLLMCoreAlphaMoneyTool
from kaos_llm_core.integrations.mcp.alpha_number import KaosLLMCoreAlphaNumberTool
from kaos_llm_core.integrations.mcp.alpha_percent import KaosLLMCoreAlphaPercentTool


def _struct(result: ToolResult) -> dict[str, Any]:
    """Extract structuredContent as a non-None dict for ty/test access."""
    s = result.structuredContent
    assert s is not None, "tool produced no structured content"
    return s


ALPHA_TOOL_CLASSES = [
    KaosLLMCoreAlphaDateTool,
    KaosLLMCoreAlphaEntityTool,
    KaosLLMCoreAlphaMoneyTool,
    KaosLLMCoreAlphaNumberTool,
    KaosLLMCoreAlphaPercentTool,
    KaosLLMCoreAlphaDurationTool,
]


# ---------------------------------------------------------------------------
# Metadata + annotations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tool_cls", ALPHA_TOOL_CLASSES)
class TestMetadataAndAnnotations:
    """All 6 tools must declare the same kelvin-conformant annotation
    block (read-only, idempotent, closed-world, non-destructive)."""

    def test_name_prefix(self, tool_cls: type) -> None:
        tool = tool_cls()
        assert tool.metadata.name.startswith("kaos-llm-core-alpha-")

    def test_module_name(self, tool_cls: type) -> None:
        tool = tool_cls()
        assert tool.metadata.module_name == "kaos-llm-core"

    def test_read_only(self, tool_cls: type) -> None:
        tool = tool_cls()
        assert tool.metadata.annotations.readOnlyHint is True

    def test_not_destructive(self, tool_cls: type) -> None:
        tool = tool_cls()
        assert tool.metadata.annotations.destructiveHint is False

    def test_idempotent(self, tool_cls: type) -> None:
        tool = tool_cls()
        assert tool.metadata.annotations.idempotentHint is True

    def test_closed_world(self, tool_cls: type) -> None:
        tool = tool_cls()
        # Alpha extractors are pure-local — no provider calls, no I/O.
        assert tool.metadata.annotations.openWorldHint is False

    def test_has_text_param(self, tool_cls: type) -> None:
        tool = tool_cls()
        param_names = [p.name for p in tool.metadata.input_schema]
        assert "text" in param_names

    def test_description_nonempty(self, tool_cls: type) -> None:
        tool = tool_cls()
        # tool-design.md mandates substantive descriptions.
        assert len(tool.metadata.description) > 100


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tool_cls", ALPHA_TOOL_CLASSES)
class TestErrors:
    @pytest.mark.asyncio
    async def test_missing_text(self, tool_cls: type) -> None:
        tool = tool_cls()
        result = await tool.execute({})
        assert result.isError is True
        # tool-design.md: errors must include what + how + alternative.
        msg = result.text or ""
        assert "text" in msg.lower()

    @pytest.mark.asyncio
    async def test_non_string_text(self, tool_cls: type) -> None:
        tool = tool_cls()
        result = await tool.execute({"text": 42})
        assert result.isError is True

    @pytest.mark.asyncio
    async def test_empty_string_returns_no_spans(self, tool_cls: type) -> None:
        """Empty input is valid — no spans, but not an error."""
        tool = tool_cls()
        result = await tool.execute({"text": ""})
        assert result.isError is False
        assert result.structuredContent is not None
        assert _struct(result)["total_matches"] == 0
        assert _struct(result)["spans"] == []
        assert _struct(result)["has_more"] is False


# ---------------------------------------------------------------------------
# Per-tool happy paths
# ---------------------------------------------------------------------------


class TestAlphaDateTool:
    @pytest.mark.asyncio
    async def test_extracts_iso_date(self) -> None:
        tool = KaosLLMCoreAlphaDateTool()
        result = await tool.execute({"text": "Agreement made February 17, 1999."})
        s = _struct(result)
        assert s["total_matches"] == 1
        assert s["spans"][0]["value"] == "1999-02-17"

    @pytest.mark.asyncio
    async def test_ordinal_form(self) -> None:
        tool = KaosLLMCoreAlphaDateTool()
        result = await tool.execute({"text": "entered into this 21st day of January 2003"})
        s = _struct(result)
        assert any(span["value"] == "2003-01-21" for span in s["spans"])

    @pytest.mark.asyncio
    async def test_span_round_trips(self) -> None:
        tool = KaosLLMCoreAlphaDateTool()
        text = "signed January 15, 2024 by parties"
        result = await tool.execute({"text": text})
        s = _struct(result)["spans"][0]
        assert text[s["start"] : s["end"]] == s["snippet"]


class TestAlphaEntityTool:
    @pytest.mark.asyncio
    async def test_finds_two_parties(self) -> None:
        tool = KaosLLMCoreAlphaEntityTool()
        result = await tool.execute({"text": "Acme Corp. and Beta LLC are parties."})
        s = _struct(result)
        names = {span["name"] for span in s["spans"]}
        assert "Acme" in names
        assert "Beta" in names

    @pytest.mark.asyncio
    async def test_entity_type_field(self) -> None:
        tool = KaosLLMCoreAlphaEntityTool()
        result = await tool.execute({"text": "Sony K.K. manufactures electronics."})
        s = _struct(result)
        assert any(span["entity_type"] == "K.K." for span in s["spans"])


class TestAlphaMoneyTool:
    @pytest.mark.asyncio
    async def test_dollar_thousands(self) -> None:
        tool = KaosLLMCoreAlphaMoneyTool()
        result = await tool.execute({"text": "Cap is $1,000,000 per claim."})
        s = _struct(result)
        first = s["spans"][0]
        assert first["amount"] == "1000000"
        assert first["currency"] == "USD"

    @pytest.mark.asyncio
    async def test_decimal_preserved(self) -> None:
        tool = KaosLLMCoreAlphaMoneyTool()
        result = await tool.execute({"text": "Paid $13.50 for lunch."})
        s = _struct(result)
        assert s["spans"][0]["amount"] == "13.50"


class TestAlphaNumberTool:
    @pytest.mark.asyncio
    async def test_arabic_and_written(self) -> None:
        tool = KaosLLMCoreAlphaNumberTool()
        result = await tool.execute({"text": "I worked 100 hours and read three books."})
        s = _struct(result)
        values = {span["value"] for span in s["spans"]}
        assert "100" in values
        assert "3" in values

    @pytest.mark.asyncio
    async def test_decimal_string_preserved(self) -> None:
        tool = KaosLLMCoreAlphaNumberTool()
        result = await tool.execute({"text": "Sold for 1,234.56 USD."})
        s = _struct(result)
        assert any(span["value"] == "1234.56" for span in s["spans"])


class TestAlphaPercentTool:
    @pytest.mark.asyncio
    async def test_percent_sign(self) -> None:
        tool = KaosLLMCoreAlphaPercentTool()
        result = await tool.execute({"text": "Rate is 5.25% per annum."})
        s = _struct(result)
        assert s["spans"][0]["value"] == "0.0525"

    @pytest.mark.asyncio
    async def test_basis_points(self) -> None:
        tool = KaosLLMCoreAlphaPercentTool()
        result = await tool.execute({"text": "Spread widened 50bps."})
        s = _struct(result)
        assert s["spans"][0]["value"] == "0.0050"


class TestAlphaDurationTool:
    @pytest.mark.asyncio
    async def test_days(self) -> None:
        tool = KaosLLMCoreAlphaDurationTool()
        result = await tool.execute({"text": "Notice of 90 days required."})
        s = _struct(result)
        first = s["spans"][0]
        assert first["quantity"] == "90"
        assert first["unit"] == "days"
        assert first["total_seconds"] == "7776000"

    @pytest.mark.asyncio
    async def test_written_quantity(self) -> None:
        tool = KaosLLMCoreAlphaDurationTool()
        result = await tool.execute({"text": "term of thirteen months."})
        s = _struct(result)
        first = s["spans"][0]
        assert first["quantity"] == "13"
        assert first["unit"] == "months"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_registers_six_alpha_tools(self) -> None:
        """register_llm_core_tools must include all 6 alpha tools."""
        from kaos_core import KaosRuntime

        from kaos_llm_core.integrations.mcp.registration import register_llm_core_tools

        # Fresh instance — KaosRuntime.default() is a process-wide
        # singleton; mutating it leaks into other test files.
        runtime = KaosRuntime()
        count = register_llm_core_tools(runtime)
        # At least 23 (pre-existing) + 6 (alpha) = 29.
        assert count >= 29
        registered = runtime.tools.list_tool_objects()
        registered_names = {t.metadata.name for t in registered}
        for name in [
            "kaos-llm-core-alpha-date",
            "kaos-llm-core-alpha-entity",
            "kaos-llm-core-alpha-money",
            "kaos-llm-core-alpha-number",
            "kaos-llm-core-alpha-percent",
            "kaos-llm-core-alpha-duration",
        ]:
            assert name in registered_names, f"missing tool: {name}"


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


class TestTruncation:
    @pytest.mark.asyncio
    async def test_max_results_caps_spans(self) -> None:
        """Pathological input that yields > ALPHA_MAX_RESULTS spans should
        be truncated and report has_more=True."""
        tool = KaosLLMCoreAlphaNumberTool()
        # Generate 300 numbers — should cap at ALPHA_MAX_RESULTS (256).
        text = " ".join(str(i) for i in range(300))
        result = await tool.execute({"text": text})
        s = _struct(result)
        assert s["total_matches"] == ALPHA_MAX_RESULTS
        assert s["has_more"] is True
