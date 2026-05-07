"""KaosLLMCoreAlphaDateTool — MCP wrapper for AlphaDateExtractor (PR-6f.7)."""

from __future__ import annotations

from typing import Any, ClassVar

from kaos_core import KaosContext, ToolResult
from kaos_core.types.annotations import ToolAnnotations
from kaos_core.types.enums import ToolCapability, ToolCategory
from kaos_core.types.parameters import ParameterSchema
from kaos_nlp_core.extract.alpha import AlphaDateExtractor

from kaos_llm_core.integrations.mcp._alpha_common import (
    ALPHA_ANNOTATIONS,
    ALPHA_MAX_RESULTS,
    ALPHA_TEXT_PARAM,
    common_text_input,
)
from kaos_llm_core.integrations.mcp._common import BaseLLMCoreTool

_EXTRACTOR = AlphaDateExtractor()


class KaosLLMCoreAlphaDateTool(BaseLLMCoreTool):
    """Extract calendar dates from text using a deterministic regex +
    gazetteer pipeline (no LLM)."""

    _NAME: ClassVar[str] = "kaos-llm-core-alpha-date"
    _DISPLAY_NAME: ClassVar[str] = "Alpha Date Extractor"
    _DESCRIPTION: ClassVar[str] = (
        "Deterministic, rule-based date extraction from text. Recognizes "
        "MM/DD/YYYY separators, 'Month DD, YYYY', 'DD Month YYYY', "
        "ordinal forms ('21st of January 2003'), and the English legal "
        "'Nth day of Month YYYY' phrasing. Returns ISO 8601 dates "
        "alongside character offsets into the source text. "
        "Use this tool BEFORE asking an LLM for a date — it never "
        "hallucinates and is much cheaper. Pair with "
        "kaos-llm-core-call when the date format is so unusual that the "
        "alpha extractor returns no matches."
    )
    _CATEGORY: ClassVar[ToolCategory] = ToolCategory.UTILITY
    _CAPABILITY: ClassVar[ToolCapability] = ToolCapability.EXTRACT
    _ANNOTATIONS: ClassVar[ToolAnnotations] = ALPHA_ANNOTATIONS
    _PARAMETERS: ClassVar[list[ParameterSchema]] = [ALPHA_TEXT_PARAM]
    _ERROR_HINT: ClassVar[str] = (
        "Pass 'text' as a non-empty string. Alternative: use "
        "kaos-llm-core-call with a typed Signature for free-form date extraction."
    )

    async def _run(
        self,
        inputs: dict[str, Any],
        context: KaosContext | None = None,
    ) -> ToolResult:
        text = common_text_input(inputs)
        if text is None:
            return ToolResult.create_error(
                "kaos-llm-core-alpha-date requires 'text' as a string. "
                "Got: missing or non-string. " + self._ERROR_HINT
            )

        spans = []
        for span in _EXTRACTOR.extract_spans(text):
            # AlphaDateExtractor returns datetime; serialize as ISO 8601
            # date only (no time component) — calendar dates from
            # contracts don't carry a meaningful time-of-day.
            value = span.value
            iso = value.date().isoformat() if hasattr(value, "date") else value.isoformat()
            spans.append(
                {
                    "value": iso,
                    "start": span.start,
                    "end": span.end,
                    "snippet": text[span.start : span.end],
                }
            )
            if len(spans) >= ALPHA_MAX_RESULTS:
                break

        return ToolResult.create_success(
            output={
                "spans": spans,
                "total_matches": len(spans),
                "has_more": len(spans) >= ALPHA_MAX_RESULTS,
            },
            summary=f"Found {len(spans)} date span(s).",
        )
