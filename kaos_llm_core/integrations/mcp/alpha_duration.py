"""KaosLLMCoreAlphaDurationTool — MCP wrapper for AlphaDurationExtractor (PR-6f.7)."""

from __future__ import annotations

from typing import Any, ClassVar

from kaos_core import KaosContext, ToolResult
from kaos_core.types.annotations import ToolAnnotations
from kaos_core.types.enums import ToolCapability, ToolCategory
from kaos_core.types.parameters import ParameterSchema
from kaos_nlp_core.extract.alpha import AlphaDurationExtractor

from kaos_llm_core.integrations.mcp._alpha_common import (
    ALPHA_ANNOTATIONS,
    ALPHA_MAX_RESULTS,
    ALPHA_TEXT_PARAM,
    coerce_decimal,
    common_text_input,
)
from kaos_llm_core.integrations.mcp._common import BaseLLMCoreTool

_EXTRACTOR = AlphaDurationExtractor()


class KaosLLMCoreAlphaDurationTool(BaseLLMCoreTool):
    """Extract time durations and normalize to (quantity, unit, total_seconds)."""

    _NAME: ClassVar[str] = "kaos-llm-core-alpha-duration"
    _DISPLAY_NAME: ClassVar[str] = "Alpha Duration Extractor"
    _DESCRIPTION: ClassVar[str] = (
        "Deterministic, rule-based duration extraction. Recognizes "
        "arabic + word quantities ('90 days', 'thirteen months'), "
        "indefinite articles ('a year' → 1), and the full unit "
        "vocabulary (seconds, minutes, hours, days, weeks, months, "
        "years, anniversary). Returns each match as a "
        "(quantity, unit, total_seconds) tuple. Calendar approximations: "
        "month ≈ 30 days, year ≈ 365 days. "
        "Documented gaps: 'business days' / 'calendar months' / "
        "'working days' modifier tokens are intentionally skipped "
        "(kelvin-faithful) — the LLM tier handles those distinctions."
    )
    _CATEGORY: ClassVar[ToolCategory] = ToolCategory.UTILITY
    _CAPABILITY: ClassVar[ToolCapability] = ToolCapability.EXTRACT
    _ANNOTATIONS: ClassVar[ToolAnnotations] = ALPHA_ANNOTATIONS
    _PARAMETERS: ClassVar[list[ParameterSchema]] = [ALPHA_TEXT_PARAM]
    _ERROR_HINT: ClassVar[str] = (
        "Pass 'text' as a non-empty string. Alternative: use "
        "kaos-llm-core-call for business-day / calendar-day distinctions."
    )

    async def _run(
        self,
        inputs: dict[str, Any],
        context: KaosContext | None = None,
    ) -> ToolResult:
        text = common_text_input(inputs)
        if text is None:
            return ToolResult.create_error(
                "kaos-llm-core-alpha-duration requires 'text' as a string. "
                "Got: missing or non-string. " + self._ERROR_HINT
            )

        spans = []
        for span in _EXTRACTOR.extract_spans(text):
            spans.append(
                {
                    "quantity": coerce_decimal(span.value.quantity),
                    "unit": span.value.unit,
                    "total_seconds": coerce_decimal(span.value.total_seconds),
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
            summary=f"Found {len(spans)} duration(s).",
        )
