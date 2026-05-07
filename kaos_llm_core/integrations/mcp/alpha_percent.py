# ruff: noqa: RUF001
"""KaosLLMCoreAlphaPercentTool — MCP wrapper for AlphaPercentExtractor (PR-6f.7)."""

from __future__ import annotations

from typing import Any, ClassVar

from kaos_core import KaosContext, ToolResult
from kaos_core.types.annotations import ToolAnnotations
from kaos_core.types.enums import ToolCapability, ToolCategory
from kaos_core.types.parameters import ParameterSchema
from kaos_nlp_core.extract.alpha import AlphaPercentExtractor

from kaos_llm_core.integrations.mcp._alpha_common import (
    ALPHA_ANNOTATIONS,
    ALPHA_MAX_RESULTS,
    ALPHA_TEXT_PARAM,
    coerce_decimal,
    common_text_input,
)
from kaos_llm_core.integrations.mcp._common import BaseLLMCoreTool

_EXTRACTOR = AlphaPercentExtractor()


class KaosLLMCoreAlphaPercentTool(BaseLLMCoreTool):
    """Extract percentage values, basis points, ppm/ppb — returns
    fractional Decimals (30% → 0.30)."""

    _NAME: ClassVar[str] = "kaos-llm-core-alpha-percent"
    _DISPLAY_NAME: ClassVar[str] = "Alpha Percent Extractor"
    _DESCRIPTION: ClassVar[str] = (
        "Deterministic, rule-based percentage extraction. Two branches: "
        "(1) symbol-suffix forms ('30%', '5.25%', '50bps', '10‰', "
        "'1‱', plus fullwidth ％ and small-form ﹪), (2) word forms "
        "('five percent', '100 basis points', '400 ppm', '5 ppb'). "
        "Returns FRACTIONAL Decimals: 30% → 0.30, 50bps → 0.005. "
        "Basis-point context guard: 'basis' only counts as a percent "
        "when followed by 'point'/'points'/'pts'. For percentages-of-"
        "percentages or percentage-point arithmetic, use "
        "kaos-llm-core-call with a typed Signature."
    )
    _CATEGORY: ClassVar[ToolCategory] = ToolCategory.UTILITY
    _CAPABILITY: ClassVar[ToolCapability] = ToolCapability.EXTRACT
    _ANNOTATIONS: ClassVar[ToolAnnotations] = ALPHA_ANNOTATIONS
    _PARAMETERS: ClassVar[list[ParameterSchema]] = [ALPHA_TEXT_PARAM]
    _ERROR_HINT: ClassVar[str] = (
        "Pass 'text' as a non-empty string. Alternative: use "
        "kaos-llm-core-call for percentage arithmetic / composition."
    )

    async def _run(
        self,
        inputs: dict[str, Any],
        context: KaosContext | None = None,
    ) -> ToolResult:
        text = common_text_input(inputs)
        if text is None:
            return ToolResult.create_error(
                "kaos-llm-core-alpha-percent requires 'text' as a string. "
                "Got: missing or non-string. " + self._ERROR_HINT
            )

        spans = []
        for span in _EXTRACTOR.extract_spans(text):
            spans.append(
                {
                    "value": coerce_decimal(span.value),
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
            summary=f"Found {len(spans)} percent value(s).",
        )
