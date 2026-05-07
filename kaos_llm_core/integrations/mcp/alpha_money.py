"""KaosLLMCoreAlphaMoneyTool — MCP wrapper for AlphaMoneyExtractor (PR-6f.7)."""

from __future__ import annotations

from typing import Any, ClassVar

from kaos_core import KaosContext, ToolResult
from kaos_core.types.annotations import ToolAnnotations
from kaos_core.types.enums import ToolCapability, ToolCategory
from kaos_core.types.parameters import ParameterSchema
from kaos_nlp_core.extract.alpha import AlphaMoneyExtractor

from kaos_llm_core.integrations.mcp._alpha_common import (
    ALPHA_ANNOTATIONS,
    ALPHA_MAX_RESULTS,
    ALPHA_TEXT_PARAM,
    coerce_decimal,
    common_text_input,
)
from kaos_llm_core.integrations.mcp._common import BaseLLMCoreTool

_EXTRACTOR = AlphaMoneyExtractor()


class KaosLLMCoreAlphaMoneyTool(BaseLLMCoreTool):
    """Extract monetary amounts (USD, EUR, GBP, JPY, etc.) and normalize
    to ISO 4217 currency codes."""

    _NAME: ClassVar[str] = "kaos-llm-core-alpha-money"
    _DISPLAY_NAME: ClassVar[str] = "Alpha Money Extractor"
    _DESCRIPTION: ClassVar[str] = (
        "Deterministic, rule-based money extraction. Recognizes "
        "currency-symbol prefix/suffix forms ($13.50, €250, 100$), "
        "currency-word forms ('ten dollars', '500 euros'), and "
        "indefinite articles ('a dollar' → 1). Always normalizes to "
        "ISO 4217 codes (USD, GBP, EUR, JPY, CNY, INR, KRW, CHF). "
        "Returns Decimal amounts as strings to preserve precision. "
        "Documented gaps: two-word currency phrases ('US dollars'), "
        "decorated quantities ('$1 million'), and redacted values "
        "('$[***]') are NOT extracted — for those, fall back to "
        "kaos-llm-core-call."
    )
    _CATEGORY: ClassVar[ToolCategory] = ToolCategory.UTILITY
    _CAPABILITY: ClassVar[ToolCapability] = ToolCapability.EXTRACT
    _ANNOTATIONS: ClassVar[ToolAnnotations] = ALPHA_ANNOTATIONS
    _PARAMETERS: ClassVar[list[ParameterSchema]] = [ALPHA_TEXT_PARAM]
    _ERROR_HINT: ClassVar[str] = (
        "Pass 'text' as a non-empty string. Alternative: use "
        "kaos-llm-core-call for non-USD-style currency phrases."
    )

    async def _run(
        self,
        inputs: dict[str, Any],
        context: KaosContext | None = None,
    ) -> ToolResult:
        text = common_text_input(inputs)
        if text is None:
            return ToolResult.create_error(
                "kaos-llm-core-alpha-money requires 'text' as a string. "
                "Got: missing or non-string. " + self._ERROR_HINT
            )

        spans = []
        for span in _EXTRACTOR.extract_spans(text):
            spans.append(
                {
                    "amount": coerce_decimal(span.value.amount),
                    "currency": span.value.currency,
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
            summary=f"Found {len(spans)} money amount(s).",
        )
