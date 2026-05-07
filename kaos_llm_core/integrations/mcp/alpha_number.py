"""KaosLLMCoreAlphaNumberTool — MCP wrapper for AlphaNumberExtractor (PR-6f.7)."""

from __future__ import annotations

from typing import Any, ClassVar

from kaos_core import KaosContext, ToolResult
from kaos_core.types.annotations import ToolAnnotations
from kaos_core.types.enums import ToolCapability, ToolCategory
from kaos_core.types.parameters import ParameterSchema
from kaos_nlp_core.extract.alpha import AlphaNumberExtractor

from kaos_llm_core.integrations.mcp._alpha_common import (
    ALPHA_ANNOTATIONS,
    ALPHA_MAX_RESULTS,
    ALPHA_TEXT_PARAM,
    coerce_decimal,
    common_text_input,
)
from kaos_llm_core.integrations.mcp._common import BaseLLMCoreTool

_EXTRACTOR = AlphaNumberExtractor()


class KaosLLMCoreAlphaNumberTool(BaseLLMCoreTool):
    """Extract numeric values from text — Arabic, Roman, or written."""

    _NAME: ClassVar[str] = "kaos-llm-core-alpha-number"
    _DISPLAY_NAME: ClassVar[str] = "Alpha Number Extractor"
    _DESCRIPTION: ClassVar[str] = (
        "Deterministic, rule-based numeric extraction. Three branches: "
        "(1) Arabic numbers with English-locale radix ('1,234.56'), "
        "(2) Roman numerals ('IV', 'XIV') — bare 'I' suppressed to "
        "avoid first-person-pronoun false positives, (3) written numbers "
        "('twenty-three' → 23 via additive hyphenation). Returns Decimal "
        "values as strings (10-digit precision, banker's rounding). "
        "Documented gaps: 'two-hundred' → 102 not 200 (kelvin-faithful "
        "additive), and 'one hundred and five' returns three separate "
        "spans. For free-form numeric extraction with composition, "
        "use kaos-llm-core-call."
    )
    _CATEGORY: ClassVar[ToolCategory] = ToolCategory.UTILITY
    _CAPABILITY: ClassVar[ToolCapability] = ToolCapability.EXTRACT
    _ANNOTATIONS: ClassVar[ToolAnnotations] = ALPHA_ANNOTATIONS
    _PARAMETERS: ClassVar[list[ParameterSchema]] = [ALPHA_TEXT_PARAM]
    _ERROR_HINT: ClassVar[str] = (
        "Pass 'text' as a non-empty string. Alternative: use "
        "kaos-llm-core-call for compositional numeric extraction."
    )

    async def _run(
        self,
        inputs: dict[str, Any],
        context: KaosContext | None = None,
    ) -> ToolResult:
        text = common_text_input(inputs)
        if text is None:
            return ToolResult.create_error(
                "kaos-llm-core-alpha-number requires 'text' as a string. "
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
            summary=f"Found {len(spans)} number(s).",
        )
