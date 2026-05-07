"""KaosLLMCoreAlphaEntityTool — MCP wrapper for AlphaEntityExtractor (PR-6f.7)."""

from __future__ import annotations

from typing import Any, ClassVar

from kaos_core import KaosContext, ToolResult
from kaos_core.types.annotations import ToolAnnotations
from kaos_core.types.enums import ToolCapability, ToolCategory
from kaos_core.types.parameters import ParameterSchema
from kaos_nlp_core.extract.alpha import AlphaEntityExtractor

from kaos_llm_core.integrations.mcp._alpha_common import (
    ALPHA_ANNOTATIONS,
    ALPHA_MAX_RESULTS,
    ALPHA_TEXT_PARAM,
    common_text_input,
)
from kaos_llm_core.integrations.mcp._common import BaseLLMCoreTool

_EXTRACTOR = AlphaEntityExtractor()


class KaosLLMCoreAlphaEntityTool(BaseLLMCoreTool):
    """Extract corporate-entity names via a 108-entry suffix gazetteer
    (Inc, LLC, GmbH, SARL, K.K., etc.)."""

    _NAME: ClassVar[str] = "kaos-llm-core-alpha-entity"
    _DISPLAY_NAME: ClassVar[str] = "Alpha Entity Extractor"
    _DESCRIPTION: ClassVar[str] = (
        "Deterministic, rule-based corporate-entity extraction. Identifies "
        "company names by detecting common entity-type suffixes "
        "(Inc, LLC, Corp, GmbH, SARL, S.p.A., K.K., AB, plc, and 100+ "
        "more across 9 language families) and scanning backward to "
        "collect the capitalized name tokens. Returns (name, entity_type) "
        "pairs with character offsets. "
        "Documented gaps: hyphenated names ('Coca-Cola'), lowercase-first "
        "names ('eBay'), and suffix-less mentions ('Acme' alone) are not "
        "caught — for those, fall back to kaos-llm-core-call with a "
        "typed Signature."
    )
    _CATEGORY: ClassVar[ToolCategory] = ToolCategory.UTILITY
    _CAPABILITY: ClassVar[ToolCapability] = ToolCapability.EXTRACT
    _ANNOTATIONS: ClassVar[ToolAnnotations] = ALPHA_ANNOTATIONS
    _PARAMETERS: ClassVar[list[ParameterSchema]] = [ALPHA_TEXT_PARAM]
    _ERROR_HINT: ClassVar[str] = (
        "Pass 'text' as a non-empty string. Alternative: use "
        "kaos-llm-core-call for entity types not in the gazetteer."
    )

    async def _run(
        self,
        inputs: dict[str, Any],
        context: KaosContext | None = None,
    ) -> ToolResult:
        text = common_text_input(inputs)
        if text is None:
            return ToolResult.create_error(
                "kaos-llm-core-alpha-entity requires 'text' as a string. "
                "Got: missing or non-string. " + self._ERROR_HINT
            )

        spans = []
        for span in _EXTRACTOR.extract_spans(text):
            spans.append(
                {
                    "name": span.value.name,
                    "entity_type": span.value.entity_type,
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
            summary=f"Found {len(spans)} entity match(es).",
        )
