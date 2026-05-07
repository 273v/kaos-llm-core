"""Rule-based (alpha) extractors.

Parallel to :mod:`kaos_llm_core.programs` — the alpha extractors are
deterministic, LLM-free, and designed to be UNIONED with LLM output in
the WS-TR Extract pipeline rather than replace it. See
``docs/design/ws-tr-alpha-extractor-sprint.md`` for the full rationale
and the layer decomposition. The pattern ports kelvin-nlp's
``kelvin.nlp.extract.alpha`` tree.

Usage::

    from kaos_nlp_core.extract.alpha import AlphaDateExtractor

    extractor = AlphaDateExtractor()
    for span in extractor.extract_spans("Agreement dated February 17, 1999."):
        print(span.value, span.start, span.end)

The merge composition with the LLM Extract program lands in PR-6f.3.
"""

from __future__ import annotations

from kaos_nlp_core.extract.base_extractor import (
    AlphaSpan,
    BaseAlphaExtractor,
    ExtractorValueType,
)

__all__ = [
    "AlphaSpan",
    "BaseAlphaExtractor",
    "ExtractorValueType",
]
