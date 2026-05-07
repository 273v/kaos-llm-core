"""Deprecation shim — moved to :mod:`kaos_nlp_core.extract.alpha`.

The rule-based alpha extractors moved to ``kaos_nlp_core`` in 2026-05.
They have no LLM dependency — only the kaos-nlp-core tokenizer and
locale gazetteers — so they belong with the other NLP primitives.
``AlphaLLMMerger`` (which DOES use LLM output) remains in
``kaos_llm_core.extract.merge``.

Migration::

    # before
    from kaos_llm_core.extract.alpha import AlphaDateExtractor

    # after
    from kaos_nlp_core.extract.alpha import AlphaDateExtractor
"""

from __future__ import annotations

import warnings

from kaos_nlp_core.extract.alpha import (
    AlphaDateExtractor,
    AlphaDurationExtractor,
    AlphaEntityExtractor,
    AlphaMoneyExtractor,
    AlphaNumberExtractor,
    AlphaPercentExtractor,
    DurationMatch,
    EntityMatch,
    MoneyMatch,
)

warnings.warn(
    "kaos_llm_core.extract.alpha moved to kaos_nlp_core.extract.alpha. Update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "AlphaDateExtractor",
    "AlphaDurationExtractor",
    "AlphaEntityExtractor",
    "AlphaMoneyExtractor",
    "AlphaNumberExtractor",
    "AlphaPercentExtractor",
    "DurationMatch",
    "EntityMatch",
    "MoneyMatch",
]
