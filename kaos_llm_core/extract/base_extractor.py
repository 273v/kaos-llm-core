"""Deprecation shim — moved to :mod:`kaos_nlp_core.extract.base_extractor`.

The rule-based extractor base class and :class:`AlphaSpan` value type
moved to ``kaos_nlp_core.extract`` in 2026-05. They have no LLM
dependency, so they belong with the other NLP primitives. This module
re-exports the new symbols and emits a :class:`DeprecationWarning` on
import so existing callers know to update.

Migration::

    # before
    from kaos_llm_core.extract.base_extractor import (
        AlphaSpan, BaseAlphaExtractor, ExtractorValueType,
    )

    # after
    from kaos_nlp_core.extract.base_extractor import (
        AlphaSpan, BaseAlphaExtractor, ExtractorValueType,
    )
"""

from __future__ import annotations

import warnings

from kaos_nlp_core.extract.base_extractor import (
    AlphaSpan,
    BaseAlphaExtractor,
    ExtractorValueType,
)

warnings.warn(
    "kaos_llm_core.extract.base_extractor moved to "
    "kaos_nlp_core.extract.base_extractor. Update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "AlphaSpan",
    "BaseAlphaExtractor",
    "ExtractorValueType",
]
