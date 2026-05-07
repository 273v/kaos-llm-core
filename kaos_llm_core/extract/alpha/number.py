"""Deprecation shim — moved to :mod:`kaos_nlp_core.extract.alpha.number`.

See :mod:`kaos_llm_core.extract.alpha` for the migration note. This
module re-exports the new symbols and emits a :class:`DeprecationWarning`.
"""

from __future__ import annotations

import warnings

from kaos_nlp_core.extract.alpha.number import AlphaNumberExtractor

warnings.warn(
    "kaos_llm_core.extract.alpha.number moved to kaos_nlp_core.extract.alpha.number. "
    "Update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "AlphaNumberExtractor",
]
