"""Deprecation shim — moved to :mod:`kaos_nlp_core.extract.alpha.money`.

See :mod:`kaos_llm_core.extract.alpha` for the migration note. This
module re-exports the new symbols and emits a :class:`DeprecationWarning`.
"""

from __future__ import annotations

import warnings

from kaos_nlp_core.extract.alpha.money import AlphaMoneyExtractor, MoneyMatch

warnings.warn(
    "kaos_llm_core.extract.alpha.money moved to kaos_nlp_core.extract.alpha.money. "
    "Update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "AlphaMoneyExtractor",
    "MoneyMatch",
]
