"""Search algorithms for optimizers — Phase 17.1.

This package holds standalone, swappable search algorithms that
optimizers can plug into. The first member is :class:`TPESearcher`,
a minimal categorical Tree-structured Parzen Estimator used by
:class:`MiproV2Optimizer` to search the joint (instruction x demos)
space per predictor.

Future searchers (random, grid, gaussian-process, optuna-bridge)
should implement the :class:`CategoricalSearcher` protocol so they
are swappable in one line.
"""

from __future__ import annotations

from kaos_llm_core.optimization.search.tpe import (
    CategoricalSearcher,
    TPESearcher,
    TPESearcherConfig,
)

__all__ = [
    "CategoricalSearcher",
    "TPESearcher",
    "TPESearcherConfig",
]
