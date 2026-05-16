"""Composition primitives for summarization and classification programs.

Two protocols and their concrete strategies:

- :class:`Aggregator` — combine multiple per-chunk
  :class:`~kaos_llm_core.results.Classification` results into a single
  document-level decision.
- :class:`Reducer` — combine multiple per-chunk
  :class:`~kaos_llm_core.results.Summary` instances into a single
  document-level summary, optionally using an async merge function for
  the actual generation step.

The strategies are pure orchestration. They contain no LLM calls
themselves; the LLM work is delegated to callables passed in by the
program. This keeps the composers unit-testable with deterministic
stubs and avoids the strategies needing knowledge of provider
selection, retries, caching, or budgets.
"""

from __future__ import annotations

from kaos_llm_core.composition.aggregate import (
    Aggregator,
    IntersectionAggregator,
    MajorityAggregator,
    MaxScoreAggregator,
    UnionAggregator,
    VoteAggregator,
    WeightedAggregator,
    resolve_aggregator,
)
from kaos_llm_core.composition.reduce import (
    Cluster,
    ClusterEmbedder,
    MapReduce,
    Reducer,
    Refine,
    Tree,
)

__all__ = [
    "Aggregator",
    "Cluster",
    "ClusterEmbedder",
    "IntersectionAggregator",
    "MajorityAggregator",
    "MapReduce",
    "MaxScoreAggregator",
    "Reducer",
    "Refine",
    "Tree",
    "UnionAggregator",
    "VoteAggregator",
    "WeightedAggregator",
    "resolve_aggregator",
]
