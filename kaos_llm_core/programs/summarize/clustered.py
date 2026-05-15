"""Clustered long-document summarization (plan §6.1).

:class:`ClusteredSummary` reuses the long-doc scaffolding from
:class:`~kaos_llm_core.programs.summarize.long_doc._LongDocBase`
(chunk → leaf-summarize each chunk → cache + budget) but swaps the
default reducer for :class:`~kaos_llm_core.composition.reduce.Cluster`.
That reducer embeds every leaf summary, runs spherical k-means in
cosine space, merges within each cluster, then does a final
cross-cluster merge. The result is a single
:class:`~kaos_llm_core.results.Summary` that organises a noisy,
multi-topic source by *theme* rather than by source order — which is
what plan §6.1 lists this Program for ("good for multi-doc").
"""

from __future__ import annotations

from typing import Any, Literal

from kaos_llm_core.composition import Cluster, ClusterEmbedder, Reducer
from kaos_llm_core.programs.summarize.long_doc import _LongDocBase


class ClusteredSummary(_LongDocBase):
    """Long-doc summarizer that groups leaves by theme.

    Args:
        embedder: A :class:`ClusterEmbedder` (canonical:
            ``kaos_nlp_transformers.EmbeddingModel``). Used by the
            :class:`Cluster` reducer to embed leaf summaries and run
            spherical k-means. The same embedder protocol is
            consumed by
            :class:`~kaos_llm_core.programs.classify.PrototypeClassify`
            and
            :class:`~kaos_llm_core.programs.summarize.QueryFocusedSummary`.
        k: Number of clusters. ``"auto"`` (default) resolves to
            ``max(2, min(round(sqrt(n)), 8))`` at reduce-time, where
            ``n`` is the number of non-empty leaves.
        max_iter: Hard cap on Lloyd's-algorithm iterations passed to
            the :class:`Cluster` reducer. Default ``25``.
        seed: PRNG seed for the reducer's initial-centroid pick.
            Default ``42``.

        Any additional keyword arguments forward to
        :class:`_LongDocBase` — ``chunker``, ``model``, ``codec``,
        ``client``, ``settings``, ``core_settings``, ``examples``,
        ``instructions``, ``max_retries``, ``leaf_summarizer``,
        ``max_concurrency``, ``cache``, ``budget``.
    """

    program_name: str = "ClusteredSummary"

    def __init__(
        self,
        *,
        embedder: ClusterEmbedder,
        k: int | Literal["auto"] = "auto",
        max_iter: int = 25,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._embedder = embedder
        self._k = k
        self._max_iter = max_iter
        self._seed = seed

    def _reducer(self) -> Reducer:
        return Cluster(
            embedder=self._embedder,
            k=self._k,
            max_iter=self._max_iter,
            seed=self._seed,
        )


__all__ = ["ClusteredSummary"]
