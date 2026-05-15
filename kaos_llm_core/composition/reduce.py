"""Reducer strategies for long-document summarization.

A :class:`Reducer` organizes the recursion that combines many leaf
:class:`~kaos_llm_core.results.Summary` instances into one
document-level summary. It does **not** generate text itself — the
caller supplies an async ``merge_fn`` that takes a group of summaries
and returns one merged summary. The reducer is responsible for
deciding how to *group* the leaves.

Four concrete strategies:

- :class:`MapReduce` — single merge call across all leaves.
- :class:`Refine` — left-to-right walk, applying ``merge_fn`` to the
  running summary plus each new leaf.
- :class:`Tree` — k-ary bottom-up reduction with bounded branching
  factor and depth.
- :class:`Cluster` — embed each leaf, k-means in cosine space,
  per-cluster merge then final merge. Good for multi-doc inputs
  where source order is not the natural grouping axis. Shipped in
  0.1.0a10 (plan §5.1 Phase-2 leftover).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Sequence
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from kaos_llm_core.results import Summary

# A ``MergeFn`` is the async callable that fuses a group of leaf
# summaries into one. Programs supply this; reducers call it.
MergeFn = Callable[[Sequence[Summary]], Awaitable[Summary]]


@runtime_checkable
class Reducer(Protocol):
    """Strategy for combining many leaf summaries into one.

    Implementations are async because the underlying ``merge_fn`` is
    nearly always an LLM call. Implementations should run sibling
    merges concurrently when the strategy allows it (Map and Tree
    can; Refine cannot).
    """

    async def reduce(
        self,
        leaves: Sequence[Summary],
        merge_fn: MergeFn,
    ) -> Summary:
        """Combine ``leaves`` into a single :class:`Summary`.

        Args:
            leaves: Per-chunk summaries in source order. Must contain
                at least one element.
            merge_fn: Async callable that produces a single
                :class:`Summary` from a sequence of summaries. The
                reducer never modifies summary text itself; all
                generation happens here.

        Returns:
            The reduced summary. Its ``depth`` field reflects the
            number of merge levels the reducer applied
            (e.g., depth ``1`` for a single map-reduce pass,
            depth ``log_k(n)`` for a k-ary tree).
        """
        ...


def _pool_chunks_used(leaves: Sequence[Summary], current: list[str]) -> list[str]:
    """Merge ``chunks_used`` from ``leaves`` into ``current`` preserving order."""
    seen = set(current)
    pooled = list(current)
    for leaf in leaves:
        for chunk_id in leaf.chunks_used:
            if chunk_id not in seen:
                pooled.append(chunk_id)
                seen.add(chunk_id)
    return pooled


class MapReduce:
    """Single-call reducer: produce one merged summary across all leaves.

    Cheapest and simplest reducer. Use when the leaves' combined
    token count comfortably fits the reduce model's context window.
    """

    async def reduce(
        self,
        leaves: Sequence[Summary],
        merge_fn: MergeFn,
    ) -> Summary:
        if not leaves:
            raise ValueError("MapReduce requires at least one leaf summary")
        merged = await merge_fn(list(leaves))
        # Preserve merge_fn's output but enrich provenance and
        # bump depth.
        pooled_chunks = _pool_chunks_used(leaves, list(merged.chunks_used))
        return merged.model_copy(
            update={
                "chunks_used": pooled_chunks,
                "depth": max(merged.depth, max(leaf.depth for leaf in leaves) + 1),
                "metadata": {
                    **dict(merged.metadata),
                    "reducer": "MapReduce",
                    "reducer.input_leaves": len(leaves),
                },
            }
        )


class Refine:
    """Sequential reducer: walk leaves left-to-right with a running summary.

    On each step, the reducer asks ``merge_fn`` to fuse the current
    running summary with the next leaf. Preserves narrative order at
    the cost of serialization — every step waits for the previous one.
    """

    async def reduce(
        self,
        leaves: Sequence[Summary],
        merge_fn: MergeFn,
    ) -> Summary:
        if not leaves:
            raise ValueError("Refine requires at least one leaf summary")
        if len(leaves) == 1:
            only = leaves[0]
            return only.model_copy(
                update={
                    "depth": only.depth + 1,
                    "metadata": {
                        **dict(only.metadata),
                        "reducer": "Refine",
                        "reducer.input_leaves": 1,
                    },
                }
            )
        running = leaves[0]
        for index, leaf in enumerate(leaves[1:], start=1):
            running = await merge_fn([running, leaf])
            running = running.model_copy(
                update={
                    "metadata": {
                        **dict(running.metadata),
                        "reducer": "Refine",
                        "reducer.step": index,
                    },
                }
            )
        pooled_chunks = _pool_chunks_used(leaves, list(running.chunks_used))
        return running.model_copy(
            update={
                "chunks_used": pooled_chunks,
                "depth": max(running.depth, max(leaf.depth for leaf in leaves) + 1),
                "metadata": {
                    **dict(running.metadata),
                    "reducer": "Refine",
                    "reducer.input_leaves": len(leaves),
                },
            }
        )


class Tree:
    """K-ary bottom-up reducer.

    Groups leaves into batches of ``branching`` and merges each batch
    concurrently to produce a smaller list of summaries. Repeats until
    one summary remains or until ``max_depth`` levels have been
    applied.

    Args:
        branching: Group size at every level. Must be >= 2. Default
            ``4``.
        max_depth: Hard cap on tree depth. When reached, the
            remaining summaries are merged in a final pass regardless
            of count. Default ``8`` (effectively unbounded for any
            realistic document, since ``4 ** 8 == 65_536`` chunks).
    """

    def __init__(self, *, branching: int = 4, max_depth: int = 8) -> None:
        if branching < 2:
            raise ValueError(f"branching must be >= 2, got {branching}")
        if max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")
        self.branching = branching
        self.max_depth = max_depth

    async def reduce(
        self,
        leaves: Sequence[Summary],
        merge_fn: MergeFn,
    ) -> Summary:
        if not leaves:
            raise ValueError("Tree requires at least one leaf summary")
        if len(leaves) == 1:
            only = leaves[0]
            return only.model_copy(
                update={
                    "depth": only.depth + 1,
                    "metadata": {
                        **dict(only.metadata),
                        "reducer": "Tree",
                        "reducer.input_leaves": 1,
                    },
                }
            )

        current: list[Summary] = list(leaves)
        depth_applied = 0
        while len(current) > 1 and depth_applied < self.max_depth:
            groups = [
                current[i : i + self.branching] for i in range(0, len(current), self.branching)
            ]
            # If the last group has only one summary, fold it into the
            # previous group so we don't waste an LLM call on a
            # single-element merge.
            if len(groups) >= 2 and len(groups[-1]) == 1:
                tail = groups.pop()
                groups[-1] = list(groups[-1]) + list(tail)
            merged_groups = await asyncio.gather(*(merge_fn(list(g)) for g in groups))
            depth_applied += 1
            current = list(merged_groups)
            if len(groups) == 1:
                break

        # If we still have more than one summary (max_depth reached),
        # do a final flat merge.
        if len(current) > 1:
            current = [await merge_fn(current)]
            depth_applied += 1

        result = current[0]
        pooled_chunks = _pool_chunks_used(leaves, list(result.chunks_used))
        return result.model_copy(
            update={
                "chunks_used": pooled_chunks,
                "depth": max(result.depth, max(leaf.depth for leaf in leaves) + depth_applied),
                "metadata": {
                    **dict(result.metadata),
                    "reducer": "Tree",
                    "reducer.input_leaves": len(leaves),
                    "reducer.branching": self.branching,
                    "reducer.depth_applied": depth_applied,
                },
            }
        )


@runtime_checkable
class ClusterEmbedder(Protocol):
    """Embedding backend consumed by :class:`Cluster`.

    Same shape as the
    :class:`~kaos_llm_core.programs.classify.prototype.Embedder`
    protocol — returns a ``(N, dim)`` ``float32`` numpy array. Rows
    should be L2-normalised so the spherical-k-means inner loop runs
    on unit vectors; :class:`Cluster` defends with a normalise pass
    on the returned matrix.
    """

    def embed(
        self, texts: Iterable[str], *, batch_size: int = 32
    ) -> np.ndarray:  # pragma: no cover - protocol
        ...


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Return ``matrix`` with every row L2-normalised; ``copy=False`` when possible."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (matrix / norms).astype(np.float32, copy=False)


def _spherical_kmeans(
    embeddings: np.ndarray,
    k: int,
    *,
    max_iter: int,
    seed: int,
) -> np.ndarray:
    """k-means in cosine space — Lloyd's algorithm on unit vectors.

    Inputs must be L2-normalised (the caller is responsible). The
    inner loop maximises ``embeddings @ centroids.T`` (cosine
    similarity) instead of minimising squared L2 distance. Centroids
    are re-normalised after each step so they stay unit-norm. The
    initialisation is deterministic via ``seed``: pick the first
    centroid uniformly, then each subsequent centroid as the row that
    is *least similar* to the existing centroids (a cheap k-means++
    cousin that avoids the random-weighted-sample cost).

    Returns:
        ``assignments`` of shape ``(n,)`` — the cluster index for
        every row of ``embeddings``.
    """
    n = embeddings.shape[0]
    rng = np.random.default_rng(seed)
    # Initial centroid: a uniformly-chosen row.
    first = int(rng.integers(0, n))
    centroid_idxs: list[int] = [first]
    # Greedy "farthest from any existing centroid" picks.
    while len(centroid_idxs) < k:
        sim = embeddings @ embeddings[centroid_idxs].T  # (n, k_so_far)
        max_sim = sim.max(axis=1)
        # Pick the row least similar to any existing centroid; break
        # ties deterministically by smallest index.
        next_idx = int(np.argmin(max_sim))
        if next_idx in centroid_idxs:
            # All remaining rows already centroid; bail.
            break
        centroid_idxs.append(next_idx)

    centroids = embeddings[centroid_idxs].copy()

    assignments = np.zeros(n, dtype=np.int64)
    for _ in range(max_iter):
        sims = embeddings @ centroids.T  # (n, k)
        new_assignments = np.argmax(sims, axis=1)
        if np.array_equal(new_assignments, assignments):
            break
        assignments = new_assignments
        # Recompute centroids as the L2-normalised sum of cluster
        # members. Empty clusters keep their previous centroid.
        new_centroids = np.zeros_like(centroids)
        for cluster_idx in range(len(centroids)):
            mask = assignments == cluster_idx
            if not mask.any():
                new_centroids[cluster_idx] = centroids[cluster_idx]
                continue
            mean = embeddings[mask].sum(axis=0)
            norm = float(np.linalg.norm(mean))
            new_centroids[cluster_idx] = mean / norm if norm > 0 else centroids[cluster_idx]
        centroids = new_centroids
    return assignments


class Cluster:
    """Cluster reducer: embed → k-means → per-cluster merge → final merge.

    Workflow:

    1. Build a list of ``(text, leaf)`` pairs. The clustering text
       is the leaf's :attr:`Summary.text` — leaves with empty text
       are coalesced into a single "(empty)" cluster.
    2. Embed every text via ``embedder.embed(...)``. L2-normalise the
       returned rows defensively.
    3. Run spherical k-means with the chosen ``k`` (or the
       ``"auto"`` heuristic) for at most ``max_iter`` passes; seed is
       fixed so the assignment is reproducible.
    4. For each non-empty cluster, call ``merge_fn`` on the leaves in
       source order (sorted by ``Summary.chunks_used`` lexicographically
       when present, otherwise input order).
    5. Final ``merge_fn`` over the per-cluster summaries, in cluster-id
       order, produces the document-level summary.

    Args:
        embedder: Object conforming to :class:`ClusterEmbedder`. The
            canonical implementation is
            :class:`kaos_nlp_transformers.EmbeddingModel`.
        k: Number of clusters. ``"auto"`` (default) resolves to
            ``max(2, min(round(sqrt(n)), 8))``.
        max_iter: Hard cap on Lloyd's-algorithm iterations. Default
            ``25`` — k-means in cosine space converges quickly on
            most inputs.
        seed: PRNG seed for the initial-centroid pick. Default ``42``.

    Notes:
        This reducer needs at least one leaf with non-empty text. A
        run with all-empty leaves degenerates to a single-cluster
        merge (the leaves go through ``merge_fn`` once).
    """

    def __init__(
        self,
        *,
        embedder: ClusterEmbedder,
        k: int | Literal["auto"] = "auto",
        max_iter: int = 25,
        seed: int = 42,
    ) -> None:
        if isinstance(k, int) and k < 1:
            raise ValueError(f"k must be >= 1 or 'auto', got {k}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        self._embedder = embedder
        self._k_request = k
        self._max_iter = max_iter
        self._seed = seed

    @staticmethod
    def _resolve_k(n_leaves: int, requested: int | Literal["auto"]) -> int:
        if requested == "auto":
            return max(2, min(round(n_leaves**0.5), 8))
        # Cap k at the number of available rows; nothing to cluster
        # otherwise.
        return min(int(requested), n_leaves)

    async def reduce(
        self,
        leaves: Sequence[Summary],
        merge_fn: MergeFn,
    ) -> Summary:
        if not leaves:
            raise ValueError("Cluster requires at least one leaf summary")
        if len(leaves) == 1:
            only = leaves[0]
            return only.model_copy(
                update={
                    "depth": only.depth + 1,
                    "metadata": {
                        **dict(only.metadata),
                        "reducer": "Cluster",
                        "reducer.input_leaves": 1,
                        "reducer.k": 1,
                    },
                }
            )

        texts = [leaf.text for leaf in leaves]
        non_empty_idxs = [i for i, t in enumerate(texts) if t]
        if not non_empty_idxs:
            # Degenerate case: everything is empty. Single-cluster
            # merge so the caller still gets a Summary out.
            merged = await merge_fn(list(leaves))
            pooled_chunks = _pool_chunks_used(leaves, list(merged.chunks_used))
            return merged.model_copy(
                update={
                    "chunks_used": pooled_chunks,
                    "depth": max(merged.depth, max(leaf.depth for leaf in leaves) + 1),
                    "metadata": {
                        **dict(merged.metadata),
                        "reducer": "Cluster",
                        "reducer.input_leaves": len(leaves),
                        "reducer.k": 1,
                        "reducer.degenerate": "all_empty",
                    },
                }
            )

        k = self._resolve_k(len(non_empty_idxs), self._k_request)
        if k <= 1:
            # Same shape as MapReduce — one flat merge.
            merged = await merge_fn(list(leaves))
            pooled_chunks = _pool_chunks_used(leaves, list(merged.chunks_used))
            return merged.model_copy(
                update={
                    "chunks_used": pooled_chunks,
                    "depth": max(merged.depth, max(leaf.depth for leaf in leaves) + 1),
                    "metadata": {
                        **dict(merged.metadata),
                        "reducer": "Cluster",
                        "reducer.input_leaves": len(leaves),
                        "reducer.k": 1,
                    },
                }
            )

        # Embed only the non-empty leaves; empties get assigned to the
        # nearest cluster post-hoc (we put them all in cluster 0 here).
        embeddings = self._embedder.embed([texts[i] for i in non_empty_idxs])
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        embeddings = _l2_normalize_rows(embeddings)
        non_empty_assignments = _spherical_kmeans(
            embeddings,
            k=k,
            max_iter=self._max_iter,
            seed=self._seed,
        )
        full_assignments = np.zeros(len(leaves), dtype=np.int64)
        for slot, leaf_idx in enumerate(non_empty_idxs):
            full_assignments[leaf_idx] = int(non_empty_assignments[slot])
        # Empties land in cluster 0; that's deterministic and harmless.

        clusters: list[list[Summary]] = [[] for _ in range(k)]
        for idx, cluster_id in enumerate(full_assignments):
            clusters[int(cluster_id)].append(leaves[idx])

        # Merge each non-empty cluster concurrently; preserve cluster-id
        # order so the final-merge inputs are deterministic.
        non_empty_clusters: list[list[Summary]] = [c for c in clusters if c]
        cluster_summaries = await asyncio.gather(*(merge_fn(list(c)) for c in non_empty_clusters))

        # Final merge across the per-cluster summaries.
        if len(cluster_summaries) == 1:
            root = cluster_summaries[0]
        else:
            root = await merge_fn(list(cluster_summaries))

        pooled_chunks = _pool_chunks_used(leaves, list(root.chunks_used))
        return root.model_copy(
            update={
                "chunks_used": pooled_chunks,
                "depth": max(root.depth, max(leaf.depth for leaf in leaves) + 2),
                "metadata": {
                    **dict(root.metadata),
                    "reducer": "Cluster",
                    "reducer.input_leaves": len(leaves),
                    "reducer.k": len(non_empty_clusters),
                    "reducer.k_requested": (
                        self._k_request if self._k_request != "auto" else "auto"
                    ),
                    "reducer.cluster_sizes": [len(c) for c in non_empty_clusters],
                },
            }
        )


__all__ = [
    "Cluster",
    "ClusterEmbedder",
    "MapReduce",
    "MergeFn",
    "Reducer",
    "Refine",
    "Tree",
]
