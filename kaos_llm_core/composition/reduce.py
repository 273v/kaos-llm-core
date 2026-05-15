"""Reducer strategies for long-document summarization.

A :class:`Reducer` organizes the recursion that combines many leaf
:class:`~kaos_llm_core.results.Summary` instances into one
document-level summary. It does **not** generate text itself — the
caller supplies an async ``merge_fn`` that takes a group of summaries
and returns one merged summary. The reducer is responsible for
deciding how to *group* the leaves.

Three concrete strategies ship at Phase 2:

- :class:`MapReduce` — single merge call across all leaves.
- :class:`Refine` — left-to-right walk, applying ``merge_fn`` to the
  running summary plus each new leaf.
- :class:`Tree` — k-ary bottom-up reduction with bounded branching
  factor and depth.

A clustering reducer (``Cluster``) is deferred to Phase 5 since it
depends on the embeddings exposed by ``kaos-nlp-transformers``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol, runtime_checkable

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


__all__ = [
    "MapReduce",
    "MergeFn",
    "Reducer",
    "Refine",
    "Tree",
]
