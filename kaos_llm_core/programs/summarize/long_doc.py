"""Long-document summarization Programs.

Three Programs ship at Phase 3:

- :class:`MapReduceSummary` — chunk → summarize each chunk in parallel
  via :func:`batch_run` semantics (asyncio.gather here for simplicity)
  → single reduce call.
- :class:`RefineSummary` — chunk → walk left-to-right with a running
  summary.
- :class:`HierarchicalSummary` — chunk → k-ary bottom-up merge tree.

All three accept a :class:`~kaos_nlp_core.chunking.Chunker` to drive
the chunking step. The default is a paragraph chunker. The leaf
summarizer is an :class:`~kaos_llm_core.programs.summarize.abstractive.AbstractiveSummary`;
the merge step is its own :class:`Call` configured with a merge-aware
signature.

Phase-5 wiring (0.1.0a10): each Program accepts an optional
``cache: ChunkCache`` and ``budget: Budget`` so callers can reuse
per-chunk summaries across runs and bound end-to-end LLM cost.
Plan §5.3 + §6.1 + §8.5 P1-7 / P1-8.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from kaos_llm_client import BaseProviderClient
from kaos_llm_client.settings import KaosLLMSettings
from kaos_nlp_core.chunking import Chunk, Chunker, ParagraphChunker

from kaos_llm_core.cache.chunk import ChunkCache, ChunkCacheKey
from kaos_llm_core.codecs.base import Codec
from kaos_llm_core.composition import MapReduce, Reducer, Tree
from kaos_llm_core.composition.reduce import Refine as RefineReducer
from kaos_llm_core.optimization.budget import Budget, BudgetTracker
from kaos_llm_core.programs.base import Program
from kaos_llm_core.programs.call import Call
from kaos_llm_core.programs.summarize.abstractive import AbstractiveSummary
from kaos_llm_core.results import Summary
from kaos_llm_core.settings import KaosLLMCoreSettings
from kaos_llm_core.signatures import InputField, OutputField, Signature
from kaos_llm_core.types import Example


def _model_hint(program: Program) -> str:
    """Best-effort opaque model identifier for cache discrimination.

    Reads the leaf program's ``summarizer`` / ``classifier`` /
    ``cited_summarizer`` :class:`Call` attribute and returns the
    ``_model`` string. Returns ``""`` when no Call child is present
    (e.g. a no-LLM ``ExtractiveSummary``) — the cache key still works,
    just without a per-model discriminator.
    """
    for attr in ("summarizer", "classifier", "cited_summarizer"):
        sub = getattr(program, attr, None)
        if sub is not None:
            model = getattr(sub, "_model", None)
            if model:
                return str(model)
    return ""


class _MergeSummariesSignature(Signature):
    """Merge several partial summaries into a single coherent summary.

    The merge should integrate the partial summaries without
    duplicating content, preserve their chronological/structural
    order, and produce a single summary at the same level of
    abstraction as the inputs.
    """

    summaries: list[str] = InputField(
        description="Ordered list of partial summaries to merge.",
    )
    summary: str = OutputField(
        description=(
            "Single merged summary that integrates the partial summaries "
            "without duplicating content. Should be the same length scale "
            "as any one partial summary, not the concatenation."
        ),
    )


class _LongDocBase(Program):
    """Shared scaffold for long-document summarization Programs.

    Subclasses pick a reducer and set it on ``self._reducer``. The
    base class handles chunking, parallel per-chunk summarization,
    and the final reducer invocation.
    """

    program_name: str = "LongDocSummary"

    def __init__(
        self,
        *,
        chunker: Chunker | None = None,
        model: str | None = None,
        codec: Codec | None = None,
        client: BaseProviderClient | None = None,
        settings: KaosLLMSettings | None = None,
        core_settings: KaosLLMCoreSettings | None = None,
        examples: list[Example] | None = None,
        instructions: str | None = None,
        max_retries: int | None = None,
        leaf_summarizer: AbstractiveSummary | None = None,
        max_concurrency: int = 8,
        cache: ChunkCache | None = None,
        budget: Budget | None = None,
        **kwargs: Any,
    ) -> None:
        self._chunker: Chunker = chunker or ParagraphChunker(max_tokens=1024)
        self._max_concurrency = max_concurrency
        self._cache = cache
        self._budget = budget
        # The leaf summarizer is reused per chunk. Auto-registered as
        # a Program child via __setattr__.
        self.summarize_chunk = leaf_summarizer or AbstractiveSummary(
            model=model,
            codec=codec,
            client=client,
            settings=settings,
            core_settings=core_settings,
            examples=examples,
            instructions=instructions,
            max_retries=max_retries,
            **kwargs,
        )
        # The merge call fuses several partial summaries into one.
        self.merge = Call(
            _MergeSummariesSignature,
            model=model,
            codec=codec,
            client=client,
            settings=settings,
            core_settings=core_settings,
            examples=examples,
            instructions=instructions,
            max_retries=max_retries,
            **kwargs,
        )

    # Subclass hook: return the Reducer used in :meth:`forward`.
    def _reducer(self) -> Reducer:  # pragma: no cover - abstract
        raise NotImplementedError

    async def _chunk(self, text: str, parent_id: str | None) -> list[Chunk]:
        chunks = self._chunker.chunk(text, parent_id=parent_id)
        return chunks

    def _chunk_cache_key(self, chunk: Chunk) -> ChunkCacheKey:
        return ChunkCacheKey(
            chunk_id=chunk.chunk_id,
            program_name=type(self.summarize_chunk).__name__,
            model_hint=_model_hint(self.summarize_chunk),
        )

    async def _summarize_one_chunk(
        self,
        chunk: Chunk,
        budget_tracker: BudgetTracker | None,
    ) -> tuple[Summary[str], bool]:
        """Summarise a single chunk with cache + budget instrumentation.

        Returns ``(summary, from_cache)``. ``from_cache=True`` means
        the leaf invocation was skipped because the cache hit; the
        budget tracker is not consumed in that case.
        """
        cache = self._cache
        cache_key = self._chunk_cache_key(chunk) if cache is not None else None

        if cache is not None and cache_key is not None:
            cached = await cache.get(cache_key)
            if cached is not None:
                # Cached payloads come back as either a Summary (when
                # the same in-memory cache was used in this process)
                # or a dict (when round-tripped through a JSON cache).
                if isinstance(cached, Summary):
                    summary_cached: Summary[str] = cached
                else:
                    summary_cached = Summary[str].model_validate(cached)
                summary_cached = summary_cached.model_copy(
                    update={
                        "metadata": {
                            **dict(summary_cached.metadata),
                            "cache.hit": True,
                        },
                    }
                )
                return self._tag_chunk_metadata(summary_cached, chunk), True

        # Cache miss (or no cache). Use ``invoke`` so we get usage for
        # the budget tracker.
        invocation = await self.summarize_chunk.invoke(
            text=chunk.text,
            parent_id=chunk.chunk_id,
        )
        summary = invocation.output

        if budget_tracker is not None:
            budget_tracker.consume(
                trials=0,
                cost_usd=float(invocation.usage.cost_usd or 0.0),
                tokens=int(invocation.usage.total_tokens or 0),
            )

        if cache is not None and cache_key is not None:
            await cache.set(cache_key, summary)

        return self._tag_chunk_metadata(summary, chunk), False

    @staticmethod
    def _tag_chunk_metadata(summary: Summary[str], chunk: Chunk) -> Summary[str]:
        """Decorate a leaf summary with chunk-level provenance."""
        if chunk.chunk_id in summary.chunks_used:
            chunks_used = summary.chunks_used
        else:
            chunks_used = [chunk.chunk_id, *summary.chunks_used]
        return summary.model_copy(
            update={
                "chunks_used": chunks_used,
                "metadata": {
                    **dict(summary.metadata),
                    "chunk.parent_id": chunk.parent_id,
                    "chunk.start": chunk.start,
                    "chunk.end": chunk.end,
                },
            }
        )

    async def _summarize_chunks(
        self,
        chunks: Sequence[Chunk],
        budget_tracker: BudgetTracker | None,
    ) -> tuple[list[Summary[str]], int, int]:
        """Run the leaf summarizer over each chunk with bounded concurrency.

        Returns ``(leaves, cache_hits, n_processed)``. When the
        ``budget_tracker`` reports exhausted, processing stops early
        and ``n_processed`` is strictly less than ``len(chunks)`` — the
        caller materialises a partial summary and tags the metadata.
        """
        if budget_tracker is None:
            sem = asyncio.Semaphore(self._max_concurrency)

            async def _one(chunk: Chunk) -> tuple[Summary[str], bool]:
                async with sem:
                    return await self._summarize_one_chunk(chunk, None)

            paired = await asyncio.gather(*(_one(chunk) for chunk in chunks))
            return [p[0] for p in paired], sum(1 for p in paired if p[1]), len(paired)

        # With a budget tracker, drop to serial-with-early-exit so the
        # tracker can short-circuit on the very next leaf after the cap
        # is reached. ``max_concurrency`` is honoured implicitly (=1).
        leaves: list[Summary[str]] = []
        cache_hits = 0
        for chunk in chunks:
            if budget_tracker.exhausted() is not None:
                break
            summary, from_cache = await self._summarize_one_chunk(chunk, budget_tracker)
            leaves.append(summary)
            if from_cache:
                cache_hits += 1
        return leaves, cache_hits, len(leaves)

    async def _merge_group(self, group: Sequence[Summary[str]]) -> Summary[str]:
        """Async merge callback that the Reducer calls."""
        if len(group) == 1:
            return group[0]
        result = await self.merge(summaries=[summary.text for summary in group])
        return Summary[str](
            text=result.summary,
            method="abstractive",
            metadata={
                "program": self.program_name,
                "merge.input_count": len(group),
            },
        )

    async def forward(  # ty: ignore[invalid-method-override]
        self,
        *,
        text: str,
        parent_id: str | None = None,
    ) -> Summary[str]:
        chunks = await self._chunk(text, parent_id)
        if not chunks:
            # Empty input: synthesize an empty Summary so callers
            # always get a well-formed result.
            return Summary[str](
                text="",
                method="abstractive",
                metadata={
                    "program": self.program_name,
                    "chunks.count": 0,
                },
            )
        tracker = self._budget.make_tracker() if self._budget is not None else None
        leaves, cache_hits, n_processed = await self._summarize_chunks(chunks, tracker)
        if not leaves:
            # Budget was already exhausted before the first leaf
            # finished, or every chunk hit the cache and returned no
            # rows (shouldn't happen — defensive). Return an empty
            # Summary tagged with the stop reason.
            stop_reason = tracker.exhausted() if tracker is not None else None
            return Summary[str](
                text="",
                method="abstractive",
                metadata={
                    "program": self.program_name,
                    "chunks.count": len(chunks),
                    "chunks.processed": 0,
                    "chunker": type(self._chunker).__name__,
                    "cache.hits": cache_hits,
                    **({"budget.exhausted": str(stop_reason)} if stop_reason is not None else {}),
                },
            )
        reduced = await self._reducer().reduce(leaves, self._merge_group)
        stop_reason = tracker.exhausted() if tracker is not None else None
        meta_extras: dict[str, Any] = {
            "program": self.program_name,
            "chunks.count": len(chunks),
            "chunks.processed": n_processed,
            "chunker": type(self._chunker).__name__,
            "cache.hits": cache_hits,
        }
        if tracker is not None:
            meta_extras["budget.cost_usd"] = round(tracker.cost_usd, 6)
            meta_extras["budget.tokens"] = tracker.tokens
        if stop_reason is not None:
            meta_extras["budget.exhausted"] = str(stop_reason)
            meta_extras["partial"] = True
        return reduced.model_copy(
            update={
                "metadata": {
                    **dict(reduced.metadata),
                    **meta_extras,
                },
            }
        )


class MapReduceSummary(_LongDocBase):
    """Long-document summarizer: parallel map then single reduce call.

    Each chunk is summarized concurrently (bounded by
    ``max_concurrency``); the partial summaries are then merged in
    one shot via :class:`~kaos_llm_core.composition.MapReduce`.
    """

    program_name = "MapReduceSummary"

    def _reducer(self) -> Reducer:
        return MapReduce()


class RefineSummary(_LongDocBase):
    """Long-document summarizer: sequential refinement.

    Each chunk extends a running summary via the
    :class:`~kaos_llm_core.composition.reduce.Refine` reducer. Order
    is preserved at the cost of serial merge calls.
    """

    program_name = "RefineSummary"

    def _reducer(self) -> Reducer:
        return RefineReducer()


class HierarchicalSummary(_LongDocBase):
    """Long-document summarizer: k-ary bottom-up merge tree.

    Each level of the merge tree runs sibling merges concurrently;
    this is the general workhorse for very long documents because
    the total merge cost scales as O(n / (k-1)) with depth O(log_k n).

    Args:
        branching: Tree branching factor passed to
            :class:`~kaos_llm_core.composition.Tree`. Default ``4``.
        max_depth: Hard cap on tree depth. Default ``8``.

    Any remaining keyword arguments forward to the base
    :class:`_LongDocBase` constructor.
    """

    program_name = "HierarchicalSummary"

    def __init__(
        self,
        *,
        branching: int = 4,
        max_depth: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._branching = branching
        self._max_depth = max_depth

    def _reducer(self) -> Reducer:
        return Tree(branching=self._branching, max_depth=self._max_depth)


__all__ = [
    "HierarchicalSummary",
    "MapReduceSummary",
    "RefineSummary",
]
