"""Long-document classification wrapper.

:class:`ChunkedClassify` is **not** itself an LLM classifier. It
chunks the input, runs any per-chunk classifier in parallel, and
combines the per-chunk results into one
:class:`~kaos_llm_core.results.Classification` via an
:class:`~kaos_llm_core.composition.Aggregator`.

This separation is deliberate: the chunking strategy, the per-chunk
classifier, and the aggregation rule are three orthogonal axes of
the long-document classification problem.

Phase-5 wiring (0.1.0a10): ``ChunkedClassify`` accepts an optional
``cache: ChunkCache`` and ``budget: Budget`` so callers can reuse
per-chunk classifications across runs and bound end-to-end cost.
Plan §5.3 + §6.2 + §8.5 P1-7 / P1-8.
"""

from __future__ import annotations

import asyncio
from typing import Any

from kaos_nlp_core.chunking import Chunk, Chunker, ParagraphChunker

from kaos_llm_core.cache.chunk import ChunkCache, ChunkCacheKey
from kaos_llm_core.composition import Aggregator, MajorityAggregator
from kaos_llm_core.labels import LabelSet
from kaos_llm_core.optimization.budget import Budget, BudgetTracker
from kaos_llm_core.programs.base import Program
from kaos_llm_core.results import Classification


def _model_hint(program: Program) -> str:
    """Best-effort opaque model id for cache discrimination.

    See ``kaos_llm_core.programs.summarize.long_doc._model_hint`` —
    same contract: read ``program.classifier._model`` /
    ``program.summarizer._model`` when present, else return ``""``.
    """
    for attr in ("classifier", "summarizer", "cited_summarizer"):
        sub = getattr(program, attr, None)
        if sub is not None:
            model = getattr(sub, "_model", None)
            if model:
                return str(model)
    return ""


class ChunkedClassify(Program):
    """Chunk → classify each chunk in parallel → aggregate.

    Args:
        labels: Label space; must match the per-chunk classifier's
            label space.
        per_chunk: A classifier :class:`Program` whose ``forward``
            accepts ``text=`` and (optionally) ``parent_id=``,
            and returns a :class:`Classification`.
        chunker: Deterministic chunker. Defaults to
            :class:`ParagraphChunker`.
        aggregator: Strategy for combining per-chunk results.
            Defaults to :class:`MajorityAggregator` for exclusive
            label sets, :class:`UnionAggregator` for multi-label.
        max_concurrency: Cap on concurrent per-chunk classifier
            invocations. Default ``8``.
        cache: Optional :class:`ChunkCache`. When provided, every
            per-chunk classification is keyed by
            ``(chunk_id, per_chunk_program_name, model_hint)``;
            cache hits skip the per-chunk Program invocation
            entirely.
        budget: Optional :class:`Budget`. When provided, a fresh
            :class:`BudgetTracker` is created per :meth:`forward`
            call; the tracker is consumed from each per-chunk
            invocation's :attr:`~kaos_llm_core.programs._invocation.TokenUsage`.
            Once the tracker reports exhausted, processing of
            remaining chunks halts and the aggregated result carries
            ``metadata["budget.exhausted"]`` + ``metadata["partial"] = True``.
    """

    def __init__(
        self,
        *,
        labels: LabelSet,
        per_chunk: Program,
        chunker: Chunker | None = None,
        aggregator: Aggregator | None = None,
        max_concurrency: int = 8,
        cache: ChunkCache | None = None,
        budget: Budget | None = None,
    ) -> None:
        self._label_set = labels
        self.per_chunk = per_chunk  # auto-registers as child for trace
        self._chunker = chunker or ParagraphChunker(max_tokens=1024)
        self._aggregator = aggregator or self._default_aggregator(labels)
        self._max_concurrency = max_concurrency
        self._cache = cache
        self._budget = budget

    @staticmethod
    def _default_aggregator(labels: LabelSet) -> Aggregator:
        from kaos_llm_core.composition.aggregate import (
            MajorityAggregator,
            UnionAggregator,
        )

        return MajorityAggregator() if labels.exclusive else UnionAggregator()

    def _chunk_cache_key(self, chunk: Chunk) -> ChunkCacheKey:
        return ChunkCacheKey(
            chunk_id=chunk.chunk_id,
            program_name=type(self.per_chunk).__name__,
            model_hint=_model_hint(self.per_chunk),
        )

    async def _classify_one_chunk(
        self,
        chunk: Chunk,
        budget_tracker: BudgetTracker | None,
    ) -> tuple[Classification, bool]:
        """Classify a single chunk with cache + budget instrumentation.

        Returns ``(classification, from_cache)``. ``from_cache=True``
        means the per-chunk Program invocation was skipped; the
        budget tracker is not consumed in that case.
        """
        cache = self._cache
        cache_key = self._chunk_cache_key(chunk) if cache is not None else None

        if cache is not None and cache_key is not None:
            cached = await cache.get(cache_key)
            if cached is not None:
                if isinstance(cached, Classification):
                    classification_cached: Classification = cached
                else:
                    classification_cached = Classification.model_validate(cached)
                classification_cached = classification_cached.model_copy(
                    update={
                        "metadata": {
                            **dict(classification_cached.metadata),
                            "cache.hit": True,
                        },
                    }
                )
                return self._tag_chunk_metadata(classification_cached, chunk), True

        invocation = await self.per_chunk.invoke(text=chunk.text, parent_id=chunk.chunk_id)
        classification: Classification = invocation.output
        if budget_tracker is not None:
            budget_tracker.consume(
                trials=0,
                cost_usd=float(invocation.usage.cost_usd or 0.0),
                tokens=int(invocation.usage.total_tokens or 0),
            )
        if cache is not None and cache_key is not None:
            await cache.set(cache_key, classification)
        return self._tag_chunk_metadata(classification, chunk), False

    @staticmethod
    def _tag_chunk_metadata(classification: Classification, chunk: Chunk) -> Classification:
        if chunk.chunk_id in classification.chunks_used:
            return classification
        return classification.model_copy(
            update={
                "chunks_used": [chunk.chunk_id, *classification.chunks_used],
            }
        )

    async def _classify_chunks(
        self,
        text: str,
        parent_id: str | None,
        budget_tracker: BudgetTracker | None,
    ) -> tuple[list[Classification], int, int]:
        """Classify every chunk; return ``(results, cache_hits, processed)``.

        Mirror of the long-doc summary helper. With no budget tracker
        the per-chunk Program runs concurrently under ``max_concurrency``;
        with a tracker it drops to serial so the tracker can short-circuit
        on the next leaf after the cap is hit.
        """
        chunks = self._chunker.chunk(text, parent_id=parent_id)
        if not chunks:
            return [], 0, 0

        if budget_tracker is None:
            sem = asyncio.Semaphore(self._max_concurrency)

            async def _one(chunk: Chunk) -> tuple[Classification, bool]:
                async with sem:
                    return await self._classify_one_chunk(chunk, None)

            paired = await asyncio.gather(*(_one(chunk) for chunk in chunks))
            return [p[0] for p in paired], sum(1 for p in paired if p[1]), len(paired)

        out: list[Classification] = []
        cache_hits = 0
        for chunk in chunks:
            if budget_tracker.exhausted() is not None:
                break
            classification, from_cache = await self._classify_one_chunk(chunk, budget_tracker)
            out.append(classification)
            if from_cache:
                cache_hits += 1
        return out, cache_hits, len(out)

    async def forward(  # ty: ignore[invalid-method-override]
        self,
        *,
        text: str,
        parent_id: str | None = None,
    ) -> Classification:
        tracker = self._budget.make_tracker() if self._budget is not None else None
        per_chunk_results, cache_hits, n_processed = await self._classify_chunks(
            text, parent_id, tracker
        )
        if not per_chunk_results:
            stop_reason = tracker.exhausted() if tracker is not None else None
            return Classification(
                labels=[],
                abstained=True,
                metadata={
                    "program": "ChunkedClassify",
                    "chunks.count": 0,
                    "chunks.processed": 0,
                    "cache.hits": cache_hits,
                    **({"budget.exhausted": str(stop_reason)} if stop_reason is not None else {}),
                },
            )
        combined = self._aggregator.combine(per_chunk_results, self._label_set)
        stop_reason = tracker.exhausted() if tracker is not None else None
        meta_extras: dict[str, Any] = {
            "program": "ChunkedClassify",
            "chunks.count": n_processed,
            "chunks.processed": n_processed,
            "chunker": type(self._chunker).__name__,
            "aggregator": type(self._aggregator).__name__,
            "per_chunk_program": type(self.per_chunk).__name__,
            "cache.hits": cache_hits,
        }
        if tracker is not None:
            meta_extras["budget.cost_usd"] = round(tracker.cost_usd, 6)
            meta_extras["budget.tokens"] = tracker.tokens
        if stop_reason is not None:
            meta_extras["budget.exhausted"] = str(stop_reason)
            meta_extras["partial"] = True
        return combined.model_copy(
            update={
                "metadata": {
                    **dict(combined.metadata),
                    **meta_extras,
                },
            }
        )


# Re-export aggregators commonly bundled with ChunkedClassify so callers
# only need a single import.
__all__ = [
    "ChunkedClassify",
    "MajorityAggregator",
]
