"""Long-document classification wrapper.

:class:`ChunkedClassify` is **not** itself an LLM classifier. It
chunks the input, runs any per-chunk classifier in parallel, and
combines the per-chunk results into one
:class:`~kaos_llm_core.results.Classification` via an
:class:`~kaos_llm_core.composition.Aggregator`.

This separation is deliberate: the chunking strategy, the per-chunk
classifier, and the aggregation rule are three orthogonal axes of
the long-document classification problem.
"""

from __future__ import annotations

import asyncio

from kaos_nlp_core.chunking import Chunker, ParagraphChunker

from kaos_llm_core.composition import Aggregator, MajorityAggregator
from kaos_llm_core.labels import LabelSet
from kaos_llm_core.programs.base import Program
from kaos_llm_core.results import Classification


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
    """

    def __init__(
        self,
        *,
        labels: LabelSet,
        per_chunk: Program,
        chunker: Chunker | None = None,
        aggregator: Aggregator | None = None,
        max_concurrency: int = 8,
    ) -> None:
        self._label_set = labels
        self.per_chunk = per_chunk  # auto-registers as child for trace
        self._chunker = chunker or ParagraphChunker(max_tokens=1024)
        self._aggregator = aggregator or self._default_aggregator(labels)
        self._max_concurrency = max_concurrency

    @staticmethod
    def _default_aggregator(labels: LabelSet) -> Aggregator:
        from kaos_llm_core.composition.aggregate import (
            MajorityAggregator,
            UnionAggregator,
        )

        return MajorityAggregator() if labels.exclusive else UnionAggregator()

    async def _classify_chunks(self, text: str, parent_id: str | None) -> list[Classification]:
        chunks = self._chunker.chunk(text, parent_id=parent_id)
        if not chunks:
            return []
        sem = asyncio.Semaphore(self._max_concurrency)

        async def _one(chunk_text: str, chunk_id: str) -> Classification:
            async with sem:
                result: Classification = await self.per_chunk(
                    text=chunk_text,
                    parent_id=chunk_id,
                )
                # Replace chunks_used with our chunk id (the inner
                # program might have used a different naming scheme).
                if chunk_id not in result.chunks_used:
                    return result.model_copy(
                        update={
                            "chunks_used": [chunk_id, *result.chunks_used],
                        }
                    )
                return result

        return await asyncio.gather(*(_one(chunk.text, chunk.chunk_id) for chunk in chunks))

    async def forward(  # ty: ignore[invalid-method-override]
        self,
        *,
        text: str,
        parent_id: str | None = None,
    ) -> Classification:
        per_chunk_results = await self._classify_chunks(text, parent_id)
        if not per_chunk_results:
            return Classification(
                labels=[],
                abstained=True,
                metadata={
                    "program": "ChunkedClassify",
                    "chunks.count": 0,
                },
            )
        combined = self._aggregator.combine(per_chunk_results, self._label_set)
        return combined.model_copy(
            update={
                "metadata": {
                    **dict(combined.metadata),
                    "program": "ChunkedClassify",
                    "chunks.count": len(per_chunk_results),
                    "chunker": type(self._chunker).__name__,
                    "aggregator": type(self._aggregator).__name__,
                    "per_chunk_program": type(self.per_chunk).__name__,
                },
            }
        )


# Re-export aggregators commonly bundled with ChunkedClassify so callers
# only need a single import.
__all__ = [
    "ChunkedClassify",
    "MajorityAggregator",
]
