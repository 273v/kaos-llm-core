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
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from kaos_llm_client import BaseProviderClient
from kaos_llm_client.settings import KaosLLMSettings
from kaos_nlp_core.chunking import Chunk, Chunker, ParagraphChunker

from kaos_llm_core.codecs.base import Codec
from kaos_llm_core.composition import MapReduce, Reducer, Tree
from kaos_llm_core.composition.reduce import Refine as RefineReducer
from kaos_llm_core.programs.base import Program
from kaos_llm_core.programs.call import Call
from kaos_llm_core.programs.summarize.abstractive import AbstractiveSummary
from kaos_llm_core.results import Summary
from kaos_llm_core.settings import KaosLLMCoreSettings
from kaos_llm_core.signatures import InputField, OutputField, Signature
from kaos_llm_core.types import Example


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
        **kwargs: Any,
    ) -> None:
        self._chunker: Chunker = chunker or ParagraphChunker(max_tokens=1024)
        self._max_concurrency = max_concurrency
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

    async def _summarize_chunks(self, chunks: Sequence[Chunk]) -> list[Summary[str]]:
        """Run the leaf summarizer over each chunk with bounded concurrency."""
        sem = asyncio.Semaphore(self._max_concurrency)

        async def _one(chunk: Chunk) -> Summary[str]:
            async with sem:
                summary = await self.summarize_chunk(text=chunk.text, parent_id=chunk.chunk_id)
                # Tag chunk-level metadata for downstream provenance.
                return summary.model_copy(
                    update={
                        "chunks_used": [chunk.chunk_id, *summary.chunks_used],
                        "metadata": {
                            **dict(summary.metadata),
                            "chunk.parent_id": chunk.parent_id,
                            "chunk.start": chunk.start,
                            "chunk.end": chunk.end,
                        },
                    }
                )

        return await asyncio.gather(*(_one(chunk) for chunk in chunks))

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
        leaves = await self._summarize_chunks(chunks)
        reduced = await self._reducer().reduce(leaves, self._merge_group)
        return reduced.model_copy(
            update={
                "metadata": {
                    **dict(reduced.metadata),
                    "program": self.program_name,
                    "chunks.count": len(chunks),
                    "chunker": type(self._chunker).__name__,
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
