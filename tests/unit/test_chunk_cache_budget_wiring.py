"""Tests for the §8.5 cache + budget wiring across long-doc Programs.

Covers ``HierarchicalSummary`` / ``MapReduceSummary`` / ``RefineSummary``
(via ``_LongDocBase``) and ``ChunkedClassify``. The LLM calls are
stubbed with :class:`FunctionClient` so the tests stay offline and
deterministic.

What's asserted:

- On cache hit, the per-chunk Program invocation is skipped (the
  stub LLM function counts calls).
- On budget exhaustion partway through processing, the Program
  short-circuits and returns a partial result tagged with
  ``metadata["budget.exhausted"]`` + ``metadata["partial"] = True``.
- The aggregated result's metadata reports cache hits, budget
  spend, and processed-chunk count.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest
from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart
from kaos_nlp_core.chunking import SentenceChunker

from kaos_llm_core.cache import InMemoryChunkCache
from kaos_llm_core.labels import Label, LabelSet
from kaos_llm_core.optimization.budget import Budget
from kaos_llm_core.programs.classify import ChunkedClassify, ZeroShotClassify
from kaos_llm_core.programs.summarize import MapReduceSummary


def _three_chunk_chunker() -> SentenceChunker:
    """SentenceChunker(max_tokens=1) splits _DOC into 3 single-sentence chunks.

    Using a one-sentence-per-chunk chunker is the cheapest way to get
    a deterministic 3-leaf workload — the default
    ``ParagraphChunker(max_tokens=1024)`` would pack the entire
    fixture into one chunk.
    """
    return SentenceChunker(max_tokens=1)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


def _json_response(data: dict[str, Any]) -> ProviderResponse:
    return ProviderResponse.model_construct(
        provider="function",
        model="function-test",
        raw={},
        parts=[ContentPart.model_construct(type="text", text=json.dumps(data))],
        usage=UsageInfo.model_construct(input_tokens=100, output_tokens=50, total_tokens=150),
        stop_reason="end_turn",
        response_id="test-id",
        status_code=200,
        response_headers={},
        latency_ms=10.0,
    )


class _CountingFn:
    """A FunctionClient ``function`` that counts calls and returns canned data."""

    def __init__(self, payload_factory: Callable[[int], dict[str, Any]]) -> None:
        self._payload_factory = payload_factory
        self.calls: int = 0

    def __call__(self, messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
        idx = self.calls
        self.calls += 1
        return _json_response(self._payload_factory(idx))


# ---------------------------------------------------------------------------
# Long-doc summary wiring
# ---------------------------------------------------------------------------


# Three sentences. ``SentenceChunker(max_tokens=1)`` puts each in its
# own chunk so the test workload is deterministic 3 leaves.
_DOC = (
    "Paragraph one talks about lease terms. "
    "Paragraph two discusses indemnification. "
    "Paragraph three covers limitation of liability."
)


class TestLongDocCacheWiring:
    @pytest.mark.asyncio
    async def test_cache_hit_skips_subsequent_llm_calls(self) -> None:
        cache = InMemoryChunkCache()

        # The chunker emits one chunk per paragraph (3 chunks total),
        # but for a cache HIT to fire on the SECOND run, the second
        # run's chunk_ids must match the first. ParagraphChunker is
        # deterministic, so identical input → identical chunk_ids.
        fn = _CountingFn(lambda i: {"summary": f"chunk-{i}"})

        prog_cold = MapReduceSummary(
            model="function-test",
            client=FunctionClient(function=fn),
            chunker=_three_chunk_chunker(),
            cache=cache,
        )
        cold = await prog_cold(text=_DOC)
        # 3 leaf calls + 1 merge call = 4 LLM calls cold.
        assert fn.calls == 4
        assert cold.metadata["cache.hits"] == 0
        assert cold.metadata["chunks.processed"] == 3
        # Cache populated with 3 leaf entries (the merge call is NOT
        # cache-keyed at this layer — it's an internal Call, not a
        # per-chunk Program).
        assert len(cache) == 3

        # Second run uses the same cache. The 3 leaf invocations
        # should hit the cache; only the merge call goes to the LLM.
        fn_warm = _CountingFn(lambda i: {"summary": f"chunk-{i}"})
        prog_warm = MapReduceSummary(
            model="function-test",
            client=FunctionClient(function=fn_warm),
            chunker=_three_chunk_chunker(),
            cache=cache,
        )
        warm = await prog_warm(text=_DOC)
        # Only the merge call hit the LLM.
        assert fn_warm.calls == 1
        assert warm.metadata["cache.hits"] == 3
        assert warm.metadata["chunks.processed"] == 3

    @pytest.mark.asyncio
    async def test_no_cache_runs_every_leaf(self) -> None:
        fn = _CountingFn(lambda i: {"summary": f"chunk-{i}"})
        prog = MapReduceSummary(
            model="function-test",
            client=FunctionClient(function=fn),
            chunker=_three_chunk_chunker(),
        )
        result = await prog(text=_DOC)
        # Cache wasn't wired -> no cache.hits key (or 0).
        assert result.metadata["cache.hits"] == 0
        # 3 leaves + 1 merge.
        assert fn.calls == 4


class TestLongDocBudgetWiring:
    @pytest.mark.asyncio
    async def test_budget_exhaustion_stops_processing(self) -> None:
        # Each leaf reports 150 tokens. Cap at 250 tokens -> the first
        # leaf consumes 150 (under the cap, not yet exhausted), the
        # second leaf consumes another 150 -> 300 total -> exhausted.
        # Processing should stop after the second leaf.
        fn = _CountingFn(lambda i: {"summary": f"chunk-{i}"})
        budget = Budget(max_tokens=250)
        prog = MapReduceSummary(
            model="function-test",
            client=FunctionClient(function=fn),
            chunker=_three_chunk_chunker(),
            budget=budget,
        )
        result = await prog(text=_DOC)
        # Two leaves processed, plus the final merge call -> 3 LLM
        # calls total. The third chunk was never sent to the LLM.
        assert fn.calls == 3
        assert result.metadata["chunks.count"] == 3
        assert result.metadata["chunks.processed"] == 2
        assert result.metadata["partial"] is True
        assert "budget_tokens" in result.metadata["budget.exhausted"]
        # Tracker stops consuming when it short-circuits; the recorded
        # spend should be >= the cap.
        assert result.metadata["budget.tokens"] >= 250

    @pytest.mark.asyncio
    async def test_budget_not_exhausted_finishes_normally(self) -> None:
        fn = _CountingFn(lambda i: {"summary": f"chunk-{i}"})
        budget = Budget(max_tokens=10_000)  # well above 3 * 150
        prog = MapReduceSummary(
            model="function-test",
            client=FunctionClient(function=fn),
            chunker=_three_chunk_chunker(),
            budget=budget,
        )
        result = await prog(text=_DOC)
        assert fn.calls == 4  # 3 leaves + merge
        assert result.metadata["chunks.processed"] == 3
        assert "partial" not in result.metadata
        assert "budget.exhausted" not in result.metadata


# ---------------------------------------------------------------------------
# ChunkedClassify wiring
# ---------------------------------------------------------------------------


_LABELS = LabelSet(
    labels=[
        Label(name="contract", description="A binding agreement."),
        Label(name="memo", description="An internal note."),
    ],
    exclusive=True,
)


class TestChunkedClassifyCacheWiring:
    @pytest.mark.asyncio
    async def test_cache_hits_skip_llm_calls(self) -> None:
        fn = _CountingFn(lambda i: {"label": "contract", "confidence": 0.9, "rationale": "stub"})
        cache = InMemoryChunkCache()
        per_chunk = ZeroShotClassify(
            labels=_LABELS,
            model="function-test",
            client=FunctionClient(function=fn),
        )
        prog_cold = ChunkedClassify(
            labels=_LABELS,
            per_chunk=per_chunk,
            chunker=_three_chunk_chunker(),
            cache=cache,
        )
        cold = await prog_cold(text=_DOC)
        assert fn.calls == 3
        assert cold.metadata["cache.hits"] == 0
        # Now wire a fresh classifier sharing the same cache; every
        # leaf should hit.
        fn_warm = _CountingFn(
            lambda i: {"label": "contract", "confidence": 0.9, "rationale": "stub"}
        )
        per_chunk_warm = ZeroShotClassify(
            labels=_LABELS,
            model="function-test",
            client=FunctionClient(function=fn_warm),
        )
        prog_warm = ChunkedClassify(
            labels=_LABELS,
            per_chunk=per_chunk_warm,
            chunker=_three_chunk_chunker(),
            cache=cache,
        )
        warm = await prog_warm(text=_DOC)
        assert fn_warm.calls == 0
        assert warm.metadata["cache.hits"] == 3
        assert warm.metadata["chunks.processed"] == 3


class TestChunkedClassifyBudgetWiring:
    @pytest.mark.asyncio
    async def test_budget_exhaustion_stops_processing(self) -> None:
        fn = _CountingFn(lambda i: {"label": "contract", "confidence": 0.9, "rationale": "stub"})
        per_chunk = ZeroShotClassify(
            labels=_LABELS,
            model="function-test",
            client=FunctionClient(function=fn),
        )
        budget = Budget(max_tokens=250)
        prog = ChunkedClassify(
            labels=_LABELS,
            per_chunk=per_chunk,
            chunker=_three_chunk_chunker(),
            budget=budget,
        )
        result = await prog(text=_DOC)
        # Two leaves processed before the tracker tripped.
        assert fn.calls == 2
        assert result.metadata["chunks.processed"] == 2
        assert result.metadata["partial"] is True
        assert "budget_tokens" in result.metadata["budget.exhausted"]
