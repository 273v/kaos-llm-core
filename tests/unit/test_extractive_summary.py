"""Tests for :class:`~kaos_llm_core.programs.summarize.ExtractiveSummary`.

These tests stay fully offline: the :class:`Ranker` protocol lets us
substitute a deterministic stub for the kaos-nlp-transformers
``ExtractiveRanker``. Live coverage against a real embedder lives in
``tests/quality`` (Phase 5 leftover live harness, behind
``@pytest.mark.live``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pytest
from kaos_nlp_core.types import Segment

from kaos_llm_core.programs.summarize import ExtractiveSummary, RankedSegment
from kaos_llm_core.results import Summary


@dataclass(frozen=True, slots=True)
class _StubScoredSegment:
    """Minimal :class:`RankedSegment` for offline tests."""

    text: str
    start: int
    end: int
    score: float
    rank: int


class _StaticRanker:
    """Ranker stub that returns a pre-baked top-k order.

    Useful for asserting that :class:`ExtractiveSummary`:
    - forwards ``query``, ``top_k``, and ``diversify`` correctly
    - sorts picks back into source order for the output
    - preserves score-ordered metadata
    """

    def __init__(self, picks_by_index: list[int], score_curve: list[float] | None = None) -> None:
        self._picks_by_index = picks_by_index
        self._score_curve = score_curve
        self.last_kwargs: dict[str, object] = {}

    def rank(
        self,
        sentences: Sequence[Segment],
        *,
        query: str | None = None,
        top_k: int | None = None,
        diversify: float = 0.0,
    ) -> Sequence[RankedSegment]:
        self.last_kwargs = {
            "n_sentences": len(sentences),
            "query": query,
            "top_k": top_k,
            "diversify": diversify,
        }
        # Honour the requested cap; the protocol's rank() always
        # truncates to top_k itself in the production implementation.
        cap = top_k if top_k is not None else len(self._picks_by_index)
        picks = self._picks_by_index[:cap]
        out: list[_StubScoredSegment] = []
        for rank, idx in enumerate(picks):
            seg = sentences[idx]
            score = self._score_curve[rank] if self._score_curve else 1.0 - 0.1 * rank
            out.append(
                _StubScoredSegment(
                    text=seg.text,
                    start=seg.start,
                    end=seg.end,
                    score=score,
                    rank=rank,
                )
            )
        return out


# The fixture text has four sentences. Their offsets after
# kaos_nlp_core.segmentation.segment_sentences():
#   0: "Alice signed the lease on Monday."
#   1: "Bob countersigned on Tuesday."
#   2: "The term is two years."
#   3: "Rent is due monthly."
_FIXTURE = (
    "Alice signed the lease on Monday. "
    "Bob countersigned on Tuesday. "
    "The term is two years. "
    "Rent is due monthly."
)


class TestExtractiveSummary:
    @pytest.mark.asyncio
    async def test_returns_summary_with_extractive_method(self) -> None:
        ranker = _StaticRanker(picks_by_index=[0, 1])
        program = ExtractiveSummary(ranker=ranker, top_k=2)
        result = await program(text=_FIXTURE)
        assert isinstance(result, Summary)
        assert result.method == "extractive"
        assert result.metadata["program"] == "ExtractiveSummary"
        assert result.metadata["ranker"] == "_StaticRanker"
        assert result.metadata["n_sentences"] == 4

    @pytest.mark.asyncio
    async def test_picks_sorted_to_source_order(self) -> None:
        # Ranker says sentence 2 ("The term...") and sentence 0 ("Alice...")
        # are the top picks (in that score order). The summary text should
        # nevertheless read in source order: Alice first, then the term.
        ranker = _StaticRanker(picks_by_index=[2, 0])
        program = ExtractiveSummary(ranker=ranker, top_k=2, joiner=" ")
        result = await program(text=_FIXTURE)
        assert result.text == "Alice signed the lease on Monday. The term is two years."

    @pytest.mark.asyncio
    async def test_source_spans_match_picks_in_source_order(self) -> None:
        ranker = _StaticRanker(picks_by_index=[2, 0])
        program = ExtractiveSummary(ranker=ranker, top_k=2)
        result = await program(text=_FIXTURE, parent_id="doc-1")
        # Source-order ascending.
        assert [span.start for span in result.source_spans] == sorted(
            span.start for span in result.source_spans
        )
        # Two picks, two spans.
        assert len(result.source_spans) == 2
        # parent_id propagates onto every span.
        for span in result.source_spans:
            assert span.source_id == "doc-1"

    @pytest.mark.asyncio
    async def test_score_order_preserved_in_metadata(self) -> None:
        ranker = _StaticRanker(picks_by_index=[2, 0], score_curve=[0.9, 0.7])
        program = ExtractiveSummary(ranker=ranker, top_k=2)
        result = await program(text=_FIXTURE)
        picks_meta = result.metadata["picks"]
        # picks_meta is in *score* order (highest first), independent of
        # the source-order rearrangement applied to result.text.
        assert [m["rank"] for m in picks_meta] == [0, 1]
        assert picks_meta[0]["score"] > picks_meta[1]["score"]
        # Sentence-2 ("The term...") scored highest and so appears first
        # in the score-ordered metadata even though it appears second in
        # result.text.
        assert _FIXTURE[picks_meta[0]["start"] : picks_meta[0]["end"]].startswith("The term")

    @pytest.mark.asyncio
    async def test_forwards_query_and_diversify(self) -> None:
        ranker = _StaticRanker(picks_by_index=[0])
        program = ExtractiveSummary(ranker=ranker, top_k=1, diversify=0.5)
        await program(text=_FIXTURE, query="who signed?")
        assert ranker.last_kwargs["query"] == "who signed?"
        assert ranker.last_kwargs["diversify"] == 0.5

    @pytest.mark.asyncio
    async def test_per_call_top_k_overrides_constructor(self) -> None:
        ranker = _StaticRanker(picks_by_index=[0, 1, 2])
        program = ExtractiveSummary(ranker=ranker, top_k=1)
        result = await program(text=_FIXTURE, top_k=3)
        assert ranker.last_kwargs["top_k"] == 3
        # Three picks, three spans.
        assert len(result.source_spans) == 3

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty_summary(self) -> None:
        ranker = _StaticRanker(picks_by_index=[])
        program = ExtractiveSummary(ranker=ranker, top_k=5)
        result = await program(text="")
        assert result.text == ""
        assert result.source_spans == []
        assert result.metadata["picks"] == []
        assert result.metadata["n_sentences"] == 0

    @pytest.mark.asyncio
    async def test_parent_id_recorded_in_chunks_used(self) -> None:
        ranker = _StaticRanker(picks_by_index=[0])
        program = ExtractiveSummary(ranker=ranker, top_k=1)
        result = await program(text=_FIXTURE, parent_id="doc-42")
        assert "doc-42" in result.chunks_used

    @pytest.mark.asyncio
    async def test_invocation_has_zero_usage_and_a_trace(self) -> None:
        ranker = _StaticRanker(picks_by_index=[0])
        program = ExtractiveSummary(ranker=ranker, top_k=1)
        invocation = await program.invoke(text=_FIXTURE)
        # No LLM call -> zero usage.
        assert invocation.usage.input_tokens == 0
        assert invocation.usage.output_tokens == 0

    @pytest.mark.asyncio
    async def test_custom_joiner(self) -> None:
        ranker = _StaticRanker(picks_by_index=[0, 1])
        program = ExtractiveSummary(ranker=ranker, top_k=2, joiner="\n\n")
        result = await program(text=_FIXTURE)
        # In source order: sentence 0 then sentence 1.
        assert result.text == "Alice signed the lease on Monday.\n\nBob countersigned on Tuesday."

    def test_top_k_must_be_positive(self) -> None:
        ranker = _StaticRanker(picks_by_index=[])
        with pytest.raises(ValueError, match="top_k must be > 0"):
            ExtractiveSummary(ranker=ranker, top_k=0)

    def test_diversify_must_be_in_range(self) -> None:
        ranker = _StaticRanker(picks_by_index=[])
        with pytest.raises(ValueError, match=r"diversify must be in \[0, 1\]"):
            ExtractiveSummary(ranker=ranker, top_k=1, diversify=1.5)

    @pytest.mark.asyncio
    async def test_per_call_top_k_must_be_positive(self) -> None:
        ranker = _StaticRanker(picks_by_index=[])
        program = ExtractiveSummary(ranker=ranker, top_k=2)
        with pytest.raises(ValueError, match="top_k must be > 0"):
            await program(text=_FIXTURE, top_k=0)
