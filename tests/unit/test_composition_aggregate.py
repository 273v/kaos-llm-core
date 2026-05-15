"""Tests for :mod:`kaos_llm_core.composition.aggregate` strategies."""

from __future__ import annotations

import pytest

from kaos_llm_core.composition import (
    Aggregator,
    IntersectionAggregator,
    MajorityAggregator,
    MaxScoreAggregator,
    UnionAggregator,
    VoteAggregator,
    WeightedAggregator,
)
from kaos_llm_core.labels import Label, LabelSet
from kaos_llm_core.results import Classification, SourceSpan


def _picks(label_set: LabelSet, names: list[str]) -> list[Label]:
    return [label_set.by_name(n) for n in names]


def _make(
    label_set: LabelSet,
    names: list[str],
    *,
    scores: dict[str, float] | None = None,
    chunks_used: list[str] | None = None,
    source_spans: list[SourceSpan] | None = None,
) -> Classification[Label]:
    return Classification[Label](
        labels=_picks(label_set, names),
        scores=scores or {},
        chunks_used=chunks_used or [],
        source_spans=source_spans or [],
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def label_set() -> LabelSet:
    return LabelSet(
        labels=[Label(name="a"), Label(name="b"), Label(name="c")],
        exclusive=True,
        allow_abstain=True,
    )


@pytest.fixture
def multi_label_set() -> LabelSet:
    return LabelSet(
        labels=[Label(name="a"), Label(name="b"), Label(name="c")],
        exclusive=False,
        allow_abstain=True,
    )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [
        VoteAggregator,
        MajorityAggregator,
        UnionAggregator,
        IntersectionAggregator,
        WeightedAggregator,
        MaxScoreAggregator,
    ],
)
def test_strategy_implements_aggregator_protocol(factory) -> None:
    assert isinstance(factory(), Aggregator)


# ---------------------------------------------------------------------------
# VoteAggregator
# ---------------------------------------------------------------------------


class TestVoteAggregator:
    def test_plurality_wins(self, label_set: LabelSet) -> None:
        chunks = [
            _make(label_set, ["a"]),
            _make(label_set, ["b"]),
            _make(label_set, ["a"]),
        ]
        result = VoteAggregator().combine(chunks, label_set)
        assert result.names == ["a"]
        assert result.abstained is False

    def test_empty_returns_abstained(self, label_set: LabelSet) -> None:
        result = VoteAggregator().combine([], label_set)
        assert result.abstained is True
        assert result.labels == []

    def test_provenance_pooled(self, label_set: LabelSet) -> None:
        chunks = [
            _make(label_set, ["a"], chunks_used=["c1"]),
            _make(label_set, ["a"], chunks_used=["c2"]),
        ]
        result = VoteAggregator().combine(chunks, label_set)
        assert set(result.chunks_used) == {"c1", "c2"}

    def test_label_histogram_in_metadata(self, label_set: LabelSet) -> None:
        chunks = [
            _make(label_set, ["a"]),
            _make(label_set, ["a"]),
            _make(label_set, ["b"]),
        ]
        result = VoteAggregator().combine(chunks, label_set)
        histogram = result.metadata["aggregator.label_histogram"]
        assert histogram == {"a": 2, "b": 1}


# ---------------------------------------------------------------------------
# MajorityAggregator
# ---------------------------------------------------------------------------


class TestMajorityAggregator:
    def test_threshold_below_50_returns_abstain(self, label_set: LabelSet) -> None:
        chunks = [
            _make(label_set, ["a"]),
            _make(label_set, ["b"]),
            _make(label_set, ["c"]),
        ]
        result = MajorityAggregator().combine(chunks, label_set)
        assert result.abstained is True

    def test_strict_threshold(self, label_set: LabelSet) -> None:
        chunks = [
            _make(label_set, ["a"]),
            _make(label_set, ["a"]),
            _make(label_set, ["a"]),
            _make(label_set, ["b"]),
        ]
        result = MajorityAggregator(threshold=0.75).combine(chunks, label_set)
        assert result.names == ["a"]

    def test_invalid_threshold(self) -> None:
        with pytest.raises(ValueError):
            MajorityAggregator(threshold=0)
        with pytest.raises(ValueError):
            MajorityAggregator(threshold=1.5)


# ---------------------------------------------------------------------------
# UnionAggregator
# ---------------------------------------------------------------------------


class TestUnionAggregator:
    def test_picks_any(self, multi_label_set: LabelSet) -> None:
        chunks = [
            _make(multi_label_set, ["a"]),
            _make(multi_label_set, ["b", "c"]),
        ]
        result = UnionAggregator().combine(chunks, multi_label_set)
        assert set(result.names) == {"a", "b", "c"}
        assert result.abstained is False

    def test_preserves_label_set_order(self, multi_label_set: LabelSet) -> None:
        # Even though chunk-1 has c first, the output should follow
        # the LabelSet order (a, b, c).
        chunks = [
            _make(multi_label_set, ["c"]),
            _make(multi_label_set, ["a"]),
        ]
        result = UnionAggregator().combine(chunks, multi_label_set)
        assert result.names == ["a", "c"]

    def test_empty_returns_abstain(self, multi_label_set: LabelSet) -> None:
        result = UnionAggregator().combine([], multi_label_set)
        assert result.abstained is True


# ---------------------------------------------------------------------------
# IntersectionAggregator
# ---------------------------------------------------------------------------


class TestIntersectionAggregator:
    def test_basic(self, multi_label_set: LabelSet) -> None:
        chunks = [
            _make(multi_label_set, ["a", "b"]),
            _make(multi_label_set, ["b", "c"]),
        ]
        result = IntersectionAggregator().combine(chunks, multi_label_set)
        assert result.names == ["b"]

    def test_no_overlap_returns_abstain(self, multi_label_set: LabelSet) -> None:
        chunks = [
            _make(multi_label_set, ["a"]),
            _make(multi_label_set, ["b"]),
        ]
        result = IntersectionAggregator().combine(chunks, multi_label_set)
        assert result.abstained is True


# ---------------------------------------------------------------------------
# WeightedAggregator
# ---------------------------------------------------------------------------


class TestWeightedAggregator:
    def test_uniform_weights(self, label_set: LabelSet) -> None:
        chunks = [
            _make(label_set, ["a"]),
            _make(label_set, ["a"]),
            _make(label_set, ["b"]),
        ]
        result = WeightedAggregator(threshold=0.5).combine(chunks, label_set)
        assert result.names == ["a"]

    def test_chunk_weight_callable(self, label_set: LabelSet) -> None:
        chunks = [
            _make(label_set, ["a"], scores={"a": 0.1}),
            _make(label_set, ["b"], scores={"b": 0.9}),
        ]

        # Weight by confidence (top score).
        def weight_fn(c: Classification[Label]) -> float:
            return next(iter(c.scores.values())) if c.scores else 1.0

        result = WeightedAggregator(threshold=0.5, chunk_weight=weight_fn).combine(
            chunks, label_set
        )
        # b has much higher weight, so it wins.
        assert result.names == ["b"]

    def test_multi_label_mode(self, multi_label_set: LabelSet) -> None:
        chunks = [
            _make(multi_label_set, ["a", "b"]),
            _make(multi_label_set, ["b", "c"]),
        ]
        # Total weight = 2.0. threshold=0.75 requires ≥ 1.5 → only 'b'
        # qualifies (appears in both chunks for total weight 2.0).
        # 'a' and 'c' each appear once for weight 1.0 < 1.5.
        result = WeightedAggregator(threshold=0.75).combine(chunks, multi_label_set)
        assert result.names == ["b"]


# ---------------------------------------------------------------------------
# MaxScoreAggregator
# ---------------------------------------------------------------------------


class TestMaxScoreAggregator:
    def test_picks_highest_max_score(self, label_set: LabelSet) -> None:
        chunks = [
            _make(label_set, ["a"], scores={"a": 0.4, "b": 0.3}),
            _make(label_set, ["b"], scores={"a": 0.1, "b": 0.9}),
        ]
        result = MaxScoreAggregator().combine(chunks, label_set)
        assert result.names == ["b"]
        assert result.scores["b"] == pytest.approx(0.9)
        assert result.scores["a"] == pytest.approx(0.4)

    def test_below_threshold_returns_abstain(self, label_set: LabelSet) -> None:
        chunks = [_make(label_set, ["a"], scores={"a": 0.2})]
        result = MaxScoreAggregator(threshold=0.5).combine(chunks, label_set)
        assert result.abstained is True

    def test_multi_label_with_threshold(self, multi_label_set: LabelSet) -> None:
        chunks = [
            _make(multi_label_set, ["a", "b"], scores={"a": 0.5, "b": 0.4, "c": 0.1}),
        ]
        result = MaxScoreAggregator(threshold=0.45).combine(chunks, multi_label_set)
        assert result.names == ["a"]


# ---------------------------------------------------------------------------
# Top-level exposure
# ---------------------------------------------------------------------------


def test_aggregators_exposed_from_top_level_module() -> None:
    import kaos_llm_core

    for name in [
        "Aggregator",
        "VoteAggregator",
        "MajorityAggregator",
        "UnionAggregator",
        "IntersectionAggregator",
        "WeightedAggregator",
        "MaxScoreAggregator",
    ]:
        assert hasattr(kaos_llm_core, name)
        assert name in kaos_llm_core.__all__
