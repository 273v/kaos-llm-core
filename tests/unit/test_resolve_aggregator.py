"""Tests for :func:`kaos_llm_core.composition.resolve_aggregator`."""

from __future__ import annotations

import pytest

from kaos_llm_core.composition import (
    IntersectionAggregator,
    MajorityAggregator,
    MaxScoreAggregator,
    UnionAggregator,
    VoteAggregator,
    WeightedAggregator,
    resolve_aggregator,
)
from kaos_llm_core.errors import CallError


@pytest.mark.parametrize(
    ("name", "expected_cls"),
    [
        ("vote", VoteAggregator),
        ("majority", MajorityAggregator),
        ("union", UnionAggregator),
        ("intersection", IntersectionAggregator),
        ("weighted", WeightedAggregator),
        ("max_score", MaxScoreAggregator),
    ],
)
def test_resolve_aggregator_by_name(name: str, expected_cls: type) -> None:
    out = resolve_aggregator(name)
    assert isinstance(out, expected_cls)


def test_resolve_aggregator_passes_through_instance() -> None:
    # The pass-through case: callers can construct an aggregator with
    # custom kwargs and still flow through the same resolver API.
    inst = MajorityAggregator()
    assert resolve_aggregator(inst) is inst


def test_resolve_aggregator_unknown_string_raises() -> None:
    with pytest.raises(CallError, match="unknown aggregator"):
        resolve_aggregator("not_a_real_aggregator")


def test_resolve_aggregator_error_lists_allowed_names() -> None:
    with pytest.raises(CallError, match="union") as excinfo:
        resolve_aggregator("garbage")
    # Sanity-check: error message enumerates the allowed names so an
    # agent that hits this error can recover without reading the docs.
    msg = str(excinfo.value)
    for name in ("vote", "majority", "union", "intersection", "weighted", "max_score"):
        assert name in msg
