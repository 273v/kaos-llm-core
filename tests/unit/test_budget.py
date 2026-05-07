"""Tests for Budget and BudgetTracker."""

from __future__ import annotations

import time

import pytest

from kaos_llm_core.optimization.budget import (
    Budget,
    StopReason,
    estimate_eval_cost,
)


class TestBudgetValidation:
    def test_rejects_empty_budget(self) -> None:
        with pytest.raises(ValueError, match="at least one cap"):
            Budget()

    def test_rejects_negative_trials(self) -> None:
        with pytest.raises(ValueError, match="max_trials"):
            Budget(max_trials=0)

    def test_rejects_negative_time(self) -> None:
        with pytest.raises(ValueError, match="max_wall_seconds"):
            Budget(max_wall_seconds=0)

    def test_rejects_negative_cost(self) -> None:
        with pytest.raises(ValueError, match="max_cost_usd"):
            Budget(max_cost_usd=-1.0)

    def test_rejects_negative_tokens(self) -> None:
        with pytest.raises(ValueError, match="max_tokens"):
            Budget(max_tokens=-1)

    def test_accepts_single_cap(self) -> None:
        b = Budget(max_trials=5)
        assert b.max_trials == 5
        assert b.max_cost_usd is None


class TestBudgetTracker:
    def test_consume_accumulates(self) -> None:
        tracker = Budget(max_trials=10).make_tracker()
        tracker.consume(trials=1, cost_usd=0.01, tokens=100)
        tracker.consume(trials=2, cost_usd=0.02, tokens=200)
        assert tracker.trials == 3
        assert tracker.cost_usd == pytest.approx(0.03)
        assert tracker.tokens == 300

    def test_consume_rejects_negative(self) -> None:
        tracker = Budget(max_trials=10).make_tracker()
        with pytest.raises(ValueError, match="negative"):
            tracker.consume(trials=-1)

    def test_exhausted_none_when_under_cap(self) -> None:
        tracker = Budget(max_trials=10).make_tracker()
        tracker.consume(trials=5)
        assert tracker.exhausted() is None

    def test_exhausted_trials(self) -> None:
        tracker = Budget(max_trials=3).make_tracker()
        tracker.consume(trials=3)
        assert tracker.exhausted() == StopReason.BUDGET_TRIALS

    def test_exhausted_cost(self) -> None:
        tracker = Budget(max_cost_usd=0.10).make_tracker()
        tracker.consume(cost_usd=0.10)
        assert tracker.exhausted() == StopReason.BUDGET_COST

    def test_exhausted_tokens(self) -> None:
        tracker = Budget(max_tokens=500).make_tracker()
        tracker.consume(tokens=500)
        assert tracker.exhausted() == StopReason.BUDGET_TOKENS

    def test_exhausted_time(self) -> None:
        tracker = Budget(max_wall_seconds=0.01).make_tracker()
        time.sleep(0.02)
        assert tracker.exhausted() == StopReason.BUDGET_TIME

    def test_make_tracker_isolation(self) -> None:
        """Two trackers from the same Budget are independent."""
        budget = Budget(max_trials=5)
        t1 = budget.make_tracker()
        t2 = budget.make_tracker()
        t1.consume(trials=3)
        assert t2.trials == 0
        assert t1.trials == 3


class TestEstimateEvalCost:
    def test_returns_zero_for_empty(self) -> None:
        class _R:
            def __init__(self) -> None:
                self.per_example: list[object] = []

        cost, tokens = estimate_eval_cost(_R())
        assert cost == 0.0
        assert tokens == 0

    def test_handles_missing_trace(self) -> None:
        class _Ex:
            def __init__(self) -> None:
                self.prediction = object()

        class _R:
            def __init__(self) -> None:
                self.per_example = [_Ex()]

        cost, tokens = estimate_eval_cost(_R())
        assert cost == 0.0
        assert tokens == 0
