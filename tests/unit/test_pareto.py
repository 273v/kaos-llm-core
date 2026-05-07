"""Tests for ParetoOptimizer and compute_pareto_frontier."""

from __future__ import annotations

from kaos_llm_core.optimization.pareto import compute_pareto_frontier


class TestComputeParetoFrontier:
    def test_single_trial_is_on_frontier(self) -> None:
        trials = [({"m": "a"}, 0.9, 0.05)]
        assert compute_pareto_frontier(trials) == trials

    def test_dominated_excluded(self) -> None:
        trials = [
            ({"m": "a"}, 0.9, 0.05),  # dominates b
            ({"m": "b"}, 0.8, 0.10),  # dominated by a
        ]
        frontier = compute_pareto_frontier(trials)
        assert len(frontier) == 1
        assert frontier[0][0] == {"m": "a"}

    def test_two_non_dominated(self) -> None:
        # cheap-ok and expensive-great
        trials = [
            ({"m": "cheap"}, 0.7, 0.01),
            ({"m": "mid"}, 0.8, 0.05),  # dominated by great on metric OR by cheap on cost? No.
            ({"m": "great"}, 0.95, 0.20),
        ]
        frontier = compute_pareto_frontier(trials)
        models = {c["m"] for c, _, _ in frontier}
        assert "cheap" in models
        assert "great" in models
        assert "mid" in models  # not dominated (cheap is worse metric, great is worse cost)

    def test_mid_dominated_by_combo(self) -> None:
        # A point dominated by a single other point.
        trials = [
            ({"m": "cheap"}, 0.9, 0.01),  # dominates mid
            ({"m": "mid"}, 0.85, 0.05),
            ({"m": "great"}, 0.95, 0.20),
        ]
        frontier = compute_pareto_frontier(trials)
        models = {c["m"] for c, _, _ in frontier}
        assert "mid" not in models
        assert "cheap" in models
        assert "great" in models

    def test_equal_points_not_dominated(self) -> None:
        # Two identical points: neither strictly better, both on frontier.
        trials = [
            ({"m": "a"}, 0.9, 0.10),
            ({"m": "b"}, 0.9, 0.10),
        ]
        frontier = compute_pareto_frontier(trials)
        # With "at least one strict" rule, neither dominates the other.
        assert len(frontier) == 2

    def test_frontier_sorted_by_cost(self) -> None:
        trials = [
            ({"m": "great"}, 0.95, 0.20),
            ({"m": "cheap"}, 0.7, 0.01),
        ]
        frontier = compute_pareto_frontier(trials)
        costs = [c for _, _, c in frontier]
        assert costs == sorted(costs)
