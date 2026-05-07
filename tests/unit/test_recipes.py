"""Tests for optimization recipes."""

from __future__ import annotations

from typing import Any

from kaos_llm_core.optimization.co_optimizer import CoOptimizer
from kaos_llm_core.optimization.model_optimizer import ModelOptimizer
from kaos_llm_core.optimization.recipes import (
    CostAwareModelSelector,
    ExampleFirstTuner,
    FriendlyPromptTuner,
    QuickWin,
)


def _metric(pred: Any, gold: dict[str, Any]) -> float:
    return 0.0


class TestRecipes:
    def test_friendly_prompt_returns_co_optimizer(self) -> None:
        opt = FriendlyPromptTuner(metric=_metric)
        assert isinstance(opt, CoOptimizer)
        assert opt.strategies == ["instruction", "bootstrap"]

    def test_example_first_returns_co_optimizer(self) -> None:
        opt = ExampleFirstTuner(metric=_metric)
        assert isinstance(opt, CoOptimizer)
        assert opt.strategies == ["bootstrap", "instruction"]

    def test_cost_aware_returns_model_optimizer(self) -> None:
        opt = CostAwareModelSelector(metric=_metric, models=["openai:gpt-5.4-nano"], min_score=0.9)
        assert isinstance(opt, ModelOptimizer)
        assert opt.models == ["openai:gpt-5.4-nano"]
        assert opt.min_score == 0.9

    def test_default_proposer_model(self) -> None:
        opt = FriendlyPromptTuner(metric=_metric)
        assert "claude" in opt.proposer_model

    def test_quickwin_runs_all_three_strategies(self) -> None:
        """QuickWin (F4) — runs every strategy with tight default trial budgets."""
        opt = QuickWin(metric=_metric)
        assert isinstance(opt, CoOptimizer)
        assert opt.strategies == ["bootstrap", "instruction", "hyperparameter"]
        # The aggressive trial caps that distinguish QuickWin from raw CoOptimizer.
        assert opt.max_instruction_trials == 2
        assert opt.max_hyperparam_trials == 4

    def test_quickwin_threads_budget(self) -> None:
        """QuickWin must forward an externally-supplied Budget."""
        from kaos_llm_core.optimization.budget import Budget

        budget = Budget(max_trials=5)
        opt = QuickWin(metric=_metric, budget=budget)
        assert opt.budget is budget

    def test_cost_aware_default_threshold(self) -> None:
        opt = CostAwareModelSelector(metric=_metric, models=["m"])
        assert opt.min_score == 0.85
