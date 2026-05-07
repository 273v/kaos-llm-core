"""Optimization recipes — pre-built end-to-end workflows.

Each recipe is a one-line factory that returns a configured optimizer
instance with sensible defaults. They wrap the primitives in
:mod:`kaos_llm_core.optimization` without adding new behavior.

Example::

    from kaos_llm_core.optimization.recipes import FriendlyPromptTuner

    optimizer = FriendlyPromptTuner(metric=exact_match)
    result = await optimizer.optimize(call, train_set=train, val_set=val)
"""

from __future__ import annotations

from kaos_llm_core.optimization.budget import Budget
from kaos_llm_core.optimization.co_optimizer import CoOptimizer
from kaos_llm_core.optimization.evaluation import MetricFn
from kaos_llm_core.optimization.model_optimizer import ModelOptimizer

__all__ = [
    "CostAwareModelSelector",
    "ExampleFirstTuner",
    "FriendlyPromptTuner",
    "QuickWin",
]


def FriendlyPromptTuner(
    metric: MetricFn,
    *,
    proposer_model: str = "anthropic:claude-sonnet-4-6",
    budget: Budget | None = None,
) -> CoOptimizer:
    """Instruction tuning first, then few-shot bootstrap.

    The "friendly prompt" workflow: improve the instruction until the
    easy failures are fixed, then let bootstrap fill in demonstrations
    for the hard cases. Skips hyperparameter search by default — most
    users reach for hyperparameters last.
    """
    return CoOptimizer(
        metric=metric,
        strategies=["instruction", "bootstrap"],
        proposer_model=proposer_model,
        budget=budget,
    )


def ExampleFirstTuner(
    metric: MetricFn,
    *,
    proposer_model: str = "anthropic:claude-sonnet-4-6",
    budget: Budget | None = None,
) -> CoOptimizer:
    """Few-shot examples first, then instruction refinement.

    The "example first" workflow: let bootstrap pick the best demos,
    then have the instruction tuner address any remaining failure
    patterns on top of those demos.
    """
    return CoOptimizer(
        metric=metric,
        strategies=["bootstrap", "instruction"],
        proposer_model=proposer_model,
        budget=budget,
    )


def CostAwareModelSelector(
    metric: MetricFn,
    models: list[str],
    *,
    min_score: float = 0.85,
    budget: Budget | None = None,
) -> ModelOptimizer:
    """Find the cheapest model at or above a quality threshold.

    Thin wrapper over :class:`ModelOptimizer` with a sharper default
    threshold (0.85) than the bare optimizer (0.80).
    """
    return ModelOptimizer(
        metric=metric,
        models=models,
        min_score=min_score,
        budget=budget,
    )


def QuickWin(
    metric: MetricFn,
    *,
    proposer_model: str = "anthropic:claude-sonnet-4-6",
    budget: Budget | None = None,
) -> CoOptimizer:
    """Run all three Co-strategies under a tight default trial budget.

    The "quick win" workflow: try every available improvement vector
    (bootstrap, instruction, hyperparameter) but cap each one to a small
    number of trials so the whole pipeline finishes in seconds with the
    biggest cheap improvement. This is the recipe to reach for when you
    do not yet know which strategy will help — let CoOptimizer try all
    three and then go deeper with a focused recipe afterward.

    Defaults are intentionally aggressive on speed:

    * ``max_bootstrap_examples=4`` — same as :class:`CoOptimizer`'s default.
    * ``max_instruction_trials=2`` — half of CoOptimizer's default.
    * ``max_hyperparam_trials=4`` — well below the default 10.

    If you supply a ``budget`` it is shared across stages and the
    short-circuit logic in :class:`CoOptimizer` will exit cleanly when
    the cap is reached.
    """
    return CoOptimizer(
        metric=metric,
        strategies=["bootstrap", "instruction", "hyperparameter"],
        proposer_model=proposer_model,
        max_instruction_trials=2,
        max_hyperparam_trials=4,
        budget=budget,
    )
