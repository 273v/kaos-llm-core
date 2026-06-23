"""HyperparameterOptimizer — search over generation parameters.

Tries different values of temperature, top_p, and other generation kwargs,
evaluating each configuration against a validation set. Keeps the best.

Supports grid search (exhaustive) and random search.

Example::

    optimizer = HyperparameterOptimizer(
        metric=exact_match,
        search_space={"temperature": [0.0, 0.3, 0.7, 1.0]},
    )
    result = await optimizer.optimize(call, val_set)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from kaos_core.logging import get_logger

from kaos_llm_core.optimization.base import OptimizerBase, resolve_config
from kaos_llm_core.optimization.budget import Budget, StopReason
from kaos_llm_core.optimization.evaluation import (
    EvalResult,
    MetricFn,
    summarize_eval_errors,
)
from kaos_llm_core.optimization.mutations import MutationLog, RunContext
from kaos_llm_core.programs.call import Call
from kaos_llm_core.types import Example

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class HyperparameterResult:
    """Result of hyperparameter optimization."""

    metric_before: float
    metric_after: float
    best_params: dict[str, Any]
    original_params: dict[str, Any]
    configs_tried: int
    accepted: bool
    eval_before: EvalResult
    eval_after: EvalResult
    stop_reason: str = StopReason.COMPLETED.value


def _grid_configs(search_space: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Generate all combinations from a search space."""
    if not search_space:
        return [{}]
    keys = list(search_space.keys())
    configs: list[dict[str, Any]] = [{}]
    for key in keys:
        new_configs = []
        for config in configs:
            for value in search_space[key]:
                new_configs.append({**config, key: value})
        configs = new_configs
    return configs


def _random_configs(
    search_space: dict[str, list[Any]], n: int, seed: int | None = None
) -> list[dict[str, Any]]:
    """Sample n random configurations from a search space."""
    rng = random.Random(seed)
    configs: list[dict[str, Any]] = []
    for _ in range(n):
        config = {key: rng.choice(values) for key, values in search_space.items()}
        configs.append(config)
    return configs


def _latin_hypercube_samples(
    param_grid: dict[str, list[Any]],
    n_samples: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Latin Hypercube Sampling over a discrete parameter grid.

    For each parameter, its list of candidate values is partitioned into
    ``n_samples`` bins; one representative is drawn from each bin and the
    bin order is then shuffled independently per parameter. Zipping the
    shuffled per-parameter columns yields ``n_samples`` stratified
    configurations — each parameter's bins are represented exactly once.

    Numeric and categorical parameters are treated uniformly as discrete
    choices from the grid. When ``n_samples > len(values)`` for some
    parameter, that parameter's column is oversampled with replacement
    (each bin reused in cyclic order).
    """
    if not param_grid or n_samples <= 0:
        return []

    columns: dict[str, list[Any]] = {}
    for key, values in param_grid.items():
        if not values:
            raise ValueError(
                f"LatinHypercubeSearch: parameter {key!r} has no candidate values. "
                "Provide at least one value per parameter in search_space."
            )
        # Stratify by cycling the values list up to n_samples entries.
        bins = [values[i % len(values)] for i in range(n_samples)]
        rng.shuffle(bins)
        columns[key] = bins

    return [{key: columns[key][i] for key in param_grid} for i in range(n_samples)]


@dataclass(slots=True)
class HyperparameterConfig:
    """Typed configuration for :class:`HyperparameterOptimizer`. Phase 16.5.

    ``search_space`` is required at construction (no default). The
    optimizer cannot run without it.
    """

    search_space: dict[str, list[Any]] = field(default_factory=dict)
    strategy: str = "grid"
    max_trials: int = 20
    seed: int | None = None

    def __post_init__(self) -> None:
        if not self.search_space:
            msg = (
                "HyperparameterConfig.search_space is required and must be "
                "a non-empty dict mapping parameter name to list of values."
            )
            raise ValueError(msg)
        if self.strategy not in ("grid", "random", "latin_hypercube"):
            msg = (
                f"HyperparameterConfig: unknown strategy {self.strategy!r}. "
                "Use one of 'grid', 'random', or 'latin_hypercube'."
            )
            raise ValueError(msg)
        if self.max_trials < 1:
            msg = "HyperparameterConfig.max_trials must be >= 1"
            raise ValueError(msg)


class HyperparameterOptimizer(OptimizerBase):
    """Search over generation parameters (temperature, top_p, etc.).

    Phase 13C: migrated onto :class:`OptimizerBase`. Trial cost +
    duration are pulled directly off the per-trial :class:`Trial`
    accumulator instead of via ``estimate_eval_cost(eval_result)`` +
    ``time.monotonic`` deltas.

    Args:
        metric: Scoring function (prediction, gold_outputs) -> float in [0, 1].
        search_space: Dict mapping parameter names to lists of values to try.
        strategy: "grid" for exhaustive search, "random" for random sampling.
        max_trials: Max configurations to try (for random search).
        seed: Random seed for reproducibility (random search only).
        mutation_log: Optional log for recording optimization history.
    """

    def __init__(
        self,
        metric: MetricFn,
        search_space: dict[str, list[Any]] | None = None,
        *,
        config: HyperparameterConfig | None = None,
        mutation_log: MutationLog | None = None,
        budget: Budget | None = None,
        **overrides: Any,
    ) -> None:
        # search_space is required positionally for backwards compat;
        # if it's passed positionally we lift it into the config.
        if search_space is not None and "search_space" not in overrides:
            overrides["search_space"] = search_space
        super().__init__(metric=metric, mutation_log=mutation_log, budget=budget)
        cfg = resolve_config(HyperparameterConfig, config, overrides)
        self.config = cfg
        self.search_space = cfg.search_space
        self.strategy = cfg.strategy
        self.max_trials = cfg.max_trials
        self.seed = cfg.seed

    async def optimize(
        self,
        call: Call,
        val_set: list[Example],
        *,
        max_concurrent: int = 5,
        run_context: RunContext | None = None,
    ) -> HyperparameterResult:
        """Run hyperparameter search.

        For each configuration, temporarily sets the kwargs on the Call,
        evaluates on the val set, and records the result. Restores the
        best configuration at the end.
        """
        # Generate configurations
        if self.strategy == "grid":
            configs = _grid_configs(self.search_space)
        elif self.strategy == "random":
            configs = _random_configs(self.search_space, self.max_trials, self.seed)
        else:  # latin_hypercube
            rng = random.Random(self.seed)
            configs = _latin_hypercube_samples(self.search_space, self.max_trials, rng)

        configs = configs[: self.max_trials]
        tracker, ctx, runner = self._make_run_state(run_context)
        stop_reason: str = StopReason.COMPLETED.value

        # Save original kwargs
        original_kwargs = dict(call._kwargs)

        # Baseline with current kwargs
        logger.debug("HyperparameterOptimizer: baseline evaluation")
        eval_before, baseline_trial = await self._evaluate_in_trial(
            call,
            val_set,
            runner=runner,
            trial_name="hyperparameter_baseline",
            max_concurrent=max_concurrent,
        )
        # Audit fix: charge the baseline into the budget tracker, matching
        # Bootstrap and Instruction. Pre-fix the baseline trial was
        # discarded, under-counting the cap by one evaluation.
        baseline_exhausted = self._consume_trial(tracker, baseline_trial)
        if baseline_exhausted is not None:
            stop_reason = baseline_exhausted.value
        best_score = eval_before.score
        best_params = dict(original_kwargs)
        best_eval = eval_before

        logger.debug(
            "HyperparameterOptimizer: trying %d configurations (strategy=%s)",
            len(configs),
            self.strategy,
        )

        for i, config in enumerate(configs):
            if tracker is not None:
                exhausted = tracker.exhausted()
                if exhausted is not None:
                    stop_reason = exhausted.value
                    logger.debug(
                        "HyperparameterOptimizer: budget exhausted (%s) after %d configs",
                        stop_reason,
                        i,
                    )
                    break

            # Apply this config
            effective_kwargs = {**original_kwargs, **config}
            call._kwargs = effective_kwargs

            trial_eval, trial = await self._evaluate_in_trial(
                call,
                val_set,
                runner=runner,
                trial_name=f"hyperparameter_trial_{i + 1}",
                max_concurrent=max_concurrent,
            )

            # Surface the most-common error from the trial when error_rate is
            # high enough that the optimizer should warn the user. (See F1.)
            trial_error = summarize_eval_errors(trial_eval)
            if trial_eval.error_rate >= 0.5 and trial_error:
                logger.warning(
                    "HyperparameterOptimizer: trial %d/%d has %.0f%% error rate; "
                    "most common: %s. Config: %s",
                    i + 1,
                    len(configs),
                    trial_eval.error_rate * 100,
                    trial_error,
                    config,
                )

            # GAP-3 fix: ``before`` is the FULL effective kwargs of the
            # currently best configuration, ``after`` is the FULL effective
            # kwargs of THIS trial — symmetric.
            mutation = ctx.make_mutation(
                strategy="hyperparameter",
                mutation_type="change_params",
                call_name=call.signature.__name__,
                before=best_params,
                after=effective_kwargs,
                rationale=f"Config {i + 1}/{len(configs)}: {config}",
                metric_before=best_score,
                metric_after=trial_eval.score,
                tokens_used=trial.total_tokens,
                cost_usd=trial.cost_usd,
                accepted=trial_eval.score > best_score,
                duration_ms=trial.duration_ms,
                error=trial_error,
            )
            self.mutation_log.record(mutation)
            self._consume_trial(tracker, trial)

            if trial_eval.score > best_score:
                logger.debug(
                    "HyperparameterOptimizer: config %d improved %.1f%% → %.1f%% %s",
                    i + 1,
                    best_score * 100,
                    trial_eval.score * 100,
                    config,
                )
                best_score = trial_eval.score
                best_params = {**original_kwargs, **config}
                best_eval = trial_eval
            else:
                logger.debug(
                    "HyperparameterOptimizer: config %d scored %.1f%% (best=%.1f%%)",
                    i + 1,
                    trial_eval.score * 100,
                    best_score * 100,
                )

        # Apply best params
        accepted = best_score > eval_before.score
        call._kwargs = best_params if accepted else original_kwargs

        return HyperparameterResult(
            metric_before=eval_before.score,
            metric_after=best_score,
            best_params=best_params,
            original_params=original_kwargs,
            configs_tried=len(configs),
            accepted=accepted,
            eval_before=eval_before,
            eval_after=best_eval,
            stop_reason=stop_reason,
        )
