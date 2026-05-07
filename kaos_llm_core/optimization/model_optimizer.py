"""ModelOptimizer — pick the cheapest model that hits a quality threshold.

Scores a Call under each candidate model, then selects the lowest-cost
model whose score meets ``min_score``. If no candidate hits the threshold,
returns the highest-scoring model with ``stop_reason="threshold_not_met"``.
If the budget is exhausted mid-run, returns the best scored so far.

Cost is approximated from ``kaos-llm-core.observability.cost`` pricing
tables plus the evaluation's accumulated token usage. Models outside the
pricing table get ``cost=0.0`` and win ties only when no priced candidate
meets the threshold.

Example::

    optimizer = ModelOptimizer(
        metric=exact_match,
        models=["openai:gpt-5.4-nano", "anthropic:claude-haiku-4-5"],
        min_score=0.9,
    )
    result = await optimizer.optimize(call, val_set)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kaos_core.logging import get_logger

from kaos_llm_core.observability.cost import PRICING
from kaos_llm_core.optimization.base import OptimizerBase, resolve_config
from kaos_llm_core.optimization.budget import Budget, StopReason
from kaos_llm_core.optimization.codec_optimizer import _primary_call
from kaos_llm_core.optimization.evaluation import MetricFn, summarize_eval_errors
from kaos_llm_core.optimization.mutations import Mutation, MutationLog, RunContext
from kaos_llm_core.types import Example

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ModelOptimizationResult:
    """Result of a model-selection run."""

    best_model: str
    best_score: float
    scores_by_model: dict[str, float]
    cost_by_model: dict[str, float]
    tokens_by_model: dict[str, int]
    mutations: list[Mutation] = field(default_factory=list)
    stop_reason: str = StopReason.COMPLETED.value


# Default token split between input and output for cost estimation.
# 70/30 reflects typical LLM workloads where prompts (system + user + context)
# consume most tokens and completions are comparatively short. Adjust if your
# workload has a different profile (e.g., code generation may be closer to 50/50).
_DEFAULT_INPUT_TOKEN_RATIO = 0.7
_DEFAULT_OUTPUT_TOKEN_RATIO = 0.3


def _estimated_model_cost(model: str, tokens: int) -> float:
    """Rough per-model cost from the pricing table.

    Splits tokens using ``_DEFAULT_INPUT_TOKEN_RATIO`` /
    ``_DEFAULT_OUTPUT_TOKEN_RATIO`` — a heuristic used only for tie-breaking
    in selection. Real cost tracking uses :func:`estimate_eval_cost` on the
    trial trace.
    """
    pricing = PRICING.get(model)
    if pricing is None:
        return 0.0
    input_tokens = int(tokens * _DEFAULT_INPUT_TOKEN_RATIO)
    output_tokens = tokens - input_tokens
    return pricing.estimate(input_tokens, output_tokens)


@dataclass(slots=True)
class ModelOptimizerConfig:
    """Typed configuration for :class:`ModelOptimizer`. Phase 16.5."""

    models: list[str] = field(default_factory=list)
    min_score: float = 0.8

    def __post_init__(self) -> None:
        if not self.models:
            msg = (
                "ModelOptimizerConfig.models is empty. Pass at least one "
                "candidate model id (e.g. 'openai:gpt-5.4-nano')."
            )
            raise ValueError(msg)
        if not 0.0 <= self.min_score <= 1.0:
            msg = "ModelOptimizerConfig.min_score must be in [0.0, 1.0]"
            raise ValueError(msg)


class ModelOptimizer(OptimizerBase):
    """Pick the cheapest model that hits a metric threshold.

    Phase 13E: migrated onto :class:`OptimizerBase`. Per-model trials
    run inside named ``Trial`` scopes; cost / tokens / duration come
    straight off the trial accumulator.

    Args:
        metric: Scoring function (prediction, gold_outputs) -> float in [0, 1].
        models: Candidate ``provider:model`` ids to try.
        min_score: Quality threshold; filters candidates before cost-sort.
        budget: Optional :class:`Budget` cap.
        mutation_log: Optional shared mutation log.
    """

    def __init__(
        self,
        metric: MetricFn,
        models: list[str] | None = None,
        *,
        config: ModelOptimizerConfig | None = None,
        budget: Budget | None = None,
        mutation_log: MutationLog | None = None,
        **overrides: Any,
    ) -> None:
        # ``models`` is required positionally for backwards compat;
        # if it's passed positionally we lift it into the overrides.
        if models is not None and "models" not in overrides:
            overrides["models"] = models
        super().__init__(metric=metric, mutation_log=mutation_log, budget=budget)
        cfg = resolve_config(ModelOptimizerConfig, config, overrides)
        self.config = cfg
        self.models = list(cfg.models)
        self.min_score = cfg.min_score

    async def optimize(
        self,
        program: Any,
        val_set: list[Example],
        *,
        max_concurrent: int = 5,
        run_context: RunContext | None = None,
    ) -> ModelOptimizationResult:
        call = _primary_call(program)
        tracker, ctx, runner = self._make_run_state(run_context)

        original_model = call._model
        scores_by_model: dict[str, float] = {}
        cost_by_model: dict[str, float] = {}
        tokens_by_model: dict[str, int] = {}
        mutations: list[Mutation] = []
        stop_reason: str = StopReason.COMPLETED.value

        logger.info(
            "ModelOptimizer: scoring %d model(s) on %d examples (min_score=%.2f)",
            len(self.models),
            len(val_set),
            self.min_score,
        )

        try:
            for model in self.models:
                if tracker is not None:
                    exhausted = tracker.exhausted()
                    if exhausted is not None:
                        stop_reason = exhausted.value
                        logger.info("ModelOptimizer: budget exhausted (%s)", stop_reason)
                        break

                call._model = model
                trial_eval, trial = await self._evaluate_in_trial(
                    call,
                    val_set,
                    runner=runner,
                    trial_name=f"model_{model}",
                    max_concurrent=max_concurrent,
                )
                score = trial_eval.score
                cost = trial.cost_usd
                tokens = trial.total_tokens
                if cost == 0.0:
                    # Fall back to pricing-table estimate so we can still tie-break.
                    cost = _estimated_model_cost(model, tokens)
                trial_error = summarize_eval_errors(trial_eval)
                if trial_eval.error_rate >= 0.5 and trial_error:
                    logger.warning(
                        "ModelOptimizer: %s has %.0f%% error rate; most common: %s",
                        model,
                        trial_eval.error_rate * 100,
                        trial_error,
                    )

                scores_by_model[model] = score
                cost_by_model[model] = cost
                tokens_by_model[model] = tokens

                mutation = ctx.make_mutation(
                    strategy="model_selection",
                    mutation_type="model_swap",
                    call_name=call.signature.__name__,
                    before={"model": original_model or ""},
                    after={"model": model},
                    rationale=f"Scored {model}: {score:.3f} @ ${cost:.6f}",
                    metric_before=0.0,
                    metric_after=score,
                    tokens_used=tokens,
                    cost_usd=cost,
                    accepted=score >= self.min_score,
                    duration_ms=trial.duration_ms,
                    error=trial_error,
                )
                self.mutation_log.record(mutation)
                mutations.append(mutation)

                self._consume_trial(tracker, trial)
        finally:
            # Will set to final pick below; restore original for now.
            call._model = original_model

        if not scores_by_model:
            raise RuntimeError(
                "ModelOptimizer: no models were evaluated. "
                "Either the budget was exhausted before the first trial, "
                "or every candidate model failed. Check the budget and the "
                "model identifiers (provider:model format)."
            )

        # Filter to models meeting the threshold.
        qualifying = [m for m, s in scores_by_model.items() if s >= self.min_score]

        if qualifying:
            best_model = min(qualifying, key=lambda m: (cost_by_model[m], self.models.index(m)))
            best_score = scores_by_model[best_model]
            if stop_reason == StopReason.COMPLETED.value:
                stop_reason = StopReason.COMPLETED.value
        else:
            # Fall back to highest-scoring overall.
            best_model = max(
                scores_by_model.keys(),
                key=lambda m: (scores_by_model[m], -self.models.index(m)),
            )
            best_score = scores_by_model[best_model]
            if stop_reason == StopReason.COMPLETED.value:
                stop_reason = StopReason.THRESHOLD_NOT_MET.value

        # Apply the selected model to the Call.
        call._model = best_model

        logger.info(
            "ModelOptimizer: selected %s (score=%.3f, cost=$%.6f, stop=%s)",
            best_model,
            best_score,
            cost_by_model.get(best_model, 0.0),
            stop_reason,
        )

        return ModelOptimizationResult(
            best_model=best_model,
            best_score=best_score,
            scores_by_model=scores_by_model,
            cost_by_model=cost_by_model,
            tokens_by_model=tokens_by_model,
            mutations=mutations,
            stop_reason=stop_reason,
        )
