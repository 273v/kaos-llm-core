"""BootstrapOptimizer — automatic few-shot example selection.

Runs the program on training data, evaluates each output with the metric,
and keeps the passing examples as few-shot demonstrations. This is the
simplest optimizer: it doesn't modify instructions or hyperparameters,
it just finds good examples that teach the LLM by demonstration.

Algorithm:
1. Run the program on each training example (no few-shot yet)
2. Score each output with the metric
3. Examples where score >= threshold become candidate few-shot demos
4. Re-evaluate on validation set with the selected demos
5. If validation score improves, accept the new demos

Example::

    optimizer = BootstrapOptimizer(
        metric=exact_match,
        max_examples=5,
    )
    improved = await optimizer.optimize(
        call=my_call,
        train_set=train_examples,
        val_set=val_examples,
    )
    print(f"Before: {improved.metric_before:.1%}, After: {improved.metric_after:.1%}")
    print(f"Selected {len(improved.examples_added)} few-shot demos")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kaos_core.logging import get_logger

from kaos_llm_core.optimization.base import OptimizerBase, resolve_config
from kaos_llm_core.optimization.budget import Budget, StopReason
from kaos_llm_core.optimization.evaluation import EvalResult, MetricFn, evaluate
from kaos_llm_core.optimization.mutations import MutationLog, RunContext
from kaos_llm_core.optimization.trial_runner import Trial, TrialRunner
from kaos_llm_core.programs.call import Call
from kaos_llm_core.types import Example


async def bootstrap_one_demo_set(
    *,
    call: Call,
    train_slice: list[Example],
    metric: MetricFn,
    runner: TrialRunner,
    trial_name: str,
    max_demos: int,
    score_threshold: float = 1.0,
    max_concurrent: int = 5,
) -> tuple[list[Example], Trial]:
    """Run ``call`` on a training slice and return the passing examples.

    Phase 17.1: extracted from :meth:`BootstrapOptimizer.optimize`'s
    inner loop so :class:`MiproV2Optimizer` can call it N times to
    build N candidate demo sets per predictor (DSPy's
    ``create_n_fewshot_demo_sets`` semantics, single-predictor
    variant).

    The semantics: evaluate ``call`` against ``train_slice`` inside a
    ``runner.trial(trial_name)`` scope so the cost is attributed,
    keep every example whose per-example score is at or above
    ``score_threshold`` and which did not raise an error, and return
    the first ``max_demos`` such examples paired with the trial
    accumulator.

    The caller is responsible for shuffling the train slice (the
    helper does not shuffle internally so the caller controls the
    seed).

    Returns
    -------
    tuple[list[Example], Trial]
        ``(selected_demos, trial)`` where ``selected_demos`` is a
        possibly empty list of length ``<= max_demos`` and ``trial``
        is the trial accumulator with cost / token / duration totals
        for the train-slice evaluation.
    """
    with runner.trial(trial_name) as trial:
        train_eval = await evaluate(call, train_slice, metric, max_concurrent=max_concurrent)

    candidates = [
        r.example for r in train_eval.per_example if r.score >= score_threshold and r.error is None
    ]
    return candidates[:max_demos], trial


@dataclass(frozen=True, slots=True)
class BootstrapConfig:
    """Typed configuration for :class:`BootstrapOptimizer`. Phase 16.5."""

    max_examples: int = 5
    score_threshold: float = 1.0

    def __post_init__(self) -> None:
        if self.max_examples < 1:
            msg = "BootstrapConfig.max_examples must be >= 1"
            raise ValueError(msg)
        if not 0.0 <= self.score_threshold <= 1.0:
            msg = "BootstrapConfig.score_threshold must be in [0.0, 1.0]"
            raise ValueError(msg)


logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """Result of bootstrap optimization."""

    metric_before: float
    metric_after: float
    examples_added: list[Example]
    eval_before: EvalResult
    eval_after: EvalResult
    accepted: bool
    stop_reason: str = StopReason.COMPLETED.value


class BootstrapOptimizer(OptimizerBase):
    """Select few-shot examples by running on training data and keeping successes.

    Phase 13B: migrated onto :class:`OptimizerBase`. The hand-rolled
    ``BudgetTracker`` plumbing and ``estimate_eval_cost`` walks are
    replaced with :meth:`OptimizerBase._evaluate_in_trial` /
    :meth:`OptimizerBase._consume_trial`. Every Call invocation inside
    a trial scope automatically charges its cost to the matching
    :class:`Trial` via the ``publish_invocation`` hook on
    ``Call._run_pipeline`` — no walking ``EvalResult.per_example`` to
    estimate cost after the fact.

    Args:
        metric: Scoring function (prediction, gold_outputs) -> float in [0, 1].
        max_examples: Maximum number of few-shot examples to select.
        score_threshold: Minimum score for an example to be considered a demo.
        mutation_log: Optional MutationLog for recording optimization history.
    """

    def __init__(
        self,
        metric: MetricFn,
        *,
        config: BootstrapConfig | None = None,
        mutation_log: MutationLog | None = None,
        budget: Budget | None = None,
        **overrides: Any,
    ) -> None:
        super().__init__(metric=metric, mutation_log=mutation_log, budget=budget)
        cfg = resolve_config(BootstrapConfig, config, overrides)
        self.config = cfg
        self.max_examples = cfg.max_examples
        self.score_threshold = cfg.score_threshold

    async def optimize(
        self,
        call: Call,
        train_set: list[Example],
        val_set: list[Example],
        *,
        max_concurrent: int = 5,
        run_context: RunContext | None = None,
    ) -> BootstrapResult:
        """Run bootstrap optimization.

        1. Evaluate on val_set without examples (baseline)
        2. Run on train_set, collect passing examples
        3. Add passing examples as few-shot demos
        4. Re-evaluate on val_set with demos
        5. Accept if validation improves
        """
        tracker, ctx, runner = self._make_run_state(run_context)
        stop_reason: str = StopReason.COMPLETED.value

        # Step 1: Baseline evaluation on val_set
        logger.debug("BootstrapOptimizer: baseline evaluation on %d val examples", len(val_set))
        eval_before, baseline_trial = await self._evaluate_in_trial(
            call,
            val_set,
            runner=runner,
            trial_name="bootstrap_baseline",
            max_concurrent=max_concurrent,
        )
        metric_before = eval_before.score
        exhausted = self._consume_trial(tracker, baseline_trial)
        if exhausted is not None:
            stop_reason = exhausted.value
            logger.debug("BootstrapOptimizer: budget exhausted after baseline (%s)", stop_reason)
            return BootstrapResult(
                metric_before=metric_before,
                metric_after=metric_before,
                examples_added=[],
                eval_before=eval_before,
                eval_after=eval_before,
                accepted=False,
                stop_reason=stop_reason,
            )

        # Step 2 + 3: Run on the training set and select passing examples
        # via the shared helper. The helper does the same evaluate +
        # filter + slice that the inline code used to do; extracted in
        # Phase 17.1 so MiproV2Optimizer can reuse it for its
        # multi-set bootstrap loop.
        logger.debug("BootstrapOptimizer: running on %d training examples", len(train_set))
        selected, train_trial = await bootstrap_one_demo_set(
            call=call,
            train_slice=train_set,
            metric=self.metric,
            runner=runner,
            trial_name="bootstrap_train",
            max_demos=self.max_examples,
            score_threshold=self.score_threshold,
            max_concurrent=max_concurrent,
        )
        self._consume_trial(tracker, train_trial)

        if not selected:
            logger.debug("BootstrapOptimizer: no passing examples found in training set")
            mutation = ctx.make_mutation(
                strategy="bootstrap",
                mutation_type="add_examples",
                call_name=call.signature.__name__,
                before={"n_examples": len(call.examples)},
                after={"n_examples": len(call.examples)},
                rationale="No training examples passed the score threshold.",
                metric_before=metric_before,
                metric_after=metric_before,
                accepted=False,
                duration_ms=baseline_trial.duration_ms,
            )
            self.mutation_log.record(mutation)
            return BootstrapResult(
                metric_before=metric_before,
                metric_after=metric_before,
                examples_added=[],
                eval_before=eval_before,
                eval_after=eval_before,
                accepted=False,
                stop_reason=stop_reason,
            )

        # Step 4: Add demos and re-evaluate
        original_examples = list(call.examples)
        call.examples = original_examples + selected

        logger.debug("BootstrapOptimizer: re-evaluating with %d demos", len(selected))
        eval_after, after_trial = await self._evaluate_in_trial(
            call,
            val_set,
            runner=runner,
            trial_name="bootstrap_with_demos",
            max_concurrent=max_concurrent,
        )
        metric_after = eval_after.score

        # Step 5: Accept if improved
        accepted = metric_after > metric_before
        if not accepted:
            call.examples = original_examples
            logger.debug(
                "BootstrapOptimizer: rejected — val score %.1f%% → %.1f%%",
                metric_before * 100,
                metric_after * 100,
            )
        else:
            logger.debug(
                "BootstrapOptimizer: accepted — val score %.1f%% → %.1f%% (+%.1f%%)",
                metric_before * 100,
                metric_after * 100,
                (metric_after - metric_before) * 100,
            )

        # Trial cost/tokens come straight off the trial accumulator —
        # no ``estimate_eval_cost`` walk required.
        mutation = ctx.make_mutation(
            strategy="bootstrap",
            mutation_type="add_examples",
            call_name=call.signature.__name__,
            before={"n_examples": len(original_examples)},
            after={"n_examples": len(call.examples)},
            rationale=f"Selected {len(selected)} demos from {len(train_set)} training examples "
            f"(threshold={self.score_threshold}).",
            metric_before=metric_before,
            metric_after=metric_after,
            tokens_used=after_trial.total_tokens,
            cost_usd=after_trial.cost_usd,
            accepted=accepted,
            duration_ms=after_trial.duration_ms,
        )
        self.mutation_log.record(mutation)

        exhausted = self._consume_trial(tracker, after_trial)
        if exhausted is not None:
            stop_reason = exhausted.value

        return BootstrapResult(
            metric_before=metric_before,
            metric_after=metric_after,
            examples_added=selected if accepted else [],
            eval_before=eval_before,
            eval_after=eval_after,
            accepted=accepted,
            stop_reason=stop_reason,
        )
