"""CoOptimizer — hybrid multi-strategy optimization.

Runs multiple optimization strategies in sequence, each building on the
previous result. The combined effect is greater than any single strategy.

Default pipeline: bootstrap → instruction → hyperparameter.
Each stage only runs if the previous stage's result is accepted or if
there are still failures to improve.

Example::

    optimizer = CoOptimizer(
        metric=exact_match,
        proposer_model="anthropic:claude-sonnet-4-6",
        search_space={"temperature": [0.0, 0.3, 0.7]},
    )
    result = await optimizer.optimize(
        call=my_call,
        train_set=train_examples,
        val_set=val_examples,
    )
    print(f"Combined: {result.metric_before:.0%} → {result.metric_after:.0%}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kaos_core.logging import get_logger

from kaos_llm_core.optimization.base import MetaOptimizerBase, resolve_config
from kaos_llm_core.optimization.bootstrap import BootstrapOptimizer, BootstrapResult
from kaos_llm_core.optimization.budget import Budget, StopReason
from kaos_llm_core.optimization.evaluation import MetricFn
from kaos_llm_core.optimization.hyperparameter import (
    HyperparameterOptimizer,
    HyperparameterResult,
)
from kaos_llm_core.optimization.instruction import InstructionOptimizer, InstructionResult
from kaos_llm_core.optimization.mutations import MutationLog, RunContext
from kaos_llm_core.programs.call import Call
from kaos_llm_core.types import Example

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CoOptimizerResult:
    """Result of multi-strategy optimization."""

    metric_before: float
    metric_after: float
    bootstrap_result: BootstrapResult | None
    instruction_result: InstructionResult | None
    hyperparameter_result: HyperparameterResult | None
    stages_run: list[str]
    mutation_log: MutationLog
    stop_reason: str = StopReason.COMPLETED.value


@dataclass(slots=True)
class CoOptimizerConfig:
    """Typed configuration for :class:`CoOptimizer`. Phase 16.5.

    Not frozen because ``strategies`` is a list and ``search_space``
    is a dict — both are mutable types that the existing API takes
    by reference. The optimizer constructor copies them defensively
    so callers cannot accidentally mutate the optimizer's state.
    """

    strategies: list[str] | None = None
    proposer_model: str = "anthropic:claude-sonnet-4-6"
    max_bootstrap_examples: int = 4
    max_instruction_trials: int = 3
    search_space: dict[str, list[Any]] | None = None
    max_hyperparam_trials: int = 10

    def __post_init__(self) -> None:
        if self.max_bootstrap_examples < 1:
            msg = "CoOptimizerConfig.max_bootstrap_examples must be >= 1"
            raise ValueError(msg)
        if self.max_instruction_trials < 1:
            msg = "CoOptimizerConfig.max_instruction_trials must be >= 1"
            raise ValueError(msg)
        if self.max_hyperparam_trials < 1:
            msg = "CoOptimizerConfig.max_hyperparam_trials must be >= 1"
            raise ValueError(msg)


class CoOptimizer(MetaOptimizerBase):
    """Run multiple optimization strategies in sequence.

    Phase 13H: migrated onto :class:`MetaOptimizerBase`. Each child
    optimizer (Bootstrap, Instruction, Hyperparameter) is itself an
    ``OptimizerBase`` subclass after Phase 13B/F/C, so they all share
    the same trial-runner cost-attribution machinery. The composite
    threads ONE :class:`RunContext` plus the shared
    :class:`MutationLog` through every child via
    :meth:`MetaOptimizerBase._build_child_kwargs`, keeping the
    cross-stage mutation history coherent.

    Pipeline: bootstrap → instruction tuning → hyperparameter search.
    Each stage builds on the previous, and the mutation log tracks
    all trials across all strategies.

    Args:
        metric: Scoring function.
        strategies: Which strategies to run. Default: all three.
        proposer_model: Model for instruction tuning proposals.
        max_bootstrap_examples: Max few-shot examples to select.
        max_instruction_trials: Max instruction tuning attempts.
        search_space: Hyperparameter search space. None to skip.
        max_hyperparam_trials: Max hyperparameter configurations.
        mutation_log: Shared log across all strategies.
    """

    def __init__(
        self,
        metric: MetricFn,
        *,
        config: CoOptimizerConfig | None = None,
        mutation_log: MutationLog | None = None,
        budget: Budget | None = None,
        **overrides: Any,
    ) -> None:
        super().__init__(metric=metric, mutation_log=mutation_log, budget=budget)
        cfg = resolve_config(CoOptimizerConfig, config, overrides)
        self.config = cfg
        # Mirror config fields for backwards compatibility.
        self.strategies = (
            list(cfg.strategies)
            if cfg.strategies is not None
            else [
                "bootstrap",
                "instruction",
                "hyperparameter",
            ]
        )
        self.proposer_model = cfg.proposer_model
        self.max_bootstrap_examples = cfg.max_bootstrap_examples
        self.max_instruction_trials = cfg.max_instruction_trials
        self.search_space = (
            dict(cfg.search_space)
            if cfg.search_space is not None
            else {"temperature": [0.0, 0.3, 0.7, 1.0]}
        )
        self.max_hyperparam_trials = cfg.max_hyperparam_trials

    async def optimize(
        self,
        call: Call,
        train_set: list[Example] | None = None,
        val_set: list[Example] | None = None,
        *,
        max_concurrent: int = 5,
        run_context: RunContext | None = None,
    ) -> CoOptimizerResult:
        """Run the optimization pipeline.

        Args:
            call: The Call to optimize (modified in place).
            train_set: Training examples (required for bootstrap).
            val_set: Validation examples (required for all strategies).
            max_concurrent: Max concurrent evaluations.
            run_context: Optional shared :class:`RunContext`. CoOptimizer
                creates ONE RunContext at the top of its pipeline and passes
                that single instance to every child stage so that all
                mutations across bootstrap/instruction/hyperparameter share
                a single ``run_id`` and continue the trial counter rather
                than each child starting from zero.
        """
        if val_set is None:
            msg = "CoOptimizer requires val_set"
            raise ValueError(msg)

        # Composite-runtime state. ``shared_tracker`` is the cumulative
        # budget cap enforced across every child stage — without it, each
        # child would build its own fresh tracker and silently multiply
        # the configured ``Budget`` by the number of stages. Audit fix
        # for the Phase 13H regression.
        shared_tracker, ctx, runner = self._make_run_state(run_context)

        # Baseline runs through ``_evaluate_in_trial`` which uses the
        # composite's runner — its trial accumulator charges into
        # ``shared_tracker`` via ``_consume_trial`` below.
        eval_before, baseline_trial = await self._evaluate_in_trial(
            call,
            val_set,
            runner=runner,
            trial_name="co_baseline_eval",
            max_concurrent=max_concurrent,
        )
        metric_before = eval_before.score
        baseline_exhausted = self._consume_trial(shared_tracker, baseline_trial)
        logger.info("CoOptimizer: baseline %.1f%%", metric_before * 100)

        bootstrap_result: BootstrapResult | None = None
        instruction_result: InstructionResult | None = None
        hyperparameter_result: HyperparameterResult | None = None
        stages_run: list[str] = []
        stop_reason: str = StopReason.COMPLETED.value
        if baseline_exhausted is not None:
            stop_reason = baseline_exhausted.value

        def _budget_hit() -> bool:
            if shared_tracker is None:
                return False
            if stop_reason != StopReason.COMPLETED.value:
                return True
            # Direct check too — a child may have exhausted the
            # shared tracker without setting its own ``stop_reason``
            # (e.g. the cap was hit on the very last trial of the
            # stage and the child returned cleanly).
            return shared_tracker.exhausted() is not None

        # Stage 1: Bootstrap
        if "bootstrap" in self.strategies and train_set and not _budget_hit():
            logger.info("CoOptimizer: running bootstrap stage")
            stages_run.append("bootstrap")
            opt = BootstrapOptimizer(
                metric=self.metric,
                max_examples=self.max_bootstrap_examples,
                mutation_log=self.mutation_log,
                budget=self.budget,
            )
            self._attach_shared_tracker(opt, shared_tracker)
            bootstrap_result = await opt.optimize(
                call,
                train_set,
                val_set,
                max_concurrent=max_concurrent,
                run_context=ctx,
            )
            current = bootstrap_result.metric_after
            if bootstrap_result.stop_reason != StopReason.COMPLETED.value:
                stop_reason = bootstrap_result.stop_reason
            logger.info("CoOptimizer: after bootstrap %.1f%%", current * 100)

        # Stage 2: Instruction tuning
        if "instruction" in self.strategies and not _budget_hit():
            logger.info("CoOptimizer: running instruction stage")
            stages_run.append("instruction")
            opt_inst = InstructionOptimizer(
                metric=self.metric,
                proposer_model=self.proposer_model,
                max_trials=self.max_instruction_trials,
                mutation_log=self.mutation_log,
                budget=self.budget,
            )
            self._attach_shared_tracker(opt_inst, shared_tracker)
            instruction_result = await opt_inst.optimize(
                call, val_set, max_concurrent=max_concurrent, run_context=ctx
            )
            current = instruction_result.metric_after
            if instruction_result.stop_reason != StopReason.COMPLETED.value:
                stop_reason = instruction_result.stop_reason
            logger.info("CoOptimizer: after instruction %.1f%%", current * 100)

        # Stage 3: Hyperparameter search
        if "hyperparameter" in self.strategies and not _budget_hit():
            logger.info("CoOptimizer: running hyperparameter stage")
            stages_run.append("hyperparameter")
            opt_hp = HyperparameterOptimizer(
                metric=self.metric,
                search_space=self.search_space,
                max_trials=self.max_hyperparam_trials,
                mutation_log=self.mutation_log,
                budget=self.budget,
            )
            self._attach_shared_tracker(opt_hp, shared_tracker)
            hyperparameter_result = await opt_hp.optimize(
                call, val_set, max_concurrent=max_concurrent, run_context=ctx
            )
            current = hyperparameter_result.metric_after
            if hyperparameter_result.stop_reason != StopReason.COMPLETED.value:
                stop_reason = hyperparameter_result.stop_reason
            logger.info("CoOptimizer: after hyperparameter %.1f%%", current * 100)

        # Final score
        eval_after, final_trial = await self._evaluate_in_trial(
            call,
            val_set,
            runner=runner,
            trial_name="co_final_eval",
            max_concurrent=max_concurrent,
        )
        self._consume_trial(shared_tracker, final_trial)
        metric_after = eval_after.score

        logger.info(
            "CoOptimizer: complete %.1f%% → %.1f%%", metric_before * 100, metric_after * 100
        )

        return CoOptimizerResult(
            metric_before=metric_before,
            metric_after=metric_after,
            bootstrap_result=bootstrap_result,
            instruction_result=instruction_result,
            hyperparameter_result=hyperparameter_result,
            stages_run=stages_run,
            mutation_log=self.mutation_log,
            stop_reason=stop_reason,
        )
