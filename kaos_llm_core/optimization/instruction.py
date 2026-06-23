"""InstructionOptimizer — LLM-driven instruction improvement.

Uses a strong model to analyze failures from the current program and
propose better instructions. The proposed instructions are evaluated
against a validation set; if they improve the metric, they're accepted.

Algorithm:
1. Evaluate the program on the validation set (baseline)
2. Collect failures (examples where score < 1.0)
3. Send failures + current instruction to a proposer LLM
4. The proposer generates an improved instruction
5. Swap in the new instruction and re-evaluate
6. Accept if validation score improves; repeat for max_trials

Example::

    optimizer = InstructionOptimizer(
        metric=exact_match,
        proposer_model="anthropic:claude-sonnet-4-6",
        max_trials=3,
    )
    result = await optimizer.optimize(
        call=my_call,
        val_set=val_examples,
    )
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from kaos_core.logging import get_logger

from kaos_llm_core.optimization.base import OptimizerBase, resolve_config
from kaos_llm_core.optimization.budget import Budget, StopReason
from kaos_llm_core.optimization.evaluation import EvalResult, MetricFn, evaluate
from kaos_llm_core.optimization.mutations import MutationLog, RunContext
from kaos_llm_core.programs.call import Call
from kaos_llm_core.signatures import InputField, OutputField, Signature

logger = get_logger(__name__)


class ProposeInstruction(Signature):
    """You are an expert prompt engineer. Analyze why an LLM program is failing
    on certain examples and propose an improved system instruction.

    Study the failure cases carefully. Identify patterns — what types of inputs
    cause errors? What does the LLM get wrong? Then write a clear, specific
    instruction that addresses those failure patterns while maintaining good
    performance on the cases that already work.
    """

    current_instruction: str = InputField(description="The current system instruction")
    signature_description: str = InputField(
        description="Description of the LLM function's purpose and output schema"
    )
    failures_json: str = InputField(
        description="JSON array of failure cases: [{inputs, expected, got, score}]"
    )
    proposed_instruction: str = OutputField(
        description=(
            "The improved system instruction. Must be complete and self-contained. "
            "Address the specific failure patterns identified."
        )
    )
    rationale: str = OutputField(description="Explanation of what was changed and why")


@dataclass(frozen=True, slots=True)
class InstructionResult:
    """Result of instruction optimization."""

    metric_before: float
    metric_after: float
    instruction_before: str
    instruction_after: str
    rationale: str
    trials: int
    accepted: bool
    eval_before: EvalResult
    eval_after: EvalResult
    stop_reason: str = StopReason.COMPLETED.value


@dataclass(frozen=True, slots=True)
class InstructionConfig:
    """Typed configuration for :class:`InstructionOptimizer`. Phase 16.5."""

    proposer_model: str = "anthropic:claude-sonnet-4-6"
    max_trials: int = 3
    patience: int = 2
    max_failures_shown: int = 10

    def __post_init__(self) -> None:
        if self.max_trials < 1:
            msg = "InstructionConfig.max_trials must be >= 1"
            raise ValueError(msg)
        if self.patience < 1:
            msg = "InstructionConfig.patience must be >= 1"
            raise ValueError(msg)
        if self.max_failures_shown < 1:
            msg = "InstructionConfig.max_failures_shown must be >= 1"
            raise ValueError(msg)


class InstructionOptimizer(OptimizerBase):
    """Improve a Call's instruction by analyzing failures with a proposer LLM.

    Phase 13F: migrated onto :class:`OptimizerBase`. Each trial scope
    wraps BOTH the proposer Call (which produces the new instruction)
    AND the evaluation under that instruction, so the trial accumulator
    automatically captures the proposer's cost — the same structural
    fix that solves the Reflective proposer-cost bug in Phase 13G.

    Args:
        metric: Scoring function (prediction, gold_outputs) -> float in [0, 1].
        proposer_model: Model to use for proposing new instructions.
        max_trials: Maximum optimization attempts.
        patience: Stop after this many trials without improvement.
        max_failures_shown: Max failure cases to show the proposer.
        mutation_log: Optional log for recording optimization history.
    """

    def __init__(
        self,
        metric: MetricFn,
        *,
        config: InstructionConfig | None = None,
        mutation_log: MutationLog | None = None,
        budget: Budget | None = None,
        **overrides: Any,
    ) -> None:
        super().__init__(metric=metric, mutation_log=mutation_log, budget=budget)
        cfg = resolve_config(InstructionConfig, config, overrides)
        self.config = cfg
        self.proposer_model = cfg.proposer_model
        self.max_trials = cfg.max_trials
        self.patience = cfg.patience
        self.max_failures_shown = cfg.max_failures_shown

    async def optimize(
        self,
        call: Call,
        val_set: list[Any],
        *,
        max_concurrent: int = 5,
        run_context: RunContext | None = None,
    ) -> InstructionResult:
        """Run instruction optimization.

        Args:
            call: The Call to optimize.
            val_set: Validation examples (list[Example]).
            max_concurrent: Max concurrent evaluations.
            run_context: Optional shared :class:`RunContext` for stamping
                mutations with cross-stage identity. If None, a fresh one is
                created so this optimizer can run standalone.
        """
        from kaos_llm_core.types import Example

        examples: list[Example] = list(val_set)

        tracker, ctx, runner = self._make_run_state(run_context)
        stop_reason: str = StopReason.COMPLETED.value

        # Baseline evaluation
        logger.debug("InstructionOptimizer: baseline evaluation on %d examples", len(examples))
        eval_before, baseline_trial = await self._evaluate_in_trial(
            call,
            examples,
            runner=runner,
            trial_name="instruction_baseline",
            max_concurrent=max_concurrent,
        )
        best_score = eval_before.score
        best_instruction = call.instructions
        original_instruction = call.instructions
        best_eval = eval_before
        trials_without_improvement = 0
        self._consume_trial(tracker, baseline_trial)

        proposer = Call(ProposeInstruction, model=self.proposer_model)

        trial = -1
        for trial in range(self.max_trials):
            if tracker is not None:
                exhausted = tracker.exhausted()
                if exhausted is not None:
                    stop_reason = exhausted.value
                    logger.debug("InstructionOptimizer: budget exhausted (%s)", stop_reason)
                    break
            logger.debug(
                "InstructionOptimizer: trial %d/%d (best=%.1f%%)",
                trial + 1,
                self.max_trials,
                best_score * 100,
            )

            # Collect failures from the most recent evaluation
            current_eval = best_eval
            failures = current_eval.failures()[: self.max_failures_shown]

            if not failures:
                logger.debug("InstructionOptimizer: no failures to analyze, stopping")
                break

            # Format failures for the proposer
            failure_dicts = []
            for f in failures:
                entry: dict[str, Any] = {
                    "inputs": f.example.inputs,
                    "expected": f.example.outputs,
                    "score": f.score,
                }
                if f.prediction is not None:
                    try:
                        entry["got"] = {
                            k: v for k, v in f.prediction.__dict__.items() if not k.startswith("_")
                        }
                    except AttributeError:
                        entry["got"] = str(f.prediction)
                failure_dicts.append(entry)

            # Get the signature's docstring as description
            sig_desc = call.signature.__doc__ or call.signature.__name__

            # Per-trial scope: BOTH the proposer call AND the
            # evaluation charge into the same trial accumulator. The
            # proposer-cost leak that the Phase 9c manual fix patched
            # is now structurally impossible — every Call inside the
            # ``with runner.trial(...)`` block publishes its
            # invocation usage automatically via
            # ``Call._run_pipeline``.
            with runner.trial(f"instruction_trial_{trial + 1}") as trial_acc:
                proposal = await proposer(
                    current_instruction=call.instructions,
                    signature_description=sig_desc,
                    failures_json=json.dumps(failure_dicts, indent=2, default=str),
                )

                proposed_instruction = proposal.proposed_instruction
                rationale = proposal.rationale

                # Swap instruction and re-evaluate
                call.instructions = proposed_instruction
                trial_eval = await evaluate(
                    call, examples, self.metric, max_concurrent=max_concurrent
                )
            trial_score = trial_eval.score

            # Record mutation
            mutation = ctx.make_mutation(
                strategy="instruction_tuning",
                mutation_type="change_instructions",
                call_name=call.signature.__name__,
                before={"instructions": best_instruction},
                after={"instructions": proposed_instruction},
                rationale=rationale,
                metric_before=best_score,
                metric_after=trial_score,
                tokens_used=trial_acc.total_tokens,
                cost_usd=trial_acc.cost_usd,
                accepted=trial_score > best_score,
                duration_ms=trial_acc.duration_ms,
            )
            self.mutation_log.record(mutation)
            self._consume_trial(tracker, trial_acc)

            if trial_score > best_score:
                logger.debug(
                    "InstructionOptimizer: trial %d improved %.1f%% → %.1f%%",
                    trial + 1,
                    best_score * 100,
                    trial_score * 100,
                )
                best_score = trial_score
                best_instruction = proposed_instruction
                best_eval = trial_eval
                trials_without_improvement = 0
            else:
                logger.debug(
                    "InstructionOptimizer: trial %d did not improve (%.1f%% vs %.1f%%)",
                    trial + 1,
                    trial_score * 100,
                    best_score * 100,
                )
                call.instructions = best_instruction
                trials_without_improvement += 1
                if trials_without_improvement >= self.patience:
                    logger.debug("InstructionOptimizer: patience exhausted, stopping")
                    stop_reason = StopReason.PATIENCE.value
                    break

        # Apply best instruction
        accepted = best_score > eval_before.score
        call.instructions = best_instruction if accepted else original_instruction

        return InstructionResult(
            metric_before=eval_before.score,
            metric_after=best_score,
            instruction_before=original_instruction,
            instruction_after=best_instruction if accepted else original_instruction,
            rationale=self.mutation_log.mutations[-1].rationale
            if self.mutation_log.mutations
            else "",
            trials=min(trial + 1, self.max_trials) if trial >= 0 else 0,
            accepted=accepted,
            eval_before=eval_before,
            eval_after=best_eval,
            stop_reason=stop_reason,
        )
