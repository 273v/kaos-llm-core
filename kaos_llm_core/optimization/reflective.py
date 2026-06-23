"""ReflectiveOptimizer — self-critique-driven improvement.

Generates output, has a critic LLM identify weaknesses, then uses the
critique to improve instructions. Unlike InstructionOptimizer which
analyzes failure cases against gold labels, ReflectiveOptimizer works
without labeled data — it uses the LLM's own judgment to improve.

Algorithm:
1. Run the program on sample inputs
2. Have a critic LLM evaluate each output (no gold labels needed)
3. Collect critiques into an improvement proposal
4. Generate improved instructions from the critiques
5. Re-run and verify improvement via critic scores

Example::

    optimizer = ReflectiveOptimizer(
        critic_model="anthropic:claude-sonnet-4-6",
        max_trials=3,
    )
    result = await optimizer.optimize(call, sample_inputs)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from kaos_core.logging import get_logger

from kaos_llm_core.optimization.base import OptimizerBase, resolve_config
from kaos_llm_core.optimization.budget import Budget, StopReason
from kaos_llm_core.optimization.mutations import MutationLog, RunContext
from kaos_llm_core.programs.call import Call
from kaos_llm_core.signatures import InputField, OutputField, Signature

logger = get_logger(__name__)


class CritiqueSignature(Signature):
    """Critique the quality of an LLM output.

    You are a rigorous quality reviewer. Evaluate the output for accuracy,
    completeness, clarity, and adherence to the task description. Provide
    a quality score and specific, actionable feedback.
    """

    task_description: str = InputField(description="What the LLM was asked to do")
    input_text: str = InputField(description="The input that was given")
    output_text: str = InputField(description="The output to critique")
    quality_score: float = OutputField(
        description="Quality score 0.0 (terrible) to 1.0 (excellent)"
    )
    strengths: list[str] = OutputField(description="What the output does well")
    weaknesses: list[str] = OutputField(description="Specific weaknesses or errors")
    suggestions: list[str] = OutputField(description="Concrete suggestions for improvement")


class ImproveInstructionSignature(Signature):
    """Generate an improved system instruction based on critique feedback.

    You are an expert prompt engineer. Given the current instruction and
    a set of critiques, write a better instruction that addresses the
    identified weaknesses while preserving the strengths.
    """

    current_instruction: str = InputField(description="The current system instruction")
    task_description: str = InputField(description="What the LLM function does")
    critiques_json: str = InputField(
        description="JSON array of critiques with scores, weaknesses, and suggestions"
    )
    improved_instruction: str = OutputField(
        description="The complete improved instruction. Self-contained and specific."
    )
    changes_made: str = OutputField(description="Summary of what was changed and why")


@dataclass(frozen=True, slots=True)
class ReflectiveResult:
    """Result of reflective optimization."""

    avg_score_before: float
    avg_score_after: float
    instruction_before: str
    instruction_after: str
    changes_made: str
    trials: int
    accepted: bool
    critiques: list[dict[str, Any]]
    stop_reason: str = StopReason.COMPLETED.value


@dataclass(frozen=True, slots=True)
class ReflectiveConfig:
    """Typed configuration for :class:`ReflectiveOptimizer`. Phase 16.5."""

    critic_model: str = "anthropic:claude-sonnet-4-6"
    proposer_model: str | None = None
    max_trials: int = 3
    patience: int = 2
    sample_count: int = 5
    quality_threshold: float = 0.9

    def __post_init__(self) -> None:
        if self.max_trials < 1:
            msg = "ReflectiveConfig.max_trials must be >= 1"
            raise ValueError(msg)
        if self.patience < 1:
            msg = "ReflectiveConfig.patience must be >= 1"
            raise ValueError(msg)
        if self.sample_count < 1:
            msg = "ReflectiveConfig.sample_count must be >= 1"
            raise ValueError(msg)
        if not 0.0 <= self.quality_threshold <= 1.0:
            msg = "ReflectiveConfig.quality_threshold must be in [0.0, 1.0]"
            raise ValueError(msg)


class ReflectiveOptimizer(OptimizerBase):
    """Improve a Call's instruction through self-critique, without labeled data.

    Phase 13G: migrated onto :class:`OptimizerBase`. The Phase 9c
    manual proposer-cost accumulation is GONE — every Call inside a
    trial scope (the trainee, the critic, the proposer) automatically
    charges its cost via ``publish_invocation`` in
    ``Call._run_pipeline``. ``_critique_round`` no longer threads
    cost / token tuples through its return value; the trial
    accumulator owns that bookkeeping.

    **No external metric required.** Unlike :class:`InstructionOptimizer`
    or :class:`MiproV2Optimizer`, this optimizer works without labeled
    data or a :class:`MetricFn`. The LLM critic
    (:class:`CritiqueSignature`) serves as the quality signal: it
    produces a ``quality_score`` (0.0--1.0) for each sample output,
    and the per-trial average drives the accept/reject decision. The
    ``self.metric`` attribute inherited from :class:`OptimizerBase` is
    a sentinel that raises :class:`NotImplementedError` if called --
    this is intentional, not a missing implementation.

    Args:
        critic_model: Model for critiquing outputs (should be strong).
        proposer_model: Model for proposing improved instructions. Defaults to critic_model.
        max_trials: Maximum improvement attempts.
        patience: Stop after this many trials without improvement.
        sample_count: Number of sample inputs to critique per trial.
        quality_threshold: Stop if average quality exceeds this.
        mutation_log: Optional log for recording optimization history.
    """

    def __init__(
        self,
        *,
        config: ReflectiveConfig | None = None,
        mutation_log: MutationLog | None = None,
        budget: Budget | None = None,
        **overrides: Any,
    ) -> None:
        # ReflectiveOptimizer does NOT use ``self.metric`` — the LLM
        # critic is the quality signal (see ``_critique_round``).
        # OptimizerBase requires a metric kwarg to satisfy its contract,
        # so we pass a sentinel that raises immediately if ever invoked.
        # This turns a silent 0.0 no-op into a loud, diagnosable error
        # if future code accidentally routes through the metric path.
        def _reflective_metric_not_supported(_pred: Any, _gold: dict[str, Any]) -> float:
            msg = (
                "ReflectiveOptimizer does not use a MetricFn. Quality is measured "
                "by the LLM critic (CritiqueSignature), not by an external metric. "
                "If you need metric-based optimization, use InstructionOptimizer "
                "or MiproV2Optimizer instead."
            )
            raise NotImplementedError(msg)

        super().__init__(
            metric=_reflective_metric_not_supported,  # type: ignore[arg-type]
            mutation_log=mutation_log,
            budget=budget,
        )
        cfg = resolve_config(ReflectiveConfig, config, overrides)
        self.config = cfg
        self.critic_model = cfg.critic_model
        self.proposer_model = cfg.proposer_model or cfg.critic_model
        self.max_trials = cfg.max_trials
        self.patience = cfg.patience
        self.sample_count = cfg.sample_count
        self.quality_threshold = cfg.quality_threshold

    async def optimize(
        self,
        call: Call,
        sample_inputs: list[dict[str, Any]],
        *,
        run_context: RunContext | None = None,
    ) -> ReflectiveResult:
        """Run reflective optimization.

        Args:
            call: The Call to optimize.
            sample_inputs: List of input dicts to run through the Call.
            run_context: Optional shared :class:`RunContext` for stamping
                mutations with cross-stage identity. If None, a fresh one is
                created.
        """
        tracker, ctx, runner = self._make_run_state(run_context)
        stop_reason: str = StopReason.COMPLETED.value

        critic = Call(CritiqueSignature, model=self.critic_model)
        proposer = Call(ImproveInstructionSignature, model=self.proposer_model)

        task_desc = call.signature.__doc__ or call.signature.__name__
        original_instruction = call.instructions
        best_instruction = original_instruction
        best_score = 0.0
        all_critiques: list[dict[str, Any]] = []
        trials_without_improvement = 0

        samples = sample_inputs[: self.sample_count]

        # Initial critique round inside its own trial scope. The trial
        # accumulator captures the trainee + critic costs automatically;
        # ``_critique_round`` no longer hand-rolls a cost rollup.
        with runner.trial("reflective_baseline") as init_trial:
            best_score, critiques, _init_error = await self._critique_round(
                call, critic, samples, task_desc
            )
        all_critiques.extend(critiques)
        self._consume_trial(tracker, init_trial)
        logger.debug("ReflectiveOptimizer: initial avg quality %.1f%%", best_score * 100)

        if best_score >= self.quality_threshold:
            logger.debug("ReflectiveOptimizer: already above threshold, no optimization needed")
            return ReflectiveResult(
                avg_score_before=best_score,
                avg_score_after=best_score,
                instruction_before=original_instruction,
                instruction_after=original_instruction,
                changes_made="No changes needed — quality already above threshold.",
                trials=0,
                accepted=False,
                critiques=all_critiques,
            )

        initial_score = best_score
        trial = -1

        for trial in range(self.max_trials):
            if tracker is not None:
                exhausted = tracker.exhausted()
                if exhausted is not None:
                    stop_reason = exhausted.value
                    logger.debug("ReflectiveOptimizer: budget exhausted (%s)", stop_reason)
                    break
            logger.debug(
                "ReflectiveOptimizer: trial %d/%d (best=%.1f%%)",
                trial + 1,
                self.max_trials,
                best_score * 100,
            )

            # Propose improved instruction based on critiques
            low_scoring = [c for c in critiques if c["quality_score"] < 0.8]
            if not low_scoring:
                low_scoring = critiques

            # Phase 13G: a single trial scope wraps BOTH the proposer
            # call AND the critique round. The trial accumulator
            # captures every Invocation that runs inside the block —
            # proposer, trainee call invocations from
            # ``_critique_round``, critic invocations — and the Phase
            # 9c manual proposer-cost fix becomes structurally
            # impossible to re-introduce. Cost attribution is now a
            # property of the framework, not of optimizer-specific
            # plumbing.
            with runner.trial(f"reflective_trial_{trial + 1}") as trial_acc:
                proposer_invocation = await proposer.invoke(
                    current_instruction=call.instructions,
                    task_description=task_desc,
                    critiques_json=json.dumps(low_scoring, indent=2, default=str),
                )
                proposal = proposer_invocation.output

                # Apply and re-critique
                call.instructions = proposal.improved_instruction
                trial_score, trial_critiques, trial_error = await self._critique_round(
                    call, critic, samples, task_desc
                )
            all_critiques.extend(trial_critiques)
            self._consume_trial(tracker, trial_acc)

            mutation = ctx.make_mutation(
                strategy="reflective",
                mutation_type="change_instructions",
                call_name=call.signature.__name__,
                before={"instructions": best_instruction},
                after={"instructions": proposal.improved_instruction},
                rationale=proposal.changes_made,
                metric_before=best_score,
                metric_after=trial_score,
                tokens_used=trial_acc.total_tokens,
                cost_usd=trial_acc.cost_usd,
                accepted=trial_score > best_score,
                duration_ms=trial_acc.duration_ms,
                error=trial_error,
            )
            self.mutation_log.record(mutation)

            if trial_score > best_score:
                logger.debug(
                    "ReflectiveOptimizer: trial %d improved %.1f%% → %.1f%%",
                    trial + 1,
                    best_score * 100,
                    trial_score * 100,
                )
                best_score = trial_score
                best_instruction = proposal.improved_instruction
                critiques = trial_critiques
                trials_without_improvement = 0

                if best_score >= self.quality_threshold:
                    logger.debug("ReflectiveOptimizer: reached quality threshold, stopping")
                    break
            else:
                logger.debug(
                    "ReflectiveOptimizer: trial %d no improvement (%.1f%%)",
                    trial + 1,
                    trial_score * 100,
                )
                call.instructions = best_instruction
                trials_without_improvement += 1
                if trials_without_improvement >= self.patience:
                    logger.debug("ReflectiveOptimizer: patience exhausted, stopping")
                    stop_reason = StopReason.PATIENCE.value
                    break

        accepted = best_score > initial_score
        call.instructions = best_instruction if accepted else original_instruction

        return ReflectiveResult(
            avg_score_before=initial_score,
            avg_score_after=best_score,
            instruction_before=original_instruction,
            instruction_after=best_instruction if accepted else original_instruction,
            changes_made=self.mutation_log.mutations[-1].rationale
            if self.mutation_log.mutations
            else "",
            trials=min(trial + 1, self.max_trials) if trial >= 0 else 0,
            accepted=accepted,
            critiques=all_critiques,
            stop_reason=stop_reason,
        )

    async def _critique_round(
        self,
        call: Call,
        critic: Call,
        samples: list[dict[str, Any]],
        task_desc: str,
    ) -> tuple[float, list[dict[str, Any]], str | None]:
        """Run call on samples and critique each output.

        Returns ``(avg_score, critiques, error_summary)``.

        Phase 13G: cost and token bookkeeping is GONE. Every
        ``call.invoke()`` and ``critic.invoke()`` inside this method
        runs inside the trial scope opened by ``optimize()``, so the
        active :class:`Trial` accumulator captures their usage
        automatically via ``publish_invocation``. The Phase 9c manual
        rollup tuple is no longer threaded through the return value.
        """
        critiques: list[dict[str, Any]] = []
        error_classes: list[str] = []
        first_error: str | None = None

        for sample_input in samples:
            try:
                call_invocation = await call.invoke(**sample_input)
                output = call_invocation.output
                output_text = json.dumps(
                    {k: v for k, v in output.__dict__.items() if not k.startswith("_")},
                    default=str,
                )
            except Exception as e:
                # The trial accumulator already captured any partial
                # cost via ``publish_invocation`` from the
                # validation-exhaustion / provider-error path of
                # ``Call._run_pipeline``.
                error_classes.append(type(e).__name__)
                if first_error is None:
                    first_error = f"{type(e).__name__}: {str(e)[:120]}"
                critiques.append(
                    {
                        "input": sample_input,
                        "quality_score": 0.0,
                        "weaknesses": [f"Call failed: {e}"],
                        "strengths": [],
                        "suggestions": ["Fix the error before optimizing."],
                    }
                )
                continue

            input_text = next(
                (str(v) for v in sample_input.values() if isinstance(v, str)),
                json.dumps(sample_input, default=str),
            )

            try:
                critic_invocation = await critic.invoke(
                    task_description=task_desc,
                    input_text=input_text,
                    output_text=output_text,
                )
                critique = critic_invocation.output
                critiques.append(
                    {
                        "input": sample_input,
                        "quality_score": critique.quality_score,
                        "strengths": critique.strengths,
                        "weaknesses": critique.weaknesses,
                        "suggestions": critique.suggestions,
                    }
                )
            except Exception as e:
                error_classes.append(type(e).__name__)
                if first_error is None:
                    first_error = f"{type(e).__name__}: {str(e)[:120]}"
                logger.warning("ReflectiveOptimizer: critique failed: %s", e)
                critiques.append(
                    {
                        "input": sample_input,
                        "quality_score": 0.5,
                        "weaknesses": [f"Critique failed: {e}"],
                        "strengths": [],
                        "suggestions": [],
                    }
                )

        scores = [c["quality_score"] for c in critiques]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        # Build a one-line error summary if at least half the round errored.
        # This mirrors F1's diagnostic surface for the other optimizers.
        error_summary: str | None = None
        if error_classes and len(error_classes) >= max(1, len(samples) // 2):
            from collections import Counter

            most_common, count = Counter(error_classes).most_common(1)[0]
            error_summary = f"{most_common}: {first_error or 'unknown'} (n={count})"

        return avg_score, critiques, error_summary
