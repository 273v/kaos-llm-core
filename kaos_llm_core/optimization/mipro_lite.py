"""MiproLiteOptimizer — joint search over (instructions, demos).

Phase 17.0 ("MIPRO-lite"). The minimum implementation of the MIPROv2
algorithm that delivers the architectural value the existing KAOS
``CoOptimizer`` cannot: joint search over (instruction, demo set)
pairs instead of sequential coordinate descent.

This is **not** the full DSPy MIPROv2. The full version uses Optuna
TPE for surrogate-guided sampling and a "GroundedProposer" that
walks the dataset and bootstrapped demos before each instruction
proposal. Phase 17.0 ships:

  - **Random search over the cartesian product** of K instruction
    candidates and L demo candidates. K * L is small enough (default
    24) that random search is competitive with TPE for the v0.
  - **Reused candidate generators**: instruction candidates come
    from the existing :class:`InstructionOptimizer` proposer prompt
    (one Call per instruction); demo candidates come from
    bootstrapping multiple times against random shards of the train
    set with varied score thresholds.
  - **Minibatch eval + full-eval promotion**: every random sample
    is scored on a `minibatch_size` slice of the val set; the top-N
    samples are then promoted to full-val scoring to break ties.
  - **Phase 13A trial machinery**: every eval (proposer call,
    minibatch, full-val promotion) runs inside a TrialRunner trial
    so cost attribution is automatic.

The full TPE / GroundedProposer / multi-predictor version is
specified in ``docs/internal/design/mipro-v2-equivalent.md`` as
Phase 17.1.

The architectural value vs. CoOptimizer:

  CoOptimizer (sequential):
    bootstrap → demo set D
    instruction with D fixed → instruction I
    return (I, D)
  Search space: 1 demo set x N_instr instructions = N_instr trials

  MIPRO-lite (joint):
    bootstrap K times → {D1, ..., DK}
    propose L instructions (each grounded on a random Dk) → {I1, ..., IL}
    random search over the K x L pairs, evaluate on minibatch,
    promote top-N to full eval, return the best (Ii, Dj)
  Search space: K x L pairs explored via random sampling
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from kaos_core.logging import get_logger

from kaos_llm_core.optimization.base import MetaOptimizerBase, MetricFn, resolve_config
from kaos_llm_core.optimization.bootstrap import bootstrap_one_demo_set
from kaos_llm_core.optimization.budget import Budget, StopReason
from kaos_llm_core.optimization.instruction import ProposeInstruction
from kaos_llm_core.optimization.mutations import MutationLog, RunContext
from kaos_llm_core.programs.call import Call
from kaos_llm_core.types import Example

logger = get_logger(__name__)


@dataclass
class _Candidate:
    """One sampled (instruction, demos) configuration."""

    index: int
    instruction: str
    demos: list[Example]
    minibatch_score: float = 0.0
    full_score: float | None = None
    promoted: bool = False


@dataclass(frozen=True, slots=True)
class MiproLiteResult:
    """Outcome of a MIPRO-lite optimization run."""

    metric_before: float
    metric_after: float
    best_instruction: str
    best_demos_count: int
    n_instruction_candidates: int
    n_demo_candidates: int
    n_trials: int
    n_promoted: int
    accepted: bool
    stop_reason: str
    candidates: list[_Candidate] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class MiproLiteConfig:
    """Typed configuration for :class:`MiproLiteOptimizer`.

    Phase 16.5: every optimizer now ships a frozen config dataclass.
    Construct via ``MiproLiteOptimizer(metric, config=MiproLiteConfig(...))``
    OR via the backwards-compatible kwargs shim
    ``MiproLiteOptimizer(metric, n_trials=10)``.
    """

    n_instruction_candidates: int = 4
    n_demo_candidates: int = 3
    max_demos: int = 3
    n_trials: int = 8
    minibatch_size: int = 5
    n_promote: int = 2
    proposer_model: str = "anthropic:claude-haiku-4-5"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_instruction_candidates < 1:
            msg = "n_instruction_candidates must be >= 1"
            raise ValueError(msg)
        if self.n_demo_candidates < 1:
            msg = "n_demo_candidates must be >= 1"
            raise ValueError(msg)
        if self.n_trials < 1:
            msg = "n_trials must be >= 1"
            raise ValueError(msg)
        if self.minibatch_size < 1:
            msg = "minibatch_size must be >= 1"
            raise ValueError(msg)
        if self.n_promote < 1:
            msg = "n_promote must be >= 1"
            raise ValueError(msg)


class MiproLiteOptimizer(MetaOptimizerBase):
    """MIPRO-lite: random-search joint optimization over (instructions, demos).

    Construct via either pattern:

    1. Typed config (preferred for new code)::

        opt = MiproLiteOptimizer(
            metric=accuracy,
            config=MiproLiteConfig(n_trials=12, minibatch_size=10),
        )

    2. Backwards-compatible kwargs (pre-Phase-16.5 surface)::

        opt = MiproLiteOptimizer(metric=accuracy, n_trials=12, minibatch_size=10)

    Both produce the same result. Adding a new field to ``MiproLiteConfig``
    is safe — neither pattern needs to change for callers that pass
    only a subset.
    """

    def __init__(
        self,
        metric: MetricFn,
        *,
        config: MiproLiteConfig | None = None,
        mutation_log: MutationLog | None = None,
        budget: Budget | None = None,
        **overrides: Any,
    ) -> None:
        super().__init__(metric=metric, mutation_log=mutation_log, budget=budget)
        cfg = resolve_config(MiproLiteConfig, config, overrides)
        self.config = cfg
        # Mirror config fields onto instance attrs for backwards
        # compatibility with callers that read them directly.
        self.n_instruction_candidates = cfg.n_instruction_candidates
        self.n_demo_candidates = cfg.n_demo_candidates
        self.max_demos = cfg.max_demos
        self.n_trials = cfg.n_trials
        self.minibatch_size = cfg.minibatch_size
        self.n_promote = cfg.n_promote
        self.proposer_model = cfg.proposer_model
        self.seed = cfg.seed

    async def optimize(
        self,
        call: Call,
        train_set: list[Example],
        val_set: list[Example],
        *,
        max_concurrent: int = 5,
        run_context: RunContext | None = None,
    ) -> MiproLiteResult:
        """Run the joint search.

        Args:
            call: The Call to optimize (modified in place on success).
            train_set: Training examples for bootstrapping demo sets.
            val_set: Validation examples for minibatch + full eval.
            max_concurrent: Max concurrent evals per trial.
            run_context: Optional shared run context.
        """
        if not val_set:
            msg = "MiproLiteOptimizer requires a non-empty val_set"
            raise ValueError(msg)
        if not train_set:
            msg = "MiproLiteOptimizer requires a non-empty train_set for demo bootstrap"
            raise ValueError(msg)

        rng = random.Random(self.seed)
        shared_tracker, _ctx, runner = self._make_run_state(run_context)
        stop_reason: str = StopReason.COMPLETED.value
        original_instruction = call.instructions
        original_examples = list(call.examples or [])

        # Baseline
        eval_before, baseline_trial = await self._evaluate_in_trial(
            call,
            val_set,
            runner=runner,
            trial_name="mipro_baseline",
            max_concurrent=max_concurrent,
        )
        metric_before = eval_before.score
        baseline_exhausted = self._consume_trial(shared_tracker, baseline_trial)
        if baseline_exhausted is not None:
            stop_reason = baseline_exhausted.value
        logger.debug("MiproV2: baseline %.1f%%", metric_before * 100)

        # ---- Step 1: bootstrap K demo candidate sets ----
        # Phase 17.1 C3: switched from BootstrapOptimizer.optimize()
        # (which did a wasted mini-val-set re-eval per candidate) to
        # the shared :func:`bootstrap_one_demo_set` helper. This saves
        # K * minibatch_size task-model calls per MIPROv2 lite run.
        demo_candidates: list[list[Example]] = [original_examples]  # baseline
        for k in range(self.n_demo_candidates):
            if shared_tracker is not None and shared_tracker.exhausted() is not None:
                _stop = shared_tracker.exhausted()
                if _stop is not None:
                    stop_reason = _stop.value
                break
            # Bootstrap each candidate against a different random shard
            # of the train set so the demo sets diverge.
            shard = rng.sample(train_set, k=min(len(train_set), max(8, self.minibatch_size * 2)))
            try:
                selected, bs_trial = await bootstrap_one_demo_set(
                    call=call,
                    train_slice=shard,
                    metric=self.metric,
                    runner=runner,
                    trial_name=f"mipro_bootstrap_{k}",
                    max_demos=self.max_demos,
                    score_threshold=1.0 if k % 2 == 0 else 0.5,
                    max_concurrent=max_concurrent,
                )
                self._consume_trial(shared_tracker, bs_trial)
                if selected:
                    demo_candidates.append(selected)
                    logger.debug(
                        "MiproV2: demo candidate %d collected (%d examples)",
                        k + 1,
                        len(selected),
                    )
            except Exception as exc:
                logger.warning("MiproV2: demo bootstrap %d failed: %s", k, exc)

        # ---- Step 2: propose L instruction candidates ----
        # Each instruction is grounded on a random demo candidate from
        # the bootstrap pool. The proposer LM sees the signature, the
        # current instruction, and a random sampled demo set as context.
        instruction_candidates: list[str] = [original_instruction or ""]
        proposer = Call(ProposeInstruction, model=self.proposer_model)
        for ell in range(self.n_instruction_candidates):
            if shared_tracker is not None and shared_tracker.exhausted() is not None:
                _stop = shared_tracker.exhausted()
                if _stop is not None:
                    stop_reason = _stop.value
                break
            import json as _json

            grounded_demos = rng.choice(demo_candidates) if demo_candidates else []
            failure_dicts = [
                {
                    "inputs": ex.inputs,
                    "expected": ex.outputs,
                    "score": 0.0,
                }
                for ex in grounded_demos[:3]
            ]
            sig_desc = call.signature.__doc__ or call.signature.__name__
            try:
                with runner.trial(f"mipro_propose_{ell}") as trial:
                    proposed = await proposer(
                        signature_description=sig_desc,
                        current_instruction=original_instruction or "",
                        failures_json=_json.dumps(failure_dicts, default=str),
                    )
                    instruction_candidates.append(str(proposed.proposed_instruction))
                self._consume_trial(shared_tracker, trial)
                logger.debug(
                    "MiproV2: instruction candidate %d proposed (%d chars)",
                    ell + 1,
                    len(str(proposed.proposed_instruction)),
                )
            except Exception as exc:
                logger.warning("MiproV2: instruction proposal %d failed: %s", ell, exc)

        # ---- Step 3: random search over the K x L pairs on minibatches ----
        candidates: list[_Candidate] = []
        pair_pool = [
            (i, d) for i in range(len(instruction_candidates)) for d in range(len(demo_candidates))
        ]
        rng.shuffle(pair_pool)
        sampled_pairs = pair_pool[: self.n_trials]

        minibatch = val_set[: self.minibatch_size]
        for trial_idx, (i_idx, d_idx) in enumerate(sampled_pairs):
            if shared_tracker is not None and shared_tracker.exhausted() is not None:
                _stop = shared_tracker.exhausted()
                if _stop is not None:
                    stop_reason = _stop.value
                break
            instruction = instruction_candidates[i_idx]
            demos = demo_candidates[d_idx]

            # Mutate in place + restore (deepcopy of a Call risks
            # transport client cloning failures).
            call.instructions = instruction
            call.examples = list(demos)
            try:
                eval_result, eval_trial = await self._evaluate_in_trial(
                    call,
                    minibatch,
                    runner=runner,
                    trial_name=f"mipro_minibatch_{trial_idx}",
                    max_concurrent=max_concurrent,
                )
                cand = _Candidate(
                    index=trial_idx,
                    instruction=instruction,
                    demos=demos,
                    minibatch_score=eval_result.score,
                )
                candidates.append(cand)
                self._consume_trial(shared_tracker, eval_trial)
                logger.debug(
                    "MiproV2: trial %d/%d (instr=%d, demos=%d) minibatch=%.1f%%",
                    trial_idx + 1,
                    len(sampled_pairs),
                    i_idx,
                    d_idx,
                    eval_result.score * 100,
                )
            except Exception as exc:
                logger.warning("MiproV2: trial %d failed: %s", trial_idx, exc)
            finally:
                # Restore between trials
                call.instructions = original_instruction
                call.examples = list(original_examples)

        # ---- Step 4: promote top-N to full-val eval ----
        candidates.sort(key=lambda c: c.minibatch_score, reverse=True)
        promoted = candidates[: self.n_promote]
        for cand in promoted:
            if shared_tracker is not None and shared_tracker.exhausted() is not None:
                _stop = shared_tracker.exhausted()
                if _stop is not None:
                    stop_reason = _stop.value
                break
            call.instructions = cand.instruction
            call.examples = list(cand.demos)
            try:
                full_eval, full_trial = await self._evaluate_in_trial(
                    call,
                    val_set,
                    runner=runner,
                    trial_name=f"mipro_promote_{cand.index}",
                    max_concurrent=max_concurrent,
                )
                cand.full_score = full_eval.score
                cand.promoted = True
                self._consume_trial(shared_tracker, full_trial)
                logger.debug(
                    "MiproV2: promotion candidate %d full=%.1f%% (mini=%.1f%%)",
                    cand.index,
                    full_eval.score * 100,
                    cand.minibatch_score * 100,
                )
            except Exception as exc:
                logger.warning("MiproV2: promotion %d failed: %s", cand.index, exc)
            finally:
                call.instructions = original_instruction
                call.examples = list(original_examples)

        # ---- Step 5: pick the winner ----
        promoted_with_scores = [c for c in promoted if c.full_score is not None]
        if promoted_with_scores:
            best = max(promoted_with_scores, key=lambda c: c.full_score or 0.0)
            metric_after = best.full_score or metric_before
        else:
            metric_after = metric_before
            best = None

        accepted = best is not None and metric_after > metric_before
        if accepted and best is not None:
            call.instructions = best.instruction
            call.examples = list(best.demos)
            logger.debug(
                "MiproV2: ACCEPTED — %.1f%% → %.1f%% (instr_idx=%d, demos=%d)",
                metric_before * 100,
                metric_after * 100,
                best.index,
                len(best.demos),
            )
        else:
            call.instructions = original_instruction
            call.examples = list(original_examples)
            logger.debug(
                "MiproV2: REJECTED — best %.1f%% did not improve baseline %.1f%%",
                metric_after * 100,
                metric_before * 100,
            )

        return MiproLiteResult(
            metric_before=metric_before,
            metric_after=metric_after,
            best_instruction=best.instruction if best else (original_instruction or ""),
            best_demos_count=len(best.demos) if best else len(original_examples),
            n_instruction_candidates=len(instruction_candidates),
            n_demo_candidates=len(demo_candidates),
            n_trials=len(candidates),
            n_promoted=len(promoted_with_scores),
            accepted=accepted,
            stop_reason=stop_reason,
            candidates=candidates,
        )
