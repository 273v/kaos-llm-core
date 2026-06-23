"""MiproV2Optimizer — full DSPy-style joint (instruction, demos) search.

Phase 17.1 — the full version of the MIPROv2 algorithm. Where
:class:`MiproLiteOptimizer` (Phase 17.0) ships random search over
the cartesian product, this version ships the three pieces that
make MIPROv2 the canonical missing optimizer in the KAOS catalog:

1. **GroundedInstructionProposer** — instructions are not generated
   from a bare proposer prompt; they are grounded in a one-shot
   dataset summary, a program description, a per-module
   description, a randomly sampled bootstrapped demo set, and one
   of six hand-tuned stylistic tips. (Port of DSPy's
   ``GroundedProposer``.)
2. **Categorical TPE search** over the joint
   (instruction_idx, demos_idx) space, with cold-start uniform
   random for the first ``n_startup_trials`` and Laplace-smoothed
   Tree-structured Parzen Estimator after that. The per-dim
   factorization is what makes joint search win over the sequential
   coordinate descent CoOptimizer does.
3. **Full-eval promotion** every ``minibatch_full_eval_steps``
   trials: pick the candidate with the highest mean minibatch
   score that has not yet been fully evaluated, run it on the full
   val set, and re-inject the result back into the TPE posterior
   so the next draw is informed by the full-fidelity score.

Phase 17.1 scope restriction
============================

Single-predictor only. The DSPy multi-predictor variant requires a
teacher-trajectory bootstrap (DSPy's ``BootstrapFewShot`` runs the
student program end-to-end and captures per-predictor traces); KAOS
``BootstrapOptimizer`` filters by metric threshold rather than
capturing trajectories, which makes the multi-predictor demo
bootstrap a different problem that needs its own design. The
optimizer raises if a Program with more than one named call is
passed.

References
==========

* DSPy MIPROv2 source:
  ``stanfordnlp/dspy/main/dspy/teleprompt/mipro_optimizer_v2.py``
  (SHA ``4ece27f7494e3ae3ac4311034481f9c697edb69a``).
* Design doc: ``docs/internal/design/mipro-v2-equivalent.md``.
* Bergstra et al. 2011. *Algorithms for Hyper-Parameter
  Optimization*. NeurIPS — for the categorical TPE formulation.

DSPy-bug avoidance
==================

DSPy's ``_select_and_insert_instructions_and_demos`` (line 698 of
mipro_optimizer_v2.py at the pinned SHA) writes the *instruction*
index into the demos key of the raw params dict used by Optuna's
trial reinjection. This means the promoted trial Optuna sees has
wrong demo coordinates. KAOS does NOT replicate this bug — see
``MiproV2Optimizer._build_raw_params``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Literal

from kaos_core.logging import get_logger

from kaos_llm_core.optimization.base import MetaOptimizerBase, MetricFn, resolve_config
from kaos_llm_core.optimization.bootstrap import bootstrap_one_demo_set
from kaos_llm_core.optimization.budget import Budget, StopReason
from kaos_llm_core.optimization.evaluation import EvalResult
from kaos_llm_core.optimization.mutations import MutationLog, RunContext
from kaos_llm_core.optimization.proposers.grounded import (
    GroundedInstructionProposer,
    ProposedInstruction,
)
from kaos_llm_core.optimization.search.tpe import TPESearcher, TPESearcherConfig
from kaos_llm_core.programs.base import Program
from kaos_llm_core.programs.call import Call
from kaos_llm_core.types import Example

logger = get_logger(__name__)

__all__ = [
    "AutoMode",
    "MiproV2BestConfig",
    "MiproV2Config",
    "MiproV2Optimizer",
    "MiproV2Result",
]


# ---------------------------------------------------------------------------
# Auto-mode presets — verbatim from DSPy mipro_optimizer_v2.py:33-37
# ---------------------------------------------------------------------------


AutoMode = Literal["light", "medium", "heavy"]


@dataclass(frozen=True, slots=True)
class _AutoPreset:
    n: int
    val_size: int


_AUTO_PRESETS: dict[AutoMode, _AutoPreset] = {
    "light": _AutoPreset(n=6, val_size=100),
    "medium": _AutoPreset(n=12, val_size=300),
    "heavy": _AutoPreset(n=18, val_size=1000),
}


def _derive_num_trials(num_candidates: int, num_predictors: int, zeroshot: bool) -> int:
    """Mirror of DSPy's ``_set_num_trials_from_num_candidates``.

    The DSPy formula (mipro_optimizer_v2.py:228 at the pinned SHA) is::

        num_vars = num_predictors
        if not zeroshot: num_vars *= 2
        num_trials = int(max(2 * num_vars * log2(N), 1.5 * N))
    """
    num_vars = num_predictors
    if not zeroshot:
        num_vars *= 2
    if num_candidates <= 1:
        return max(math.ceil(1.5 * num_candidates), 2)
    return max(int(2 * num_vars * math.log2(num_candidates)), math.ceil(1.5 * num_candidates))


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MiproV2BestConfig:
    """The (instruction, demos) bundle MIPROv2 selected as best."""

    instruction: str
    demos: list[Example]


@dataclass(frozen=True, slots=True)
class MiproV2Result:
    """Outcome of a :class:`MiproV2Optimizer` run.

    Attributes
    ----------
    metric_before:
        Baseline full-eval score before optimization.
    metric_after:
        Best full-eval score observed during the search. Equal to
        ``metric_before`` if no candidate ever beat the baseline.
    best_config:
        The (instruction, demos) bundle that achieved
        ``metric_after``. Always populated even on rejection — the
        caller can apply it manually if needed.
    eval_before:
        The :class:`EvalResult` from the baseline evaluation.
    eval_after:
        The :class:`EvalResult` from the final accepted candidate.
        Equals ``eval_before`` on rejection.
    trials_run:
        Number of TPE search trials that completed (excludes the
        baseline + dataset-summary + proposer setup phases).
    proposer_calls:
        Number of grounded proposer LM invocations
        (DescribeDataset + DescribeProgram + DescribeModule + N
        instruction draws).
    accepted:
        ``True`` iff ``metric_after > metric_before`` strictly.
    stop_reason:
        One of :class:`StopReason` values, indicating why the
        search stopped (``completed``, ``budget_cost``,
        ``budget_trials``, ``budget_wall_seconds``, ``budget_tokens``).
    """

    metric_before: float
    metric_after: float
    best_config: MiproV2BestConfig
    eval_before: EvalResult
    eval_after: EvalResult
    trials_run: int
    proposer_calls: int
    accepted: bool
    stop_reason: str = StopReason.COMPLETED.value


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MiproV2Config:
    """Typed configuration for :class:`MiproV2Optimizer`.

    Not frozen because nothing here is mutated, but ``frozen=True``
    blocks mirror-on-instance usage in the constructor and the rest
    of the optimizer catalog uses unfrozen mirrors. Construct via
    either ``MiproV2Optimizer(metric, config=MiproV2Config(...))`` or
    the kwargs shim ``MiproV2Optimizer(metric, num_candidates=6, ...)``.

    See the design doc §4 for the rationale behind every default.
    """

    proposer_model: str = "anthropic:claude-sonnet-4-6"
    auto: AutoMode | None = "light"
    num_candidates: int | None = None
    num_trials: int | None = None
    max_bootstrapped_demos: int = 4
    max_labeled_demos: int = 4
    minibatch: bool = True
    minibatch_size: int = 35
    minibatch_full_eval_steps: int = 5
    metric_threshold: float | None = None
    seed: int = 9
    init_temperature: float = 1.0
    program_aware_proposer: bool = True
    data_aware_proposer: bool = True
    tip_aware_proposer: bool = True
    fewshot_aware_proposer: bool = True
    view_data_batch_size: int = 10

    def __post_init__(self) -> None:
        if self.auto is not None and self.auto not in _AUTO_PRESETS:
            msg = f"auto must be one of {sorted(_AUTO_PRESETS)} or None; got {self.auto!r}"
            raise ValueError(msg)
        if self.num_candidates is not None and self.num_candidates < 2:
            msg = "num_candidates must be >= 2 when set"
            raise ValueError(msg)
        if self.num_trials is not None and self.num_trials < 1:
            msg = "num_trials must be >= 1 when set"
            raise ValueError(msg)
        if self.max_bootstrapped_demos < 0:
            msg = "max_bootstrapped_demos must be >= 0"
            raise ValueError(msg)
        if self.minibatch_size < 1:
            msg = "minibatch_size must be >= 1"
            raise ValueError(msg)
        if self.minibatch_full_eval_steps < 1:
            msg = "minibatch_full_eval_steps must be >= 1"
            raise ValueError(msg)
        if self.view_data_batch_size < 1:
            msg = "view_data_batch_size must be >= 1"
            raise ValueError(msg)


# ---------------------------------------------------------------------------
# Internal: per-trial bookkeeping
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _TrialRecord:
    trial_index: int
    params: dict[str, int]
    score: float
    full_eval: bool


# ---------------------------------------------------------------------------
# The optimizer
# ---------------------------------------------------------------------------


class MiproV2Optimizer(MetaOptimizerBase):
    """KAOS-native port of DSPy's MIPROv2.

    Composes :class:`GroundedInstructionProposer` and
    :class:`TPESearcher` under one shared
    :class:`RunContext` / :class:`MutationLog` / :class:`BudgetTracker`.

    Pipeline (see design doc §4):

    1. **Baseline** — full-eval the call as-is to anchor
       ``metric_before``.
    2. **Bootstrap** — call :func:`bootstrap_one_demo_set` N times
       over shuffled training slices to build N candidate demo sets.
       Index 0 is reserved for the baseline (the call's existing
       ``examples``).
    3. **Propose** — build the grounded proposer's static context
       (dataset summary + program description + module description),
       then draw N instruction candidates per Call. Index 0 is the
       baseline (the call's existing ``instructions``).
    4. **TPE search** — sample (instr_idx, demos_idx) per trial via
       :class:`TPESearcher`, evaluate on a fresh minibatch of the
       val set, observe the score in the TPE posterior. Every
       ``minibatch_full_eval_steps`` trials, promote the candidate
       with the highest mean minibatch score (that hasn't been
       promoted yet) to a full-val evaluation, observe the
       full-eval score in the TPE posterior, and update
       ``metric_after`` if it strictly exceeds the previous best.
    5. **Apply** — if the best full-eval score strictly exceeds
       ``metric_before``, mutate the original call in place (set
       ``call.instructions`` and ``call.examples`` to the best
       config). Otherwise restore the original.

    Concurrency
    -----------

    Phase 17.1 forces ``max_concurrent=1`` at the trial level
    because the optimizer mutates ``call.instructions`` /
    ``call.examples`` in place. Within-trial evaluation concurrency
    is the caller's value; only the trial loop is sequential.

    Args
    ----

    metric:
        Scoring function: ``(prediction, gold_outputs) -> float``
        in [0, 1].
    config:
        Optional :class:`MiproV2Config`. Defaults match DSPy
        ``auto="light"``.
    mutation_log:
        Optional shared mutation log.
    budget:
        Optional cumulative budget cap. Enforced across the
        proposer setup, demo bootstraps, and search trials via
        the shared :class:`BudgetTracker`.
    **overrides:
        Backwards-compatible kwargs shim — any
        :class:`MiproV2Config` field can be passed directly.
    """

    def __init__(
        self,
        metric: MetricFn,
        *,
        config: MiproV2Config | None = None,
        mutation_log: MutationLog | None = None,
        budget: Budget | None = None,
        **overrides: Any,
    ) -> None:
        super().__init__(metric=metric, mutation_log=mutation_log, budget=budget)
        cfg = resolve_config(MiproV2Config, config, overrides)
        self.config = cfg
        # Mirror config fields onto the instance.
        self.proposer_model = cfg.proposer_model
        self.auto = cfg.auto
        self.num_candidates = cfg.num_candidates
        self.num_trials = cfg.num_trials
        self.max_bootstrapped_demos = cfg.max_bootstrapped_demos
        self.max_labeled_demos = cfg.max_labeled_demos
        self.minibatch = cfg.minibatch
        self.minibatch_size = cfg.minibatch_size
        self.minibatch_full_eval_steps = cfg.minibatch_full_eval_steps
        self.metric_threshold = cfg.metric_threshold
        self.seed = cfg.seed
        self.init_temperature = cfg.init_temperature
        self.program_aware_proposer = cfg.program_aware_proposer
        self.data_aware_proposer = cfg.data_aware_proposer
        self.tip_aware_proposer = cfg.tip_aware_proposer
        self.fewshot_aware_proposer = cfg.fewshot_aware_proposer
        self.view_data_batch_size = cfg.view_data_batch_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def optimize(
        self,
        target: Call | Program,
        train_set: list[Example],
        val_set: list[Example] | None = None,
        *,
        max_concurrent: int = 5,
        run_context: RunContext | None = None,
    ) -> MiproV2Result:
        """Run the MIPROv2 pipeline.

        Phase 17.1 supports only single-predictor targets (a bare
        :class:`Call` or a :class:`Program` with exactly one named
        sub-Call). Multi-predictor support is a Phase 17.2 follow-up.
        """
        call = self._unwrap_target(target)
        self._validate_inputs(call, train_set, val_set)
        assert val_set is not None  # narrowed by _validate_inputs

        if self.proposer_model == self._target_model(call):
            logger.warning(
                "MIPROv2: proposer_model == target model (%s). The proposer "
                "cannot correct biases it shares with the target. Consider "
                "passing a stronger proposer_model.",
                self.proposer_model,
            )

        # Resolve auto preset (or honor explicit num_candidates / num_trials).
        num_candidates, _val_size = self._resolve_auto()
        zeroshot = self.max_bootstrapped_demos == 0 and self.max_labeled_demos == 0
        num_trials = self.num_trials or _derive_num_trials(
            num_candidates=num_candidates, num_predictors=1, zeroshot=zeroshot
        )

        rng = random.Random(self.seed)
        shared_tracker, ctx, runner = self._make_run_state(run_context)
        stop_reason: str = StopReason.COMPLETED.value

        original_instruction = call.instructions or ""
        original_examples = list(call.examples or [])

        proposer_call_count = 0

        # ------------------------------------------------------------------
        # Phase 1: baseline full-eval
        # ------------------------------------------------------------------

        eval_before, baseline_trial = await self._evaluate_in_trial(
            call,
            val_set,
            runner=runner,
            trial_name="mipro_v2_baseline",
            max_concurrent=max_concurrent,
        )
        metric_before = eval_before.score
        baseline_exhausted = self._consume_trial(shared_tracker, baseline_trial)
        if baseline_exhausted is not None:
            stop_reason = baseline_exhausted.value
            return self._abort(
                metric_before=metric_before,
                eval_before=eval_before,
                original_instruction=original_instruction,
                original_examples=original_examples,
                trials_run=0,
                proposer_calls=proposer_call_count,
                stop_reason=stop_reason,
            )
        logger.debug("MIPROv2: baseline %.1f%%", metric_before * 100)

        # ------------------------------------------------------------------
        # Phase 2: bootstrap N demo candidate sets (or skip if zero-shot)
        # ------------------------------------------------------------------

        demo_candidates: list[list[Example]] = [original_examples]
        if not zeroshot:
            for k in range(num_candidates - 1):  # baseline already at index 0
                if shared_tracker is not None and shared_tracker.exhausted() is not None:
                    _stop = shared_tracker.exhausted()
                    if _stop is not None:
                        stop_reason = _stop.value
                    break
                shard_size = min(len(train_set), max(8, self.minibatch_size))
                shard = rng.sample(train_set, k=shard_size)
                try:
                    selected, bs_trial = await bootstrap_one_demo_set(
                        call=call,
                        train_slice=shard,
                        metric=self.metric,
                        runner=runner,
                        trial_name=f"mipro_v2_bootstrap_{k}",
                        max_demos=self.max_bootstrapped_demos,
                        score_threshold=self.metric_threshold or 1.0,
                        max_concurrent=max_concurrent,
                    )
                    consumed = self._consume_trial(shared_tracker, bs_trial)
                    self._record_bootstrap_mutation(
                        ctx=ctx,
                        call=call,
                        set_index=k + 1,
                        size=len(selected),
                        cost_usd=bs_trial.cost_usd,
                        tokens_used=bs_trial.total_tokens,
                        duration_ms=bs_trial.duration_ms,
                    )
                    if selected:
                        demo_candidates.append(selected)
                    if consumed is not None:
                        stop_reason = consumed.value
                        break
                except Exception as exc:
                    logger.warning("MIPROv2: demo bootstrap %d failed: %s", k, exc)

        # ------------------------------------------------------------------
        # Phase 3: build the grounded proposer's static context + draw N
        # instruction candidates
        # ------------------------------------------------------------------

        proposer = GroundedInstructionProposer(
            proposer_model=self.proposer_model,
            program_aware=self.program_aware_proposer,
            data_aware=self.data_aware_proposer,
            tip_aware=self.tip_aware_proposer,
            fewshot_aware=self.fewshot_aware_proposer,
            init_temperature=self.init_temperature,
            view_data_batch_size=self.view_data_batch_size,
            seed=self.seed,
        )

        try:
            with runner.trial("mipro_v2_summarize_dataset") as summary_trial:
                await proposer.build_static_context(call=call, train_set=train_set)
            consumed = self._consume_trial(shared_tracker, summary_trial)
            # Count: 1 (DescribeDataset) + 1 (DescribeProgram) + 1 (DescribeModule)
            # if all toggles are on, otherwise fewer.
            proposer_call_count += sum(
                [
                    int(self.data_aware_proposer),
                    int(self.program_aware_proposer),  # DescribeProgram
                    int(self.program_aware_proposer),  # DescribeModule
                ]
            )
            if consumed is not None:
                stop_reason = consumed.value
        except Exception as exc:
            logger.warning("MIPROv2: dataset summarizer failed: %s", exc)
            return self._abort(
                metric_before=metric_before,
                eval_before=eval_before,
                original_instruction=original_instruction,
                original_examples=original_examples,
                trials_run=0,
                proposer_calls=proposer_call_count,
                stop_reason=stop_reason,
            )

        instruction_candidates: list[str] = [original_instruction]
        if shared_tracker is None or shared_tracker.exhausted() is None:
            try:
                with runner.trial(f"mipro_v2_propose_{call.signature.__name__}") as propose_trial:
                    proposed: list[
                        ProposedInstruction
                    ] = await proposer.propose_n_instructions_for_call(
                        call=call,
                        n=num_candidates - 1,  # baseline already at index 0
                        demo_candidates=demo_candidates if len(demo_candidates) > 1 else None,
                    )
                consumed = self._consume_trial(shared_tracker, propose_trial)
                proposer_call_count += len(proposed)
                for i, p in enumerate(proposed, start=1):
                    instruction_candidates.append(p.instruction)
                    self._record_propose_mutation(
                        ctx=ctx,
                        call=call,
                        index=i,
                        proposed=p,
                        cost_usd=propose_trial.cost_usd / max(len(proposed), 1),
                        tokens_used=propose_trial.total_tokens // max(len(proposed), 1),
                        duration_ms=propose_trial.duration_ms / max(len(proposed), 1),
                    )
                if consumed is not None:
                    stop_reason = consumed.value
            except Exception as exc:
                logger.warning("MIPROv2: instruction proposal failed: %s", exc)

        # ------------------------------------------------------------------
        # Phase 4: TPE search over (instr_idx, demos_idx)
        # ------------------------------------------------------------------

        n_instr = len(instruction_candidates)
        n_demos = len(demo_candidates)
        space: dict[str, int] = {"instr_idx": n_instr}
        if not zeroshot and n_demos > 1:
            space["demos_idx"] = n_demos

        tpe = TPESearcher(
            space=space,
            num_trials=num_trials,
            config=TPESearcherConfig(),
            seed=self.seed,
        )
        # Pre-seed the baseline (instr=0, demos=0) with its known
        # full-eval score so TPE has at least one observation to
        # anchor its posterior on. Matches DSPy mipro_v2.py:578-588.
        baseline_params = {"instr_idx": 0}
        if "demos_idx" in space:
            baseline_params["demos_idx"] = 0
        tpe.observe(baseline_params, score=metric_before, full_eval=True)

        records: list[_TrialRecord] = []
        # Track per-categorical-key mean scores so we can promote the
        # best-mean candidate to full eval (DSPy's
        # ``param_score_dict`` semantics).
        param_score_dict: dict[tuple[int, int], list[float]] = {}
        promoted_keys: set[tuple[int, int]] = set()
        best_full_score = metric_before
        best_eval_after = eval_before
        best_config = MiproV2BestConfig(
            instruction=original_instruction, demos=list(original_examples)
        )

        rng_minibatch = random.Random(self.seed * 17 + 3)

        for trial_idx in range(num_trials):
            if shared_tracker is not None and shared_tracker.exhausted() is not None:
                _stop = shared_tracker.exhausted()
                if _stop is not None:
                    stop_reason = _stop.value
                break

            point = tpe.suggest()
            instr_idx = point["instr_idx"]
            demos_idx = point.get("demos_idx", 0)
            instruction = instruction_candidates[instr_idx]
            demos = demo_candidates[demos_idx]

            # In-place mutate + try/finally restore (per design doc §7.2
            # and the mipro_lite template).
            try:
                call.instructions = instruction
                call.examples = list(demos)
                if self.minibatch:
                    minibatch_set = rng_minibatch.sample(
                        val_set, k=min(self.minibatch_size, len(val_set))
                    )
                else:
                    minibatch_set = val_set
                eval_result, eval_trial = await self._evaluate_in_trial(
                    call,
                    minibatch_set,
                    runner=runner,
                    trial_name=f"mipro_v2_minibatch_{trial_idx}",
                    max_concurrent=max_concurrent,
                )
                trial_score = eval_result.score
                tpe.observe(point, score=trial_score, full_eval=False)
                records.append(
                    _TrialRecord(
                        trial_index=trial_idx,
                        params=dict(point),
                        score=trial_score,
                        full_eval=False,
                    )
                )
                key = (instr_idx, demos_idx)
                param_score_dict.setdefault(key, []).append(trial_score)
                self._record_search_trial_mutation(
                    ctx=ctx,
                    call=call,
                    trial_index=trial_idx,
                    params=point,
                    score=trial_score,
                    minibatch_size=len(minibatch_set),
                    accepted=False,  # minibatch trials are never "accepted" — only promotions are
                    cost_usd=eval_trial.cost_usd,
                    tokens_used=eval_trial.total_tokens,
                    duration_ms=eval_trial.duration_ms,
                )
                consumed = self._consume_trial(shared_tracker, eval_trial)
                if consumed is not None:
                    stop_reason = consumed.value
                    break
                logger.debug(
                    "MIPROv2: trial %d/%d (instr=%d, demos=%d) minibatch=%.1f%%",
                    trial_idx + 1,
                    num_trials,
                    instr_idx,
                    demos_idx,
                    trial_score * 100,
                )

                # Promotion path: every (minibatch_full_eval_steps + 1)
                # trials, OR on the very last trial.
                is_promote_step = self.minibatch and (
                    (trial_idx + 1) % (self.minibatch_full_eval_steps + 1) == 0
                    or trial_idx == num_trials - 1
                )
                if is_promote_step:
                    promote_result = await self._promote_best_mean(
                        runner=runner,
                        call=call,
                        val_set=val_set,
                        param_score_dict=param_score_dict,
                        promoted_keys=promoted_keys,
                        instruction_candidates=instruction_candidates,
                        demo_candidates=demo_candidates,
                        max_concurrent=max_concurrent,
                        trial_index=trial_idx,
                    )
                    if promote_result is not None:
                        promote_score, promote_eval, promote_key, promote_trial = promote_result
                        # Re-inject into TPE posterior with full_eval=True
                        promoted_params = {"instr_idx": promote_key[0]}
                        if "demos_idx" in space:
                            promoted_params["demos_idx"] = promote_key[1]
                        tpe.observe(promoted_params, score=promote_score, full_eval=True)
                        records.append(
                            _TrialRecord(
                                trial_index=trial_idx,
                                params=promoted_params,
                                score=promote_score,
                                full_eval=True,
                            )
                        )
                        self._record_promote_mutation(
                            ctx=ctx,
                            call=call,
                            trial_index=trial_idx,
                            params=promoted_params,
                            promoted_from_mean=sum(param_score_dict[promote_key])
                            / len(param_score_dict[promote_key]),
                            full_eval_score=promote_score,
                            val_size=len(val_set),
                            cost_usd=promote_trial.cost_usd,
                            tokens_used=promote_trial.total_tokens,
                            duration_ms=promote_trial.duration_ms,
                        )
                        consumed = self._consume_trial(shared_tracker, promote_trial)
                        if promote_score > best_full_score:
                            best_full_score = promote_score
                            best_eval_after = promote_eval
                            best_config = MiproV2BestConfig(
                                instruction=instruction_candidates[promote_key[0]],
                                demos=list(demo_candidates[promote_key[1]]),
                            )
                        if consumed is not None:
                            stop_reason = consumed.value
                            break
            finally:
                # Always restore the original between trials so the
                # next iteration starts from a clean slate.
                call.instructions = original_instruction
                call.examples = list(original_examples)

        # ------------------------------------------------------------------
        # Phase 5: apply the best config (or restore the original)
        # ------------------------------------------------------------------

        accepted = best_full_score > metric_before
        if accepted:
            call.instructions = best_config.instruction
            call.examples = list(best_config.demos)
            metric_after = best_full_score
            logger.debug(
                "MIPROv2: ACCEPTED %.1f%% -> %.1f%%",
                metric_before * 100,
                metric_after * 100,
            )
        else:
            call.instructions = original_instruction
            call.examples = list(original_examples)
            metric_after = metric_before
            best_eval_after = eval_before
            logger.debug(
                "MIPROv2: REJECTED — best %.1f%% did not improve baseline %.1f%%",
                best_full_score * 100,
                metric_before * 100,
            )

        self._record_apply_mutation(
            ctx=ctx,
            call=call,
            metric_before=metric_before,
            metric_after=metric_after,
            best_config=best_config,
            original_instruction=original_instruction,
            original_examples=original_examples,
            accepted=accepted,
        )

        return MiproV2Result(
            metric_before=metric_before,
            metric_after=metric_after,
            best_config=best_config,
            eval_before=eval_before,
            eval_after=best_eval_after,
            trials_run=len([r for r in records if not r.full_eval]),
            proposer_calls=proposer_call_count,
            accepted=accepted,
            stop_reason=stop_reason,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _unwrap_target(self, target: Call | Program) -> Call:
        """Extract the single Call from the target.

        Phase 17.1 only supports single-predictor targets. Raises a
        clear error if the caller passes a multi-predictor Program.
        """
        if isinstance(target, Call):
            return target
        if isinstance(target, Program):
            children = target.named_calls()
            calls_only = [
                (name, child) for name, child in children.items() if isinstance(child, Call)
            ]
            if len(calls_only) != 1:
                msg = (
                    f"MiproV2Optimizer Phase 17.1 supports only single-predictor "
                    f"targets; got a Program with {len(calls_only)} sub-calls. "
                    f"Use CoOptimizer for sequential coordinate descent or wait "
                    f"for Phase 17.2 multi-predictor support."
                )
                raise ValueError(msg)
            return calls_only[0][1]
        msg = f"target must be a Call or Program; got {type(target).__name__}"
        raise TypeError(msg)

    def _validate_inputs(
        self,
        call: Call,
        train_set: list[Example],
        val_set: list[Example] | None,
    ) -> None:
        if val_set is None:
            msg = (
                "MiproV2Optimizer requires an explicit val_set. Auto-splitting "
                "the train_set hides a correctness footgun on small datasets."
            )
            raise ValueError(msg)
        if not val_set:
            msg = "MiproV2Optimizer requires a non-empty val_set"
            raise ValueError(msg)
        if len(train_set) < 4:
            msg = (
                f"MiproV2Optimizer requires at least 4 train_set examples for "
                f"meaningful demo bootstrap; got {len(train_set)}"
            )
            raise ValueError(msg)
        _ = call  # Reserved for future per-call validation.

    def _resolve_auto(self) -> tuple[int, int]:
        """Return ``(num_candidates, val_size)`` honoring the auto preset."""
        if self.auto is not None and self.num_candidates is None:
            preset = _AUTO_PRESETS[self.auto]
            return preset.n, preset.val_size
        n = self.num_candidates if self.num_candidates is not None else 6
        return n, 100

    def _target_model(self, call: Call) -> str:
        return getattr(call, "_model", "") or ""

    def _abort(
        self,
        *,
        metric_before: float,
        eval_before: EvalResult,
        original_instruction: str,
        original_examples: list[Example],
        trials_run: int,
        proposer_calls: int,
        stop_reason: str,
    ) -> MiproV2Result:
        """Return a non-accepting result on early termination."""
        return MiproV2Result(
            metric_before=metric_before,
            metric_after=metric_before,
            best_config=MiproV2BestConfig(
                instruction=original_instruction, demos=list(original_examples)
            ),
            eval_before=eval_before,
            eval_after=eval_before,
            trials_run=trials_run,
            proposer_calls=proposer_calls,
            accepted=False,
            stop_reason=stop_reason,
        )

    async def _promote_best_mean(
        self,
        *,
        runner: Any,
        call: Call,
        val_set: list[Example],
        param_score_dict: dict[tuple[int, int], list[float]],
        promoted_keys: set[tuple[int, int]],
        instruction_candidates: list[str],
        demo_candidates: list[list[Example]],
        max_concurrent: int,
        trial_index: int,
    ) -> tuple[float, EvalResult, tuple[int, int], Any] | None:
        """Pick the highest-mean unpromoted candidate, full-eval it,
        return ``(score, eval_result, key, trial)`` or ``None`` if no
        eligible candidate exists.
        """
        eligible = [
            (key, scores) for key, scores in param_score_dict.items() if key not in promoted_keys
        ]
        if not eligible:
            return None
        # Pick the categorical key with the highest mean score.
        eligible.sort(key=lambda kv: sum(kv[1]) / len(kv[1]), reverse=True)
        best_key, _scores = eligible[0]
        promoted_keys.add(best_key)

        # Apply the candidate.
        instr_idx, demos_idx = best_key
        call.instructions = instruction_candidates[instr_idx]
        call.examples = list(demo_candidates[demos_idx])

        try:
            full_eval, full_trial = await self._evaluate_in_trial(
                call,
                val_set,
                runner=runner,
                trial_name=f"mipro_v2_promote_{trial_index}",
                max_concurrent=max_concurrent,
            )
            return full_eval.score, full_eval, best_key, full_trial
        except Exception as exc:
            logger.warning("MIPROv2: promotion at trial %d failed: %s", trial_index, exc)
            return None

    # ------------------------------------------------------------------
    # Mutation log helpers
    # ------------------------------------------------------------------

    def _record_bootstrap_mutation(
        self,
        *,
        ctx: RunContext,
        call: Call,
        set_index: int,
        size: int,
        cost_usd: float,
        tokens_used: int,
        duration_ms: float,
    ) -> None:
        mutation = ctx.make_mutation(
            strategy="mipro_v2",
            mutation_type="bootstrap_demo_set",
            call_name=call.signature.__name__,
            before={"set_index": set_index, "size": 0},
            after={"size": size, "source": "student"},
            rationale=f"Bootstrap candidate set {set_index} ({size} demos).",
            metric_before=0.0,
            metric_after=0.0,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            duration_ms=duration_ms,
            accepted=size > 0,
        )
        self.mutation_log.record(mutation)

    def _record_propose_mutation(
        self,
        *,
        ctx: RunContext,
        call: Call,
        index: int,
        proposed: ProposedInstruction,
        cost_usd: float,
        tokens_used: int,
        duration_ms: float,
    ) -> None:
        mutation = ctx.make_mutation(
            strategy="mipro_v2",
            mutation_type="propose_instruction",
            call_name=call.signature.__name__,
            before={"index": index},
            after={
                "instruction": proposed.instruction,
                "tip": proposed.tip,
                "verbatim_copy": proposed.verbatim_copy,
            },
            rationale=proposed.rationale or f"Tip: {proposed.tip}",
            metric_before=0.0,
            metric_after=0.0,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            duration_ms=duration_ms,
            accepted=not proposed.verbatim_copy,
        )
        self.mutation_log.record(mutation)

    def _record_search_trial_mutation(
        self,
        *,
        ctx: RunContext,
        call: Call,
        trial_index: int,
        params: dict[str, int],
        score: float,
        minibatch_size: int,
        accepted: bool,
        cost_usd: float,
        tokens_used: int,
        duration_ms: float,
    ) -> None:
        mutation = ctx.make_mutation(
            strategy="mipro_v2",
            mutation_type="search_trial",
            call_name=call.signature.__name__,
            before={"trial_index": trial_index, "config": dict(params)},
            after={
                "score": score,
                "full_eval": False,
                "minibatch_size": minibatch_size,
            },
            rationale=f"TPE trial {trial_index} on minibatch (size={minibatch_size}).",
            metric_before=0.0,
            metric_after=score,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            duration_ms=duration_ms,
            accepted=accepted,
        )
        self.mutation_log.record(mutation)

    def _record_promote_mutation(
        self,
        *,
        ctx: RunContext,
        call: Call,
        trial_index: int,
        params: dict[str, int],
        promoted_from_mean: float,
        full_eval_score: float,
        val_size: int,
        cost_usd: float,
        tokens_used: int,
        duration_ms: float,
    ) -> None:
        mutation = ctx.make_mutation(
            strategy="mipro_v2",
            mutation_type="search_full_eval",
            call_name=call.signature.__name__,
            before={"trial_index": trial_index, "promoted_from_mean": promoted_from_mean},
            after={
                "config": dict(params),
                "full_eval_score": full_eval_score,
                "val_size": val_size,
            },
            rationale=(
                f"Promoted candidate at trial {trial_index} (mean minibatch "
                f"{promoted_from_mean:.3f}) to full-val evaluation."
            ),
            metric_before=promoted_from_mean,
            metric_after=full_eval_score,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            duration_ms=duration_ms,
            accepted=full_eval_score > promoted_from_mean,
        )
        self.mutation_log.record(mutation)

    def _record_apply_mutation(
        self,
        *,
        ctx: RunContext,
        call: Call,
        metric_before: float,
        metric_after: float,
        best_config: MiproV2BestConfig,
        original_instruction: str,
        original_examples: list[Example],
        accepted: bool,
    ) -> None:
        mutation = ctx.make_mutation(
            strategy="mipro_v2",
            mutation_type="apply_best_config",
            call_name=call.signature.__name__,
            before={
                "instructions": original_instruction,
                "n_examples": len(original_examples),
            },
            after={
                "instructions": best_config.instruction,
                "n_examples": len(best_config.demos),
            },
            rationale=(
                f"Applied best (instruction, demos) bundle "
                f"({metric_before:.3f} -> {metric_after:.3f})."
                if accepted
                else f"Rejected — best did not improve baseline ({metric_before:.3f})."
            ),
            metric_before=metric_before,
            metric_after=metric_after,
            accepted=accepted,
        )
        self.mutation_log.record(mutation)


# Suppress unused-import warnings on the field import (kept for
# future use when MiproV2Result needs default-factory fields).
_ = field
