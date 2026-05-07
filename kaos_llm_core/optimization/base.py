"""OptimizerBase / MetaOptimizerBase — shared scaffolding for trial-loop optimizers.

Phase 13A: every existing optimizer in the package implements its own
``optimize()`` method that:

1. Builds a :class:`BudgetTracker` from the (optional) :class:`Budget`.
2. Runs a baseline ``evaluate()`` and records the score.
3. Loops over candidate mutations (codecs / models / instructions /
   examples / hyperparameters / proposer outputs).
4. For each mutation: applies it, runs ``evaluate()``, calls
   ``estimate_eval_cost()`` to feed the tracker, accepts/rejects, and
   records a :class:`Mutation` in the :class:`MutationLog`.
5. On accepted mutations, persists the change on the underlying Call.

The hand-rolled implementations diverged on the small details — some
optimizers consumed the baseline cost into the tracker, some did not;
some optimizers stamped the trial duration on the Mutation record, some
did not; the Reflective optimizer had to manually accumulate proposer
costs because the framework had no place for cost from a sub-Call that
was not the trainee.

``OptimizerBase`` is the shared scaffolding. Subclasses implement two
methods:

- :meth:`OptimizerBase._propose` — yield candidate mutations.
- :meth:`OptimizerBase._apply_and_evaluate` — apply one mutation,
  run the trainee evaluation under :meth:`TrialRunner.trial`, return
  the new score.

The base owns:

- Baseline evaluation under its own trial scope so the baseline cost
  is charged to a named trial just like every other trial.
- Budget tracker construction and the early-exit checks after each
  trial.
- :class:`MutationLog` integration with the :class:`RunContext`
  threading optimizer-internal trial counters.
- Stop-reason classification (``COMPLETED`` / budget cap codes /
  ``THRESHOLD_NOT_MET`` for the recipe layer).

Optimizers that have a meta-shape (composing other optimizers, e.g.
:class:`CoOptimizer`) inherit from :class:`MetaOptimizerBase`, which
overrides ``optimize()`` to delegate to children but still owns the
shared trial-scope semantics.

This module is foundational for Phase 13B-H. The tests for this phase
exercise the runner + base in isolation; the seven optimizer migrations
follow in 13B/C/D/E/F/G/H.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from kaos_core.logging import get_logger

from kaos_llm_core.optimization.budget import Budget, BudgetTracker, StopReason
from kaos_llm_core.optimization.evaluation import EvalResult, MetricFn, evaluate
from kaos_llm_core.optimization.mutations import MutationLog, RunContext
from kaos_llm_core.optimization.trial_runner import Trial, TrialRunner
from kaos_llm_core.programs.call import Call
from kaos_llm_core.types import Example

logger = get_logger(__name__)

__all__ = [
    "MetaOptimizerBase",
    "OptimizerBase",
    "resolve_config",
]


def resolve_config[C](
    config_cls: type[C],
    config: C | None,
    overrides: dict[str, Any],
) -> C:
    """Materialize a per-optimizer Config dataclass from explicit + override args.

    Phase 16.5 helper used by every optimizer's ``__init__`` to support
    three construction patterns simultaneously:

      1. ``Optimizer(metric, config=Cfg(field=value))`` — typed config object
      2. ``Optimizer(metric, field=value)`` — backwards-compatible kwargs
      3. ``Optimizer(metric, config=cfg, field=override)`` — config + overrides

    Pattern (2) is the pre-Phase-16.5 surface. Pattern (1) is the new
    typed surface that prevents constructor sprawl. Pattern (3) is the
    "tweak one field on a shared base config" ergonomic.

    Raises ``TypeError`` if any override key is not a valid field of
    ``config_cls`` so typos surface immediately rather than silently
    becoming no-ops.
    """
    if overrides:
        valid = {f.name for f in dataclasses.fields(config_cls)}
        unknown = set(overrides) - valid
        if unknown:
            msg = (
                f"Unknown {config_cls.__name__} field(s): {sorted(unknown)}. "
                f"Valid fields: {sorted(valid)}."
            )
            raise TypeError(msg)
    if config is None:
        return config_cls(**overrides)
    if overrides:
        # ty: dataclasses.replace generic narrowing fails on the
        # TypeVar; the runtime contract is correct because we already
        # validated the override keys are valid fields above.
        return dataclasses.replace(config, **overrides)  # ty: ignore[invalid-argument-type]
    return config


class OptimizerBase:
    """Shared scaffolding for trial-loop optimizers.

    Subclasses inherit budget, mutation-log, and trial-scope plumbing.
    The base does not define ``optimize()`` itself — each optimizer's
    workflow is too domain-specific to abstract into a single template.
    Instead, the base offers helper methods that each ``optimize()``
    implementation calls in the right order:

    - :meth:`_make_run_state` — build the budget tracker, run context,
      and trial runner for one ``optimize()`` invocation.
    - :meth:`_evaluate_in_trial` — run :func:`evaluate` against a
      Call/Program inside a named trial scope. Returns the
      :class:`EvalResult` and the populated :class:`Trial`.
    - :meth:`_consume_trial` — fold a completed :class:`Trial` into the
      budget tracker. Returns the matching :class:`StopReason` if the
      tracker is now exhausted.

    The contract is intentionally permissive: optimizers can mix
    framework helpers with their existing logic. Phase 13B/C/D/E/F/G/H
    each migrate one optimizer onto this base; the result is that
    every optimizer charges its evaluations through the same trial
    runner, so cost attribution is uniform.
    """

    def __init__(
        self,
        *,
        metric: MetricFn,
        mutation_log: MutationLog | None = None,
        budget: Budget | None = None,
    ) -> None:
        self.metric = metric
        self.mutation_log = mutation_log or MutationLog()
        self.budget = budget
        # Phase 13H followup (audit fix): meta-optimizers inject a
        # shared :class:`BudgetTracker` here so a composite Budget cap
        # is enforced cumulatively across child stages instead of
        # silently multiplied. ``None`` means "build a fresh tracker
        # in :meth:`_make_run_state`".
        self._inherited_tracker: BudgetTracker | None = None

    def _make_run_state(
        self, run_context: RunContext | None = None
    ) -> tuple[BudgetTracker | None, RunContext, TrialRunner]:
        """Build the per-``optimize()`` runtime state.

        Returns ``(tracker, ctx, runner)``:

        - ``tracker`` is the inherited :class:`BudgetTracker` (set by a
          meta-optimizer via :attr:`_inherited_tracker`) when present,
          otherwise a fresh tracker from the configured
          :class:`Budget`, or ``None`` when no budget is set.
        - ``ctx`` is the :class:`RunContext` — caller-supplied (for
          composite optimizers) or freshly minted.
        - ``runner`` is a fresh :class:`TrialRunner`. Each
          ``optimize()`` call gets its own runner; concurrent
          ``optimize()`` invocations are isolated by the contextvar.
        """
        if self._inherited_tracker is not None:
            tracker: BudgetTracker | None = self._inherited_tracker
        else:
            tracker = self.budget.make_tracker() if self.budget else None
        ctx = run_context or RunContext()
        runner = TrialRunner()
        return tracker, ctx, runner

    async def _evaluate_in_trial(
        self,
        target: Call,
        dataset: list[Example],
        *,
        runner: TrialRunner,
        trial_name: str,
        max_concurrent: int = 5,
    ) -> tuple[EvalResult, Trial]:
        """Run :func:`evaluate` against ``target`` inside a named trial scope.

        Every Call invocation that completes inside the scope (the
        trainee, any sub-Calls in a Program, any auxiliary Calls the
        metric callable makes) charges its
        ``invocation.usage.cost_usd`` into the returned :class:`Trial`.
        Reading the trial after this method returns gives an exact
        cost / token total for the evaluation — no
        ``estimate_eval_cost`` walk required.
        """
        with runner.trial(trial_name) as trial:
            eval_result = await evaluate(
                target, dataset, self.metric, max_concurrent=max_concurrent
            )
        return eval_result, trial

    @staticmethod
    def _consume_trial(
        tracker: BudgetTracker | None,
        trial: Trial,
        *,
        trials: int = 1,
    ) -> StopReason | None:
        """Fold a completed trial into the budget tracker.

        Returns the matching :class:`StopReason` when the tracker is
        now exhausted, or ``None`` otherwise. Optimizers call this
        after every trial and bail when it returns non-None.
        """
        if tracker is None:
            return None
        tracker.consume(
            trials=trials,
            cost_usd=trial.cost_usd,
            tokens=trial.total_tokens,
        )
        return tracker.exhausted()


class MetaOptimizerBase(OptimizerBase):
    """Scaffolding for optimizers that compose other optimizers.

    The canonical example is :class:`CoOptimizer`, which runs a sequence
    of strategies (Bootstrap → Hyperparameter → Instruction → ...) on
    the same Call and merges their mutation logs into one history.

    A meta-optimizer:

    - Forwards its :class:`MutationLog` to every child optimizer so all
      child mutations land in one log with consistent identity.
    - Forwards a single :class:`RunContext` to every child so the
      ``run_id`` is shared across the composite trial sequence and the
      ``trial_id`` counter is monotonic across child boundaries.
    - Forwards its own :class:`Budget` cap (or splits it) according to
      the meta-optimizer's specific contract.

    Phase 13H migrates :class:`CoOptimizer` onto this class.
    """

    def _attach_shared_tracker(
        self,
        child: OptimizerBase,
        tracker: BudgetTracker | None,
    ) -> None:
        """Inject a shared :class:`BudgetTracker` into a child optimizer.

        Composite optimizers call this immediately before invoking
        ``child.optimize(...)`` to enforce one cumulative Budget cap
        across all child stages. Without this, every child's
        :meth:`_make_run_state` builds its own fresh tracker from
        ``self.budget``, silently multiplying the cap by the number
        of stages — which is the bug Phase 13H originally aimed to
        avoid.

        ``tracker`` may be ``None`` (no budget configured) — the
        child then falls through to its own ``self.budget`` lookup,
        matching standalone semantics.
        """
        child._inherited_tracker = tracker
