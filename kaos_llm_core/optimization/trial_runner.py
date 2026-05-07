"""TrialRunner — automatic Invocation cost attribution for optimizer trials.

Phase 13A: every existing optimizer hand-rolls trial bookkeeping.
``BootstrapOptimizer`` calls ``estimate_eval_cost(eval_result)`` after
each ``evaluate()`` to feed its :class:`BudgetTracker`. ``ReflectiveOptimizer``
goes further — it manually accumulates cost from each ``invoke()`` it
makes for the proposer/critic Calls because the optimizer's
``evaluate()`` only sees the trainee Call's cost, not the proposer's.
The Phase 9c manual fix-up patched the proposer-cost leak but left
the underlying contract fragile: every new optimizer that runs
auxiliary LLM calls outside the trainee's ``evaluate()`` has to
remember to wire its own accumulator.

The ``TrialRunner`` solves this structurally:

1. Optimizers open a trial scope with ``runner.trial(name)`` (a context
   manager).
2. Inside the scope, every successful :meth:`Call._run_pipeline`
   completion calls :func:`publish_invocation` to charge the active
   :class:`Trial` accumulator.
3. The publishing is task-isolated via a ContextVar so concurrent
   optimizer runs do not cross-contaminate. Calls running outside any
   trial scope are no-ops.

The result: an optimizer that uses :meth:`TrialRunner.trial` gets
correct cost attribution for **every** Call inside the scope —
trainee, proposer, critic, judge, sub-Programs, the lot — without
calling ``estimate_eval_cost`` or any other helper. The Phase 9c
proposer-cost bug becomes structurally impossible.

This module is foundational for Phase 13B-H (the seven optimizer
migrations onto :class:`OptimizerBase`).
"""

from __future__ import annotations

import contextvars
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kaos_llm_core.programs._invocation import Invocation

__all__ = [
    "Trial",
    "TrialRunner",
    "current_trial",
    "publish_invocation",
]


@dataclass(slots=True)
class Trial:
    """Per-trial accumulator for cost / token / invocation telemetry.

    A :class:`Trial` is created by :meth:`TrialRunner.trial` and exposed
    inside the ``with`` block. Every :class:`Invocation` that completes
    inside the block (via :func:`publish_invocation`) accumulates into
    this trial. Optimizers read the final ``cost_usd`` and ``total_tokens``
    from the trial after the scope closes — no need to walk an
    ``EvalResult`` and call ``estimate_eval_cost``.

    Attributes:
        name: Human-readable trial label (typically a strategy name + index,
            e.g. ``"bootstrap_trial_3"``). Stamped onto :class:`Mutation`
            records by the optimizer for traceability.
        cost_usd: Sum of ``invocation.usage.cost_usd`` over every Call
            invocation inside the trial scope.
        input_tokens: Sum of ``invocation.usage.input_tokens``.
        output_tokens: Sum of ``invocation.usage.output_tokens``.
        total_tokens: Sum of ``invocation.usage.total_tokens``.
        n_invocations: Count of invocations charged to this trial.
        duration_ms: Wall-clock duration of the trial scope, populated
            on context exit.
    """

    name: str
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    n_invocations: int = 0
    duration_ms: float = 0.0
    _start: float = field(default=0.0, repr=False)


_active_trial_var: contextvars.ContextVar[Trial | None] = contextvars.ContextVar(
    "kaos_llm_core_active_trial",
    default=None,
)
"""ContextVar holding the active :class:`Trial`, or ``None`` outside any scope.

Task-isolated so concurrent optimizer runs (or concurrent trials within
the same optimizer, if a future feature adds parallelism) do not
cross-contaminate. Calls running outside any trial scope see ``None``
and :func:`publish_invocation` becomes a no-op.
"""


def current_trial() -> Trial | None:
    """Return the active :class:`Trial`, or ``None`` outside any trial scope.

    Public surface for callers that want to read the running totals
    inside a trial without going through the runner. The accumulator
    is shared mutable state — read fields off it, do not assign.
    """
    return _active_trial_var.get()


def publish_invocation(invocation: Invocation) -> None:
    """Charge a completed :class:`Invocation` to the active :class:`Trial`.

    Called by :meth:`Call._run_pipeline` on the success and validation-
    exhaustion paths once the Invocation's ``usage`` slot has been
    finalized via ``apply_cost_estimates``. When no trial scope is
    active this is a no-op (the common case — most Call invocations
    happen outside any optimizer).

    The publish is **additive** — every invocation pushes its own
    deltas. Programs that fan out into multiple sub-Calls (BestOfN,
    Refine, ReAct) charge each sub-Call individually because each
    sub-Call's ``_run_pipeline`` calls this function. The Program-
    level Invocation built by ``Program.invoke`` is NOT published
    (it is a synthetic envelope, not a wire-level Call) — its
    children already accounted for the work.
    """
    trial = _active_trial_var.get()
    if trial is None:
        return
    usage = invocation.usage
    trial.cost_usd += usage.cost_usd
    trial.input_tokens += usage.input_tokens
    trial.output_tokens += usage.output_tokens
    trial.total_tokens += usage.total_tokens
    trial.n_invocations += 1


class TrialRunner:
    """Run optimizer trials with automatic cost attribution.

    Usage::

        runner = TrialRunner()
        with runner.trial("bootstrap_train") as trial:
            await evaluate(call, train_set, metric)
            await some_other_call(...)
        # trial.cost_usd now sums every Call inside the with block.

    The runner does not own the optimizer's bookkeeping (mutation log,
    budget tracker, accept/reject logic) — those concerns belong to
    :class:`OptimizerBase`. The runner only owns the contextvar
    machinery for routing :class:`Invocation` usage into the active
    :class:`Trial`.
    """

    @contextmanager
    def trial(self, name: str) -> Iterator[Trial]:
        """Open a trial scope.

        Yields a :class:`Trial` that accumulates Invocation usage from
        every Call inside the ``with`` block. On exit, the trial's
        ``duration_ms`` is populated with the wall-clock elapsed time.

        Trials do not nest — opening a trial inside another trial
        replaces the outer trial for the duration of the inner one.
        Optimizers that need nested cost accounting should sum the
        inner trial into the outer trial themselves after the inner
        scope closes.
        """
        trial = Trial(name=name, _start=time.monotonic())
        token = _active_trial_var.set(trial)
        try:
            yield trial
        finally:
            trial.duration_ms = (time.monotonic() - trial._start) * 1000
            _active_trial_var.reset(token)
