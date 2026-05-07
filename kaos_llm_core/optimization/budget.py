"""Budget — trial/time/cost/token caps for optimization loops.

Every optimizer in Phase 6 (and every existing optimizer) accepts an optional
``Budget`` object. When set, the optimizer instantiates a :class:`BudgetTracker`
and consumes from it after each trial. The tracker surfaces the first cap that
has been reached via :meth:`BudgetTracker.exhausted`, and the optimizer halts
with a matching :class:`StopReason`.

``Budget`` is greenfield — there is no ``budget_tokens`` parameter to migrate.
The parameter is optional everywhere so existing behavior is preserved when a
caller does not opt in.

Example::

    budget = Budget(max_trials=10, max_cost_usd=0.05)
    optimizer = BootstrapOptimizer(metric=exact_match, budget=budget)
    result = await optimizer.optimize(call, train_set, val_set)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum

from kaos_core.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "Budget",
    "BudgetTracker",
    "StopReason",
    "estimate_eval_cost",
]


def estimate_eval_cost(eval_result: object) -> tuple[float, int]:
    """Best-effort (cost_usd, tokens) from an :class:`EvalResult`.

    Walks ``per_example`` and sums ``cost_usd`` / ``total_tokens`` from each
    example's trace. Returns ``(0.0, 0)`` for results we cannot introspect.

    Phase 10: ``ExampleResult.trace`` is the canonical per-example trace,
    captured directly by ``evaluate()`` from each ``Invocation`` returned
    by ``call.invoke()``. The legacy ``prediction._last_trace`` fallback
    is gone — there is no result-mutation pattern in the runtime
    contract.

    This is intentionally approximate — it is used for budget enforcement,
    not for billing.
    """
    cost = 0.0
    tokens = 0
    per_example = getattr(eval_result, "per_example", None)
    if not per_example:
        return cost, tokens
    for er in per_example:
        trace = getattr(er, "trace", None)
        if trace is None:
            continue
        try:
            cost += float(getattr(trace, "total_cost_usd", 0.0) or 0.0)
            tokens += int(getattr(trace, "total_tokens", 0) or 0)
        except Exception as exc:
            logger.debug("estimate_eval_cost: failed to extract cost/tokens from trace: %s", exc)
            continue
    return cost, tokens


class StopReason(StrEnum):
    """Reasons an optimizer loop may terminate."""

    BUDGET_TRIALS = "budget_trials"
    BUDGET_TIME = "budget_time"
    BUDGET_COST = "budget_cost"
    BUDGET_TOKENS = "budget_tokens"
    PATIENCE = "patience"
    CONVERGED = "converged"
    COMPLETED = "completed"
    THRESHOLD_NOT_MET = "threshold_not_met"


@dataclass(slots=True)
class Budget:
    """A cap on optimization work.

    At least one cap must be set, otherwise the budget has no effect and is
    a configuration error.

    Args:
        max_trials: Hard cap on the number of trial evaluations.
        max_wall_seconds: Wall-clock budget (seconds) since tracker start.
        max_cost_usd: Total estimated cost cap in USD.
        max_tokens: Total LLM tokens consumed cap.
    """

    max_trials: int | None = None
    max_wall_seconds: float | None = None
    max_cost_usd: float | None = None
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        if (
            self.max_trials is None
            and self.max_wall_seconds is None
            and self.max_cost_usd is None
            and self.max_tokens is None
        ):
            raise ValueError(
                "Budget must set at least one cap (max_trials, max_wall_seconds, "
                "max_cost_usd, or max_tokens). Passing an empty Budget silently "
                "disables every cap. To opt out of budgeting, pass budget=None "
                "to the optimizer instead."
            )
        if self.max_trials is not None and self.max_trials < 1:
            raise ValueError(
                f"Budget.max_trials must be >= 1, got {self.max_trials}. "
                "Use None to disable the trial cap."
            )
        if self.max_wall_seconds is not None and self.max_wall_seconds <= 0:
            raise ValueError(
                f"Budget.max_wall_seconds must be > 0, got {self.max_wall_seconds}. "
                "Use None to disable the wall-clock cap."
            )
        if self.max_cost_usd is not None and self.max_cost_usd < 0:
            raise ValueError(
                f"Budget.max_cost_usd must be >= 0, got {self.max_cost_usd}. "
                "Use None to disable the cost cap."
            )
        if self.max_tokens is not None and self.max_tokens < 0:
            raise ValueError(
                f"Budget.max_tokens must be >= 0, got {self.max_tokens}. "
                "Use None to disable the token cap."
            )

    def make_tracker(self) -> BudgetTracker:
        """Return a fresh :class:`BudgetTracker` for this budget."""
        return BudgetTracker(budget=self)


@dataclass(slots=True)
class BudgetTracker:
    """Mutable tracker for a single optimization run against a :class:`Budget`.

    Tracker isolation: every call to :meth:`Budget.make_tracker` returns a new
    tracker with a fresh ``_start`` timestamp and zero counters. Two concurrent
    optimizations against the same ``Budget`` instance get independent trackers.
    """

    budget: Budget
    trials: int = 0
    cost_usd: float = 0.0
    tokens: int = 0
    _start: float = field(default_factory=time.monotonic)

    def wall_elapsed(self) -> float:
        """Seconds elapsed since the tracker was created."""
        return time.monotonic() - self._start

    def consume(
        self,
        *,
        trials: int = 0,
        cost_usd: float = 0.0,
        tokens: int = 0,
    ) -> None:
        """Record work performed since the last ``consume`` call.

        Optimizers call this after each trial with estimated cost and tokens
        from the trial's execution trace. All arguments are additive.
        """
        if trials < 0 or cost_usd < 0 or tokens < 0:
            raise ValueError(
                f"BudgetTracker.consume received a negative value "
                f"(trials={trials}, cost_usd={cost_usd}, tokens={tokens}). "
                "All consumption must be non-negative."
            )
        self.trials += trials
        self.cost_usd += cost_usd
        self.tokens += tokens

    def exhausted(self) -> StopReason | None:
        """Return the first cap that has been reached, or ``None``.

        Order of precedence: trials → time → cost → tokens. This is arbitrary
        but deterministic — callers should not rely on it beyond "some cap
        matched" when interpreting the :class:`StopReason`.
        """
        b = self.budget
        if b.max_trials is not None and self.trials >= b.max_trials:
            return StopReason.BUDGET_TRIALS
        if b.max_wall_seconds is not None and self.wall_elapsed() >= b.max_wall_seconds:
            return StopReason.BUDGET_TIME
        if b.max_cost_usd is not None and self.cost_usd >= b.max_cost_usd:
            return StopReason.BUDGET_COST
        if b.max_tokens is not None and self.tokens >= b.max_tokens:
            return StopReason.BUDGET_TOKENS
        return None
