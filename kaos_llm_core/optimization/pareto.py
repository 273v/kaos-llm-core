"""ParetoOptimizer — track the (metric, cost) Pareto frontier.

Runs an inner optimizer (CodecOptimizer, ModelOptimizer, ...) and intercepts
its mutation records to compute the Pareto frontier across all trials.

A trial ``A`` **dominates** trial ``B`` if ``A.metric >= B.metric`` and
``A.cost <= B.cost`` with at least one strict inequality. The frontier is
the set of non-dominated trials — points where no other trial is strictly
better on both axes simultaneously.

Integration strategy: the Pareto wrapper does **not** instrument the inner
loop. It runs ``inner.optimize()`` to completion, then reads the mutations
recorded on the inner optimizer's ``mutation_log`` (or on the result's
``mutations`` attribute when present) and computes the frontier from the
``(metric_after, cost_usd)`` pair on each mutation. This keeps the wrapper
dead simple and works with every optimizer that writes a mutation log.

Example::

    inner = ModelOptimizer(metric=exact_match, models=[...])
    pareto = ParetoOptimizer(metric=exact_match, inner=inner)
    result = await pareto.optimize(call, val_set)
    for config, metric, cost in result.frontier:
        print(f"{config} → {metric:.3f} @ ${cost:.4f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kaos_core.logging import get_logger

from kaos_llm_core.optimization.budget import Budget, StopReason
from kaos_llm_core.optimization.evaluation import MetricFn
from kaos_llm_core.optimization.mutations import Mutation
from kaos_llm_core.types import Example

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ParetoResult:
    """Result of Pareto multi-objective optimization."""

    frontier: list[tuple[dict[str, Any], float, float]]
    all_trials: list[tuple[dict[str, Any], float, float]]
    stop_reason: str = StopReason.COMPLETED.value
    inner_result: Any = field(default=None)


def compute_pareto_frontier(
    trials: list[tuple[dict[str, Any], float, float]],
) -> list[tuple[dict[str, Any], float, float]]:
    """Return the Pareto frontier of (config, metric, cost) trials.

    Higher metric is better; lower cost is better. A trial is on the
    frontier iff no other trial dominates it. Dominance: other.metric >=
    self.metric AND other.cost <= self.cost AND at least one strict.
    """
    frontier: list[tuple[dict[str, Any], float, float]] = []
    for i, (cfg_i, m_i, c_i) in enumerate(trials):
        dominated = False
        for j, (_, m_j, c_j) in enumerate(trials):
            if i == j:
                continue
            if m_j >= m_i and c_j <= c_i and (m_j > m_i or c_j < c_i):
                dominated = True
                break
        if not dominated:
            frontier.append((cfg_i, m_i, c_i))
    # Sort by cost ascending for deterministic output.
    frontier.sort(key=lambda t: (t[2], -t[1]))
    return frontier


class ParetoOptimizer:
    """Wrap an inner optimizer and compute the (metric, cost) Pareto frontier.

    Args:
        metric: Scoring function (unused directly, kept for API parity).
        inner: Any optimizer with an ``optimize(program, val_set)`` coroutine
            that records mutations to ``self.mutation_log`` or returns a
            result with a ``mutations`` attribute.
        budget: Optional budget. Forwarded to the inner optimizer if it
            exposes a ``budget`` attribute; otherwise ignored.
    """

    def __init__(
        self,
        metric: MetricFn,
        inner: Any,
        *,
        budget: Budget | None = None,
    ) -> None:
        self.metric = metric
        self.inner = inner
        self.budget = budget
        if budget is not None and hasattr(inner, "budget") and inner.budget is None:
            inner.budget = budget

    async def optimize(
        self,
        program: Any,
        val_set: list[Example],
        *,
        max_concurrent: int = 5,
    ) -> ParetoResult:
        inner_result = await self.inner.optimize(program, val_set, max_concurrent=max_concurrent)

        mutations: list[Mutation] = []
        inner_muts = getattr(inner_result, "mutations", None)
        if isinstance(inner_muts, list) and inner_muts:
            mutations = [m for m in inner_muts if isinstance(m, Mutation)]
        else:
            log = getattr(self.inner, "mutation_log", None)
            if log is not None:
                mutations = list(log.mutations)

        trials: list[tuple[dict[str, Any], float, float]] = []
        for m in mutations:
            cfg = dict(m.after) if isinstance(m.after, dict) else {"value": m.after}
            trials.append((cfg, float(m.metric_after), float(m.cost_usd)))

        frontier = compute_pareto_frontier(trials)

        stop_reason = getattr(inner_result, "stop_reason", StopReason.COMPLETED.value)

        logger.debug("ParetoOptimizer: %d trial(s), %d on frontier", len(trials), len(frontier))

        return ParetoResult(
            frontier=frontier,
            all_trials=trials,
            stop_reason=stop_reason,
            inner_result=inner_result,
        )
