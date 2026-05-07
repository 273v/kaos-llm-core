"""Mutation log analysis layer.

Pure analysis functions on top of :class:`MutationLog`. Loads JSONL logs,
groups by strategy, computes contributions, and renders trial cards. Does
NOT mutate the log — write-side concerns live in
:mod:`kaos_llm_core.optimization.mutations`.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from kaos_core.logging import get_logger

from kaos_llm_core.optimization.mutations import Mutation

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class TrialCard:
    """Rendered summary of a single optimization trial."""

    trial_id: int
    strategy: str
    mutation_type: str
    call_name: str
    metric_before: float
    metric_after: float
    improvement: float
    accepted: bool
    tokens_used: int
    cost_usd: float
    timestamp: datetime
    diff: str

    def render(self) -> str:
        """Multi-line plain-text rendering of this trial. Deterministic."""
        accepted_marker = "ACCEPTED" if self.accepted else "rejected"
        improvement_sign = "+" if self.improvement >= 0 else ""
        lines = [
            f"Trial #{self.trial_id} [{accepted_marker}]",
            f"  strategy:    {self.strategy}",
            f"  mutation:    {self.mutation_type}",
            f"  call:        {self.call_name}",
            f"  metric:      {self.metric_before:.4f} -> {self.metric_after:.4f} "
            f"({improvement_sign}{self.improvement:.4f})",
            f"  tokens:      {self.tokens_used}",
            f"  cost_usd:    {self.cost_usd:.6f}",
            f"  timestamp:   {self.timestamp.isoformat()}",
        ]
        if self.diff:
            lines.append("  diff:")
            for diff_line in self.diff.splitlines():
                lines.append(f"    {diff_line}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class StrategyContribution:
    """Aggregate contribution of a single optimizer strategy."""

    strategy: str
    trials: int
    wins: int
    total_improvement: float
    average_cost: float


def load_mutations(path: str | Path) -> list[Mutation]:
    """Load a MutationLog JSONL file into a list of :class:`Mutation` records.

    **Error policy:**

    * Blank lines are skipped silently (they have no semantic content).
    * The **last** non-blank line is allowed to fail parsing — that case is
      treated as a partially-written final record (the writer was killed mid-
      flush) and skipped with no warning.
    * Any **non-final** broken line is logged at WARNING level with the line
      number and continuation. The line is dropped from the result so the
      caller still gets the parseable records, but the log makes the dropped
      line visible. Previously every broken line was swallowed silently — a
      ``valid-invalid-valid`` JSONL file loaded as 2 records with no warning,
      making trial summaries silently inaccurate.

    Raises :class:`FileNotFoundError` if the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Mutation log not found at {p}. "
            "Check the path. Alternative: pass the path returned from "
            "MutationLog(path=...).record(...)."
        )

    # Materialize first so we can identify the *last* non-blank line and apply
    # the partial-write tolerance only to it. JSONL files are typically small
    # enough that this is fine.
    raw_lines: list[tuple[int, str]] = []  # (1-based line number, stripped)
    with p.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            stripped = line.strip()
            if stripped:
                raw_lines.append((lineno, stripped))

    if not raw_lines:
        return []

    last_lineno = raw_lines[-1][0]

    mutations: list[Mutation] = []
    for lineno, stripped in raw_lines:
        try:
            mutations.append(Mutation.model_validate_json(stripped))
        except Exception as exc:
            if lineno == last_lineno:
                # Partial-write tolerance — silent.
                logger.debug(
                    "load_mutations: dropping unparseable final line %d of %s "
                    "(treated as partial write): %s",
                    lineno,
                    p,
                    exc,
                )
            else:
                logger.warning(
                    "load_mutations: dropping unparseable line %d of %s "
                    "(non-final, NOT a partial-write): %s. The remainder of "
                    "the file is still being loaded; trial summaries computed "
                    "from this log are missing one record.",
                    lineno,
                    p,
                    exc,
                )
    return mutations


def _render_diff(before: dict[str, Any], after: dict[str, Any]) -> str:
    """Render a unified diff between two dicts using their JSON representation."""
    try:
        before_text = json.dumps(before, indent=2, sort_keys=True, default=str)
    except Exception as exc:
        logger.debug("_render_diff: json.dumps(before) failed, falling back to repr(): %s", exc)
        before_text = repr(before)
    try:
        after_text = json.dumps(after, indent=2, sort_keys=True, default=str)
    except Exception as exc:
        logger.debug("_render_diff: json.dumps(after) failed, falling back to repr(): %s", exc)
        after_text = repr(after)
    diff_lines = difflib.unified_diff(
        before_text.splitlines(),
        after_text.splitlines(),
        fromfile="before",
        tofile="after",
        lineterm="",
    )
    return "\n".join(diff_lines)


def make_trial_cards(mutations: list[Mutation]) -> list[TrialCard]:
    """Convert mutations into trial cards with rendered diffs."""
    cards: list[TrialCard] = []
    for idx, m in enumerate(mutations):
        cards.append(
            TrialCard(
                trial_id=idx,
                strategy=m.strategy,
                mutation_type=m.mutation_type,
                call_name=m.call_name,
                metric_before=m.metric_before,
                metric_after=m.metric_after,
                improvement=m.improvement,
                accepted=m.accepted,
                tokens_used=m.tokens_used,
                cost_usd=m.cost_usd,
                timestamp=m.timestamp,
                diff=_render_diff(m.before, m.after),
            )
        )
    return cards


def strategy_contributions(mutations: list[Mutation]) -> list[StrategyContribution]:
    """Group mutations by strategy and compute contribution stats.

    Sorted by ``total_improvement`` descending, then by strategy name ascending
    for deterministic ordering on ties.
    """
    grouped: dict[str, list[Mutation]] = {}
    for m in mutations:
        grouped.setdefault(m.strategy, []).append(m)

    contributions: list[StrategyContribution] = []
    for strategy, items in grouped.items():
        trials = len(items)
        wins = sum(1 for m in items if m.improvement > 0)
        total_improvement = sum(m.improvement for m in items if m.improvement > 0)
        total_cost = sum(m.cost_usd for m in items)
        average_cost = total_cost / trials if trials else 0.0
        contributions.append(
            StrategyContribution(
                strategy=strategy,
                trials=trials,
                wins=wins,
                total_improvement=total_improvement,
                average_cost=average_cost,
            )
        )

    contributions.sort(key=lambda c: (-c.total_improvement, c.strategy))
    return contributions


@dataclass
class _RunSummary:
    total_trials: int
    accepted: int
    wins: int
    total_cost_usd: float
    total_tokens: int
    best_metric_after: float
    best_improvement: float
    strategies: list[StrategyContribution] = field(default_factory=list)


def summarize_run(mutations: list[Mutation]) -> dict[str, Any]:
    """Top-level summary of a mutation log. Deterministic output for tests.

    Returns a dict with:

    - ``total_trials`` — number of mutation records
    - ``accepted`` — number marked accepted=True
    - ``wins`` — number with positive improvement
    - ``total_cost_usd`` — sum of cost_usd
    - ``total_tokens`` — sum of tokens_used
    - ``best_metric_after`` — highest metric_after observed (0.0 if empty)
    - ``best_improvement`` — largest positive improvement (0.0 if empty)
    - ``strategies`` — list of :class:`StrategyContribution` dicts (sorted)
    """
    if not mutations:
        return {
            "total_trials": 0,
            "accepted": 0,
            "wins": 0,
            "total_cost_usd": 0.0,
            "total_tokens": 0,
            "best_metric_after": 0.0,
            "best_improvement": 0.0,
            "strategies": [],
        }

    accepted = sum(1 for m in mutations if m.accepted)
    wins = sum(1 for m in mutations if m.improvement > 0)
    total_cost = sum(m.cost_usd for m in mutations)
    total_tokens = sum(m.tokens_used for m in mutations)
    best_after = max(m.metric_after for m in mutations)
    best_improvement = max((m.improvement for m in mutations), default=0.0)

    return {
        "total_trials": len(mutations),
        "accepted": accepted,
        "wins": wins,
        "total_cost_usd": total_cost,
        "total_tokens": total_tokens,
        "best_metric_after": best_after,
        "best_improvement": max(0.0, best_improvement),
        "strategies": [
            {
                "strategy": c.strategy,
                "trials": c.trials,
                "wins": c.wins,
                "total_improvement": c.total_improvement,
                "average_cost": c.average_cost,
            }
            for c in strategy_contributions(mutations)
        ],
    }
