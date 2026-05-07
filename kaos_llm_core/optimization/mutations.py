"""Mutation records — typed, observable optimization history.

Every optimization trial produces a Mutation record documenting what changed,
why, and whether it helped. Mutations are persisted as JSONL for reproducibility,
cost tracking, and debugging.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kaos_core.types.content import KaosModel
from pydantic import ConfigDict, Field


def _new_id() -> str:
    """Generate a unique mutation/run identifier.

    Uses ``uuid4().hex`` (32-char hex). The schema doc specifies UUIDv7 for
    time-orderability, but Python 3.13 stdlib does not yet expose ``uuid7``
    (lands in 3.14). uuid4 satisfies the uniqueness contract today; we can
    swap to ``uuid.uuid7().hex`` once Python 3.14 is the minimum.
    """
    return uuid.uuid4().hex


class RunContext:
    """Per-run identity helper that every optimizer threads through its trials.

    A ``RunContext`` is created at the top of an ``optimize()`` call and used
    to stamp every ``Mutation`` produced by that run with consistent identity:

    * ``run_id`` — uuid hex shared by all mutations in this run.
    * ``trial_id`` — monotonic integer that increments before each mutation.
    * ``parent_mutation_id`` — linked-list pointer to the previous mutation,
      regardless of acceptance. Set to None at the start of the run.

    Composite optimizers (``CoOptimizer``) construct **one** RunContext at the
    top and pass the same instance to every child optimizer's ``optimize()``
    so the child mutations share the parent's run_id and continue the trial
    counter rather than starting their own. Standalone optimizers create
    their own RunContext when none is supplied.

    This is the typed alternative to the previous "thread three integers
    through every Mutation construction site" pattern, which was both verbose
    and error-prone (every optimizer would have had to remember to bump the
    trial counter).
    """

    __slots__ = ("_last_mutation_id", "_next_trial_id", "run_id")

    def __init__(self, *, run_id: str | None = None) -> None:
        self.run_id = run_id or _new_id()
        self._next_trial_id = 0
        self._last_mutation_id: str | None = None

    def make_mutation(self, **fields: Any) -> Mutation:
        """Construct a :class:`Mutation` stamped with this run's identity.

        Increments the internal trial counter and updates ``parent_mutation_id``
        so the next mutation links back to this one. ``fields`` are forwarded
        to the :class:`Mutation` constructor verbatim — callers should NOT
        pass ``run_id``, ``trial_id``, ``parent_mutation_id``, or
        ``mutation_id``; if they do, the explicit values win, but doing so
        defeats the linked-list discipline.
        """
        trial_id = self._next_trial_id
        self._next_trial_id += 1
        # Build the mutation. The default mutation_id factory generates a
        # fresh uuid for each instance.
        mutation = Mutation(
            run_id=self.run_id,
            trial_id=trial_id,
            parent_mutation_id=self._last_mutation_id,
            **fields,
        )
        self._last_mutation_id = mutation.mutation_id
        return mutation


# Phase 16.5: mutation log schema version. Bump on any breaking change
# (field rename, type change). Additive changes (new optional fields)
# do NOT need a bump because the model is `extra="ignore"`.
MUTATION_SCHEMA_VERSION = 1


class Mutation(KaosModel):
    """Record of a single optimization trial.

    Serializable via ``model_dump()`` / ``model_validate()`` (KaosModel).

    Schema-evolution policy: this model overrides ``KaosModel``'s default
    ``extra="forbid"`` with ``extra="ignore"`` so that newer writers can add
    fields (per ``docs/internal/design/mutation-log-schema.md``) without
    breaking older readers. The mutation log is a forward-compatible wire
    format; the rest of KaosModel's wire types stay strict.

    The fields ``mutation_id``, ``run_id``, ``trial_id``,
    ``parent_mutation_id``, ``duration_ms``, and ``error`` correspond to
    GAP-1, GAP-6, GAP-7, GAP-8, GAP-4, and GAP-5 in the schema doc and
    enable the Phase 7.3 analysis layer to answer the queries enumerated
    in §7 of that doc.

    **Schema versioning (Phase 16.5).** The class-level
    ``MUTATION_SCHEMA_VERSION`` is the contract version of this dataclass.
    Every record stamps it as the ``schema_version`` field on serialization
    so a future loader can detect the source version. Bump on any breaking
    change (field rename, type change). Pre-Phase-16.5 records have no
    ``schema_version`` field and are loaded as version 1.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    # Phase 16.5: schema version stamp. Every serialized mutation record
    # carries this so future loaders can detect drift. Default value is
    # the constant below; downstream code should NOT override it on
    # construction.
    schema_version: int = 1

    # Identity (GAP-1, GAP-6, GAP-7, GAP-8) -----------------------------------
    # ``mutation_id`` is unique per record (uuid4 hex). ``run_id`` is shared
    # across every mutation produced by a single ``optimize()`` invocation
    # (CoOptimizer shares one ``run_id`` across all its child stages).
    # ``trial_id`` is monotonic within a run, starting at 0.
    # ``parent_mutation_id`` chains mutations linearly: it points at the
    # previously-recorded mutation in the same run, regardless of acceptance.
    # The reader uses this to reconstruct the trial history.
    mutation_id: str = Field(default_factory=_new_id)
    run_id: str | None = None
    trial_id: int = 0
    parent_mutation_id: str | None = None

    # Mutation content --------------------------------------------------------
    strategy: str
    mutation_type: str
    call_name: str
    before: dict[str, Any]
    after: dict[str, Any]
    rationale: str = ""

    # Outcome -----------------------------------------------------------------
    metric_before: float = 0.0
    metric_after: float = 0.0
    tokens_used: int = 0
    cost_usd: float = 0.0
    duration_ms: float = 0.0  # GAP-4: wall time spent on this trial
    accepted: bool = False
    # GAP-5: when an optimizer trial errors out (provider 4xx, validation
    # exhausted, etc.) the optimizer may still want to record what was tried.
    # ``error`` carries the most-common-exception class name + message; an
    # empty/None value means the trial completed without raising.
    error: str | None = None

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def improvement(self) -> float:
        """Metric improvement (positive = better)."""
        return self.metric_after - self.metric_before


class MutationLog:
    """Append-only log of Mutation records, backed by JSONL."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else None
        self._mutations: list[Mutation] = []

    @property
    def mutations(self) -> list[Mutation]:
        return list(self._mutations)

    def record(self, mutation: Mutation) -> None:
        """Record a mutation and optionally persist to JSONL."""
        self._mutations.append(mutation)
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(mutation.model_dump_json() + "\n")

    def accepted(self) -> list[Mutation]:
        """Return only accepted mutations."""
        return [m for m in self._mutations if m.accepted]

    def total_cost(self) -> float:
        return sum(m.cost_usd for m in self._mutations)

    def total_tokens(self) -> int:
        return sum(m.tokens_used for m in self._mutations)

    def best_improvement(self) -> Mutation | None:
        accepted = self.accepted()
        if not accepted:
            return None
        return max(accepted, key=lambda m: m.improvement)

    def summary(self) -> str:
        lines = [
            f"Optimization Log: {len(self._mutations)} trials",
            f"  Accepted: {len(self.accepted())}",
            f"  Total cost: ${self.total_cost():.4f}",
            f"  Total tokens: {self.total_tokens():,}",
        ]
        best = self.best_improvement()
        if best:
            lines.append(
                f"  Best improvement: {best.metric_before:.1%} → {best.metric_after:.1%} "
                f"({best.strategy}/{best.mutation_type})"
            )
        return "\n".join(lines)

    @classmethod
    def load(cls, path: str | Path) -> MutationLog:
        """Load a JSONL mutation log from disk.

        Phase 16.5: rejects logs that carry a ``schema_version`` newer
        than ``MUTATION_SCHEMA_VERSION``. Pre-Phase-16.5 lines without
        a ``schema_version`` field are accepted as version 1 (the
        ``Mutation`` model defaults the field).
        """
        log = cls(path=path)
        p = Path(path)
        if not p.exists():
            return log
        with p.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                # Cheap version check before full pydantic validation
                # so a future-version log fails loudly without leaking
                # validation errors that obscure the real cause.
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    msg = f"MutationLog.load: line {line_no} of {p} is not valid JSON: {exc}"
                    raise ValueError(msg) from exc
                version = raw.get("schema_version", 1)
                if not isinstance(version, int) or version > MUTATION_SCHEMA_VERSION:
                    msg = (
                        f"MutationLog.load: line {line_no} of {p} has "
                        f"schema_version={version!r} which is newer than "
                        f"this kaos-llm-core release supports "
                        f"(max {MUTATION_SCHEMA_VERSION}). Upgrade kaos-llm-core "
                        "to read this log."
                    )
                    raise ValueError(msg)
                log._mutations.append(Mutation.model_validate(raw))
        return log
