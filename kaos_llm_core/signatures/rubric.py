"""ScoringRubric value types — typed criteria + qualitative tiebreaker.

The rubric is the second-half partner of an ``ExtractionSchema``: where
the schema describes *what* to extract from a corpus, the rubric describes
*how to compare* the extracted rows. Agents synthesize a rubric at runtime
(via a ``Call(RubricDesignerSignature)``), pass it to ``apply_rubric``
(see :mod:`kaos_llm_core.programs.scoring`), and get a scored
``TabularDocument`` back.

Design (synthesised from the 2026-05-10 program-synthesis prototypes):

- **Typed ``Criterion`` list** — auditable, machine-evaluable predicates.
  Each criterion has a column name, an operator (from a fixed enum),
  a value, a weight, and a free-form rationale. Scored in pure Python:
  no LLM call on the hot path.
- **Free-form ``qualitative_guidance`` string** — the tiebreaker channel
  for residual professional judgment (e.g. "are the indemnity caps
  reasonable for a deal of this size?"). Fed to one LLM call per row
  via the dedicated ``RowJudgmentSignature``.
- **``hard_weight`` + ``soft_weight``** — must sum to 1.0. The final
  per-row score combines both channels.

Boundary modes fall out as configurations:

- Pure deterministic mode → empty ``qualitative_guidance``,
  ``hard_weight=1.0``. Zero LLM calls in ``apply_rubric``.
- Pure qualitative mode → empty ``criteria``, ``hard_weight=0.0``.
  One LLM call per row.
- Hybrid mode (default) → ``hard_weight=0.6``, ``soft_weight=0.4``.
  Both channels active.

Columns are referenced by **name** (``str``), not by a typed
``ColumnSpec`` reference. JSON round-trip is trivial; missing columns
silently score 0 for that criterion (the row's ``.get(col)`` returns
``None``, which fails all numeric/equality operators except ``falsy`` and
``not_in``). Schema-validation against an ``ExtractionSchema`` is a
separate, optional concern, not part of the type itself.

The ``HybridRubric`` round-trips cleanly through ``model_dump_json()`` /
``model_validate_json()`` so agents that synthesise it via Call output
get a typed, immutable artifact back. The whole thing is frozen.

Why operators are a fixed enum rather than free-form strings: the
2026-05-10 prototypes found that an LLM-emitted predicate string like
``"value > 24 and <= 36"`` requires another LLM call to interpret. The
fixed operator enum collapses that into deterministic Python and saves
~$0.04/run on a 5-document corpus while eliminating scoring drift.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# The 11-operator vocabulary covers the cases observed across legal
# contract comparison, financial filing review, and general structured
# extraction. Any operator the rubric designer invents outside this set
# fails at Pydantic decode time, which is the right place to catch it.
CriterionOperator = Literal[
    "eq",
    "neq",
    "lt",
    "lte",
    "gt",
    "gte",
    "in",
    "not_in",
    "contains",
    "truthy",
    "falsy",
]


class Criterion(BaseModel):
    """One auditable rule in a ``HybridRubric``.

    Attributes:
        column: Column name to read from each row (string, not
            :class:`~kaos_llm_core.signatures.extraction.ColumnSpec`).
            Missing columns score 0 for this criterion.
        operator: One of the 11 fixed operators (see
            :data:`CriterionOperator`). The LLM that synthesises the
            rubric is constrained to these — anything else fails
            Pydantic Literal validation at decode time.
        value: JSON-native value compared against the cell. Type
            depends on the operator: numeric for ``lt`` / ``lte`` /
            ``gt`` / ``gte``; collection for ``in`` / ``not_in``;
            unused for ``truthy`` / ``falsy``.
        weight: Magnitude of this criterion's contribution to the
            hard-channel score. Negative weights are allowed (penalty
            criteria) — normalisation handles both signs.
        rationale: Free-form explanation. Surfaces in the per-row
            ``_reasoning`` output so audit trails are human-readable.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    column: str = Field(min_length=1)
    operator: CriterionOperator
    value: Any = None
    weight: float
    rationale: str = ""


class HybridRubric(BaseModel):
    """Typed criteria + qualitative tiebreaker.

    Synthesised by an agent at runtime; consumed by
    :func:`~kaos_llm_core.programs.scoring.apply_rubric` to score each
    row of a ``TabularDocument``. Frozen and JSON round-trippable —
    agents emit these via ``Call`` output and the wire layer
    serialises them with ``model_dump_json()`` for audit + replay.

    Attributes:
        objective: One-sentence statement of what the rubric optimises
            for (surfaces in the per-row reasoning and the recommender
            prompt). Required.
        criteria: Tuple of typed :class:`Criterion`. Evaluated in pure
            Python — no LLM calls. Can be empty (then ``hard_weight``
            must be 0.0).
        qualitative_guidance: Free-form prompt for residual judgment.
            Fed to one LLM call per row via
            :class:`~kaos_llm_core.programs.scoring.RowJudgmentSignature`.
            Empty string means pure deterministic mode.
        hard_weight: Weight of the typed-criteria channel in the
            aggregate score. Default 0.6.
        soft_weight: Weight of the qualitative channel. Default 0.4.
            Must satisfy ``hard_weight + soft_weight == 1.0``.

    Boundary modes:

    - Empty ``criteria`` + non-empty ``qualitative_guidance`` + matching
      weights → pure qualitative mode (one LLM call per row).
    - Non-empty ``criteria`` + empty ``qualitative_guidance`` +
      ``hard_weight=1.0`` → pure deterministic mode (zero LLM calls).
    - The constructor rejects fully-empty rubrics (both ``criteria`` and
      ``qualitative_guidance`` empty) — they're meaningless.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    objective: str = Field(min_length=1)
    criteria: tuple[Criterion, ...] = Field(default_factory=tuple)
    qualitative_guidance: str = ""
    hard_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    soft_weight: float = Field(default=0.4, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_invariants(self) -> HybridRubric:
        if abs(self.hard_weight + self.soft_weight - 1.0) > 1e-9:
            raise ValueError(
                f"HybridRubric: hard_weight + soft_weight must equal 1.0, "
                f"got {self.hard_weight} + {self.soft_weight} = "
                f"{self.hard_weight + self.soft_weight}. "
                "Adjust the weights so they sum to exactly 1.0."
            )
        if not self.criteria and not self.qualitative_guidance.strip():
            raise ValueError(
                "HybridRubric must have at least one of: typed criteria or "
                "qualitative_guidance. An empty rubric is meaningless. "
                "How to fix: add at least one Criterion, or provide a "
                "non-empty qualitative_guidance string."
            )
        if not self.criteria and self.hard_weight > 0:
            raise ValueError(
                f"HybridRubric: no criteria provided but hard_weight="
                f"{self.hard_weight} > 0. With no typed criteria, set "
                "hard_weight=0.0 and soft_weight=1.0 for pure qualitative "
                "mode."
            )
        if not self.qualitative_guidance.strip() and self.soft_weight > 0:
            raise ValueError(
                f"HybridRubric: no qualitative_guidance but soft_weight="
                f"{self.soft_weight} > 0. With no qualitative channel, "
                "set soft_weight=0.0 and hard_weight=1.0."
            )
        return self


__all__ = [
    "Criterion",
    "CriterionOperator",
    "HybridRubric",
]
