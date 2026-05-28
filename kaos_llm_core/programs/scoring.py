"""Rubric-based row scoring for ``TabularDocument``.

Given a :class:`~kaos_llm_core.signatures.rubric.HybridRubric` and a
``TabularDocument`` (typically produced by ``extract_corpus`` over a
corpus of documents), :func:`apply_rubric` returns a new typed table
with ``_score`` and ``_reasoning`` columns appended.

Design:

- **Hard channel**: deterministic, pure-Python evaluation of typed
  :class:`~kaos_llm_core.signatures.rubric.Criterion`. Each cell goes
  through :func:`_evaluate_criterion` which dispatches on the
  operator; sum of weighted matches normalised to ``[0, 1]``. No LLM
  involvement on this path.
- **Soft channel**: when ``rubric.qualitative_guidance`` is non-empty,
  one ``Call(RowJudgmentSignature)`` per row, executed concurrently
  via ``asyncio.gather``. The signature explicitly names the
  row-scoring contract — distinct from :class:`Judge` which scores
  producer/judge generation pairs.
- **Aggregate**: ``hard * rubric.hard_weight + soft * rubric.soft_weight``.
  Weights sum to 1.0 (enforced by ``HybridRubric``).

Why not :class:`Judge` for the soft channel: empirically tested in the
2026-05-10 program-synthesis prototypes. Judge is wired around a
producer/judge pair — feeding it a row makes the model grade it as
"you echoed input without analysing anything," collapsing all scores
to ~0.10. The dedicated ``RowJudgmentSignature`` resolves this by
naming the inputs (``objective``, ``qualitative_guidance``, ``row_json``)
and outputs (``score``, ``reasoning``) so the model knows what game it's
playing. After the swap, live Sonnet 4.6 produced sensibly ordered
scores (0.87 / 0.73 / 0.38) on a 3-row synthetic table.

Cost: roughly ~$0.008 per row at ``anthropic:claude-sonnet-4-6`` when
the soft channel is active; zero when ``qualitative_guidance`` is empty.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from kaos_content.model.tabular import Column, ColumnType, Table, TabularDocument

from kaos_llm_core.programs.call import Call
from kaos_llm_core.signatures.fields import InputField, OutputField
from kaos_llm_core.signatures.rubric import Criterion, HybridRubric
from kaos_llm_core.signatures.signature import Signature

# Default judge model. Tracks the platform-wide research/sonnet default.
_DEFAULT_JUDGE_MODEL: str = "anthropic:claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Row-judgment signature — the LLM-facing contract for the soft channel
# ---------------------------------------------------------------------------


class RowJudgmentSignature(Signature):
    """Score one row of a ``TabularDocument`` against qualitative guidance.

    The row IS the thing being scored. Read the objective and
    qualitative_guidance, then return a 0.0--1.0 score with a
    one-paragraph reasoning string.
    """

    objective: str = InputField(description="The scoring objective.")
    qualitative_guidance: str = InputField(
        description="Free-form qualitative criteria for the soft channel.",
    )
    row_json: str = InputField(
        description="The row to score, serialized as a JSON object.",
    )
    score: float = OutputField(
        description=(
            "Quality score from 0.0 (terrible fit against the guidance) to "
            "1.0 (perfect fit). Avoid 0.5 unless genuinely ambiguous."
        ),
    )
    reasoning: str = OutputField(description="One-paragraph explanation of the score.")


# ---------------------------------------------------------------------------
# Hard channel — deterministic, pure-Python evaluation
# ---------------------------------------------------------------------------


def _evaluate_criterion(criterion: Criterion, cell: Any) -> bool:
    """Return True iff ``cell`` satisfies ``criterion``.

    Missing cells (``None``) match only ``falsy`` and ``not_in``.
    Type-mismatched comparisons (e.g., ``"michigan" < 5``) return False
    silently — the rubric synthesizer is responsible for type-aligning
    criteria with column types. The mismatch shows up in the row's
    reasoning string so audit can spot the misconfiguration.
    """
    op = criterion.operator
    target = criterion.value

    if op == "truthy":
        return bool(cell)
    if op == "falsy":
        return not cell
    if cell is None:
        return op == "not_in"

    try:
        if op == "eq":
            return cell == target
        if op == "neq":
            return cell != target
        if op == "lt":
            return cell < target
        if op == "lte":
            return cell <= target
        if op == "gt":
            return cell > target
        if op == "gte":
            return cell >= target
        if op == "in":
            return cell in target
        if op == "not_in":
            return cell not in target
        if op == "contains":
            return str(target) in str(cell)
    except (TypeError, ValueError):
        return False
    return False


def _hard_score(rubric: HybridRubric, row: dict[str, Any]) -> tuple[float, list[str]]:
    """Compute the deterministic hard score for one row.

    Returns ``(score_in_[0, 1], explanation_lines)``. Normalisation:
    ``sum(weight x matched) / sum(|weight|)`` gives a value in ``[-1, +1]``;
    linearly remap to ``[0, 1]``.
    """
    if not rubric.criteria:
        return (0.5, ["no hard criteria; neutral score 0.5"])

    explanations: list[str] = []
    raw = 0.0
    max_magnitude = sum(abs(c.weight) for c in rubric.criteria)

    for crit in rubric.criteria:
        cell = row.get(crit.column)
        matched = _evaluate_criterion(crit, cell)
        if matched:
            raw += crit.weight
            explanations.append(
                f"  [{crit.weight:+.2f}] {crit.column} {crit.operator} "
                f"{crit.value!r} matched (cell={cell!r}) — {crit.rationale}"
            )
        else:
            explanations.append(
                f"  [ 0.00] {crit.column} {crit.operator} {crit.value!r} "
                f"NOT matched (cell={cell!r})"
            )

    if max_magnitude == 0:
        return (0.5, [*explanations, "max_magnitude=0; neutral score 0.5"])
    normalised = raw / max_magnitude  # in [-1, +1]
    score_01 = (normalised + 1.0) / 2.0
    return (score_01, explanations)


# ---------------------------------------------------------------------------
# apply_rubric — the public entry point
# ---------------------------------------------------------------------------


async def apply_rubric(
    rubric: HybridRubric,
    doc: TabularDocument,
    *,
    judge_model: str = _DEFAULT_JUDGE_MODEL,
    table_name: str | None = None,
) -> TabularDocument:
    """Score every row of ``doc`` against ``rubric``.

    Returns a new ``TabularDocument`` with two extra columns appended
    to the chosen table:

    - ``_score`` (FLOAT, ``[0, 1]``): aggregate score
    - ``_reasoning`` (TEXT): per-row explanation

    The original table's columns and metadata are preserved. The
    aggregate score is:

    .. code-block:: text

        agg = rubric.hard_weight * hard_score + rubric.soft_weight * soft_score

    Args:
        rubric: The synthesised :class:`~kaos_llm_core.signatures.rubric.HybridRubric`.
        doc: Input ``TabularDocument`` — typically the output of
            ``extract_corpus`` over a corpus of documents.
        judge_model: Model identifier for the soft-channel
            :class:`RowJudgmentSignature` Call. Defaults to
            ``anthropic:claude-sonnet-4-6``. Ignored when the rubric
            has no ``qualitative_guidance``.
        table_name: Which table in ``doc`` to score. Defaults to the
            first table.

    Raises:
        ValueError: if ``doc`` has no tables, or ``table_name`` is set
            and no table by that name exists.

    Cost: ~$0.008/row at Sonnet 4.6 when the soft channel is active;
    zero LLM cost when ``rubric.qualitative_guidance`` is empty. Rows
    are scored concurrently via ``asyncio.gather``.
    """
    if not doc.tables:
        raise ValueError(
            "apply_rubric: input TabularDocument has no tables. "
            "How to fix: pass a TabularDocument built from at least one "
            "extraction result. Alternative: use "
            "TabularDocument.from_cells(...) to wrap extraction cells."
        )

    if table_name is not None:
        try:
            table = doc.get_table(table_name)
        except (KeyError, AttributeError) as exc:
            available = ", ".join(t.name for t in doc.tables) or "(none)"
            raise ValueError(
                f"apply_rubric: no table named {table_name!r}. Available: {available}."
            ) from exc
    else:
        table = doc.tables[0]

    col_names = tuple(c.name for c in table.columns)

    # See module docstring for why we use Call(RowJudgmentSignature)
    # directly rather than kaos_llm_core.programs.judge.Judge.
    judge_call = Call(RowJudgmentSignature, model=judge_model)

    async def _score_one_row(row: tuple[Any, ...]) -> tuple[float, str]:
        row_dict = dict(zip(col_names, row, strict=False))
        hard, hard_lines = _hard_score(rubric, row_dict)

        soft = 0.5
        soft_reason = "(no qualitative guidance)"
        if rubric.qualitative_guidance:
            try:
                judgment = await judge_call(
                    objective=rubric.objective,
                    qualitative_guidance=rubric.qualitative_guidance,
                    row_json=json.dumps(row_dict, default=str),
                )
                soft = float(judgment.score)
                soft_reason = str(judgment.reasoning)
            except Exception as exc:  # pragma: no cover — defensive
                # Live-call defensive: surface the failure in the
                # reasoning string rather than poisoning the whole
                # table. The hard channel still contributes its score.
                soft_reason = f"(judge error: {type(exc).__name__}: {exc})"

        agg = rubric.hard_weight * hard + rubric.soft_weight * soft
        # Clamp to [0, 1] defensively — a misbehaving judge could return
        # a score outside the contract.
        agg = max(0.0, min(1.0, agg))

        reasoning = "\n".join(
            [
                f"objective: {rubric.objective}",
                f"hard={hard:.3f} (weight {rubric.hard_weight}):",
                *hard_lines,
                f"soft={soft:.3f} (weight {rubric.soft_weight}): {soft_reason}",
                f"aggregate={agg:.3f}",
            ]
        )
        return (agg, reasoning)

    results = await asyncio.gather(
        *(_score_one_row(r) for r in table.rows), return_exceptions=False
    )

    new_columns = (
        *table.columns,
        Column(name="_score", column_type=ColumnType.FLOAT, nullable=False),
        Column(name="_reasoning", column_type=ColumnType.TEXT, nullable=False),
    )
    new_rows = tuple(
        (*row, score, reasoning)
        for row, (score, reasoning) in zip(table.rows, results, strict=True)
    )
    new_metadata = dict(table.metadata)
    new_metadata["scored_by"] = rubric.objective[:80]
    new_table = Table(
        name=table.name,
        columns=new_columns,
        rows=new_rows,
        row_count=len(new_rows),
        metadata=new_metadata,
    )
    return TabularDocument(
        metadata=doc.metadata,
        tables=(new_table,),
        provenance=doc.provenance,
    )


__all__ = [
    "RowJudgmentSignature",
    "apply_rubric",
]
