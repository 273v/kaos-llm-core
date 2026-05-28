"""DealReviewProgram — runtime program synthesis for portfolio review.

The reference implementation of the "agent as program synthesizer"
architecture. Given a user question, a document corpus, and optional
preferences, the five-phase pipeline:

1. **Design schema** — :func:`~kaos_llm_core.programs.designers.design_schema`
   synthesizes an :class:`~kaos_llm_core.signatures.extraction.ExtractionSchema`
   tailored to the question.
2. **Extract** — :func:`~kaos_llm_core.programs.extract.extract_corpus`
   runs the schema across every document in the corpus.
3. **Design rubric** — :func:`~kaos_llm_core.programs.designers.design_rubric`
   synthesizes a :class:`~kaos_llm_core.signatures.rubric.HybridRubric`
   from the schema + user preferences.
4. **Score** — :func:`~kaos_llm_core.programs.scoring.apply_rubric`
   applies the rubric to the extracted ``TabularDocument``,
   appending ``_score`` and ``_reasoning`` columns.
5. **Recommend** — a final synthesis Call produces a structured
   recommendation (endorsed / not_endorsed lists + narrative)
   from the scored table.

This is a ``Program``, not an agent. The agent layer (kaos-agents'
``DealReviewAgent``) wraps this Program for the wire-facing surfaces
(CLI / API / MCP) — see ``kaos_agents/patterns/deal_review.py``.

Why a Program subclass rather than a Program v3 envelope (the 2026-05-10
prototype comparison): the envelope packaging is 54% more expensive
and 2x more lines because the envelope schema has no ``extract`` step
kind, no list builder, and no corpus-loop combinator. The Program
subclass executes; the envelope role becomes audit-replay artifact
(future work — emit a content-addressed envelope alongside the
DealReviewResult).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from kaos_llm_core.programs.base import Program
from kaos_llm_core.programs.call import Call
from kaos_llm_core.programs.designers import (
    design_rubric,
    design_schema,
    sample_corpus_text,
)
from kaos_llm_core.programs.scoring import apply_rubric
from kaos_llm_core.signatures.extraction import ExtractionSchema
from kaos_llm_core.signatures.fields import InputField, OutputField
from kaos_llm_core.signatures.rubric import HybridRubric
from kaos_llm_core.signatures.signature import Signature

_DEFAULT_MODEL: str = "anthropic:claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Final synthesis signature
# ---------------------------------------------------------------------------


class _RecommendationSignature(Signature):
    """Synthesize a structured recommendation from the scored table.

    Used by ``DealReviewProgram`` as the final phase.
    """

    question: str = InputField(description="The user's original review question.")
    preferences: str = InputField(
        description="The user's stated preferences (may be empty).",
    )
    scored_table_json: str = InputField(
        description=(
            "The scored TabularDocument as JSON. Each row carries the "
            "extracted column values plus ``_score`` (float, 0..1) and "
            "``_reasoning`` (str)."
        ),
    )
    rubric_summary: str = InputField(
        description="One-paragraph summary of the rubric's criteria + weights.",
    )
    endorsed: list[str] = OutputField(
        description=(
            "doc_ids of contracts the analysis recommends signing/accepting. "
            "Empty if none qualify. Reference rows by their ``doc_id`` "
            "column value."
        ),
    )
    not_endorsed: list[str] = OutputField(
        description=(
            "doc_ids of contracts the analysis recommends declining or "
            "renegotiating before signing."
        ),
    )
    narrative: str = OutputField(
        description=(
            "2-3 paragraph human-readable explanation of the "
            "recommendation, citing specific rows + scores by their "
            "doc_id."
        ),
    )


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------


class DealReviewResult(BaseModel):
    """Output of a :class:`DealReviewProgram` run.

    Carries every typed artifact the pipeline produced — schema,
    rubric, scored table — alongside the final recommendation. The
    artifacts are persistable as a "deal-room playbook" for replay
    against future corpora without re-paying the design cost.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    extraction_schema: ExtractionSchema = Field(
        description=(
            "The synthesized ExtractionSchema (Phase 1 output). Named "
            "``extraction_schema`` (not ``schema``) to avoid shadowing "
            "``BaseModel.schema``."
        ),
    )
    rubric: HybridRubric = Field(description="The synthesized HybridRubric (Phase 3 output).")
    scored_table_json: str = Field(
        description=(
            "The scored TabularDocument serialized as JSON. Round-trips "
            "via ``kaos_content.artifacts.store_tabular`` for persisted "
            "audit. Kept as JSON rather than the typed value to avoid a "
            "hard kaos-content dependency on the Program v3 envelope path."
        ),
    )
    endorsed: tuple[str, ...] = Field(
        description="doc_ids the analysis recommends signing/accepting."
    )
    not_endorsed: tuple[str, ...] = Field(
        description="doc_ids the analysis recommends declining or renegotiating."
    )
    narrative: str = Field(description="The human-readable recommendation paragraph.")


# ---------------------------------------------------------------------------
# DealReviewProgram — the orchestrator
# ---------------------------------------------------------------------------


class DealReviewProgram(Program):
    """Five-phase deal-review pipeline.

    Composes ``design_schema`` → ``extract_corpus`` → ``design_rubric``
    → ``apply_rubric`` → final ``Call(_RecommendationSignature)``.

    Args (to ``__init__``):
        designer_model: Model for the schema + rubric synthesizers.
            Defaults to ``anthropic:claude-sonnet-4-6``.
        extractor_model: Model passed to ``extract_corpus``. Defaults
            to the designer model.
        judge_model: Model for the rubric's soft channel. Defaults to
            the designer model.
        recommender_model: Model for the final synthesis Call. Defaults
            to the designer model.

    Args (to ``forward``):
        question: The user's review question.
        corpus: List of documents (typically ``ContentDocument``
            instances). Anything with a ``to_text`` method or string
            representation works.
        doc_ids: Optional explicit doc ids. If omitted, ``doc_<i>``
            is used. Must align with ``corpus`` by index.
        preferences: User's preferences in natural language. Optional.
        domain_hint: Domain context for ``design_schema``. Optional.
        schema_id: Stable id assigned to the synthesized schema.
            Default ``"deal-review"``.

    Returns:
        :class:`DealReviewResult` with all typed artifacts + final
        recommendation.
    """

    def __init__(
        self,
        *,
        designer_model: str = _DEFAULT_MODEL,
        extractor_model: str | None = None,
        judge_model: str | None = None,
        recommender_model: str | None = None,
    ) -> None:
        super().__init__()
        self.designer_model = designer_model
        self.extractor_model = extractor_model or designer_model
        self.judge_model = judge_model or designer_model
        self.recommender_model = recommender_model or designer_model

    async def forward(self, **inputs: Any) -> DealReviewResult:
        """Run the five-phase pipeline.

        Inputs (passed via ``__call__`` / ``invoke`` as keyword args):
            question: ``str`` — the user's review question (required).
            corpus: ``list`` — documents to review (required).
            doc_ids: ``list[str] | None`` — optional explicit ids.
            preferences: ``str`` — user preferences (default empty).
            domain_hint: ``str`` — domain context (default empty).
            schema_id: ``str`` — schema id (default ``"deal-review"``).
        """
        try:
            question: str = inputs["question"]
            corpus: list[Any] = inputs["corpus"]
        except KeyError as exc:
            raise ValueError(
                f"DealReviewProgram: missing required input {exc.args[0]!r}. "
                "Pass via __call__(question=..., corpus=...) or invoke(...)."
            ) from exc
        doc_ids: list[str] | None = inputs.get("doc_ids")
        preferences: str = inputs.get("preferences", "")
        domain_hint: str = inputs.get("domain_hint", "")
        schema_id: str = inputs.get("schema_id", "deal-review")

        # Resolve doc ids — used downstream for endorsed/not_endorsed
        # references that must round-trip back to specific documents.
        if doc_ids is None:
            doc_ids = [f"doc_{i}" for i in range(len(corpus))]
        if len(doc_ids) != len(corpus):
            raise ValueError(
                f"doc_ids has length {len(doc_ids)}, corpus has "
                f"length {len(corpus)}; they must match. How to fix: "
                "either omit doc_ids (auto-assigned) or pass one id per doc."
            )

        # Phase 1: design schema -------------------------------------
        sample = sample_corpus_text(corpus)
        schema = await design_schema(
            question=question,
            corpus_sample=sample,
            domain_hint=domain_hint,
            schema_id=schema_id,
            model=self.designer_model,
        )

        # Phase 2: extract -------------------------------------------
        # Lazy import — extract_corpus pulls kaos-content; keep the
        # Program class importable without that dep.
        from kaos_content.model.tabular import (
            Column,
            ColumnType,
            Table,
            TabularDocument,
        )

        from kaos_llm_core.programs.extract import Extract

        extractor = Extract(schema=schema, model=self.extractor_model, provenance="none")
        # Build a TabularDocument from per-doc extractions. We could
        # use ``extract_corpus`` for batched fan-out, but the prototype
        # found per-doc execution simpler to reason about and the cost
        # delta is negligible for <100-doc corpora. ``extract_corpus``
        # is the right call once corpus sizes grow.
        column_objs = (
            Column(name="doc_id", column_type=ColumnType.TEXT, nullable=False),
            *(
                Column(
                    name=spec.id,
                    column_type=_column_type_for(spec.column_type),
                    nullable=not spec.required,
                )
                for spec in schema.columns
            ),
        )
        rows: list[tuple[Any, ...]] = []
        for doc_id, doc in zip(doc_ids, corpus, strict=True):
            text = _doc_text(doc)
            result = await extractor(source_text=text)
            cell_lookup = {c.column_id: c.ai_value for c in result.cells}
            row = (doc_id, *(cell_lookup.get(spec.id) for spec in schema.columns))
            rows.append(row)

        table = Table(
            name="extracted",
            columns=column_objs,
            rows=tuple(rows),
            row_count=len(rows),
        )
        extracted_doc = TabularDocument(tables=(table,))

        # Phase 3: design rubric -------------------------------------
        rubric = await design_rubric(
            objective=question,
            schema=schema,
            preferences=preferences,
            model=self.designer_model,
        )

        # Phase 4: score ---------------------------------------------
        scored_doc = await apply_rubric(rubric, extracted_doc, judge_model=self.judge_model)

        # Phase 5: recommend -----------------------------------------
        scored_table = scored_doc.tables[0]
        rubric_summary = _summarize_rubric(rubric)
        scored_table_json = _serialize_table_for_recommendation(scored_table)

        recommend_call = Call(_RecommendationSignature, model=self.recommender_model)
        rec_output = await recommend_call(
            question=question,
            preferences=preferences or "(no explicit preferences)",
            scored_table_json=scored_table_json,
            rubric_summary=rubric_summary,
        )

        return DealReviewResult(
            extraction_schema=schema,
            rubric=rubric,
            scored_table_json=scored_table_json,
            endorsed=tuple(rec_output.endorsed),
            not_endorsed=tuple(rec_output.not_endorsed),
            narrative=rec_output.narrative,
        )


# ---------------------------------------------------------------------------
# Helpers (module-private)
# ---------------------------------------------------------------------------


def _doc_text(doc: Any) -> str:
    """Best-effort text extraction from any document-like object."""
    if hasattr(doc, "to_text"):
        return str(doc.to_text())
    if isinstance(doc, str):
        return doc
    return str(doc)


def _column_type_for(name: str) -> Any:
    """Map an ExtractionSchema column_type name to a kaos-content ColumnType.

    The schema designer emits a subset of the full ColumnTypeName
    vocabulary; this helper resolves each to its TabularDocument peer.
    Unknown names fall through to TEXT — they'll still extract fine
    but lose downstream type-aware features (sort/aggregate).
    """
    from kaos_content.model.tabular import ColumnType

    mapping = {
        "string": ColumnType.TEXT,
        "verbatim_quote": ColumnType.TEXT,
        "number": ColumnType.FLOAT,
        "integer": ColumnType.INTEGER,
        "date": ColumnType.DATE,
        "datetime": ColumnType.DATETIME,
        "boolean": ColumnType.BOOLEAN,
        "money": ColumnType.MONEY,
        "score": ColumnType.SCORE,
        "enum": ColumnType.TEXT,
        "list": ColumnType.LIST,
        "object": ColumnType.STRUCT,
        "entity_role": ColumnType.ENTITY_ROLE,
    }
    return mapping.get(name, ColumnType.TEXT)


def _summarize_rubric(rubric: HybridRubric) -> str:
    """Build a compact, prompt-friendly summary of a HybridRubric."""
    lines = [f"objective: {rubric.objective}"]
    if rubric.criteria:
        lines.append("criteria:")
        for c in rubric.criteria:
            lines.append(
                f"  - {c.column} {c.operator} {c.value!r} weight={c.weight:+.2f}"
                + (f" — {c.rationale}" if c.rationale else "")
            )
    if rubric.qualitative_guidance:
        lines.append(f"qualitative_guidance: {rubric.qualitative_guidance}")
    lines.append(f"weights: hard={rubric.hard_weight} soft={rubric.soft_weight}")
    return "\n".join(lines)


def _serialize_table_for_recommendation(table: Any) -> str:
    """Serialize a Table to a compact JSON-records string for the recommender."""
    import json

    col_names = [c.name for c in table.columns]
    records = [dict(zip(col_names, row, strict=False)) for row in table.rows]
    return json.dumps(records, default=str, separators=(",", ":"))


__all__ = [
    "DealReviewProgram",
    "DealReviewResult",
]
