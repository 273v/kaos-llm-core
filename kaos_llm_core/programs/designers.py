"""SchemaDesigner + RubricDesigner — runtime program synthesizers.

Agents call these to synthesize typed Programs at runtime instead of
reasoning over retrieved passages. ``design_schema`` produces an
:class:`~kaos_llm_core.signatures.extraction.ExtractionSchema` that
``extract_corpus`` then runs against a document corpus; ``design_rubric``
produces a :class:`~kaos_llm_core.signatures.rubric.HybridRubric` that
:func:`~kaos_llm_core.programs.scoring.apply_rubric` applies to the
extracted ``TabularDocument``.

The 2026-05-10 program-synthesis prototypes confirmed both designers are
stable enough as single ``Call`` invocations — sub-agent #1 observed
9 of 10 columns identical across 3 runs of the same question + corpus.
Refinement / critique wrappers are a future optimization (``Refine``
or ``BestOfN`` over the inner Call); they don't ship with the v1.

This module is the meta-programming surface: the LLM functions here
emit typed artifacts that other Programs consume. The artifacts are
JSON-round-trippable so they can be persisted as reusable "deal-room
playbooks" and re-applied to fresh corpora without re-paying the
design cost.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from kaos_llm_core.programs.call import Call
from kaos_llm_core.signatures.extraction import ColumnSpec, ExtractionSchema
from kaos_llm_core.signatures.fields import InputField, OutputField
from kaos_llm_core.signatures.rubric import HybridRubric
from kaos_llm_core.signatures.signature import Signature

_DEFAULT_DESIGNER_MODEL: str = "anthropic:claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# SchemaDesigner — synthesizes an ExtractionSchema from a question + corpus
# ---------------------------------------------------------------------------


# Subset of ``ColumnTypeName`` that the synthesizer is allowed to emit.
# Restricting the LLM's choice catches "freshly invented" column types
# at Pydantic decode time rather than silently in extraction. Full set:
# string / verbatim_quote / number / integer / date / datetime /
# boolean / money / score / enum / list / object / entity_role.
_DesignerColumnType = Literal[
    "string",
    "verbatim_quote",
    "number",
    "integer",
    "date",
    "datetime",
    "boolean",
    "money",
    "score",
    "enum",
]


class _ColumnProposal(BaseModel):
    """One column the designer proposes for the synthesized schema.

    Maps directly to a :class:`~kaos_llm_core.signatures.extraction.ColumnSpec`
    after validation — see ``_to_column_spec``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(
        min_length=1,
        pattern=r"^[a-z_][a-z0-9_]*$",
        description=(
            "Snake-case identifier. Must be a valid Python attribute name. "
            "Becomes the column id in the downstream TabularDocument."
        ),
    )
    column_type: _DesignerColumnType = Field(
        description=(
            "Logical column type from the supported set. Pick the most "
            "specific type: prefer 'integer'/'number' over 'string' for "
            "numeric values, 'date'/'datetime' over 'string' for dates."
        ),
    )
    description: str = Field(
        min_length=1,
        description=(
            "Natural-language description shown to the extraction LLM. "
            "Be specific: 'The contract's effective date in ISO 8601 "
            "format' rather than 'date'."
        ),
    )
    required: bool = Field(
        default=True,
        description=(
            "True if every document MUST have this field. False if the "
            "extractor may emit null when the document doesn't carry it."
        ),
    )


class SchemaDesignerSignature(Signature):
    """Synthesize an ExtractionSchema from a question + corpus sample."""

    question: str = InputField(
        description=(
            "The user's question or objective for the review. Drives "
            "which columns the schema needs."
        ),
    )
    corpus_sample: str = InputField(
        description=(
            "Concatenated samples (typically the first ~500 chars of "
            "each document) to ground the column proposals in the "
            "corpus's actual content and variation."
        ),
    )
    domain_hint: str = InputField(
        description=(
            "Domain context: e.g. 'mutual NDAs', 'commercial real estate "
            "leases', 'sponsorship agreements'. Empty string is allowed."
        ),
    )
    columns: list[_ColumnProposal] = OutputField(
        description=(
            "Ordered list of column proposals. Include only columns "
            "relevant to answering the user's question; do NOT propose "
            "columns the question doesn't ask about. Typically 5-12 "
            "columns for legal review; more for forensic extraction."
        ),
    )


async def design_schema(
    question: str,
    corpus_sample: str,
    domain_hint: str = "",
    *,
    schema_id: str = "synthesized",
    model: str = _DEFAULT_DESIGNER_MODEL,
) -> ExtractionSchema:
    """Synthesize an ``ExtractionSchema`` for the given review question.

    Args:
        question: The user's review question or objective.
        corpus_sample: Concatenated samples (typically ~500 chars from
            each document) so the designer sees corpus variation.
        domain_hint: Optional domain context. Defaults to empty.
        schema_id: Stable id assigned to the resulting schema. Defaults
            to ``"synthesized"`` — callers reviewing distinct deals
            should pass a unique id (e.g., deal name) so downstream
            cell tags remain auditable.
        model: Designer model. Defaults to
            ``anthropic:claude-sonnet-4-6`` — the platform's research-
            grade default. Cheap models under-design columns.

    Returns:
        A typed ``ExtractionSchema`` ready for ``Extract`` or
        ``extract_corpus``.

    Cost: ~$0.02 per call at Sonnet 4.6 for 5-doc legal corpora.
    Sub-agent #1's 2026-05-10 prototype observed 9 of 10 columns
    identical across 3 runs of the same question/corpus — synthesis
    is stable enough that a single Call suffices.
    """
    call = Call(SchemaDesignerSignature, model=model)
    output = await call(
        question=question,
        corpus_sample=corpus_sample,
        domain_hint=domain_hint,
    )

    column_specs = tuple(
        ColumnSpec(
            id=p.id,
            label=p.id.replace("_", " ").title(),
            column_type=p.column_type,
            description=p.description,
            required=p.required,
        )
        for p in output.columns
    )
    return ExtractionSchema(id=schema_id, version=1, columns=column_specs)


# ---------------------------------------------------------------------------
# RubricDesigner — synthesizes a HybridRubric from a schema + preferences
# ---------------------------------------------------------------------------


class RubricDesignerSignature(Signature):
    """Synthesize a HybridRubric from an ExtractionSchema + user preferences.

    Outputs a :class:`HybridRubric` directly — the LLM's structured
    output is the typed value, no re-interpretation needed.
    """

    objective: str = InputField(
        description="The user's objective — what the rubric optimises for.",
    )
    extraction_schema_json: str = InputField(
        description=(
            "The ExtractionSchema (as JSON) that was used to extract the "
            "data this rubric will score. Criteria reference column ids "
            "from this schema. (Named ``extraction_schema_json`` rather "
            "than ``schema_json`` to avoid shadowing ``BaseModel.schema``.)"
        ),
    )
    preferences: str = InputField(
        description=(
            "User's stated preferences, in natural language. E.g. "
            "'prefer Michigan law, terms ≤ 2 years, avoid non-solicit "
            "clauses'."
        ),
    )
    rubric: HybridRubric = OutputField(
        description=(
            "The synthesized rubric. Criteria must reference columns that "
            "exist in the schema. Weights should reflect preference "
            "strength (positive for preferred, negative for penalised). "
            "Use the qualitative_guidance channel for residual judgment "
            "that doesn't reduce to typed criteria. hard_weight + "
            "soft_weight must equal 1.0."
        ),
    )


async def design_rubric(
    objective: str,
    schema: ExtractionSchema,
    preferences: str,
    *,
    model: str = _DEFAULT_DESIGNER_MODEL,
) -> HybridRubric:
    """Synthesize a ``HybridRubric`` from a schema + user preferences.

    Args:
        objective: One-sentence statement of what the rubric optimises
            for. Will be carried on the rubric for the per-row
            reasoning trail.
        schema: The schema whose columns the criteria will reference.
        preferences: User's preferences in natural language.
        model: Designer model. Defaults to Sonnet 4.6.

    Returns:
        A typed ``HybridRubric``. Validation (hard+soft sum to 1.0,
        at least one of criteria/qualitative_guidance) is enforced by
        the type itself.

    Cost: ~$0.02 per call at Sonnet 4.6.
    """
    call = Call(RubricDesignerSignature, model=model)
    output = await call(
        objective=objective,
        extraction_schema_json=schema.model_dump_json(),
        preferences=preferences,
    )
    return output.rubric


# ---------------------------------------------------------------------------
# Helpers — corpus sampling for design_schema
# ---------------------------------------------------------------------------


def sample_corpus_text(
    docs: list[Any],
    *,
    chars_per_doc: int = 500,
    separator: str = "\n\n---\n\n",
) -> str:
    """Build a corpus_sample string from a list of ContentDocuments.

    Sub-agent #1 finding: passing the first ~500 chars of EVERY doc
    (not the full text of one doc) lets the designer see corpus
    variation. Cheap, generalises to heterogeneous corpora.

    Args:
        docs: List of objects with a ``model_dump_json`` method or a
            ``__str__`` representation. Most commonly
            ``ContentDocument`` instances after ``parse_docx`` /
            ``parse_pdf`` /  ``parse_markdown``.
        chars_per_doc: Truncate each doc's text to this many leading
            characters. 500 is the prototype-validated default.
        separator: Joining string between samples.

    Returns:
        The concatenated sample string.
    """
    samples = []
    for doc in docs:
        # Best-effort text extraction without taking a hard dep on
        # kaos-content (the consumer may not have it installed).
        text = ""
        if hasattr(doc, "to_text"):
            text = str(doc.to_text())
        elif hasattr(doc, "model_dump_json"):
            text = doc.model_dump_json()
        else:
            text = str(doc)
        samples.append(text[:chars_per_doc])
    return separator.join(samples)


__all__ = [
    "RubricDesignerSignature",
    "SchemaDesignerSignature",
    "design_rubric",
    "design_schema",
    "sample_corpus_text",
]
