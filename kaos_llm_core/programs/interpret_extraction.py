"""InterpretExtraction — the synthesizer that reformulates typed extraction
rows into the deliverable shape the user actually asked for.

A complement to :func:`~kaos_llm_core.programs.designers.design_schema` +
the per-document extraction fan-out: the designer + extractor produce
typed grounded rows; this Signature reformulates them into a memo,
table, comparison, or whatever the user's prompt implied.

Two key properties separate this from a free-form LLM rewrite:

1. **Bounded by the extraction.** The synthesizer only sees the typed
   rows in its ``extracted_rows`` input. It cannot fabricate facts that
   aren't in those rows (or if it does, the contract makes the
   fabrication auditable — every cited claim points back to a row).

2. **Iterative-aware.** When the synthesizer identifies that the
   current rows are insufficient to fully answer the user's question,
   it sets ``needs_more_extraction=True`` and proposes specific column
   ids in ``requested_columns``. The dispatcher loop (which lives in
   the caller — typically a kaos-agents tool) takes those proposals,
   augments the schema, re-extracts, and re-invokes the synthesizer.
   See the iterative ReAct pattern in ``kaos-agents``' design plan.

This is the "interpret" half of the dynamic-deliverable-schema
architecture. The "extract" half lives in
:mod:`~kaos_llm_core.programs.designers` + the kaos-agents
``kaos-agent-design-extraction`` tool.
"""

from __future__ import annotations

from kaos_llm_core.signatures.fields import InputField, OutputField
from kaos_llm_core.signatures.signature import Signature


class InterpretExtractionSignature(Signature):
    """Draft a memo answering the user's question, grounded on EXTRACTED_ROWS.

    EXTRACTED_ROWS is a typed table — one row per source document, with
    named columns of verified values. Every substantive claim in your
    memo MUST trace to a specific (row.document, column) cell. Do NOT
    introduce facts that aren't in EXTRACTED_ROWS.

    When the user's question requires data that EXTRACTED_ROWS does NOT
    contain, set ``needs_more_extraction=true`` and list specific column
    proposals in ``requested_columns``. Each proposal is one string in
    the form ``"<column_id>: <one-sentence description>"`` so the
    downstream extractor knows exactly what to fetch.

    When the existing rows are sufficient to fully answer the user's
    question, set ``needs_more_extraction=false`` and produce the
    final memo. Do NOT request more extractions when:

    - the gap is unanswerable from the source documents
    - the user's question is already fully answered
    - the requested column would duplicate an existing one
    - the loop has reached its iteration cap (see ``iteration``)

    Write the memo in the user's preferred shape (infer from the
    question — table, narrative, exec summary, comparison, etc.). Use
    rich markdown. Cite each substantive claim by document name in
    parentheses.
    """

    user_question: str = InputField(
        description="The user's original prompt that the memo must answer.",
    )
    extracted_rows: str = InputField(
        description=(
            "JSON-serialized typed extraction rows. Shape: "
            '{"columns": [{"id": "...", "description": "..."}, ...], '
            '"rows": [{"document": "<filename>", "cells": {"<col_id>": '
            "<value>, ...}}, ...]}. This is the only source of facts "
            "you may cite — nothing else."
        ),
    )
    deliverable_hint: str = InputField(
        default="",
        description=(
            "Optional caller-supplied hint about the deliverable shape, "
            'e.g. "one-page exec summary for non-lawyer CEO", '
            '"CSV-ready table", "compare A and B". Empty when no hint.'
        ),
    )
    iteration: int = InputField(
        default=1,
        ge=1,
        description=(
            "Which iteration of the extract↔interpret loop this is "
            "(1-indexed). When the caller signals this is the LAST "
            "iteration (typically by passing iteration >= some cap), "
            "set ``needs_more_extraction=false`` regardless of gaps."
        ),
    )

    memo: str = OutputField(
        description=(
            "The grounded memo, markdown-formatted, with parenthetical "
            "document citations on every substantive claim."
        ),
    )
    score: int = OutputField(
        default=7,
        ge=1,
        le=10,
        description=(
            "Self-rated 1-10 completeness + accuracy of the memo given "
            "the rows currently available in EXTRACTED_ROWS."
        ),
    )
    needs_more_extraction: bool = OutputField(
        default=False,
        description=(
            "True only when adding columns to EXTRACTED_ROWS would "
            "meaningfully improve the memo. False when the existing "
            "rows are sufficient OR the gap is unanswerable from source."
        ),
    )
    requested_columns: tuple[str, ...] = OutputField(
        default=(),
        description=(
            "Column proposals when ``needs_more_extraction=true``. "
            'Format: ``"<column_id>: <one-sentence description>"``. '
            "Empty when ``needs_more_extraction=false``. Maximum 5 "
            "columns per iteration."
        ),
    )
