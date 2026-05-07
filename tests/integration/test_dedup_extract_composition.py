"""INTEG-3: Dedup → Extract composition test.

Proves the full ingest → dedup → extract → aggregate pipeline works
with duplicates removed BEFORE LLM extraction (saving cost and
preventing double-counted rows).

Test scenario:
- 8 documents total: 5 unique + 3 exact duplicates
- Dedup pipeline (text_hash level) should remove 3 dups
- Extract runs on 5 unique docs only
- TabularDocument has 5 rows, not 8

This is the first test that composes kaos-content's dedup pipeline
with kaos-llm-core's Extract program.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

FIXTURE_BASE = Path(__file__).parent.parent / "fixtures"


def _load_text_docs() -> list[tuple[str, str]]:
    """Load (doc_id, text) pairs from the grounding corpus.

    Then add 3 exact duplicates to test dedup.
    """
    corpus_dir = FIXTURE_BASE / "grounding-corpus"
    docs: list[tuple[str, str]] = []
    for f in sorted(corpus_dir.glob("*.txt")):
        text = f.read_text(errors="replace")
        if len(text) > 50:
            docs.append((f.stem, text))

    unique_count = len(docs)
    # Add 3 exact duplicates of the first 3 docs
    for i in range(min(3, unique_count)):
        docs.append((f"dup_{docs[i][0]}", docs[i][1]))

    return docs


@pytest.mark.live
@pytest.mark.asyncio
async def test_dedup_then_extract() -> None:
    """Dedup removes duplicates → Extract runs on unique only → correct row count."""
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("KAOS_LLM_ANTHROPIC_API_KEY")):
        pytest.skip("ANTHROPIC_API_KEY not set")

    from kaos_content.dedup import DedupDocument, DedupPipeline
    from kaos_content.dedup.levels import TextHashLevel
    from kaos_content.dedup.pipeline import DedupPipelineConfig

    from kaos_llm_core.programs.extract import Extract
    from kaos_llm_core.signatures.extraction import ExtractionSchema

    all_docs = _load_text_docs()
    total_with_dups = len(all_docs)

    # Step 1: Dedup
    dedup_docs = [DedupDocument(doc_id=doc_id, text=text) for doc_id, text in all_docs]
    config = DedupPipelineConfig(levels=(TextHashLevel(),))
    report = DedupPipeline(config).run(dedup_docs)

    print(
        f"\nDedup: {report.total_input} input → {report.total_unique} unique "
        f"({report.total_duplicates} removed)"
    )
    assert report.total_duplicates >= 3, (
        f"Expected >=3 duplicates removed, got {report.total_duplicates}"
    )

    # Step 2: Filter to unique docs only (singletons + canonicals)
    unique_ids = set(report.singletons)
    for cluster in report.clusters:
        unique_ids.add(cluster.canonical_doc_id)

    unique_texts = {doc_id: text for doc_id, text in all_docs if doc_id in unique_ids}

    # Step 3: Extract on unique docs
    schema = ExtractionSchema.from_dict(
        {
            "id": "dedup-extract-test-v1",
            "columns": [
                {
                    "id": "document_type",
                    "column_type": "enum",
                    "description": (
                        "Type of document: 'legal', 'technical', 'scientific', or 'other'."
                    ),
                    "constraints": {"values": ["legal", "technical", "scientific", "other"]},
                },
                {
                    "id": "primary_topic",
                    "column_type": "string",
                    "description": "Main subject in 5-10 words.",
                },
            ],
        }
    )

    all_cells: list[Any] = []
    for doc_id, text in unique_texts.items():
        extract = Extract(schema, model="anthropic:claude-haiku-4-5", provenance="none")
        inv = await extract.invoke(source_text=text[:15000], doc_id=doc_id)
        all_cells.extend(inv.output.cells)

    # Step 4: Aggregate into TabularDocument
    from kaos_content.model.tabular import ColumnType, TabularDocument

    tabular = TabularDocument.from_cells(
        all_cells,
        column_specs=(
            ("document_type", ColumnType.TEXT),
            ("primary_topic", ColumnType.TEXT),
        ),
        table_name="dedup_extract",
    )

    table = tabular.tables[0]
    print(f"Extract: {len(unique_texts)} unique docs → {table.row_count} rows")
    assert table.row_count == len(unique_texts), (
        f"Row count {table.row_count} != unique doc count {len(unique_texts)}"
    )
    assert table.row_count < total_with_dups, (
        f"Dedup should have reduced row count: {table.row_count} vs {total_with_dups} total"
    )

    # Verify dedup lineage is available for audit
    assert len(report.clusters) >= 1
    for cluster in report.clusters:
        assert cluster.size >= 2
        print(f"  Dedup cluster: {cluster.canonical_doc_id} + {cluster.duplicate_doc_ids}")
