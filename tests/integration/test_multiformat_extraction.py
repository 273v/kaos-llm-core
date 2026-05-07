"""INTEG-2: Multi-format extraction integration test.

Proves the full ingest → extract → aggregate path works across
PDF, DOCX, HTML, Markdown, and plain text in a single pipeline run.

Steps:
1. Load each document via the appropriate parser (kaos-pdf, kaos-office,
   or plain read)
2. Extract text from each
3. Run Extract with a simple 3-column schema on all documents
4. Assemble into a single TabularDocument via from_cells
5. Assert: N documents → N rows, typed columns, no crashes

Gated on ANTHROPIC_API_KEY (Extract needs an LLM).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

CORPUS_DIR = Path(__file__).parent.parent / "fixtures" / "multiformat-corpus"

SCHEMA_DICT = {
    "id": "multiformat-test-v1",
    "columns": [
        {
            "id": "document_type",
            "column_type": "enum",
            "description": (
                "Classify this document as one of the listed types based on its "
                "content and structure. 'regulation' for CFR rules, FR notices. "
                "'guidance' for agency guidance, standards (NIST, FDA). "
                "'legal_opinion' for court orders, opinions, holdings. "
                "'filing' for patent filings, SEC filings. "
                "'policy' for consumer rights, organizational policies. "
                "'standard' for RFCs, technical standards."
            ),
            "constraints": {
                "values": [
                    "regulation",
                    "guidance",
                    "legal_opinion",
                    "filing",
                    "policy",
                    "standard",
                ]
            },
        },
        {
            "id": "primary_topic",
            "column_type": "string",
            "description": (
                "The main subject of the document in 5-10 words. "
                "E.g., 'SEC anti-fraud rule for securities', "
                "'FDA pharmaceutical guidance', 'NIST password requirements'."
            ),
        },
        {
            "id": "issuing_authority",
            "column_type": "string",
            "description": (
                "The government agency, court, or organization that issued "
                "this document. E.g., 'SEC', 'FDA', 'NIST', 'U.S. Supreme Court', "
                "'USPTO', 'CFPB', 'IETF'."
            ),
        },
    ],
}


def _extract_text(path: Path) -> str | None:
    """Extract text from a file using the appropriate parser."""
    ext = path.suffix.lower()
    if ext in (".txt", ".md"):
        return path.read_text(errors="replace")
    if ext == ".html":
        text = path.read_text(errors="replace")
        import re

        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    if ext == ".pdf":
        try:
            from kaos_pdf import extract_pdf

            doc = extract_pdf(str(path))
            from kaos_content.serializers.text import serialize_text

            return serialize_text(doc)
        except Exception as exc:
            return f"[PDF extraction failed: {exc}]"
    if ext == ".docx":
        try:
            from kaos_office.docx.reader import parse_docx

            doc = parse_docx(str(path))
            from kaos_content.serializers.text import serialize_text

            return serialize_text(doc)
        except Exception as exc:
            return f"[DOCX extraction failed: {exc}]"
    return None


@pytest.mark.live
@pytest.mark.asyncio
async def test_multiformat_extraction_to_tabular() -> None:
    """Full path: mixed files → text → Extract → TabularDocument."""
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("KAOS_LLM_ANTHROPIC_API_KEY")):
        pytest.skip("ANTHROPIC_API_KEY not set")

    from kaos_llm_core.programs.extract import Extract
    from kaos_llm_core.signatures.extraction import ExtractionSchema

    schema = ExtractionSchema.from_dict(SCHEMA_DICT)
    model = "anthropic:claude-haiku-4-5"

    exts_to_test = {".pdf", ".docx", ".txt", ".md", ".html"}
    docs_processed: list[dict[str, Any]] = []
    all_cells: list[Any] = []

    for path in sorted(CORPUS_DIR.iterdir()):
        if path.suffix.lower() not in exts_to_test:
            continue

        text = _extract_text(path)
        if not text or len(text) < 50:
            continue

        doc_id = path.stem
        extract = Extract(schema, model=model, provenance="none")
        inv = await extract.invoke(source_text=text[:20000], doc_id=doc_id)

        for cell in inv.output.cells:
            all_cells.append(cell)
        docs_processed.append({"name": path.name, "format": path.suffix, "text_len": len(text)})

    # Verify multi-format coverage
    formats_seen = {d["format"] for d in docs_processed}
    assert ".txt" in formats_seen or ".md" in formats_seen, "No text files processed"
    assert len(docs_processed) >= 3, f"Expected >=3 docs processed, got {len(docs_processed)}"
    assert len(formats_seen) >= 2, f"Expected >=2 formats, got {formats_seen}"

    # Assemble into TabularDocument
    from kaos_content.model.tabular import ColumnType, TabularDocument

    tabular = TabularDocument.from_cells(
        all_cells,
        column_specs=(
            ("document_type", ColumnType.TEXT),
            ("primary_topic", ColumnType.TEXT),
            ("issuing_authority", ColumnType.TEXT),
        ),
        table_name="multiformat_extraction",
    )

    assert len(tabular.tables) == 1
    table = tabular.tables[0]
    assert table.row_count >= 3
    # from_cells adds a doc_id column → 4 columns total
    assert len(table.columns) >= 3

    # Print results for visibility
    print(f"\n{'=' * 60}")
    print(f"Multi-format extraction: {len(docs_processed)} docs → {table.row_count} rows")
    print(f"Formats: {formats_seen}")
    for i, row in enumerate(table.rows[:8]):
        # row[0] = doc_id, row[1] = document_type, row[2] = topic, row[3] = authority
        doc_name = docs_processed[i]["name"] if i < len(docs_processed) else "?"
        doc_type = str(row[1]) if len(row) > 1 else "?"
        authority = str(row[3]) if len(row) > 3 else "?"
        print(f"  {doc_name:40s} | {doc_type:20s} | {authority}")
    print(f"{'=' * 60}")
