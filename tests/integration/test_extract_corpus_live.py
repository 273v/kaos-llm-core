"""Live integration tests for extract_corpus — WS-TR.PR-3 acceptance.

Runs the full corpus fan-out pipeline (ExtractionSchema → Extract →
batch_run → JSONL round-trip → TabularDocument assembly) against a real
provider with a small 3-doc mixed-format corpus carved from the WS-3.7
multiformat fixture.

Per CLAUDE.md: live tests are the acceptance gate. Unit tests alone are
insufficient.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kaos_llm_core.programs.extract import CorpusExtractionResult, extract_corpus
from kaos_llm_core.signatures.extraction import ExtractionSchema

requires_anthropic = pytest.mark.skipif(
    not (os.getenv("KAOS_LLM_ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")),
    reason="Anthropic API key not set",
)


# Three documents from the multiformat fixture, each with known content we can
# verify extraction against. Kept small so the live test cost is negligible.
_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "multiformat-corpus"


def _load_text_corpus() -> dict[str, str]:
    """Load a 3-doc dict corpus from the multiformat fixture.

    Only the plain-text docs are used here so the test doesn't depend on
    optional parsing extras (kaos-pdf, kaos-office, kaos-web) being
    installed in the integration-test environment.
    """
    docs = {
        "rfc_2119": (_FIXTURE_DIR / "rfc_2119.txt").read_text(encoding="utf-8"),
        "nist_passwords": (_FIXTURE_DIR / "nist_passwords.txt").read_text(encoding="utf-8"),
        "pfas_summary": (_FIXTURE_DIR / "federal_register_pfas_summary.md").read_text(
            encoding="utf-8"
        ),
    }
    return docs


# Small extraction schema spanning 3 columns. Each column is answerable from
# every document in the fixture (broadly enough that we get a clean signal).
CORPUS_SCHEMA = ExtractionSchema.from_dict(
    {
        "id": "document-metadata-smoke",
        "version": 1,
        "columns": [
            {
                "id": "document_type",
                "column_type": "string",
                "description": (
                    "The general document type (e.g., 'RFC', 'NIST Special Publication', "
                    "'Federal Register notice')."
                ),
            },
            {
                "id": "primary_topic",
                "column_type": "string",
                "description": "One short phrase describing the primary topic.",
            },
            {
                "id": "publishing_organization",
                "column_type": "string",
                "description": (
                    "The organization that authored/published this document "
                    "(e.g., 'IETF', 'NIST', 'US EPA')."
                ),
            },
        ],
    }
)


@pytest.mark.integration
@requires_anthropic
class TestExtractCorpusLive:
    async def test_3_doc_corpus_end_to_end(self, tmp_path: Path) -> None:
        """Fan out Extract over a 3-doc corpus and assert each doc's row."""
        corpus = _load_text_corpus()

        result = await extract_corpus(
            CORPUS_SCHEMA,
            corpus,
            output_dir=str(tmp_path / "corpus-run"),
            model="anthropic:claude-haiku-4-5",
            provenance="none",
            max_concurrency=3,
        )

        assert isinstance(result, CorpusExtractionResult)
        assert result.schema_id == "document-metadata-smoke"
        assert result.n_docs_processed == 3
        assert result.n_docs_errored == 0

        # Verify the TabularDocument has one row per doc with expected content.
        assert len(result.table.tables) == 1
        table = result.table.tables[0]
        assert table.row_count == 3
        assert table.column_names() == (
            "doc_id",
            "document_type",
            "primary_topic",
            "publishing_organization",
        )

        rows_by_doc = {row[0]: row for row in table.rows}
        # RFC 2119 — IETF document about requirement-level keywords.
        rfc_row = rows_by_doc["rfc_2119"]
        assert "RFC" in str(rfc_row[1])
        assert "IETF" in str(rfc_row[3]) or "Internet" in str(rfc_row[3])

        # NIST passwords — NIST SP about authenticators / memorized secrets.
        nist_row = rows_by_doc["nist_passwords"]
        assert "NIST" in str(nist_row[3])

        # PFAS summary — EPA Federal Register notice about PFAS.
        pfas_row = rows_by_doc["pfas_summary"]
        assert "EPA" in str(pfas_row[3]) or "Environmental" in str(pfas_row[3])

        # Cost + latency sanity.
        assert result.cost_usd >= 0.0
        assert result.tokens_total > 0

    async def test_resume_round_trip_live(self, tmp_path: Path) -> None:
        """Run once, resume — second invocation skips all docs.

        Covers the batch_run resume contract end-to-end with a real LLM
        call path. The first run exhausts the fixture; the second run
        must detect the existing items.jsonl + matching program_hash and
        skip every item.
        """
        corpus = _load_text_corpus()
        output_dir = str(tmp_path / "resume-live")

        # Run 1 — processes all 3 docs.
        result1 = await extract_corpus(
            CORPUS_SCHEMA,
            corpus,
            output_dir=output_dir,
            model="anthropic:claude-haiku-4-5",
            provenance="none",
            max_concurrency=3,
        )
        assert result1.n_docs_processed == 3

        # Run 2 — resume with the same schema + output_dir. Zero new LLM
        # calls should happen (delta-cost should be ~0 modulo rounding).
        result2 = await extract_corpus(
            CORPUS_SCHEMA,
            corpus,
            output_dir=output_dir,
            model="anthropic:claude-haiku-4-5",
            provenance="none",
            max_concurrency=3,
        )
        assert result2.n_docs_skipped == 3
        # Hydrated result still has all 3 rows (resume reads the full log).
        assert result2.n_docs_processed == 3
        # Same extracted values — byte-identical extractions from the log.
        table1_rows = sorted(result1.table.tables[0].rows)
        table2_rows = sorted(result2.table.tables[0].rows)
        assert table1_rows == table2_rows
