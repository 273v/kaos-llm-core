"""Unit tests for extract_corpus — corpus fan-out via batch_run.

Exercises the WS-TR.PR-3 corpus primitive: a dict corpus fans out over
``Extract``, results round-trip through the JSONL batch log, and a
``TabularDocument`` is assembled from the per-doc cells. All with a
FunctionClient (no HTTP).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart

from kaos_llm_core.programs.extract import (
    CorpusExtractionResult,
    CorpusInputSource,
    Extract,
    ExtractionResult,
    extract_corpus,
)
from kaos_llm_core.signatures.extraction import ExtractionSchema


def _json_client(payloads: dict[str, dict[str, Any]]) -> FunctionClient:
    """FunctionClient that returns a different payload per call by index.

    ``payloads`` is a dict keyed by call-index as a string ("0", "1", ...).
    The FunctionClient doesn't know which BatchItem it's processing, so we
    return payloads in insertion order, cycling back to the last one when
    we run out (defensive).
    """
    ordered = list(payloads.values())
    call_ix = [0]

    def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
        idx = min(call_ix[0], len(ordered) - 1)
        call_ix[0] += 1
        payload = ordered[idx]
        return ProviderResponse.model_construct(
            provider="function",
            model="function-test",
            raw={},
            parts=[
                ContentPart.model_construct(
                    type="text",
                    text=json.dumps(payload),
                )
            ],
            usage=UsageInfo.model_construct(input_tokens=10, output_tokens=5, total_tokens=15),
            stop_reason="end_turn",
            status_code=200,
            response_headers={},
        )

    return FunctionClient(function=fn)


@pytest.mark.asyncio
class TestCorpusInputSource:
    async def test_iterates_dict_corpus_as_batch_items(self) -> None:
        corpus = {"doc-1": "first text", "doc-2": "second text"}
        source = CorpusInputSource(corpus)

        items = [item async for item in source]
        assert len(items) == 2
        assert items[0].custom_id == "doc-1"
        assert items[0].inputs == {"source_text": "first text", "doc_id": "doc-1"}
        assert items[1].custom_id == "doc-2"
        assert items[1].inputs == {"source_text": "second text", "doc_id": "doc-2"}

    async def test_describe_reports_size(self) -> None:
        source = CorpusInputSource({"a": "x", "b": "y", "c": "z"})
        described = source.describe()
        assert described["n_docs"] == 3
        assert "kaos_tr" in described["type"]

    async def test_iteration_is_deterministic(self) -> None:
        """Order is critical for resume — two iterations yield the same order."""
        corpus = {"doc-1": "a", "doc-2": "b", "doc-3": "c"}
        source = CorpusInputSource(corpus)
        run1 = [i.custom_id async for i in source]
        run2 = [i.custom_id async for i in source]
        assert run1 == run2


@pytest.mark.asyncio
class TestExtractCorpus:
    async def test_basic_dict_corpus(self, tmp_path: Path) -> None:
        schema = ExtractionSchema.from_dict(
            {
                "id": "tiny-schema",
                "columns": [{"id": "name", "column_type": "string"}],
            }
        )
        corpus = {"doc-1": "First doc about Acme.", "doc-2": "Second doc about Beta."}
        client = _json_client(
            {
                "doc-1": {"name": "Acme"},
                "doc-2": {"name": "Beta"},
            }
        )

        # Pre-build the Extract so we can inject the fake client. extract_corpus
        # would build its own, so we call batch_run manually instead for this
        # test; the direct-batch path is what extract_corpus composes.
        extractor = Extract(schema, model="function:test", provenance="none")
        extractor.call._client = client

        from kaos_llm_core.programs.batch import batch_run

        source = CorpusInputSource(corpus)
        batch_result = await batch_run(
            extractor,
            source,
            output_dir=str(tmp_path / "run-1"),
            max_concurrency=2,
        )
        assert batch_result.n_succeeded == 2
        assert batch_result.n_errored == 0

        # Verify JSONL log + hydration.
        log_lines = [
            json.loads(line)
            for line in (tmp_path / "run-1" / "items.jsonl").read_text().splitlines()
            if line.strip()
        ]
        item_records = [r for r in log_lines if r.get("kind") == "item"]
        assert len(item_records) == 2
        # The output field is a serialized ExtractionResult.
        for r in item_records:
            result = ExtractionResult.model_validate(r["output"])
            assert result.doc_id in corpus
            assert len(result.cells) == 1

    async def test_round_trip_through_jsonl(self, tmp_path: Path) -> None:
        """Cells with the full 4-state machine round-trip via model_dump()."""
        schema = ExtractionSchema.from_dict(
            {
                "id": "rt",
                "columns": [
                    {"id": "name", "column_type": "string"},
                    {"id": "optional_f", "column_type": "string", "required": False},
                ],
            }
        )
        # Model answers the required field but not the optional one.
        corpus = {"doc-1": "contents"}
        client = _json_client({"doc-1": {"name": "Acme", "optional_f": None}})

        extractor = Extract(schema, model="function:test", provenance="none")
        extractor.call._client = client

        from kaos_llm_core.programs.batch import batch_run

        await batch_run(
            extractor,
            CorpusInputSource(corpus),
            output_dir=str(tmp_path / "rt-run"),
            max_concurrency=1,
        )

        records = [
            json.loads(line)
            for line in (tmp_path / "rt-run" / "items.jsonl").read_text().splitlines()
            if line.strip()
        ]
        item = next(r for r in records if r.get("kind") == "item")
        result = ExtractionResult.model_validate(item["output"])
        # The required cell is extracted, the optional one is
        # not_in_document.
        by_id = {c.column_id: c for c in result.cells}
        assert by_id["name"].status == "extracted"
        assert by_id["name"].ai_value == "Acme"
        assert by_id["optional_f"].status == "not_in_document"

    async def test_extract_corpus_builds_tabular_document(self, tmp_path: Path) -> None:
        """End-to-end: dict corpus → extract_corpus → TabularDocument.

        extract_corpus constructs its own Extract instance, so we patch the
        module-level Call class to inject the FunctionClient via a
        monkeypatch-style override instead. For this test we stub batch_run
        to prove the assembly path.
        """
        schema = ExtractionSchema.from_dict(
            {
                "id": "small",
                "columns": [
                    {"id": "name", "column_type": "string"},
                    {"id": "amount", "column_type": "integer"},
                ],
            }
        )
        client = _json_client(
            {
                "doc-1": {"name": "Acme", "amount": 100},
                "doc-2": {"name": "Beta", "amount": 200},
            }
        )
        corpus = {"doc-1": "first", "doc-2": "second"}

        # Monkey-patch the Extract.__init__ to inject our client after construction.
        # Simplest path: run extract_corpus, then replace the extractor's client
        # via a subclass.
        import kaos_llm_core.programs.extract as extract_mod

        original_extract = extract_mod.Extract

        class _StubExtract(original_extract):  # type: ignore[misc]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self.call._client = client  # type: ignore[attr-defined]

        # ty: assigning a subclass to the module attribute is safe here but
        # fails static analysis. Use setattr to sidestep the invalid-assignment
        # rule without a blanket ignore.
        setattr(extract_mod, "Extract", _StubExtract)  # noqa: B010
        try:
            result = await extract_corpus(
                schema,
                corpus,
                output_dir=str(tmp_path / "tab-run"),
                model="function:test",
                provenance="none",
                max_concurrency=2,
            )
        finally:
            setattr(extract_mod, "Extract", original_extract)  # noqa: B010

        assert isinstance(result, CorpusExtractionResult)
        assert result.n_docs_processed == 2
        assert result.n_docs_errored == 0
        assert result.schema_id == "small"
        assert len(result.rows) == 2

        # TabularDocument assembly — one Table with two rows.
        from kaos_content.model.tabular import ColumnType, TabularDocument

        assert isinstance(result.table, TabularDocument)
        assert len(result.table.tables) == 1
        table = result.table.tables[0]
        assert table.name == "small"
        # doc_id + name + amount
        assert table.column_names() == ("doc_id", "name", "amount")
        # ColumnType for the 'amount' column should be INTEGER, not widened.
        types = {c.name: c.column_type for c in table.columns}
        assert types["name"] == ColumnType.TEXT
        assert types["amount"] == ColumnType.INTEGER
        # Rows carry the extracted values.
        rows = sorted(table.rows)
        assert rows[0] == ("doc-1", "Acme", 100)
        assert rows[1] == ("doc-2", "Beta", 200)

    async def test_resume_skips_completed_docs(self, tmp_path: Path) -> None:
        """Run once, then resume — the second run must skip completed items."""
        schema = ExtractionSchema.from_dict(
            {"id": "resumable", "columns": [{"id": "name", "column_type": "string"}]}
        )
        corpus = {"doc-1": "a", "doc-2": "b", "doc-3": "c"}
        client = _json_client(
            {
                "doc-1": {"name": "Acme"},
                "doc-2": {"name": "Beta"},
                "doc-3": {"name": "Gamma"},
            }
        )

        extractor = Extract(schema, model="function:test", provenance="none")
        extractor.call._client = client

        from kaos_llm_core.programs.batch import batch_run

        # First run — all 3 succeed.
        result1 = await batch_run(
            extractor,
            CorpusInputSource(corpus),
            output_dir=str(tmp_path / "resume-run"),
            max_concurrency=1,
        )
        assert result1.n_succeeded == 3
        assert result1.n_skipped == 0

        # Second run with resume=True (default) — should skip all 3.
        # batch_run reports the cumulative n_succeeded from the log (3); the
        # delta signal that this run did zero work is n_skipped=3.
        result2 = await batch_run(
            extractor,
            CorpusInputSource(corpus),
            output_dir=str(tmp_path / "resume-run"),
            max_concurrency=1,
        )
        assert result2.n_skipped == 3
        # No new items were processed (n_succeeded is the cumulative count,
        # not the delta; all 3 rows in the log were pre-existing).


class TestExtractionResultRoundTrip:
    def test_model_dump_validate_round_trip(self) -> None:
        """Extract a full ExtractionResult and round-trip through JSON."""

        from kaos_content.model.extraction import (
            ExtractionCell,
            ExtractionCitation,
        )

        cells = (
            ExtractionCell(
                doc_id="doc-1",
                column_id="name",
                schema_version=1,
                status="extracted",
                ai_value="Acme",
                confidence=0.95,
                citations=(
                    ExtractionCitation(
                        block_ref="#/span/doc-1",
                        page=1,
                        char_span=(0, 4),
                        snippet="Acme",
                        snippet_sha256="a" * 64,
                        confidence=0.9,
                    ),
                ),
                model="anthropic:claude-haiku-4-5",
                cost_usd=0.0001,
                tokens_total=512,
                latency_ms=1000.0,
            ),
        )
        result = ExtractionResult(
            cells=cells,
            schema_id="s",
            schema_version=1,
            doc_id="doc-1",
            is_verified=True,
            cost_usd=0.0001,
            tokens_total=512,
            latency_ms=1000.0,
            model="anthropic:claude-haiku-4-5",
        )

        dumped = result.model_dump_json()
        restored = ExtractionResult.model_validate_json(dumped)
        assert restored == result
