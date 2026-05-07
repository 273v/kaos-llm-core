"""Unit tests for ChunkRetry — WS-TR.PR-6b.

Covers: chunking algorithm (sentence boundaries, fallback path,
greedy accumulation, hard-split for oversize sentences); per-cell
retry policy (which statuses trigger retry, which don't); cost-cap
enforcement; multi-candidate merging; doc-too-large skip; integration
with Extract.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Literal

import pytest
from kaos_content.model.extraction import ExtractionCell

from kaos_llm_core.programs.chunk_retry import (
    ChunkExtractResult,
    ChunkRetry,
    ChunkRetryConfig,
    ChunkSpan,
    _chunk_text,
    iter_chunks,
)
from kaos_llm_core.signatures.extraction import ColumnSpec

# ---------------------------------------------------------------------------
# Cell-builder helper (mirrors the merger tests pattern)
# ---------------------------------------------------------------------------


def _cell(
    column_id: str,
    *,
    status: Literal["extracted", "not_in_document", "unclear", "error"],
    ai_value: object | None = None,
    confidence: float | None = None,
) -> ExtractionCell:
    if status == "extracted":
        return ExtractionCell(
            doc_id="doc-1",
            column_id=column_id,
            schema_version=1,
            status=status,
            ai_value=ai_value,
            confidence=confidence if confidence is not None else 0.9,
        )
    if status == "unclear":
        return ExtractionCell(
            doc_id="doc-1",
            column_id=column_id,
            schema_version=1,
            status=status,
            confidence=confidence if confidence is not None else 0.4,
        )
    return ExtractionCell(
        doc_id="doc-1",
        column_id=column_id,
        schema_version=1,
        status=status,
    )


COL_A = ColumnSpec(id="parties", column_type="string")
COL_B = ColumnSpec(id="amount", column_type="money")


# ---------------------------------------------------------------------------
# Chunking algorithm
# ---------------------------------------------------------------------------


class TestChunking:
    def test_short_text_returns_single_chunk(self) -> None:
        text = "This is a short sentence. So is this one."
        chunks = _chunk_text(text, max_chars=1000, min_chars=100)
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].start == 0
        assert chunks[0].end == len(text)

    def test_long_text_splits_at_sentences(self) -> None:
        sentence = "This is one sentence with some words to take up space. "
        text = sentence * 30  # ~1700 chars
        chunks = _chunk_text(text, max_chars=500, min_chars=100)
        assert len(chunks) > 1
        # Every chunk respects max_chars.
        for c in chunks:
            assert c.end - c.start <= 500
        # Concatenation covers the original (modulo whitespace).
        joined = "".join(text[c.start : c.end] for c in chunks)
        assert joined.replace(" ", "") == text.replace(" ", "")

    def test_chunk_spans_round_trip(self) -> None:
        text = "First sentence. Second sentence. Third sentence." * 10
        chunks = _chunk_text(text, max_chars=200, min_chars=50)
        for c in chunks:
            assert text[c.start : c.end] == c.text

    def test_iter_chunks_public_api(self) -> None:
        text = "Sentence one. Sentence two." * 50
        chunks = list(iter_chunks(text, max_chars=200, min_chars=50))
        assert len(chunks) > 1
        assert all(isinstance(c, ChunkSpan) for c in chunks)

    def test_paragraph_fallback_when_segmentation_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Force the kaos-nlp-core import to fail to exercise the
        paragraph fallback path."""
        import builtins as _builtins

        real_import = _builtins.__import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "kaos_nlp_core.segmentation":
                raise ImportError("simulated missing dep")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(_builtins, "__import__", fake_import)
        text = "Para one with some words.\n\nPara two with more.\n\nPara three." * 10
        chunks = _chunk_text(text, max_chars=200, min_chars=50)
        assert len(chunks) > 0
        for c in chunks:
            assert text[c.start : c.end] == c.text


# ---------------------------------------------------------------------------
# Refusal-status filter
# ---------------------------------------------------------------------------


class TestRefusalFilter:
    @pytest.mark.asyncio
    async def test_extracted_cells_skip_retry(self) -> None:
        retry = ChunkRetry(config=ChunkRetryConfig(initial_chunk_chars=100))
        cells = (_cell("parties", status="extracted", ai_value="Acme Inc."),)
        call_count = 0

        async def fake_extract(text: str, doc_id: str) -> ChunkExtractResult:
            nonlocal call_count
            call_count += 1
            return ChunkExtractResult(cells=(), cost_usd=0.0)

        new_cells, cost = await retry.retry_refused_cells(
            cells,
            columns=(COL_A,),
            source_text="Long text " * 200,
            doc_id="doc-1",
            extract_chunk=fake_extract,
        )
        assert call_count == 0  # extracted status skips retry entirely
        assert new_cells == cells
        assert cost == 0.0

    @pytest.mark.asyncio
    async def test_error_cells_skip_retry(self) -> None:
        """Error-status cells are transient API failures handled by
        kaos-llm-client retry, not by chunk-retry."""
        retry = ChunkRetry(config=ChunkRetryConfig(initial_chunk_chars=100))
        # Build an error cell directly (bypass _cell helper which
        # doesn't handle the error case fully).
        from kaos_content.model.extraction import ExtractionError

        err_cell = ExtractionCell(
            doc_id="doc-1",
            column_id="parties",
            schema_version=1,
            status="error",
            error=ExtractionError(
                code="model_api_error", message="rate limit", retry_recommended=True
            ),
        )
        call_count = 0

        async def fake_extract(text: str, doc_id: str) -> ChunkExtractResult:
            nonlocal call_count
            call_count += 1
            return ChunkExtractResult(cells=(), cost_usd=0.0)

        new_cells, _cost = await retry.retry_refused_cells(
            (err_cell,),
            columns=(COL_A,),
            source_text="Long text " * 200,
            doc_id="doc-1",
            extract_chunk=fake_extract,
        )
        assert call_count == 0
        assert new_cells == (err_cell,)


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------


class TestRecovery:
    @pytest.mark.asyncio
    async def test_refused_cell_recovered_from_chunk(self) -> None:
        """The hypothesis: LLM refuses on full text but a specific chunk
        contains the answer. Chunk-retry surfaces that answer."""
        retry = ChunkRetry(
            config=ChunkRetryConfig(
                initial_chunk_chars=200, min_chunk_chars=50, max_chunks_per_doc=99
            )
        )
        cells = (_cell("parties", status="not_in_document"),)
        long_text = "Lorem ipsum dolor sit amet. " * 50  # ~1400 chars
        call_count = 0

        async def fake_extract(text: str, doc_id: str) -> ChunkExtractResult:
            nonlocal call_count
            call_count += 1
            # Simulate the LLM finding the value on the third chunk.
            if call_count == 3:
                return ChunkExtractResult(
                    cells=(_cell("parties", status="extracted", ai_value="Acme Inc."),),
                    cost_usd=0.001,
                )
            return ChunkExtractResult(
                cells=(_cell("parties", status="not_in_document"),),
                cost_usd=0.001,
            )

        new_cells, cost = await retry.retry_refused_cells(
            cells,
            columns=(COL_A,),
            source_text=long_text,
            doc_id="doc-1",
            extract_chunk=fake_extract,
        )
        assert new_cells[0].status == "extracted"
        assert new_cells[0].ai_value == "Acme Inc."
        assert new_cells[0].confidence == 0.7  # confidence_on_recovery default
        assert "chunk-retry recovered" in (new_cells[0].rationale or "")
        assert cost > 0

    @pytest.mark.asyncio
    async def test_no_chunks_recover_returns_unchanged(self) -> None:
        retry = ChunkRetry(
            config=ChunkRetryConfig(
                initial_chunk_chars=200, max_chunks_per_doc=99, min_chunk_chars=50
            )
        )
        cells = (_cell("parties", status="not_in_document"),)
        long_text = "Filler text. " * 200

        async def fake_extract(text: str, doc_id: str) -> ChunkExtractResult:
            return ChunkExtractResult(
                cells=(_cell("parties", status="not_in_document"),),
                cost_usd=0.001,
            )

        new_cells, _ = await retry.retry_refused_cells(
            cells,
            columns=(COL_A,),
            source_text=long_text,
            doc_id="doc-1",
            extract_chunk=fake_extract,
        )
        assert new_cells[0].status == "not_in_document"

    @pytest.mark.asyncio
    async def test_multiple_candidates_first_wins(self) -> None:
        """When chunks return different values, leftmost-in-document wins
        and the others contribute citations."""
        retry = ChunkRetry(
            config=ChunkRetryConfig(
                initial_chunk_chars=200,
                min_chunk_chars=50,
                max_chunks_per_doc=99,
                max_candidates_per_cell=5,
            )
        )
        cells = (_cell("parties", status="not_in_document"),)
        long_text = "Filler. " * 100

        call_idx = 0

        async def fake_extract(text: str, doc_id: str) -> ChunkExtractResult:
            nonlocal call_idx
            call_idx += 1
            return ChunkExtractResult(
                cells=(_cell("parties", status="extracted", ai_value=f"value_{call_idx}"),),
                cost_usd=0.001,
            )

        new_cells, _ = await retry.retry_refused_cells(
            cells,
            columns=(COL_A,),
            source_text=long_text,
            doc_id="doc-1",
            extract_chunk=fake_extract,
        )
        assert new_cells[0].status == "extracted"
        # First chunk's value wins.
        assert new_cells[0].ai_value == "value_1"
        # Rationale mentions multiple candidates.
        assert "candidates merged" in (new_cells[0].rationale or "")

    @pytest.mark.asyncio
    async def test_disabled_config_no_op(self) -> None:
        retry = ChunkRetry(config=ChunkRetryConfig(enabled=False))
        cells = (_cell("parties", status="not_in_document"),)
        call_count = 0

        async def fake_extract(text: str, doc_id: str) -> ChunkExtractResult:
            nonlocal call_count
            call_count += 1
            return ChunkExtractResult(cells=(), cost_usd=0.0)

        new_cells, cost = await retry.retry_refused_cells(
            cells,
            columns=(COL_A,),
            source_text="Long text " * 100,
            doc_id="doc-1",
            extract_chunk=fake_extract,
        )
        assert call_count == 0
        assert cost == 0.0
        assert new_cells == cells


# ---------------------------------------------------------------------------
# Cost cap + safety valves
# ---------------------------------------------------------------------------


class TestCostCap:
    @pytest.mark.asyncio
    async def test_max_total_cost_short_circuits(self) -> None:
        """Once cumulative cost exceeds the budget, remaining chunks
        are skipped — even if they would have recovered the value."""
        retry = ChunkRetry(
            config=ChunkRetryConfig(
                initial_chunk_chars=200,
                min_chunk_chars=50,
                max_chunks_per_doc=99,
                max_total_cost_usd=0.005,  # only 5 calls at $0.001 each
            )
        )
        cells = (_cell("parties", status="not_in_document"),)
        long_text = "Padding " * 500
        call_count = 0

        async def fake_extract(text: str, doc_id: str) -> ChunkExtractResult:
            nonlocal call_count
            call_count += 1
            return ChunkExtractResult(
                cells=(_cell("parties", status="not_in_document"),),
                cost_usd=0.001,
            )

        _new_cells, cost = await retry.retry_refused_cells(
            cells,
            columns=(COL_A,),
            source_text=long_text,
            doc_id="doc-1",
            extract_chunk=fake_extract,
        )
        # Loop should exit once cumulative cost exceeds 0.005.
        # Each call adds 0.001; the budget check fires AT THE TOP of
        # the loop, so we make calls until the cumulative meets/exceeds
        # the cap, then stop on the next iteration.
        assert call_count <= 6
        assert cost <= 0.006

    @pytest.mark.asyncio
    async def test_doc_too_large_skips_retry(self) -> None:
        """Documents that would yield more than max_chunks_per_doc are
        skipped entirely — pathological inputs aren't worth retrying."""
        retry = ChunkRetry(
            config=ChunkRetryConfig(
                initial_chunk_chars=100,
                min_chunk_chars=20,
                max_chunks_per_doc=3,
            )
        )
        cells = (_cell("parties", status="not_in_document"),)
        long_text = "Sentence. " * 200  # would yield ~20 chunks
        call_count = 0

        async def fake_extract(text: str, doc_id: str) -> ChunkExtractResult:
            nonlocal call_count
            call_count += 1
            return ChunkExtractResult(cells=(), cost_usd=0.0)

        new_cells, _cost = await retry.retry_refused_cells(
            cells,
            columns=(COL_A,),
            source_text=long_text,
            doc_id="doc-1",
            extract_chunk=fake_extract,
        )
        assert call_count == 0
        assert new_cells == cells

    @pytest.mark.asyncio
    async def test_single_chunk_skips_retry(self) -> None:
        """When the document fits in one chunk, the LLM already saw the
        whole thing on the original call. Chunk-retry would just repeat
        the same call. Skip."""
        retry = ChunkRetry(
            config=ChunkRetryConfig(
                initial_chunk_chars=10000,
                max_chunks_per_doc=99,
            )
        )
        cells = (_cell("parties", status="not_in_document"),)
        call_count = 0

        async def fake_extract(text: str, doc_id: str) -> ChunkExtractResult:
            nonlocal call_count
            call_count += 1
            return ChunkExtractResult(cells=(), cost_usd=0.0)

        _new_cells, _ = await retry.retry_refused_cells(
            cells,
            columns=(COL_A,),
            source_text="Short text",  # fits in one chunk
            doc_id="doc-1",
            extract_chunk=fake_extract,
        )
        assert call_count == 0


# ---------------------------------------------------------------------------
# Chunk failure tolerance
# ---------------------------------------------------------------------------


class TestChunkFailures:
    @pytest.mark.asyncio
    async def test_chunk_exception_continues_loop(self) -> None:
        """If one chunk's extraction throws, the remaining chunks still
        run — the goal is recovery, and one bad chunk shouldn't kill it."""
        retry = ChunkRetry(
            config=ChunkRetryConfig(
                initial_chunk_chars=200,
                min_chunk_chars=50,
                max_chunks_per_doc=99,
            )
        )
        cells = (_cell("parties", status="not_in_document"),)
        long_text = "Sentence here. " * 50
        call_count = 0

        async def fake_extract(text: str, doc_id: str) -> ChunkExtractResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("simulated chunk failure")
            return ChunkExtractResult(
                cells=(_cell("parties", status="extracted", ai_value="recovered"),),
                cost_usd=0.001,
            )

        new_cells, _ = await retry.retry_refused_cells(
            cells,
            columns=(COL_A,),
            source_text=long_text,
            doc_id="doc-1",
            extract_chunk=fake_extract,
        )
        assert call_count > 1  # didn't stop at the failure
        assert new_cells[0].status == "extracted"
        assert new_cells[0].ai_value == "recovered"


# ---------------------------------------------------------------------------
# Multi-column scenarios
# ---------------------------------------------------------------------------


class TestMultiColumn:
    @pytest.mark.asyncio
    async def test_only_refused_columns_collected(self) -> None:
        """Cells already in extracted status don't have candidates
        collected for them, even if a chunk happens to extract them."""
        retry = ChunkRetry(
            config=ChunkRetryConfig(
                initial_chunk_chars=200,
                min_chunk_chars=50,
                max_chunks_per_doc=99,
            )
        )
        cells = (
            _cell("parties", status="extracted", ai_value="LLM-found"),
            _cell("amount", status="not_in_document"),
        )
        long_text = "Filler. " * 100

        async def fake_extract(text: str, doc_id: str) -> ChunkExtractResult:
            # Each chunk returns extracted values for BOTH columns.
            return ChunkExtractResult(
                cells=(
                    _cell("parties", status="extracted", ai_value="chunk-found"),
                    _cell(
                        "amount",
                        status="extracted",
                        ai_value={"amount": Decimal("100"), "currency": "USD"},
                    ),
                ),
                cost_usd=0.001,
            )

        new_cells, _ = await retry.retry_refused_cells(
            cells,
            columns=(COL_A, COL_B),
            source_text=long_text,
            doc_id="doc-1",
            extract_chunk=fake_extract,
        )
        # Parties was already extracted — left alone.
        assert new_cells[0].ai_value == "LLM-found"
        assert new_cells[0].status == "extracted"
        assert new_cells[0].confidence == 0.9
        # Amount was refused — recovered from chunks.
        assert new_cells[1].status == "extracted"
        assert new_cells[1].confidence == 0.7


# ---------------------------------------------------------------------------
# Extract integration
# ---------------------------------------------------------------------------


class TestExtractIntegration:
    def test_extract_init_default_disables_chunk_retry(self) -> None:
        """Chunk-retry is opt-in (default None) to preserve existing
        latency profiles."""
        from kaos_llm_core.programs.extract import Extract
        from kaos_llm_core.signatures.extraction import ExtractionSchema

        schema = ExtractionSchema.from_dict(
            {
                "id": "test",
                "columns": [{"id": "parties", "column_type": "string"}],
            }
        )
        extractor = Extract(schema, model="openai:gpt-5.4-nano")
        assert extractor._chunk_retry is None

    def test_extract_init_default_keyword(self) -> None:
        from kaos_llm_core.programs.extract import Extract
        from kaos_llm_core.signatures.extraction import ExtractionSchema

        schema = ExtractionSchema.from_dict(
            {
                "id": "test",
                "columns": [{"id": "parties", "column_type": "string"}],
            }
        )
        extractor = Extract(schema, model="openai:gpt-5.4-nano", chunk_retry="default")
        assert extractor._chunk_retry is not None
        assert isinstance(extractor._chunk_retry, ChunkRetry)
        assert extractor._chunk_retry.config.enabled is True

    def test_extract_init_custom_config(self) -> None:
        from kaos_llm_core.programs.extract import Extract
        from kaos_llm_core.signatures.extraction import ExtractionSchema

        schema = ExtractionSchema.from_dict(
            {
                "id": "test",
                "columns": [{"id": "parties", "column_type": "string"}],
            }
        )
        cfg = ChunkRetryConfig(initial_chunk_chars=4000, max_total_cost_usd=0.10)
        extractor = Extract(schema, model="openai:gpt-5.4-nano", chunk_retry=cfg)
        assert extractor._chunk_retry is not None
        assert extractor._chunk_retry.config.initial_chunk_chars == 4000
        assert extractor._chunk_retry.config.max_total_cost_usd == 0.10

    def test_extract_init_custom_instance(self) -> None:
        from kaos_llm_core.programs.extract import Extract
        from kaos_llm_core.signatures.extraction import ExtractionSchema

        schema = ExtractionSchema.from_dict(
            {
                "id": "test",
                "columns": [{"id": "parties", "column_type": "string"}],
            }
        )
        custom = ChunkRetry(config=ChunkRetryConfig(confidence_on_recovery=0.6))
        extractor = Extract(schema, model="openai:gpt-5.4-nano", chunk_retry=custom)
        assert extractor._chunk_retry is custom
