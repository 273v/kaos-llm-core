"""Unit tests for the Extract program.

Uses FunctionClient to drive deterministic fake LLM responses and asserts
the Cell-projection pipeline, error classification, and refusal policy.
Live tests live in ``tests/integration/test_extract_live.py``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from kaos_content.model.extraction import ExtractionCell
from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart

from kaos_llm_core.programs.extract import Extract, ExtractionResult
from kaos_llm_core.signatures.extraction import ExtractionSchema


def _json_client(payload: dict[str, Any]) -> FunctionClient:
    """FunctionClient that always returns ``payload`` as a JSON text block."""

    def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
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


class TestExtractCompiles:
    def test_accepts_extraction_schema(self) -> None:
        schema = ExtractionSchema.from_dict(
            {"id": "x", "columns": [{"id": "name", "column_type": "string"}]}
        )
        ext = Extract(schema, model="function:test", provenance="none")
        assert ext.schema is schema

    def test_accepts_dict(self) -> None:
        ext = Extract(
            {"id": "x", "columns": [{"id": "name", "column_type": "string"}]},
            model="function:test",
            provenance="none",
        )
        assert ext.schema.id == "x"

    def test_accepts_pydantic(self) -> None:
        from pydantic import BaseModel

        class Contract(BaseModel):
            name: str
            amount: float

        ext = Extract(Contract, model="function:test", provenance="none")
        assert ext.schema.id == "Contract"
        assert len(ext.schema.columns) == 2

    def test_rejects_invalid_schema(self) -> None:
        from typing import cast

        with pytest.raises(TypeError, match="ExtractionSchema"):
            Extract(cast(Any, "not a schema"), model="function:test")


@pytest.mark.asyncio
class TestExtractBareProvenance:
    async def test_extracts_simple_fields(self) -> None:
        schema = ExtractionSchema.from_dict(
            {
                "id": "contract",
                "columns": [
                    {"id": "name", "column_type": "string"},
                    {"id": "amount", "column_type": "integer"},
                ],
            }
        )
        client = _json_client({"name": "Acme", "amount": 1000})
        ext = Extract(schema, model="function:test", provenance="none")
        ext.call._client = client

        result = await ext.extract(text="contract text", doc_id="doc-1")

        assert isinstance(result, ExtractionResult)
        assert result.doc_id == "doc-1"
        assert result.schema_id == "contract"
        assert result.is_verified is True
        assert len(result.cells) == 2

        by_id = {c.column_id: c for c in result.cells}
        assert by_id["name"].status == "extracted"
        assert by_id["name"].ai_value == "Acme"
        assert by_id["name"].confidence == 1.0
        assert by_id["amount"].ai_value == 1000
        assert not result.refused_columns


@pytest.mark.asyncio
class TestExtractCitedProvenance:
    async def test_projects_cited_spans_to_citations(self) -> None:
        schema = ExtractionSchema.from_dict(
            {"id": "x", "columns": [{"id": "name", "column_type": "string"}]}
        )
        source_text = "The party is Acme Corp."
        client = _json_client(
            {
                "name": {
                    "value": "Acme Corp",
                    "spans": [
                        {
                            "source_uri": "doc-1",
                            "char_span": [13, 22],
                            "quote": "Acme Corp",
                        }
                    ],
                    "confidence": 0.95,
                }
            }
        )
        ext = Extract(schema, model="function:test", provenance="cited")
        ext.call._client = client

        result = await ext.extract(text=source_text, doc_id="doc-1")
        cell = result.cells[0]
        assert cell.status == "extracted"
        assert cell.ai_value == "Acme Corp"
        assert cell.confidence == 0.95
        assert len(cell.citations) == 1
        assert cell.citations[0].snippet == "Acme Corp"
        assert cell.citations[0].char_span == (13, 22)

    async def test_verification_failure_marks_unverified(self) -> None:
        """A span that doesn't match the corpus text fails verification
        (is_verified=False) but does not retry in ``cited`` mode."""
        schema = ExtractionSchema.from_dict(
            {"id": "x", "columns": [{"id": "name", "column_type": "string"}]}
        )
        source_text = "The party is Acme Corp."
        # LLM emits a quote that doesn't appear in the source.
        client = _json_client(
            {
                "name": {
                    "value": "Acme",
                    "spans": [
                        {
                            "source_uri": "doc-1",
                            "char_span": [0, 4],
                            "quote": "NONEXISTENT_QUOTE_XYZ",
                        }
                    ],
                    "confidence": 0.9,
                }
            }
        )
        ext = Extract(schema, model="function:test", provenance="cited")
        ext.call._client = client

        result = await ext.extract(text=source_text, doc_id="doc-1")
        assert result.is_verified is False
        assert result.verification_error_count >= 1


@pytest.mark.asyncio
class TestExtractRefusal:
    async def test_field_refusal_allowed_by_default(self) -> None:
        """Optional fields can be None; refused_columns populated."""
        schema = ExtractionSchema.from_dict(
            {
                "id": "x",
                "columns": [
                    {"id": "name", "column_type": "string"},
                    {"id": "termination", "column_type": "string", "required": False},
                ],
            }
        )
        client = _json_client({"name": "Acme", "termination": None})
        ext = Extract(schema, model="function:test", provenance="none", refusal="field")
        ext.call._client = client

        result = await ext.extract(text="...", doc_id="doc-1")
        assert "termination" in result.refused_columns
        by_id = {c.column_id: c for c in result.cells}
        assert by_id["termination"].status == "not_in_document"
        assert by_id["name"].status == "extracted"

    async def test_refusal_never_errors_out(self) -> None:
        """When refusal='never', any refused column fails the row."""
        schema = ExtractionSchema.from_dict(
            {
                "id": "x",
                "columns": [
                    {"id": "name", "column_type": "string"},
                    {"id": "termination", "column_type": "string", "required": False},
                ],
            }
        )
        client = _json_client({"name": "Acme", "termination": None})
        ext = Extract(schema, model="function:test", provenance="none", refusal="never")
        ext.call._client = client

        result = await ext.extract(text="...", doc_id="doc-1")
        # All cells now have status=error (the whole row errored out).
        assert all(c.status == "error" for c in result.cells)

    async def test_refusal_row_fails_all_cells(self) -> None:
        schema = ExtractionSchema.from_dict(
            {
                "id": "x",
                "columns": [
                    {"id": "name", "column_type": "string"},
                    {"id": "termination", "column_type": "string", "required": False},
                ],
            }
        )
        client = _json_client({"name": "Acme", "termination": None})
        ext = Extract(schema, model="function:test", provenance="none", refusal="row")
        ext.call._client = client

        result = await ext.extract(text="...", doc_id="doc-1")
        assert all(isinstance(c, ExtractionCell) for c in result.cells)
        assert all(c.status == "error" for c in result.cells)


@pytest.mark.asyncio
class TestExtractErrorClassification:
    async def test_provider_error_becomes_error_cells(self) -> None:
        """A raised provider error is caught and cells are tagged error."""

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            from kaos_llm_client.errors import KaosLLMProviderError

            raise KaosLLMProviderError(
                "500 Internal Server Error",
                provider="function",
                status_code=500,
            )

        schema = ExtractionSchema.from_dict(
            {
                "id": "x",
                "columns": [
                    {"id": "name", "column_type": "string"},
                    {"id": "amount", "column_type": "integer"},
                ],
            }
        )
        ext = Extract(schema, model="function:test", provenance="none")
        ext.call._client = FunctionClient(function=fn)

        result = await ext.extract(text="...", doc_id="doc-1")
        assert all(c.status == "error" for c in result.cells)
        # All errors should share the same diagnostic.
        assert all(c.error is not None for c in result.cells)
        first_error = result.cells[0].error
        assert first_error is not None
        assert first_error.code == "model_api_error"
        assert first_error.retry_recommended is True


@pytest.mark.asyncio
class TestExtractCostAndLatency:
    async def test_latency_recorded(self) -> None:
        schema = ExtractionSchema.from_dict(
            {"id": "x", "columns": [{"id": "name", "column_type": "string"}]}
        )
        client = _json_client({"name": "Acme"})
        ext = Extract(schema, model="function:test", provenance="none")
        ext.call._client = client

        result = await ext.extract(text="...", doc_id="doc-1")
        assert result.latency_ms > 0
        assert result.attempts == 1
