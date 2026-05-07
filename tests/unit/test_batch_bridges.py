"""Phase 15.4 cross-module BatchInputSource bridge tests.

Exercises the PDF-pages and tabular-rows input sources end-to-end via
``batch_run()`` against a deterministic FunctionClient. Each test
constructs its own PDF/CSV fixture in-place via the optional
dependencies' real APIs — no canned binary blobs.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart

from kaos_llm_core.observability.cost import PRICING, ModelPricing
from kaos_llm_core.programs.batch import batch_run
from kaos_llm_core.programs.batch_bridges import (
    PdfPagesInputSource,
    TabularRowsInputSource,
    pdf_pages_input_source,
    tabular_rows_input_source,
)
from kaos_llm_core.programs.call import Call
from kaos_llm_core.signatures import InputField, OutputField, Signature

# Optional deps — guard tests that exercise the real bridges
_has_kaos_pdf = importlib.util.find_spec("kaos_pdf") is not None
_has_kaos_tabular = importlib.util.find_spec("kaos_tabular") is not None


class _Sig(Signature):
    """Echo a value."""

    text: str = InputField(description="Input text")
    answer: str = OutputField(description="Echoed answer")


def _json_response(data: dict[str, Any]) -> ProviderResponse:
    return ProviderResponse.model_construct(
        provider="function",
        model="function-test",
        raw={},
        parts=[ContentPart.model_construct(type="text", text=json.dumps(data))],
        usage=UsageInfo.model_construct(input_tokens=10, output_tokens=5, total_tokens=15),
        stop_reason="end_turn",
        status_code=200,
        response_headers={},
    )


def _make_call() -> Call:
    counter = {"n": 0}

    def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
        i = counter["n"]
        counter["n"] += 1
        return _json_response({"answer": f"reply-{i}"})

    call = Call(_Sig, model="function-test")
    call._client = FunctionClient(function=fn)
    return call


@pytest.fixture(autouse=True)
def _pricing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        PRICING,
        "function-test",
        ModelPricing(input_per_mtok=1.0, output_per_mtok=2.0),
    )


# ---------------------------------------------------------------------------
# PDF fixture: build a small multi-page PDF in tmp_path via pypdfium2 helpers
# ---------------------------------------------------------------------------


def _build_test_pdf(path: Path, pages_text: list[str]) -> None:
    """Generate a tiny multi-page PDF using reportlab if available, else
    fall back to a hand-rolled minimal PDF that pypdfium2 can read."""
    try:
        from reportlab.lib.pagesizes import letter  # ty: ignore[unresolved-import]
        from reportlab.pdfgen import canvas  # ty: ignore[unresolved-import]
    except ImportError:
        # Hand-rolled minimal PDF (one text object per page).
        _build_minimal_pdf(path, pages_text)
        return

    c = canvas.Canvas(str(path), pagesize=letter)
    for text in pages_text:
        c.drawString(72, 720, text)
        c.showPage()
    c.save()


def _build_minimal_pdf(path: Path, pages_text: list[str]) -> None:
    """Construct a valid minimal PDF with one Helvetica text object per page.

    pypdfium2 needs a real Page tree, content streams, and at least one
    font dict. The generated file is small (<2 KB) and round-trips
    through ``extract_page_text``.
    """
    objects: list[bytes] = []

    def add(obj: bytes) -> int:
        objects.append(obj)
        return len(objects)

    font_obj = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    font_id = add(font_obj)

    page_ids: list[int] = []
    content_ids: list[int] = []
    for text in pages_text:
        # Escape parens
        safe = text.replace("\\", "\\\\").replace("(", r"\(").replace(")", r"\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({safe}) Tj ET\n".encode()
        content = (
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"endstream"
        )
        content_id = add(content)
        content_ids.append(content_id)

    # Pages need to know the parent /Pages id, so we reserve it now.
    pages_id = len(objects) + len(pages_text) + 1

    for content_id in content_ids:
        page_obj = (
            f"<< /Type /Page /Parent {pages_id} 0 R "
            f"/MediaBox [0 0 612 792] "
            f"/Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>"
        ).encode()
        page_ids.append(add(page_obj))

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    pages_obj = (f"<< /Type /Pages /Count {len(page_ids)} /Kids [{kids}] >>").encode()
    actual_pages_id = add(pages_obj)
    assert actual_pages_id == pages_id, "page tree id mismatch"

    catalog_obj = f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode()
    catalog_id = add(catalog_obj)

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for idx, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{idx} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_offset = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        b"trailer\n<< /Size "
        + str(len(objects) + 1).encode()
        + b" /Root "
        + str(catalog_id).encode()
        + b" 0 R >>\nstartxref\n"
        + str(xref_offset).encode()
        + b"\n%%EOF\n"
    )
    path.write_bytes(bytes(out))


# ---------------------------------------------------------------------------
# PDF bridge tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _has_kaos_pdf,
    reason="kaos-pdf not installed (optional dependency)",
)
class TestPdfPagesInputSource:
    async def test_yields_one_item_per_page(self, tmp_path: Path) -> None:
        pdf = tmp_path / "doc.pdf"
        _build_test_pdf(pdf, ["page one body", "page two body", "page three body"])
        source = PdfPagesInputSource(pdf, program_hash_value="sha256:test")
        items = []
        async for item in source:
            items.append(item)
        assert len(items) == 3
        # Inputs carry only the page text; page number lives on metadata
        for i, item in enumerate(items):
            assert "text" in item.inputs
            assert "page" not in item.inputs
            assert item.metadata["page"] == i
            assert item.inputs["text"].strip()

    async def test_skip_blank_drops_empty_pages(self, tmp_path: Path) -> None:
        pdf = tmp_path / "blank.pdf"
        _build_test_pdf(pdf, ["filled", "", "filled again"])
        source = PdfPagesInputSource(pdf, program_hash_value="sha256:test", skip_blank=True)
        items = [item async for item in source]
        # The blank page is dropped → only 2 items
        assert len(items) == 2
        assert items[0].metadata["page"] == 0
        assert items[1].metadata["page"] == 2

    async def test_page_range_slices(self, tmp_path: Path) -> None:
        pdf = tmp_path / "ranged.pdf"
        _build_test_pdf(pdf, [f"page {i}" for i in range(5)])
        source = PdfPagesInputSource(pdf, program_hash_value="sha256:test", page_range=(1, 4))
        items = [item async for item in source]
        assert len(items) == 3
        assert [it.metadata["page"] for it in items] == [1, 2, 3]

    async def test_describe(self, tmp_path: Path) -> None:
        pdf = tmp_path / "d.pdf"
        _build_test_pdf(pdf, ["x"])
        source = pdf_pages_input_source(pdf, program_hash_value="sha256:test", page_range=(0, 1))
        desc = source.describe()
        assert desc["type"] == "pdf_pages"
        assert desc["page_range"] == [0, 1]
        assert desc["skip_blank"] is True

    async def test_deterministic_id_per_page(self, tmp_path: Path) -> None:
        pdf = tmp_path / "ids.pdf"
        _build_test_pdf(pdf, ["a", "b"])
        source1 = PdfPagesInputSource(pdf, program_hash_value="sha256:test")
        source2 = PdfPagesInputSource(pdf, program_hash_value="sha256:test")
        ids1 = [item.custom_id async for item in source1]
        ids2 = [item.custom_id async for item in source2]
        assert ids1 == ids2  # content-addressed → stable across iterators

    async def test_pdf_bridge_drives_batch_run(self, tmp_path: Path) -> None:
        pdf = tmp_path / "batch.pdf"
        _build_test_pdf(pdf, [f"document chunk {i} sentinel content" for i in range(4)])
        call = _make_call()
        result = await batch_run(
            call,
            PdfPagesInputSource(pdf, program_hash_value="sha256:test", skip_blank=False),
            output_dir=str(tmp_path / "out"),
            resume=False,
        )
        assert result.n_total == 4
        assert result.n_succeeded == 4
        assert result.n_errored == 0

    async def test_missing_file_raises(self, tmp_path: Path) -> None:
        from kaos_llm_core.programs.batch import BatchError

        source = PdfPagesInputSource(tmp_path / "nope.pdf", program_hash_value="sha256:test")
        with pytest.raises(BatchError, match="does not exist"):
            async for _ in source:
                pass


# ---------------------------------------------------------------------------
# Tabular bridge tests
# ---------------------------------------------------------------------------


def _build_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


@pytest.mark.skipif(
    not _has_kaos_tabular,
    reason="kaos-tabular not installed (optional dependency)",
)
class TestTabularRowsInputSource:
    async def test_yields_one_item_per_row(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "data.csv"
        _build_csv(
            csv_path,
            ["text", "label"],
            [
                ["positive review", "good"],
                ["negative review", "bad"],
                ["neutral review", "meh"],
            ],
        )
        source = TabularRowsInputSource(csv_path, program_hash_value="sha256:test")
        items = [item async for item in source]
        assert len(items) == 3
        assert items[0].inputs == {"text": "positive review", "label": "good"}
        assert items[2].inputs == {"text": "neutral review", "label": "meh"}

    async def test_sql_filter(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "filt.csv"
        _build_csv(
            csv_path,
            ["text", "score"],
            [["a", "1"], ["b", "5"], ["c", "9"]],
        )
        source = TabularRowsInputSource(
            csv_path,
            program_hash_value="sha256:test",
            sql="SELECT text FROM data WHERE CAST(score AS INTEGER) >= 5",
        )
        items = [item async for item in source]
        assert len(items) == 2
        assert items[0].inputs == {"text": "b"}
        assert items[1].inputs == {"text": "c"}

    async def test_max_rows_caps_iteration(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "many.csv"
        _build_csv(csv_path, ["text"], [[f"row-{i}"] for i in range(20)])
        source = TabularRowsInputSource(csv_path, program_hash_value="sha256:test", max_rows=5)
        items = [item async for item in source]
        assert len(items) == 5

    async def test_describe(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "d.csv"
        _build_csv(csv_path, ["text"], [["x"]])
        source = tabular_rows_input_source(
            csv_path,
            program_hash_value="sha256:test",
            sql="SELECT * FROM data",
            max_rows=42,
        )
        desc = source.describe()
        assert desc["type"] == "tabular_rows"
        assert desc["sql"] == "SELECT * FROM data"
        assert desc["max_rows"] == 42

    async def test_tabular_bridge_drives_batch_run(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "batch.csv"
        _build_csv(
            csv_path,
            ["text"],
            [[f"row text {i}"] for i in range(6)],
        )
        call = _make_call()
        result = await batch_run(
            call,
            TabularRowsInputSource(csv_path, program_hash_value="sha256:test"),
            output_dir=str(tmp_path / "out"),
            resume=False,
        )
        assert result.n_total == 6
        assert result.n_succeeded == 6
        assert result.n_errored == 0

    async def test_missing_file_raises(self, tmp_path: Path) -> None:
        from kaos_llm_core.programs.batch import BatchError

        source = TabularRowsInputSource(tmp_path / "nope.csv", program_hash_value="sha256:test")
        with pytest.raises(BatchError, match="does not exist"):
            async for _ in source:
                pass

    async def test_invalid_sql_raises_actionable_error(self, tmp_path: Path) -> None:
        from kaos_llm_core.programs.batch import BatchError

        csv_path = tmp_path / "bad.csv"
        _build_csv(csv_path, ["text"], [["x"]])
        source = TabularRowsInputSource(
            csv_path,
            program_hash_value="sha256:test",
            sql="SELECT * FROM nonexistent_table",
        )
        with pytest.raises(BatchError, match="failed to query"):
            async for _ in source:
                pass


# ---------------------------------------------------------------------------
# Descriptor → BatchInputSource dispatch (the MCP-tool integration path)
# ---------------------------------------------------------------------------


class TestBuildInputSourceBridges:
    @pytest.mark.skipif(
        not _has_kaos_pdf,
        reason="kaos-pdf not installed (optional dependency)",
    )
    async def test_pdf_pages_descriptor(self, tmp_path: Path) -> None:
        from kaos_llm_core.integrations.mcp._batch_helpers import build_input_source

        pdf = tmp_path / "desc.pdf"
        _build_test_pdf(pdf, ["one", "two"])
        result = build_input_source(
            {"type": "pdf_pages", "path": str(pdf)},
            program_hash_value="sha256:test",
            context=None,
        )
        # ToolResult on error, tuple on success
        assert isinstance(result, tuple), result
        source, count = result
        assert count == 2
        items = [item async for item in source]
        assert len(items) == 2

    @pytest.mark.skipif(
        not _has_kaos_tabular,
        reason="kaos-tabular not installed (optional dependency)",
    )
    async def test_tabular_rows_descriptor(self, tmp_path: Path) -> None:
        from kaos_llm_core.integrations.mcp._batch_helpers import build_input_source

        csv_path = tmp_path / "desc.csv"
        _build_csv(csv_path, ["text"], [["a"], ["b"]])
        result = build_input_source(
            {"type": "tabular_rows", "path": str(csv_path)},
            program_hash_value="sha256:test",
            context=None,
        )
        assert isinstance(result, tuple), result
        source, count = result
        # tabular doesn't pre-count rows
        assert count is None
        items = [item async for item in source]
        assert len(items) == 2

    async def test_unknown_type(self) -> None:
        from kaos_core import ToolResult

        from kaos_llm_core.integrations.mcp._batch_helpers import build_input_source

        result = build_input_source({"type": "wat"}, program_hash_value="sha256:test", context=None)
        assert isinstance(result, ToolResult)
        assert result.isError
