"""Phase 15.4 cross-module BatchInputSource bridges.

These bridges let ``batch_run()`` iterate the native artifacts of
sibling KAOS modules — PDF pages from kaos-pdf, tabular rows from
kaos-tabular — by importing their Python APIs directly. They are
**library-level** bridges, not MCP-chained tools: per-row MCP
overhead is fatal at thousand-document scale.

Each source is gated on an optional extra:

  - ``kaos-llm-core[pdf]``     → PdfPagesInputSource
  - ``kaos-llm-core[tabular]`` → TabularRowsInputSource

Importing this module is always cheap; the heavy dependency import
happens lazily on first iteration. ``BatchBridgeUnavailableError`` is
raised at iteration time when the optional package is not installed,
with a clear "pip install kaos-llm-core[X]" message.

Sub-design: ``docs/internal/plans/kaos-llm-core-roadmap.md §4 Phase 15.4``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from pathlib import Path
from typing import Any

from kaos_llm_core.errors import KaosLLMCoreError
from kaos_llm_core.programs.batch import BatchError, BatchItem


class BatchBridgeUnavailableError(KaosLLMCoreError):
    """Raised when a bridge's optional dependency is not installed.

    The exception message names the optional extra (e.g.
    ``kaos-llm-core[pdf]``) so the agent can fix it with a single
    ``pip install`` invocation.
    """


# ---------------------------------------------------------------------------
# kaos-pdf bridge: one BatchItem per page
# ---------------------------------------------------------------------------


class PdfPagesInputSource:
    """Yield one :class:`BatchItem` per page in a PDF.

    Each item's ``inputs`` is ``{"text": <page text>}`` so it matches a
    program whose top-level inputs declare a ``text`` field. The 0-based
    page number lives on ``BatchItem.metadata`` (``"page"``) for
    provenance — keeping it out of ``inputs`` prevents
    SignatureError/extra-field errors when the program does not declare
    a ``page`` input. The ``custom_id`` is content-addressed against
    the program hash plus the (path, page) pair, so resume is idempotent.

    Args:
        path: Filesystem path to the PDF (already resolved through
            the VFS by the calling tool).
        program_hash_value: Program hash for ``BatchItem.deterministic_id``.
        page_range: Optional ``(start, end)`` 0-based half-open page
            slice. ``None`` means every page.
        skip_blank: When True (default), pages whose extracted text is
            empty/whitespace are silently dropped from the iteration.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        program_hash_value: str,
        page_range: tuple[int, int] | None = None,
        skip_blank: bool = True,
    ) -> None:
        self._path = Path(path)
        self._program_hash = program_hash_value
        self._page_range = page_range
        self._skip_blank = skip_blank

    def __aiter__(self) -> AsyncIterator[BatchItem]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[BatchItem]:
        try:
            # kaos-pdf is the optional [pdf] extra; not declared in
            # 0.1.0a1 because it isn't on PyPI yet (returns in 0.1.0a2).
            from kaos_pdf.extract import (  # ty: ignore[unresolved-import]
                extract_page_text,
                get_page_count,
            )
        except ImportError as exc:
            msg = (
                "PdfPagesInputSource requires the optional 'pdf' extra. "
                "Install with: pip install 'kaos-llm-core[pdf]' "
                "(or `uv add kaos-pdf` if you manage deps directly)."
            )
            raise BatchBridgeUnavailableError(msg) from exc

        if not self._path.exists():
            msg = (
                f"PdfPagesInputSource: file {self._path} does not exist. "
                "Resolve the VFS path before constructing the source, or "
                "pass an absolute disk path."
            )
            raise BatchError(msg)

        n_pages = get_page_count(self._path)
        start, end = (0, n_pages) if self._page_range is None else self._page_range
        start = max(0, start)
        end = min(n_pages, end)
        for page_num in range(start, end):
            try:
                text = extract_page_text(self._path, page_num)
            except Exception as exc:
                # Skip unreadable pages but record the failure as a
                # synthetic empty item — the batch error policy decides
                # how to handle it. Wrap as BatchError so the message is
                # actionable in the JSONL log.
                msg = (
                    f"PdfPagesInputSource: failed to extract page {page_num} of {self._path}: {exc}"
                )
                raise BatchError(msg) from exc
            if self._skip_blank and not text.strip():
                continue
            inputs = {"text": text}
            cid = BatchItem.deterministic_id(
                self._program_hash,
                {"path": str(self._path), "page": page_num},
            )
            yield BatchItem(
                custom_id=cid,
                inputs=inputs,
                metadata={"source": "pdf", "path": str(self._path), "page": page_num},
            )

    def describe(self) -> dict[str, Any]:
        return {
            "type": "pdf_pages",
            "path": str(self._path),
            "page_range": list(self._page_range) if self._page_range else None,
            "skip_blank": self._skip_blank,
        }


def pdf_pages_input_source(
    path: str | Path,
    *,
    program_hash_value: str,
    page_range: tuple[int, int] | None = None,
    skip_blank: bool = True,
) -> PdfPagesInputSource:
    """Build a PdfPagesInputSource. See class docstring for arg details."""
    return PdfPagesInputSource(
        path,
        program_hash_value=program_hash_value,
        page_range=page_range,
        skip_blank=skip_blank,
    )


# ---------------------------------------------------------------------------
# kaos-tabular bridge: one BatchItem per row
# ---------------------------------------------------------------------------


class TabularRowsInputSource:
    """Yield one :class:`BatchItem` per row of a tabular query.

    Each item's ``inputs`` is the row as a dict mapping column name to
    Python value. ``custom_id`` is content-addressed against the program
    hash plus the row dict so resume is content-stable.

    Args:
        path: Filesystem path to the tabular file (CSV / Parquet /
            JSON). Already resolved through the VFS by the caller.
        program_hash_value: Program hash for ``BatchItem.deterministic_id``.
        sql: Optional SQL query to apply against the registered file.
            ``None`` means ``SELECT * FROM data``. ``$TABLE`` is
            substituted with the registered table name (default
            ``data``).
        max_rows: Hard cap on rows iterated. Defaults to 10_000 to
            match kaos-tabular's safety cap. Pass a smaller value to
            constrain a specific batch.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        program_hash_value: str,
        sql: str | None = None,
        max_rows: int = 10_000,
    ) -> None:
        self._path = Path(path)
        self._program_hash = program_hash_value
        self._sql = sql
        self._max_rows = max_rows

    def __aiter__(self) -> AsyncIterator[BatchItem]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[BatchItem]:
        try:
            # kaos-tabular is the optional [tabular] extra; not declared
            # in 0.1.0a1 because it isn't on PyPI yet (returns in 0.1.0a2).
            from kaos_tabular.engine import (  # ty: ignore[unresolved-import]
                TabularEngine,
            )
        except ImportError as exc:
            msg = (
                "TabularRowsInputSource requires the optional 'tabular' extra. "
                "Install with: pip install 'kaos-llm-core[tabular]'."
            )
            raise BatchBridgeUnavailableError(msg) from exc

        if not self._path.exists():
            msg = (
                f"TabularRowsInputSource: file {self._path} does not exist. "
                "Resolve the VFS path before constructing the source."
            )
            raise BatchError(msg)

        engine = TabularEngine()
        try:
            table_name = engine.register_file(self._path, table_name="data")
            sql = self._sql or f"SELECT * FROM {table_name}"
            sql = sql.replace("$TABLE", table_name)
            table = engine.execute(sql, max_rows=self._max_rows)
        except Exception as exc:
            msg = (
                f"TabularRowsInputSource: failed to query {self._path}: {exc}. "
                "Check (1) the file format is supported (csv/parquet/json), "
                "(2) the SQL is valid against table 'data', (3) max_rows is "
                "within the 10,000 hard cap."
            )
            raise BatchError(msg) from exc

        column_names = [c.name for c in table.columns]
        for row in table.rows:
            row_dict = dict(zip(column_names, row, strict=False))
            cid = BatchItem.deterministic_id(self._program_hash, row_dict)
            yield BatchItem(
                custom_id=cid,
                inputs=row_dict,
                metadata={
                    "source": "tabular",
                    "path": str(self._path),
                },
            )

    def describe(self) -> dict[str, Any]:
        return {
            "type": "tabular_rows",
            "path": str(self._path),
            "sql": self._sql,
            "max_rows": self._max_rows,
        }


def tabular_rows_input_source(
    path: str | Path,
    *,
    program_hash_value: str,
    sql: str | None = None,
    max_rows: int = 10_000,
) -> TabularRowsInputSource:
    """Build a TabularRowsInputSource. See class docstring for arg details."""
    return TabularRowsInputSource(
        path,
        program_hash_value=program_hash_value,
        sql=sql,
        max_rows=max_rows,
    )


__all__ = [
    "BatchBridgeUnavailableError",
    "PdfPagesInputSource",
    "TabularRowsInputSource",
    "pdf_pages_input_source",
    "tabular_rows_input_source",
]


# Silence unused-import lint
_ = Iterable
