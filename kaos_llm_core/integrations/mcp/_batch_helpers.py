"""Shared helpers for the four Phase 15.3 batch MCP tool modules.

The four tools (``batch-create``, ``batch-run``, ``batch-status``,
``batch-results``) all need to:
  - Open a :class:`WorkspaceMetadata` against the runtime VFS.
  - Load a Program v3 envelope (inline OR via VFS path).
  - Build a :class:`BatchInputSource` from a JSON descriptor.
  - Resolve a VFS output_dir into a real disk path.
  - Convert a :class:`BatchRecord` into a JSON dict for the tool result.

Centralizing here keeps each tool module focused on its KaosTool
subclass and its argument schema.

Sub-design: ``docs/internal/design/batch-workspace-schema.md``.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import asdict
from pathlib import Path
from typing import Any

from kaos_core import KaosContext, ToolResult
from kaos_core.logging import get_logger

from kaos_llm_core.programs.batch import (
    BatchError,
    BatchInputSource,
    JsonlInputSource,
    ListInputSource,
)
from kaos_llm_core.programs.envelope import (
    ProgramEnvelopeError,
    from_envelope,
    program_hash,
)
from kaos_llm_core.workspace import (
    BatchRecord,
    WorkspaceMetadata,
    WorkspaceUnsupportedBackendError,
)

logger = get_logger(__name__)

# Pinned VFS context_id for every batch-related lookup. The batch
# subsystem is a shared, per-VFS resource: workspace SQLite, persisted
# envelopes, output directories, and JSONL logs must all be reachable
# from any MCP request, regardless of which client_id / request_id the
# wire happens to attach to a given call. By pinning to a fixed
# sentinel we sidestep VFS context isolation entirely so a batch
# created in request A can be run/polled/read in requests B/C/D.
_BATCH_CONTEXT_ID = "__kaos_llm_core_batch__"


# ---------------------------------------------------------------------------
# Standard error helpers
# ---------------------------------------------------------------------------
#
# Every batch-tool input error follows the same what/how/alternative
# template so callers see a consistent recovery hint without each tool
# inventing its own phrasing.


def missing_batch_id_error() -> ToolResult:
    """Standard error for a missing-or-non-string ``batch_id`` input."""
    return ToolResult.create_error(
        "batch_id is required and must be a non-empty string. "
        "Use the batch_id returned by kaos-llm-core-batch-create, or list "
        "recent batches via the workspace SQLite at "
        "${VFS_ROOT}/.kaos-llm-core/workspace.sqlite."
    )


def unknown_batch_error(batch_id: str) -> ToolResult:
    """Standard error for a batch_id that is not in the workspace."""
    return ToolResult.create_error(
        f"No batch with id {batch_id!r} in this workspace. "
        "Confirm the id from the kaos-llm-core-batch-create response, or "
        "check that the runtime VFS root matches the one the batch was "
        "created against."
    )


# ---------------------------------------------------------------------------
# Workspace open
# ---------------------------------------------------------------------------


def open_workspace(context: KaosContext | None) -> WorkspaceMetadata | ToolResult:
    """Open the per-VFS workspace SQLite store, or return a ToolResult error.

    Returns the :class:`WorkspaceMetadata` instance on success, or a
    :class:`ToolResult` error if the runtime is missing or the VFS
    backend cannot host the workspace file.

    Callers MUST close the returned workspace when done — prefer the
    :func:`workspace_or_error` context-manager helper below to make
    that automatic. Pre-Phase-16.5, several tool callers leaked the
    SQLite connection because the close was missing.
    """
    if context is None or context.runtime is None:
        return ToolResult.create_error(
            "Batch tools require a KaosRuntime context. The workspace SQLite "
            "metadata DB lives at ${VFS_ROOT}/.kaos-llm-core/workspace.sqlite. "
            "Run this tool through an MCP server (kaos-llm-core-serve) so the "
            "runtime is available."
        )
    try:
        return WorkspaceMetadata.from_runtime(context.runtime, context)
    except WorkspaceUnsupportedBackendError as exc:
        return ToolResult.create_error(
            f"Batch workspace unavailable: {exc}. v1 requires the disk VFS "
            "backend. Configure the runtime with VFS_BACKEND=disk."
        )


import contextlib  # noqa: E402
from collections.abc import Iterator  # noqa: E402


@contextlib.contextmanager
def workspace_or_error(
    context: KaosContext | None,
) -> Iterator[WorkspaceMetadata | ToolResult]:
    """Context-managed wrapper around :func:`open_workspace`.

    Yields either a live :class:`WorkspaceMetadata` or a
    :class:`ToolResult` error. On exit, closes the workspace if one
    was opened. Phase 16.5 fix for the SQLite connection leak the
    reviewer caught — callers were forgetting to close the handle and
    pytest emitted ``ResourceWarning: unclosed database`` for every
    batch tool invocation.

    Usage::

        with workspace_or_error(context) as ws:
            if isinstance(ws, ToolResult):
                return ws
            record = ws.get_batch(batch_id)
            ...
    """
    ws_or_err = open_workspace(context)
    try:
        yield ws_or_err
    finally:
        if isinstance(ws_or_err, WorkspaceMetadata):
            ws_or_err.close()


# ---------------------------------------------------------------------------
# Envelope loading
# ---------------------------------------------------------------------------


async def load_envelope(
    *,
    envelope: Any | None,
    envelope_path: str | None,
    context: KaosContext | None,
) -> tuple[dict[str, Any], str] | ToolResult:
    """Load a Program v3 envelope from inline JSON or a VFS path.

    Returns ``(envelope_dict, source_path)`` on success where
    ``source_path`` is the VFS path that was read, or a synthesized
    placeholder for inline envelopes (``"<inline>"``).
    """
    if envelope is None and envelope_path is None:
        return ToolResult.create_error(
            "Provide either 'envelope' (inline JSON object) or "
            "'envelope_path' (VFS path to a saved envelope JSON file)."
        )
    if envelope is not None and envelope_path is not None:
        return ToolResult.create_error("Provide either 'envelope' or 'envelope_path', not both.")
    if envelope_path is not None:
        if context is None or context.runtime is None:
            return ToolResult.create_error(
                f"envelope_path={envelope_path!r} requires a runtime VFS. "
                "Inline envelopes work without a runtime; envelope_path does not."
            )
        try:
            data = await context.runtime.vfs.read(envelope_path, context_id=_BATCH_CONTEXT_ID)
            envelope_dict = json.loads(data.decode("utf-8"))
        except json.JSONDecodeError as exc:
            return ToolResult.create_error(
                f"envelope_path={envelope_path!r}: file is not valid JSON ({exc})."
            )
        except FileNotFoundError:
            return ToolResult.create_error(
                f"envelope_path={envelope_path!r}: file not found in the VFS."
            )
        except Exception as exc:
            return ToolResult.create_error(f"envelope_path={envelope_path!r}: read failed: {exc}.")
        return envelope_dict, envelope_path

    # Inline envelope
    if isinstance(envelope, str):
        try:
            envelope_dict = json.loads(envelope)
        except json.JSONDecodeError as exc:
            return ToolResult.create_error(f"envelope is a string but not valid JSON: {exc}.")
    elif isinstance(envelope, dict):
        envelope_dict = envelope
    else:
        return ToolResult.create_error(
            f"envelope must be a JSON object, got {type(envelope).__name__}."
        )
    return envelope_dict, "<inline>"


def validate_envelope(envelope_dict: dict[str, Any]) -> str | ToolResult:
    """Validate an envelope and return its program hash, or a ToolResult error."""
    try:
        from_envelope(envelope_dict)
    except ProgramEnvelopeError as exc:
        return ToolResult.create_error(
            f"Envelope rejected: {exc}. See "
            "docs/internal/design/program-envelope-v3.md for the schema."
        )
    return program_hash(envelope_dict)


# ---------------------------------------------------------------------------
# Input source descriptor → BatchInputSource
# ---------------------------------------------------------------------------


def build_input_source(
    descriptor: dict[str, Any],
    *,
    program_hash_value: str,
    context: KaosContext | None,
) -> tuple[BatchInputSource, int | None] | ToolResult:
    """Construct a :class:`BatchInputSource` from a JSON descriptor.

    Supported types (v1):
      - ``{"type": "list", "items": [{...}, ...]}`` — inline list of inputs.
      - ``{"type": "jsonl", "path": "/vfs/inputs.jsonl"}`` — JSONL on disk.
      - ``{"type": "pdf_pages", "path": "/vfs/file.pdf",
            "page_range": [start, end]?, "skip_blank": true?}`` —
            one item per PDF page (Phase 15.4, requires the ``pdf`` extra).
      - ``{"type": "tabular_rows", "path": "/vfs/file.csv",
            "sql": "SELECT ...", "max_rows": 1000}`` — one item per row
            of a tabular file (Phase 15.4, requires the ``tabular`` extra).

    Returns ``(source, count_hint)`` where ``count_hint`` is None if
    counting would be expensive.
    """
    if not isinstance(descriptor, dict):
        return ToolResult.create_error(
            "inputs_source must be a JSON object with a 'type' key. "
            "Examples: {'type':'list','items':[...]}, "
            "{'type':'jsonl','path':'/vfs/inputs.jsonl'}."
        )
    src_type = descriptor.get("type")
    if src_type == "list":
        items = descriptor.get("items")
        if not isinstance(items, list):
            return ToolResult.create_error(
                "inputs_source type='list' requires an 'items' list of input dicts."
            )
        source = ListInputSource(items, program_hash_value=program_hash_value)
        return source, len(items)
    if src_type == "jsonl":
        path = descriptor.get("path")
        if not isinstance(path, str):
            return ToolResult.create_error(
                "inputs_source type='jsonl' requires a 'path' string (VFS path)."
            )
        # Resolve the JSONL through the VFS to a real disk path.
        disk_path = _resolve_vfs_to_disk(path, context)
        if isinstance(disk_path, ToolResult):
            return disk_path
        source = JsonlInputSource(disk_path, program_hash_value=program_hash_value)
        # Cheap line count for the hint
        try:
            with disk_path.open("rb") as handle:
                count = sum(1 for line in handle if line.strip())
        except OSError:
            count = None
        return source, count
    if src_type == "pdf_pages":
        path = descriptor.get("path")
        if not isinstance(path, str):
            return ToolResult.create_error(
                "inputs_source type='pdf_pages' requires a 'path' string (VFS path)."
            )
        disk_path = _resolve_vfs_to_disk(path, context)
        if isinstance(disk_path, ToolResult):
            return disk_path
        page_range_arg = descriptor.get("page_range")
        page_range: tuple[int, int] | None = None
        if page_range_arg is not None:
            if not isinstance(page_range_arg, (list, tuple)) or len(page_range_arg) != 2:
                return ToolResult.create_error(
                    "pdf_pages page_range must be a [start, end] pair (0-based, half-open)."
                )
            page_range = (int(page_range_arg[0]), int(page_range_arg[1]))
        try:
            from kaos_llm_core.programs.batch_bridges import (
                BatchBridgeUnavailableError,
                PdfPagesInputSource,
            )

            source = PdfPagesInputSource(
                disk_path,
                program_hash_value=program_hash_value,
                page_range=page_range,
                skip_blank=bool(descriptor.get("skip_blank", True)),
            )
            # Count via kaos-pdf for the hint (cheap; just opens metadata).
            try:
                # kaos-pdf is the optional [pdf] extra; not declared in
                # 0.1.0a1 because it isn't on PyPI yet (returns in 0.1.0a2).
                from kaos_pdf.extract import (  # ty: ignore[unresolved-import]
                    get_page_count,
                )

                total = get_page_count(disk_path)
                if page_range is None:
                    count = total
                else:
                    count = max(0, min(total, page_range[1]) - max(0, page_range[0]))
            except Exception as exc:
                logger.debug("pdf_pages bridge: get_page_count failed: %s", exc)
                count = None
            return source, count
        except BatchBridgeUnavailableError as exc:
            return ToolResult.create_error(
                f"PDF pages bridge is unavailable: {exc}. "
                "Install the optional dependency with: "
                "pip install 'kaos-llm-core[pdf]' (or uv add kaos-pdf). "
                "Alternative: use inputs_source type='jsonl' with pre-extracted text."
            )
        except ImportError as exc:
            return ToolResult.create_error(
                f"pdf_pages bridge unavailable: {exc}. "
                "Install with: pip install 'kaos-llm-core[pdf]'."
            )
    if src_type == "tabular_rows":
        path = descriptor.get("path")
        if not isinstance(path, str):
            return ToolResult.create_error(
                "inputs_source type='tabular_rows' requires a 'path' string (VFS path)."
            )
        disk_path = _resolve_vfs_to_disk(path, context)
        if isinstance(disk_path, ToolResult):
            return disk_path
        sql = descriptor.get("sql")
        if sql is not None and not isinstance(sql, str):
            return ToolResult.create_error("tabular_rows 'sql' must be a string when provided.")
        max_rows = int(descriptor.get("max_rows") or 10_000)
        try:
            from kaos_llm_core.programs.batch_bridges import (
                BatchBridgeUnavailableError,
                TabularRowsInputSource,
            )

            source = TabularRowsInputSource(
                disk_path,
                program_hash_value=program_hash_value,
                sql=sql,
                max_rows=max_rows,
            )
            # No cheap count without running the query — leave hint None.
            return source, None
        except BatchBridgeUnavailableError as exc:
            return ToolResult.create_error(
                f"Tabular rows bridge is unavailable: {exc}. "
                "Install the optional dependency with: "
                "pip install 'kaos-llm-core[tabular]' (or uv add kaos-tabular). "
                "Alternative: use inputs_source type='jsonl' with pre-exported rows."
            )
        except ImportError as exc:
            return ToolResult.create_error(
                f"tabular_rows bridge unavailable: {exc}. "
                "Install with: pip install 'kaos-llm-core[tabular]'."
            )
    return ToolResult.create_error(
        f"Unsupported inputs_source type {src_type!r}. v1 supports 'list', 'jsonl', "
        "'pdf_pages' (requires [pdf] extra), and 'tabular_rows' (requires [tabular] extra)."
    )


def _resolve_vfs_to_disk(
    vfs_path: str,
    context: KaosContext | None,
) -> Path | ToolResult:
    """Resolve a VFS path to a real disk Path, or return a ToolResult error.

    Absolute disk paths are rejected — every batch input/output path must
    flow through ``runtime.vfs.resolve_disk_path`` so that VFS containment
    holds. An MCP caller cannot escape the workspace by passing
    ``/etc/cron.d`` or similar.
    """
    if Path(vfs_path).is_absolute():
        return ToolResult.create_error(
            f"Absolute disk paths are rejected for batch inputs ({vfs_path!r}). "
            "Pass a VFS-relative path; the path will be resolved against the "
            "current runtime's VFS root."
        )
    if context is None or context.runtime is None:
        return ToolResult.create_error(
            f"Cannot resolve VFS path {vfs_path!r} without a runtime context. "
            "Run this tool through an MCP server (kaos-llm-core-serve) so the "
            "runtime + VFS are available."
        )
    try:
        disk = context.runtime.vfs.resolve_disk_path(vfs_path, context_id=_BATCH_CONTEXT_ID)
    except (AttributeError, NotImplementedError) as exc:
        return ToolResult.create_error(
            f"VFS backend cannot resolve {vfs_path!r} to a disk path: {exc}. "
            "Batch I/O requires the disk VFS backend; configure the runtime "
            "with VFS_BACKEND=disk."
        )
    if disk is None:
        return ToolResult.create_error(
            f"VFS path {vfs_path!r} is not on the disk backend. "
            "Batch I/O requires VFS_BACKEND=disk so inputs/outputs can be "
            "streamed to real files. Either switch the runtime to the disk "
            "backend, or use an inline 'list' input source instead of a path."
        )
    return Path(disk)


def resolve_output_dir(
    vfs_path: str,
    context: KaosContext | None,
) -> Path | ToolResult:
    """Resolve an output directory VFS path, creating parents if needed.

    Absolute disk paths are rejected so that batch results cannot escape
    the workspace VFS — see ``_resolve_vfs_to_disk`` for the rationale.
    """
    if Path(vfs_path).is_absolute():
        return ToolResult.create_error(
            f"Absolute disk paths are rejected for batch output_dir ({vfs_path!r}). "
            "Pass a VFS-relative path; the directory will be created under the "
            "current runtime's VFS root."
        )
    if context is None or context.runtime is None:
        return ToolResult.create_error(
            f"Cannot resolve output_dir {vfs_path!r} without a runtime context. "
            "Run this tool through an MCP server (kaos-llm-core-serve) so the "
            "runtime + VFS are available."
        )
    try:
        disk = context.runtime.vfs.resolve_disk_path(vfs_path, context_id=_BATCH_CONTEXT_ID)
    except (AttributeError, NotImplementedError) as exc:
        return ToolResult.create_error(
            f"VFS backend cannot resolve output_dir {vfs_path!r}: {exc}. "
            "Batch I/O requires the disk VFS backend; configure the runtime "
            "with VFS_BACKEND=disk."
        )
    if disk is None:
        return ToolResult.create_error(
            f"output_dir {vfs_path!r} is not on the disk backend. "
            "Batch output requires VFS_BACKEND=disk so the JSONL log and "
            "manifest can be written to real files. Switch the runtime to "
            "the disk backend, or omit output_dir to fall back to the "
            "workspace default directory."
        )
    Path(disk).mkdir(parents=True, exist_ok=True)
    return Path(disk)


# ---------------------------------------------------------------------------
# batch_id generation
# ---------------------------------------------------------------------------


def generate_batch_id(name: str | None = None) -> str:
    """Generate a unique batch_id.

    Format: ``batch-{YYYYMMDD}-{slug?}-{6 hex chars}``. The hex suffix
    makes collisions astronomically unlikely; the date prefix makes
    grep-by-date trivial.
    """
    from datetime import UTC, datetime

    date = datetime.now(UTC).strftime("%Y%m%d")
    suffix = secrets.token_hex(3)
    if name:
        slug = "".join(c if c.isalnum() else "-" for c in name.lower()).strip("-")
        slug = "-".join(filter(None, slug.split("-")))[:32]
        if slug:
            return f"batch-{date}-{slug}-{suffix}"
    return f"batch-{date}-{suffix}"


# ---------------------------------------------------------------------------
# BatchRecord → JSON dict for tool results
# ---------------------------------------------------------------------------


def record_to_dict(record: BatchRecord) -> dict[str, Any]:
    """Convert a BatchRecord into a flat JSON-serializable dict."""
    d = asdict(record)
    return d


def record_to_status_dict(record: BatchRecord) -> dict[str, Any]:
    """Build the batch-status output payload from a BatchRecord."""
    duration_so_far: float | None = None
    if record.started_at and record.completed_at:
        duration_so_far = record.duration_s
    elif record.started_at and record.last_progress_at:
        from datetime import datetime as _dt

        try:
            t0 = _dt.fromisoformat(record.started_at.replace("Z", "+00:00"))
            t1 = _dt.fromisoformat(record.last_progress_at.replace("Z", "+00:00"))
            duration_so_far = (t1 - t0).total_seconds()
        except ValueError:
            duration_so_far = None

    n_done = record.n_succeeded + record.n_errored
    eta_seconds: float | None = None
    if (
        record.n_total is not None
        and record.n_total > 0
        and n_done > 0
        and duration_so_far is not None
        and record.status == "running"
    ):
        rate = duration_so_far / n_done
        eta_seconds = (record.n_total - n_done) * rate

    errors_by_type: dict[str, int] = {}
    if record.error_summary and isinstance(record.error_summary, dict):
        ebt = record.error_summary.get("errors_by_type")
        if isinstance(ebt, dict):
            errors_by_type = {str(k): int(v) for k, v in ebt.items()}

    return {
        "batch_id": record.batch_id,
        "status": record.status,
        "name": record.name,
        "envelope_path": record.envelope_path,
        "program_name": record.program_name,
        "program_hash": record.program_hash,
        "n_total": record.n_total,
        "n_succeeded": record.n_succeeded,
        "n_errored": record.n_errored,
        "n_skipped": record.n_skipped,
        "cost_usd_so_far": record.cost_usd,
        "tokens_so_far": {
            "input": record.input_tokens,
            "output": record.output_tokens,
            "total": record.total_tokens,
        },
        "started_at": record.started_at,
        "completed_at": record.completed_at,
        "last_progress_at": record.last_progress_at,
        "duration_s_so_far": duration_so_far,
        "eta_seconds": eta_seconds,
        "errors_by_type": errors_by_type,
        "log_path": record.log_path,
        "manifest_path": record.manifest_path,
    }


# Re-export for tools that want raw access to BatchError handling
__all__ = [
    "BatchError",
    "build_input_source",
    "generate_batch_id",
    "load_envelope",
    "open_workspace",
    "record_to_dict",
    "record_to_status_dict",
    "resolve_output_dir",
    "validate_envelope",
]
