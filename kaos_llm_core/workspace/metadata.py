"""Per-VFS SQLite metadata store for kaos-llm-core batch jobs.

Phase 15.3. ONE table — ``batches`` — that persists batch state across
MCP requests so an agent can:

  1. Call ``batch-create`` at turn N to define a batch.
  2. Call ``batch-run`` at turn N+1 to execute it.
  3. Call ``batch-status`` at turn N+2 to poll progress.
  4. Call ``batch-results`` at turn N+5 to read the manifest.

Each call is a separate MCP request with no shared in-memory state. The
SQLite file is the source of truth for batch *state*; the JSONL log
that ``batch_run()`` writes is the source of truth for per-item
*results*.

Sub-design: ``docs/internal/design/batch-workspace-schema.md``.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from kaos_llm_core.errors import KaosLLMCoreError
from kaos_llm_core.programs.batch import BatchProgress

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class WorkspaceError(KaosLLMCoreError):
    """Base error for workspace metadata operations."""


class WorkspaceUnsupportedBackendError(WorkspaceError):
    """The current VFS backend cannot host the workspace SQLite file.

    v1 requires the disk backend because the metadata DB needs a real
    file path. Memory and S3 backends raise this on workspace open.
    """


class BatchAlreadyExistsError(WorkspaceError):
    """A batch with the given ``batch_id`` already exists in the table.

    PRIMARY KEY collision on the ``batches`` table is mapped to this
    typed error so MCP handlers can return a clean message rather than
    leaking ``sqlite3.IntegrityError``.
    """


# ---------------------------------------------------------------------------
# Status enum
# ---------------------------------------------------------------------------


BatchStatus = Literal["pending", "running", "completed", "failed", "cancelled"]
TERMINAL_STATUSES: frozenset[BatchStatus] = frozenset({"completed", "failed", "cancelled"})


# ---------------------------------------------------------------------------
# In-memory record
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BatchRecord:
    """In-memory mirror of a row in the ``batches`` table.

    Every column is represented. JSON-encoded columns
    (``inputs_source``, ``error_summary``, ``metadata_json``) are
    decoded into Python dicts on read.
    """

    # Identity
    batch_id: str
    name: str | None
    created_at: str
    created_by: str | None
    # Program reference
    envelope_path: str
    program_hash: str
    program_name: str | None
    # Source
    inputs_source: dict[str, Any]
    inputs_count_hint: int | None
    # Output paths
    output_dir: str
    log_path: str
    manifest_path: str
    # Run config
    max_concurrency: int = 8
    error_policy: str = "continue"
    max_errors: int | None = None
    mode: str = "live"
    item_timeout_s: float | None = None
    checkpoint_every: int = 1
    # Lifecycle
    status: BatchStatus = "pending"
    started_at: str | None = None
    completed_at: str | None = None
    duration_s: float | None = None
    cancel_requested_at: str | None = None
    # Progress
    n_total: int | None = None
    n_succeeded: int = 0
    n_errored: int = 0
    n_skipped: int = 0
    # Cost
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    # Diagnostics
    error_summary: dict[str, Any] | None = None
    last_progress_at: str | None = None
    # Reserved
    parent_batch_id: str | None = None
    metadata_json: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _row_to_record(row: sqlite3.Row) -> BatchRecord:
    return BatchRecord(
        batch_id=row["batch_id"],
        name=row["name"],
        created_at=row["created_at"],
        created_by=row["created_by"],
        envelope_path=row["envelope_path"],
        program_hash=row["program_hash"],
        program_name=row["program_name"],
        inputs_source=json.loads(row["inputs_source"]) if row["inputs_source"] else {},
        inputs_count_hint=row["inputs_count_hint"],
        output_dir=row["output_dir"],
        log_path=row["log_path"],
        manifest_path=row["manifest_path"],
        max_concurrency=row["max_concurrency"],
        error_policy=row["error_policy"],
        max_errors=row["max_errors"],
        mode=row["mode"],
        item_timeout_s=row["item_timeout_s"],
        checkpoint_every=row["checkpoint_every"],
        status=row["status"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        duration_s=row["duration_s"],
        cancel_requested_at=row["cancel_requested_at"],
        n_total=row["n_total"],
        n_succeeded=row["n_succeeded"],
        n_errored=row["n_errored"],
        n_skipped=row["n_skipped"],
        cost_usd=row["cost_usd"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        total_tokens=row["total_tokens"],
        error_summary=json.loads(row["error_summary"]) if row["error_summary"] else None,
        last_progress_at=row["last_progress_at"],
        parent_batch_id=row["parent_batch_id"],
        metadata_json=json.loads(row["metadata_json"]) if row["metadata_json"] else None,
    )


# ---------------------------------------------------------------------------
# Schema (single source of truth — also documented in batch-workspace-schema.md)
# ---------------------------------------------------------------------------


# Phase 16.5 — workspace SQLite schema version. Bump on additive
# changes (new columns / new indexes); design a one-shot migration
# from the prior version when the bump happens. Pre-Phase-16.5
# databases were unstamped (user_version=0) and will be auto-stamped
# to 1 on first open by a kaos-llm-core>=16.5 release.
WORKSPACE_SCHEMA_VERSION = 1


_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS batches (
    batch_id            TEXT PRIMARY KEY NOT NULL,
    name                TEXT,
    created_at          TEXT NOT NULL,
    created_by          TEXT,
    envelope_path       TEXT NOT NULL,
    program_hash        TEXT NOT NULL,
    program_name        TEXT,
    inputs_source       TEXT NOT NULL,
    inputs_count_hint   INTEGER,
    output_dir          TEXT NOT NULL,
    log_path            TEXT NOT NULL,
    manifest_path       TEXT NOT NULL,
    max_concurrency     INTEGER NOT NULL DEFAULT 8,
    error_policy        TEXT NOT NULL DEFAULT 'continue',
    max_errors          INTEGER,
    mode                TEXT NOT NULL DEFAULT 'live',
    item_timeout_s      REAL,
    checkpoint_every    INTEGER NOT NULL DEFAULT 1,
    status              TEXT NOT NULL DEFAULT 'pending',
    started_at          TEXT,
    completed_at        TEXT,
    duration_s          REAL,
    cancel_requested_at TEXT,
    n_total             INTEGER,
    n_succeeded         INTEGER NOT NULL DEFAULT 0,
    n_errored           INTEGER NOT NULL DEFAULT 0,
    n_skipped           INTEGER NOT NULL DEFAULT 0,
    cost_usd            REAL NOT NULL DEFAULT 0.0,
    input_tokens        INTEGER NOT NULL DEFAULT 0,
    output_tokens       INTEGER NOT NULL DEFAULT 0,
    total_tokens        INTEGER NOT NULL DEFAULT 0,
    error_summary       TEXT,
    last_progress_at    TEXT,
    parent_batch_id     TEXT,
    metadata_json       TEXT
);

CREATE INDEX IF NOT EXISTS idx_batches_status ON batches(status);
CREATE INDEX IF NOT EXISTS idx_batches_created_at ON batches(created_at);
CREATE INDEX IF NOT EXISTS idx_batches_program_hash ON batches(program_hash);
"""


# ---------------------------------------------------------------------------
# WorkspaceMetadata
# ---------------------------------------------------------------------------


class WorkspaceMetadata:
    """Per-VFS SQLite metadata store for batch jobs.

    Thread-safe and multi-process-safe via SQLite WAL mode. The
    connection uses ``isolation_level=None`` (autocommit) so each CRUD
    method's single statement commits immediately.
    """

    def __init__(self, db_path: Path | str):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._closed = False
        self._con = self._connect()

    @classmethod
    def from_runtime(
        cls,
        runtime: Any,
        context: Any | None = None,
    ) -> WorkspaceMetadata:
        """Resolve the workspace SQLite path via the runtime VFS.

        Raises ``WorkspaceUnsupportedBackendError`` if the VFS backend
        does not expose a real disk path (memory / S3 backends).
        """
        if runtime is None or not hasattr(runtime, "vfs"):
            msg = (
                "WorkspaceMetadata requires a KaosRuntime with a VFS to "
                "resolve the workspace path. Pass a runtime explicitly or "
                "construct WorkspaceMetadata(db_path=...) directly."
            )
            raise WorkspaceUnsupportedBackendError(msg)
        # The workspace SQLite is a shared, per-VFS resource — every MCP
        # session in this process must see the same DB file. We pin the
        # context_id to a fixed sentinel so VFS context isolation does
        # NOT split the workspace across per-session subdirectories.
        _ = context  # parameter retained for API symmetry; intentionally unused
        try:
            disk_path = runtime.vfs.resolve_disk_path(
                ".kaos-llm-core/workspace.sqlite",
                context_id="__kaos_llm_core_workspace__",
            )
        except (AttributeError, NotImplementedError) as exc:
            msg = (
                f"WorkspaceMetadata: VFS backend does not support "
                f"resolve_disk_path ({exc}). v1 requires the disk backend. "
                "Configure the runtime with the disk VFS backend or pass an "
                "explicit db_path to WorkspaceMetadata()."
            )
            raise WorkspaceUnsupportedBackendError(msg) from exc
        if disk_path is None:
            msg = (
                "WorkspaceMetadata: VFS backend returned no disk path "
                "(non-disk backend in use). v1 requires the disk backend "
                "for the workspace SQLite file."
            )
            raise WorkspaceUnsupportedBackendError(msg)
        return cls(Path(disk_path))

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(
            str(self._db_path),
            isolation_level=None,
            check_same_thread=False,
            timeout=30.0,
        )
        con.execute("PRAGMA journal_mode = WAL")
        con.execute("PRAGMA synchronous = NORMAL")
        con.row_factory = sqlite3.Row
        con.executescript(_SCHEMA_DDL)
        # Phase 16.5: stamp the schema version so future migrations can
        # detect the current shape. Bump WORKSPACE_SCHEMA_VERSION
        # whenever a column is added/removed/renamed and ship a
        # one-shot migration that runs from the prior version.
        current_version = con.execute("PRAGMA user_version").fetchone()[0]
        if current_version == 0:
            con.execute(f"PRAGMA user_version = {WORKSPACE_SCHEMA_VERSION}")
        elif current_version > WORKSPACE_SCHEMA_VERSION:
            msg = (
                f"Workspace SQLite at {self._db_path} has user_version="
                f"{current_version}, which is newer than the maximum "
                f"version this kaos-llm-core release knows ({WORKSPACE_SCHEMA_VERSION}). "
                "Upgrade kaos-llm-core or open the database with a newer release."
            )
            raise WorkspaceError(msg)
        return con

    def close(self) -> None:
        """Close the underlying SQLite connection.

        Idempotent — safe to call multiple times. After ``close()``,
        any further CRUD call will raise ``sqlite3.ProgrammingError``
        from the underlying sqlite3 connection. We do NOT set
        ``self._con = None`` so the type annotation stays
        ``sqlite3.Connection`` and ty does not have to narrow at every
        CRUD call site.
        """
        if not self._closed:
            self._closed = True
            with contextlib.suppress(Exception):
                self._con.close()

    def __enter__(self) -> WorkspaceMetadata:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        # Safety net — eviction by the garbage collector should NOT
        # leak connections. Phase 16.5 reviewer caught a leak in the
        # batch MCP tool path because callers were not closing
        # explicitly. The tools are now fixed to close in finally
        # blocks; this destructor is the second line of defense.
        with contextlib.suppress(Exception):
            self.close()

    # ---------------------------------------------------------------- CRUD

    def create_batch(self, record: BatchRecord) -> None:
        """Insert a new batch row.

        Raises ``BatchAlreadyExistsError`` on PRIMARY KEY collision.
        """
        try:
            self._con.execute(
                """
                INSERT INTO batches (
                    batch_id, name, created_at, created_by,
                    envelope_path, program_hash, program_name,
                    inputs_source, inputs_count_hint,
                    output_dir, log_path, manifest_path,
                    max_concurrency, error_policy, max_errors, mode,
                    item_timeout_s, checkpoint_every,
                    status, started_at, completed_at, duration_s, cancel_requested_at,
                    n_total, n_succeeded, n_errored, n_skipped,
                    cost_usd, input_tokens, output_tokens, total_tokens,
                    error_summary, last_progress_at,
                    parent_batch_id, metadata_json
                ) VALUES (
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?,
                    ?, ?
                )
                """,
                (
                    record.batch_id,
                    record.name,
                    record.created_at,
                    record.created_by,
                    record.envelope_path,
                    record.program_hash,
                    record.program_name,
                    json.dumps(record.inputs_source, sort_keys=True),
                    record.inputs_count_hint,
                    record.output_dir,
                    record.log_path,
                    record.manifest_path,
                    record.max_concurrency,
                    record.error_policy,
                    record.max_errors,
                    record.mode,
                    record.item_timeout_s,
                    record.checkpoint_every,
                    record.status,
                    record.started_at,
                    record.completed_at,
                    record.duration_s,
                    record.cancel_requested_at,
                    record.n_total,
                    record.n_succeeded,
                    record.n_errored,
                    record.n_skipped,
                    record.cost_usd,
                    record.input_tokens,
                    record.output_tokens,
                    record.total_tokens,
                    json.dumps(record.error_summary) if record.error_summary else None,
                    record.last_progress_at,
                    record.parent_batch_id,
                    json.dumps(record.metadata_json) if record.metadata_json else None,
                ),
            )
        except sqlite3.IntegrityError as exc:
            msg = (
                f"Batch ID {record.batch_id!r} already exists. Use a different "
                "name or call batch-status to inspect the existing record."
            )
            raise BatchAlreadyExistsError(msg) from exc

    def get_batch(self, batch_id: str) -> BatchRecord | None:
        cur = self._con.execute("SELECT * FROM batches WHERE batch_id = ?", (batch_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return _row_to_record(row)

    def list_batches(
        self,
        *,
        status: BatchStatus | None = None,
        limit: int = 50,
    ) -> list[BatchRecord]:
        if status is not None:
            cur = self._con.execute(
                "SELECT * FROM batches WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            )
        else:
            cur = self._con.execute(
                "SELECT * FROM batches ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
        return [_row_to_record(r) for r in cur.fetchall()]

    def update_progress(self, batch_id: str, progress: BatchProgress) -> None:
        """Update the progress columns from a BatchProgress snapshot."""
        self._con.execute(
            """
            UPDATE batches
               SET n_succeeded      = ?,
                   n_errored        = ?,
                   n_skipped        = ?,
                   cost_usd         = ?,
                   total_tokens     = ?,
                   last_progress_at = ?
             WHERE batch_id = ?
            """,
            (
                progress.n_succeeded,
                progress.n_errored,
                progress.n_skipped,
                progress.cost_usd_so_far,
                progress.tokens_so_far,
                _now_iso(),
                batch_id,
            ),
        )

    def update_status(
        self,
        batch_id: str,
        status: BatchStatus,
        **terminal_fields: Any,
    ) -> None:
        """Update status plus an arbitrary set of terminal columns.

        Allowed ``terminal_fields`` keys: ``started_at``, ``completed_at``,
        ``duration_s``, ``n_total``, ``n_succeeded``, ``n_errored``,
        ``n_skipped``, ``cost_usd``, ``input_tokens``, ``output_tokens``,
        ``total_tokens``, ``error_summary``.
        """
        allowed = {
            "started_at",
            "completed_at",
            "duration_s",
            "n_total",
            "n_succeeded",
            "n_errored",
            "n_skipped",
            "cost_usd",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "error_summary",
        }
        sets: list[str] = ["status = ?"]
        params: list[Any] = [status]
        for key, value in terminal_fields.items():
            if key not in allowed:
                msg = f"update_status: unknown terminal field {key!r}"
                raise WorkspaceError(msg)
            sets.append(f"{key} = ?")
            if key == "error_summary" and value is not None:
                params.append(json.dumps(value))
            else:
                params.append(value)
        params.append(batch_id)
        sql = f"UPDATE batches SET {', '.join(sets)} WHERE batch_id = ?"
        self._con.execute(sql, params)

    def cancel_batch(self, batch_id: str) -> None:
        """Mark a batch as cancel-requested. The runner observes and drains."""
        self._con.execute(
            "UPDATE batches SET cancel_requested_at = ? WHERE batch_id = ?",
            (_now_iso(), batch_id),
        )


# Re-export for the schema constant inspection in tests
__all__ = [
    "TERMINAL_STATUSES",
    "WORKSPACE_SCHEMA_VERSION",
    "_SCHEMA_DDL",
    "BatchAlreadyExistsError",
    "BatchRecord",
    "BatchStatus",
    "WorkspaceError",
    "WorkspaceMetadata",
    "WorkspaceUnsupportedBackendError",
    "_now_iso",
]


# Silence unused-import lint for `field` (kept available for downstream
# subclasses that want to extend BatchRecord with field defaults).
_ = field
