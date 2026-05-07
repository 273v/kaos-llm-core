"""``batch_run()`` — streaming, resumable batch execution primitive.

Phase 15.2. The missing batch primitive that closes the PRD §6 promise
that "a pipeline author doing batch classification or extraction across
thousands of inputs" is a Primary User. ``evaluate()`` is eval-shaped
(gold labels per item, aggregate metric out); ``batch_run()`` is
batch-shaped (no labels, per-item streamed to disk, checkpoint +
resume + cost attribution).

Sub-design: ``docs/internal/design/batch-run.md``.

Constraints (load-bearing, from the sub-design):

- **JSONL append-only log, NOT SQLite.** Crash-safe trivially.
  Streamable to MCP as a resource. Direct lift from the OpenAI/Anthropic
  provider-batch wire format.
- **Write-after-each-item (WAE) checkpointing.** Append + ``fsync`` per
  row. Crash-safe to the last completed item.
- **Deterministic ``custom_id``s, NOT UUIDs.** UUIDs make resume
  impossible because the second run cannot recognize the same input.
- **Resume scans the JSONL log, builds a skip-set, re-streams the
  source.** No separate state file. The JSONL log IS the source of
  truth. The source iterator MUST be deterministic across runs.
- **Cost attribution via the existing Phase 13A ``publish_invocation``
  hook.** Wrap the entire batch in ``runner.trial("batch_run")``.
  Every per-item Call charges into the trial's cost total
  automatically. Zero new cost-tracking code.
- **Three error policies**: ``continue`` (default), ``stop``,
  ``stop_after_n``. The third is the money-safety valve.
- **``KaosContext.report_progress()``** is the existing kaos-core
  mechanism the default progress callback forwards to.
- **VFS append API gap**: ``VirtualFileSystem`` has no ``append`` method.
  v1 bypasses VFS via ``runtime.vfs.resolve_disk_path(path,
  context_id)`` for the log file (disk backend only). Phase 16 should
  add ``VirtualFileSystem.append()`` to the backend protocol.

Package layout (Phase 16.5 — was a single 1000-LOC file pre-split):

  - :mod:`._errors` — :class:`BatchError`, :class:`BatchResumeMismatchError`,
    :class:`BatchStopAfterNError`, :class:`BatchUnsupportedBackendError`.
  - :mod:`.models` — :class:`BatchItem`, :class:`BatchInputSource`
    Protocol, :class:`BatchWriter` Protocol, :class:`BatchProgress`,
    :class:`BatchResult`.
  - :mod:`.sources` — :class:`ListInputSource`, :class:`JsonlInputSource`
    plus the factory functions.
  - :mod:`.writer` — :class:`JsonlBatchWriter`.
  - :mod:`.resume` — :func:`_scan_existing_log`.
  - :mod:`.records` — per-item record builders + manifest writer.
  - :mod:`.targets` — target normalization (Call / Program /
    ProgramEnvelope / dict → Program + hash + disk path).
  - :mod:`.runner` — :func:`batch_run` itself.

Public API is re-exported from this :mod:`__init__` so callers can
keep importing from ``kaos_llm_core.programs.batch`` unchanged.
"""

from __future__ import annotations

from kaos_llm_core.programs.batch._errors import (
    BatchError,
    BatchResumeMismatchError,
    BatchStopAfterNError,
    BatchUnsupportedBackendError,
)
from kaos_llm_core.programs.batch.models import (
    BatchInputSource,
    BatchItem,
    BatchProgress,
    BatchResult,
    BatchWriter,
)
from kaos_llm_core.programs.batch.records import (
    _error_record,
    _now_iso,
    _success_record,
    _write_manifest,
)
from kaos_llm_core.programs.batch.resume import _scan_existing_log
from kaos_llm_core.programs.batch.runner import batch_run
from kaos_llm_core.programs.batch.sources import (
    JsonlInputSource,
    ListInputSource,
    jsonl_input_source,
    list_input_source,
)
from kaos_llm_core.programs.batch.targets import (
    _hash_target,
    _resolve_output_disk_path,
    _resolve_target,
)
from kaos_llm_core.programs.batch.writer import JsonlBatchWriter

__all__ = [
    "BatchError",
    "BatchInputSource",
    "BatchItem",
    "BatchProgress",
    "BatchResult",
    "BatchResumeMismatchError",
    "BatchStopAfterNError",
    "BatchUnsupportedBackendError",
    "BatchWriter",
    "JsonlBatchWriter",
    "JsonlInputSource",
    "ListInputSource",
    # Private but re-exported for tests + cross-package callers
    "_error_record",
    "_hash_target",
    "_now_iso",
    "_resolve_output_disk_path",
    "_resolve_target",
    "_scan_existing_log",
    "_success_record",
    "_write_manifest",
    "batch_run",
    "jsonl_input_source",
    "list_input_source",
]
