"""Append-only JSONL log writer with per-record fsync."""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Any


class JsonlBatchWriter:
    """Append-only JSONL log writer with per-record fsync.

    Crash-safe to the last completed item. The log file is opened in
    append mode (``a``) so concurrent runs and resumes do not truncate
    each other.

    Records are JSON objects, one per line, with a ``kind`` discriminator:
    ``header``, ``item``, ``checkpoint``. See
    ``docs/internal/design/batch-run.md`` § "Wire format" for the schema.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._handle: Any = None

    def _ensure_open(self) -> None:
        if self._handle is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self._path.open("a", encoding="utf-8")

    async def _write_record(self, record: dict[str, Any]) -> None:
        self._ensure_open()
        line = json.dumps(record, separators=(",", ":"), default=str) + "\n"
        self._handle.write(line)
        self._handle.flush()
        # fsync may fail on some filesystems (e.g. tmpfs); flush is best-effort.
        with contextlib.suppress(OSError):
            os.fsync(self._handle.fileno())

    async def write_header(self, header: dict[str, Any]) -> None:
        await self._write_record({"kind": "header", **header})

    async def write_item(self, item_record: dict[str, Any]) -> None:
        await self._write_record({"kind": "item", **item_record})

    async def write_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        await self._write_record({"kind": "checkpoint", **checkpoint})

    async def close(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None

    @property
    def path(self) -> Path:
        return self._path
