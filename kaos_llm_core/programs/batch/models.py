"""Batch data models — BatchItem, Protocols, BatchProgress, BatchResult."""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


@dataclass(slots=True, frozen=True)
class BatchItem:
    """One item in a batch run.

    The custom_id MUST be deterministic. Resume scans the JSONL log,
    builds a skip-set on custom_ids, and re-runs missing rows. Random
    UUIDs make this impossible.

    The ``deterministic_id`` classmethod is the recommended way to
    derive a custom_id from a program hash + the input dict.
    """

    custom_id: str
    inputs: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def deterministic_id(cls, program_hash_value: str, inputs: dict[str, Any]) -> str:
        """Return a 16-char sha256 hash combining the program and the input row.

        Resume identity. Two runs with the same program + same input row
        produce the same custom_id; two runs with a changed instruction
        / model / hyperparameter / few-shot demo set produce different
        custom_ids and the second run correctly re-executes the items.
        """
        canonical = json.dumps(
            {"program_hash": program_hash_value, "inputs": inputs},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class BatchInputSource(Protocol):
    """Streaming, deterministic, resumable iterator over inputs.

    The source MUST yield items in the same order across runs given
    the same source state. This is a hard requirement of resume:
    if a resumed run sees a different ordering, the skip-set logic
    silently corrupts the output.
    """

    def __aiter__(self) -> AsyncIterator[BatchItem]:
        """Yield BatchItem instances. The iterator must be deterministic across runs."""
        ...

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serializable provenance dict for the manifest header."""
        ...


class BatchWriter(Protocol):
    """Streams per-item results to durable storage as they complete."""

    async def write_header(self, header: dict[str, Any]) -> None: ...

    async def write_item(self, item_record: dict[str, Any]) -> None: ...

    async def write_checkpoint(self, checkpoint: dict[str, Any]) -> None: ...

    async def close(self) -> None: ...


@dataclass(slots=True)
class BatchProgress:
    """Snapshot of an in-progress batch."""

    n_done: int
    n_total: int | None
    n_succeeded: int
    n_errored: int
    n_skipped: int
    cost_usd_so_far: float
    tokens_so_far: int
    last_custom_id: str | None
    last_status: Literal["success", "error", "skipped"] | None


@dataclass(frozen=True, slots=True)
class BatchResult:
    """Final result of a completed batch run."""

    run_id: str
    output_dir: str
    log_path: str
    manifest_path: str
    n_total: int
    n_succeeded: int
    n_errored: int
    n_skipped: int
    cost_usd: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    duration_s: float
    status: Literal["completed", "stopped", "cancelled", "failed"]
    started_at: str
    completed_at: str | None
    program_hash: str
    errors_by_type: dict[str, int] = field(default_factory=dict)
