"""Per-item record builders + manifest writer for the batch primitive."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kaos_llm_core.programs.batch.models import BatchItem


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _success_record(
    item: BatchItem,
    invocation: Any,
    duration_ms: float,
) -> dict[str, Any]:
    output = invocation.output
    if hasattr(output, "model_dump"):
        output_dict = output.model_dump()
    elif isinstance(output, dict):
        output_dict = output
    else:
        output_dict = {"result": str(output)}
    usage = invocation.usage
    return {
        "custom_id": item.custom_id,
        "input_ref": item.metadata,
        "status": "success",
        "output": output_dict,
        "usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
            "cost_usd": round(usage.cost_usd, 6),
        },
        "duration_ms": round(duration_ms, 2),
        "attempts": 1,
        "ended_at": _now_iso(),
    }


def _error_record(
    item: BatchItem,
    exc: BaseException,
    duration_ms: float,
) -> dict[str, Any]:
    # Read partial usage from a tagged Invocation if Phase 10A's exception
    # tagging caught a partial result.
    partial_inv = getattr(exc, "invocation", None)
    if partial_inv is not None:
        usage = partial_inv.usage
        usage_dict = {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
            "cost_usd": round(usage.cost_usd, 6),
        }
    else:
        usage_dict = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
        }
    return {
        "custom_id": item.custom_id,
        "input_ref": item.metadata,
        "status": "error",
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
            "recoverable": False,
        },
        "usage": usage_dict,
        "duration_ms": round(duration_ms, 2),
        "attempts": 1,
        "ended_at": _now_iso(),
    }


def _write_manifest(
    manifest_path: Path,
    *,
    run_id: str,
    program_hash_value: str,
    log_path: Path,
    n_total: int,
    n_succeeded: int,
    n_errored: int,
    n_skipped: int,
    cost_usd: float,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    started_at: str,
    completed_at: str | None,
    duration_s: float,
    status: str,
    errors_by_type: dict[str, int],
) -> None:
    """Write the human-and-tool-friendly manifest summary."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "kaos_batch_manifest": "1",
        "run_id": run_id,
        "log_path": log_path.name,
        "program_hash": program_hash_value,
        "n_total": n_total,
        "n_succeeded": n_succeeded,
        "n_errored": n_errored,
        "n_skipped": n_skipped,
        "cost_usd": round(cost_usd, 6),
        "tokens": {
            "input": input_tokens,
            "output": output_tokens,
            "total": total_tokens,
        },
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_s": round(duration_s, 3),
        "status": status,
        "errors_by_type": errors_by_type,
    }
    # Atomic write via tempfile + rename.
    tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    tmp_path.replace(manifest_path)
