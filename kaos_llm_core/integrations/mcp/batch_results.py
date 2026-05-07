"""KaosLLMCoreBatchResultsTool — return the final manifest / log for a batch.

Phase 15.3. Returns the manifest as the default payload, or the JSONL
log inline (size-capped) for small batches, or a compact summary.

Sub-design: ``docs/internal/design/batch-workspace-schema.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

from kaos_core import KaosContext, ToolResult
from kaos_core.types.annotations import ToolAnnotations
from kaos_core.types.enums import ToolCapability, ToolCategory
from kaos_core.types.parameters import ParameterSchema

from kaos_llm_core.integrations.mcp._batch_helpers import (
    missing_batch_id_error,
    unknown_batch_error,
    workspace_or_error,
)
from kaos_llm_core.integrations.mcp._common import BaseLLMCoreTool

# Pure read of the manifest file + SQLite row. Read-only and idempotent.
_BATCH_RESULTS_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)

# 16 KB inline cap on JSONL output (matches kaos-core's INLINE_THRESHOLD).
_INLINE_THRESHOLD = 16 * 1024


class KaosLLMCoreBatchResultsTool(BaseLLMCoreTool):
    """Return the final results of a completed batch."""

    _NAME: ClassVar[str] = "kaos-llm-core-batch-results"
    _DISPLAY_NAME: ClassVar[str] = "LLM Core Batch Results"
    _DESCRIPTION: ClassVar[str] = (
        "Return the final results of a batch. The default 'manifest' "
        "format returns the aggregate manifest (cost, tokens, status, "
        "errors_by_type) plus paths to the JSONL log and manifest file. "
        "format='jsonl' returns the per-item rows inline (capped at the "
        "16 KB threshold; for larger logs, read the log file via "
        "kaos-core-vfs-read). format='summary' returns the manifest "
        "without the file paths (smaller payload). For batches that are "
        "still running, the manifest may not exist yet — in that case "
        "the tool returns whatever the workspace row knows so far."
    )
    _CATEGORY: ClassVar[ToolCategory] = ToolCategory.INTEGRATION
    _CAPABILITY: ClassVar[ToolCapability] = ToolCapability.QUERY
    _ANNOTATIONS: ClassVar[ToolAnnotations] = _BATCH_RESULTS_ANNOTATIONS
    _PARAMETERS: ClassVar[list[ParameterSchema]] = [
        ParameterSchema(
            name="batch_id",
            type="string",
            description="The batch_id returned by kaos-llm-core-batch-create.",
            required=True,
        ),
        ParameterSchema(
            name="format",
            type="string",
            description=(
                "Output format: 'manifest' (default — aggregate stats + "
                "paths), 'summary' (manifest without paths, smaller), or "
                "'jsonl' (per-item rows inline, capped at 16 KB)."
            ),
            required=False,
        ),
    ]

    async def _run(self, inputs: dict[str, Any], context: KaosContext | None = None) -> ToolResult:
        with workspace_or_error(context) as ws:
            if isinstance(ws, ToolResult):
                return ws
            batch_id = inputs.get("batch_id")
            if not isinstance(batch_id, str) or not batch_id:
                return missing_batch_id_error()
            record = ws.get_batch(batch_id)
            if record is None:
                return unknown_batch_error(batch_id)
            fmt = inputs.get("format", "manifest")
            if fmt not in ("manifest", "summary", "jsonl"):
                return ToolResult.create_error(
                    f"format must be 'manifest', 'summary', or 'jsonl'. Got {fmt!r}. "
                    "Use 'manifest' (default; aggregate stats + paths), 'summary' "
                    "(stats only), or 'jsonl' (per-item rows inline, capped at 16 KB)."
                )

            # Common aggregate payload built from the SQLite row + on-disk
            # manifest.json (if present).
            manifest_path = Path(record.manifest_path)
            manifest_data: dict[str, Any] | None = None
            if manifest_path.exists():
                try:
                    manifest_data = json.loads(manifest_path.read_text())
                except (OSError, json.JSONDecodeError):
                    manifest_data = None

            base = {
                "batch_id": batch_id,
                "status": record.status,
                "n_total": record.n_total,
                "n_succeeded": record.n_succeeded,
                "n_errored": record.n_errored,
                "n_skipped": record.n_skipped,
                "cost_usd": round(record.cost_usd, 6),
                "tokens": {
                    "input": record.input_tokens,
                    "output": record.output_tokens,
                    "total": record.total_tokens,
                },
                "duration_s": record.duration_s,
                "errors_by_type": (
                    (record.error_summary or {}).get("errors_by_type", {})
                    if record.error_summary
                    else {}
                ),
            }

            if fmt == "manifest":
                base["log_path"] = record.log_path
                base["manifest_path"] = record.manifest_path
                if manifest_data is not None:
                    base["manifest"] = manifest_data
            elif fmt == "jsonl":
                log_path = Path(record.log_path)
                if not log_path.exists():
                    return ToolResult.create_error(
                        f"Batch {batch_id}: log file {log_path} does not exist yet. "
                        "Wait for the batch to start, or check the workspace row."
                    )
                size = log_path.stat().st_size
                if size > _INLINE_THRESHOLD:
                    return ToolResult.create_error(
                        f"Batch {batch_id}: log is {size} bytes, larger than the "
                        f"{_INLINE_THRESHOLD} byte inline cap. Read the file directly "
                        f"via kaos-core-vfs-read at {record.log_path}."
                    )
                with log_path.open("r", encoding="utf-8") as handle:
                    rows: list[dict[str, Any]] = []
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                base["rows"] = rows
                base["log_path"] = record.log_path
            # fmt == "summary" → just `base`

            summary = (
                f"Batch {batch_id} {record.status}: "
                f"{record.n_succeeded} succeeded, {record.n_errored} errored, "
                f"${record.cost_usd:.6f}"
            )
            return ToolResult.create_success(output=base, summary=summary)
