"""KaosLLMCoreBatchStatusTool — read the latest state of a batch.

Phase 15.3. Pure read of the workspace SQLite ``batches`` row.
Includes a derived ETA when the batch is running and a count is known.

Sub-design: ``docs/internal/design/batch-workspace-schema.md``.
"""

from __future__ import annotations

from typing import Any, ClassVar

from kaos_core import KaosContext, ToolResult
from kaos_core.types.annotations import ToolAnnotations
from kaos_core.types.enums import ToolCapability, ToolCategory
from kaos_core.types.parameters import ParameterSchema

from kaos_llm_core.integrations.mcp._batch_helpers import (
    record_to_status_dict,
    workspace_or_error,
)
from kaos_llm_core.integrations.mcp._common import BaseLLMCoreTool

# Pure read of one SQLite row. Read-only and idempotent.
_BATCH_STATUS_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)


class KaosLLMCoreBatchStatusTool(BaseLLMCoreTool):
    """Return the latest known state of a batch."""

    _NAME: ClassVar[str] = "kaos-llm-core-batch-status"
    _DISPLAY_NAME: ClassVar[str] = "LLM Core Batch Status"
    _DESCRIPTION: ClassVar[str] = (
        "Read the latest state of a batch from the workspace SQLite. "
        "Returns the status (pending|running|completed|failed|cancelled), "
        "progress counters (n_total, n_succeeded, n_errored, n_skipped), "
        "cost spent so far in USD, token counts, started_at / "
        "last_progress_at timestamps, and a derived ETA in seconds when "
        "the batch is still running and a count hint is known. Pair this "
        "tool with kaos-llm-core-batch-run to poll long-running batches "
        "from a separate MCP request — the workspace SQLite is the "
        "shared state."
    )
    _CATEGORY: ClassVar[ToolCategory] = ToolCategory.INTEGRATION
    _CAPABILITY: ClassVar[ToolCapability] = ToolCapability.QUERY
    _ANNOTATIONS: ClassVar[ToolAnnotations] = _BATCH_STATUS_ANNOTATIONS
    _PARAMETERS: ClassVar[list[ParameterSchema]] = [
        ParameterSchema(
            name="batch_id",
            type="string",
            description="The batch_id returned by kaos-llm-core-batch-create.",
            required=True,
        ),
    ]
    _ERROR_HINT: ClassVar[str] = (
        "Check the batch_id and that the runtime VFS uses the disk backend."
    )

    async def _run(self, inputs: dict[str, Any], context: KaosContext | None = None) -> ToolResult:
        with workspace_or_error(context) as ws:
            if isinstance(ws, ToolResult):
                return ws
            batch_id = inputs.get("batch_id")
            if not isinstance(batch_id, str) or not batch_id:
                return ToolResult.create_error("batch_id is required (string).")
            record = ws.get_batch(batch_id)
            if record is None:
                return ToolResult.create_error(
                    f"No batch with id {batch_id!r}. Confirm the id from the "
                    "output of kaos-llm-core-batch-create."
                )
            output = record_to_status_dict(record)
            n_done = record.n_succeeded + record.n_errored
            total_str = str(record.n_total) if record.n_total is not None else "?"
            summary = (
                f"Batch {batch_id} {record.status}: {n_done}/{total_str} done, "
                f"${record.cost_usd:.6f} so far"
            )
            return ToolResult.create_success(output=output, summary=summary)
