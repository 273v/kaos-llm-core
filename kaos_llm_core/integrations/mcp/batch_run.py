"""KaosLLMCoreBatchRunTool — execute (or resume) a defined batch.

Phase 15.3. Synchronous in v1: the tool returns when the batch reaches
a terminal status. Per-item progress flows into the workspace SQLite
row via a progress callback so other MCP requests (``batch-status``)
see live updates.

Sub-design: ``docs/internal/design/batch-workspace-schema.md``.
"""

from __future__ import annotations

import contextlib
from typing import Any, ClassVar

from kaos_core import KaosContext, ToolResult
from kaos_core.types.annotations import ToolAnnotations
from kaos_core.types.enums import ToolCapability, ToolCategory
from kaos_core.types.parameters import ParameterSchema

from kaos_llm_core.integrations.mcp._batch_helpers import (
    build_input_source,
    load_envelope,
    workspace_or_error,
)
from kaos_llm_core.integrations.mcp._common import BaseLLMCoreTool, _settings_for
from kaos_llm_core.programs.batch import BatchProgress, batch_run
from kaos_llm_core.programs.envelope import from_envelope
from kaos_llm_core.workspace.metadata import _now_iso

# batch-run is the only batch tool that mutates external state (calls
# providers, writes JSONL log). Not idempotent: re-running with
# resume=True is OK; re-running a completed batch is rejected.
_BATCH_RUN_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=False,
    openWorldHint=True,
)


class KaosLLMCoreBatchRunTool(BaseLLMCoreTool):
    """Execute (or resume) a previously created batch."""

    _NAME: ClassVar[str] = "kaos-llm-core-batch-run"
    _DISPLAY_NAME: ClassVar[str] = "LLM Core Batch Run"
    _DESCRIPTION: ClassVar[str] = (
        "Execute (or resume) a batch defined via kaos-llm-core-batch-create. "
        "Synchronous in v1: returns when the batch reaches a terminal status "
        "(completed | failed | stopped | cancelled). Per-item progress is "
        "persisted to the workspace SQLite row as the run proceeds, so other "
        "MCP requests (kaos-llm-core-batch-status) see live updates. Use "
        "resume=true (default for failed/cancelled batches) to skip items "
        "that already completed in a prior run via the JSONL log skip-set. "
        "On a fresh run, resume defaults to false. The cost cap from the "
        "creation-time error_policy is enforced — error_policy='stop_after_n' "
        "with a small max_errors is the recommended money-safety valve. "
        "For programmatic use, the Python API kaos_llm_core.programs.batch."
        "batch_run() exposes the same primitive without the workspace layer."
    )
    _CATEGORY: ClassVar[ToolCategory] = ToolCategory.INTEGRATION
    _CAPABILITY: ClassVar[ToolCapability] = ToolCapability.QUERY
    _ANNOTATIONS: ClassVar[ToolAnnotations] = _BATCH_RUN_ANNOTATIONS
    _PARAMETERS: ClassVar[list[ParameterSchema]] = [
        ParameterSchema(
            name="batch_id",
            type="string",
            description=("The batch_id returned by kaos-llm-core-batch-create."),
            required=True,
        ),
        ParameterSchema(
            name="resume",
            type="boolean",
            description=(
                "When true, skip items already in the JSONL log via the "
                "deterministic custom_id skip-set. Default: true if the "
                "batch's current status is 'failed' or 'cancelled', "
                "otherwise false."
            ),
            required=False,
        ),
    ]
    _ERROR_HINT: ClassVar[str] = (
        "Check (1) the batch_id exists, (2) the envelope_path on the "
        "row resolves in the VFS, (3) provider API keys are configured, "
        "and (4) the runtime VFS uses the disk backend."
    )

    async def _run(self, inputs: dict[str, Any], context: KaosContext | None = None) -> ToolResult:
        with workspace_or_error(context) as ws:
            if isinstance(ws, ToolResult):
                return ws
            return await self._execute_with_ws(ws, inputs, context)

    async def _execute_with_ws(
        self,
        ws: Any,
        inputs: dict[str, Any],
        context: KaosContext | None,
    ) -> ToolResult:
        batch_id = inputs.get("batch_id")
        if not isinstance(batch_id, str) or not batch_id:
            return ToolResult.create_error(
                "batch_id is required (string). Get one from kaos-llm-core-batch-create."
            )
        record = ws.get_batch(batch_id)
        if record is None:
            return ToolResult.create_error(
                f"No batch with id {batch_id!r}. List existing batches via the "
                "workspace SQLite, or call kaos-llm-core-batch-create first."
            )

        resume_arg = inputs.get("resume")
        if resume_arg is None:
            resume = record.status in ("failed", "cancelled")
        else:
            resume = bool(resume_arg)

        if record.status == "running":
            return ToolResult.create_error(
                f"Batch {batch_id} is already running. Use kaos-llm-core-batch-status "
                "to monitor progress, or wait for the prior run to finish."
            )
        if record.status == "completed" and not resume:
            return ToolResult.create_error(
                f"Batch {batch_id} is already completed. To re-run with resume "
                "skipping prior items, pass resume=true. To run from scratch, "
                "create a new batch."
            )

        # Re-load the envelope from its persisted location. Inline
        # envelopes were copied to ${output_dir}/envelope.json at
        # create-time so this re-load works for both inputs.
        loaded = await load_envelope(
            envelope=None,
            envelope_path=record.envelope_path,
            context=context,
        )
        if isinstance(loaded, ToolResult):
            return loaded
        envelope_dict, _ = loaded

        # Build the program. Settings flow into each child Call.
        program = from_envelope(envelope_dict)
        settings = _settings_for(context)
        for child in program.named_calls().values():
            if hasattr(child, "_core_settings"):
                child._core_settings = settings

        # Build the input source descriptor → BatchInputSource.
        built = build_input_source(
            record.inputs_source,
            program_hash_value=record.program_hash,
            context=context,
        )
        if isinstance(built, ToolResult):
            return built
        source, _count_hint = built

        # Mark running. Preserve original started_at on resume.
        started_at = record.started_at or _now_iso()
        ws.update_status(
            batch_id,
            "running",
            started_at=started_at,
            completed_at=None,
        )

        async def _progress_cb(progress: BatchProgress) -> None:
            ws.update_progress(batch_id, progress)
            if context is not None and hasattr(context, "report_progress"):
                with contextlib.suppress(Exception):
                    await context.report_progress(
                        progress=progress.n_done,
                        total=progress.n_total,
                        message=(f"{progress.n_done} done — ${progress.cost_usd_so_far:.4f}"),
                    )

        # Use the absolute disk dir we resolved + cached at create-time.
        # Passing runtime=None makes batch_run treat output_dir as a
        # literal disk path, sidestepping per-session VFS scoping (the
        # batch subsystem is shared across all MCP requests).
        from pathlib import Path as _Path

        output_disk_dir = str(_Path(record.log_path).parent)

        try:
            result = await batch_run(
                program,
                source,
                output_dir=output_disk_dir,
                runtime=None,
                context=context,
                run_id=batch_id,
                max_concurrency=record.max_concurrency,
                error_policy=record.error_policy,
                max_errors=record.max_errors,
                mode="live",
                resume=resume,
                progress_callback=_progress_cb,
                item_timeout_s=record.item_timeout_s,
            )
        except Exception as exc:
            ws.update_status(
                batch_id,
                "failed",
                completed_at=_now_iso(),
                error_summary={
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            )
            return ToolResult.create_error(
                f"Batch {batch_id} failed: {exc}. Inspect the workspace row "
                "via kaos-llm-core-batch-status. Re-run with resume=true to "
                "continue from the last successful item."
            )

        terminal_status: str
        if result.status == "completed":
            terminal_status = "completed"
        elif result.status == "stopped":
            terminal_status = "failed"
        else:
            terminal_status = result.status

        ws.update_status(
            batch_id,
            terminal_status,  # type: ignore[arg-type]
            completed_at=_now_iso(),
            duration_s=result.duration_s,
            n_total=result.n_total,
            n_succeeded=result.n_succeeded,
            n_errored=result.n_errored,
            n_skipped=result.n_skipped,
            cost_usd=result.cost_usd,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            total_tokens=result.total_tokens,
            error_summary=(
                {"errors_by_type": result.errors_by_type} if result.errors_by_type else None
            ),
        )

        output = {
            "batch_id": batch_id,
            "status": terminal_status,
            "n_total": result.n_total,
            "n_succeeded": result.n_succeeded,
            "n_errored": result.n_errored,
            "n_skipped": result.n_skipped,
            "cost_usd": round(result.cost_usd, 6),
            "duration_s": round(result.duration_s, 3),
            "tokens": {
                "input": result.input_tokens,
                "output": result.output_tokens,
                "total": result.total_tokens,
            },
            "log_path": result.log_path,
            "manifest_path": result.manifest_path,
            "errors_by_type": result.errors_by_type,
        }
        summary = (
            f"Batch {batch_id} {terminal_status}: "
            f"{result.n_succeeded}/{result.n_total} succeeded, "
            f"{result.n_errored} errored, "
            f"${result.cost_usd:.6f}, {result.duration_s:.2f}s"
        )
        return ToolResult.create_success(output=output, summary=summary)
