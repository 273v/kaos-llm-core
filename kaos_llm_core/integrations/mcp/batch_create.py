"""KaosLLMCoreBatchCreateTool — define a batch job (no execution).

Phase 15.3. Inserts a row into the workspace SQLite ``batches`` table
with status ``pending``. Returns a ``batch_id`` plus a coarse cost
estimate. The agent then calls ``kaos-llm-core-batch-run`` to execute.

Sub-design: ``docs/internal/design/batch-workspace-schema.md``.
"""

from __future__ import annotations

from typing import Any, ClassVar

from kaos_core import KaosContext, ToolResult
from kaos_core.types.annotations import ToolAnnotations
from kaos_core.types.enums import ToolCapability, ToolCategory
from kaos_core.types.parameters import ParameterSchema

from kaos_llm_core.integrations.mcp._batch_helpers import (
    generate_batch_id,
    load_envelope,
    record_to_dict,
    resolve_output_dir,
    validate_envelope,
    workspace_or_error,
)
from kaos_llm_core.integrations.mcp._common import BaseLLMCoreTool
from kaos_llm_core.workspace import BatchAlreadyExistsError, BatchRecord
from kaos_llm_core.workspace.metadata import _now_iso

# batch-create is a write op (single INSERT) but idempotent in spirit:
# the same envelope+source+output_dir is safe to recreate as long as
# the batch_id differs. We model it as not destructive, not idempotent
# (each call creates a new batch row).
_BATCH_CREATE_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=False,
    openWorldHint=False,
)


class KaosLLMCoreBatchCreateTool(BaseLLMCoreTool):
    """Define a new batch job in the workspace.

    Inserts a ``batches`` row in pending state. Returns the
    ``batch_id`` and a count hint for the input source. Does NOT
    execute the batch — call ``kaos-llm-core-batch-run`` for that.
    """

    _NAME: ClassVar[str] = "kaos-llm-core-batch-create"
    _DISPLAY_NAME: ClassVar[str] = "LLM Core Batch Create"
    _DESCRIPTION: ClassVar[str] = (
        "Define a new batch job to run a Program v3 envelope across many "
        "inputs. Inserts a row in the workspace SQLite metadata DB with "
        "status='pending' and returns a batch_id. Does NOT start running. "
        "Call kaos-llm-core-batch-run with the returned batch_id to execute, "
        "then poll with kaos-llm-core-batch-status and retrieve results "
        "with kaos-llm-core-batch-results. The 'envelope' XOR 'envelope_path' "
        "argument supplies the program; 'inputs_source' is a JSON descriptor "
        "for the input stream (type='list' with inline items, OR type='jsonl' "
        "pointing at a JSONL file in the VFS); 'output_dir' is the VFS dir "
        "that will receive items.jsonl + manifest.json. Optional flags: "
        "name, max_concurrency (default 8), error_policy "
        "(continue|stop|stop_after_n), max_errors (required iff "
        "error_policy='stop_after_n'), item_timeout_s. The cost-safety valve "
        "is error_policy='stop_after_n' with a small max_errors. See "
        "docs/internal/design/batch-workspace-schema.md for the full schema."
    )
    _CATEGORY: ClassVar[ToolCategory] = ToolCategory.INTEGRATION
    _CAPABILITY: ClassVar[ToolCapability] = ToolCapability.QUERY
    _ANNOTATIONS: ClassVar[ToolAnnotations] = _BATCH_CREATE_ANNOTATIONS
    _PARAMETERS: ClassVar[list[ParameterSchema]] = [
        ParameterSchema(
            name="envelope",
            type="object",
            description=(
                "Inline Program v3 envelope (a JSON object). XOR with "
                "envelope_path. Use this for small composes."
            ),
            required=False,
        ),
        ParameterSchema(
            name="envelope_path",
            type="string",
            description=(
                "VFS path to a saved Program v3 envelope JSON. XOR with "
                "envelope. Preferred for non-trivial programs."
            ),
            required=False,
        ),
        ParameterSchema(
            name="inputs_source",
            type="object",
            description=(
                "JSON descriptor for the input stream. Supported: "
                "{'type':'list','items':[{...},...]} for inline lists, "
                "{'type':'jsonl','path':'/vfs/inputs.jsonl'} for streaming "
                "from a file in the VFS."
            ),
            required=True,
        ),
        ParameterSchema(
            name="output_dir",
            type="string",
            description=(
                "VFS directory that will receive items.jsonl + manifest.json. Created if missing."
            ),
            required=True,
        ),
        ParameterSchema(
            name="name",
            type="string",
            description="Optional human-friendly batch name (used as id slug).",
            required=False,
        ),
        ParameterSchema(
            name="max_concurrency",
            type="integer",
            description="Async semaphore cap on in-flight items. Default: 8.",
            required=False,
        ),
        ParameterSchema(
            name="error_policy",
            type="string",
            description=(
                "How to react to per-item failures: 'continue' (default, "
                "log and keep going), 'stop' (abort on first failure), "
                "'stop_after_n' (abort after max_errors failures — the "
                "money-safety valve)."
            ),
            required=False,
        ),
        ParameterSchema(
            name="max_errors",
            type="integer",
            description=(
                "Required when error_policy='stop_after_n'. The batch aborts "
                "after this many per-item failures."
            ),
            required=False,
        ),
        ParameterSchema(
            name="item_timeout_s",
            type="number",
            description="Optional per-item asyncio timeout in seconds.",
            required=False,
        ),
    ]
    _ERROR_HINT: ClassVar[str] = (
        "Check that (1) the envelope passes validation, (2) the "
        "inputs_source descriptor is well-formed, (3) the output_dir is "
        "writable, and (4) the runtime VFS uses the disk backend."
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
        envelope_arg = inputs.get("envelope")
        envelope_path_arg = inputs.get("envelope_path")
        loaded = await load_envelope(
            envelope=envelope_arg,
            envelope_path=envelope_path_arg,
            context=context,
        )
        if isinstance(loaded, ToolResult):
            return loaded
        envelope_dict, envelope_source = loaded

        program_hash_value = validate_envelope(envelope_dict)
        if isinstance(program_hash_value, ToolResult):
            return program_hash_value

        inputs_source = inputs.get("inputs_source")
        if not isinstance(inputs_source, dict):
            return ToolResult.create_error(
                "inputs_source must be a JSON object with a 'type' key. "
                "See docs/internal/design/batch-workspace-schema.md."
            )
        # Pre-count for the cost hint. Build the source via the helper but
        # don't iterate yet — we just need the count.
        from kaos_llm_core.integrations.mcp._batch_helpers import build_input_source

        built = build_input_source(
            inputs_source,
            program_hash_value=program_hash_value,
            context=context,
        )
        if isinstance(built, ToolResult):
            return built
        _source, count_hint = built

        output_dir_arg = inputs.get("output_dir")
        if not isinstance(output_dir_arg, str):
            return ToolResult.create_error("output_dir must be a string (VFS directory path).")
        disk_dir = resolve_output_dir(output_dir_arg, context)
        if isinstance(disk_dir, ToolResult):
            return disk_dir

        # Persist inline envelopes alongside the batch's output_dir so
        # batch-run can re-load them on a separate MCP request. This is
        # the v1 design — without it, inline envelopes are unreachable
        # from later tool calls because the workspace SQLite only
        # records *paths*, not envelope bodies. The persisted file is
        # the canonical source of truth from this point forward.
        if envelope_source == "<inline>":
            import json as _json

            envelope_disk = disk_dir / "envelope.json"
            envelope_disk.write_text(_json.dumps(envelope_dict))
            envelope_source = f"{output_dir_arg.rstrip('/')}/envelope.json"

        # Validate error_policy. The MCP wire may deliver an explicit
        # ``None`` for optional fields the caller omitted, so coerce to
        # the default before validating.
        error_policy = inputs.get("error_policy") or "continue"
        if error_policy not in ("continue", "stop", "stop_after_n"):
            return ToolResult.create_error(
                f"error_policy must be 'continue', 'stop', or 'stop_after_n'. Got {error_policy!r}."
            )
        max_errors = inputs.get("max_errors")
        if error_policy == "stop_after_n" and max_errors is None:
            return ToolResult.create_error(
                "error_policy='stop_after_n' requires max_errors. Pick a small "
                "value (e.g. 5) as a money-safety valve."
            )

        batch_id = generate_batch_id(name=inputs.get("name"))
        log_path = disk_dir / "items.jsonl"
        manifest_path = disk_dir / "manifest.json"

        record = BatchRecord(
            batch_id=batch_id,
            name=inputs.get("name"),
            created_at=_now_iso(),
            created_by=getattr(context, "session_id", None) if context else None,
            envelope_path=envelope_source,
            program_hash=program_hash_value,
            program_name=envelope_dict.get("name"),
            inputs_source=inputs_source,
            inputs_count_hint=count_hint,
            output_dir=output_dir_arg,
            log_path=str(log_path),
            manifest_path=str(manifest_path),
            max_concurrency=int(inputs.get("max_concurrency") or 8),
            error_policy=error_policy,
            max_errors=int(max_errors) if max_errors is not None else None,
            mode="live",
            item_timeout_s=(
                float(inputs["item_timeout_s"])
                if inputs.get("item_timeout_s") is not None
                else None
            ),
        )

        try:
            ws.create_batch(record)
        except BatchAlreadyExistsError as exc:
            return ToolResult.create_error(
                f"Batch '{batch_id}' already exists: {exc}. "
                "Use a different batch_id or check existing batches with "
                "kaos-llm-core-batch-status. To re-run a failed batch, "
                "use kaos-llm-core-batch-run which supports resume."
            )

        output = {
            "batch_id": batch_id,
            "status": "pending",
            "envelope_path": envelope_source,
            "program_hash": program_hash_value,
            "program_name": envelope_dict.get("name"),
            "inputs_count_hint": count_hint,
            "output_dir": output_dir_arg,
            "log_path": str(log_path),
            "manifest_path": str(manifest_path),
            "max_concurrency": record.max_concurrency,
            "error_policy": record.error_policy,
            "max_errors": record.max_errors,
        }
        count_str = str(count_hint) if count_hint is not None else "?"
        summary = (
            f"Batch {batch_id} created (pending, {count_str} inputs). "
            f"Call kaos-llm-core-batch-run to execute."
        )
        return ToolResult.create_success(output=output, summary=summary)


# Silence ty: record_to_dict re-exported for symmetry with sibling tools
_ = record_to_dict
