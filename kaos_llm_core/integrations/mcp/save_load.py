"""KaosLLMCoreSaveLoadTool — see kaos_llm_core.tools (Phase 14B split)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

from kaos_core import KaosContext, ToolResult
from kaos_core.types.annotations import ToolAnnotations
from kaos_core.types.enums import ToolCapability, ToolCategory
from kaos_core.types.parameters import ParameterSchema

from kaos_llm_core.integrations.mcp._common import BaseLLMCoreTool


class KaosLLMCoreSaveLoadTool(BaseLLMCoreTool):
    """Validate and round-trip a saved Program envelope (v2 schema)."""

    _NAME: ClassVar[str] = "kaos-llm-core-save-load"
    _DISPLAY_NAME: ClassVar[str] = "LLM Core Save/Load"
    _DESCRIPTION: ClassVar[str] = (
        "Inspect, validate, and round-trip a Program save envelope produced "
        "by Program.save(). Three modes: "
        "(1) 'save' is rejected with an explanatory error — saving requires "
        "an in-memory Program from the Python API; "
        "(2) 'load' parses the file at path (or an inline program_state "
        "object), validates the v2 envelope shape, and returns a summary "
        "({valid, program, version, num_calls, state_keys}); v1 envelopes "
        "are loadable but flagged with deprecated=true; "
        "(3) 'round-trip' reads, re-serializes, and writes to a temp file, "
        "returning the temp file path so callers can verify their envelope "
        "is healthy. The temp file is the caller's responsibility to delete."
    )
    _CATEGORY: ClassVar[ToolCategory] = ToolCategory.UTILITY
    _CAPABILITY: ClassVar[ToolCapability] = ToolCapability.TRANSFORM
    _ANNOTATIONS: ClassVar[ToolAnnotations] = ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
    _PARAMETERS: ClassVar[list[ParameterSchema]] = [
        ParameterSchema(
            name="mode",
            type="string",
            description="One of 'save', 'load', or 'round-trip'.",
            required=True,
            constraints={"enum": ["save", "load", "round-trip"]},
        ),
        ParameterSchema(
            name="path",
            type="string",
            description=(
                "Local file path. Required for 'round-trip' and "
                "'load' (unless program_state is provided)."
            ),
            required=False,
        ),
        ParameterSchema(
            name="program_state",
            type="object",
            description=(
                "Inline JSON envelope to validate. Used as an alternative "
                "to 'path' when mode='load'."
            ),
            required=False,
        ),
    ]
    _ERROR_HINT: ClassVar[str] = (
        "Check the file path and envelope shape. "
        "Alternative: use Program.save()/load() in the Python API for full control."
    )

    async def _run(self, inputs: dict[str, Any], context: KaosContext | None = None) -> ToolResult:
        import tempfile
        from pathlib import Path

        from kaos_llm_core.programs.base import Program

        mode = inputs.get("mode")
        if mode not in ("save", "load", "round-trip"):
            return ToolResult.create_error(
                f"LLM Core Save/Load: mode={mode!r} is invalid. "
                "Use one of: 'save', 'load', 'round-trip'. "
                "Most callers want 'load' to validate a file or "
                "'round-trip' to confirm an envelope is healthy."
            )

        if mode == "save":
            return ToolResult.create_error(
                "LLM Core Save/Load: 'save' mode requires an in-memory Program "
                "instance and is not supportable over MCP. Use Program.save() "
                "in the Python API. This MCP tool can validate ('load') and "
                "round-trip envelopes via 'round-trip'."
            )

        path_str = inputs.get("path")

        # Reject absolute disk paths so an MCP caller cannot read arbitrary
        # files (e.g. /etc/hostname or ~/.aws/credentials). All inputs must
        # be VFS-relative; the path is resolved against the runtime's VFS
        # root by ``_resolve_input_path`` below.
        if path_str and Path(path_str).is_absolute():
            return ToolResult.create_error(
                f"LLM Core Save/Load: absolute disk paths are rejected "
                f"({path_str!r}). Pass a VFS-relative path; the file will be "
                "resolved against the runtime's VFS root. Alternative: use "
                "Program.load() in the Python API for direct disk access."
            )

        if mode == "load":
            program_state = inputs.get("program_state")
            if program_state is None:
                if not path_str:
                    return ToolResult.create_error(
                        "LLM Core Save/Load: 'load' mode requires either 'path' "
                        "or an inline 'program_state' object. Provide one and retry."
                    )
                file_path_or_err = self._resolve_input_path(path_str, context)
                if isinstance(file_path_or_err, ToolResult):
                    return file_path_or_err
                file_path = file_path_or_err
                if not file_path.exists():
                    return ToolResult.create_error(
                        f"LLM Core Save/Load: file not found at the resolved VFS "
                        f"path {path_str!r}. Check the path and ensure the file "
                        "was produced by Program.save()."
                    )
                try:
                    data = json.loads(file_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as jde:
                    return ToolResult.create_error(
                        f"LLM Core Save/Load: file at {path_str!r} is not valid "
                        f"JSON: {jde}. The file may be corrupted. Re-save with "
                        "Program.save()."
                    )
            else:
                if isinstance(program_state, str):
                    try:
                        data = json.loads(program_state)
                    except json.JSONDecodeError as jde:
                        return ToolResult.create_error(
                            f"LLM Core Save/Load: program_state is not valid JSON: {jde}. "
                            "Provide a JSON object or a JSON-encoded string."
                        )
                else:
                    data = program_state

            if not isinstance(data, dict):
                return ToolResult.create_error(
                    "LLM Core Save/Load: envelope must be a JSON object. "
                    "Re-save with Program.save() to produce the correct shape."
                )

            if "state" not in data:
                return ToolResult.create_error(
                    "LLM Core Save/Load: envelope is missing the required 'state' key. "
                    "Valid envelopes have {program, version, state}. "
                    "Re-save with Program.save() to produce a healthy file."
                )

            version = int(data.get("version", 1))
            program_name = str(data.get("program", "unknown"))
            state = data.get("state") or {}
            state_keys = sorted(state.keys()) if isinstance(state, dict) else []
            num_calls = len(state_keys)
            deprecated = version < Program.SCHEMA_VERSION

            output: dict[str, Any] = {
                "valid": True,
                "program": program_name,
                "version": version,
                "num_calls": num_calls,
                "state_keys": state_keys,
            }
            if deprecated:
                output["deprecated"] = True

            summary = f"Loaded {program_name} v{version} with {num_calls} sub-call(s)" + (
                " [DEPRECATED v1]" if deprecated else ""
            )
            return ToolResult.create_success(output=output, summary=summary)

        # mode == "round-trip"
        if not path_str:
            return ToolResult.create_error(
                "LLM Core Save/Load: 'round-trip' mode requires 'path' to read from. "
                "Provide a path to a file produced by Program.save()."
            )
        file_path_or_err = self._resolve_input_path(path_str, context)
        if isinstance(file_path_or_err, ToolResult):
            return file_path_or_err
        file_path = file_path_or_err
        if not file_path.exists():
            return ToolResult.create_error(
                f"LLM Core Save/Load: file not found at the resolved VFS path "
                f"{path_str!r}. Check the path and ensure the file was produced "
                "by Program.save()."
            )
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as jde:
            return ToolResult.create_error(
                f"LLM Core Save/Load: file at {path_str!r} is not valid JSON: "
                f"{jde}. The file may be corrupted. Re-save with Program.save()."
            )
        if not isinstance(data, dict) or "state" not in data:
            return ToolResult.create_error(
                "LLM Core Save/Load: envelope is missing the required 'state' key. "
                "Cannot round-trip a malformed envelope. Re-save with Program.save()."
            )

        serialized = json.dumps(data, indent=2, sort_keys=True, default=str)
        # Round-trip output goes to a tempdir but the caller only sees a
        # synthetic VFS-style relative name in the response. The tempdir
        # path itself is logged at debug only — never exposed to the
        # caller — so it cannot be probed.
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            prefix="kaos-llm-core-save-load-",
            delete=False,
        ) as tmp:
            tmp.write(serialized)
            tmp_name = tmp.name

        output_rt: dict[str, Any] = {
            "success": True,
            "original_path": path_str,
            "round_trip_path": tmp_name,
            "size_bytes": len(serialized.encode("utf-8")),
        }
        return ToolResult.create_success(
            output=output_rt,
            summary=(
                f"Round-trip OK: {Path(path_str).name} -> {tmp_name} "
                f"({output_rt['size_bytes']} bytes; caller must clean up the temp file)"
            ),
        )

    @staticmethod
    def _resolve_input_path(
        path_str: str,
        context: KaosContext | None,
    ) -> Path | ToolResult:
        """Resolve a VFS-relative path to a disk path under the runtime VFS root.

        Absolute paths must already have been rejected by the caller; this
        method enforces VFS containment for the relative case.
        """
        from pathlib import Path

        if context is None or context.runtime is None:
            # Outside of a runtime context (e.g. legacy callers), fall back
            # to the bare path as-is — but it must be relative; rejecting
            # absolute paths is enforced at the call site.
            return Path(path_str)
        try:
            disk = context.runtime.vfs.resolve_disk_path(
                path_str, context_id="kaos-llm-core-save-load"
            )
        except (AttributeError, NotImplementedError) as exc:
            return ToolResult.create_error(
                f"LLM Core Save/Load: VFS backend cannot resolve {path_str!r}: "
                f"{exc}. Use Program.load() in the Python API for direct access."
            )
        if disk is None:
            return ToolResult.create_error(
                f"LLM Core Save/Load: path {path_str!r} is not on the disk-backed VFS."
            )
        return Path(disk)
