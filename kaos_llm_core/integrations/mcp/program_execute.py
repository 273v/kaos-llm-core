"""KaosLLMCoreProgramExecuteTool — execute a Program v3 envelope.

Phase 15.1. The first MCP tool that lets an agent describe a
multi-step LLM program in JSON, save it to the VFS, and execute
it through the wire.

Sub-design: ``docs/internal/design/program-envelope-v3.md``.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from kaos_core import KaosContext, ToolResult
from kaos_core.types.annotations import ToolAnnotations
from kaos_core.types.enums import ToolCapability, ToolCategory
from kaos_core.types.parameters import ParameterSchema

from kaos_llm_core.integrations.mcp._common import (
    _LLM_CORE_ANNOTATIONS,
    BaseLLMCoreTool,
    _settings_for,
)


class KaosLLMCoreProgramExecuteTool(BaseLLMCoreTool):
    """Execute a multi-step LLM program described as a Program v3 envelope.

    The Program v3 envelope is the JSON-declarable equivalent of a Python
    ``Program`` subclass: a typed multi-step composition with input
    substitution between steps. An agent constructs the envelope (or
    reads it from the VFS), passes it to this tool, and gets back the
    final outputs plus a trace and per-step cost. See
    ``docs/internal/design/program-envelope-v3.md`` for the full schema.
    """

    _NAME: ClassVar[str] = "kaos-llm-core-program-execute"
    _DISPLAY_NAME: ClassVar[str] = "LLM Core Program Execute"
    _DESCRIPTION: ClassVar[str] = (
        "Execute a multi-step LLM program described as a Program v3 envelope. "
        "An envelope is JSON that declares typed inputs, symbolic clients, "
        "shared types, and an ordered list of steps where each step is one "
        "underlying primitive. Executable step kinds: call, reason, judge, "
        "multi_chain_comparison, program_of_thought. NOTE: the schema also "
        "accepts react, refine, and best_of_n for forward compatibility but "
        "these are NOT YET EXECUTABLE and will fail at runtime — use the "
        "dedicated MCP tools (kaos-llm-core-react, kaos-llm-core-refine, "
        "kaos-llm-core-best-of-n) for those patterns instead. "
        "Steps reference upstream values via JSON-pointer-like syntax "
        "($.inputs.X or $.steps.<id>.output.<field>). The envelope must declare a "
        "top-level 'output' mapping that produces the final result dict. "
        "Pass 'envelope' for an inline envelope (small composes) or 'envelope_path' "
        "for a saved envelope in the VFS (preferred for anything non-trivial — "
        "the agent typically saves via kaos-core-vfs-write first). 'inputs' is "
        "the dict of values to invoke the program with — the keys must match the "
        "envelope's top-level 'inputs' block. Returns the final outputs dict, "
        "the cost in USD, and the total tokens. For the schema, edge cases, and "
        "examples see docs/internal/design/program-envelope-v3.md."
    )
    _CATEGORY: ClassVar[ToolCategory] = ToolCategory.INTEGRATION
    _CAPABILITY: ClassVar[ToolCapability] = ToolCapability.QUERY
    _ANNOTATIONS: ClassVar[ToolAnnotations] = _LLM_CORE_ANNOTATIONS
    _PARAMETERS: ClassVar[list[ParameterSchema]] = [
        ParameterSchema(
            name="envelope",
            type="object",
            description=(
                "Inline Program v3 envelope (a JSON object with kaos_program='1', "
                "name, inputs, clients, steps, output, capabilities). XOR with "
                "envelope_path. Use this for small composes; for saved programs "
                "or envelopes larger than the inline-result threshold, use "
                "envelope_path instead."
            ),
            required=False,
        ),
        ParameterSchema(
            name="envelope_path",
            type="string",
            description=(
                "VFS path to a saved Program v3 envelope JSON file (e.g. "
                "'/vfs/programs/contract-triage-v1.json'). XOR with envelope. "
                "Resolves the file via the runtime VFS, then executes."
            ),
            required=False,
        ),
        ParameterSchema(
            name="inputs",
            type="object",
            description=(
                "The input values to invoke the program with. Keys must match "
                "the envelope's top-level 'inputs' block. Values are passed as "
                "kwargs into program.invoke(**inputs)."
            ),
            required=True,
        ),
    ]
    _ERROR_HINT: ClassVar[str] = (
        "Check that (1) the envelope passes schema validation per "
        "docs/internal/design/program-envelope-v3.md, (2) the inputs dict "
        "covers every required top-level input, (3) every step's client "
        "references a real provider model, and (4) the API key for that "
        "provider is configured. Alternative: build the program in Python "
        "via kaos_llm_core.programs.envelope.from_envelope() for richer error "
        "messages."
    )

    async def _run(self, inputs: dict[str, Any], context: KaosContext | None = None) -> ToolResult:
        from kaos_llm_core.programs.envelope import (
            ProgramEnvelopeError,
            from_envelope,
        )

        envelope_arg = inputs.get("envelope")
        envelope_path = inputs.get("envelope_path")
        program_inputs = inputs.get("inputs", {})

        # Validate XOR semantics
        if envelope_arg is None and envelope_path is None:
            return ToolResult.create_error(
                "Provide either 'envelope' (inline JSON) or 'envelope_path' "
                "(VFS path). Both are missing. "
                "For small composes use envelope; for saved programs use envelope_path."
            )
        if envelope_arg is not None and envelope_path is not None:
            return ToolResult.create_error(
                "Provide either 'envelope' or 'envelope_path', not both. "
                "If you have a saved envelope file, just pass envelope_path."
            )

        # Load the envelope
        if envelope_path is not None:
            envelope_dict = await self._load_envelope_from_vfs(envelope_path, context)
            if isinstance(envelope_dict, ToolResult):
                return envelope_dict  # error
        else:
            if isinstance(envelope_arg, str):
                try:
                    envelope_dict = json.loads(envelope_arg)
                except json.JSONDecodeError as exc:
                    return ToolResult.create_error(
                        f"envelope is a string but not valid JSON: {exc}. "
                        "Pass it as a parsed object, not a JSON string."
                    )
            elif isinstance(envelope_arg, dict):
                envelope_dict = envelope_arg
            else:
                return ToolResult.create_error(
                    f"envelope must be a JSON object, got {type(envelope_arg).__name__}. "
                    "See docs/internal/design/program-envelope-v3.md for the schema."
                )

        # Settings + per-request overrides
        settings = _settings_for(context)

        # Build the program
        try:
            program = from_envelope(envelope_dict)
        except ProgramEnvelopeError as exc:
            return ToolResult.create_error(
                f"LLM Core Program Execute: envelope rejected. {exc} "
                "See docs/internal/design/program-envelope-v3.md for the schema."
            )

        # Thread runtime settings into every step's underlying Call
        for child in program.named_calls().values():
            if hasattr(child, "_core_settings"):
                child._core_settings = settings

        # Execute
        invocation = await program.invoke(**program_inputs)

        # Build the result envelope
        output = {
            "outputs": invocation.output,
            "tokens": {
                "input": invocation.usage.input_tokens,
                "output": invocation.usage.output_tokens,
                "total": invocation.usage.total_tokens,
            },
            "cost_usd": round(invocation.usage.cost_usd, 6),
        }

        program_name = envelope_dict.get("name", "program")
        n_steps = len(envelope_dict.get("steps", []))
        summary = (
            f"{program_name}: {n_steps} step{'s' if n_steps != 1 else ''}, "
            f"${invocation.usage.cost_usd:.6f}, "
            f"{invocation.usage.total_tokens} tokens"
        )

        return ToolResult.create_success(
            output=output,
            summary=summary,
        )

    async def _load_envelope_from_vfs(
        self, path: str, context: KaosContext | None
    ) -> dict[str, Any] | ToolResult:
        """Read an envelope JSON file from the VFS."""
        if context is None or context.runtime is None:
            return ToolResult.create_error(
                f"Cannot resolve envelope_path={path!r}: no runtime context. "
                "Inline envelopes work without a runtime; envelope_path requires "
                "a KaosRuntime to access the VFS. Pass envelope inline or run "
                "this tool through an MCP server."
            )
        try:
            data = await context.runtime.vfs.read(path, context_id=context.session_id)
            return json.loads(data.decode("utf-8"))
        except json.JSONDecodeError as exc:
            return ToolResult.create_error(
                f"envelope_path={path!r}: file is not valid JSON ({exc}). "
                "Confirm the file was written by kaos-core-vfs-write with the "
                "Program v3 envelope shape from docs/internal/design/program-envelope-v3.md."
            )
        except FileNotFoundError:
            return ToolResult.create_error(
                f"envelope_path={path!r}: file not found in the VFS. "
                "Confirm the path with kaos-core-vfs-list or write the envelope "
                "first with kaos-core-vfs-write."
            )
        except Exception as exc:
            return ToolResult.create_error(
                f"envelope_path={path!r}: read failed: {exc}. "
                "Check the path is correct and the runtime VFS is accessible."
            )
