"""KaosLLMCoreProgramOfThoughtTool — code-as-reasoning over the MCP wire.

Dedicated wrapper for the ``ProgramOfThought`` program (Phase 16.2).
Counterpart to ``kaos-llm-core-react`` / ``kaos-llm-core-refine`` /
``kaos-llm-core-best-of-n``: a single MCP tool that exposes one
program kind without requiring the agent to construct a full
Program v3 envelope.

Security: ``ProgramOfThought`` executes model-generated Python in a
subprocess sandbox. The underlying Program requires an explicit
``allow_code_execution=True`` opt-in. The MCP wrapper enforces the
same contract — the tool refuses to run unless the caller passes
``allow_code_execution=true`` AND the runtime
:class:`KaosLLMCoreSettings.allow_code_execution_via_mcp` flag is
also true. Both gates default to off. See
``kaos_llm_core/programs/program_of_thought.py`` for the v1 sandbox
threat model.
"""

from __future__ import annotations

from typing import Any, ClassVar

from kaos_core import KaosContext, ToolResult
from kaos_core.types.annotations import ToolAnnotations
from kaos_core.types.enums import ToolCapability, ToolCategory
from kaos_core.types.parameters import ParameterSchema

from kaos_llm_core.integrations.mcp._common import (
    BaseLLMCoreTool,
    _settings_for,
)


class KaosLLMCoreProgramOfThoughtTool(BaseLLMCoreTool):
    """Code-as-reasoning over MCP.

    Writer LLM emits Python, a subprocess sandbox runs it, and an
    interpreter LLM parses the captured stdout into a typed answer.
    """

    _NAME: ClassVar[str] = "kaos-llm-core-program-of-thought"
    _DISPLAY_NAME: ClassVar[str] = "LLM Core Program-of-Thought"
    _DESCRIPTION: ClassVar[str] = (
        "Solve a task by having one LLM write a small Python program, "
        "executing it in a subprocess sandbox, and having a second LLM "
        "parse the captured stdout into a structured answer. Best for "
        "arithmetic-heavy, list-manipulation, or symbolic reasoning that "
        "LLMs are unreliable at executing in their head. "
        "DANGEROUS BY DEFAULT: executes model-generated code. Both this "
        "tool and the runtime setting `allow_code_execution_via_mcp` must "
        "be opted in explicitly (`allow_code_execution=true` AND "
        "`KAOS_LLM_CORE_ALLOW_CODE_EXECUTION_VIA_MCP=1`). The sandbox is "
        "an isolated Python interpreter with POSIX rlimits (memory, CPU, "
        "fsize), a fresh tempdir cwd, stdin closed, no env vars except "
        "PATH, and a wall-clock timeout. It blocks obvious mistakes but "
        "does not stop a determined adversary — wrap in a container "
        "sandbox (gVisor, firecracker) for untrusted task input. "
        "Returns the interpreted answer, the generated code, and the raw "
        "stdout/stderr so callers can audit what ran."
    )
    _CATEGORY: ClassVar[ToolCategory] = ToolCategory.INTEGRATION
    _CAPABILITY: ClassVar[ToolCapability] = ToolCapability.ANALYZE
    _ANNOTATIONS: ClassVar[ToolAnnotations] = ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    )
    _PARAMETERS: ClassVar[list[ParameterSchema]] = [
        ParameterSchema(
            name="producer_model",
            type="string",
            description="Model identifier for the code-writer LLM.",
            required=True,
        ),
        ParameterSchema(
            name="instruction",
            type="string",
            description="Task description shown to both the writer and interpreter LLMs.",
            required=True,
        ),
        ParameterSchema(
            name="input_text",
            type="string",
            description="The task input the program should reason over.",
            required=True,
        ),
        ParameterSchema(
            name="output_field",
            type="string",
            description=(
                "Name of the single structured output field the interpreter "
                "produces (e.g. 'answer', 'total', 'count')."
            ),
            required=True,
        ),
        ParameterSchema(
            name="allow_code_execution",
            type="boolean",
            description=(
                "REQUIRED OPT-IN. Must be true to actually run the generated "
                "code. Defaults to false at every layer — passing false (or "
                "omitting it) makes the tool refuse before any subprocess "
                "spawns. Confirms the caller has accepted the v1 sandbox "
                "threat model."
            ),
            required=True,
        ),
        ParameterSchema(
            name="interpreter_model",
            type="string",
            description=(
                "Optional model identifier for the interpreter LLM. Defaults "
                "to producer_model when absent. Use a cheaper model here for "
                "cost savings when interpretation is straightforward."
            ),
            required=False,
        ),
        ParameterSchema(
            name="timeout_s",
            type="number",
            description=(
                "Wall-clock seconds before the sandbox subprocess is killed. "
                "Default 8.0. Hard cap 30.0 to keep MCP requests bounded."
            ),
            required=False,
            constraints={"minimum": 0.1, "maximum": 30.0},
        ),
        ParameterSchema(
            name="memory_mb",
            type="integer",
            description=(
                "RLIMIT_AS cap for the sandbox subprocess in megabytes. "
                "Default 256 MB. Hard cap 1024 MB."
            ),
            required=False,
            constraints={"minimum": 16, "maximum": 1024},
        ),
        ParameterSchema(
            name="cpu_seconds",
            type="integer",
            description="RLIMIT_CPU cap in seconds. Default 5. Hard cap 30.",
            required=False,
            constraints={"minimum": 1, "maximum": 30},
        ),
    ]
    _ERROR_HINT: ClassVar[str] = (
        "Check that (1) allow_code_execution=true was passed, "
        "(2) KAOS_LLM_CORE_ALLOW_CODE_EXECUTION_VIA_MCP=1 is set on the "
        "server, (3) producer_model is valid and reachable, and (4) the "
        "generated code finished within timeout_s. "
        "Alternative: use kaos-llm-core-call for direct LLM answers when "
        "code execution is unnecessary."
    )

    async def _run(self, inputs: dict[str, Any], context: KaosContext | None = None) -> ToolResult:
        from pydantic import create_model

        from kaos_llm_core import InputField, OutputField, Signature
        from kaos_llm_core.programs.program_of_thought import ProgramOfThought

        try:
            producer_model = inputs["producer_model"]
            instruction = inputs["instruction"]
            input_text = inputs["input_text"]
            output_field = inputs["output_field"]
            allow_code_execution = inputs["allow_code_execution"]
        except KeyError as ke:
            return ToolResult.create_error(
                f"LLM Core Program-of-Thought missing required parameter: {ke}. "
                "Required: producer_model, instruction, input_text, output_field, "
                "allow_code_execution. Alternative: use kaos-llm-core-call for "
                "direct LLM answers without code execution."
            )

        if not isinstance(allow_code_execution, bool):
            return ToolResult.create_error(
                f"LLM Core Program-of-Thought: allow_code_execution={allow_code_execution!r} "
                "must be a boolean. Pass true to opt in to model-generated code "
                "execution, or false (or omit the param) to refuse the run."
            )
        if not allow_code_execution:
            return ToolResult.create_error(
                "LLM Core Program-of-Thought refused: allow_code_execution=false. "
                "This tool runs model-generated Python in a subprocess sandbox; "
                "you must pass allow_code_execution=true to opt in. Alternative: "
                "use kaos-llm-core-call or kaos-llm-core-reason when LLM-only "
                "reasoning is sufficient."
            )

        # Phase 9d: honor _meta.kaos_config per-request overrides.
        settings = _settings_for(context)
        if not getattr(settings, "allow_code_execution_via_mcp", False):
            return ToolResult.create_error(
                "LLM Core Program-of-Thought refused at the runtime gate: the "
                "KAOS_LLM_CORE_ALLOW_CODE_EXECUTION_VIA_MCP setting is off (the "
                "default). Set it to 1 in the server environment to allow this "
                "tool to execute code. Alternative: use the Python API where "
                "the security model is auditable in-process."
            )

        interpreter_model = inputs.get("interpreter_model") or producer_model
        timeout_s = float(inputs.get("timeout_s") or 8.0)
        memory_mb = int(inputs.get("memory_mb") or 256)
        cpu_seconds = int(inputs.get("cpu_seconds") or 5)

        field_defs: dict[str, Any] = {
            "text": (str, InputField(description="Input text")),
            output_field: (
                str,
                OutputField(description=f"The {output_field} computed from the program output."),
            ),
        }
        sig = create_model(
            "DynamicProgramOfThoughtSignature",
            __base__=Signature,
            __doc__=instruction,
            **field_defs,
        )

        program = ProgramOfThought(
            sig,
            producer_model=producer_model,
            interpreter_model=interpreter_model,
            allow_code_execution=True,
            timeout_s=timeout_s,
            memory_bytes=memory_mb * 1024 * 1024,
            cpu_seconds=cpu_seconds,
        )

        result = await program(text=input_text)

        execution = result.execution
        answer_value = getattr(result.outputs, output_field, None)

        output: dict[str, Any] = {
            "output": answer_value,
            "code": execution.code,
            "stdout": execution.stdout,
            "stderr": execution.stderr,
            "return_code": execution.return_code,
            "timed_out": execution.timed_out,
        }

        summary = (
            f"Program-of-Thought: producer={producer_model}, interpreter={interpreter_model}, "
            f"timed_out={execution.timed_out}, return_code={execution.return_code}"
        )

        return ToolResult.create_success(output=output, summary=summary)
