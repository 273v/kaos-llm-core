"""KaosLLMCoreReActTool — see kaos_llm_core.tools (Phase 14B split)."""

from __future__ import annotations

import json
from typing import Any, ClassVar, cast

from kaos_core import KaosContext, ToolResult
from kaos_core.types.annotations import ToolAnnotations
from kaos_core.types.enums import ToolCapability, ToolCategory
from kaos_core.types.parameters import ParameterSchema

from kaos_llm_core.integrations.mcp._common import (
    BaseLLMCoreTool,
    _settings_for,
)


class KaosLLMCoreReActTool(BaseLLMCoreTool):
    """Run a ReAct (reason-act-observe) loop with tool specs and dry-run executors.

    Tool executors are dry-run stubs by design — MCP cannot transport Python
    callables. Each tool spec is exposed to the model with its real schema, but
    when the model emits a tool call the executor returns a canned dry-run
    envelope so the loop can demonstrate end-to-end behaviour without side
    effects. Use the kaos-llm-core Python API for real tool execution.
    """

    _NAME: ClassVar[str] = "kaos-llm-core-react"
    _DISPLAY_NAME: ClassVar[str] = "LLM Core ReAct"
    _DESCRIPTION: ClassVar[str] = (
        "DRY-RUN ONLY: Run a ReAct (reason -> act -> observe -> reason) loop "
        "with SIMULATED tool execution. Tool executors are dry-run stubs — "
        "MCP cannot transport Python callables, so every tool call returns "
        '{"dry_run": true, "tool": <name>, "arguments": <args>} instead of '
        "real results. The model reasons about hypothetical tool outputs. "
        "Use this for prototyping loop shapes and inspecting trajectories, "
        "NOT for executing real tool workflows. For real tool execution, use "
        "the kaos-llm-core Python API directly. "
        "Provide an instruction, an input, the name of a single output "
        "field, and a list of tool specs (name, description, JSON-Schema "
        "parameters). Returns the final output plus a full per-iteration "
        "trajectory including tool calls and (simulated) tool results. "
        "Alternative: use kaos-llm-core-call for a one-shot typed call "
        "without tool simulation."
    )
    _CATEGORY: ClassVar[ToolCategory] = ToolCategory.INTEGRATION
    _CAPABILITY: ClassVar[ToolCapability] = ToolCapability.ANALYZE
    _ANNOTATIONS: ClassVar[ToolAnnotations] = ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    )
    _PARAMETERS: ClassVar[list[ParameterSchema]] = [
        ParameterSchema(
            name="model",
            type="string",
            description=(
                "Model identifier in 'provider:model' format. "
                "Example: 'anthropic:claude-haiku-4-5'."
            ),
            required=True,
        ),
        ParameterSchema(
            name="instruction",
            type="string",
            description="System instruction describing the loop's objective.",
            required=True,
        ),
        ParameterSchema(
            name="input_text",
            type="string",
            description="The user input to feed into the loop.",
            required=True,
        ),
        ParameterSchema(
            name="output_field",
            type="string",
            description=(
                "Name of the single output field on the dynamic Signature (e.g. 'answer')."
            ),
            required=True,
        ),
        ParameterSchema(
            name="tool_specs",
            type="array",
            description=(
                "List of tool specifications. Each item is "
                '{"name": <string>, "description": <string>, '
                '"parameters": <JSON-Schema object>}. Executors are '
                "dry-run stubs that return a canned envelope."
            ),
            required=True,
        ),
        ParameterSchema(
            name="system",
            type="string",
            description="Optional additional system context (currently appended to instruction).",
            required=False,
        ),
        ParameterSchema(
            name="max_iterations",
            type="integer",
            description="Maximum loop iterations before MAX_ITERATIONS stop.",
            required=False,
            default=10,
            constraints={"minimum": 1, "maximum": 50},
        ),
    ]
    _ERROR_HINT: ClassVar[str] = (
        "Check the model identifier, API key, and tool_specs shape. "
        "Alternative: use kaos-llm-core-call for a one-shot typed call."
    )

    async def _run(self, inputs: dict[str, Any], context: KaosContext | None = None) -> ToolResult:
        from kaos_llm_client.types import ToolDefinition
        from pydantic import create_model

        from kaos_llm_core import InputField, OutputField, Signature
        from kaos_llm_core.programs.react import ReAct
        from kaos_llm_core.programs.tool import Tool

        try:
            model = inputs["model"]
            instruction = inputs["instruction"]
            input_text = inputs["input_text"]
            output_field = inputs["output_field"]
            tool_specs = inputs["tool_specs"]
        except KeyError as ke:
            return ToolResult.create_error(
                f"LLM Core ReAct missing required parameter: {ke}. "
                "Required: model, instruction, input_text, output_field, tool_specs. "
                "Alternative: use kaos-llm-core-call for a single typed call without a loop."
            )

        system = inputs.get("system")
        max_iterations = int(inputs.get("max_iterations", 10) or 10)

        if isinstance(tool_specs, str):
            try:
                tool_specs = json.loads(tool_specs)
            except json.JSONDecodeError as jde:
                return ToolResult.create_error(
                    f"LLM Core ReAct: tool_specs is a string but is not valid JSON: {jde}. "
                    "Provide tool_specs as a list of objects, or a JSON-encoded list. "
                    "Alternative: pass an empty list [] to run the loop without tools."
                )

        if not isinstance(tool_specs, list):
            return ToolResult.create_error(
                "LLM Core ReAct: tool_specs must be a list of "
                "{name, description, parameters} objects. "
                "Alternative: pass an empty list [] to run a tool-less loop."
            )

        # Build dynamic single-field Signature.
        full_instruction = instruction if not system else f"{system}\n\n{instruction}"
        field_defs: dict[str, Any] = {
            "text": (str, InputField(description="Input text")),
            output_field: (str, OutputField(description=f"The {output_field} output")),
        }
        sig = create_model(
            "DynamicReActSignature",
            __base__=Signature,
            __doc__=full_instruction,
            **field_defs,
        )

        # Build Tool objects with dry-run executors. Each executor returns a
        # canned envelope so the model gets *something* back and can finish
        # the loop end-to-end without real side effects.
        def _make_dry_run_executor(tool_name: str) -> Any:
            def _dry_run(**kwargs: Any) -> dict[str, Any]:
                return {"dry_run": True, "tool": tool_name, "arguments": dict(kwargs)}

            return _dry_run

        tools_list: list[Tool] = []
        for idx, raw_spec in enumerate(tool_specs):
            if not isinstance(raw_spec, dict):
                return ToolResult.create_error(
                    f"LLM Core ReAct: tool_specs[{idx}] is not an object. "
                    'Each spec must be {"name": str, "description": str, "parameters": object}. '
                    "Fix the malformed entry and retry."
                )
            spec = cast("dict[str, Any]", raw_spec)
            name = spec.get("name")
            description = spec.get("description", "")
            parameters = spec.get("parameters") or {
                "type": "object",
                "properties": {},
                "required": [],
            }
            if not name or not isinstance(name, str):
                return ToolResult.create_error(
                    f"LLM Core ReAct: tool_specs[{idx}] is missing a 'name' string. "
                    "Each tool spec needs a non-empty name."
                )
            definition = ToolDefinition(
                name=name,
                description=str(description) if description else None,
                parameters=parameters,
            )
            tools_list.append(Tool(definition=definition, executor=_make_dry_run_executor(name)))

        # Phase 9c follow-up: honor _meta.kaos_config per-request overrides.
        settings = _settings_for(context)
        react = ReAct(
            sig,
            tools=tools_list,
            model=model,
            max_iterations=max_iterations,
            instructions=full_instruction,
            core_settings=settings,
        )
        result = await react(text=input_text)

        output_value: Any = None
        if result.outputs is not None:
            output_value = result.outputs.get(output_field)

        trajectory: list[dict[str, Any]] = []
        for it in result.trajectory:
            trajectory.append(
                {
                    "iteration": it.iteration,
                    "text": it.text,
                    "tool_calls": [
                        {"name": tc.name, "arguments": dict(tc.arguments)} for tc in it.tool_calls
                    ],
                    "tool_results": [
                        {
                            "tool_name": obs.tool_name,
                            "result": obs.result,
                            "is_error": obs.is_error,
                        }
                        for obs in it.tool_results
                    ],
                    "error": it.error,
                }
            )

        output: dict[str, Any] = {
            "output": output_value,
            "iterations_used": result.iterations_used,
            "stop_reason": result.stop_reason,
            "trajectory": trajectory,
        }

        return ToolResult.create_success(
            output=output,
            summary=(
                f"ReAct: {result.iterations_used} iter, "
                f"stop={result.stop_reason}, tools={len(tools_list)}"
            ),
        )
