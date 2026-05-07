"""KaosLLMCoreCallTool — see kaos_llm_core.tools (Phase 14B split)."""

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


class KaosLLMCoreCallTool(BaseLLMCoreTool):
    """Make a typed LLM call with a natural-language instruction and structured output."""

    _NAME: ClassVar[str] = "kaos-llm-core-call"
    _DISPLAY_NAME: ClassVar[str] = "LLM Core Call"
    _DESCRIPTION: ClassVar[str] = (
        "Make a typed LLM call: provide an instruction, input text, and expected "
        "output fields. Returns structured JSON output validated against the schema. "
        "Use this when you need structured extraction, classification, or analysis "
        "from an LLM with typed output validation."
    )
    _CATEGORY: ClassVar[ToolCategory] = ToolCategory.INTEGRATION
    _CAPABILITY: ClassVar[ToolCapability] = ToolCapability.QUERY
    _ANNOTATIONS: ClassVar[ToolAnnotations] = _LLM_CORE_ANNOTATIONS
    _PARAMETERS: ClassVar[list[ParameterSchema]] = [
        ParameterSchema(
            name="instruction",
            type="string",
            description="What the LLM should do (system instruction).",
            required=True,
        ),
        ParameterSchema(
            name="text",
            type="string",
            description="The input text to process.",
            required=True,
        ),
        ParameterSchema(
            name="output_fields",
            type="object",
            description=(
                "JSON object mapping field names to descriptions. "
                'Example: {"entities": "list of entity names", "summary": "brief summary"}'
            ),
            required=True,
        ),
        ParameterSchema(
            name="model",
            type="string",
            description=(
                "Model identifier in 'provider:model' format. "
                "Examples: 'anthropic:claude-haiku-4-5', 'openai:gpt-5.4-nano', "
                "'google:gemini-2.5-flash'."
            ),
            required=False,
        ),
        ParameterSchema(
            name="temperature",
            type="number",
            description="Sampling temperature (0.0-2.0). Lower = more deterministic.",
            required=False,
            constraints={"minimum": 0.0, "maximum": 2.0},
        ),
    ]
    _ERROR_HINT: ClassVar[str] = (
        "Check that the model identifier is correct and API key is set. "
        "Alternative: use kaos-llm-chat for simpler untyped chat."
    )

    async def _run(self, inputs: dict[str, Any], context: KaosContext | None = None) -> ToolResult:
        from kaos_llm_core import Call, InputField, OutputField, Signature

        instruction = inputs["instruction"]
        text = inputs["text"]
        output_fields_spec = inputs["output_fields"]
        model = inputs.get("model")
        temperature = inputs.get("temperature")

        if isinstance(output_fields_spec, str):
            output_fields_spec = json.loads(output_fields_spec)

        # Build a dynamic Signature
        from pydantic import create_model

        field_defs: dict[str, Any] = {
            "text": (str, InputField(description="Input text")),
        }
        for name, desc in output_fields_spec.items():
            field_defs[name] = (str, OutputField(description=str(desc)))

        sig = create_model(
            "DynamicSignature",
            __base__=Signature,
            __doc__=instruction,
            **field_defs,
        )

        # Phase 9c follow-up: honor _meta.kaos_config per-request overrides.
        settings = _settings_for(context)
        # Resolve model
        if not model:
            model = settings.default_model
        if not model:
            return ToolResult.create_error(
                "No model specified. Set model= parameter or KAOS_LLM_CORE_DEFAULT_MODEL env var. "
                "Example models: 'anthropic:claude-haiku-4-5', 'openai:gpt-5.4-nano'."
            )

        kwargs: dict[str, Any] = {}
        if temperature is not None:
            kwargs["temperature"] = temperature

        call = Call(
            sig,
            model=model,
            instructions=instruction,
            core_settings=settings,
            **kwargs,
        )
        invocation = await call.invoke(text=text)
        result = invocation.output

        output: dict[str, Any] = {}
        for name in output_fields_spec:
            output[name] = getattr(result, name, None)

        trace = invocation.trace
        summary_parts = [f"Model: {model}"]
        if trace:
            summary_parts.append(f"Tokens: {trace.total_tokens}")
            summary_parts.append(f"Latency: {trace.latency_ms:.0f}ms")

        return ToolResult.create_success(
            output=output,
            summary=", ".join(summary_parts),
        )
