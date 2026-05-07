"""KaosLLMCoreChainOfThoughtTool — see kaos_llm_core.tools (Phase 14B split)."""

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


class KaosLLMCoreChainOfThoughtTool(BaseLLMCoreTool):
    """Make an LLM call with step-by-step reasoning before the answer."""

    _NAME: ClassVar[str] = "kaos-llm-core-reason"
    _DISPLAY_NAME: ClassVar[str] = "LLM Core Reason"
    _DESCRIPTION: ClassVar[str] = (
        "Make an LLM call with chain-of-thought reasoning. The model thinks "
        "step-by-step before producing structured output. Returns reasoning + "
        "structured fields. Use for complex analysis, classification, or tasks "
        "that benefit from explicit reasoning."
    )
    _CATEGORY: ClassVar[ToolCategory] = ToolCategory.INTEGRATION
    _CAPABILITY: ClassVar[ToolCapability] = ToolCapability.ANALYZE
    _ANNOTATIONS: ClassVar[ToolAnnotations] = _LLM_CORE_ANNOTATIONS
    _PARAMETERS: ClassVar[list[ParameterSchema]] = [
        ParameterSchema(
            name="instruction",
            type="string",
            description="What the LLM should analyze (system instruction).",
            required=True,
        ),
        ParameterSchema(
            name="text",
            type="string",
            description="The input text to analyze.",
            required=True,
        ),
        ParameterSchema(
            name="output_fields",
            type="object",
            description=(
                "JSON object mapping field names to descriptions. "
                "A 'reasoning' field is added automatically."
            ),
            required=True,
        ),
        ParameterSchema(
            name="model",
            type="string",
            description="Model identifier (e.g., 'anthropic:claude-sonnet-4-6').",
            required=False,
        ),
    ]
    _ERROR_HINT: ClassVar[str] = (
        "Check model and API key. "
        "Alternative: use kaos-llm-core-call for simpler calls without reasoning."
    )

    async def _run(self, inputs: dict[str, Any], context: KaosContext | None = None) -> ToolResult:
        from kaos_llm_core import InputField, OutputField, Signature
        from kaos_llm_core.programs.chain_of_thought import ChainOfThought

        instruction = inputs["instruction"]
        text = inputs["text"]
        output_fields_spec = inputs["output_fields"]
        model = inputs.get("model")

        if isinstance(output_fields_spec, str):
            output_fields_spec = json.loads(output_fields_spec)

        from pydantic import create_model

        field_defs: dict[str, Any] = {
            "text": (str, InputField(description="Input text")),
        }
        for name, desc in output_fields_spec.items():
            field_defs[name] = (str, OutputField(description=str(desc)))

        sig = create_model(
            "DynamicReasonSignature",
            __base__=Signature,
            __doc__=instruction,
            **field_defs,
        )

        # Phase 9c follow-up: honor _meta.kaos_config per-request overrides.
        settings = _settings_for(context)
        if not model:
            model = settings.default_model
        if not model:
            return ToolResult.create_error(
                "No model specified. Set model= parameter or KAOS_LLM_CORE_DEFAULT_MODEL. "
                "Recommended: 'anthropic:claude-sonnet-4-6' for reasoning tasks."
            )

        cot = ChainOfThought(sig, model=model, instructions=instruction, core_settings=settings)
        result = await cot(text=text)

        output: dict[str, Any] = {"reasoning": getattr(result, "reasoning", "")}
        for name in output_fields_spec:
            output[name] = getattr(result, name, None)

        return ToolResult.create_success(
            output=output,
            summary=f"Model: {model}, reasoning length: {len(output['reasoning'])} chars",
        )
