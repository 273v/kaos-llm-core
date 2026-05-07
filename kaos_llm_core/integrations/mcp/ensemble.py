"""KaosLLMCoreEnsembleTool — see kaos_llm_core.tools (Phase 14B split)."""

from __future__ import annotations

import json
from typing import Any, ClassVar

from kaos_core import KaosContext, ToolResult
from kaos_core.types.annotations import ToolAnnotations
from kaos_core.types.enums import ToolCapability, ToolCategory
from kaos_core.types.parameters import ParameterSchema

from kaos_llm_core.integrations.mcp._common import (
    BaseLLMCoreTool,
    _settings_for,
)


class KaosLLMCoreEnsembleTool(BaseLLMCoreTool):
    """Run the same task across multiple models and return consensus results."""

    _NAME: ClassVar[str] = "kaos-llm-core-ensemble"
    _DISPLAY_NAME: ClassVar[str] = "LLM Core Ensemble"
    _DESCRIPTION: ClassVar[str] = (
        "Run the same instruction across multiple models in parallel and return "
        "all results with a consensus. Use for high-stakes tasks where you want "
        "agreement across models. Requires multiple model API keys. "
        "For single-model evaluation, use kaos-llm-core-judge instead."
    )
    _CATEGORY: ClassVar[ToolCategory] = ToolCategory.INTEGRATION
    _CAPABILITY: ClassVar[ToolCapability] = ToolCapability.QUERY
    _ANNOTATIONS: ClassVar[ToolAnnotations] = ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    )
    _PARAMETERS: ClassVar[list[ParameterSchema]] = [
        ParameterSchema(
            name="models",
            type="array",
            description=(
                "List of model identifiers to run in parallel. "
                "Example: ['anthropic:claude-haiku-4-5', 'openai:gpt-5.4-nano', "
                "'google:gemini-2.5-flash']."
            ),
            required=True,
        ),
        ParameterSchema(
            name="instruction",
            type="string",
            description="What the LLMs should do (system instruction).",
            required=True,
        ),
        ParameterSchema(
            name="input_text",
            type="string",
            description="The input text to process.",
            required=True,
        ),
        ParameterSchema(
            name="output_fields",
            type="object",
            description=(
                "JSON object mapping field names to descriptions. "
                'Example: {"answer": "the answer", "confidence": "confidence level"}. '
                "If omitted, a single 'answer' field is used."
            ),
            required=False,
        ),
    ]
    _ERROR_HINT: ClassVar[str] = (
        "Check that all model identifiers are valid and API keys are set. "
        "Alternative: use kaos-llm-core-call to query models individually."
    )

    async def _run(self, inputs: dict[str, Any], context: KaosContext | None = None) -> ToolResult:
        from pydantic import create_model

        from kaos_llm_core import InputField, OutputField, Signature
        from kaos_llm_core.programs.ensemble import Ensemble

        models = inputs["models"]
        instruction = inputs["instruction"]
        input_text = inputs["input_text"]
        output_fields_spec = inputs.get("output_fields")

        if isinstance(models, str):
            models = json.loads(models)

        if not models or not isinstance(models, list):
            return ToolResult.create_error(
                "models must be a non-empty list of model identifiers. "
                "Example: ['anthropic:claude-haiku-4-5', 'openai:gpt-5.4-nano']. "
                "For a single model, use kaos-llm-core-call instead."
            )

        if output_fields_spec is None:
            output_fields_spec = {"answer": "the answer"}
        elif isinstance(output_fields_spec, str):
            output_fields_spec = json.loads(output_fields_spec)

        field_defs: dict[str, Any] = {
            "text": (str, InputField(description="Input text")),
        }
        output_names = list(output_fields_spec.keys())
        for name, desc in output_fields_spec.items():
            field_defs[name] = (str, OutputField(description=str(desc)))

        sig = create_model(
            "DynamicEnsembleSignature",
            __base__=Signature,
            __doc__=instruction,
            **field_defs,
        )

        # Use first output field as vote field for majority vote
        vote_field = output_names[0] if output_names else None

        # Phase 9d: honor _meta.kaos_config per-request overrides.
        settings = _settings_for(context)
        ensemble = Ensemble(
            sig,
            models=models,
            aggregation="majority_vote" if vote_field else "all",
            vote_field=vote_field,
            instructions=instruction,
            core_settings=settings,
        )
        result = await ensemble(text=input_text)

        # Build per-model results
        per_model: list[dict[str, Any]] = []
        for i, r in enumerate(result.all_results):
            model_name = models[i] if i < len(models) else f"model-{i}"
            entry: dict[str, Any] = {"model": model_name}
            output_dict: dict[str, Any] = {}
            for name in output_names:
                output_dict[name] = getattr(r, name, None)
            entry["output"] = output_dict
            per_model.append(entry)

        # Build consensus from the selected result
        consensus_parts: list[str] = []
        for name in output_names:
            val = getattr(result.selected, name, None)
            consensus_parts.append(f"{name}: {val}")
        consensus = "; ".join(consensus_parts) if consensus_parts else ""

        output: dict[str, Any] = {
            "results": per_model,
            "consensus": consensus,
        }

        return ToolResult.create_success(
            output=output,
            summary=f"Ensemble: {len(per_model)} models, consensus: {consensus[:100]}",
        )
