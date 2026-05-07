"""KaosLLMCoreOptimizeModelTool — see kaos_llm_core.tools (Phase 14B split)."""

from __future__ import annotations

import json
from typing import Any, ClassVar, cast

from kaos_core import KaosContext, ToolResult
from kaos_core.types.annotations import ToolAnnotations
from kaos_core.types.enums import ToolCapability, ToolCategory
from kaos_core.types.parameters import ParameterSchema

from kaos_llm_core.integrations.mcp._common import (
    _PHASE6_ANNOTATIONS,
    BaseLLMCoreTool,
    _build_eval_dataset,
    _resolve_metric_by_name,
)


class KaosLLMCoreOptimizeModelTool(BaseLLMCoreTool):
    """Pick the cheapest model that hits a metric threshold."""

    _NAME: ClassVar[str] = "kaos-llm-core-optimize-model"
    _DISPLAY_NAME: ClassVar[str] = "LLM Core Optimize Model"
    _DESCRIPTION: ClassVar[str] = (
        "Score a Call under multiple candidate models and return the cheapest "
        "model that meets the quality threshold. If no candidate meets the "
        "threshold, returns the highest-scoring model with stop_reason="
        "'threshold_not_met'. Use to find the cheapest-good-enough model for a "
        "task. Metric transport: specify metric_name; for custom metrics use "
        "ModelOptimizer in the Python API."
    )
    _CATEGORY: ClassVar[ToolCategory] = ToolCategory.INTEGRATION
    _CAPABILITY: ClassVar[ToolCapability] = ToolCapability.TRANSFORM
    _ANNOTATIONS: ClassVar[ToolAnnotations] = _PHASE6_ANNOTATIONS
    _PARAMETERS: ClassVar[list[ParameterSchema]] = [
        ParameterSchema(
            name="models",
            type="array",
            description=(
                "List of provider:model identifiers to try. "
                "Example: ['openai:gpt-5.4-nano', 'anthropic:claude-haiku-4-5']."
            ),
            required=True,
        ),
        ParameterSchema(
            name="instruction",
            type="string",
            description="System instruction for the Call.",
            required=True,
        ),
        ParameterSchema(
            name="examples",
            type="array",
            description="Validation examples: {input, expected_output} objects.",
            required=True,
        ),
        ParameterSchema(
            name="min_score",
            type="number",
            description="Quality threshold; default 0.8.",
            required=False,
            constraints={"minimum": 0.0, "maximum": 1.0},
        ),
        ParameterSchema(
            name="metric_name",
            type="string",
            description=(
                "Metric: 'exact_match' (default), 'case_insensitive_match', 'length_ratio'."
            ),
            required=False,
            constraints={
                "enum": [
                    "normalized_match",
                    "exact_match",
                    "case_insensitive_match",
                    "length_ratio",
                ]
            },
        ),
    ]
    _ERROR_HINT: ClassVar[str] = (
        "Check model identifiers and API keys. Alternative: use ModelOptimizer in the Python API."
    )

    async def _run(self, inputs: dict[str, Any], context: KaosContext | None = None) -> ToolResult:
        from kaos_llm_core.optimization.model_optimizer import ModelOptimizer
        from kaos_llm_core.programs.call import Call
        from kaos_llm_core.settings import KaosLLMCoreSettings

        settings = (
            KaosLLMCoreSettings.from_context(context)
            if context is not None
            else KaosLLMCoreSettings()
        )

        models_raw = inputs["models"]
        if isinstance(models_raw, str):
            models_raw = json.loads(models_raw)
        if not isinstance(models_raw, list) or not models_raw:
            return ToolResult.create_error(
                "models must be a non-empty list of provider:model identifiers. "
                "Example: ['openai:gpt-5.4-nano', 'anthropic:claude-haiku-4-5']."
            )

        metric_name = inputs.get("metric_name", "normalized_match")
        metric = _resolve_metric_by_name(metric_name)
        if metric is None:
            return ToolResult.create_error(
                f"Unknown metric_name={metric_name!r}. "
                "Use 'exact_match', 'case_insensitive_match', or 'length_ratio'."
            )

        parsed = _build_eval_dataset(inputs["examples"])
        if isinstance(parsed, ToolResult):
            return parsed
        sig, dataset = parsed

        min_score = float(inputs.get("min_score", 0.8))

        call = Call(
            sig,
            model=models_raw[0],
            instructions=inputs["instruction"],
            core_settings=settings,
        )
        optimizer = ModelOptimizer(
            metric=metric, models=cast(list[str], models_raw), min_score=min_score
        )
        result = await optimizer.optimize(call, dataset)

        output: dict[str, Any] = {
            "best_model": result.best_model,
            "best_score": result.best_score,
            "scores_by_model": result.scores_by_model,
            "cost_by_model": result.cost_by_model,
            "stop_reason": result.stop_reason,
        }
        return ToolResult.create_success(
            output=output,
            summary=(
                f"Best model: {result.best_model} "
                f"(score={result.best_score:.3f}, stop={result.stop_reason})"
            ),
        )
