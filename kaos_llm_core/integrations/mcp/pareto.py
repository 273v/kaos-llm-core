"""KaosLLMCoreParetoTool — see kaos_llm_core.tools (Phase 14B split)."""

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


class KaosLLMCoreParetoTool(BaseLLMCoreTool):
    """Compute the (metric, cost) Pareto frontier across candidate models."""

    _NAME: ClassVar[str] = "kaos-llm-core-pareto"
    _DISPLAY_NAME: ClassVar[str] = "LLM Core Pareto"
    _DESCRIPTION: ClassVar[str] = (
        "Score a Call under multiple candidate models and return the Pareto "
        "frontier of (metric, cost) trade-offs. Use to explore the quality/cost "
        "space before committing to a single model with kaos-llm-core-optimize-model. "
        "The frontier contains all non-dominated (model, score, cost) trials."
    )
    _CATEGORY: ClassVar[ToolCategory] = ToolCategory.INTEGRATION
    _CAPABILITY: ClassVar[ToolCapability] = ToolCapability.ANALYZE
    _ANNOTATIONS: ClassVar[ToolAnnotations] = _PHASE6_ANNOTATIONS
    _PARAMETERS: ClassVar[list[ParameterSchema]] = [
        ParameterSchema(
            name="models",
            type="array",
            description="List of provider:model identifiers to explore.",
            required=True,
        ),
        ParameterSchema(
            name="instruction",
            type="string",
            description="System instruction.",
            required=True,
        ),
        ParameterSchema(
            name="examples",
            type="array",
            description="Validation examples: {input, expected_output} objects.",
            required=True,
        ),
        ParameterSchema(
            name="metric_name",
            type="string",
            description="Metric name; see kaos-llm-core-optimize-model.",
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
        "Check model identifiers and examples format. "
        "Alternative: use ParetoOptimizer in the Python API."
    )

    async def _run(self, inputs: dict[str, Any], context: KaosContext | None = None) -> ToolResult:
        from kaos_llm_core.optimization.model_optimizer import ModelOptimizer
        from kaos_llm_core.optimization.pareto import ParetoOptimizer
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
                "models must be a non-empty list of provider:model identifiers."
            )

        metric_name = inputs.get("metric_name", "normalized_match")
        metric = _resolve_metric_by_name(metric_name)
        if metric is None:
            return ToolResult.create_error(f"Unknown metric_name={metric_name!r}.")

        parsed = _build_eval_dataset(inputs["examples"])
        if isinstance(parsed, ToolResult):
            return parsed
        sig, dataset = parsed

        call = Call(
            sig,
            model=models_raw[0],
            instructions=inputs["instruction"],
            core_settings=settings,
        )
        inner = ModelOptimizer(metric=metric, models=cast(list[str], models_raw), min_score=0.0)
        pareto = ParetoOptimizer(metric=metric, inner=inner)
        result = await pareto.optimize(call, dataset)

        output: dict[str, Any] = {
            "frontier": [{"config": cfg, "metric": m, "cost": c} for cfg, m, c in result.frontier],
            "all_trials": [
                {"config": cfg, "metric": m, "cost": c} for cfg, m, c in result.all_trials
            ],
            "stop_reason": result.stop_reason,
        }
        return ToolResult.create_success(
            output=output,
            summary=(
                f"Pareto frontier: {len(result.frontier)} non-dominated / "
                f"{len(result.all_trials)} total trials"
            ),
        )
