"""KaosLLMCoreRecipeTuneTool — see kaos_llm_core.tools (Phase 14B split)."""

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


class KaosLLMCoreRecipeTuneTool(BaseLLMCoreTool):
    """Run a pre-built optimization recipe."""

    _NAME: ClassVar[str] = "kaos-llm-core-recipe-tune"
    _DISPLAY_NAME: ClassVar[str] = "LLM Core Recipe Tune"
    _DESCRIPTION: ClassVar[str] = (
        "Run a pre-built optimization recipe end-to-end. Three recipes: "
        "'friendly_prompt' (instruction → bootstrap), 'example_first' "
        "(bootstrap → instruction), 'cost_aware_model' (pick cheapest model "
        "above threshold). Each wraps CoOptimizer / ModelOptimizer with sensible "
        "defaults so callers get a one-step workflow. For fine-grained control, "
        "use the individual optimizer tools."
    )
    _CATEGORY: ClassVar[ToolCategory] = ToolCategory.INTEGRATION
    _CAPABILITY: ClassVar[ToolCapability] = ToolCapability.TRANSFORM
    _ANNOTATIONS: ClassVar[ToolAnnotations] = _PHASE6_ANNOTATIONS
    _PARAMETERS: ClassVar[list[ParameterSchema]] = [
        ParameterSchema(
            name="recipe",
            type="string",
            description=(
                "Recipe name. 'friendly_prompt' and 'example_first' require "
                "'model' + train_set via 'examples'. 'cost_aware_model' requires "
                "'models' list."
            ),
            required=True,
            constraints={"enum": ["friendly_prompt", "example_first", "cost_aware_model"]},
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
            description="Training/validation examples: {input, expected_output}.",
            required=True,
        ),
        ParameterSchema(
            name="model",
            type="string",
            description="Single model (for friendly_prompt / example_first).",
            required=False,
        ),
        ParameterSchema(
            name="models",
            type="array",
            description="Model list (for cost_aware_model).",
            required=False,
        ),
        ParameterSchema(
            name="min_score",
            type="number",
            description="Quality threshold for cost_aware_model (default 0.85).",
            required=False,
            constraints={"minimum": 0.0, "maximum": 1.0},
        ),
        ParameterSchema(
            name="metric_name",
            type="string",
            description="Metric name.",
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
        "Check the recipe name, model(s), and examples. "
        "Alternative: use the underlying optimizers directly via the Python API."
    )

    async def _run(self, inputs: dict[str, Any], context: KaosContext | None = None) -> ToolResult:
        from kaos_llm_core.optimization.recipes import (
            CostAwareModelSelector,
            ExampleFirstTuner,
            FriendlyPromptTuner,
        )
        from kaos_llm_core.programs.call import Call
        from kaos_llm_core.settings import KaosLLMCoreSettings

        settings = (
            KaosLLMCoreSettings.from_context(context)
            if context is not None
            else KaosLLMCoreSettings()
        )

        recipe = inputs["recipe"]
        metric_name = inputs.get("metric_name", "normalized_match")
        metric = _resolve_metric_by_name(metric_name)
        if metric is None:
            return ToolResult.create_error(f"Unknown metric_name={metric_name!r}.")

        parsed = _build_eval_dataset(inputs["examples"])
        if isinstance(parsed, ToolResult):
            return parsed
        sig, dataset = parsed

        if recipe in ("friendly_prompt", "example_first"):
            model = inputs.get("model")
            if not model:
                return ToolResult.create_error(
                    f"Recipe {recipe!r} requires 'model' parameter. "
                    "Alternative: use cost_aware_model with a list of models."
                )
            call = Call(
                sig,
                model=model,
                instructions=inputs["instruction"],
                core_settings=settings,
            )
            factory = FriendlyPromptTuner if recipe == "friendly_prompt" else ExampleFirstTuner
            optimizer = factory(metric=metric)
            result = await optimizer.optimize(call, train_set=dataset, val_set=dataset)
            output: dict[str, Any] = {
                "recipe": recipe,
                "metric_before": result.metric_before,
                "metric_after": result.metric_after,
                "stages_run": result.stages_run,
                "stop_reason": result.stop_reason,
            }
            return ToolResult.create_success(
                output=output,
                summary=(
                    f"Recipe {recipe}: {result.metric_before:.1%} → {result.metric_after:.1%}"
                ),
            )

        # cost_aware_model
        models_raw = inputs.get("models")
        if isinstance(models_raw, str):
            models_raw = json.loads(models_raw)
        if not isinstance(models_raw, list) or not models_raw:
            return ToolResult.create_error(
                "Recipe 'cost_aware_model' requires a non-empty 'models' list."
            )
        min_score = float(inputs.get("min_score", 0.85))
        call = Call(
            sig,
            model=models_raw[0],
            instructions=inputs["instruction"],
            core_settings=settings,
        )
        optimizer = CostAwareModelSelector(
            metric=metric, models=cast(list[str], models_raw), min_score=min_score
        )
        result = await optimizer.optimize(call, dataset)
        output = {
            "recipe": recipe,
            "best_model": result.best_model,
            "best_score": result.best_score,
            "scores_by_model": result.scores_by_model,
            "cost_by_model": result.cost_by_model,
            "stop_reason": result.stop_reason,
        }
        return ToolResult.create_success(
            output=output,
            summary=(
                f"Recipe cost_aware_model: picked {result.best_model} "
                f"(score={result.best_score:.3f})"
            ),
        )
