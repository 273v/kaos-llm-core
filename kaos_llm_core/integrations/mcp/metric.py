"""KaosLLMCoreMetricTool — see kaos_llm_core.tools (Phase 14B split)."""

from __future__ import annotations

from typing import Any, ClassVar

from kaos_core import KaosContext, ToolResult
from kaos_core.types.annotations import ToolAnnotations
from kaos_core.types.enums import ToolCapability, ToolCategory
from kaos_core.types.parameters import ParameterSchema

from kaos_llm_core.integrations.mcp._common import (
    _PHASE7_METRIC_ANNOTATIONS,
    _PHASE7_METRIC_NAMES,
    BaseLLMCoreTool,
)


class KaosLLMCoreMetricTool(BaseLLMCoreTool):
    """Compute a named metric on a (prediction, gold) pair."""

    _NAME: ClassVar[str] = "kaos-llm-core-metric"
    _DISPLAY_NAME: ClassVar[str] = "LLM Core Metric"
    _DESCRIPTION: ClassVar[str] = (
        "Compute a named metric on a (prediction, gold) string pair. "
        "Supports deterministic text/structured metrics (exact_match, "
        "case_insensitive_match, normalized_match, accuracy, numeric_ratio) "
        "as well as the llm_judge metric, which uses an LLM to score the "
        "prediction against a rubric. Most invocations are local; only "
        "metric_name='llm_judge' makes a provider call (hence openWorld). "
        "For full programmatic control, use kaos_llm_core.metrics directly."
    )
    _CATEGORY: ClassVar[ToolCategory] = ToolCategory.UTILITY
    _CAPABILITY: ClassVar[ToolCapability] = ToolCapability.ANALYZE
    _ANNOTATIONS: ClassVar[ToolAnnotations] = _PHASE7_METRIC_ANNOTATIONS
    _PARAMETERS: ClassVar[list[ParameterSchema]] = [
        ParameterSchema(
            name="metric_name",
            type="string",
            description=(
                "Metric to compute. Choose llm_judge for LLM-as-a-metric "
                "scoring; the rest are local string/numeric metrics."
            ),
            required=True,
            constraints={"enum": list(_PHASE7_METRIC_NAMES)},
        ),
        ParameterSchema(
            name="prediction",
            type="string",
            description="The model output to evaluate.",
            required=True,
        ),
        ParameterSchema(
            name="gold",
            type="string",
            description="The reference / expected output.",
            required=True,
        ),
        ParameterSchema(
            name="judge_model",
            type="string",
            description=(
                "Model id for llm_judge (provider:name). Required only "
                "when metric_name='llm_judge'."
            ),
            required=False,
        ),
        ParameterSchema(
            name="rubric",
            type="string",
            description=(
                "Rubric for llm_judge: 'helpfulness', 'factuality', "
                "'conciseness', or a free-form rubric string. Default: 'helpfulness'."
            ),
            required=False,
        ),
    ]
    _ERROR_HINT: ClassVar[str] = (
        "Check metric_name, inputs, and (for llm_judge) the judge_model and API key. "
        "Alternative: use kaos_llm_core.metrics from the Python API."
    )

    async def _run(
        self,
        inputs: dict[str, Any],
        context: KaosContext | None = None,
    ) -> ToolResult:
        from kaos_llm_core.metrics import (
            LLMJudge,
            accuracy,
            case_insensitive_match,
            exact_match,
            normalized_match,
            numeric_ratio,
        )

        metric_name = str(inputs.get("metric_name", "")).strip().lower()
        prediction = inputs.get("prediction")
        gold = inputs.get("gold")
        if prediction is None or gold is None:
            return ToolResult.create_error(
                "kaos-llm-core-metric requires both 'prediction' and 'gold'. "
                "Pass them as strings. Alternative: use kaos-llm-core-evaluate "
                "to score against a labeled dataset."
            )

        deterministic = {
            "exact_match": exact_match,
            "case_insensitive_match": case_insensitive_match,
            "normalized_match": normalized_match,
            "accuracy": accuracy,
            "numeric_ratio": numeric_ratio,
        }

        if metric_name in deterministic:
            score = float(deterministic[metric_name](prediction, gold))
            return ToolResult.create_success(
                output={"metric": metric_name, "score": score},
                summary=f"{metric_name}={score:.4f}",
            )

        if metric_name == "llm_judge":
            judge_model = inputs.get("judge_model")
            if not judge_model:
                return ToolResult.create_error(
                    "metric_name='llm_judge' requires 'judge_model' "
                    "(e.g., 'anthropic:claude-haiku-4-5'). "
                    "Alternative: use a deterministic metric like exact_match."
                )
            rubric = inputs.get("rubric") or "helpfulness"
            judge = LLMJudge(model=str(judge_model), rubric=str(rubric))
            score = await judge.acall(prediction, gold)
            return ToolResult.create_success(
                output={
                    "metric": "llm_judge",
                    "rubric": rubric,
                    "model": judge_model,
                    "score": score,
                },
                summary=f"llm_judge[{rubric}]={score:.4f}",
            )

        return ToolResult.create_error(
            f"Unknown metric_name={metric_name!r}. "
            f"Allowed: {sorted(_PHASE7_METRIC_NAMES)}. "
            "Alternative: use kaos_llm_core.metrics directly for custom metrics."
        )
