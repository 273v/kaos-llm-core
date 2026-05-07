"""KaosLLMCoreJudgeTool — see kaos_llm_core.tools (Phase 14B split)."""

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


class KaosLLMCoreJudgeTool(BaseLLMCoreTool):
    """Use an LLM to evaluate the quality of another LLM's output against criteria."""

    _NAME: ClassVar[str] = "kaos-llm-core-judge"
    _DISPLAY_NAME: ClassVar[str] = "LLM Core Judge"
    _DESCRIPTION: ClassVar[str] = (
        "Use an LLM as a judge to evaluate the quality of an output against "
        "specified criteria. Returns a quality score (0.0-1.0) and reasoning. "
        "Use after kaos-llm-core-call to assess output quality, or to compare "
        "outputs from different models. For multi-model comparison, use "
        "kaos-llm-core-ensemble instead."
    )
    _CATEGORY: ClassVar[ToolCategory] = ToolCategory.INTEGRATION
    _CAPABILITY: ClassVar[ToolCapability] = ToolCapability.ANALYZE
    _ANNOTATIONS: ClassVar[ToolAnnotations] = ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    )
    _PARAMETERS: ClassVar[list[ParameterSchema]] = [
        ParameterSchema(
            name="model",
            type="string",
            description=(
                "Model to use as the judge (e.g., 'anthropic:claude-sonnet-4-6'). "
                "Use a strong model for reliable evaluation."
            ),
            required=True,
        ),
        ParameterSchema(
            name="input_text",
            type="string",
            description="The original input that produced the output.",
            required=True,
        ),
        ParameterSchema(
            name="output_text",
            type="string",
            description="The output to evaluate.",
            required=True,
        ),
        ParameterSchema(
            name="criteria",
            type="string",
            description=(
                "Evaluation criteria (e.g., 'accuracy, completeness, and "
                "correctness of entity extraction')."
            ),
            required=True,
        ),
        ParameterSchema(
            name="system",
            type="string",
            description="Optional system instruction describing the original task.",
            required=False,
        ),
    ]
    _ERROR_HINT: ClassVar[str] = (
        "Check that the model identifier is valid and API key is set. "
        "Alternative: use kaos-llm-core-reason to manually analyze output quality."
    )

    async def _run(self, inputs: dict[str, Any], context: KaosContext | None = None) -> ToolResult:
        from kaos_llm_core.programs.call import Call
        from kaos_llm_core.programs.judge import JudgmentSignature

        model = inputs["model"]
        input_text = inputs["input_text"]
        output_text = inputs["output_text"]
        criteria = inputs["criteria"]
        system = inputs.get("system", "Produce a high-quality response")

        # Phase 9c follow-up: honor _meta.kaos_config per-request overrides.
        settings = _settings_for(context)
        judge_call = Call(JudgmentSignature, model=model, core_settings=settings)
        judgment = await judge_call(
            original_input=input_text,
            task_description=system,
            response=output_text,
            criteria=criteria,
        )

        score = float(judgment.quality_score)
        reasoning = str(judgment.reasoning)

        output: dict[str, Any] = {
            "score": score,
            "reasoning": reasoning,
            "model": model,
        }

        return ToolResult.create_success(
            output=output,
            summary=f"Judge score: {score:.2f} (model: {model})",
        )
