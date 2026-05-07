"""KaosLLMCoreRefineTool — see kaos_llm_core.tools (Phase 14B split)."""

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


class KaosLLMCoreRefineTool(BaseLLMCoreTool):
    """Run a producer → judge → refine loop until a quality threshold is met."""

    _NAME: ClassVar[str] = "kaos-llm-core-refine"
    _DISPLAY_NAME: ClassVar[str] = "LLM Core Refine"
    _DESCRIPTION: ClassVar[str] = (
        "Run an inline producer-critic-refine loop. The producer Call "
        "generates an output, a Judge scores it against criteria, and the "
        "loop iterates with feedback injected into the producer's "
        "instructions until either min_score is reached or max_iterations "
        "is exhausted. On overrun the BEST output across iterations is "
        "returned (never the most recent). "
        "Workflow: pair with kaos-llm-core-judge for one-off scoring; "
        "use kaos-llm-core-call for unrefined single-shot generation."
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
            name="producer_model",
            type="string",
            description="Model identifier for the producer Call.",
            required=True,
        ),
        ParameterSchema(
            name="judge_model",
            type="string",
            description="Model identifier for the Judge (typically a stronger model).",
            required=True,
        ),
        ParameterSchema(
            name="producer_instruction",
            type="string",
            description="System instruction for the producer.",
            required=True,
        ),
        ParameterSchema(
            name="judge_instruction",
            type="string",
            description="Evaluation criteria the judge uses to score outputs.",
            required=True,
        ),
        ParameterSchema(
            name="input_text",
            type="string",
            description="The producer's input text.",
            required=True,
        ),
        ParameterSchema(
            name="output_field",
            type="string",
            description="Name of the producer's single output field.",
            required=True,
        ),
        ParameterSchema(
            name="max_iterations",
            type="integer",
            description="Maximum refine iterations (default 3).",
            required=False,
            default=3,
            constraints={"minimum": 1, "maximum": 20},
        ),
        ParameterSchema(
            name="min_score",
            type="number",
            description="Quality threshold for early stop (default 0.8).",
            required=False,
            default=0.8,
            constraints={"minimum": 0.0, "maximum": 1.0},
        ),
    ]
    _ERROR_HINT: ClassVar[str] = (
        "Check both model identifiers and API keys. "
        "Alternative: use kaos-llm-core-call followed by kaos-llm-core-judge."
    )

    async def _run(self, inputs: dict[str, Any], context: KaosContext | None = None) -> ToolResult:
        from pydantic import create_model

        from kaos_llm_core import InputField, OutputField, Signature
        from kaos_llm_core.programs.call import Call
        from kaos_llm_core.programs.judge import Judge
        from kaos_llm_core.programs.refine import Refine

        try:
            producer_model = inputs["producer_model"]
            judge_model = inputs["judge_model"]
            producer_instruction = inputs["producer_instruction"]
            judge_instruction = inputs["judge_instruction"]
            input_text = inputs["input_text"]
            output_field = inputs["output_field"]
        except KeyError as ke:
            return ToolResult.create_error(
                f"LLM Core Refine missing required parameter: {ke}. "
                "Required: producer_model, judge_model, producer_instruction, "
                "judge_instruction, input_text, output_field. "
                "Alternative: use kaos-llm-core-call + kaos-llm-core-judge separately."
            )

        max_iterations = int(inputs.get("max_iterations", 3) or 3)
        min_score = float(inputs.get("min_score", 0.8) or 0.8)

        # Build producer Signature with a single text input + the requested output field.
        field_defs: dict[str, Any] = {
            "text": (str, InputField(description="Input text")),
            output_field: (str, OutputField(description=f"The {output_field} output")),
        }
        producer_sig = create_model(
            "DynamicRefineProducerSignature",
            __base__=Signature,
            __doc__=producer_instruction,
            **field_defs,
        )

        # Phase 9d: honor _meta.kaos_config per-request overrides.
        settings = _settings_for(context)
        producer = Call(
            producer_sig,
            model=producer_model,
            instructions=producer_instruction,
            core_settings=settings,
        )
        judge = Judge(
            producer_sig,
            producer_model=producer_model,
            judge_model=judge_model,
            criteria=judge_instruction,
            core_settings=settings,
        )

        # Refine reads ``feedback_field`` from the judgment object. The
        # JudgmentSignature exposes ``reasoning`` (not ``feedback``), so we
        # point Refine at it.
        refine = Refine(
            producer,
            judge,
            max_iterations=max_iterations,
            min_score=min_score,
            feedback_field="reasoning",
            core_settings=settings,
        )

        result = await refine(text=input_text)

        output_value: Any = None
        try:
            output_value = getattr(result.outputs, output_field, None)
        except AttributeError:
            output_value = None

        history: list[dict[str, Any]] = []
        for entry in result.history:
            entry_output: Any = None
            try:
                entry_output = getattr(entry.output, output_field, None)
            except AttributeError:
                entry_output = None
            history.append(
                {
                    "iteration": entry.iteration,
                    "output": entry_output,
                    "score": entry.score,
                    "feedback": entry.feedback,
                }
            )

        output: dict[str, Any] = {
            "output": output_value,
            "final_score": result.final_score,
            "iterations": result.iterations,
            "stop_reason": result.stop_reason,
            "history": history,
        }

        return ToolResult.create_success(
            output=output,
            summary=(
                f"Refine: {result.iterations} iter, score={result.final_score:.2f}, "
                f"stop={result.stop_reason}"
            ),
        )
