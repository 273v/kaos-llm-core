"""KaosLLMCoreBestOfNTool — see kaos_llm_core.tools (Phase 14B split)."""

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


class KaosLLMCoreBestOfNTool(BaseLLMCoreTool):
    """Sample one model N times and pick the best candidate."""

    _NAME: ClassVar[str] = "kaos-llm-core-best-of-n"
    _DISPLAY_NAME: ClassVar[str] = "LLM Core Best-of-N"
    _DESCRIPTION: ClassVar[str] = (
        "Sample the same producer N times against one model and pick the "
        "best candidate. If judge_model is provided, a Judge selector "
        "scores each candidate on quality. WARNING: Without judge_model, "
        "selection defaults to longest output (a dummy heuristic that "
        "optimizes verbosity, NOT quality). Always provide judge_model "
        "for meaningful selection. The result without a judge "
        "should not be considered authoritative. "
        "Use this for sampling-based reliability — distinct from "
        "kaos-llm-core-ensemble which votes across DIFFERENT models. "
        "n must be >= 2."
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
            description="Producer model identifier.",
            required=True,
        ),
        ParameterSchema(
            name="instruction",
            type="string",
            description="System instruction for the producer.",
            required=True,
        ),
        ParameterSchema(
            name="input_text",
            type="string",
            description="The input text to sample against.",
            required=True,
        ),
        ParameterSchema(
            name="output_field",
            type="string",
            description="Name of the producer's single output field.",
            required=True,
        ),
        ParameterSchema(
            name="n",
            type="integer",
            description="Number of samples (must be >= 2).",
            required=True,
            constraints={"minimum": 2, "maximum": 50},
        ),
        ParameterSchema(
            name="judge_model",
            type="string",
            description=(
                "Optional judge model. When set, a Judge selector scores "
                "each candidate. When absent, a dummy length selector is used."
            ),
            required=False,
        ),
        ParameterSchema(
            name="judge_criteria",
            type="string",
            description="Required when judge_model is set: criteria the judge applies.",
            required=False,
        ),
    ]
    _ERROR_HINT: ClassVar[str] = (
        "Check the producer model identifier and API key. "
        "Alternative: use kaos-llm-core-call to make individual samples manually."
    )

    async def _run(self, inputs: dict[str, Any], context: KaosContext | None = None) -> ToolResult:
        from pydantic import create_model

        from kaos_llm_core import InputField, OutputField, Signature
        from kaos_llm_core.programs.best_of_n import BestOfN
        from kaos_llm_core.programs.call import Call
        from kaos_llm_core.programs.judge import Judge

        try:
            model = inputs["model"]
            instruction = inputs["instruction"]
            input_text = inputs["input_text"]
            output_field = inputs["output_field"]
            n_raw = inputs["n"]
        except KeyError as ke:
            return ToolResult.create_error(
                f"LLM Core Best-of-N missing required parameter: {ke}. "
                "Required: model, instruction, input_text, output_field, n. "
                "Alternative: use kaos-llm-core-call for a single sample."
            )

        try:
            n = int(n_raw)
        except (TypeError, ValueError):
            return ToolResult.create_error(
                f"LLM Core Best-of-N: n={n_raw!r} is not an integer. "
                "Pass n as an integer >= 2 (e.g. 3 or 5). "
                "For a single sample, use kaos-llm-core-call instead."
            )

        if n < 2:
            return ToolResult.create_error(
                f"LLM Core Best-of-N: n={n} is invalid. "
                "n must be >= 2 (otherwise there is nothing to choose between). "
                "For a single sample, use kaos-llm-core-call instead."
            )

        judge_model = inputs.get("judge_model")
        judge_criteria = inputs.get("judge_criteria")
        if judge_model and not judge_criteria:
            return ToolResult.create_error(
                "LLM Core Best-of-N: judge_model is set but judge_criteria is missing. "
                "Provide judge_criteria (e.g. 'accuracy and clarity') so the Judge "
                "knows what to score against. Alternative: omit judge_model to use "
                "the deterministic length-based selector."
            )

        field_defs: dict[str, Any] = {
            "text": (str, InputField(description="Input text")),
            output_field: (str, OutputField(description=f"The {output_field} output")),
        }
        sig = create_model(
            "DynamicBestOfNSignature",
            __base__=Signature,
            __doc__=instruction,
            **field_defs,
        )

        # Phase 9d: honor _meta.kaos_config per-request overrides.
        settings = _settings_for(context)
        producer = Call(sig, model=model, instructions=instruction, core_settings=settings)

        selector: Any
        if judge_model:
            selector = Judge(
                sig,
                producer_model=model,
                judge_model=judge_model,
                criteria=str(judge_criteria),
                core_settings=settings,
            )
            selection_method = "judge"
        else:
            # Dummy deterministic selector: length of the output field as
            # a string. Documented as a non-authoritative fallback.
            def _length_selector(output: Any, _inputs: dict[str, Any]) -> float:
                val = getattr(output, output_field, None)
                return float(len(str(val))) if val is not None else 0.0

            selector = _length_selector
            selection_method = "length"

        best_of_n = BestOfN(producer, n=n, selector=selector, core_settings=settings)
        result = await best_of_n(text=input_text)

        candidates_serialized: list[Any] = []
        for cand in result.candidates:
            cand_value = getattr(cand, output_field, None)
            candidates_serialized.append(cand_value)

        selected_value = getattr(result.outputs, output_field, None)

        output: dict[str, Any] = {
            "output": selected_value,
            "selected_index": result.selected_index,
            "candidates": candidates_serialized,
            "scores": list(result.scores),
            "selection_method": selection_method,
        }

        return ToolResult.create_success(
            output=output,
            summary=(
                f"BestOfN: n={n}, selected={result.selected_index}, method={selection_method}"
            ),
        )
