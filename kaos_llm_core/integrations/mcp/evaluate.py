"""KaosLLMCoreEvaluateTool — see kaos_llm_core.tools (Phase 14B split)."""

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


class KaosLLMCoreEvaluateTool(BaseLLMCoreTool):
    """Evaluate an LLM against labeled examples and compute accuracy."""

    _NAME: ClassVar[str] = "kaos-llm-core-evaluate"
    _DISPLAY_NAME: ClassVar[str] = "LLM Core Evaluate"
    _DESCRIPTION: ClassVar[str] = (
        "Evaluate an LLM's performance on labeled examples. For each example, "
        "calls the model and compares the output to the expected result. Returns "
        "accuracy and per-example results. Use before kaos-llm-core-optimize to "
        "establish a baseline, or after optimization to verify improvement."
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
            description="Model to evaluate (e.g., 'anthropic:claude-haiku-4-5').",
            required=True,
        ),
        ParameterSchema(
            name="instruction",
            type="string",
            description="System instruction for the LLM task.",
            required=True,
        ),
        ParameterSchema(
            name="examples",
            type="array",
            description=(
                "Array of {input, expected_output} objects. Each input is the text "
                "to process, expected_output is the expected result string. "
                'Example: [{"input": "SEC v. Acme", "expected_output": "lawsuit"}].'
            ),
            required=True,
        ),
    ]
    _ERROR_HINT: ClassVar[str] = (
        "Check model identifier and API key. "
        "Alternative: use kaos-llm-core-call to test individual examples manually."
    )

    async def _run(self, inputs: dict[str, Any], context: KaosContext | None = None) -> ToolResult:
        from pydantic import create_model

        from kaos_llm_core import InputField, OutputField, Signature
        from kaos_llm_core.optimization.evaluation import evaluate
        from kaos_llm_core.programs.call import Call
        from kaos_llm_core.types import Example

        model = inputs["model"]
        instruction = inputs["instruction"]
        examples_raw = inputs["examples"]

        if isinstance(examples_raw, str):
            examples_raw = json.loads(examples_raw)

        if not examples_raw or not isinstance(examples_raw, list):
            return ToolResult.create_error(
                "examples must be a non-empty list of {input, expected_output} objects. "
                'Example: [{"input": "text to process", "expected_output": "expected result"}].'
            )

        # Build a simple signature with text input and answer output
        sig = create_model(
            "DynamicEvalSignature",
            __base__=Signature,
            __doc__=instruction,
            text=(str, InputField(description="Input text")),
            answer=(str, OutputField(description="The answer")),
        )

        # Build Example objects
        dataset: list[Example] = []
        for ex in examples_raw:
            if "input" not in ex or "expected_output" not in ex:
                return ToolResult.create_error(
                    "Each example must have 'input' and 'expected_output' keys. "
                    f"Got keys: {list(ex.keys())}. "
                    'Example: {{"input": "some text", "expected_output": "expected result"}}.'
                )
            dataset.append(
                Example(
                    inputs={"text": ex["input"]},
                    outputs={"answer": ex["expected_output"]},
                )
            )

        # Phase 9c follow-up: honor _meta.kaos_config per-request overrides.
        settings = _settings_for(context)
        call = Call(sig, model=model, instructions=instruction, core_settings=settings)

        def exact_match(prediction: Any, gold: dict[str, Any]) -> float:
            pred_answer = str(getattr(prediction, "answer", "")).strip().lower()
            gold_answer = str(gold.get("answer", "")).strip().lower()
            return 1.0 if pred_answer == gold_answer else 0.0

        eval_result = await evaluate(call, dataset, exact_match)

        per_example: list[dict[str, Any]] = []
        for er in eval_result.per_example:
            entry: dict[str, Any] = {
                "input": er.example.inputs.get("text", ""),
                "expected": er.example.outputs.get("answer", ""),
                "actual": str(getattr(er.prediction, "answer", "")) if er.prediction else "",
                "correct": er.score >= 1.0,
            }
            if er.error:
                entry["error"] = er.error
            per_example.append(entry)

        output: dict[str, Any] = {
            "accuracy": eval_result.accuracy,
            "results": per_example,
        }

        return ToolResult.create_success(
            output=output,
            summary=(
                f"Evaluation: {eval_result.accuracy:.1%} accuracy "
                f"({eval_result.n_correct}/{eval_result.n_total}), "
                f"model: {model}"
            ),
        )
