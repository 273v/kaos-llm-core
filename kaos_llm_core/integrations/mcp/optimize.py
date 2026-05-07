"""KaosLLMCoreOptimizeTool — see kaos_llm_core.tools (Phase 14B split)."""

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


class KaosLLMCoreOptimizeTool(BaseLLMCoreTool):
    """Optimize an LLM instruction using labeled examples."""

    _NAME: ClassVar[str] = "kaos-llm-core-optimize"
    _DISPLAY_NAME: ClassVar[str] = "LLM Core Optimize"
    _DESCRIPTION: ClassVar[str] = (
        "Optimize an LLM system instruction using labeled examples. Analyzes "
        "failures and proposes improved instructions. Use after kaos-llm-core-evaluate "
        "reveals low accuracy. Requires labeled examples with expected outputs. "
        "For evaluation without optimization, use kaos-llm-core-evaluate."
    )
    _CATEGORY: ClassVar[ToolCategory] = ToolCategory.INTEGRATION
    _CAPABILITY: ClassVar[ToolCapability] = ToolCapability.TRANSFORM
    _ANNOTATIONS: ClassVar[ToolAnnotations] = ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    )
    _PARAMETERS: ClassVar[list[ParameterSchema]] = [
        ParameterSchema(
            name="model",
            type="string",
            description=(
                "Model to optimize the instruction for (e.g., 'anthropic:claude-haiku-4-5')."
            ),
            required=True,
        ),
        ParameterSchema(
            name="instruction",
            type="string",
            description="The current system instruction to optimize.",
            required=True,
        ),
        ParameterSchema(
            name="examples",
            type="array",
            description=(
                "Array of {input, expected_output} objects for optimization. "
                "More examples (5-20) give better results. "
                'Example: [{"input": "SEC v. Acme", "expected_output": "lawsuit"}].'
            ),
            required=True,
        ),
        ParameterSchema(
            name="num_iterations",
            type="integer",
            description="Number of optimization iterations (default 3).",
            required=False,
            constraints={"minimum": 1, "maximum": 10},
        ),
    ]
    _ERROR_HINT: ClassVar[str] = (
        "Check model identifier and API key. "
        "Alternative: use kaos-llm-core-evaluate to diagnose failures before optimizing."
    )

    async def _run(self, inputs: dict[str, Any], context: KaosContext | None = None) -> ToolResult:
        from pydantic import create_model

        from kaos_llm_core import InputField, OutputField, Signature
        from kaos_llm_core.optimization.instruction import InstructionOptimizer
        from kaos_llm_core.programs.call import Call
        from kaos_llm_core.types import Example

        model = inputs["model"]
        instruction = inputs["instruction"]
        examples_raw = inputs["examples"]
        num_iterations = inputs.get("num_iterations", 3)

        if isinstance(examples_raw, str):
            examples_raw = json.loads(examples_raw)

        if not examples_raw or not isinstance(examples_raw, list):
            return ToolResult.create_error(
                "examples must be a non-empty list of {input, expected_output} objects. "
                "Provide 5-20 labeled examples for best results. "
                "Use kaos-llm-core-evaluate first to establish a baseline."
            )

        # Build a simple signature
        sig = create_model(
            "DynamicOptSignature",
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

        optimizer = InstructionOptimizer(
            metric=exact_match,
            max_trials=num_iterations,
        )
        result = await optimizer.optimize(call=call, val_set=dataset)

        output: dict[str, Any] = {
            "original_instruction": result.instruction_before,
            "optimized_instruction": result.instruction_after,
            "improvement_notes": result.rationale,
        }

        summary_parts = [
            f"Score: {result.metric_before:.1%} -> {result.metric_after:.1%}",
            f"accepted={result.accepted}",
            f"trials={result.trials}",
        ]

        return ToolResult.create_success(
            output=output,
            summary=", ".join(summary_parts),
        )
