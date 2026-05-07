"""KaosLLMCoreOptimizeCodecTool — see kaos_llm_core.tools (Phase 14B split)."""

from __future__ import annotations

import json
from typing import Any, ClassVar

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


class KaosLLMCoreOptimizeCodecTool(BaseLLMCoreTool):
    """Try each codec (JSON/Chat/XML) and pick the best for a given Call."""

    _NAME: ClassVar[str] = "kaos-llm-core-optimize-codec"
    _DISPLAY_NAME: ClassVar[str] = "LLM Core Optimize Codec"
    _DESCRIPTION: ClassVar[str] = (
        "Try each registered codec (JSONCodec, ChatCodec, XMLCodec) against "
        "a labeled validation set and return the best codec by metric score. "
        "Use after kaos-llm-core-evaluate reveals that codec format matters for "
        "your model. For cross-model search, use kaos-llm-core-optimize-model. "
        "Metric transport: specify metric_name since MCP cannot carry callables; "
        "for custom metrics use the Python API (CodecOptimizer)."
    )
    _CATEGORY: ClassVar[ToolCategory] = ToolCategory.INTEGRATION
    _CAPABILITY: ClassVar[ToolCapability] = ToolCapability.TRANSFORM
    _ANNOTATIONS: ClassVar[ToolAnnotations] = _PHASE6_ANNOTATIONS
    _PARAMETERS: ClassVar[list[ParameterSchema]] = [
        ParameterSchema(
            name="model",
            type="string",
            description="Model identifier (e.g., 'anthropic:claude-haiku-4-5').",
            required=True,
        ),
        ParameterSchema(
            name="instruction",
            type="string",
            description="System instruction for the Call under optimization.",
            required=True,
        ),
        ParameterSchema(
            name="examples",
            type="array",
            description=("Validation examples: array of {input, expected_output} objects."),
            required=True,
        ),
        ParameterSchema(
            name="codecs",
            type="array",
            description=(
                "Optional list of codec names to try. Default: all three. "
                "Valid names: 'JSONCodec', 'ChatCodec', 'XMLCodec'."
            ),
            required=False,
        ),
        ParameterSchema(
            name="metric_name",
            type="string",
            description=(
                "Metric to optimize against. One of 'exact_match' (default), "
                "'case_insensitive_match', 'length_ratio'. For custom metrics "
                "use the Python API."
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
        "Check the model identifier and the examples format. "
        "Alternative: use CodecOptimizer in the Python API."
    )

    async def _run(self, inputs: dict[str, Any], context: KaosContext | None = None) -> ToolResult:
        from kaos_llm_core.codecs import ChatCodec, JSONCodec, XMLCodec
        from kaos_llm_core.optimization.codec_optimizer import CodecOptimizer
        from kaos_llm_core.programs.call import Call
        from kaos_llm_core.settings import KaosLLMCoreSettings

        settings = (
            KaosLLMCoreSettings.from_context(context)
            if context is not None
            else KaosLLMCoreSettings()
        )

        metric_name = inputs.get("metric_name", "normalized_match")
        metric = _resolve_metric_by_name(metric_name)
        if metric is None:
            return ToolResult.create_error(
                f"Unknown metric_name={metric_name!r}. "
                "Use 'exact_match', 'case_insensitive_match', or 'length_ratio'. "
                "For custom metrics use the Python API (CodecOptimizer)."
            )

        parsed = _build_eval_dataset(inputs["examples"])
        if isinstance(parsed, ToolResult):
            return parsed
        sig, dataset = parsed

        codec_name_map: dict[str, Any] = {
            "JSONCodec": JSONCodec,
            "ChatCodec": ChatCodec,
            "XMLCodec": XMLCodec,
        }
        codec_names = inputs.get("codecs")
        if codec_names:
            if isinstance(codec_names, str):
                codec_names = json.loads(codec_names)
            try:
                codecs = [codec_name_map[n] for n in codec_names]
            except KeyError as ke:
                return ToolResult.create_error(
                    f"Unknown codec name: {ke}. "
                    "Valid codec names: 'JSONCodec', 'ChatCodec', 'XMLCodec'."
                )
        else:
            codecs = None

        call = Call(
            sig,
            model=inputs["model"],
            instructions=inputs["instruction"],
            core_settings=settings,
        )
        optimizer = CodecOptimizer(metric=metric, codecs=codecs)
        result = await optimizer.optimize(call, dataset)

        output: dict[str, Any] = {
            "best_codec": result.best_codec.__name__,
            "best_score": result.best_score,
            "scores_by_codec": result.scores_by_codec,
            "stop_reason": result.stop_reason,
        }
        return ToolResult.create_success(
            output=output,
            summary=(
                f"Best codec: {result.best_codec.__name__} "
                f"(score={result.best_score:.3f}, tried {len(result.scores_by_codec)})"
            ),
        )
