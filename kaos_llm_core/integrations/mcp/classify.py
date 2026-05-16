"""KaosLLMCoreClassifyTool — declarative classification (plan §7.3).

Wraps :func:`kaos_llm_core.starter.classify_doc` so an agent can ask
"classify this against these labels" without constructing a
:class:`ZeroShotClassify` / :class:`ChunkedClassify` Program graph by
hand. The tool returns the full
:class:`~kaos_llm_core.results.Classification` (picked labels +
per-label scores + abstain flag + metadata).
"""

from __future__ import annotations

from typing import Any, ClassVar

from kaos_core import KaosContext, ToolResult
from kaos_core.types.enums import ToolCapability, ToolCategory
from kaos_core.types.parameters import ParameterSchema

from kaos_llm_core.integrations.mcp._common import (
    _LLM_CORE_ANNOTATIONS,
    BaseLLMCoreTool,
)


class KaosLLMCoreClassifyTool(BaseLLMCoreTool):
    """Declarative classification façade — plan §7.3."""

    _NAME: ClassVar[str] = "kaos-llm-core-classify"
    _DISPLAY_NAME: ClassVar[str] = "LLM Core Classify"
    _DESCRIPTION: ClassVar[str] = (
        "Classify a text document against a label set using the kaos-llm-core "
        "§7.1 declarative façade. Picks ZeroShotClassify (or FewShotClassify "
        "when supervision='few_shot') for short inputs and wraps it in "
        "ChunkedClassify for long inputs automatically. Returns the full "
        "Classification structure (labels + scores + abstained + metadata). "
        "The labels parameter accepts either a flat list of names or a full "
        "serialized LabelSet (with policy flags + per-label descriptions)."
    )
    _CATEGORY: ClassVar[ToolCategory] = ToolCategory.INTEGRATION
    _CAPABILITY: ClassVar[ToolCapability] = ToolCapability.QUERY
    _ANNOTATIONS = _LLM_CORE_ANNOTATIONS
    _PARAMETERS: ClassVar[list[ParameterSchema]] = [
        ParameterSchema(
            name="text",
            type="string",
            description="Source text to classify. Must be non-empty.",
            required=True,
        ),
        ParameterSchema(
            name="labels",
            type="array",
            description=(
                "Either a flat list of label name strings (e.g. "
                "['contract','memo','letter']) — the tool builds a flat "
                "exclusive LabelSet — OR a single-element array containing "
                "a serialized LabelSet object (LabelSet.model_dump()) with "
                "explicit exclusive / allow_abstain / hierarchical flags. "
                "Use the second form when you need multi-label or a "
                "hierarchical taxonomy."
            ),
            required=True,
        ),
        ParameterSchema(
            name="model",
            type="string",
            description=("Provider-prefixed model id. Defaults to KAOS_LLM_CORE_DEFAULT_MODEL."),
            required=False,
        ),
        ParameterSchema(
            name="supervision",
            type="string",
            description=(
                "'zero_shot' (default) or 'few_shot'. The few-shot path "
                "requires the 'examples' parameter."
            ),
            required=False,
            constraints={"enum": ["zero_shot", "few_shot"]},
        ),
        ParameterSchema(
            name="long_strategy",
            type="string",
            description=(
                "One of 'auto' (default — single for inputs <= 12k chars, "
                "chunk otherwise), 'single', or 'chunk'."
            ),
            required=False,
            constraints={"enum": ["auto", "single", "chunk"]},
        ),
        ParameterSchema(
            name="aggregator",
            type="string",
            description=(
                "Short name of the aggregation strategy used by the "
                "chunked path. One of 'vote', 'majority', 'union', "
                "'intersection', 'weighted', 'max_score'. Ignored on "
                "the single-call path. Defaults to 'majority' for "
                "exclusive labels and 'union' for multi-label."
            ),
            required=False,
            constraints={
                "enum": [
                    "vote",
                    "majority",
                    "union",
                    "intersection",
                    "weighted",
                    "max_score",
                ]
            },
        ),
        ParameterSchema(
            name="budget_tokens",
            type="integer",
            description="Optional token budget cap.",
            required=False,
        ),
        ParameterSchema(
            name="budget_usd",
            type="number",
            description="Optional cost budget cap in USD.",
            required=False,
        ),
    ]
    _ERROR_HINT: ClassVar[str] = (
        "Pass a non-empty 'text' parameter and a non-empty 'labels' "
        "parameter (either a list of label-name strings or a single "
        "serialized LabelSet)."
    )

    async def _run(self, inputs: dict[str, Any], context: KaosContext | None = None) -> ToolResult:
        from kaos_llm_core.labels import LabelSet
        from kaos_llm_core.optimization.budget import Budget
        from kaos_llm_core.starter import classify_doc

        text = inputs.get("text")
        if not isinstance(text, str) or not text:
            return ToolResult.create_error(
                "classify requires a non-empty 'text' parameter (string)."
            )

        raw_labels = inputs.get("labels")
        if not isinstance(raw_labels, list) or not raw_labels:
            return ToolResult.create_error(
                "classify requires a non-empty 'labels' array — either label "
                "names or a single serialized LabelSet object."
            )

        labels: LabelSet
        if len(raw_labels) == 1 and isinstance(raw_labels[0], dict):
            labels = LabelSet.model_validate(raw_labels[0])
        elif all(isinstance(item, str) for item in raw_labels):
            labels = LabelSet.from_names(raw_labels)
        else:
            return ToolResult.create_error(
                "classify 'labels' must be either a list of name strings or "
                "a single-element list containing a serialized LabelSet."
            )

        budget_tokens = inputs.get("budget_tokens")
        budget_usd = inputs.get("budget_usd")
        budget: Budget | None = None
        if budget_tokens is not None or budget_usd is not None:
            budget = Budget(
                max_tokens=int(budget_tokens) if budget_tokens is not None else None,
                max_cost_usd=float(budget_usd) if budget_usd is not None else None,
            )

        classification = await classify_doc(
            text,
            labels,
            model=inputs.get("model"),
            supervision=inputs.get("supervision", "zero_shot"),
            long_strategy=inputs.get("long_strategy", "auto"),
            aggregator=inputs.get("aggregator"),
            budget=budget,
        )

        payload = classification.model_dump(mode="json")
        top = classification.top_label or "(abstained)"
        return ToolResult.create_success(
            output=payload,
            summary=(
                f"Classification: {top} "
                f"(strategy={classification.metadata.get('starter.long_strategy', 'single')})"
            ),
        )
