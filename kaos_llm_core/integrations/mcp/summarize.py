"""KaosLLMCoreSummarizeTool — declarative summarization (plan §7.3).

Wraps :func:`kaos_llm_core.starter.summarize_doc` so an agent can call
the §7.1 declarative façade without constructing a Program graph or
JSON envelope by hand. The tool returns the full
:class:`~kaos_llm_core.results.Summary` (text + provenance + metadata)
as the structured output.

Complements (does not replace) the existing
:class:`KaosLLMCoreProgramExecuteTool` and the lower-level
:class:`KaosLLMCoreCallTool`: callers reach for *this* tool when they
want the "I just want a summary, pick the right strategy for me"
shape, and reach for the others when they're building bespoke
multi-step compositions.
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


class KaosLLMCoreSummarizeTool(BaseLLMCoreTool):
    """Declarative summarization façade — plan §7.3."""

    _NAME: ClassVar[str] = "kaos-llm-core-summarize"
    _DISPLAY_NAME: ClassVar[str] = "LLM Core Summarize"
    _DESCRIPTION: ClassVar[str] = (
        "Summarize a text document using the kaos-llm-core §7.1 declarative "
        "façade. Picks single-shot abstractive for short inputs and the "
        "hierarchical tree reducer for long inputs automatically. Supports "
        "optional citation-grounding (cited=true), explicit strategy override "
        "('single' / 'tree' / 'refine'), and a token / cost budget that "
        "stops processing partway through and returns a partial Summary tagged "
        "with metadata.partial=true. Returns the full Summary structure "
        "(text + method + chunks_used + source_spans + metadata). For typed "
        "Pydantic-schema output use kaos-llm-core-program-execute with a "
        "StructuredSummary envelope instead."
    )
    _CATEGORY: ClassVar[ToolCategory] = ToolCategory.INTEGRATION
    _CAPABILITY: ClassVar[ToolCapability] = ToolCapability.QUERY
    _ANNOTATIONS = _LLM_CORE_ANNOTATIONS
    _PARAMETERS: ClassVar[list[ParameterSchema]] = [
        ParameterSchema(
            name="text",
            type="string",
            description="Source text to summarize. Must be non-empty.",
            required=True,
        ),
        ParameterSchema(
            name="model",
            type="string",
            description=(
                "Provider-prefixed model id (e.g. 'anthropic:claude-haiku-4-5'). "
                "Defaults to KAOS_LLM_CORE_DEFAULT_MODEL."
            ),
            required=False,
        ),
        ParameterSchema(
            name="long_strategy",
            type="string",
            description=(
                "One of 'auto' (default — picks single for inputs <= 12k chars, "
                "tree otherwise), 'single', 'tree', or 'refine'."
            ),
            required=False,
            constraints={"enum": ["auto", "single", "tree", "refine"]},
        ),
        ParameterSchema(
            name="cited",
            type="boolean",
            description=(
                "When true, route the single-call path through CitedSummary so "
                "every claim is grounded in verbatim source spans. Long-doc "
                "strategies stay free-form abstractive in 0.1.0a10."
            ),
            required=False,
        ),
        ParameterSchema(
            name="budget_tokens",
            type="integer",
            description="Optional token budget cap; processing halts when reached.",
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
        "Pass a non-empty 'text' parameter and ensure either 'model' is "
        "provided or KAOS_LLM_CORE_DEFAULT_MODEL is set in the environment."
    )

    async def _run(self, inputs: dict[str, Any], context: KaosContext | None = None) -> ToolResult:
        from kaos_llm_core.optimization.budget import Budget
        from kaos_llm_core.starter import summarize_doc

        text = inputs.get("text")
        if not isinstance(text, str) or not text:
            return ToolResult.create_error(
                "summarize requires a non-empty 'text' parameter (string)."
            )

        budget_tokens = inputs.get("budget_tokens")
        budget_usd = inputs.get("budget_usd")
        budget: Budget | None = None
        if budget_tokens is not None or budget_usd is not None:
            budget = Budget(
                max_tokens=int(budget_tokens) if budget_tokens is not None else None,
                max_cost_usd=float(budget_usd) if budget_usd is not None else None,
            )

        summary = await summarize_doc(
            text,
            model=inputs.get("model"),
            long_strategy=inputs.get("long_strategy", "auto"),
            cited=bool(inputs.get("cited", False)),
            budget=budget,
        )

        payload = summary.model_dump(mode="json")
        return ToolResult.create_success(
            output=payload,
            summary=(
                f"Summary ({summary.method}, "
                f"strategy={summary.metadata.get('starter.long_strategy', 'single')}): "
                f"{summary.text[:80]}{'…' if len(summary.text) > 80 else ''}"
            ),
        )
