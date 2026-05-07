"""KaosLLMCoreCostReportTool — see kaos_llm_core.tools (Phase 14B split)."""

from __future__ import annotations

import json
from typing import Any, ClassVar

from kaos_core import KaosContext, ToolResult
from kaos_core.types.annotations import ToolAnnotations
from kaos_core.types.enums import ToolCapability, ToolCategory
from kaos_core.types.parameters import ParameterSchema

from kaos_llm_core.integrations.mcp._common import BaseLLMCoreTool


class KaosLLMCoreCostReportTool(BaseLLMCoreTool):
    """Compute cost estimates from an ExecutionTrace."""

    _NAME: ClassVar[str] = "kaos-llm-core-cost-report"
    _DISPLAY_NAME: ClassVar[str] = "LLM Core Cost Report"
    _DESCRIPTION: ClassVar[str] = (
        "Parse an ExecutionTrace JSON and compute cost estimates. Returns total "
        "cost, token counts, and per-call breakdown. Use after kaos-llm-core-call "
        "or kaos-llm-core-reason to understand costs. This is a local computation "
        "that does not make API calls."
    )
    _CATEGORY: ClassVar[ToolCategory] = ToolCategory.UTILITY
    _CAPABILITY: ClassVar[ToolCapability] = ToolCapability.ANALYZE
    _ANNOTATIONS: ClassVar[ToolAnnotations] = ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
    _PARAMETERS: ClassVar[list[ParameterSchema]] = [
        ParameterSchema(
            name="trace_json",
            type="string",
            description=(
                "ExecutionTrace serialized as a JSON string. Obtain by "
                "calling ``await call.invoke(...)`` (or ``program.invoke(...)``) "
                "and serializing ``invocation.trace`` via "
                "``invocation.trace.model_dump_json()``."
            ),
            required=True,
        ),
    ]
    _ERROR_HINT: ClassVar[str] = (
        "Provide a valid ExecutionTrace JSON string. "
        "Obtain it by calling ``await call.invoke(...)`` and serializing "
        "``invocation.trace.model_dump_json()`` — there is no ``last_trace`` "
        "attribute in the Phase 10 runtime contract."
    )

    async def _run(self, inputs: dict[str, Any], context: KaosContext | None = None) -> ToolResult:
        from kaos_llm_core.observability.cost import estimate_cost, format_cost_report
        from kaos_llm_core.observability.traces import ExecutionTrace

        trace_json = inputs["trace_json"]

        if isinstance(trace_json, dict):
            trace_data = trace_json
        else:
            try:
                trace_data = json.loads(trace_json)
            except json.JSONDecodeError as jde:
                return ToolResult.create_error(
                    f"Invalid JSON in trace_json: {jde}. "
                    "Provide a valid JSON string from ExecutionTrace.model_dump_json(). "
                    "Example: call a kaos-llm-core-call tool first, then pass its trace."
                )

        trace = ExecutionTrace.model_validate(trace_data)
        total_cost = estimate_cost(trace)
        report = format_cost_report(trace)

        # Build per-call breakdown
        calls: list[dict[str, Any]] = []
        if trace.children:
            for child in trace.children:
                child_cost = estimate_cost(child)
                calls.append(
                    {
                        "name": child.call_name,
                        "model": child.model,
                        "cost_usd": round(child_cost, 6),
                        "tokens": child.total_tokens,
                    }
                )
        else:
            # Leaf trace — report itself
            calls.append(
                {
                    "name": trace.call_name,
                    "model": trace.model,
                    "cost_usd": round(total_cost, 6),
                    "tokens": trace.total_tokens,
                }
            )

        output: dict[str, Any] = {
            "total_cost_usd": round(total_cost, 6),
            "total_input_tokens": trace.input_tokens,
            "total_output_tokens": trace.output_tokens,
            "calls": calls,
            "report": report,
        }

        return ToolResult.create_success(
            output=output,
            summary=f"Cost: ${total_cost:.4f}, tokens: {trace.total_tokens:,}",
        )
