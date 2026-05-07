"""KaosLLMCoreAnalyzeTrialTool — see kaos_llm_core.tools (Phase 14B split)."""

from __future__ import annotations

from typing import Any, ClassVar

from kaos_core import KaosContext, ToolResult
from kaos_core.types.annotations import ToolAnnotations
from kaos_core.types.enums import ToolCapability, ToolCategory
from kaos_core.types.parameters import ParameterSchema

from kaos_llm_core.integrations.mcp._common import (
    _PHASE7_ANALYZE_ANNOTATIONS,
    BaseLLMCoreTool,
)


class KaosLLMCoreAnalyzeTrialTool(BaseLLMCoreTool):
    """Analyze a mutation log JSONL file and return trial summaries."""

    _NAME: ClassVar[str] = "kaos-llm-core-analyze-trial"
    _DISPLAY_NAME: ClassVar[str] = "LLM Core Analyze Trial"
    _DESCRIPTION: ClassVar[str] = (
        "Load a mutation log JSONL file (produced by any optimizer with "
        "mutation_log=...), render it into trial cards, compute per-strategy "
        "contributions, and produce a run summary. Pure local computation; "
        "no LLM calls. Use after running kaos-llm-core-optimize, "
        "kaos-llm-core-optimize-codec, kaos-llm-core-optimize-model, or any "
        "other optimizer that writes a mutation log."
    )
    _CATEGORY: ClassVar[ToolCategory] = ToolCategory.UTILITY
    _CAPABILITY: ClassVar[ToolCapability] = ToolCapability.ANALYZE
    _ANNOTATIONS: ClassVar[ToolAnnotations] = _PHASE7_ANALYZE_ANNOTATIONS
    _PARAMETERS: ClassVar[list[ParameterSchema]] = [
        ParameterSchema(
            name="mutation_log_path",
            type="string",
            description="Filesystem path to the mutation log JSONL file.",
            required=True,
        ),
    ]
    _ERROR_HINT: ClassVar[str] = (
        "Verify the path is a valid mutation log JSONL file. "
        "Alternative: use kaos_llm_core.optimization.analysis from Python."
    )

    async def _run(
        self,
        inputs: dict[str, Any],
        context: KaosContext | None = None,
    ) -> ToolResult:
        from kaos_llm_core.optimization.analysis import (
            load_mutations,
            make_trial_cards,
            strategy_contributions,
            summarize_run,
        )

        path = inputs.get("mutation_log_path")
        if not path:
            return ToolResult.create_error(
                "kaos-llm-core-analyze-trial requires 'mutation_log_path'. "
                "Pass the path returned from MutationLog(path=...). "
                "Alternative: use kaos_llm_core.optimization.analysis from Python."
            )

        try:
            mutations = load_mutations(str(path))
        except FileNotFoundError as fnf:
            return ToolResult.create_error(
                f"Trial log file not found: {path}. {fnf} "
                "Verify the path is correct and the file exists. "
                "Generate a trial log by running an optimization first "
                "(e.g., kaos-llm-core-optimize or the Python optimization API)."
            )

        cards = make_trial_cards(mutations)
        contributions = strategy_contributions(mutations)
        summary = summarize_run(mutations)

        output: dict[str, Any] = {
            "summary": summary,
            "trial_cards": [
                {
                    "trial_id": c.trial_id,
                    "strategy": c.strategy,
                    "mutation_type": c.mutation_type,
                    "call_name": c.call_name,
                    "metric_before": c.metric_before,
                    "metric_after": c.metric_after,
                    "improvement": c.improvement,
                    "accepted": c.accepted,
                    "tokens_used": c.tokens_used,
                    "cost_usd": c.cost_usd,
                    "timestamp": c.timestamp.isoformat(),
                    "diff": c.diff,
                }
                for c in cards
            ],
            "strategy_contributions": [
                {
                    "strategy": c.strategy,
                    "trials": c.trials,
                    "wins": c.wins,
                    "total_improvement": c.total_improvement,
                    "average_cost": c.average_cost,
                }
                for c in contributions
            ],
        }
        return ToolResult.create_success(
            output=output,
            summary=(f"Analyzed {len(mutations)} mutations across {len(contributions)} strategies"),
        )
