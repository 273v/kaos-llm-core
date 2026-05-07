"""register_llm_core_tools — bulk-register every kaos-llm-core MCP tool."""

from __future__ import annotations

from kaos_core import KaosRuntime, KaosTool

from kaos_llm_core.integrations.mcp.alpha_date import KaosLLMCoreAlphaDateTool
from kaos_llm_core.integrations.mcp.alpha_duration import KaosLLMCoreAlphaDurationTool
from kaos_llm_core.integrations.mcp.alpha_entity import KaosLLMCoreAlphaEntityTool
from kaos_llm_core.integrations.mcp.alpha_money import KaosLLMCoreAlphaMoneyTool
from kaos_llm_core.integrations.mcp.alpha_number import KaosLLMCoreAlphaNumberTool
from kaos_llm_core.integrations.mcp.alpha_percent import KaosLLMCoreAlphaPercentTool
from kaos_llm_core.integrations.mcp.analyze_trial import KaosLLMCoreAnalyzeTrialTool
from kaos_llm_core.integrations.mcp.batch_create import KaosLLMCoreBatchCreateTool
from kaos_llm_core.integrations.mcp.batch_results import KaosLLMCoreBatchResultsTool
from kaos_llm_core.integrations.mcp.batch_run import KaosLLMCoreBatchRunTool
from kaos_llm_core.integrations.mcp.batch_status import KaosLLMCoreBatchStatusTool
from kaos_llm_core.integrations.mcp.best_of_n import KaosLLMCoreBestOfNTool
from kaos_llm_core.integrations.mcp.call import KaosLLMCoreCallTool
from kaos_llm_core.integrations.mcp.chain_of_thought import KaosLLMCoreChainOfThoughtTool
from kaos_llm_core.integrations.mcp.cost_report import KaosLLMCoreCostReportTool
from kaos_llm_core.integrations.mcp.ensemble import KaosLLMCoreEnsembleTool
from kaos_llm_core.integrations.mcp.evaluate import KaosLLMCoreEvaluateTool
from kaos_llm_core.integrations.mcp.judge import KaosLLMCoreJudgeTool
from kaos_llm_core.integrations.mcp.metric import KaosLLMCoreMetricTool
from kaos_llm_core.integrations.mcp.mipro_v2 import KaosLLMCoreMiproV2Tool
from kaos_llm_core.integrations.mcp.optimize import KaosLLMCoreOptimizeTool
from kaos_llm_core.integrations.mcp.optimize_codec import KaosLLMCoreOptimizeCodecTool
from kaos_llm_core.integrations.mcp.optimize_model import KaosLLMCoreOptimizeModelTool
from kaos_llm_core.integrations.mcp.pareto import KaosLLMCoreParetoTool
from kaos_llm_core.integrations.mcp.program_execute import KaosLLMCoreProgramExecuteTool
from kaos_llm_core.integrations.mcp.react import KaosLLMCoreReActTool
from kaos_llm_core.integrations.mcp.recipe_tune import KaosLLMCoreRecipeTuneTool
from kaos_llm_core.integrations.mcp.refine import KaosLLMCoreRefineTool
from kaos_llm_core.integrations.mcp.save_load import KaosLLMCoreSaveLoadTool


def register_llm_core_tools(runtime: KaosRuntime) -> int:
    """Register all kaos-llm-core MCP tools with the runtime.

    Returns the number of tools registered.
    """
    from kaos_llm_core.settings import KaosLLMCoreSettings

    runtime.module_settings["llm_core"] = KaosLLMCoreSettings()

    tools: list[KaosTool] = [
        KaosLLMCoreCallTool(),
        KaosLLMCoreChainOfThoughtTool(),
        KaosLLMCoreJudgeTool(),
        KaosLLMCoreEnsembleTool(),
        KaosLLMCoreEvaluateTool(),
        KaosLLMCoreOptimizeTool(),
        KaosLLMCoreCostReportTool(),
        KaosLLMCoreReActTool(),
        KaosLLMCoreRefineTool(),
        KaosLLMCoreBestOfNTool(),
        KaosLLMCoreSaveLoadTool(),
        KaosLLMCoreOptimizeCodecTool(),
        KaosLLMCoreOptimizeModelTool(),
        KaosLLMCoreParetoTool(),
        KaosLLMCoreRecipeTuneTool(),
        KaosLLMCoreMetricTool(),
        KaosLLMCoreAnalyzeTrialTool(),
        KaosLLMCoreProgramExecuteTool(),
        KaosLLMCoreBatchCreateTool(),
        KaosLLMCoreBatchRunTool(),
        KaosLLMCoreBatchStatusTool(),
        KaosLLMCoreBatchResultsTool(),
        KaosLLMCoreMiproV2Tool(),
        # WS-TR.PR-6f.7 — rule-based extraction primitives.
        KaosLLMCoreAlphaDateTool(),
        KaosLLMCoreAlphaEntityTool(),
        KaosLLMCoreAlphaMoneyTool(),
        KaosLLMCoreAlphaNumberTool(),
        KaosLLMCoreAlphaPercentTool(),
        KaosLLMCoreAlphaDurationTool(),
    ]

    for tool in tools:
        runtime.tools.register_tool(tool)

    return len(tools)
