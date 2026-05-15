"""register_llm_core_tools — bulk-register every kaos-llm-core MCP tool.

The PRD `kaos-modules/docs/internal/dynamic-tool-planning-prd.md` §4
splits this registration into two granular entry points that match
the SessionToolSet group taxonomy in kaos-agents:

- :func:`register_llm_core_program_tools` — the 24 typed-program
  wrappers (Call, ChainOfThought, ReAct, Refine, Judge, Ensemble,
  Evaluate, optimizers, codecs, MIPRO, batch ops, recipe tuning,
  metric, cost report, save/load). These are the "programs" group:
  power-user surfaces denied by default at the ceiling and opted
  into per-session.
- :func:`register_llm_core_alpha_tools` — the 6 ``kaos-llm-core-alpha-*``
  rule-based extractors (date, duration, entity, money, number,
  percent). Same "programs" group — also denied by default — but
  separated so a SessionToolSet that wants the deterministic
  extractors without the heavier program wrappers can opt in
  selectively.

:func:`register_llm_core_tools` remains the backward-compatible
union — every existing caller continues to see the same 30 tools.
"""

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
from kaos_llm_core.integrations.mcp.program_of_thought import KaosLLMCoreProgramOfThoughtTool
from kaos_llm_core.integrations.mcp.react import KaosLLMCoreReActTool
from kaos_llm_core.integrations.mcp.recipe_tune import KaosLLMCoreRecipeTuneTool
from kaos_llm_core.integrations.mcp.refine import KaosLLMCoreRefineTool
from kaos_llm_core.integrations.mcp.save_load import KaosLLMCoreSaveLoadTool


def _ensure_settings(runtime: KaosRuntime) -> None:
    """Settings hydration is idempotent across the two split registrations."""
    from kaos_llm_core.settings import KaosLLMCoreSettings

    if "llm_core" not in runtime.module_settings:
        runtime.module_settings["llm_core"] = KaosLLMCoreSettings()


def register_llm_core_program_tools(runtime: KaosRuntime) -> int:
    """Register the 24 typed-program wrapper tools.

    Returns the number of tools registered. Per the PRD this is the
    "programs" group of the SessionToolSet ceiling — denied by default
    so that a fresh session does not expose the full optimizer /
    codec / batch surface to the LLM, but opt-in per-session for
    power users who want the typed-program callables.
    """
    _ensure_settings(runtime)
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
        KaosLLMCoreProgramOfThoughtTool(),
        KaosLLMCoreBatchCreateTool(),
        KaosLLMCoreBatchRunTool(),
        KaosLLMCoreBatchStatusTool(),
        KaosLLMCoreBatchResultsTool(),
        KaosLLMCoreMiproV2Tool(),
    ]
    for tool in tools:
        runtime.tools.register_tool(tool)
    return len(tools)


def register_llm_core_alpha_tools(runtime: KaosRuntime) -> int:
    """Register the 6 deterministic ``kaos-llm-core-alpha-*`` extractors.

    Returns the number of tools registered. Rule-based primitives
    (date, duration, entity, money, number, percent) — no provider
    calls, no I/O, no state mutation. Same "programs" group as the
    typed-program wrappers, denied by default at the SessionToolSet
    ceiling but registered in the runtime so power-user sessions
    can opt them in.
    """
    _ensure_settings(runtime)
    tools: list[KaosTool] = [
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


def register_llm_core_tools(runtime: KaosRuntime) -> int:
    """Register all kaos-llm-core MCP tools with the runtime.

    Backward-compatible union of
    :func:`register_llm_core_program_tools` and
    :func:`register_llm_core_alpha_tools`. Existing callers see the
    same 30 tools as before — no schema, name, or behavior changes.
    """
    count = register_llm_core_program_tools(runtime)
    count += register_llm_core_alpha_tools(runtime)
    return count
