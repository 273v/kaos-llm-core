"""Tests for MCP tool definitions."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from kaos_core import KaosContext, KaosRuntime
from kaos_core.vfs.core import VirtualFileSystem
from kaos_core.vfs.models import VFSConfig
from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart, ToolCall

from kaos_llm_core.integrations.mcp.analyze_trial import KaosLLMCoreAnalyzeTrialTool
from kaos_llm_core.integrations.mcp.best_of_n import KaosLLMCoreBestOfNTool
from kaos_llm_core.integrations.mcp.call import KaosLLMCoreCallTool
from kaos_llm_core.integrations.mcp.chain_of_thought import KaosLLMCoreChainOfThoughtTool
from kaos_llm_core.integrations.mcp.cost_report import KaosLLMCoreCostReportTool
from kaos_llm_core.integrations.mcp.ensemble import KaosLLMCoreEnsembleTool
from kaos_llm_core.integrations.mcp.evaluate import KaosLLMCoreEvaluateTool
from kaos_llm_core.integrations.mcp.judge import KaosLLMCoreJudgeTool
from kaos_llm_core.integrations.mcp.metric import KaosLLMCoreMetricTool
from kaos_llm_core.integrations.mcp.optimize import KaosLLMCoreOptimizeTool
from kaos_llm_core.integrations.mcp.optimize_codec import KaosLLMCoreOptimizeCodecTool
from kaos_llm_core.integrations.mcp.optimize_model import KaosLLMCoreOptimizeModelTool
from kaos_llm_core.integrations.mcp.pareto import KaosLLMCoreParetoTool
from kaos_llm_core.integrations.mcp.program_of_thought import KaosLLMCoreProgramOfThoughtTool
from kaos_llm_core.integrations.mcp.react import KaosLLMCoreReActTool
from kaos_llm_core.integrations.mcp.recipe_tune import KaosLLMCoreRecipeTuneTool
from kaos_llm_core.integrations.mcp.refine import KaosLLMCoreRefineTool
from kaos_llm_core.integrations.mcp.save_load import KaosLLMCoreSaveLoadTool


def _json_response(data: dict[str, Any]) -> ProviderResponse:
    return ProviderResponse.model_construct(
        provider="function",
        model="function-test",
        raw={},
        parts=[ContentPart.model_construct(type="text", text=json.dumps(data))],
        usage=UsageInfo.model_construct(input_tokens=10, output_tokens=5, total_tokens=15),
        stop_reason="end_turn",
        status_code=200,
        response_headers={},
    )


class TestCallTool:
    def test_metadata(self) -> None:
        tool = KaosLLMCoreCallTool()
        meta = tool.metadata
        assert meta.name == "kaos-llm-core-call"
        assert meta.annotations is not None
        assert meta.annotations.readOnlyHint is True
        assert meta.annotations.openWorldHint is True
        assert len(meta.input_schema) >= 3

    async def test_execute_success(self) -> None:
        """Tool should produce structured output."""
        import kaos_llm_core.programs.call as call_mod

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"entities": "SEC, Acme", "summary": "SEC sued Acme."})

        client = FunctionClient(function=fn)
        original = call_mod.create_client
        call_mod.create_client = lambda *a, **kw: client  # ty: ignore[invalid-assignment]

        try:
            tool = KaosLLMCoreCallTool()
            result = await tool.execute(
                {
                    "instruction": "Extract entities and summarize.",
                    "text": "The SEC filed suit against Acme Corp.",
                    "output_fields": {"entities": "entity names", "summary": "brief summary"},
                    "model": "function-test",
                }
            )

            assert not result.isError
            structured = result.require_structured()
            assert "entities" in structured
            assert "summary" in structured
        finally:
            call_mod.create_client = original

    async def test_execute_no_model_error(self) -> None:
        """Tool should return error when no model specified and no default."""
        tool = KaosLLMCoreCallTool()
        result = await tool.execute(
            {
                "instruction": "Do something.",
                "text": "Input.",
                "output_fields": {"result": "output"},
            }
        )
        assert result.isError

    async def test_execute_json_string_output_fields(self) -> None:
        """output_fields can be a JSON string."""
        import kaos_llm_core.programs.call as call_mod

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"result": "done"})

        client = FunctionClient(function=fn)
        original = call_mod.create_client
        call_mod.create_client = lambda *a, **kw: client  # ty: ignore[invalid-assignment]

        try:
            tool = KaosLLMCoreCallTool()
            result = await tool.execute(
                {
                    "instruction": "Process.",
                    "text": "Input.",
                    "output_fields": '{"result": "the output"}',
                    "model": "function-test",
                }
            )
            assert not result.isError
        finally:
            call_mod.create_client = original


class TestChainOfThoughtTool:
    def test_metadata(self) -> None:
        tool = KaosLLMCoreChainOfThoughtTool()
        meta = tool.metadata
        assert meta.name == "kaos-llm-core-reason"
        assert meta.annotations is not None
        assert meta.annotations.readOnlyHint is True


class TestJudgeTool:
    def test_metadata(self) -> None:
        tool = KaosLLMCoreJudgeTool()
        meta = tool.metadata
        assert meta.name == "kaos-llm-core-judge"
        assert meta.annotations is not None
        assert meta.annotations.readOnlyHint is True
        assert meta.annotations.openWorldHint is True
        assert meta.category == "integration"
        assert meta.capability == "analyze"
        # Verify required params
        param_names = {p.name for p in meta.input_schema}
        assert "model" in param_names
        assert "input_text" in param_names
        assert "output_text" in param_names
        assert "criteria" in param_names
        assert "system" in param_names
        # model, input_text, output_text, criteria required; system optional
        required = {p.name for p in meta.input_schema if p.required}
        assert required == {"model", "input_text", "output_text", "criteria"}

    async def test_execute_missing_params(self) -> None:
        """Judge should return error when required params are missing."""
        tool = KaosLLMCoreJudgeTool()
        result = await tool.execute({"model": "test-model"})
        assert result.isError


class TestEnsembleTool:
    def test_metadata(self) -> None:
        tool = KaosLLMCoreEnsembleTool()
        meta = tool.metadata
        assert meta.name == "kaos-llm-core-ensemble"
        assert meta.annotations is not None
        assert meta.annotations.readOnlyHint is True
        assert meta.annotations.openWorldHint is True
        assert meta.category == "integration"
        assert meta.capability == "query"
        param_names = {p.name for p in meta.input_schema}
        assert "models" in param_names
        assert "instruction" in param_names
        assert "input_text" in param_names
        assert "output_fields" in param_names
        required = {p.name for p in meta.input_schema if p.required}
        assert required == {"models", "instruction", "input_text"}

    async def test_execute_empty_models(self) -> None:
        """Ensemble should return error for empty models list."""
        tool = KaosLLMCoreEnsembleTool()
        result = await tool.execute(
            {
                "models": [],
                "instruction": "Classify.",
                "input_text": "Some text.",
            }
        )
        assert result.isError

    async def test_execute_models_string_parse(self) -> None:
        """Ensemble should handle models as JSON string."""
        tool = KaosLLMCoreEnsembleTool()
        # Empty list as string should still error
        result = await tool.execute(
            {
                "models": "[]",
                "instruction": "Classify.",
                "input_text": "Some text.",
            }
        )
        assert result.isError


class TestEvaluateTool:
    def test_metadata(self) -> None:
        tool = KaosLLMCoreEvaluateTool()
        meta = tool.metadata
        assert meta.name == "kaos-llm-core-evaluate"
        assert meta.annotations is not None
        assert meta.annotations.readOnlyHint is True
        assert meta.annotations.openWorldHint is True
        assert meta.category == "integration"
        assert meta.capability == "analyze"
        param_names = {p.name for p in meta.input_schema}
        assert "model" in param_names
        assert "instruction" in param_names
        assert "examples" in param_names
        required = {p.name for p in meta.input_schema if p.required}
        assert required == {"model", "instruction", "examples"}

    async def test_execute_empty_examples(self) -> None:
        """Evaluate should error on empty examples list."""
        tool = KaosLLMCoreEvaluateTool()
        result = await tool.execute(
            {
                "model": "test-model",
                "instruction": "Classify.",
                "examples": [],
            }
        )
        assert result.isError

    async def test_execute_missing_keys_in_examples(self) -> None:
        """Evaluate should error when example objects lack required keys."""
        tool = KaosLLMCoreEvaluateTool()
        result = await tool.execute(
            {
                "model": "test-model",
                "instruction": "Classify.",
                "examples": [{"text": "hello"}],
            }
        )
        assert result.isError
        assert "input" in (result.text or "")


class TestOptimizeTool:
    def test_metadata(self) -> None:
        tool = KaosLLMCoreOptimizeTool()
        meta = tool.metadata
        assert meta.name == "kaos-llm-core-optimize"
        assert meta.annotations is not None
        assert meta.annotations.readOnlyHint is False
        assert meta.annotations.openWorldHint is True
        assert meta.category == "integration"
        assert meta.capability == "transform"
        param_names = {p.name for p in meta.input_schema}
        assert "model" in param_names
        assert "instruction" in param_names
        assert "examples" in param_names
        assert "num_iterations" in param_names
        required = {p.name for p in meta.input_schema if p.required}
        assert required == {"model", "instruction", "examples"}

    async def test_execute_empty_examples(self) -> None:
        """Optimize should error on empty examples list."""
        tool = KaosLLMCoreOptimizeTool()
        result = await tool.execute(
            {
                "model": "test-model",
                "instruction": "Classify.",
                "examples": [],
            }
        )
        assert result.isError

    async def test_execute_missing_keys_in_examples(self) -> None:
        """Optimize should error when example objects lack required keys."""
        tool = KaosLLMCoreOptimizeTool()
        result = await tool.execute(
            {
                "model": "test-model",
                "instruction": "Classify.",
                "examples": [{"wrong": "keys"}],
            }
        )
        assert result.isError
        assert "input" in (result.text or "")


class TestCostReportTool:
    def test_metadata(self) -> None:
        tool = KaosLLMCoreCostReportTool()
        meta = tool.metadata
        assert meta.name == "kaos-llm-core-cost-report"
        assert meta.annotations is not None
        assert meta.annotations.readOnlyHint is True
        assert meta.annotations.openWorldHint is False
        assert meta.category == "utility"
        assert meta.capability == "analyze"
        param_names = {p.name for p in meta.input_schema}
        assert "trace_json" in param_names
        required = {p.name for p in meta.input_schema if p.required}
        assert required == {"trace_json"}

    async def test_execute_leaf_trace(self) -> None:
        """Cost report should handle a simple leaf trace."""
        tool = KaosLLMCoreCostReportTool()
        trace_dict = {
            "call_name": "TestCall",
            "signature": "TestSig",
            "model": "anthropic:claude-haiku-4-5",
            "input_tokens": 1000,
            "output_tokens": 200,
            "total_tokens": 1200,
            "latency_ms": 450.0,
            "children": [],
        }
        result = await tool.execute({"trace_json": json.dumps(trace_dict)})
        assert not result.isError
        structured = result.require_structured()
        assert "total_cost_usd" in structured
        assert structured["total_cost_usd"] > 0
        assert structured["total_input_tokens"] == 1000
        assert structured["total_output_tokens"] == 200
        assert len(structured["calls"]) == 1
        assert structured["calls"][0]["model"] == "anthropic:claude-haiku-4-5"
        assert structured["calls"][0]["tokens"] == 1200

    async def test_execute_program_trace_with_children(self) -> None:
        """Cost report should handle a program trace with child calls."""
        tool = KaosLLMCoreCostReportTool()
        trace_dict = {
            "call_name": "JudgeProgram",
            "model": "(program)",
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "latency_ms": 1200.0,
            "children": [
                {
                    "call_name": "producer",
                    "model": "openai:gpt-5.4-nano",
                    "input_tokens": 500,
                    "output_tokens": 100,
                    "total_tokens": 600,
                    "latency_ms": 400.0,
                    "children": [],
                },
                {
                    "call_name": "judge",
                    "model": "anthropic:claude-sonnet-4-6",
                    "input_tokens": 800,
                    "output_tokens": 150,
                    "total_tokens": 950,
                    "latency_ms": 600.0,
                    "children": [],
                },
            ],
        }
        result = await tool.execute({"trace_json": json.dumps(trace_dict)})
        assert not result.isError
        structured = result.require_structured()
        assert structured["total_cost_usd"] > 0
        assert len(structured["calls"]) == 2
        # Verify child costs are summed
        child_costs = sum(c["cost_usd"] for c in structured["calls"])
        assert abs(structured["total_cost_usd"] - child_costs) < 1e-6

    async def test_execute_dict_input(self) -> None:
        """Cost report should accept trace_json as a dict directly."""
        tool = KaosLLMCoreCostReportTool()
        trace_dict = {
            "call_name": "TestCall",
            "model": "openai:gpt-5.4-nano",
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "latency_ms": 200.0,
            "children": [],
        }
        result = await tool.execute({"trace_json": trace_dict})
        assert not result.isError
        structured = result.require_structured()
        assert structured["total_cost_usd"] > 0

    async def test_execute_invalid_json(self) -> None:
        """Cost report should return error on invalid JSON."""
        tool = KaosLLMCoreCostReportTool()
        result = await tool.execute({"trace_json": "not valid json{{"})
        assert result.isError
        assert "Invalid JSON" in (result.text or "")

    async def test_execute_unknown_model(self) -> None:
        """Cost report should handle unknown models gracefully (cost=0)."""
        tool = KaosLLMCoreCostReportTool()
        trace_dict = {
            "call_name": "TestCall",
            "model": "unknown:model-x",
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "latency_ms": 100.0,
            "children": [],
        }
        result = await tool.execute({"trace_json": json.dumps(trace_dict)})
        assert not result.isError
        structured = result.require_structured()
        assert structured["total_cost_usd"] == 0.0


# ---------------------------------------------------------------------------
# Helpers for ReAct / Refine / BestOfN tests
# ---------------------------------------------------------------------------


def _text_response(text: str) -> ProviderResponse:
    return ProviderResponse.model_construct(
        provider="function",
        model="function-test",
        raw={},
        parts=[ContentPart.model_construct(type="text", text=text)],
        usage=UsageInfo.model_construct(input_tokens=10, output_tokens=5, total_tokens=15),
        stop_reason="end_turn",
        status_code=200,
        response_headers={},
    )


def _tool_call_response(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    call_id: str = "call_1",
) -> ProviderResponse:
    parts = [
        ContentPart.model_construct(
            type="tool_use",
            tool_call=ToolCall.model_construct(id=call_id, name=tool_name, arguments=arguments),
        )
    ]
    return ProviderResponse.model_construct(
        provider="function",
        model="function-test",
        raw={},
        parts=parts,
        usage=UsageInfo.model_construct(input_tokens=12, output_tokens=8, total_tokens=20),
        stop_reason="tool_use",
        status_code=200,
        response_headers={},
    )


# ---------------------------------------------------------------------------
# KaosLLMCoreReActTool tests
# ---------------------------------------------------------------------------


class TestReActTool:
    def test_metadata(self) -> None:
        tool = KaosLLMCoreReActTool()
        meta = tool.metadata
        assert meta.name == "kaos-llm-core-react"
        assert meta.annotations is not None
        assert meta.annotations.readOnlyHint is True
        assert meta.annotations.openWorldHint is True
        assert meta.category == "integration"
        assert meta.capability == "analyze"
        # Dry-run caveat must be in the description so agents see it.
        assert "dry-run" in (meta.description or "").lower()
        param_names = {p.name for p in meta.input_schema}
        assert "model" in param_names
        assert "instruction" in param_names
        assert "input_text" in param_names
        assert "output_field" in param_names
        assert "tool_specs" in param_names
        assert "max_iterations" in param_names
        required = {p.name for p in meta.input_schema if p.required}
        assert required == {
            "model",
            "instruction",
            "input_text",
            "output_field",
            "tool_specs",
        }

    async def test_execute_missing_required_params(self) -> None:
        """ReAct should error when required params are missing."""
        tool = KaosLLMCoreReActTool()
        result = await tool.execute({"model": "function-test"})
        assert result.isError
        assert "Required" in (result.text or "")

    async def test_execute_tool_specs_must_be_list(self) -> None:
        """tool_specs string that is not JSON parseable returns error."""
        tool = KaosLLMCoreReActTool()
        result = await tool.execute(
            {
                "model": "function-test",
                "instruction": "Answer the question.",
                "input_text": "hi",
                "output_field": "answer",
                "tool_specs": "not valid json!!",
            }
        )
        assert result.isError

    async def test_execute_terminates_immediately(self) -> None:
        """ReAct that produces valid output on turn 1 returns 1 iteration."""
        import kaos_llm_core.programs.call as call_mod

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _text_response(json.dumps({"answer": "42"}))

        client = FunctionClient(model="function-test", function=handler)
        original = call_mod.create_client
        call_mod.create_client = lambda *a, **kw: client  # ty: ignore[invalid-assignment]

        try:
            tool = KaosLLMCoreReActTool()
            result = await tool.execute(
                {
                    "model": "function-test",
                    "instruction": "Answer the question.",
                    "input_text": "What is the meaning of life?",
                    "output_field": "answer",
                    "tool_specs": [],
                }
            )
            assert not result.isError
            structured = result.require_structured()
            assert structured["output"] == "42"
            assert structured["iterations_used"] == 1
            assert structured["stop_reason"] == "TERMINATED"
            assert len(structured["trajectory"]) == 1
            assert structured["trajectory"][0]["tool_calls"] == []
        finally:
            call_mod.create_client = original

    async def test_execute_dry_run_tool_then_finalizes(self) -> None:
        """ReAct that emits one tool call then finalizes: 2 iterations, dry-run stub invoked."""
        import kaos_llm_core.programs.call as call_mod

        call_count = {"n": 0}

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _tool_call_response("lookup", {"q": "life"})
            return _text_response(json.dumps({"answer": "42"}))

        client = FunctionClient(model="function-test", function=handler)
        original = call_mod.create_client
        call_mod.create_client = lambda *a, **kw: client  # ty: ignore[invalid-assignment]

        try:
            tool = KaosLLMCoreReActTool()
            result = await tool.execute(
                {
                    "model": "function-test",
                    "instruction": "Answer the question.",
                    "input_text": "What is the meaning of life?",
                    "output_field": "answer",
                    "tool_specs": [
                        {
                            "name": "lookup",
                            "description": "Look up a fact.",
                            "parameters": {
                                "type": "object",
                                "properties": {"q": {"type": "string"}},
                                "required": ["q"],
                                "additionalProperties": False,
                            },
                        }
                    ],
                }
            )
            assert not result.isError
            structured = result.require_structured()
            assert structured["iterations_used"] == 2
            assert structured["stop_reason"] == "TERMINATED"
            assert structured["output"] == "42"
            # First iteration executed the dry-run tool
            first = structured["trajectory"][0]
            assert len(first["tool_calls"]) == 1
            assert first["tool_calls"][0]["name"] == "lookup"
            assert len(first["tool_results"]) == 1
            obs = first["tool_results"][0]
            assert obs["tool_name"] == "lookup"
            assert obs["is_error"] is False
            # The dry-run envelope is returned as the tool result.
            assert isinstance(obs["result"], dict)
            assert obs["result"].get("dry_run") is True
            assert obs["result"].get("tool") == "lookup"
            assert obs["result"].get("arguments") == {"q": "life"}
        finally:
            call_mod.create_client = original

    async def test_execute_invalid_tool_spec_shape(self) -> None:
        """A malformed tool spec returns an error with recovery guidance."""
        tool = KaosLLMCoreReActTool()
        result = await tool.execute(
            {
                "model": "function-test",
                "instruction": "Answer.",
                "input_text": "hi",
                "output_field": "answer",
                "tool_specs": [{"description": "no name key"}],
            }
        )
        assert result.isError
        assert "name" in (result.text or "")


# ---------------------------------------------------------------------------
# KaosLLMCoreRefineTool tests
# ---------------------------------------------------------------------------


class TestRefineTool:
    def test_metadata(self) -> None:
        tool = KaosLLMCoreRefineTool()
        meta = tool.metadata
        assert meta.name == "kaos-llm-core-refine"
        assert meta.annotations is not None
        assert meta.annotations.readOnlyHint is True
        assert meta.annotations.openWorldHint is True
        assert meta.category == "integration"
        assert meta.capability == "analyze"
        param_names = {p.name for p in meta.input_schema}
        assert "producer_model" in param_names
        assert "judge_model" in param_names
        assert "producer_instruction" in param_names
        assert "judge_instruction" in param_names
        assert "input_text" in param_names
        assert "output_field" in param_names
        assert "max_iterations" in param_names
        assert "min_score" in param_names
        required = {p.name for p in meta.input_schema if p.required}
        assert required == {
            "producer_model",
            "judge_model",
            "producer_instruction",
            "judge_instruction",
            "input_text",
            "output_field",
        }

    async def test_execute_missing_required_params(self) -> None:
        """Refine should error when required params are missing."""
        tool = KaosLLMCoreRefineTool()
        result = await tool.execute({"producer_model": "function-test"})
        assert result.isError
        assert "Required" in (result.text or "")

    async def test_execute_quality_met_on_first_iteration(self) -> None:
        """Refine stops at iteration 1 when the judge scores >= min_score."""
        import kaos_llm_core.programs.call as call_mod

        call_count = {"producer": 0, "judge": 0}

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            # The Refine loop alternates: producer call (expects {answer}),
            # then judge call (expects {quality_score, reasoning}). We route
            # based on the presence of 'criteria' in the rendered messages,
            # which the judge always includes.
            text_blob = " ".join(
                str(m.get("content", "")) if isinstance(m, dict) else str(m) for m in messages
            )
            if "Evaluation criteria" in text_blob or "quality_score" in text_blob:
                call_count["judge"] += 1
                return _text_response(
                    json.dumps({"quality_score": 0.95, "reasoning": "Looks great"})
                )
            call_count["producer"] += 1
            return _text_response(json.dumps({"answer": "great answer"}))

        client = FunctionClient(model="function-test", function=handler)
        original = call_mod.create_client
        call_mod.create_client = lambda *a, **kw: client  # ty: ignore[invalid-assignment]

        try:
            tool = KaosLLMCoreRefineTool()
            result = await tool.execute(
                {
                    "producer_model": "function-test",
                    "judge_model": "function-test",
                    "producer_instruction": "Write a great answer.",
                    "judge_instruction": "Is the answer great?",
                    "input_text": "Say something great.",
                    "output_field": "answer",
                    "min_score": 0.8,
                    "max_iterations": 3,
                }
            )
            assert not result.isError
            structured = result.require_structured()
            assert structured["iterations"] == 1
            assert structured["stop_reason"] == "QUALITY_MET"
            assert structured["final_score"] >= 0.8
            assert structured["output"] == "great answer"
            assert len(structured["history"]) == 1
        finally:
            call_mod.create_client = original

    async def test_execute_two_iterations_then_quality_met(self) -> None:
        """Low quality on iteration 1 -> high quality on iteration 2."""
        import kaos_llm_core.programs.call as call_mod

        judge_count = {"n": 0}
        producer_count = {"n": 0}

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            text_blob = " ".join(
                str(m.get("content", "")) if isinstance(m, dict) else str(m) for m in messages
            )
            if "Evaluation criteria" in text_blob or "quality_score" in text_blob:
                judge_count["n"] += 1
                if judge_count["n"] == 1:
                    return _text_response(
                        json.dumps({"quality_score": 0.3, "reasoning": "Too short; add detail"})
                    )
                return _text_response(
                    json.dumps({"quality_score": 0.9, "reasoning": "Much better"})
                )
            producer_count["n"] += 1
            if producer_count["n"] == 1:
                return _text_response(json.dumps({"answer": "short"}))
            return _text_response(json.dumps({"answer": "a much longer and better answer"}))

        client = FunctionClient(model="function-test", function=handler)
        original = call_mod.create_client
        call_mod.create_client = lambda *a, **kw: client  # ty: ignore[invalid-assignment]

        try:
            tool = KaosLLMCoreRefineTool()
            result = await tool.execute(
                {
                    "producer_model": "function-test",
                    "judge_model": "function-test",
                    "producer_instruction": "Write an answer.",
                    "judge_instruction": "Is the answer good?",
                    "input_text": "Say something useful.",
                    "output_field": "answer",
                    "min_score": 0.8,
                    "max_iterations": 3,
                }
            )
            assert not result.isError
            structured = result.require_structured()
            assert structured["iterations"] == 2
            assert structured["stop_reason"] == "QUALITY_MET"
            assert structured["final_score"] >= 0.8
            assert len(structured["history"]) == 2
            assert structured["history"][0]["score"] < 0.8
            assert structured["history"][1]["score"] >= 0.8
        finally:
            call_mod.create_client = original


# ---------------------------------------------------------------------------
# KaosLLMCoreBestOfNTool tests
# ---------------------------------------------------------------------------


class TestBestOfNTool:
    def test_metadata(self) -> None:
        tool = KaosLLMCoreBestOfNTool()
        meta = tool.metadata
        assert meta.name == "kaos-llm-core-best-of-n"
        assert meta.annotations is not None
        assert meta.annotations.readOnlyHint is True
        assert meta.annotations.openWorldHint is True
        assert meta.category == "integration"
        assert meta.capability == "analyze"
        param_names = {p.name for p in meta.input_schema}
        assert "model" in param_names
        assert "instruction" in param_names
        assert "input_text" in param_names
        assert "output_field" in param_names
        assert "n" in param_names
        assert "judge_model" in param_names
        assert "judge_criteria" in param_names
        required = {p.name for p in meta.input_schema if p.required}
        assert required == {"model", "instruction", "input_text", "output_field", "n"}

    async def test_execute_missing_required_params(self) -> None:
        """BestOfN should error when required params are missing."""
        tool = KaosLLMCoreBestOfNTool()
        result = await tool.execute({"model": "function-test"})
        assert result.isError
        assert "Required" in (result.text or "")

    async def test_execute_n_less_than_2_returns_error(self) -> None:
        """n < 2 should return an explanatory error."""
        tool = KaosLLMCoreBestOfNTool()
        result = await tool.execute(
            {
                "model": "function-test",
                "instruction": "Answer.",
                "input_text": "hi",
                "output_field": "answer",
                "n": 1,
            }
        )
        assert result.isError
        assert "n must be >= 2" in (result.text or "") or "n=1" in (result.text or "")

    async def test_execute_non_integer_n_returns_error(self) -> None:
        """n that is not an integer returns an error."""
        tool = KaosLLMCoreBestOfNTool()
        result = await tool.execute(
            {
                "model": "function-test",
                "instruction": "Answer.",
                "input_text": "hi",
                "output_field": "answer",
                "n": "banana",
            }
        )
        assert result.isError

    async def test_execute_judge_without_criteria_returns_error(self) -> None:
        """judge_model without judge_criteria should fail with guidance."""
        tool = KaosLLMCoreBestOfNTool()
        result = await tool.execute(
            {
                "model": "function-test",
                "instruction": "Answer.",
                "input_text": "hi",
                "output_field": "answer",
                "n": 3,
                "judge_model": "function-test",
            }
        )
        assert result.isError
        assert "judge_criteria" in (result.text or "")

    async def test_execute_three_samples_with_length_selector(self) -> None:
        """n=3 with no judge_model: uses dummy length selector, returns 3 candidates."""
        import kaos_llm_core.programs.call as call_mod

        call_count = {"n": 0}

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            call_count["n"] += 1
            # Produce varied-length outputs so the length selector picks a winner.
            outputs = ["short", "medium length", "the longest answer of them all"]
            idx = (call_count["n"] - 1) % 3
            return _text_response(json.dumps({"answer": outputs[idx]}))

        client = FunctionClient(model="function-test", function=handler)
        original = call_mod.create_client
        call_mod.create_client = lambda *a, **kw: client  # ty: ignore[invalid-assignment]

        try:
            tool = KaosLLMCoreBestOfNTool()
            result = await tool.execute(
                {
                    "model": "function-test",
                    "instruction": "Answer.",
                    "input_text": "Say something.",
                    "output_field": "answer",
                    "n": 3,
                }
            )
            assert not result.isError
            structured = result.require_structured()
            assert structured["selection_method"] == "length"
            assert len(structured["candidates"]) == 3
            assert len(structured["scores"]) == 3
            assert 0 <= structured["selected_index"] < 3
        finally:
            call_mod.create_client = original


# ---------------------------------------------------------------------------
# KaosLLMCoreSaveLoadTool tests
# ---------------------------------------------------------------------------


def _save_load_context(root: Path) -> KaosContext:
    """Build a KaosContext whose VFS is rooted at ``root`` so the
    security-hardened SaveLoadTool can resolve VFS-relative paths."""
    runtime = KaosRuntime()
    runtime.vfs = VirtualFileSystem(VFSConfig(disk_base_path=root))
    return KaosContext(session_id="test-save-load", runtime=runtime)


def _save_load_write(ctx: KaosContext, rel_name: str, content: str) -> str:
    """Write ``content`` to the disk location ``rel_name`` resolves to under
    the SaveLoadTool's VFS context_id, creating parents. Returns the same
    ``rel_name`` so callers can pass it back into the tool."""
    assert ctx.runtime is not None
    disk = ctx.runtime.vfs.resolve_disk_path(rel_name, context_id="kaos-llm-core-save-load")
    assert disk is not None
    p = Path(disk)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return rel_name


class TestSaveLoadTool:
    def test_metadata(self) -> None:
        tool = KaosLLMCoreSaveLoadTool()
        meta = tool.metadata
        assert meta.name == "kaos-llm-core-save-load"
        assert meta.annotations is not None
        assert meta.annotations.readOnlyHint is True
        assert meta.annotations.openWorldHint is False
        assert meta.category == "utility"
        assert meta.capability == "transform"
        param_names = {p.name for p in meta.input_schema}
        assert "mode" in param_names
        assert "path" in param_names
        assert "program_state" in param_names
        required = {p.name for p in meta.input_schema if p.required}
        assert required == {"mode"}

    async def test_execute_save_mode_returns_error(self) -> None:
        """'save' mode is unsupported and returns an explanatory error."""
        tool = KaosLLMCoreSaveLoadTool()
        result = await tool.execute({"mode": "save", "path": "/tmp/ignored.json"})
        assert result.isError
        assert "Python API" in (result.text or "")

    async def test_execute_invalid_mode(self) -> None:
        """Unknown mode returns error."""
        tool = KaosLLMCoreSaveLoadTool()
        result = await tool.execute({"mode": "bogus"})
        assert result.isError

    async def test_execute_load_v2_envelope_from_file(self, tmp_path: Path) -> None:
        """Valid v2 envelope on disk loads and returns metadata."""
        tool = KaosLLMCoreSaveLoadTool()
        envelope = {
            "program": "TestProgram",
            "version": 2,
            "state": {
                "extract": {
                    "instructions": "Extract entities.",
                    "examples": [],
                    "hyperparameters": {"temperature": 0.0},
                    "codec": "JSONCodec",
                    "model": "anthropic:claude-haiku-4-5",
                },
                "classify": {
                    "instructions": "Classify.",
                    "examples": [],
                    "hyperparameters": {},
                    "codec": "JSONCodec",
                    "model": "openai:gpt-5.4-nano",
                },
            },
        }
        # Files must live INSIDE the runtime VFS root (under the
        # save-load tool's context_id namespace) so the security-hardened
        # save/load tool can resolve a VFS-relative path.
        ctx = _save_load_context(tmp_path)
        rel_name = _save_load_write(ctx, "envelope-v2.json", json.dumps(envelope))

        result = await tool.execute({"mode": "load", "path": rel_name}, ctx)
        assert not result.isError
        structured = result.require_structured()
        assert structured["valid"] is True
        assert structured["program"] == "TestProgram"
        assert structured["version"] == 2
        assert structured["num_calls"] == 2
        assert set(structured["state_keys"]) == {"extract", "classify"}
        assert "deprecated" not in structured

    async def test_execute_load_v1_envelope_marks_deprecated(self, tmp_path: Path) -> None:
        """v1 envelopes load but are flagged deprecated=true."""
        tool = KaosLLMCoreSaveLoadTool()
        envelope = {
            "program": "LegacyProgram",
            "version": 1,
            "state": {
                "only_call": {
                    "instructions": "Legacy.",
                    "examples": [],
                },
            },
        }
        ctx = _save_load_context(tmp_path)
        rel_name = _save_load_write(ctx, "envelope-v1.json", json.dumps(envelope))

        result = await tool.execute({"mode": "load", "path": rel_name}, ctx)
        assert not result.isError
        structured = result.require_structured()
        assert structured["valid"] is True
        assert structured["version"] == 1
        assert structured["deprecated"] is True
        assert structured["num_calls"] == 1

    async def test_execute_load_missing_state_key_returns_error(self, tmp_path: Path) -> None:
        """Envelope missing 'state' should return error with guidance."""
        tool = KaosLLMCoreSaveLoadTool()
        bad_envelope = {"program": "NoState", "version": 2}
        ctx = _save_load_context(tmp_path)
        rel_name = _save_load_write(ctx, "no-state.json", json.dumps(bad_envelope))

        result = await tool.execute({"mode": "load", "path": rel_name}, ctx)
        assert result.isError
        assert "state" in (result.text or "")

    async def test_execute_load_from_inline_program_state(self) -> None:
        """Inline program_state dict can be validated without a path."""
        tool = KaosLLMCoreSaveLoadTool()
        result = await tool.execute(
            {
                "mode": "load",
                "program_state": {
                    "program": "InlineProgram",
                    "version": 2,
                    "state": {"a": {"instructions": "..."}},
                },
            }
        )
        assert not result.isError
        structured = result.require_structured()
        assert structured["valid"] is True
        assert structured["program"] == "InlineProgram"
        assert structured["num_calls"] == 1

    async def test_execute_load_nonexistent_file(self, tmp_path: Path) -> None:
        """Nonexistent file path returns error.

        After the security hardening for KLLC-03, absolute paths are
        always rejected; this test now uses a VFS-relative name that
        resolves to a path that doesn't exist on disk."""
        tool = KaosLLMCoreSaveLoadTool()
        ctx = _save_load_context(tmp_path)
        result = await tool.execute({"mode": "load", "path": "does-not-exist.json"}, ctx)
        assert result.isError
        assert "not found" in (result.text or "").lower()

    async def test_execute_load_absolute_path_rejected(self) -> None:
        """KLLC-03 — absolute paths are rejected outright (was: returned 'not found').

        The fixture path must be absolute on the test platform. On
        POSIX ``/nonexistent/...`` is absolute; on Windows a leading
        ``/`` without a drive letter is drive-relative, so
        ``Path(...).is_absolute()`` returns ``False`` there and the
        guard never fires. Use ``Path.cwd().anchor`` to derive the
        platform's root (``/`` on POSIX, ``C:\\`` on Windows) so the
        fixture is always absolute.
        """
        import sys

        if sys.platform == "win32":
            abs_path = "C:/nonexistent/path/does_not_exist.json"
        else:
            abs_path = "/nonexistent/path/does_not_exist.json"
        tool = KaosLLMCoreSaveLoadTool()
        result = await tool.execute({"mode": "load", "path": abs_path})
        assert result.isError
        assert "absolute" in (result.text or "").lower()

    async def test_execute_load_corrupted_json(self, tmp_path: Path) -> None:
        """Corrupted JSON file returns error."""
        tool = KaosLLMCoreSaveLoadTool()
        ctx = _save_load_context(tmp_path)
        rel_name = _save_load_write(ctx, "corrupted.json", "{ not valid json at all")
        result = await tool.execute({"mode": "load", "path": rel_name}, ctx)
        assert result.isError
        assert "JSON" in (result.text or "")

    async def test_execute_round_trip_success(self, tmp_path: Path) -> None:
        """round-trip should write a temp file that can be read back."""
        tool = KaosLLMCoreSaveLoadTool()
        envelope = {
            "program": "RTProgram",
            "version": 2,
            "state": {
                "a": {
                    "instructions": "Do thing A.",
                    "examples": [],
                    "hyperparameters": {},
                    "codec": "JSONCodec",
                    "model": "anthropic:claude-haiku-4-5",
                },
            },
        }
        ctx = _save_load_context(tmp_path)
        rel_name = _save_load_write(ctx, "rt-original.json", json.dumps(envelope))

        round_trip_path: str | None = None
        try:
            result = await tool.execute({"mode": "round-trip", "path": rel_name}, ctx)
            assert not result.isError
            structured = result.require_structured()
            assert structured["success"] is True
            # Original path is echoed as the caller-supplied VFS-relative name.
            assert structured["original_path"] == rel_name
            assert structured["size_bytes"] > 0
            round_trip_path = structured["round_trip_path"]
            assert round_trip_path is not None
            loaded = json.loads(Path(round_trip_path).read_text(encoding="utf-8"))
            assert loaded["program"] == "RTProgram"
            assert loaded["version"] == 2
            assert "a" in loaded["state"]
        finally:
            if round_trip_path is not None:
                Path(round_trip_path).unlink(missing_ok=True)

    async def test_execute_round_trip_missing_path(self) -> None:
        """round-trip with no path returns error."""
        tool = KaosLLMCoreSaveLoadTool()
        result = await tool.execute({"mode": "round-trip"})
        assert result.isError
        assert "path" in (result.text or "").lower()

    async def test_execute_round_trip_malformed_envelope(self, tmp_path: Path) -> None:
        """round-trip on a JSON file missing 'state' returns error."""
        tool = KaosLLMCoreSaveLoadTool()
        ctx = _save_load_context(tmp_path)
        rel_name = _save_load_write(
            ctx, "rt-malformed.json", json.dumps({"program": "X", "version": 2})
        )
        result = await tool.execute({"mode": "round-trip", "path": rel_name}, ctx)
        assert result.isError
        assert "state" in (result.text or "")


# ---------------------------------------------------------------------------
# register_llm_core_tools registration count
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_registration_count_is_30(self) -> None:
        """register_llm_core_tools should register exactly 30 tools after #91.

        Tool count history:
        7 (pre-Phase-5) → 11 (5) → 15 (6) → 17 (7) → 18 (15.1) → 22 (15.3)
        → 23 (17.1) → 29 (WS-TR.PR-6f.7: +6 alpha extractors) →
        30 (#91: +kaos-llm-core-program-of-thought dedicated wrapper).
        """
        from kaos_core import KaosRuntime

        from kaos_llm_core.integrations.mcp.registration import register_llm_core_tools

        runtime = KaosRuntime()
        n = register_llm_core_tools(runtime)
        assert n == 30
        registered = set(runtime.tools.list_tools())
        assert "kaos-llm-core-react" in registered
        assert "kaos-llm-core-refine" in registered
        assert "kaos-llm-core-best-of-n" in registered
        assert "kaos-llm-core-save-load" in registered
        assert "kaos-llm-core-optimize-codec" in registered
        assert "kaos-llm-core-optimize-model" in registered
        assert "kaos-llm-core-mipro-v2" in registered
        assert "kaos-llm-core-pareto" in registered
        assert "kaos-llm-core-recipe-tune" in registered
        assert "kaos-llm-core-metric" in registered
        assert "kaos-llm-core-analyze-trial" in registered
        assert "kaos-llm-core-program-execute" in registered
        assert "kaos-llm-core-program-of-thought" in registered
        assert "kaos-llm-core-batch-create" in registered
        assert "kaos-llm-core-batch-run" in registered
        assert "kaos-llm-core-batch-status" in registered
        assert "kaos-llm-core-batch-results" in registered


# ---------------------------------------------------------------------------
# Phase 6 MCP tool tests
# ---------------------------------------------------------------------------


class TestOptimizeCodecTool:
    def test_metadata(self) -> None:
        tool = KaosLLMCoreOptimizeCodecTool()
        meta = tool.metadata
        assert meta.name == "kaos-llm-core-optimize-codec"
        assert meta.annotations is not None
        assert meta.annotations.readOnlyHint is True
        assert meta.annotations.openWorldHint is True
        assert meta.annotations.idempotentHint is False
        names = {p.name for p in meta.input_schema}
        assert {"model", "instruction", "examples"}.issubset(names)

    async def test_unknown_metric_error(self) -> None:
        tool = KaosLLMCoreOptimizeCodecTool()
        result = await tool.execute(
            {
                "model": "function-test",
                "instruction": "Classify.",
                "examples": [{"input": "a", "expected_output": "b"}],
                "metric_name": "nonsense",
            }
        )
        assert result.isError

    async def test_bad_codec_name(self) -> None:
        tool = KaosLLMCoreOptimizeCodecTool()
        result = await tool.execute(
            {
                "model": "function-test",
                "instruction": "Classify.",
                "examples": [{"input": "a", "expected_output": "b"}],
                "codecs": ["NotARealCodec"],
            }
        )
        assert result.isError

    async def test_happy_path(self) -> None:
        import kaos_llm_core.programs.call as call_mod

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"answer": "yes"})

        client = FunctionClient(function=fn)
        original = call_mod.create_client
        call_mod.create_client = lambda *a, **kw: client  # ty: ignore[invalid-assignment]
        try:
            tool = KaosLLMCoreOptimizeCodecTool()
            result = await tool.execute(
                {
                    "model": "function-test",
                    "instruction": "Answer.",
                    "examples": [{"input": "q", "expected_output": "yes"}],
                    "codecs": ["JSONCodec"],
                }
            )
            assert not result.isError
            output = result.require_structured()
            assert output["best_codec"] == "JSONCodec"
        finally:
            call_mod.create_client = original


class TestOptimizeModelTool:
    def test_metadata(self) -> None:
        tool = KaosLLMCoreOptimizeModelTool()
        meta = tool.metadata
        assert meta.name == "kaos-llm-core-optimize-model"
        assert meta.annotations is not None
        assert meta.annotations.openWorldHint is True

    async def test_empty_models_error(self) -> None:
        tool = KaosLLMCoreOptimizeModelTool()
        result = await tool.execute(
            {
                "models": [],
                "instruction": "x",
                "examples": [{"input": "a", "expected_output": "b"}],
            }
        )
        assert result.isError

    async def test_happy_path(self) -> None:
        import kaos_llm_core.programs.call as call_mod

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"answer": "yes"})

        client = FunctionClient(function=fn)
        original = call_mod.create_client
        call_mod.create_client = lambda *a, **kw: client  # ty: ignore[invalid-assignment]
        try:
            tool = KaosLLMCoreOptimizeModelTool()
            result = await tool.execute(
                {
                    "models": ["function-test"],
                    "instruction": "Answer.",
                    "examples": [{"input": "q", "expected_output": "yes"}],
                    "min_score": 0.5,
                }
            )
            assert not result.isError
            output = result.require_structured()
            assert output["best_model"] == "function-test"
        finally:
            call_mod.create_client = original


class TestParetoTool:
    def test_metadata(self) -> None:
        tool = KaosLLMCoreParetoTool()
        meta = tool.metadata
        assert meta.name == "kaos-llm-core-pareto"
        assert meta.capability == "analyze"

    async def test_empty_models_error(self) -> None:
        tool = KaosLLMCoreParetoTool()
        result = await tool.execute(
            {
                "models": [],
                "instruction": "x",
                "examples": [{"input": "a", "expected_output": "b"}],
            }
        )
        assert result.isError

    async def test_happy_path(self) -> None:
        import kaos_llm_core.programs.call as call_mod

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"answer": "yes"})

        client = FunctionClient(function=fn)
        original = call_mod.create_client
        call_mod.create_client = lambda *a, **kw: client  # ty: ignore[invalid-assignment]
        try:
            tool = KaosLLMCoreParetoTool()
            result = await tool.execute(
                {
                    "models": ["function-test"],
                    "instruction": "Answer.",
                    "examples": [{"input": "q", "expected_output": "yes"}],
                }
            )
            assert not result.isError
            output = result.require_structured()
            assert "frontier" in output
            assert "all_trials" in output
        finally:
            call_mod.create_client = original


class TestRecipeTuneTool:
    def test_metadata(self) -> None:
        tool = KaosLLMCoreRecipeTuneTool()
        meta = tool.metadata
        assert meta.name == "kaos-llm-core-recipe-tune"
        names = {p.name for p in meta.input_schema}
        assert "recipe" in names

    async def test_missing_model_for_friendly_prompt(self) -> None:
        tool = KaosLLMCoreRecipeTuneTool()
        result = await tool.execute(
            {
                "recipe": "friendly_prompt",
                "instruction": "x",
                "examples": [{"input": "a", "expected_output": "b"}],
            }
        )
        assert result.isError

    async def test_missing_models_for_cost_aware(self) -> None:
        tool = KaosLLMCoreRecipeTuneTool()
        result = await tool.execute(
            {
                "recipe": "cost_aware_model",
                "instruction": "x",
                "examples": [{"input": "a", "expected_output": "b"}],
            }
        )
        assert result.isError

    async def test_cost_aware_happy_path(self) -> None:
        import kaos_llm_core.programs.call as call_mod

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"answer": "yes"})

        client = FunctionClient(function=fn)
        original = call_mod.create_client
        call_mod.create_client = lambda *a, **kw: client  # ty: ignore[invalid-assignment]
        try:
            tool = KaosLLMCoreRecipeTuneTool()
            result = await tool.execute(
                {
                    "recipe": "cost_aware_model",
                    "instruction": "Answer.",
                    "examples": [{"input": "q", "expected_output": "yes"}],
                    "models": ["function-test"],
                    "min_score": 0.5,
                }
            )
            assert not result.isError
            output = result.require_structured()
            assert output["best_model"] == "function-test"
        finally:
            call_mod.create_client = original


# ---------------------------------------------------------------------------
# Phase 7 MCP tool tests
# ---------------------------------------------------------------------------


class TestMetricTool:
    def test_metadata(self) -> None:
        tool = KaosLLMCoreMetricTool()
        meta = tool.metadata
        assert meta.name == "kaos-llm-core-metric"
        assert meta.annotations is not None
        assert meta.annotations.readOnlyHint is True
        assert meta.annotations.openWorldHint is True
        names = {p.name for p in meta.input_schema}
        assert {"metric_name", "prediction", "gold"}.issubset(names)

    async def test_exact_match(self) -> None:
        tool = KaosLLMCoreMetricTool()
        result = await tool.execute(
            {"metric_name": "exact_match", "prediction": "hello", "gold": "hello"}
        )
        assert not result.isError
        output = result.require_structured()
        assert output["score"] == 1.0
        assert output["metric"] == "exact_match"

    async def test_case_insensitive(self) -> None:
        tool = KaosLLMCoreMetricTool()
        result = await tool.execute(
            {"metric_name": "case_insensitive_match", "prediction": "FOO", "gold": "foo"}
        )
        assert not result.isError
        assert result.require_structured()["score"] == 1.0

    async def test_numeric_ratio(self) -> None:
        tool = KaosLLMCoreMetricTool()
        result = await tool.execute(
            {"metric_name": "numeric_ratio", "prediction": "5", "gold": "5"}
        )
        assert not result.isError
        assert result.require_structured()["score"] == 1.0

    async def test_unknown_metric_error(self) -> None:
        tool = KaosLLMCoreMetricTool()
        result = await tool.execute({"metric_name": "nonsense", "prediction": "p", "gold": "g"})
        assert result.isError

    async def test_missing_inputs(self) -> None:
        tool = KaosLLMCoreMetricTool()
        result = await tool.execute({"metric_name": "exact_match"})
        assert result.isError

    async def test_llm_judge_requires_model(self) -> None:
        tool = KaosLLMCoreMetricTool()
        result = await tool.execute({"metric_name": "llm_judge", "prediction": "p", "gold": "g"})
        assert result.isError
        assert "judge_model" in (result.text or "")


class TestAnalyzeTrialTool:
    def test_metadata(self) -> None:
        tool = KaosLLMCoreAnalyzeTrialTool()
        meta = tool.metadata
        assert meta.name == "kaos-llm-core-analyze-trial"
        assert meta.annotations is not None
        assert meta.annotations.readOnlyHint is True
        assert meta.annotations.openWorldHint is False
        assert meta.annotations.idempotentHint is True
        names = {p.name for p in meta.input_schema}
        assert "mutation_log_path" in names

    async def test_happy_path(self) -> None:
        from kaos_llm_core.optimization.mutations import Mutation, MutationLog

        tool = KaosLLMCoreAnalyzeTrialTool()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.jsonl"
            log = MutationLog(path=path)
            log.record(
                Mutation(
                    strategy="bootstrap",
                    mutation_type="add_example",
                    call_name="DemoCall",
                    before={"examples": []},
                    after={"examples": [{"input": "x", "output": "y"}]},
                    metric_before=0.5,
                    metric_after=0.8,
                    accepted=True,
                    cost_usd=0.001,
                    tokens_used=100,
                )
            )
            result = await tool.execute({"mutation_log_path": str(path)})
            assert not result.isError
            output = result.require_structured()
            assert output["summary"]["total_trials"] == 1
            assert len(output["trial_cards"]) == 1
            assert output["trial_cards"][0]["strategy"] == "bootstrap"
            assert len(output["strategy_contributions"]) == 1

    async def test_missing_path(self) -> None:
        tool = KaosLLMCoreAnalyzeTrialTool()
        result = await tool.execute({})
        assert result.isError

    async def test_nonexistent_file(self) -> None:
        tool = KaosLLMCoreAnalyzeTrialTool()
        result = await tool.execute({"mutation_log_path": "/tmp/does-not-exist-kaos-xyz.jsonl"})
        assert result.isError


# ---------------------------------------------------------------------------
# KaosLLMCoreProgramOfThoughtTool tests
# ---------------------------------------------------------------------------


class TestProgramOfThoughtTool:
    def test_metadata(self) -> None:
        tool = KaosLLMCoreProgramOfThoughtTool()
        meta = tool.metadata
        assert meta.name == "kaos-llm-core-program-of-thought"
        assert meta.annotations is not None
        # Code execution is not read-only — it spawns a subprocess and mutates a tempdir.
        assert meta.annotations.readOnlyHint is False
        assert meta.annotations.openWorldHint is True
        assert meta.category == "integration"
        assert meta.capability == "analyze"
        param_names = {p.name for p in meta.input_schema}
        required = {p.name for p in meta.input_schema if p.required}
        assert param_names >= {
            "producer_model",
            "instruction",
            "input_text",
            "output_field",
            "allow_code_execution",
            "interpreter_model",
            "timeout_s",
            "memory_mb",
            "cpu_seconds",
        }
        assert required == {
            "producer_model",
            "instruction",
            "input_text",
            "output_field",
            "allow_code_execution",
        }

    async def test_missing_required_params(self) -> None:
        tool = KaosLLMCoreProgramOfThoughtTool()
        result = await tool.execute({"producer_model": "function-test"})
        assert result.isError
        assert "Required" in (result.text or "")

    async def test_refuses_when_allow_code_execution_false(self) -> None:
        """The wrapper short-circuits before any LLM call when opt-in is false."""
        tool = KaosLLMCoreProgramOfThoughtTool()
        result = await tool.execute(
            {
                "producer_model": "function-test",
                "instruction": "Sum the numbers.",
                "input_text": "1 + 2",
                "output_field": "answer",
                "allow_code_execution": False,
            }
        )
        assert result.isError
        assert "allow_code_execution" in (result.text or "")

    async def test_refuses_when_allow_code_execution_not_bool(self) -> None:
        tool = KaosLLMCoreProgramOfThoughtTool()
        result = await tool.execute(
            {
                "producer_model": "function-test",
                "instruction": "Sum.",
                "input_text": "1 + 2",
                "output_field": "answer",
                "allow_code_execution": "yes-please",
            }
        )
        assert result.isError
        assert "boolean" in (result.text or "")

    async def test_refuses_when_runtime_gate_off(self) -> None:
        """Caller opt-in alone is not enough — the runtime setting must also be on."""
        import os

        # Make sure the env var is NOT set for this test.
        prev = os.environ.pop("KAOS_LLM_CORE_ALLOW_CODE_EXECUTION_VIA_MCP", None)
        try:
            tool = KaosLLMCoreProgramOfThoughtTool()
            result = await tool.execute(
                {
                    "producer_model": "function-test",
                    "instruction": "Sum.",
                    "input_text": "1 + 2",
                    "output_field": "answer",
                    "allow_code_execution": True,
                }
            )
            assert result.isError
            assert "ALLOW_CODE_EXECUTION_VIA_MCP" in (result.text or "")
        finally:
            if prev is not None:
                os.environ["KAOS_LLM_CORE_ALLOW_CODE_EXECUTION_VIA_MCP"] = prev

    async def test_executes_when_both_gates_open(self) -> None:
        """With both gates on and a mocked LLM, the tool runs the program end-to-end."""
        import os

        import kaos_llm_core.programs.call as call_mod

        os.environ["KAOS_LLM_CORE_ALLOW_CODE_EXECUTION_VIA_MCP"] = "1"

        call_count = {"n": 0}

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            call_count["n"] += 1
            if call_count["n"] == 1:
                # Code-writer phase: emit a trivial program.
                return _json_response({"code": "print(3)"})
            # Interpreter phase: parse stdout into the answer field.
            return _json_response({"answer": "3"})

        client = FunctionClient(model="function-test", function=handler)
        original = call_mod.create_client
        call_mod.create_client = lambda *a, **kw: client  # ty: ignore[invalid-assignment]

        try:
            tool = KaosLLMCoreProgramOfThoughtTool()
            result = await tool.execute(
                {
                    "producer_model": "function-test",
                    "instruction": "Write a program that prints 3.",
                    "input_text": "Compute 1 + 2.",
                    "output_field": "answer",
                    "allow_code_execution": True,
                    "timeout_s": 5.0,
                }
            )
            assert not result.isError, result.text
            structured = result.require_structured()
            assert structured["output"] == "3"
            assert structured["code"] == "print(3)"
            assert structured["stdout"].strip() == "3"
            assert structured["return_code"] == 0
            assert structured["timed_out"] is False
        finally:
            call_mod.create_client = original
            os.environ.pop("KAOS_LLM_CORE_ALLOW_CODE_EXECUTION_VIA_MCP", None)
