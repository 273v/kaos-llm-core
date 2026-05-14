"""Tests for cost estimation and reporting."""

from __future__ import annotations

from kaos_llm_core.observability.cost import (
    PRICING,
    ModelPricing,
    apply_cost_estimates,
    estimate_cost,
    format_cost_report,
)
from kaos_llm_core.observability.traces import ExecutionTrace


class TestModelPricing:
    def test_estimate(self) -> None:
        pricing = ModelPricing(input_per_mtok=3.0, output_per_mtok=15.0)
        cost = pricing.estimate(input_tokens=1000, output_tokens=500)
        expected = 1000 * 3.0 / 1_000_000 + 500 * 15.0 / 1_000_000
        assert abs(cost - expected) < 1e-10

    def test_known_models_in_pricing(self) -> None:
        assert "anthropic:claude-sonnet-4-6" in PRICING
        assert "openai:gpt-4.1-nano" in PRICING
        assert "google:gemini-2.5-flash" in PRICING


class TestEstimateCost:
    def test_leaf_trace(self) -> None:
        trace = ExecutionTrace(
            model="anthropic:claude-haiku-4-5",
            input_tokens=1000,
            output_tokens=500,
        )
        cost = estimate_cost(trace)
        pricing = PRICING["anthropic:claude-haiku-4-5"]
        expected = pricing.estimate(1000, 500)
        assert abs(cost - expected) < 1e-10

    def test_unknown_model_returns_zero(self) -> None:
        trace = ExecutionTrace(model="unknown:model", input_tokens=100, output_tokens=50)
        assert estimate_cost(trace) == 0.0

    def test_hierarchical_trace(self) -> None:
        child1 = ExecutionTrace(
            model="anthropic:claude-haiku-4-5",
            input_tokens=500,
            output_tokens=200,
        )
        child2 = ExecutionTrace(
            model="anthropic:claude-sonnet-4-6",
            input_tokens=500,
            output_tokens=200,
        )
        parent = ExecutionTrace(
            model="(program)",
            children=[child1, child2],
        )
        cost = estimate_cost(parent)
        expected = PRICING["anthropic:claude-haiku-4-5"].estimate(500, 200) + PRICING[
            "anthropic:claude-sonnet-4-6"
        ].estimate(500, 200)
        assert abs(cost - expected) < 1e-10


class TestApplyCostEstimates:
    def test_fills_cost_on_leaf(self) -> None:
        trace = ExecutionTrace(
            model="openai:gpt-4.1-nano",
            input_tokens=10000,
            output_tokens=5000,
        )
        apply_cost_estimates(trace)
        expected = PRICING["openai:gpt-4.1-nano"].estimate(10000, 5000)
        assert abs(trace.cost_usd - expected) < 1e-10

    def test_fills_and_aggregates_program(self) -> None:
        child1 = ExecutionTrace(
            call_name="extract",
            model="anthropic:claude-haiku-4-5",
            input_tokens=1000,
            output_tokens=500,
        )
        child2 = ExecutionTrace(
            call_name="classify",
            model="anthropic:claude-sonnet-4-6",
            input_tokens=1000,
            output_tokens=500,
        )
        parent = ExecutionTrace(
            call_name="Pipeline",
            model="(program)",
            children=[child1, child2],
        )
        apply_cost_estimates(parent)

        # Children should have their own costs
        assert child1.cost_usd > 0
        assert child2.cost_usd > 0
        # Parent should sum children
        assert abs(parent.cost_usd - (child1.cost_usd + child2.cost_usd)) < 1e-10


class TestFormatCostReport:
    def test_report_format(self) -> None:
        child = ExecutionTrace(
            call_name="ExtractEntities",
            model="anthropic:claude-haiku-4-5",
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1500,
            latency_ms=250.0,
        )
        parent = ExecutionTrace(
            call_name="Analyzer",
            model="(program)",
            total_tokens=1500,
            latency_ms=300.0,
            children=[child],
        )
        report = format_cost_report(parent)
        assert "Analyzer" in report
        assert "ExtractEntities" in report
        assert "$" in report
        assert "tokens" in report
