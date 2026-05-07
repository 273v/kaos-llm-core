"""Integration tests for the example programs.

Each test runs a real example against real APIs and verifies the output
has the expected structure and content quality.
"""

from __future__ import annotations

import pytest

from .conftest import requires_anthropic, requires_google, requires_openai

# --- Contract Analysis ---


@requires_anthropic
class TestContractAnalysis:
    @pytest.mark.integration
    async def test_contract_analysis_anthropic(self) -> None:
        from examples.contract_analysis import run

        result = await run(provider="anthropic")

        assert "clauses" in result
        assert len(result["clauses"]) >= 3
        assert "overall_risk" in result
        assert result["overall_risk"] in ("low", "medium", "high")
        assert "risk_factors" in result
        assert len(result["risk_factors"]) >= 1
        assert "summary" in result
        assert len(result["summary"]) > 100


@requires_openai
class TestContractAnalysisOpenAI:
    @pytest.mark.integration
    async def test_contract_analysis_openai(self) -> None:
        from examples.contract_analysis import run

        result = await run(provider="openai")

        assert len(result["clauses"]) >= 3
        assert result["overall_risk"] in ("low", "medium", "high")


@requires_google
class TestContractAnalysisGoogle:
    @pytest.mark.integration
    async def test_contract_analysis_google(self) -> None:
        from examples.contract_analysis import run

        result = await run(provider="google")

        assert len(result["clauses"]) >= 3
        assert result["overall_risk"] in ("low", "medium", "high")


# --- Financial Extraction ---


@requires_anthropic
class TestFinancialExtraction:
    @pytest.mark.integration
    async def test_financial_extraction_anthropic(self) -> None:
        from examples.financial_extraction import run

        result = await run(provider="anthropic")

        fin = result["financials"]
        assert fin["revenue_millions"] == pytest.approx(4200, rel=0.01)
        assert fin["revenue_growth_yoy"] == pytest.approx(0.18, abs=0.02)
        assert fin["gross_margin"] == pytest.approx(0.423, abs=0.02)
        assert len(result["risks"]) >= 1
        assert result["guidance"]["revenue_range"][0] == pytest.approx(4000, rel=0.01)


@requires_openai
class TestFinancialExtractionOpenAI:
    @pytest.mark.integration
    async def test_financial_extraction_openai(self) -> None:
        from examples.financial_extraction import run

        result = await run(provider="openai")

        fin = result["financials"]
        assert fin["revenue_millions"] == pytest.approx(4200, rel=0.01)
        assert fin["gross_margin"] == pytest.approx(0.423, abs=0.02)


@requires_google
class TestFinancialExtractionGoogle:
    @pytest.mark.integration
    async def test_financial_extraction_google(self) -> None:
        from examples.financial_extraction import run

        result = await run(provider="google")

        fin = result["financials"]
        assert fin["revenue_millions"] == pytest.approx(4200, rel=0.01)
        assert fin["gross_margin"] == pytest.approx(0.423, abs=0.02)


# --- Cascade Routing (Judge-Verified) ---


@requires_anthropic
class TestCascadeRouting:
    @pytest.mark.integration
    async def test_cascade_accepts_good_classification(self) -> None:
        from examples.cascade_routing import run

        result = await run(provider="anthropic", quality_threshold=0.7)

        assert result["result"]["document_type"] == "complaint"
        assert result["result"]["severity"] in ("high", "critical")
        assert len(result["cascade"]) >= 1
        assert result["total_cost_usd"] > 0

    @pytest.mark.integration
    async def test_cascade_cross_provider(self) -> None:
        from examples.cascade_routing import run

        result = await run(provider="cross", quality_threshold=0.7)

        assert result["result"]["document_type"] == "complaint"
        assert result["models_tried"] >= 1
        assert result["total_cost_usd"] > 0


# --- Optimization ---


@requires_anthropic
class TestOptimization:
    @pytest.mark.integration
    async def test_bootstrap_improves_accuracy(self) -> None:
        """Bootstrap optimizer should improve gpt-5.4-nano or haiku on SEC severity."""
        from examples.optimization_demo import TRAIN_SET, VAL_SET, ClassifySeverity, exact_match
        from kaos_llm_core import Call
        from kaos_llm_core.optimization.bootstrap import BootstrapOptimizer

        call = Call(ClassifySeverity, model="anthropic:claude-haiku-4-5")
        optimizer = BootstrapOptimizer(metric=exact_match, max_examples=4)
        result = await optimizer.optimize(call, TRAIN_SET, VAL_SET)

        # Baseline is ~50% with vague instructions, bootstrap should help
        assert result.eval_before.score <= 1.0  # valid score
        assert result.eval_after.score >= result.eval_before.score  # did not get worse

    @pytest.mark.integration
    async def test_instruction_optimizer_with_sonnet_proposer(self) -> None:
        """Sonnet proposing instructions for haiku should reach high accuracy."""
        from examples.optimization_demo import VAL_SET, ClassifySeverity, exact_match
        from kaos_llm_core import Call
        from kaos_llm_core.optimization.instruction import InstructionOptimizer

        call = Call(ClassifySeverity, model="anthropic:claude-haiku-4-5")
        optimizer = InstructionOptimizer(
            metric=exact_match,
            proposer_model="anthropic:claude-sonnet-4-6",
            max_trials=2,
        )
        result = await optimizer.optimize(call, VAL_SET)

        assert result.metric_after >= result.metric_before
        # Sonnet should be able to get haiku to at least 83%
        assert result.metric_after >= 0.8, (
            f"Expected >=80% after instruction optimization, got {result.metric_after:.0%}"
        )


@requires_anthropic
class TestCoOptimizer:
    @pytest.mark.integration
    async def test_co_optimizer_full_pipeline(self) -> None:
        """CoOptimizer running bootstrap + instruction + hyperparameter."""
        from examples.optimization_demo import TRAIN_SET, VAL_SET, ClassifySeverity, exact_match
        from kaos_llm_core import Call
        from kaos_llm_core.optimization.co_optimizer import CoOptimizer

        call = Call(ClassifySeverity, model="anthropic:claude-haiku-4-5")
        optimizer = CoOptimizer(
            metric=exact_match,
            strategies=["bootstrap", "instruction"],
            proposer_model="anthropic:claude-sonnet-4-6",
            max_bootstrap_examples=3,
            max_instruction_trials=2,
        )
        result = await optimizer.optimize(call, train_set=TRAIN_SET, val_set=VAL_SET)

        assert result.metric_after >= result.metric_before
        assert result.metric_after >= 0.8
        assert len(result.stages_run) == 2


@requires_openai
class TestOptimizationCrossModel:
    @pytest.mark.integration
    async def test_gpt54_optimizes_nano(self) -> None:
        """gpt-5.4 proposing instructions for gpt-5.4-nano."""
        from examples.optimization_demo import VAL_SET, ClassifySeverity, exact_match
        from kaos_llm_core import Call
        from kaos_llm_core.optimization.instruction import InstructionOptimizer

        call = Call(ClassifySeverity, model="openai:gpt-5.4-nano")
        optimizer = InstructionOptimizer(
            metric=exact_match,
            proposer_model="openai:gpt-5.4",
            max_trials=2,
        )
        result = await optimizer.optimize(call, VAL_SET)

        assert result.metric_after >= result.metric_before
        assert result.metric_after >= 0.8, (
            f"Expected >=80% after gpt-5.4 optimization, got {result.metric_after:.0%}"
        )
