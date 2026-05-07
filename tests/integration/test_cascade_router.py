"""Live integration test for CascadeRouter across providers.

Tests cascade escalation from cheap to expensive models via Call(router=...).
Requires ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import pytest

from kaos_llm_core.programs.call import Call
from kaos_llm_core.router.cascade import CascadeRouter
from kaos_llm_core.signatures import InputField, OutputField, Signature

from .conftest import requires_anthropic


class ClassifyRisk(Signature):
    """Classify the risk level of the described scenario."""

    text: str = InputField(description="Scenario description")
    level: str = OutputField(description="Risk level: low, medium, or high")
    confidence: float = OutputField(description="Classification confidence between 0 and 1")


@requires_anthropic
class TestCascadeRouter:
    @pytest.mark.integration
    async def test_cascade_accepts_first_model(self) -> None:
        """When the cheap model produces a high-confidence result, no escalation."""
        router = CascadeRouter(
            models=["anthropic:claude-haiku-4-5", "anthropic:claude-sonnet-4-6"],
            escalation_check=lambda r: r.confidence >= 0.5,
        )
        call = Call(ClassifyRisk, router=router)
        result = await call(text="A company is dumping toxic waste into a river near a school.")

        assert result.level in ("low", "medium", "high")
        assert result.confidence >= 0.5
        assert router.model_used == "anthropic:claude-haiku-4-5"
        assert len(router.last_traces) == 1

    @pytest.mark.integration
    async def test_cascade_escalates_on_quality_fail(self) -> None:
        """When the quality check always fails, cascade tries all models."""
        router = CascadeRouter(
            models=["anthropic:claude-haiku-4-5", "anthropic:claude-sonnet-4-6"],
            escalation_check=lambda _r: False,
        )
        call = Call(ClassifyRisk, router=router)
        result = await call(text="Minor paperwork discrepancy in quarterly filing.")

        assert len(router.last_traces) == 2
        assert router.model_used == "anthropic:claude-sonnet-4-6"
        assert result.level in ("low", "medium", "high")

    @pytest.mark.integration
    async def test_cascade_traces_have_token_counts(self) -> None:
        """Each cascade attempt should produce a trace with real token counts."""
        router = CascadeRouter(
            models=["anthropic:claude-haiku-4-5"],
            escalation_check=lambda _r: True,
        )
        call = Call(ClassifyRisk, router=router)
        await call(text="Routine compliance audit found no issues.")

        assert len(router.last_traces) == 1
        trace = router.last_traces[0]
        assert trace.input_tokens > 0
        assert trace.output_tokens > 0
        assert trace.latency_ms > 0
        assert trace.model == "anthropic:claude-haiku-4-5"
