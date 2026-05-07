"""Live integration test for ChainOfThought.

Tests that reasoning is populated alongside final output.
Requires ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import pytest

from kaos_llm_core.programs.chain_of_thought import ChainOfThought
from kaos_llm_core.signatures import InputField, OutputField, Signature

from .conftest import requires_anthropic


class ClassifyRisk(Signature):
    """Classify the risk level of the described legal scenario."""

    text: str = InputField(description="Legal scenario description")
    level: str = OutputField(description="Risk level: low, medium, or high")


@requires_anthropic
class TestChainOfThoughtLive:
    @pytest.mark.integration
    async def test_cot_produces_reasoning(self) -> None:
        """ChainOfThought should produce non-trivial reasoning."""
        cot = ChainOfThought(
            ClassifyRisk,
            model="anthropic:claude-haiku-4-5",
        )
        invocation = await cot.invoke(
            text="A pharmaceutical company failed to disclose adverse trial results "
            "to the FDA for 18 months while continuing to market the drug."
        )
        result = invocation.output

        # Should have reasoning
        assert result.reasoning is not None
        assert len(result.reasoning) > 20, (
            f"Reasoning too short ({len(result.reasoning)} chars): {result.reasoning}"
        )

        # Should have classification
        assert result.level in ("low", "medium", "high")
        # This scenario is clearly high risk
        assert result.level == "high", f"Expected high risk, got '{result.level}'"

        # Trace should exist
        trace = invocation.trace
        assert trace is not None
        assert trace.input_tokens > 0
