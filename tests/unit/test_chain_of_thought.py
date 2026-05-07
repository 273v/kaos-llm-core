"""Tests for ChainOfThought — Call with automatic reasoning."""

from __future__ import annotations

import json
from typing import Any

from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart

from kaos_llm_core.programs.chain_of_thought import ChainOfThought, _inject_reasoning_field
from kaos_llm_core.signatures import InputField, OutputField, Signature
from kaos_llm_core.signatures.introspection import get_input_fields, get_output_fields


class ClassifyRisk(Signature):
    """Classify the risk level of a document."""

    text: str = InputField(description="Document text")
    level: str = OutputField(description="Risk level: low, medium, high")


class WithExistingReasoning(Signature):
    """Already has a reasoning field."""

    text: str = InputField(description="Input")
    reasoning: str = OutputField(description="Already here")
    result: str = OutputField(description="Result")


def _json_response(data: dict[str, Any]) -> ProviderResponse:
    return ProviderResponse.model_construct(
        provider="function",
        model="function-test",
        raw={},
        parts=[ContentPart.model_construct(type="text", text=json.dumps(data))],
        usage=UsageInfo.model_construct(input_tokens=50, output_tokens=25, total_tokens=75),
        stop_reason="end_turn",
        status_code=200,
        response_headers={},
    )


class TestInjectReasoningField:
    def test_adds_reasoning_field(self) -> None:
        enhanced = _inject_reasoning_field(ClassifyRisk)
        outputs = get_output_fields(enhanced)
        assert "reasoning" in outputs
        assert "level" in outputs

    def test_preserves_input_fields(self) -> None:
        enhanced = _inject_reasoning_field(ClassifyRisk)
        inputs = get_input_fields(enhanced)
        assert "text" in inputs

    def test_skips_if_already_present(self) -> None:
        result = _inject_reasoning_field(WithExistingReasoning)
        # Should return the same class, not wrap it
        assert result is WithExistingReasoning


class TestChainOfThought:
    async def test_basic_cot(self) -> None:
        """ChainOfThought returns reasoning + original output fields."""

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response(
                {
                    "reasoning": "The document mentions securities violations which are high risk.",
                    "level": "high",
                }
            )

        client = FunctionClient(function=fn)
        cot = ChainOfThought(ClassifyRisk, model="function-test", client=client)
        result = await cot(text="SEC enforcement action against firm...")

        assert result.reasoning is not None
        assert "securities" in result.reasoning.lower()
        assert result.level == "high"

    async def test_cot_instruction_includes_reasoning(self) -> None:
        """System prompt should instruct step-by-step thinking."""
        captured: list[list[dict[str, Any]]] = []

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            captured.append(messages)
            return _json_response({"reasoning": "thinking...", "level": "low"})

        client = FunctionClient(function=fn)
        cot = ChainOfThought(ClassifyRisk, model="function-test", client=client)
        await cot(text="test")

        system_msg = captured[0][0]["content"]
        assert "step-by-step" in system_msg.lower()
        assert "reasoning" in system_msg.lower()

    async def test_cot_records_trace(self) -> None:
        """Trace should be recorded like any other Call."""

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"reasoning": "...", "level": "low"})

        client = FunctionClient(function=fn)
        cot = ChainOfThought(ClassifyRisk, model="function-test", client=client)
        invocation = await cot.invoke(text="test")

        trace = invocation.trace
        assert trace is not None
        assert trace.input_tokens == 50

    async def test_cot_is_a_call(self) -> None:
        """ChainOfThought should be a subclass of Call (for Program discovery)."""
        from kaos_llm_core.programs.call import Call

        cot = ChainOfThought(ClassifyRisk, model="function-test")
        assert isinstance(cot, Call)
