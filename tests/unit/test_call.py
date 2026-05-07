"""Tests for kaos_llm_core.programs.call — Call execution with FunctionClient."""

from __future__ import annotations

import json
from typing import Any

import pytest
from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart

from kaos_llm_core.errors import CallError, ValidationRetryExhaustedError
from kaos_llm_core.programs.call import Call
from kaos_llm_core.signatures import InputField, OutputField, Signature
from kaos_llm_core.types import Example


class ExtractEntities(Signature):
    """Extract named entities from text."""

    text: str = InputField(description="Input text")
    entities: list[str] = OutputField(description="Entity names")
    confidence: float = OutputField(description="Confidence 0-1")


class ClassifyRisk(Signature):
    """Classify risk level."""

    text: str = InputField(description="Input text")
    level: str = OutputField(description="Risk level: low, medium, high")


def _json_response(data: dict[str, Any], model: str = "function-test") -> ProviderResponse:
    text = json.dumps(data)
    return ProviderResponse.model_construct(
        provider="function",
        model=model,
        raw={},
        parts=[ContentPart.model_construct(type="text", text=text)],
        usage=UsageInfo.model_construct(input_tokens=50, output_tokens=25, total_tokens=75),
        stop_reason="end_turn",
        response_id="test-id",
        status_code=200,
        response_headers={},
        latency_ms=100.0,
    )


class TestCallBasic:
    async def test_simple_call(self) -> None:
        """Call returns typed output from FunctionClient."""

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"entities": ["SEC", "Acme"], "confidence": 0.9})

        client = FunctionClient(function=fn)
        call = Call(ExtractEntities, model="function-test", client=client)
        result = await call(text="The SEC filed suit against Acme Corp.")

        assert result.entities == ["SEC", "Acme"]  # type: ignore[attr-defined]
        assert result.confidence == 0.9  # type: ignore[attr-defined]

    async def test_call_records_trace(self) -> None:
        """Trace is populated after successful call."""

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"entities": ["X"], "confidence": 0.5})

        client = FunctionClient(function=fn)
        call = Call(ExtractEntities, model="function-test", client=client)
        invocation = await call.invoke(text="test")

        trace = invocation.trace
        assert trace is not None
        assert trace.call_name == "ExtractEntities"
        assert trace.model == "function-test"
        assert trace.input_tokens == 50
        assert trace.output_tokens == 25
        assert trace.retries == 0
        assert trace.error is None

    async def test_call_with_custom_instructions(self) -> None:
        """Custom instructions override the Signature docstring."""
        captured_messages: list[list[dict[str, Any]]] = []

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            captured_messages.append(messages)
            return _json_response({"level": "high"})

        client = FunctionClient(function=fn)
        call = Call(
            ClassifyRisk,
            model="function-test",
            client=client,
            instructions="Always classify as high risk.",
        )
        result = await call(text="test")
        assert result.level == "high"  # type: ignore[attr-defined]

        # Verify custom instruction was used in the system prompt
        assert len(captured_messages) == 1
        system_msg = captured_messages[0][0]
        assert "Always classify as high risk" in system_msg["content"]

    async def test_call_with_examples(self) -> None:
        """Few-shot examples are included in the messages."""
        captured_messages: list[list[dict[str, Any]]] = []

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            captured_messages.append(messages)
            return _json_response({"level": "low"})

        client = FunctionClient(function=fn)
        examples = [
            Example(
                inputs={"text": "minor issue"},
                outputs={"level": "low"},
            ),
        ]
        call = Call(
            ClassifyRisk,
            model="function-test",
            client=client,
            examples=examples,
        )
        await call(text="test input")

        # Should have: system, user (example), assistant (example), user (actual)
        assert len(captured_messages) == 1
        msgs = captured_messages[0]
        assert len(msgs) == 4
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"


class TestCallRetry:
    async def test_validation_retry(self) -> None:
        """Call retries on codec decode failure."""
        call_count = 0

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First attempt: bad JSON
                return _json_response({"entities": ["X"]})  # missing confidence
            # Second attempt: good response
            return _json_response({"entities": ["X"], "confidence": 0.8})

        client = FunctionClient(function=fn)
        call = Call(
            ExtractEntities,
            model="function-test",
            client=client,
            max_retries=2,
        )
        result = await call(text="test")
        assert result.confidence == 0.8  # type: ignore[attr-defined]
        assert call_count == 2

    async def test_retry_exhausted(self) -> None:
        """Raises ValidationRetryExhaustedError when all retries fail."""

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            # Always return bad JSON (missing confidence)
            return _json_response({"entities": ["X"]})

        client = FunctionClient(function=fn)
        call = Call(
            ExtractEntities,
            model="function-test",
            client=client,
            max_retries=1,
        )
        with pytest.raises(ValidationRetryExhaustedError, match="validation attempts failed"):
            await call(text="test")

    async def test_validation_retry_accumulates_token_usage(self) -> None:
        """Bug 7 regression: trace.{input,output,total}_tokens must be the SUM
        across the validation-retry loop, not just the final attempt's usage.

        Previous behavior: Call._execute_inner overwrote ``trace.input_tokens =
        response.usage.input_tokens`` on each successful decode. If decode
        attempt 1 failed and attempt 2 succeeded, the trace reported only
        attempt 2's tokens, silently undercounting budgets and mutation costs.

        Repro: one failed parse + one success. Each FunctionClient response
        carries 50/25/75 input/output/total tokens (see _json_response). The
        trace should report 100/50/150, not 50/25/75.
        """
        call_count = 0

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _json_response({"entities": ["X"]})  # missing confidence
            return _json_response({"entities": ["X"], "confidence": 0.8})

        client = FunctionClient(function=fn)
        call = Call(
            ExtractEntities,
            model="function-test",
            client=client,
            max_retries=2,
        )
        invocation = await call.invoke(text="test")
        result = invocation.output
        assert call_count == 2
        # Trace lives on the Invocation returned by invoke().
        trace = invocation.trace
        assert trace is not None
        # Each FunctionClient response carries 50/25/75. With 2 attempts the
        # totals must be summed.
        assert trace.input_tokens == 100, (
            f"Expected accumulated input_tokens=100 across 2 attempts, "
            f"got {trace.input_tokens}. Bug 7 regression."
        )
        assert trace.output_tokens == 50, (
            f"Expected accumulated output_tokens=50, got {trace.output_tokens}"
        )
        assert trace.total_tokens == 150, (
            f"Expected accumulated total_tokens=150, got {trace.total_tokens}"
        )
        # The result itself is just the validated output; the trace lives on
        # the Invocation.
        assert result is not None
        assert invocation.trace.total_tokens == 150


class TestCallErrors:
    async def test_no_model_raises(self) -> None:
        """CallError when no model is specified."""
        call = Call(ExtractEntities)
        with pytest.raises(CallError, match="No model specified"):
            await call(text="test")


class TestCallLearnableState:
    def test_get_learnable_state(self) -> None:
        examples = [Example(inputs={"text": "a"}, outputs={"level": "low"})]
        call = Call(
            ClassifyRisk,
            model="test",
            examples=examples,
            instructions="Custom instructions.",
        )
        state = call.get_learnable_state()
        assert state["instructions"] == "Custom instructions."
        assert len(state["examples"]) == 1
        assert state["examples"][0]["inputs"]["text"] == "a"

    def test_set_learnable_state(self) -> None:
        call = Call(ClassifyRisk, model="test")
        call.set_learnable_state(
            {
                "instructions": "New instruction.",
                "examples": [
                    {"inputs": {"text": "b"}, "outputs": {"level": "high"}},
                ],
            }
        )
        assert call.instructions == "New instruction."
        assert len(call.examples) == 1
        assert call.examples[0].outputs["level"] == "high"

    def test_get_learnable_state_includes_v2_fields(self) -> None:
        """v2 schema: state must include hyperparameters, codec, model."""
        call = Call(
            ClassifyRisk,
            model="anthropic:claude-haiku-4-5",
            temperature=0.4,
            top_p=0.85,
            max_tokens=128,
        )
        state = call.get_learnable_state()
        assert state["hyperparameters"] == {
            "temperature": 0.4,
            "top_p": 0.85,
            "max_tokens": 128,
        }
        assert state["model"] == "anthropic:claude-haiku-4-5"
        assert state["codec"].endswith("JSONCodec")

    def test_get_learnable_state_filters_unknown_kwargs(self) -> None:
        """Only the persisted-hyperparameters allowlist round-trips."""
        call = Call(
            ClassifyRisk,
            model="test",
            temperature=0.5,
            custom_provider_extra="should-not-persist",
        )
        state = call.get_learnable_state()
        assert "temperature" in state["hyperparameters"]
        assert "custom_provider_extra" not in state["hyperparameters"]

    def test_set_learnable_state_restores_hyperparameters(self) -> None:
        """v2 round-trip: setting state must restore hyperparameters."""
        call = Call(ClassifyRisk, model="test", temperature=0.0)
        call.set_learnable_state(
            {
                "hyperparameters": {"temperature": 0.9, "top_p": 0.8},
            }
        )
        assert call._kwargs["temperature"] == 0.9
        assert call._kwargs["top_p"] == 0.8

    def test_set_learnable_state_preserves_unrelated_kwargs(self) -> None:
        """Setting hyperparameters must not wipe non-hyperparameter kwargs."""
        call = Call(ClassifyRisk, model="test", temperature=0.0, custom_extra="keep-me")
        call.set_learnable_state({"hyperparameters": {"temperature": 0.7}})
        assert call._kwargs["temperature"] == 0.7
        assert call._kwargs["custom_extra"] == "keep-me"

    def test_set_learnable_state_restores_model(self) -> None:
        call = Call(ClassifyRisk, model="old-model")
        call.set_learnable_state({"model": "new-model"})
        assert call._model == "new-model"

    def test_set_learnable_state_v1_compat_no_hyperparameters(self) -> None:
        """A v1 state dict (no hyperparameters/codec/model) loads cleanly."""
        call = Call(ClassifyRisk, model="test", temperature=0.5)
        call.set_learnable_state(
            {
                "instructions": "v1 inst",
                "examples": [{"inputs": {"text": "x"}, "outputs": {"level": "low"}}],
            }
        )
        assert call.instructions == "v1 inst"
        # In-code temperature must persist when v1 state has no hyperparameters
        assert call._kwargs.get("temperature") == 0.5

    def test_router_round_trip_cascade(self) -> None:
        """F5 regression: CascadeRouter survives get/set learnable state."""
        from kaos_llm_core.router.cascade import CascadeRouter

        router = CascadeRouter(
            models=["anthropic:claude-haiku-4-5", "anthropic:claude-sonnet-4-6"],
        )
        call = Call(ClassifyRisk, router=router)
        state = call.get_learnable_state()
        assert state["router"] is not None
        assert state["router"]["type"] == "cascade"
        assert state["router"]["models"] == [
            "anthropic:claude-haiku-4-5",
            "anthropic:claude-sonnet-4-6",
        ]

        # Restore into a fresh Call without a router and verify shape.
        fresh = Call(ClassifyRisk, model="test")
        assert fresh._router is None
        fresh.set_learnable_state(state)
        assert isinstance(fresh._router, CascadeRouter)
        assert fresh._router.models == [
            "anthropic:claude-haiku-4-5",
            "anthropic:claude-sonnet-4-6",
        ]

    def test_router_round_trip_rule(self) -> None:
        """F5 regression: RuleRouter survives get/set learnable state."""
        from kaos_llm_core.router.rules import Rule, RuleRouter

        router = RuleRouter(
            rules=[
                Rule(
                    model="anthropic:claude-haiku-4-5",
                    signature_name="ClassifyRisk",
                ),
                Rule(
                    model="anthropic:claude-sonnet-4-6",
                    input_matches={"priority": "high"},
                ),
            ],
            default_model="anthropic:claude-haiku-4-5",
        )
        call = Call(ClassifyRisk, router=router)
        state = call.get_learnable_state()
        assert state["router"]["type"] == "rule"
        assert state["router"]["default_model"] == "anthropic:claude-haiku-4-5"
        assert len(state["router"]["rules"]) == 2

        fresh = Call(ClassifyRisk, model="test")
        fresh.set_learnable_state(state)
        assert isinstance(fresh._router, RuleRouter)
        assert fresh._router.default_model == "anthropic:claude-haiku-4-5"
        assert len(fresh._router.rules) == 2
        assert fresh._router.rules[0].model == "anthropic:claude-haiku-4-5"
        assert fresh._router.rules[0].signature_name == "ClassifyRisk"
        assert fresh._router.rules[1].input_matches == {"priority": "high"}

    def test_get_learnable_state_router_none_when_no_router(self) -> None:
        """Calls without a router serialize ``router=None``."""
        call = Call(ClassifyRisk, model="test")
        state = call.get_learnable_state()
        assert state["router"] is None

    def test_set_learnable_state_v1_router_key_missing_keeps_router(self) -> None:
        """Loading a v1 envelope (no router key) leaves the existing router."""
        from kaos_llm_core.router.cascade import CascadeRouter

        router = CascadeRouter(models=["anthropic:claude-haiku-4-5"])
        call = Call(ClassifyRisk, router=router)
        call.set_learnable_state({"instructions": "v1 inst"})
        assert call._router is router
