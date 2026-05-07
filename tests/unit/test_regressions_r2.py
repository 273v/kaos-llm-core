"""Regression tests for code review round 2 findings.

Each test targets a specific finding from the second review.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart

from kaos_llm_core.codecs.chat_codec import ChatCodec
from kaos_llm_core.errors import CallError
from kaos_llm_core.observability.traces import ExecutionTrace
from kaos_llm_core.programs.base import Program
from kaos_llm_core.programs.call import Call
from kaos_llm_core.settings import KaosLLMCoreSettings
from kaos_llm_core.signatures import InputField, OutputField, Signature


class ExtractSig(Signature):
    """Extract entities."""

    text: str = InputField(description="Input text")
    entities: list[str] = OutputField(description="Entity names")


class ClassifySigWithDefault(Signature):
    """Classify with a default mode."""

    text: str = InputField(description="Input text")
    mode: str = InputField(description="Analysis mode", default="standard")
    level: str = OutputField(description="Level")


class MultiOutputSig(Signature):
    """Multiple output types."""

    text: str = InputField(description="Input")
    summary: str = OutputField(description="Summary")
    tags: list[str] = OutputField(description="Tags")


def _json_response(data: dict[str, Any]) -> ProviderResponse:
    return ProviderResponse.model_construct(
        provider="function",
        model="function-test",
        raw={},
        parts=[ContentPart.model_construct(type="text", text=json.dumps(data))],
        usage=UsageInfo.model_construct(input_tokens=20, output_tokens=10, total_tokens=30),
        stop_reason="end_turn",
        status_code=200,
        response_headers={},
    )


# --- R2 Finding 1: Provider errors must not be validation retries ---


class TestR2Finding1ProviderErrorClassification:
    async def test_runtime_error_raises_call_error(self) -> None:
        """A RuntimeError from chat_async must become CallError."""
        from kaos_llm_core.errors import ValidationRetryExhaustedError

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            raise RuntimeError("upstream boom")

        client = FunctionClient(function=fn)
        call = Call(ExtractSig, model="function-test", client=client, max_retries=3)

        with pytest.raises(CallError, match="upstream boom"):
            await call(text="test")

        # Must NOT be a ValidationRetryExhaustedError
        try:
            await call(text="test")
        except CallError as e:
            assert not isinstance(e, ValidationRetryExhaustedError)


# --- R2 Finding 4: Program trace on forward() exception ---


class TestR2Finding4ProgramExceptionTrace:
    async def test_trace_recorded_on_exception(self) -> None:
        """Program.last_trace should be set even when forward() raises."""

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"entities": ["X"]})

        client = FunctionClient(function=fn)

        class FailingProgram(Program):
            def __init__(self) -> None:
                self.extract = Call(ExtractSig, model="function-test", client=client)

            async def forward(self, **kwargs: Any) -> Any:
                await self.extract(text="test")
                raise ValueError("something broke after extraction")

        prog = FailingProgram()
        captured_inv = None
        try:
            await prog.invoke(text="test")
        except ValueError as exc:
            assert "something broke" in str(exc)
            captured_inv = getattr(exc, "invocation", None)

        # Trace should exist with error and child from the successful sub-call
        assert captured_inv is not None
        trace = captured_inv.trace
        assert trace is not None
        assert trace.error is not None
        assert "something broke" in trace.error
        assert len(trace.children) == 1


# --- R2 Finding 5: Input type validation and default materialization ---


class TestR2Finding5InputTypeDefaults:
    async def test_wrong_type_raises_call_error(self) -> None:
        """Passing text=123 should fail type validation."""

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            raise AssertionError("Should not reach LLM")

        client = FunctionClient(function=fn)
        call = Call(ExtractSig, model="function-test", client=client)

        with pytest.raises(CallError, match="input validation failed"):
            await call(text=123)  # type: ignore[arg-type]

    async def test_omitted_default_is_materialized(self) -> None:
        """An omitted field with a default should be materialized in the prompt."""
        captured: list[list[dict[str, Any]]] = []

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            captured.append(messages)
            return _json_response({"level": "low"})

        client = FunctionClient(function=fn)
        call = Call(ClassifySigWithDefault, model="function-test", client=client)

        result = await call(text="test")  # omit mode, should default to "standard"
        assert result.level == "low"

        # The prompt should contain the default value "standard"
        user_msg = captured[0][-1]["content"]
        assert "standard" in user_msg


# --- R2 Finding 6: total_cost_usd no longer double-counts ---


class TestR2Finding6CostDoubleCount:
    def test_program_trace_does_not_double_count(self) -> None:
        """Program traces with children should not double-count cost."""
        child1 = ExecutionTrace(
            call_name="c1",
            model="anthropic:claude-haiku-4-5",
            input_tokens=500,
            output_tokens=200,
            cost_usd=0.005,
        )
        child2 = ExecutionTrace(
            call_name="c2",
            model="anthropic:claude-sonnet-4-6",
            input_tokens=500,
            output_tokens=200,
            cost_usd=0.008,
        )
        parent = ExecutionTrace(
            call_name="Pipeline",
            model="(program)",
            cost_usd=0.013,  # aggregated from children
            children=[child1, child2],
        )

        # total_cost_usd should be sum of children, not cost_usd + children
        assert abs(parent.total_cost_usd - 0.013) < 1e-10
        # NOT 0.026 (which would be double-counting)
        assert parent.total_cost_usd < 0.014


# --- R2 Finding 7: ChatCodec reordered markers ---


class TestR2Finding7ChatCodecReordering:
    def test_reordered_markers_decoded_correctly(self) -> None:
        """Markers appearing out of schema order should still decode correctly."""
        codec = ChatCodec()
        # Schema order is: summary, tags. But response has tags first.
        text = '[tags]\n["legal", "compliance"]\n\n[summary]\nThis is a summary.'
        response = ProviderResponse.model_construct(
            provider="test",
            model="test",
            raw={},
            parts=[ContentPart.model_construct(type="text", text=text)],
            usage=UsageInfo.model_construct(input_tokens=10, output_tokens=5, total_tokens=15),
            stop_reason="end_turn",
            status_code=200,
            response_headers={},
        )
        result = codec.decode(MultiOutputSig, response)
        assert result["summary"] == "This is a summary."
        assert result["tags"] == ["legal", "compliance"]

    def test_list_field_parsed_as_json(self) -> None:
        """Non-string fields should be parsed via JSON when possible."""
        codec = ChatCodec()
        text = '[summary]\nBrief.\n\n[tags]\n["a", "b", "c"]'
        response = ProviderResponse.model_construct(
            provider="test",
            model="test",
            raw={},
            parts=[ContentPart.model_construct(type="text", text=text)],
            usage=UsageInfo.model_construct(input_tokens=10, output_tokens=5, total_tokens=15),
            stop_reason="end_turn",
            status_code=200,
            response_headers={},
        )
        result = codec.decode(MultiOutputSig, response)
        assert isinstance(result["tags"], list)
        assert result["tags"] == ["a", "b", "c"]


# --- R2 Finding 8: trace_enabled ---


class TestR2Finding8TraceEnabled:
    async def test_trace_disabled_produces_no_trace(self) -> None:
        """When trace_enabled=False, Call should not record a trace."""

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"entities": ["X"]})

        settings = KaosLLMCoreSettings(trace_enabled=False)
        client = FunctionClient(function=fn)
        call = Call(
            ExtractSig,
            model="function-test",
            client=client,
            core_settings=settings,
        )
        invocation = await call.invoke(text="test")
        assert invocation.output.entities == ["X"]
        assert invocation.trace is None
