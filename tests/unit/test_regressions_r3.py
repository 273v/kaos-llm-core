"""Regression tests for code review round 3 findings.

Tests the architectural refactor: Call step decomposition, ChainOfThought
as hook overrides, CascadeRouter delegation, ChatCodec line-anchored markers.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.profiles import AnthropicModelProfile, GoogleModelProfile
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart

from kaos_llm_core.codecs.chat_codec import ChatCodec
from kaos_llm_core.errors import CallError
from kaos_llm_core.programs.call import Call
from kaos_llm_core.programs.chain_of_thought import ChainOfThought
from kaos_llm_core.signatures import InputField, OutputField, Signature


class ExtractSig(Signature):
    """Extract entities."""

    text: str = InputField(description="Input text")
    entities: list[str] = OutputField(description="Entity names")


class ClassifySig(Signature):
    """Classify risk."""

    text: str = InputField(description="Input text")
    level: str = OutputField(description="Risk level")


class SigWithDefault(Signature):
    """Sig with default_factory field."""

    text: str = InputField(description="Input text")
    tags: list[str] = InputField(description="Tags", default_factory=list)
    result: str = OutputField(description="Result")


class MultiFieldSig(Signature):
    """Multiple output fields for ChatCodec tests."""

    text: str = InputField(description="Input")
    analysis: str = OutputField(description="Analysis text")
    summary: str = OutputField(description="Summary text")


def _json_response(data: dict[str, Any], model: str = "function-test") -> ProviderResponse:
    return ProviderResponse.model_construct(
        provider="function",
        model=model,
        raw={},
        parts=[ContentPart.model_construct(type="text", text=json.dumps(data))],
        usage=UsageInfo.model_construct(input_tokens=20, output_tokens=10, total_tokens=30),
        stop_reason="end_turn",
        status_code=200,
        response_headers={},
    )


def _thinking_response(text: str, thinking: str, model: str = "function-test") -> ProviderResponse:
    """Response with native thinking content."""
    return ProviderResponse.model_construct(
        provider="function",
        model=model,
        raw={},
        parts=[
            ContentPart.model_construct(type="thinking", thinking=thinking),
            ContentPart.model_construct(type="text", text=text),
        ],
        usage=UsageInfo.model_construct(input_tokens=20, output_tokens=10, total_tokens=30),
        stop_reason="end_turn",
        status_code=200,
        response_headers={},
    )


# --- R3 Finding 1: Native-thinking CoT retry works ---


class TestR3Finding1NativeThinkingRetry:
    async def test_native_thinking_retries_on_decode_failure(self) -> None:
        """Native-thinking path must go through retry loop on decode failure."""
        call_count = 0
        thinking_profile = AnthropicModelProfile(
            supports_thinking=True,
            thinking_parameter="thinking",
        )

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First attempt: completely unparseable
                return _thinking_response("not json at all", "thinking first...")
            # Second attempt: good response
            return _thinking_response('{"level": "high"}', "now I see it clearly")

        client = FunctionClient(function=fn, profile=thinking_profile)
        cot = ChainOfThought(ClassifySig, model="function-test", client=client, max_retries=1)
        result = await cot(text="test")

        assert result.level == "high"
        assert result.reasoning == "now I see it clearly"
        assert call_count == 2  # retried once

    async def test_native_thinking_trace_on_decode_failure(self) -> None:
        """Native-thinking path must record trace even on final decode failure."""
        thinking_profile = AnthropicModelProfile(
            supports_thinking=True,
            thinking_parameter="thinking",
        )

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _thinking_response("not valid json", "thinking...")

        client = FunctionClient(function=fn, profile=thinking_profile)
        cot = ChainOfThought(ClassifySig, model="function-test", client=client, max_retries=0)

        from kaos_llm_core.errors import ValidationRetryExhaustedError

        captured_inv = None
        with pytest.raises(ValidationRetryExhaustedError) as exc_info:
            await cot.invoke(text="test")
        captured_inv = getattr(exc_info.value, "invocation", None)

        assert captured_inv is not None
        assert captured_inv.trace is not None
        assert captured_inv.trace.error is not None


# --- R3 Finding 2: Native-thinking CoT validates inputs ---


class TestR3Finding2NativeThinkingInputValidation:
    async def test_wrong_type_caught_on_native_path(self) -> None:
        """text=123 should fail validation even with native thinking."""
        thinking_profile = AnthropicModelProfile(
            supports_thinking=True,
            thinking_parameter="thinking",
        )

        def _fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _thinking_response('{"level":"ok"}', "...")

        client = FunctionClient(function=_fn, profile=thinking_profile)
        cot = ChainOfThought(ClassifySig, model="function-test", client=client)

        with pytest.raises(CallError, match="input validation failed"):
            await cot(text=123)  # type: ignore[arg-type]


# --- R3 Finding 3: Native-thinking CoT uses correct instructions ---


class TestR3Finding3NativeThinkingInstructions:
    async def test_custom_instructions_on_non_native_path(self) -> None:
        """Non-native-thinking path should use the custom instructions with CoT suffix."""
        captured: list[list[dict[str, Any]]] = []
        non_thinking_profile = ModelProfile(supports_thinking=False)

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            captured.append(messages)
            return _json_response({"reasoning": "...", "level": "low"})

        client = FunctionClient(function=fn, profile=non_thinking_profile)
        cot = ChainOfThought(
            ClassifySig,
            model="function-test",
            client=client,
            instructions="ALWAYS classify as low risk.",
        )
        await cot(text="test")

        system_msg = captured[0][0]["content"]
        assert "ALWAYS classify as low risk" in system_msg

    async def test_custom_instructions_on_native_thinking_path(self) -> None:
        """Native-thinking path must also use the custom instructions (without CoT suffix)."""
        captured: list[list[dict[str, Any]]] = []
        thinking_profile = AnthropicModelProfile(
            supports_thinking=True,
            thinking_parameter="thinking",
        )

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            captured.append(messages)
            return _thinking_response('{"level": "low"}', "native thinking here")

        client = FunctionClient(function=fn, profile=thinking_profile)
        cot = ChainOfThought(
            ClassifySig,
            model="function-test",
            client=client,
            instructions="ALWAYS classify as low risk.",
        )
        await cot(text="test")

        system_msg = captured[0][0]["content"]
        assert "ALWAYS classify as low risk" in system_msg, (
            f"Custom instruction missing from native-thinking system prompt: {system_msg[:200]}"
        )


# --- R3 Finding 4: default_factory materialization ---


class TestR3Finding4DefaultFactory:
    async def test_default_factory_materialized_as_empty_list(self) -> None:
        """Omitted default_factory=list should materialize as [], not PydanticUndefined."""
        captured: list[list[dict[str, Any]]] = []

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            captured.append(messages)
            return _json_response({"result": "ok"})

        client = FunctionClient(function=fn)
        call = Call(SigWithDefault, model="function-test", client=client)
        result = await call(text="test")  # omit tags, should default to []

        assert result.result == "ok"
        # The prompt should contain the default [] not PydanticUndefined
        user_msg = captured[0][-1]["content"]
        assert "PydanticUndefined" not in user_msg


# --- R3 Finding 5: thinkingConfig for Google ---


class TestR3Finding5ThinkingConfig:
    async def test_google_thinking_config_passed(self) -> None:
        """Google models with thinkingConfig should get the param in kwargs."""
        captured_kwargs: list[dict[str, Any]] = []
        google_thinking_profile = GoogleModelProfile(
            supports_thinking=True,
            thinking_parameter="thinkingConfig",
        )

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _thinking_response('{"level": "medium"}', "google thinking")

        client = FunctionClient(function=fn, profile=google_thinking_profile)

        cot = ChainOfThought(ClassifySig, model="function-test", client=client)

        # Override _prepare_call_kwargs to capture what's passed
        original_prepare = cot._prepare_call_kwargs

        def capturing_prepare() -> dict[str, Any]:
            kwargs = original_prepare()
            captured_kwargs.append(dict(kwargs))
            return kwargs

        cot._prepare_call_kwargs = capturing_prepare  # ty: ignore[invalid-assignment]

        await cot(text="test scenario")

        assert len(captured_kwargs) == 1
        assert "thinkingConfig" in captured_kwargs[0]


# --- R3 Finding 6: ChatCodec marker collision ---


class TestR3Finding6MarkerCollision:
    def test_marker_inside_content_not_treated_as_boundary(self) -> None:
        """Literal [summary] inside another field's content should not split."""
        codec = ChatCodec()
        # The analysis content contains a literal [summary] mid-line
        text = (
            "[analysis]\n"
            "The document references [summary] statistics from Q3.\n"
            "Overall findings are positive.\n\n"
            "[summary]\n"
            "This is the actual summary."
        )
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
        result = codec.decode(MultiFieldSig, response)

        # The actual [summary] marker at start of line should be the boundary
        assert "actual summary" in result["summary"]
        # The analysis should contain the full text including [summary] reference
        assert "[summary] statistics" in result["analysis"]


# --- Verify hook-based architecture ---


class TestHookOverrideArchitecture:
    async def test_prepare_call_kwargs_is_called(self) -> None:
        """Subclass overriding _prepare_call_kwargs should have it called."""
        prepare_called = False

        class CustomCall(Call):
            def _prepare_call_kwargs(self) -> dict[str, Any]:
                nonlocal prepare_called
                prepare_called = True
                kwargs = super()._prepare_call_kwargs()
                kwargs["temperature"] = 0.0
                return kwargs

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"entities": ["X"]})

        client = FunctionClient(function=fn)
        call = CustomCall(ExtractSig, model="function-test", client=client)
        await call(text="test")
        assert prepare_called

    async def test_post_process_is_called(self) -> None:
        """Subclass overriding _post_process should have it called."""
        post_called = False

        class CustomCall(Call):
            def _post_process(
                self, result: Any, response: ProviderResponse, output_dict: dict[str, Any]
            ) -> Any:
                nonlocal post_called
                post_called = True
                return result

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"entities": ["X"]})

        client = FunctionClient(function=fn)
        call = CustomCall(ExtractSig, model="function-test", client=client)
        await call(text="test")
        assert post_called
