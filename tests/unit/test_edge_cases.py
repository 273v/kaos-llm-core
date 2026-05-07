"""Edge case and fuzz tests — boundary conditions across all modules."""

from __future__ import annotations

import json
from typing import Any

from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart

from kaos_llm_core.codecs.chat_codec import ChatCodec
from kaos_llm_core.codecs.xml_codec import XMLCodec
from kaos_llm_core.observability.traces import ExecutionTrace
from kaos_llm_core.optimization.evaluation import evaluate
from kaos_llm_core.programs.call import Call
from kaos_llm_core.router.cascade import CascadeRouter
from kaos_llm_core.signatures import InputField, OutputField, Signature
from kaos_llm_core.types import Example


class SimpleSig(Signature):
    """Simple test sig."""

    text: str = InputField(description="Input")
    result: str = OutputField(description="Result")


class OptionalSig(Signature):
    """Sig with optional output."""

    text: str = InputField(description="Input")
    result: str = OutputField(description="Required result")
    extra: str = OutputField(description="Optional extra", default="")


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


# --- evaluate() edge cases ---


class TestEvaluateEdgeCases:
    async def test_empty_dataset(self) -> None:
        """evaluate() with empty dataset should return zero score."""

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            raise AssertionError("Should not be called")

        client = FunctionClient(function=fn)
        call = Call(SimpleSig, model="function-test", client=client)
        result = await evaluate(call, [], lambda p, g: 1.0)

        assert result.score == 0.0
        assert result.n_total == 0
        assert result.n_correct == 0

    async def test_single_example(self) -> None:
        """evaluate() with one example."""

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"result": "ok"})

        client = FunctionClient(function=fn)
        call = Call(SimpleSig, model="function-test", client=client)
        dataset = [Example(inputs={"text": "hi"}, outputs={"result": "ok"})]

        result = await evaluate(call, dataset, lambda p, g: 1.0 if p.result == g["result"] else 0.0)
        assert result.score == 1.0
        assert result.n_total == 1


# --- XMLCodec edge cases ---


class TestXMLCodecEdgeCases:
    def test_nested_xml_in_content(self) -> None:
        """Content with XML-like tags should not break extraction."""
        codec = XMLCodec()
        text = "<result>The file contains &lt;html&gt; tags and <b>bold</b> text.</result>"
        response = _text_response(text)
        # This will extract everything between <result> and </result>,
        # including the nested tags
        result = codec.decode(SimpleSig, response)
        assert "html" in result["result"] or "bold" in result["result"]

    def test_optional_field_missing(self) -> None:
        """Missing optional output field should not raise."""
        codec = XMLCodec()
        text = "<result>value</result>"
        response = _text_response(text)
        result = codec.decode(OptionalSig, response)
        assert result["result"] == "value"
        assert "extra" not in result  # optional, not present

    def test_empty_tags(self) -> None:
        """Empty XML tags should return empty string."""
        codec = XMLCodec()
        text = "<result></result>"
        response = _text_response(text)
        result = codec.decode(SimpleSig, response)
        assert result["result"] == ""


# --- ChatCodec edge cases ---


class TestChatCodecEdgeCases:
    def test_optional_field_missing_ok(self) -> None:
        """Missing optional chat field should not raise."""
        codec = ChatCodec()
        text = "[result]\nThe answer is 42."
        response = _text_response(text)
        result = codec.decode(OptionalSig, response)
        assert result["result"] == "The answer is 42."

    def test_markers_at_start_of_text(self) -> None:
        """Markers at very start of response (no preamble)."""
        codec = ChatCodec()
        text = "[result]\nDone."
        response = _text_response(text)
        result = codec.decode(SimpleSig, response)
        assert result["result"] == "Done."


# --- CascadeRouter edge cases ---


class TestCascadeEdgeCases:
    def test_single_model_cascade(self) -> None:
        """Cascade with one model should work and accept immediately."""
        router = CascadeRouter(
            models=["only-model"],
            escalation_check=lambda _r: True,
        )
        assert len(router.models) == 1

    async def test_single_model_via_call(self) -> None:
        """Single-model cascade through Call."""

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"result": "ok"})

        client = FunctionClient(function=fn)

        import kaos_llm_core.programs.call as call_mod

        original = call_mod.create_client
        call_mod.create_client = lambda *a, **kw: client  # ty: ignore[invalid-assignment]
        try:
            router = CascadeRouter(models=["m1"], escalation_check=lambda _r: True)
            call = Call(SimpleSig, router=router)
            result = await call(text="test")
            assert result.result == "ok"
            assert router.model_used == "m1"
            assert len(router.last_traces) == 1
        finally:
            call_mod.create_client = original


# --- ExecutionTrace edge cases ---


class TestTraceEdgeCases:
    def test_total_cost_leaf_no_children(self) -> None:
        """Leaf trace: total_cost_usd == cost_usd."""
        trace = ExecutionTrace(cost_usd=0.005)
        assert trace.total_cost_usd == 0.005

    def test_total_cost_program_with_children(self) -> None:
        """Program trace: total_cost_usd == sum of children, not double-counted."""
        child1 = ExecutionTrace(cost_usd=0.01)
        child2 = ExecutionTrace(cost_usd=0.02)
        parent = ExecutionTrace(cost_usd=0.03, children=[child1, child2])
        assert abs(parent.total_cost_usd - 0.03) < 1e-10  # sum of children, not 0.06

    def test_nested_program_traces(self) -> None:
        """Three-level trace tree should not double-count."""
        leaf = ExecutionTrace(cost_usd=0.01)
        mid = ExecutionTrace(cost_usd=0.01, children=[leaf])
        top = ExecutionTrace(cost_usd=0.01, children=[mid])
        # top.total_cost_usd = mid.total_cost_usd = leaf.total_cost_usd = 0.01
        assert abs(top.total_cost_usd - 0.01) < 1e-10


# --- Signature edge cases ---


class TestSignatureEdgeCases:
    async def test_unicode_inputs(self) -> None:
        """Unicode text should pass through cleanly."""

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"result": "处理完毕"})

        client = FunctionClient(function=fn)
        call = Call(SimpleSig, model="function-test", client=client)
        result = await call(text="日本語テキスト")
        assert result.result == "处理完毕"

    async def test_very_long_input(self) -> None:
        """Long inputs should not crash."""

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"result": "processed"})

        client = FunctionClient(function=fn)
        call = Call(SimpleSig, model="function-test", client=client)
        long_text = "word " * 10000
        result = await call(text=long_text)
        assert result.result == "processed"

    async def test_empty_string_input(self) -> None:
        """Empty string is a valid input."""

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"result": "empty"})

        client = FunctionClient(function=fn)
        call = Call(SimpleSig, model="function-test", client=client)
        result = await call(text="")
        assert result.result == "empty"
