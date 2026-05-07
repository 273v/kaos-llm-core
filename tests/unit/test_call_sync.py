"""Tests for Call.__call_sync__ — synchronous execution wrapper."""

from __future__ import annotations

import json
from typing import Any

from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart

from kaos_llm_core.programs.call import Call
from kaos_llm_core.signatures import InputField, OutputField, Signature


class ExtractSig(Signature):
    """Extract entities."""

    text: str = InputField(description="Input text")
    entities: list[str] = OutputField(description="Entity names")


def _make_client(response_data: dict[str, Any]) -> FunctionClient:
    def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
        return ProviderResponse.model_construct(
            provider="function",
            model="function-test",
            raw={},
            parts=[ContentPart.model_construct(type="text", text=json.dumps(response_data))],
            usage=UsageInfo.model_construct(input_tokens=10, output_tokens=5, total_tokens=15),
            stop_reason="end_turn",
            status_code=200,
            response_headers={},
        )

    return FunctionClient(function=fn)


class TestCallSync:
    def test_sync_returns_typed_result(self) -> None:
        """__call_sync__ should return the same typed result as async __call__."""
        client = _make_client({"entities": ["SEC", "Acme"]})
        call = Call(ExtractSig, model="function-test", client=client)

        result = call.__call_sync__(text="The SEC filed suit against Acme Corp.")
        assert result.entities == ["SEC", "Acme"]

    def test_sync_populates_trace(self) -> None:
        """invoke() (driven synchronously) should populate the Invocation trace."""
        import asyncio

        client = _make_client({"entities": ["X"]})
        call = Call(ExtractSig, model="function-test", client=client)

        invocation = asyncio.run(call.invoke(text="test"))

        trace = invocation.trace
        assert trace is not None
        assert trace.call_name == "ExtractSig"
        assert trace.input_tokens == 10
        assert trace.output_tokens == 5
        assert trace.latency_ms > 0
