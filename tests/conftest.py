"""Shared fixtures for kaos-llm-core tests."""

from __future__ import annotations

from typing import Any

import pytest
from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient


def make_json_response(json_text: str, model: str = "function-test") -> ProviderResponse:
    """Create a ProviderResponse with the given JSON text content."""
    from kaos_llm_client.types import ContentPart

    return ProviderResponse.model_construct(
        provider="function",
        model=model,
        raw={},
        parts=[ContentPart.model_construct(type="text", text=json_text)],
        usage=UsageInfo.model_construct(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        ),
        stop_reason="end_turn",
        response_id="test-response-id",
        status_code=200,
        response_headers={},
        request_id=None,
        latency_ms=100.0,
    )


@pytest.fixture
def function_client() -> FunctionClient:
    """A FunctionClient that returns a simple JSON response.

    Override the function for specific test scenarios.
    """

    def default_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
        return make_json_response('{"result": "test output"}')

    return FunctionClient(function=default_fn)
