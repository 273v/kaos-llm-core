"""Tests for kaos_llm_core.programs.decorator — @llm_call decorator."""

from __future__ import annotations

from typing import Any

from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart
from pydantic import BaseModel

from kaos_llm_core.integrations.common.signatures import (
    build_signature_from_function as _function_to_signature,
)
from kaos_llm_core.programs.call import Call
from kaos_llm_core.programs.decorator import llm_call
from kaos_llm_core.signatures import Signature
from kaos_llm_core.signatures.introspection import (
    get_input_fields,
    get_instruction,
    get_output_fields,
)


class Entity(BaseModel):
    name: str
    type: str


class TestFunctionToSignature:
    def test_simple_function(self) -> None:
        async def extract(text: str) -> str:  # ty: ignore[empty-body]
            """Extract entities."""
            ...

        sig = _function_to_signature(extract)
        assert issubclass(sig, Signature)
        inputs = get_input_fields(sig)
        assert "text" in inputs
        outputs = get_output_fields(sig)
        assert "result" in outputs
        assert get_instruction(sig) == "Extract entities."

    def test_pydantic_return_type(self) -> None:
        async def extract(text: str) -> Entity:  # ty: ignore[empty-body]
            """Extract an entity."""
            ...

        sig = _function_to_signature(extract)
        inputs = get_input_fields(sig)
        outputs = get_output_fields(sig)
        assert "text" in inputs
        assert "name" in outputs
        assert "type" in outputs
        # Should NOT have 'result' field
        assert "result" not in outputs

    def test_multiple_params(self) -> None:
        async def classify(text: str, category: str = "general") -> str:  # ty: ignore[empty-body]
            """Classify text."""
            ...

        sig = _function_to_signature(classify)
        inputs = get_input_fields(sig)
        assert "text" in inputs
        assert "category" in inputs
        assert not inputs["category"].is_required()

    def test_list_return_type(self) -> None:
        async def extract(text: str) -> list[str]:  # ty: ignore[empty-body]
            """Extract items."""
            ...

        sig = _function_to_signature(extract)
        outputs = get_output_fields(sig)
        assert "result" in outputs


class TestLLMCallDecorator:
    async def test_decorated_function_is_callable(self) -> None:
        """Decorated function works with FunctionClient."""

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return ProviderResponse.model_construct(
                provider="function",
                model="function-test",
                raw={},
                parts=[
                    ContentPart.model_construct(
                        type="text",
                        text='{"result": "extracted entities here"}',
                    )
                ],
                usage=UsageInfo.model_construct(input_tokens=10, output_tokens=5, total_tokens=15),
                stop_reason="end_turn",
                status_code=200,
                response_headers={},
            )

        client = FunctionClient(function=fn)

        @llm_call(model="function-test")
        async def extract(text: str) -> str:  # ty: ignore[empty-body]
            """Extract entities from text."""
            ...

        # Patch the call's client
        inner_call: Call = extract._call
        inner_call._client = client

        result = await extract(text="The SEC filed suit.")
        assert result is not None

    def test_decorator_exposes_call(self) -> None:
        @llm_call(model="test-model")
        async def my_func(text: str) -> str:  # ty: ignore[empty-body]
            """Test func."""
            ...

        assert hasattr(my_func, "_call")
        inner_call = my_func._call
        assert isinstance(inner_call, Call)

    def test_decorator_exposes_signature_class(self) -> None:
        @llm_call(model="test-model")
        async def my_func(text: str) -> str:  # ty: ignore[empty-body]
            """Test func."""
            ...

        sig_cls = my_func._signature_class
        assert issubclass(sig_cls, Signature)
