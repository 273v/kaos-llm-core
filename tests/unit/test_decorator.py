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


class TitleAndConfidence(BaseModel):
    """Module-level multi-field model used by the unwrap tests.

    Defined at module scope (not inside a test method) so
    ``typing.get_type_hints`` can resolve the forward reference under
    ``from __future__ import annotations``.
    """

    title: str
    confidence: float


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


class TestSingleFieldUnwrap:
    """Single-output Signatures auto-unwrap so the function's `-> T`
    annotation matches what callers actually receive.

    The 2026-05-18 persona matrix surfaced this: a Program declared
    `-> str` was returning a synthesized SignatureOutput object, breaking
    every ``(await fn(...)).strip()`` caller. See
    ``kaos-modules/docs/plans/persona-matrix-followups.md`` §5.
    """

    def _stub_function_client(self, payload: str) -> FunctionClient:
        """Build a deterministic FunctionClient that returns ``payload``.

        Used to drive @llm_call wrappers without touching live providers.
        """

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return ProviderResponse.model_construct(
                provider="function",
                model="function-test",
                raw={},
                parts=[ContentPart.model_construct(type="text", text=payload)],
                usage=UsageInfo.model_construct(input_tokens=10, output_tokens=5, total_tokens=15),
                stop_reason="end_turn",
                status_code=200,
                response_headers={},
            )

        return FunctionClient(function=fn)

    def test_unwrap_field_chosen_for_single_output(self) -> None:
        """A `-> str` function gets the auto-unwrap path enabled."""

        @llm_call(model="test-model")
        async def make_title(text: str) -> str:  # ty: ignore[empty-body]
            """Produce a title."""
            ...

        # Decoration time: exactly one OutputField → unwrap field set.
        assert make_title._unwrap_field is not None
        # That field exists on the signature class.
        sig_cls = make_title._signature_class
        assert make_title._unwrap_field in get_output_fields(sig_cls)

    def test_unwrap_disabled_for_multi_field_signature(self) -> None:
        """A Pydantic-model return type with multiple fields disables unwrap.

        ``TitleAndConfidence`` is defined at module scope above so the
        type-hint resolver finds it.
        """

        @llm_call(model="test-model")
        async def make_titled(text: str) -> TitleAndConfidence:  # ty: ignore[empty-body]
            """Produce a titled summary."""
            ...

        # Multi-output → no unwrap; callers get the full SignatureOutput.
        # (The build_signature_from_function expansion flattens a BaseModel
        # return type into multiple OutputFields, hence "multi".)
        sig_cls = make_titled._signature_class
        n_outputs = len(get_output_fields(sig_cls))
        if n_outputs > 1:
            assert make_titled._unwrap_field is None
        else:
            # Some Signature builders wrap the BaseModel as a single
            # field instead of flattening — in that case unwrap IS the
            # right behavior (caller gets the BaseModel directly).
            assert make_titled._unwrap_field is not None

    async def test_unwrap_returns_field_value_for_string_signature(self) -> None:
        """End-to-end: `-> str` Program returns a plain ``str``, not an
        object. This is the kaos-ui SPA ``summarize_session_title``
        regression closure.
        """

        @llm_call(model="function-test")
        async def make_title(text: str) -> str:  # ty: ignore[empty-body]
            """Produce a short title for the input."""
            ...

        # Patch the inner Call's client with a deterministic stub that
        # returns a JSON-encoded single-field payload matching the
        # generated SignatureOutput shape.
        sig_cls = make_title._signature_class
        only_field = next(iter(get_output_fields(sig_cls)))
        payload = f'{{"{only_field}": "Mutual NDA Review"}}'
        make_title._call._client = self._stub_function_client(payload)

        result = await make_title(text="some content")
        # The result is the raw string — the legacy `(await ...).strip()`
        # pattern from kaos-ui's title_service is now honest.
        assert isinstance(result, str)
        assert result == "Mutual NDA Review"
        assert result.strip() == "Mutual NDA Review"
