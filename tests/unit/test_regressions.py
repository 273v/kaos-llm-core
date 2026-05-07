"""Regression tests for bugs found in code review.

Each test targets a specific finding and would have caught the bug.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart
from pydantic import BaseModel

from kaos_llm_core.errors import CallError
from kaos_llm_core.integrations.common.signatures import (
    build_signature_from_function as _function_to_signature,
)
from kaos_llm_core.programs.base import Program
from kaos_llm_core.programs.call import Call
from kaos_llm_core.programs.judge import Judge
from kaos_llm_core.signatures import InputField, OutputField, Signature
from kaos_llm_core.signatures.introspection import get_output_fields


class ExtractSig(Signature):
    """Extract entities."""

    text: str = InputField(description="Input text")
    entities: list[str] = OutputField(description="Entity names")


class ClassifySig(Signature):
    """Classify risk."""

    text: str = InputField(description="Input text")
    level: str = OutputField(description="Level")
    confidence: float = OutputField(description="Confidence 0-1")


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


# --- Finding 1: Call must validate inputs ---


class TestFinding1InputValidation:
    async def test_missing_required_input_raises(self) -> None:
        """Call should reject missing required inputs before hitting the LLM."""

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            raise AssertionError("Should not reach LLM")

        client = FunctionClient(function=fn)
        call = Call(ExtractSig, model="function-test", client=client)

        with pytest.raises(CallError, match="missing required inputs"):
            await call()  # no 'text' input

    async def test_extra_input_raises(self) -> None:
        """Call should reject unexpected inputs."""

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            raise AssertionError("Should not reach LLM")

        client = FunctionClient(function=fn)
        call = Call(ExtractSig, model="function-test", client=client)

        with pytest.raises(CallError, match="unexpected inputs"):
            await call(text="hello", bogus_field="oops")

    async def test_valid_inputs_pass(self) -> None:
        """Valid inputs should not raise."""

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"entities": ["X"]})

        client = FunctionClient(function=fn)
        call = Call(ExtractSig, model="function-test", client=client)
        result = await call(text="hello")
        assert result.entities == ["X"]


# --- Finding 2: CascadeRouter + Call integration ---


class TestFinding2CascadeIntegration:
    async def test_call_with_cascade_router_actually_cascades(self) -> None:
        """Call(..., router=CascadeRouter(...)) must invoke execute_cascade."""
        from kaos_llm_core.router.cascade import CascadeRouter

        # We can't fully test cascade with FunctionClient (create_client needs real providers)
        # but we can verify Call detects CascadeRouter and delegates.
        # Create a CascadeRouter with a mock execute_cascade to verify it's called.
        cascade_called = False
        original_execute = CascadeRouter.execute_cascade

        async def mock_execute(self_router, parent_call, inputs):  # type: ignore[no-untyped-def]
            nonlocal cascade_called
            cascade_called = True
            from kaos_llm_core.programs._invocation import Invocation, TokenUsage
            from kaos_llm_core.signatures.introspection import create_output_model

            model = create_output_model(parent_call.signature)
            output = model.model_validate({"entities": ["cascaded"]})
            return Invocation(
                client=None,
                model="mock-cascade",
                context=None,
                output=output,
                trace=None,
                usage=TokenUsage(),
            )

        router = CascadeRouter(models=["m1", "m2"])
        CascadeRouter.execute_cascade = mock_execute  # ty: ignore[invalid-assignment]
        try:
            call = Call(ExtractSig, router=router)
            result = await call(text="test")
            assert cascade_called, "execute_cascade was not called"
            assert result.entities == ["cascaded"]
        finally:
            CascadeRouter.execute_cascade = original_execute  # type: ignore[assignment]


# --- Finding 3: Judge re-judges on retry ---


class TestFinding3JudgeRetry:
    async def test_judge_rejudges_retried_output(self) -> None:
        """After retry, the judgment must describe the retried output, not the original."""
        judge_call_count = 0

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            nonlocal judge_call_count
            system = messages[0]["content"] if messages else ""
            if "Extract entities" in system:
                return _json_response({"entities": ["original"]})
            else:
                judge_call_count += 1
                if judge_call_count == 1:
                    # First judgment: low score → triggers retry
                    return _json_response({"quality_score": 0.3, "reasoning": "bad"})
                else:
                    # Second judgment: should be for the retried output
                    return _json_response({"quality_score": 0.95, "reasoning": "good retry"})

        client = FunctionClient(function=fn)
        judge = Judge(
            ExtractSig,
            producer_model="function-test",
            judge_model="function-test",
            quality_threshold=0.7,
            retry_model="function-test",
        )
        judge.produce._client = client
        judge.judge_call._client = client
        # Patch the retry call's client too — Judge creates it inside forward()
        # We need to monkeypatch create_client for this
        import kaos_llm_core.programs.call as call_mod

        original_create = call_mod.create_client
        call_mod.create_client = lambda *a, **kw: client  # ty: ignore[invalid-assignment]
        try:
            result = await judge(text="test")
            # Judgment should be from the second judge call (after retry)
            assert result.judgment.quality_score == 0.95
            assert "good retry" in result.judgment.reasoning
            assert judge_call_count == 2, f"Expected 2 judge calls, got {judge_call_count}"
        finally:
            call_mod.create_client = original_create  # type: ignore[assignment]


# --- Finding 4: Provider failure during retry ---


class TestFinding4ProviderFailureRetry:
    async def test_provider_failure_raises_call_error_not_validation_error(self) -> None:
        """Provider failures must raise CallError immediately, not ValidationRetryExhaustedError."""

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            raise ConnectionError("Network failure")

        client = FunctionClient(function=fn)
        call = Call(ExtractSig, model="function-test", client=client, max_retries=2)

        # Must be CallError, NOT ValidationRetryExhaustedError or UnboundLocalError
        with pytest.raises(CallError, match="Network failure"):
            await call(text="test")

    async def test_validation_failure_still_retries(self) -> None:
        """Codec/validation failures should still go through the retry loop."""
        call_count = 0

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _json_response({"entities": "not-a-list"})  # bad type
            return _json_response({"entities": ["recovered"]})

        client = FunctionClient(function=fn)
        call = Call(ExtractSig, model="function-test", client=client, max_retries=1)
        result = await call(text="test")
        assert result.entities == ["recovered"]
        assert call_count == 2


# --- Finding 5: Stale child traces ---


class TestFinding5StaleTraces:
    async def test_conditional_program_no_stale_traces(self) -> None:
        """A sub-Call that doesn't run this invocation should not appear in traces."""

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"entities": ["X"]})

        client = FunctionClient(function=fn)

        class ConditionalProgram(Program):
            def __init__(self) -> None:
                self.always_runs = Call(ExtractSig, model="function-test", client=client)
                self.sometimes_runs = Call(ExtractSig, model="function-test", client=client)

            async def forward(self, *, run_both: bool = False, **kwargs: Any) -> Any:
                result = await self.always_runs(text="test")
                if run_both:
                    await self.sometimes_runs(text="test")
                return result

        prog = ConditionalProgram()

        # First call: run both
        inv1 = await prog.invoke(run_both=True)
        assert inv1.trace is not None
        assert len(inv1.trace.children) == 2

        # Second call: only run always_runs
        inv2 = await prog.invoke(run_both=False)
        assert inv2.trace is not None
        # Should have 1 child, not 2 (no stale trace from sometimes_runs)
        assert len(inv2.trace.children) == 1
        assert inv2.trace.children[0].call_name == "ExtractSig"


# --- Finding 6: @llm_call optional output fields ---


class _ResultModelForFinding6(BaseModel):
    """Module-level model to avoid get_type_hints forward-ref issues."""

    name: str
    nickname: str | None = None


class TestFinding6OptionalOutputFields:
    def test_optional_output_field_not_required(self) -> None:
        """A Pydantic output field with `= None` should be optional in the Signature."""

        # Must use module-level model because get_type_hints resolves forward refs
        # against the module globals, and locally-defined classes aren't visible there.
        sig = _function_to_signature(_extract_for_finding6)
        outputs = get_output_fields(sig)
        assert "name" in outputs
        assert "nickname" in outputs
        assert outputs["name"].is_required()
        assert not outputs["nickname"].is_required(), "nickname should be optional but was required"


async def _extract_for_finding6(text: str) -> _ResultModelForFinding6:  # ty: ignore[empty-body]
    """Extract name."""
    ...
