"""Unit tests for the Phase 16.2 MultiChainComparison Program."""

from __future__ import annotations

import json
from typing import Any

import pytest
from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart

from kaos_llm_core.observability.cost import PRICING, ModelPricing
from kaos_llm_core.programs.multi_chain_comparison import (
    MultiChainComparison,
    MultiChainComparisonResult,
)
from kaos_llm_core.signatures import InputField, OutputField, Signature


class _Sig(Signature):
    """Solve a math problem."""

    question: str = InputField(description="The question")
    answer: str = OutputField(description="The numeric answer")


def _json_response(payload: dict[str, Any]) -> ProviderResponse:
    return ProviderResponse.model_construct(
        provider="function",
        model="function-test",
        raw={},
        parts=[ContentPart.model_construct(type="text", text=json.dumps(payload))],
        usage=UsageInfo.model_construct(input_tokens=10, output_tokens=5, total_tokens=15),
        stop_reason="end_turn",
        status_code=200,
        response_headers={},
    )


@pytest.fixture(autouse=True)
def _pricing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        PRICING,
        "function-test",
        ModelPricing(input_per_mtok=1.0, output_per_mtok=2.0),
    )


def _stub_aggregator_response(payload: dict[str, Any]) -> ProviderResponse:
    return _json_response(payload)


class TestMultiChainComparison:
    def test_construction(self) -> None:
        m = MultiChainComparison(_Sig, n=3, producer_model="function:function-test")
        assert m.n == 3
        # Producer is a ChainOfThought; aggregator is a Call
        from kaos_llm_core.programs.call import Call
        from kaos_llm_core.programs.chain_of_thought import ChainOfThought

        assert isinstance(m.producer, ChainOfThought)
        assert isinstance(m.aggregator, Call)

    def test_aggregator_signature_includes_chains_input(self) -> None:
        m = MultiChainComparison(_Sig, n=3, producer_model="function:function-test")
        agg_fields = m.aggregator.signature.model_fields
        assert "chains" in agg_fields
        # Original input + output fields are mirrored
        assert "question" in agg_fields
        assert "answer" in agg_fields

    def test_n_must_be_at_least_2(self) -> None:
        with pytest.raises(ValueError, match="n >= 2"):
            MultiChainComparison(_Sig, n=1, producer_model="function:function-test")

    def test_examples_forwarded_to_inner_producer(self) -> None:
        """``examples=`` flows through to the inner ChainOfThought producer.

        Callers that enforce a grounded-Signature contract (e.g. kaos-agents'
        ``Call(SigClass, examples=load_examples("..."))`` pattern) must be
        able to keep the same calibration when routing through MCC. Pre-fix
        the producer was constructed with no examples regardless of what the
        caller passed; this test pins the forwarded behavior.
        """
        from kaos_llm_core.types import Example

        examples = [
            Example(
                inputs={"question": "what is 2+2?"},
                outputs={"answer": "4"},
            ),
            Example(
                inputs={"question": "what is 3+5?"},
                outputs={"answer": "8"},
            ),
        ]
        m = MultiChainComparison(
            _Sig,
            n=3,
            producer_model="function:function-test",
            examples=examples,
        )
        # ChainOfThought is a Call subclass, so ``examples`` lands directly
        # on the producer.
        assert list(m.producer.examples) == examples

    def test_examples_default_none_keeps_existing_behaviour(self) -> None:
        """Omitting ``examples=`` leaves the producer ungrounded — back-compat."""
        m = MultiChainComparison(_Sig, n=3, producer_model="function:function-test")
        # Pre-fix shape: producer carries no caller-supplied examples.
        assert not getattr(m.producer, "examples", None)

    async def test_forward_aggregates_n_chains(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Run a 3-chain MultiChainComparison through deterministic stubs.

        Each producer sample returns a different "answer". The aggregator
        synthesizes a final answer that the test can recognize. The
        result should expose the synthesized answer plus all 3 chains.
        """
        producer_call_count = 0
        aggregator_call_count = 0

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            nonlocal producer_call_count, aggregator_call_count
            # Sniff the message content to tell producer vs aggregator apart.
            # Aggregator messages contain the word "chains".
            text_blob = json.dumps(messages)
            if "chains" in text_blob:
                aggregator_call_count += 1
                return _json_response({"answer": "synthesized-final"})
            producer_call_count += 1
            return _json_response(
                {
                    "reasoning": f"chain reasoning {producer_call_count}",
                    "answer": f"chain-{producer_call_count}",
                }
            )

        client = FunctionClient(function=fn)

        m = MultiChainComparison(_Sig, n=3, producer_model="function:function-test")
        # Inject the same client into all sub-Calls
        for sub in m.named_calls().values():
            sub._client = client

        result = await m(question="What is 2+2?")
        # 3 producer samples + 1 aggregator call
        assert producer_call_count == 3
        assert aggregator_call_count == 1
        assert isinstance(result, MultiChainComparisonResult)
        assert len(result.chains) == 3
        assert result.failed_count == 0
        # Forward attribute access through the mixin to the synthesized output
        assert result.answer == "synthesized-final"

    async def test_partial_failure_still_aggregates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If some producer samples raise, the aggregator should still
        run with the surviving chains."""
        producer_count = 0

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            nonlocal producer_count
            text_blob = json.dumps(messages)
            if "chains" in text_blob:
                return _json_response({"answer": "from-2-chains"})
            producer_count += 1
            if producer_count == 2:
                raise RuntimeError("simulated provider error")
            return _json_response(
                {"reasoning": f"chain {producer_count}", "answer": f"a{producer_count}"}
            )

        client = FunctionClient(function=fn)
        m = MultiChainComparison(_Sig, n=3, producer_model="function:function-test")
        for sub in m.named_calls().values():
            sub._client = client
            # Disable retries for the test so the simulated error
            # surfaces immediately and the chain is recorded as failed.
            sub.max_retries = 0  # ty: ignore[invalid-assignment]

        result = await m(question="What is 1+1?")
        assert isinstance(result, MultiChainComparisonResult)
        assert len(result.chains) == 2  # 2 survivors out of 3
        assert result.failed_count == 1
        assert result.answer == "from-2-chains"

    async def test_all_failures_raises(self) -> None:
        """If every producer sample raises, the program raises CallError."""
        from kaos_llm_core.errors import CallError

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            raise RuntimeError("always fails")

        client = FunctionClient(function=fn)
        m = MultiChainComparison(_Sig, n=2, producer_model="function:function-test")
        for sub in m.named_calls().values():
            sub._client = client
            sub.max_retries = 0  # ty: ignore[invalid-assignment]

        with pytest.raises(CallError, match=r"all .* producer samples failed"):
            await m(question="anything")
