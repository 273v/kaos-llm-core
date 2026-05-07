"""Tests for Ensemble — multi-model voting program."""

from __future__ import annotations

import json
from typing import Any

import pytest
from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart

from kaos_llm_core.errors import CallError
from kaos_llm_core.programs.ensemble import Ensemble
from kaos_llm_core.signatures import InputField, OutputField, Signature


class ClassifySig(Signature):
    """Classify risk level."""

    text: str = InputField(description="Input text")
    level: str = OutputField(description="Risk level: low, medium, high")


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


class TestEnsemble:
    async def test_majority_vote(self) -> None:
        """Three voters: 2 say high, 1 says medium → high wins."""
        responses = iter(
            [
                _json_response({"level": "high"}),
                _json_response({"level": "high"}),
                _json_response({"level": "medium"}),
            ]
        )

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return next(responses)

        client = FunctionClient(function=fn)
        ensemble = Ensemble(
            ClassifySig,
            models=["m1", "m2", "m3"],
            vote_field="level",
        )
        for voter in ensemble.voters:
            voter._client = client

        result = await ensemble(text="test")

        assert result.selected.level == "high"
        assert result.votes["high"] == 2
        assert result.votes["medium"] == 1
        assert len(result.all_results) == 3

    async def test_all_aggregation(self) -> None:
        """With aggregation='all', returns all results without voting."""
        responses = iter(
            [
                _json_response({"level": "low"}),
                _json_response({"level": "high"}),
            ]
        )

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return next(responses)

        client = FunctionClient(function=fn)
        ensemble = Ensemble(
            ClassifySig,
            models=["m1", "m2"],
            aggregation="all",
        )
        for voter in ensemble.voters:
            voter._client = client

        result = await ensemble(text="test")

        assert len(result.all_results) == 2
        assert result.selected.level == "low"  # first result

    async def test_tolerates_partial_failures(self) -> None:
        """Ensemble should work even if some models fail."""
        call_count = 0

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Model 1 failed")
            return _json_response({"level": "high"})

        client = FunctionClient(function=fn)
        ensemble = Ensemble(
            ClassifySig,
            models=["m1", "m2"],
            aggregation="all",
            max_retries=0,  # disable retry so the failure propagates
        )
        for voter in ensemble.voters:
            voter._client = client

        result = await ensemble(text="test")
        assert len(result.all_results) == 1
        assert result.selected.level == "high"

    def test_empty_models_raises(self) -> None:
        with pytest.raises(CallError, match="at least one model"):
            Ensemble(ClassifySig, models=[])

    def test_named_calls_includes_voters(self) -> None:
        ensemble = Ensemble(ClassifySig, models=["m1", "m2", "m3"])
        calls = ensemble.named_calls()
        assert "voter_0" in calls
        assert "voter_1" in calls
        assert "voter_2" in calls

    async def test_trace_tree(self) -> None:
        """Ensemble should have trace children from all voters."""
        responses = iter(
            [
                _json_response({"level": "high"}),
                _json_response({"level": "high"}),
            ]
        )

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return next(responses)

        client = FunctionClient(function=fn)
        ensemble = Ensemble(
            ClassifySig,
            models=["m1", "m2"],
            vote_field="level",
        )
        for voter in ensemble.voters:
            voter._client = client

        invocation = await ensemble.invoke(text="test")

        trace = invocation.trace
        assert trace is not None
        # Ensemble has voters as voter_0, voter_1 — those are discovered by named_calls
        assert trace.children is not None
