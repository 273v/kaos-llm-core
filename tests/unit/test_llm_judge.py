"""Unit tests for LLMJudge metric using FunctionClient."""

from __future__ import annotations

import json
from typing import Any

import pytest
from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart

from kaos_llm_core.metrics import LLMJudge
from kaos_llm_core.metrics.llm_judge import _BUILTIN_RUBRICS, LLMJudgeSignature


def _json_response(data: dict[str, Any]) -> ProviderResponse:
    text = json.dumps(data)
    return ProviderResponse.model_construct(
        provider="function",
        model="function-test",
        raw={},
        parts=[ContentPart.model_construct(type="text", text=text)],
        usage=UsageInfo.model_construct(input_tokens=10, output_tokens=5, total_tokens=15),
        stop_reason="end_turn",
        response_id="test-id",
        status_code=200,
        response_headers={},
        latency_ms=10.0,
    )


def _make_judge(score: float, rubric: str = "helpfulness") -> LLMJudge:
    def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
        return _json_response({"score": score})

    client = FunctionClient(function=fn)
    return LLMJudge(model="function-test", rubric=rubric, client=client)


class TestLLMJudgeMetadata:
    def test_requires_model(self) -> None:
        with pytest.raises(ValueError):
            LLMJudge(model="")

    def test_builtin_rubric_names(self) -> None:
        judge = _make_judge(0.5, rubric="helpfulness")
        assert judge.rubric_text == _BUILTIN_RUBRICS["helpfulness"]

    def test_custom_rubric_passes_through(self) -> None:
        judge = _make_judge(0.5, rubric="be terse and accurate")
        assert judge.rubric_text == "be terse and accurate"

    def test_signature_fields(self) -> None:
        from kaos_llm_core.signatures.introspection import get_input_fields

        names = set(get_input_fields(LLMJudgeSignature).keys())
        assert names == {"rubric", "prediction", "gold"}


class TestLLMJudgeAsync:
    async def test_acall_returns_score(self) -> None:
        judge = _make_judge(0.75)
        result = await judge.acall("good prediction", "expected answer")
        assert result == 0.75

    async def test_clamps_to_unit(self) -> None:
        judge = _make_judge(2.5)
        result = await judge.acall("p", "g")
        assert result == 1.0
        judge_neg = _make_judge(-0.5)
        result_neg = await judge_neg.acall("p", "g")
        assert result_neg == 0.0

    async def test_dict_inputs(self) -> None:
        judge = _make_judge(0.6)
        result = await judge.acall({"answer": "x"}, {"answer": "x"})
        assert result == 0.6

    async def test_provider_failure_returns_zero(self) -> None:
        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            raise RuntimeError("provider boom")

        client = FunctionClient(function=fn)
        judge = LLMJudge(model="function-test", rubric="helpfulness", client=client)
        result = await judge.acall("p", "g")
        assert result == 0.0


class TestLLMJudgeSync:
    def test_sync_call(self) -> None:
        judge = _make_judge(0.42)
        result = judge("prediction", "gold")
        assert result == 0.42
