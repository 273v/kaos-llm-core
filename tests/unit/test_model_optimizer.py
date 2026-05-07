"""Tests for ModelOptimizer."""

from __future__ import annotations

import json
from typing import Any

from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart

from kaos_llm_core.optimization.budget import Budget
from kaos_llm_core.optimization.model_optimizer import ModelOptimizer
from kaos_llm_core.programs.call import Call
from kaos_llm_core.signatures import InputField, OutputField, Signature
from kaos_llm_core.types import Example


class _Sig(Signature):
    """Answer."""

    text: str = InputField(description="In")
    answer: str = OutputField(description="Out")


def _client_returning(answer: str) -> FunctionClient:
    def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
        return ProviderResponse.model_construct(
            provider="function",
            model="function-test",
            raw={},
            parts=[ContentPart.model_construct(type="text", text=json.dumps({"answer": answer}))],
            usage=UsageInfo.model_construct(input_tokens=10, output_tokens=5, total_tokens=15),
            stop_reason="end_turn",
            status_code=200,
            response_headers={},
        )

    return FunctionClient(function=fn)


def _exact(pred: Any, gold: dict[str, Any]) -> float:
    return 1.0 if getattr(pred, "answer", "") == gold.get("answer") else 0.0


class TestModelOptimizer:
    async def test_rejects_empty_models(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="at least one candidate model"):
            ModelOptimizer(metric=_exact, models=[])

    async def test_selects_above_threshold(self) -> None:
        client = _client_returning("yes")
        call = Call(_Sig, model="function-test", client=client)
        val = [Example(inputs={"text": "q"}, outputs={"answer": "yes"})]

        opt = ModelOptimizer(
            metric=_exact,
            models=["function-test"],
            min_score=0.5,
        )
        result = await opt.optimize(call, val)
        assert result.best_model == "function-test"
        assert result.best_score == 1.0
        assert result.stop_reason in ("completed",)

    async def test_threshold_not_met(self) -> None:
        client = _client_returning("wrong")
        call = Call(_Sig, model="function-test", client=client)
        val = [Example(inputs={"text": "q"}, outputs={"answer": "right"})]

        opt = ModelOptimizer(
            metric=_exact,
            models=["function-test"],
            min_score=0.9,
        )
        result = await opt.optimize(call, val)
        assert result.stop_reason == "threshold_not_met"
        # Falls back to the highest-scoring model even below threshold.
        assert result.best_model == "function-test"

    async def test_budget_halts_after_first_trial(self) -> None:
        client = _client_returning("yes")
        call = Call(_Sig, model="function-test", client=client)
        val = [Example(inputs={"text": "q"}, outputs={"answer": "yes"})]

        opt = ModelOptimizer(
            metric=_exact,
            models=["function-test", "function-test"],  # two slots, same client
            min_score=0.5,
            budget=Budget(max_trials=1),
        )
        result = await opt.optimize(call, val)
        # Exactly one model should have been scored before budget exhausted.
        assert len(result.scores_by_model) == 1
        assert result.stop_reason == "budget_trials"
