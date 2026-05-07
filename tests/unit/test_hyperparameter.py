"""Tests for HyperparameterOptimizer."""

from __future__ import annotations

import json
from typing import Any

from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart

from kaos_llm_core.optimization.hyperparameter import (
    HyperparameterOptimizer,
    _grid_configs,
    _random_configs,
)
from kaos_llm_core.optimization.mutations import MutationLog
from kaos_llm_core.programs.call import Call
from kaos_llm_core.signatures import InputField, OutputField, Signature
from kaos_llm_core.types import Example


class ClassifySig(Signature):
    """Classify."""

    text: str = InputField(description="Input")
    label: str = OutputField(description="Label")


def _make_call(answer: str = "positive") -> Call:
    def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
        return ProviderResponse.model_construct(
            provider="function",
            model="function-test",
            raw={},
            parts=[ContentPart.model_construct(type="text", text=json.dumps({"label": answer}))],
            usage=UsageInfo.model_construct(input_tokens=10, output_tokens=5, total_tokens=15),
            stop_reason="end_turn",
            status_code=200,
            response_headers={},
        )

    return Call(ClassifySig, model="function-test", client=FunctionClient(function=fn))


def exact_match(pred: Any, gold: dict[str, Any]) -> float:
    return 1.0 if pred.label == gold["label"] else 0.0


class TestGridConfigs:
    def test_single_param(self) -> None:
        configs = _grid_configs({"temperature": [0.0, 0.5, 1.0]})
        assert len(configs) == 3

    def test_two_params(self) -> None:
        configs = _grid_configs({"temperature": [0.0, 1.0], "top_p": [0.5, 1.0]})
        assert len(configs) == 4

    def test_empty(self) -> None:
        configs = _grid_configs({})
        assert configs == [{}]


class TestRandomConfigs:
    def test_returns_n(self) -> None:
        configs = _random_configs({"temperature": [0.0, 0.5, 1.0]}, n=5, seed=42)
        assert len(configs) == 5

    def test_reproducible(self) -> None:
        a = _random_configs({"temperature": [0.0, 0.5, 1.0]}, n=3, seed=42)
        b = _random_configs({"temperature": [0.0, 0.5, 1.0]}, n=3, seed=42)
        assert a == b


class TestHyperparameterOptimizer:
    async def test_grid_search(self) -> None:
        call = _make_call("positive")
        val = [Example(inputs={"text": "good"}, outputs={"label": "positive"})]
        log = MutationLog()

        opt = HyperparameterOptimizer(
            metric=exact_match,
            search_space={"temperature": [0.0, 0.5, 1.0]},
            mutation_log=log,
        )
        result = await opt.optimize(call, val)

        assert result.configs_tried == 3
        assert result.metric_after >= result.metric_before
        assert len(log.mutations) == 3

    async def test_random_search(self) -> None:
        call = _make_call("positive")
        val = [Example(inputs={"text": "good"}, outputs={"label": "positive"})]

        opt = HyperparameterOptimizer(
            metric=exact_match,
            search_space={"temperature": [0.0, 0.3, 0.5, 0.7, 1.0]},
            strategy="random",
            max_trials=3,
            seed=42,
        )
        result = await opt.optimize(call, val)

        assert result.configs_tried == 3
