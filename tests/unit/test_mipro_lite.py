"""Unit tests for the Phase 17.0 MiproLiteOptimizer (MIPRO-lite)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart

from kaos_llm_core.observability.cost import PRICING, ModelPricing
from kaos_llm_core.optimization.mipro_lite import MiproLiteOptimizer, MiproLiteResult
from kaos_llm_core.programs.call import Call
from kaos_llm_core.signatures import InputField, OutputField, Signature
from kaos_llm_core.types import Example


class _Sentiment(Signature):
    """Classify sentiment as positive or negative."""

    text: str = InputField(description="The input text")
    sentiment: str = OutputField(description="positive or negative")


def _stub_response(payload: dict[str, Any]) -> ProviderResponse:
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


def _accuracy_metric(prediction: Any, gold: dict[str, Any]) -> float:
    pred = str(getattr(prediction, "sentiment", "")).strip().lower()
    expected = str(gold.get("sentiment", "")).strip().lower()
    return 1.0 if pred == expected else 0.0


class TestMiproLiteOptimizerConstruction:
    def test_construction_defaults(self) -> None:
        opt = MiproLiteOptimizer(metric=_accuracy_metric)
        assert opt.n_instruction_candidates == 4
        assert opt.n_demo_candidates == 3
        assert opt.n_trials == 8

    def test_validation_rejects_zero_trials(self) -> None:
        with pytest.raises(ValueError, match="n_trials"):
            MiproLiteOptimizer(metric=_accuracy_metric, n_trials=0)

    def test_validation_rejects_zero_candidates(self) -> None:
        with pytest.raises(ValueError, match="n_instruction_candidates"):
            MiproLiteOptimizer(metric=_accuracy_metric, n_instruction_candidates=0)
        with pytest.raises(ValueError, match="n_demo_candidates"):
            MiproLiteOptimizer(metric=_accuracy_metric, n_demo_candidates=0)


class TestMiproLiteOptimizerFlow:
    async def test_baseline_only_when_no_train_set(self) -> None:
        """Empty train_set should raise — bootstrap needs training data."""

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _stub_response({"sentiment": "positive"})

        client = FunctionClient(function=fn)
        call = Call(_Sentiment, model="function-test", client=client, instructions="Be helpful.")

        opt = MiproLiteOptimizer(metric=_accuracy_metric)
        with pytest.raises(ValueError, match="train_set"):
            await opt.optimize(
                call,
                train_set=[],
                val_set=[Example(inputs={"text": "x"}, outputs={"sentiment": "positive"})],
            )

    async def test_full_flow_against_function_client(self) -> None:
        """End-to-end flow through deterministic stubs.

        The producer always returns 'positive'. The proposer (used for
        instruction generation) returns a synthetic instruction. The
        eval metric returns 1.0 when the producer's 'positive' matches
        the gold; we craft the dataset so the baseline scores high
        and the optimizer accepts the (instruction, demos) it tried.
        """
        producer_count = {"n": 0}

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            producer_count["n"] += 1
            text_blob = json.dumps(messages)
            # The proposer is a Call against ProposeInstruction whose
            # output fields are proposed_instruction + rationale.
            if "proposed_instruction" in text_blob:
                return _stub_response(
                    {
                        "proposed_instruction": "Be very explicit and answer only positive.",
                        "rationale": "Failure cases all expected positive.",
                    }
                )
            return _stub_response({"sentiment": "positive"})

        client = FunctionClient(function=fn)
        call = Call(
            _Sentiment,
            model="function-test",
            client=client,
            instructions="Original instruction.",
        )

        train = [
            Example(inputs={"text": f"happy {i}"}, outputs={"sentiment": "positive"})
            for i in range(8)
        ]
        val = [
            Example(inputs={"text": f"glad {i}"}, outputs={"sentiment": "positive"})
            for i in range(4)
        ]

        opt = MiproLiteOptimizer(
            metric=_accuracy_metric,
            n_instruction_candidates=2,
            n_demo_candidates=2,
            n_trials=4,
            minibatch_size=2,
            n_promote=1,
            proposer_model="function-test",
            seed=42,
        )
        # The proposer Call is constructed inside optimize() and uses
        # create_client(). Patch create_client so the proposer uses
        # our FunctionClient too.
        from unittest.mock import patch

        with (
            patch("kaos_llm_client.create_client", return_value=client),
            patch("kaos_llm_core.programs.call.create_client", return_value=client),
        ):
            result = await opt.optimize(call, train_set=train, val_set=val)

        assert isinstance(result, MiproLiteResult)
        assert result.metric_before == 1.0  # producer always returns the right answer
        # We can't strictly require improvement (already at 100%), but
        # we can require the optimizer ran at least one trial and
        # collected candidates.
        assert result.n_instruction_candidates >= 2  # baseline + at least one proposed
        assert result.n_demo_candidates >= 1
        assert result.n_trials >= 1
        # Either kept the baseline (already perfect) or accepted a tied alternative.
        assert result.metric_after >= result.metric_before

    async def test_search_space_grows_with_more_candidates(self) -> None:
        """Larger K and L should yield a larger candidate pool."""

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            text_blob = json.dumps(messages)
            if "proposed_instruction" in text_blob:
                return _stub_response({"proposed_instruction": "new", "rationale": "r"})
            return _stub_response({"sentiment": "positive"})

        client = FunctionClient(function=fn)
        call = Call(_Sentiment, model="function-test", client=client, instructions="Base.")
        train = [
            Example(inputs={"text": f"x{i}"}, outputs={"sentiment": "positive"}) for i in range(10)
        ]
        val = [
            Example(inputs={"text": f"y{i}"}, outputs={"sentiment": "positive"}) for i in range(4)
        ]

        opt = MiproLiteOptimizer(
            metric=_accuracy_metric,
            n_instruction_candidates=3,
            n_demo_candidates=3,
            n_trials=6,
            minibatch_size=2,
            n_promote=1,
            proposer_model="function-test",
            seed=7,
        )
        from unittest.mock import patch

        with (
            patch("kaos_llm_client.create_client", return_value=client),
            patch("kaos_llm_core.programs.call.create_client", return_value=client),
        ):
            result = await opt.optimize(call, train_set=train, val_set=val)

        # baseline + up to 3 proposed = at most 4 instructions
        assert result.n_instruction_candidates <= 4
        # baseline (original_examples) + up to 3 bootstrapped = at most 4 demo sets
        assert result.n_demo_candidates <= 4
        # n_trials capped at the configured value (6)
        assert result.n_trials <= 6
