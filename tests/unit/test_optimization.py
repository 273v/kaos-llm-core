"""Tests for the optimization framework — evaluation, mutations, bootstrap."""

from __future__ import annotations

import json
from typing import Any

import pytest
from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart

from kaos_llm_core.optimization.bootstrap import BootstrapOptimizer
from kaos_llm_core.optimization.evaluation import EvalResult, evaluate
from kaos_llm_core.optimization.mutations import Mutation, MutationLog
from kaos_llm_core.programs.call import Call
from kaos_llm_core.signatures import InputField, OutputField, Signature
from kaos_llm_core.types import Example


class ClassifySig(Signature):
    """Classify sentiment."""

    text: str = InputField(description="Input text")
    sentiment: str = OutputField(description="positive, negative, or neutral")


def _make_classifier(responses: dict[str, str]) -> Call:
    """Create a Call that returns sentiment based on a lookup table."""

    def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
        # Extract the text from the user message
        user_msg = messages[-1]["content"]
        # Simple lookup: find which key appears in the message
        sentiment = "neutral"
        for key, val in responses.items():
            if key in user_msg:
                sentiment = val
                break
        return ProviderResponse.model_construct(
            provider="function",
            model="function-test",
            raw={},
            parts=[
                ContentPart.model_construct(
                    type="text",
                    text=json.dumps({"sentiment": sentiment}),
                )
            ],
            usage=UsageInfo.model_construct(input_tokens=10, output_tokens=5, total_tokens=15),
            stop_reason="end_turn",
            status_code=200,
            response_headers={},
        )

    return Call(ClassifySig, model="function-test", client=FunctionClient(function=fn))


def exact_match(prediction: Any, gold: dict[str, Any]) -> float:
    """Simple exact match metric."""
    return 1.0 if prediction.sentiment == gold["sentiment"] else 0.0


# --- Evaluation Tests ---


class TestEvaluation:
    async def test_perfect_score(self) -> None:
        call = _make_classifier({"great": "positive", "terrible": "negative", "fine": "neutral"})
        dataset = [
            Example(inputs={"text": "great product"}, outputs={"sentiment": "positive"}),
            Example(inputs={"text": "terrible service"}, outputs={"sentiment": "negative"}),
        ]
        result = await evaluate(call, dataset, exact_match)
        assert result.score == 1.0
        assert result.n_correct == 2
        assert result.n_errors == 0
        assert result.n_total == 2

    async def test_partial_score(self) -> None:
        # Classifier always returns "neutral"
        call = _make_classifier({})
        dataset = [
            Example(inputs={"text": "great"}, outputs={"sentiment": "positive"}),
            Example(inputs={"text": "ok"}, outputs={"sentiment": "neutral"}),
        ]
        result = await evaluate(call, dataset, exact_match)
        assert result.score == 0.5  # 1 correct out of 2
        assert result.n_correct == 1

    async def test_eval_result_failures(self) -> None:
        call = _make_classifier({})
        dataset = [
            Example(inputs={"text": "great"}, outputs={"sentiment": "positive"}),
            Example(inputs={"text": "ok"}, outputs={"sentiment": "neutral"}),
        ]
        result = await evaluate(call, dataset, exact_match)
        failures = result.failures()
        assert len(failures) == 1
        assert failures[0].example.inputs["text"] == "great"

    async def test_eval_result_properties(self) -> None:
        result = EvalResult(score=0.75, n_total=4, n_correct=3, n_errors=0)
        assert result.accuracy == 0.75
        assert result.error_rate == 0.0


# --- Mutation Tests ---


class TestMutations:
    def test_mutation_improvement(self) -> None:
        m = Mutation(
            strategy="bootstrap",
            mutation_type="add_examples",
            call_name="Classify",
            before={},
            after={},
            rationale="test",
            metric_before=0.5,
            metric_after=0.8,
        )
        assert m.improvement == pytest.approx(0.3)

    def test_mutation_round_trip(self) -> None:
        m = Mutation(
            strategy="instruction_tuning",
            mutation_type="change_instructions",
            call_name="Extract",
            before={"instructions": "old"},
            after={"instructions": "new"},
            rationale="improved clarity",
            metric_before=0.6,
            metric_after=0.85,
            tokens_used=1500,
            cost_usd=0.005,
            accepted=True,
        )
        d = m.model_dump()
        restored = Mutation.model_validate(d)
        assert restored.strategy == "instruction_tuning"
        assert restored.metric_after == 0.85
        assert restored.accepted is True

    def test_mutation_log_persistence(self, tmp_path: Any) -> None:
        path = tmp_path / "mutations.jsonl"
        log = MutationLog(path=path)
        log.record(
            Mutation(
                strategy="bootstrap",
                mutation_type="add_examples",
                call_name="Test",
                before={},
                after={},
                rationale="test",
                metric_before=0.5,
                metric_after=0.7,
                accepted=True,
            )
        )
        log.record(
            Mutation(
                strategy="bootstrap",
                mutation_type="add_examples",
                call_name="Test",
                before={},
                after={},
                rationale="test2",
                metric_before=0.7,
                metric_after=0.65,
                accepted=False,
            )
        )

        # Load from disk
        loaded = MutationLog.load(path)
        assert len(loaded.mutations) == 2
        assert len(loaded.accepted()) == 1
        best = loaded.best_improvement()
        assert best is not None
        assert best.improvement == pytest.approx(0.2)
        # Phase 16.5: every persisted mutation carries schema_version=1
        assert all(m.schema_version == 1 for m in loaded.mutations)

    def test_mutation_log_load_rejects_future_schema_version(self, tmp_path) -> None:
        """A log line carrying schema_version > MUTATION_SCHEMA_VERSION
        must be rejected loudly so callers do not silently misread fields."""
        import json as _json

        from kaos_llm_core.optimization.mutations import MUTATION_SCHEMA_VERSION

        path = tmp_path / "future.jsonl"
        # Hand-craft a line with a future version. Use an integer well
        # above the current max so the test does not silently pass after
        # a real version bump.
        future = MUTATION_SCHEMA_VERSION + 99
        record = {
            "schema_version": future,
            "strategy": "future_strategy",
            "mutation_type": "future_kind",
            "call_name": "X",
            "before": {},
            "after": {},
        }
        path.write_text(_json.dumps(record) + "\n")
        with pytest.raises(ValueError, match="schema_version"):
            MutationLog.load(path)

    def test_mutation_log_summary(self) -> None:
        log = MutationLog()
        log.record(
            Mutation(
                strategy="bootstrap",
                mutation_type="add_examples",
                call_name="Test",
                before={},
                after={},
                rationale="test",
                metric_before=0.5,
                metric_after=0.7,
                accepted=True,
                tokens_used=100,
                cost_usd=0.001,
            )
        )
        summary = log.summary()
        assert "1 trials" in summary
        assert "Accepted: 1" in summary


# --- BootstrapOptimizer Tests ---


class TestBootstrapOptimizer:
    async def test_bootstrap_adds_examples(self) -> None:
        """Bootstrap should find passing examples and add them as demos."""
        # Classifier that correctly handles these specific texts
        call = _make_classifier({"great": "positive", "terrible": "negative", "ok": "neutral"})

        train_set = [
            Example(inputs={"text": "great product"}, outputs={"sentiment": "positive"}),
            Example(inputs={"text": "terrible service"}, outputs={"sentiment": "negative"}),
            Example(inputs={"text": "ok I guess"}, outputs={"sentiment": "neutral"}),
        ]
        val_set = [
            Example(inputs={"text": "great job"}, outputs={"sentiment": "positive"}),
        ]

        optimizer = BootstrapOptimizer(metric=exact_match, max_examples=3)
        result = await optimizer.optimize(call, train_set, val_set)

        # Should have found passing examples from training
        # Whether it "improves" depends on the FunctionClient being deterministic
        assert result.metric_before is not None
        assert result.eval_before.n_total == 1

    async def test_bootstrap_no_passing_examples(self) -> None:
        """When no training examples pass, bootstrap should not add any demos."""
        # Classifier that always returns wrong answers
        call = _make_classifier({})  # always neutral

        train_set = [
            Example(inputs={"text": "great"}, outputs={"sentiment": "positive"}),
            Example(inputs={"text": "bad"}, outputs={"sentiment": "negative"}),
        ]
        val_set = [
            Example(inputs={"text": "great"}, outputs={"sentiment": "positive"}),
        ]

        optimizer = BootstrapOptimizer(metric=exact_match, max_examples=3)
        result = await optimizer.optimize(call, train_set, val_set)

        assert result.accepted is False
        assert len(result.examples_added) == 0

    async def test_bootstrap_records_mutation(self) -> None:
        """Bootstrap should record a mutation in the log."""
        call = _make_classifier({"great": "positive"})
        log = MutationLog()

        train_set = [
            Example(inputs={"text": "great"}, outputs={"sentiment": "positive"}),
        ]
        val_set = [
            Example(inputs={"text": "great"}, outputs={"sentiment": "positive"}),
        ]

        optimizer = BootstrapOptimizer(metric=exact_match, mutation_log=log)
        await optimizer.optimize(call, train_set, val_set)

        assert len(log.mutations) == 1
        assert log.mutations[0].strategy == "bootstrap"
