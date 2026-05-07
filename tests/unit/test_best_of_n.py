"""Tests for BestOfN -- sampling-based best-of-N selection program."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest
from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart

from kaos_llm_core.errors import CallError
from kaos_llm_core.programs.best_of_n import BestOfN, BestOfNResult
from kaos_llm_core.programs.call import Call
from kaos_llm_core.programs.judge import Judge
from kaos_llm_core.signatures import InputField, OutputField, Signature

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


class WriteSig(Signature):
    """Write a short answer."""

    topic: str = InputField(description="What to write about")
    answer: str = OutputField(description="The written answer")


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


def _producer_with_responses(answers: list[str]) -> Call:
    """Build a Call backed by a FunctionClient that returns the given answers in order."""
    iter_answers = iter(answers)

    def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
        return _json_response({"answer": next(iter_answers)})

    client = FunctionClient(function=fn)
    call = Call(WriteSig, model="function-test")
    call._client = client
    return call


# ----------------------------------------------------------------------
# Construction guards
# ----------------------------------------------------------------------


class TestBestOfNConstruction:
    def test_n_must_be_at_least_two(self) -> None:
        producer = _producer_with_responses(["x"])
        with pytest.raises(CallError, match="n >= 2"):
            BestOfN(producer, n=1, selector=lambda o, i: 1.0)

    def test_n_must_be_at_least_two_includes_recovery_guidance(self) -> None:
        producer = _producer_with_responses(["x"])
        with pytest.raises(CallError) as exc:
            BestOfN(producer, n=0, selector=lambda o, i: 1.0)
        msg = str(exc.value)
        assert "n=0" in msg
        # Recovery guidance: tell the user what to do instead.
        assert "call the producer directly" in msg

    def test_selector_required(self) -> None:
        producer = _producer_with_responses(["x"])
        with pytest.raises(CallError, match="explicit selector"):
            BestOfN(producer, n=3, selector=None)

    def test_selector_required_recovery_guidance(self) -> None:
        producer = _producer_with_responses(["x"])
        with pytest.raises(CallError) as exc:
            BestOfN(producer, n=3, selector=None)
        msg = str(exc.value)
        # Recovery guidance: how to fix it
        assert "Judge" in msg
        assert "lambda" in msg or "callable" in msg

    def test_seed_strategy_validated(self) -> None:
        producer = _producer_with_responses(["x"])
        # Pass via **kwargs so the literal type narrowing doesn't fire.
        bogus_kwargs: dict[str, Any] = {"seed_strategy": "bogus"}
        with pytest.raises(CallError, match="seed_strategy"):
            BestOfN(
                producer,
                n=3,
                selector=lambda o, i: 1.0,
                **bogus_kwargs,
            )


# ----------------------------------------------------------------------
# Selection mechanics
# ----------------------------------------------------------------------


class TestBestOfNSelection:
    async def test_metric_selector_picks_highest_score(self) -> None:
        """Three samples scored 0.3, 0.7, 0.5 -> selected_index=1."""
        producer = _producer_with_responses(["a", "b", "c"])
        scores_by_answer = {"a": 0.3, "b": 0.7, "c": 0.5}

        def metric(output: Any, inputs: dict[str, Any]) -> float:
            return scores_by_answer[output.answer]

        program = BestOfN(producer, n=3, selector=metric, seed_strategy="none")
        result = await program(topic="anything")

        assert isinstance(result, BestOfNResult)
        assert result.selection_method == "metric"
        assert len(result.candidates) == 3
        assert result.scores == [0.3, 0.7, 0.5]
        assert result.selected_index == 1
        assert result.outputs.answer == "b"

    async def test_judge_selector_picks_highest_score(self) -> None:
        """Three samples judged 0.4, 0.9, 0.6 -> selected_index=1."""
        producer = _producer_with_responses(["a", "b", "c"])

        # The Judge's produce call returns a placeholder; the judge_call returns
        # a deterministic score. Judge.forward() calls produce + judge_call per
        # candidate, so total judge invocations per BestOfN run = n.
        judge_scores = iter([0.4, 0.9, 0.6])

        def judge_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            system = messages[0]["content"] if messages else ""
            if "Evaluate the quality" in system or "quality_score" in system:
                return _json_response({"quality_score": next(judge_scores), "reasoning": "test"})
            return _json_response({"answer": "ignored-by-bestofn"})

        judge_client = FunctionClient(function=judge_fn)
        judge = Judge(
            WriteSig,
            producer_model="function-test",
            judge_model="function-test",
        )
        judge.produce._client = judge_client
        judge.judge_call._client = judge_client

        program = BestOfN(producer, n=3, selector=judge, seed_strategy="none")
        result = await program(topic="anything")

        assert result.selection_method == "judge"
        assert len(result.candidates) == 3
        assert result.scores == [0.4, 0.9, 0.6]
        assert result.selected_index == 1

    async def test_judge_selector_scores_in_order(self) -> None:
        """Verify that judge scoring uses the candidates in order.

        We use a counter that returns one score per *judge* call to make the
        ordering observable. ``Judge.forward()`` calls produce + judge_call,
        so each scoring round consumes two FunctionClient invocations.
        """
        producer = _producer_with_responses(["one", "two", "three"])

        # Track judge_call invocations only -- those are the actual scoring.
        judge_scores = iter([0.2, 0.95, 0.5])

        def judge_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            # Distinguish produce vs judge_call by inspecting the system prompt
            system = messages[0]["content"] if messages else ""
            if "Evaluate the quality" in system or "quality_score" in system:
                return _json_response({"quality_score": next(judge_scores), "reasoning": "test"})
            return _json_response({"answer": "ignored-by-bestofn"})

        judge_client = FunctionClient(function=judge_fn)
        judge = Judge(
            WriteSig,
            producer_model="function-test",
            judge_model="function-test",
        )
        judge.produce._client = judge_client
        judge.judge_call._client = judge_client

        program = BestOfN(producer, n=3, selector=judge, seed_strategy="none")
        result = await program(topic="anything")

        assert result.scores == [0.2, 0.95, 0.5]
        assert result.selected_index == 1

    async def test_n_equals_two(self) -> None:
        """Minimum n=2 works."""
        producer = _producer_with_responses(["x", "y"])

        def metric(output: Any, inputs: dict[str, Any]) -> float:
            return 0.1 if output.answer == "x" else 0.9

        program = BestOfN(producer, n=2, selector=metric, seed_strategy="none")
        result = await program(topic="t")

        assert len(result.candidates) == 2
        assert result.selected_index == 1
        assert result.outputs.answer == "y"

    async def test_judge_selector_actually_sees_candidate(self) -> None:
        """Bug 6 regression: BestOfN+Judge must score the SUPPLIED candidate.

        Previous behavior: ``_score_candidate`` called ``self.selector(**inputs)``,
        which invoked ``Judge.forward()``. Judge.forward then called its own
        producer with the original inputs, ignoring the candidate. Every
        candidate received the same score and selection collapsed to tie-break.

        This test makes the score depend on the *candidate content* the judge
        actually sees in its prompt. If the bug returns, every candidate scores
        the same and the assertion below fails.
        """
        producer = _producer_with_responses(["good", "bad", "average"])

        def judge_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            # Find the user message that carries the response under judgment.
            blob = " ".join(
                m.get("content", "") if isinstance(m.get("content", ""), str) else ""
                for m in messages
            )
            # Score is determined by which candidate text appears in the prompt.
            if '"answer": "good"' in blob or "'answer': 'good'" in blob:
                score = 0.95
            elif '"answer": "average"' in blob or "'answer': 'average'" in blob:
                score = 0.6
            elif '"answer": "bad"' in blob or "'answer': 'bad'" in blob:
                score = 0.1
            else:
                # Fallback: the judge never saw any of the candidates. This is
                # the bug path — the test will fail because every candidate
                # gets the same score and selected_index will be 0 by tie-break.
                score = 0.5
            return _json_response({"quality_score": score, "reasoning": "content-based"})

        judge_client = FunctionClient(function=judge_fn)
        judge = Judge(
            WriteSig,
            producer_model="function-test",
            judge_model="function-test",
        )
        judge.produce._client = judge_client
        judge.judge_call._client = judge_client

        program = BestOfN(producer, n=3, selector=judge, seed_strategy="none")
        result = await program(topic="anything")

        # The candidates should be scored by content. "good" -> 0.95, "average"
        # -> 0.6, "bad" -> 0.1. Selection picks "good" (index 0).
        assert result.candidates[0].answer == "good"
        assert result.candidates[1].answer == "bad"
        assert result.candidates[2].answer == "average"
        assert result.scores == [0.95, 0.1, 0.6]
        assert result.selected_index == 0
        assert result.outputs.answer == "good"


# ----------------------------------------------------------------------
# Failure handling
# ----------------------------------------------------------------------


class TestBestOfNFailures:
    async def test_some_samples_fail(self) -> None:
        """3 of 5 succeed, 2 raise -- only successful candidates are scored."""
        # Five calls: positions 0,2 fail; 1,3,4 succeed.
        successes = iter(["s1", "s2", "s3"])
        call_idx = {"i": 0}

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            i = call_idx["i"]
            call_idx["i"] += 1
            if i in (0, 2):
                raise RuntimeError(f"sample {i} failed")
            return _json_response({"answer": next(successes)})

        client = FunctionClient(function=fn)
        producer = Call(WriteSig, model="function-test", max_retries=0)
        producer._client = client

        # Metric assigns increasing scores so the LAST survivor wins.
        survivor_scores = iter([0.1, 0.5, 0.9])

        def metric(output: Any, inputs: dict[str, Any]) -> float:
            return next(survivor_scores)

        program = BestOfN(producer, n=5, selector=metric, seed_strategy="none")
        result = await program(topic="t")

        # 3 survivors out of 5
        assert len(result.candidates) == 3
        assert result.scores == [0.1, 0.5, 0.9]
        # selected_index references survivor list, not original positions
        assert result.selected_index == 2
        assert result.outputs.answer == "s3"

    async def test_all_samples_fail(self) -> None:
        """If every sample raises, BestOfN raises CallError chained from the last."""

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            raise RuntimeError("boom")

        client = FunctionClient(function=fn)
        producer = Call(WriteSig, model="function-test", max_retries=0)
        producer._client = client

        program = BestOfN(producer, n=3, selector=lambda o, i: 1.0, seed_strategy="none")
        with pytest.raises(CallError) as exc:
            await program(topic="t")

        msg = str(exc.value)
        assert "all 3 samples failed" in msg
        assert "boom" in msg
        # Has a __cause__ chain so the original exception isn't lost.
        assert exc.value.__cause__ is not None


# ----------------------------------------------------------------------
# Hyperparameter immutability
# ----------------------------------------------------------------------


class TestBestOfNHyperparameters:
    async def test_temperature_strategy_does_not_mutate_producer(self) -> None:
        """The producer's hyperparameters must be unchanged after a run."""
        producer = _producer_with_responses(["a", "b"])
        producer._kwargs = {"temperature": 0.0}  # caller pinned a low value
        original_kwargs_id = id(producer._kwargs)
        original_kwargs_snapshot = dict(producer._kwargs)

        program = BestOfN(
            producer,
            n=2,
            selector=lambda o, i: 0.5,
            seed_strategy="temperature",
        )
        await program(topic="t")

        # The dict object identity is preserved AND its contents are unchanged.
        assert id(producer._kwargs) == original_kwargs_id
        assert producer._kwargs == original_kwargs_snapshot
        assert producer._kwargs["temperature"] == 0.0

    async def test_temperature_strategy_overrides_low_temperature(self) -> None:
        """Verify the transient clone actually receives temperature=1.0."""
        seen_temperatures: list[float | None] = []

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"answer": "ok"})

        client = FunctionClient(function=fn)
        producer = Call(WriteSig, model="function-test")
        producer._client = client
        producer._kwargs = {"temperature": 0.2}

        # Patch _prepare_call_kwargs to record what each clone uses.
        # Easier path: inspect via a wrapper around _run_sample.
        program = BestOfN(
            producer,
            n=3,
            selector=lambda o, i: 0.5,
            seed_strategy="temperature",
        )

        original_run = program._run_sample

        async def spy(i: int, inputs: dict[str, Any]) -> Any:
            kwargs = program._kwargs_for_sample(i)
            seen_temperatures.append(kwargs.get("temperature"))
            return await original_run(i, inputs)

        # setattr to bypass static method-assign checking
        object.__setattr__(program, "_run_sample", spy)

        await program(topic="t")

        assert seen_temperatures == [1.0, 1.0, 1.0]
        # Original is still unchanged
        assert producer._kwargs["temperature"] == 0.2

    async def test_explicit_seed_strategy_passes_seeds(self) -> None:
        """seed_strategy='explicit_seed' assigns seed=i+1 for i in range(n).

        Audit finding #23: seeds start at 1, not 0, because OpenAI treats
        ``seed=0`` as "unset" so sample 0 with the legacy ``seed=i`` strategy
        got no seed at all and silently fell back to provider non-determinism.
        """
        producer = _producer_with_responses(["a", "b", "c"])

        program = BestOfN(
            producer,
            n=3,
            selector=lambda o, i: 0.5,
            seed_strategy="explicit_seed",
        )
        # Inspect the kwargs builder directly -- no need to trace through asyncio.
        assert program._kwargs_for_sample(0) == {"seed": 1}
        assert program._kwargs_for_sample(1) == {"seed": 2}
        assert program._kwargs_for_sample(2) == {"seed": 3}
        # And the producer is still unmutated
        assert producer._kwargs == {}

    async def test_none_strategy_passes_kwargs_unchanged(self) -> None:
        """seed_strategy='none' makes no changes to kwargs."""
        producer = _producer_with_responses(["a", "b"])
        producer._kwargs = {"top_p": 0.95}

        program = BestOfN(
            producer,
            n=2,
            selector=lambda o, i: 0.5,
            seed_strategy="none",
        )
        # Should be a snapshot copy, not the same dict
        sample0 = program._kwargs_for_sample(0)
        assert sample0 == {"top_p": 0.95}
        assert sample0 is not producer._kwargs


# ----------------------------------------------------------------------
# Result attribute access
# ----------------------------------------------------------------------


class TestBestOfNResultAccess:
    async def test_attribute_access_forwards_to_outputs(self) -> None:
        """`result.field_name` should resolve to `result.outputs.field_name`."""
        producer = _producer_with_responses(["hello", "world"])

        def metric(output: Any, inputs: dict[str, Any]) -> float:
            return 0.0 if output.answer == "hello" else 1.0

        program = BestOfN(producer, n=2, selector=metric, seed_strategy="none")
        result = await program(topic="t")

        # Both spellings work
        assert result.outputs.answer == "world"
        assert result.answer == "world"  # forwarded via __getattr__

    async def test_attribute_access_raises_on_missing(self) -> None:
        producer = _producer_with_responses(["x", "y"])
        program = BestOfN(producer, n=2, selector=lambda o, i: 0.5, seed_strategy="none")
        result = await program(topic="t")

        with pytest.raises(AttributeError):
            _ = result.nonexistent_field


# ----------------------------------------------------------------------
# Parallel execution
# ----------------------------------------------------------------------


class TestBestOfNParallel:
    async def test_samples_run_in_parallel(self) -> None:
        """Samples must run concurrently, not sequentially.

        We instrument the FunctionClient with an async sleep so each call
        takes ~50ms wall time. With n=5, sequential execution would take
        ~250ms. Parallel execution should finish in ~50ms (plus overhead).
        We allow generous slack for CI flakiness.
        """
        sleep_per_call = 0.05  # 50ms

        async def slow_fn(
            messages: list[dict[str, Any]], profile: ModelProfile
        ) -> ProviderResponse:
            await asyncio.sleep(sleep_per_call)
            return _json_response({"answer": "ok"})

        client = FunctionClient(function=slow_fn)
        producer = Call(WriteSig, model="function-test")
        producer._client = client

        program = BestOfN(
            producer,
            n=5,
            selector=lambda o, i: 0.5,
            seed_strategy="none",
        )

        start = time.monotonic()
        result = await program(topic="t")
        elapsed = time.monotonic() - start

        assert len(result.candidates) == 5
        # Sequential would be 5 * 50ms = 250ms.
        # Parallel should be ~50ms. Allow 150ms ceiling for slow CI.
        assert elapsed < 0.15, (
            f"Expected parallel execution (~50ms) but took {elapsed * 1000:.0f}ms; "
            f"sampling appears to be sequential."
        )

    async def test_uses_asyncio_gather(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify the implementation actually calls asyncio.gather (not a loop)."""
        producer = _producer_with_responses(["a", "b", "c"])

        gather_calls: list[int] = []
        original_gather = asyncio.gather

        async def spy_gather(*aws: Any, **kwargs: Any) -> Any:
            gather_calls.append(len(aws))
            return await original_gather(*aws, **kwargs)

        monkeypatch.setattr("kaos_llm_core.programs.best_of_n.asyncio.gather", spy_gather)

        program = BestOfN(
            producer,
            n=3,
            selector=lambda o, i: 0.5,
            seed_strategy="none",
        )
        await program(topic="t")

        # First gather call must hold all n samples.
        assert gather_calls, "asyncio.gather was never called"
        assert gather_calls[0] == 3


# ----------------------------------------------------------------------
# Optimizer / trace integration
# ----------------------------------------------------------------------


class TestBestOfNOptimizerIntegration:
    def test_named_calls_includes_producer(self) -> None:
        producer = _producer_with_responses(["x", "y"])
        program = BestOfN(producer, n=2, selector=lambda o, i: 0.5)
        names = program.named_calls()
        assert "producer" in names
        assert names["producer"] is producer

    def test_named_calls_includes_judge_selector(self) -> None:
        producer = _producer_with_responses(["x", "y"])
        judge = Judge(
            WriteSig,
            producer_model="function-test",
            judge_model="function-test",
        )
        program = BestOfN(producer, n=2, selector=judge)
        names = program.named_calls()
        assert "producer" in names
        assert "selector" in names
        assert names["selector"] is judge
