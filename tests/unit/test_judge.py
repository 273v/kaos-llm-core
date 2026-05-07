"""Tests for Judge — LLM-as-evaluator program."""

from __future__ import annotations

import json
from typing import Any

from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart

from kaos_llm_core.programs.judge import Judge
from kaos_llm_core.signatures import InputField, OutputField, Signature


class ExtractSig(Signature):
    """Extract entities from text."""

    text: str = InputField(description="Input text")
    entities: list[str] = OutputField(description="Entity names")


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


class TestJudge:
    async def test_judge_produces_and_evaluates(self) -> None:
        """Judge should call producer then evaluate with judge model."""
        call_count = {"produce": 0, "judge": 0}

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            # Inspect the system prompt to figure out which call this is
            system = messages[0]["content"] if messages else ""
            if "Extract entities" in system:
                call_count["produce"] += 1
                return _json_response({"entities": ["SEC", "Acme"]})
            else:
                call_count["judge"] += 1
                return _json_response(
                    {
                        "quality_score": 0.85,
                        "reasoning": "Good extraction, found both key entities.",
                    }
                )

        client = FunctionClient(function=fn)
        judge = Judge(
            ExtractSig,
            producer_model="function-test",
            judge_model="function-test",
            criteria="accuracy of entity extraction",
        )
        # Patch both calls' clients
        judge.produce._client = client
        judge.judge_call._client = client

        result = await judge(text="The SEC filed suit against Acme Corp.")

        assert result.output is not None
        assert result.output.entities == ["SEC", "Acme"]
        assert result.judgment.quality_score == 0.85
        assert "Good extraction" in result.judgment.reasoning
        assert call_count["produce"] == 1
        assert call_count["judge"] == 1

    async def test_judge_has_trace_tree(self) -> None:
        """Judge trace should have children from produce + judge calls."""

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            system = messages[0]["content"] if messages else ""
            if "Extract entities" in system:
                return _json_response({"entities": ["X"]})
            else:
                return _json_response({"quality_score": 0.9, "reasoning": "ok"})

        client = FunctionClient(function=fn)
        judge = Judge(
            ExtractSig,
            producer_model="function-test",
            judge_model="function-test",
        )
        judge.produce._client = client
        judge.judge_call._client = client

        invocation = await judge.invoke(text="test")

        trace = invocation.trace
        assert trace is not None
        assert len(trace.children) == 2

    def test_judge_is_a_program(self) -> None:
        judge = Judge(ExtractSig, producer_model="test", judge_model="test")
        from kaos_llm_core.programs.base import Program

        assert isinstance(judge, Program)

    def test_judge_named_calls(self) -> None:
        judge = Judge(ExtractSig, producer_model="test", judge_model="test")
        calls = judge.named_calls()
        assert "produce" in calls
        assert "judge_call" in calls

    def test_retry_produce_is_named_when_set(self) -> None:
        """Bug 9 regression: when ``retry_model`` is set, the retry Call must
        be a named attribute on the Judge so that ``Program.named_calls`` /
        ``Program.__call__`` collects its trace into ``last_trace.children``.

        Previously the retry Call was a local variable inside ``forward()``,
        so its trace was invisible to cost rollups — retry-model spend was
        silently missing from every Judge usage that triggered the retry path.
        """
        judge = Judge(
            ExtractSig,
            producer_model="primary",
            judge_model="judge",
            retry_model="strong-fallback",
        )
        calls = judge.named_calls()
        assert "retry_produce" in calls, (
            "Judge with retry_model must expose retry_produce as a named "
            "subcall so its trace gets captured by Program.__call__. "
            f"named_calls keys: {list(calls)}"
        )
        # And: when retry_model is NOT set, retry_produce stays None and is
        # NOT exposed (None values are filtered by Program.named_calls).
        no_retry_judge = Judge(ExtractSig, producer_model="primary", judge_model="judge")
        assert "retry_produce" not in no_retry_judge.named_calls()
        assert no_retry_judge.retry_produce is None

    async def test_score_response_uses_supplied_candidate(self) -> None:
        """Companion regression for the Judge refactor that fixed Bug 6.

        ``Judge.score_response`` must score the **supplied** response, not
        re-run the producer. We pass a known-distinct response object whose
        rendered JSON should appear in the judge_call's prompt; if Judge
        re-ran its producer instead, the prompt would not contain it.
        """
        seen_responses: list[str] = []

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            blob = " ".join(
                m.get("content", "") if isinstance(m.get("content", ""), str) else ""
                for m in messages
            )
            if "Evaluate the quality" in blob or "quality_score" in blob:
                seen_responses.append(blob)
                return _json_response({"quality_score": 0.42, "reasoning": "scored"})
            # Producer side — should NEVER be invoked by score_response.
            return _json_response({"entity": "should-not-appear", "level": "low"})

        client = FunctionClient(function=fn)
        judge = Judge(ExtractSig, producer_model="test", judge_model="test")
        judge.produce._client = client
        judge.judge_call._client = client

        # A simple object that renders into JSON containing a marker token
        # the judge prompt must contain.
        class FakeResponse:
            def model_dump(self) -> dict[str, Any]:
                return {"entity": "MARKER_TOKEN_42", "level": "high"}

        judged = await judge.score_response(FakeResponse(), text="anything")
        assert judged.judgment.quality_score == 0.42
        # The judge prompt should have seen the supplied response, not the
        # producer's.
        assert seen_responses, "judge_call was never invoked by score_response"
        assert any("MARKER_TOKEN_42" in r for r in seen_responses), (
            f"Judge prompt did not contain the supplied response marker. "
            f"seen prompts: {[r[:200] for r in seen_responses]}"
        )
