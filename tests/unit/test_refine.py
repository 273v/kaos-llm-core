"""Tests for Refine — inline iterative refinement program.

These tests use ``FunctionClient`` from kaos-llm-client to deterministically
simulate producer and judge responses without hitting any real LLM API.
The producer is a single ``Call`` with the standard pipeline; the judge is
a real :class:`Judge` instance whose internal ``judge_call`` is what
``Refine`` actually invokes.
"""

from __future__ import annotations

import json
from typing import Any

from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart

from kaos_llm_core.programs.call import Call
from kaos_llm_core.programs.judge import Judge
from kaos_llm_core.programs.refine import (
    STOP_MAX_ITERATIONS,
    STOP_QUALITY_MET,
    Refine,
    RefineResult,
    _build_refined_instructions,
)
from kaos_llm_core.signatures import InputField, OutputField, Signature

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class WriteHaiku(Signature):
    """Write a haiku about a topic."""

    topic: str = InputField(description="The subject of the haiku")
    haiku: str = OutputField(description="A haiku, three lines, 5-7-5 syllables")


class CritiqueSig(Signature):
    """Score a haiku and explain what to improve."""

    original_input: str = InputField(description="The original input text")
    task_description: str = InputField(description="What the task was supposed to accomplish")
    response: str = InputField(description="The response to evaluate")
    criteria: str = InputField(description="Evaluation criteria")
    quality_score: float = OutputField(description="Score 0-1")
    critique: str = OutputField(description="Free-text critique")


def _json_response(data: dict[str, Any]) -> ProviderResponse:
    return ProviderResponse.model_construct(
        provider="function",
        model="function-test",
        raw={},
        parts=[ContentPart.model_construct(type="text", text=json.dumps(data))],
        usage=UsageInfo.model_construct(input_tokens=10, output_tokens=5, total_tokens=15),
        stop_reason="end_turn",
        status_code=200,
        response_headers={},
    )


def _make_refine(
    producer_fn: Any,
    judge_fn: Any,
    *,
    max_iterations: int = 3,
    min_score: float = 0.8,
    feedback_field: str = "feedback",
) -> tuple[Refine, FunctionClient, FunctionClient]:
    """Build a Refine wired to two FunctionClients (producer + judge)."""
    producer_client = FunctionClient(function=producer_fn)
    judge_client = FunctionClient(function=judge_fn)

    producer = Call(WriteHaiku, model="function-test")
    producer._client = producer_client

    judge = Judge(
        WriteHaiku,
        producer_model="function-test",
        judge_model="function-test",
        criteria="poetic quality, imagery, syllable structure",
    )
    # Patch the judge's internal scoring Call (the only one Refine invokes)
    judge.judge_call._client = judge_client

    refine = Refine(
        producer,
        judge,
        max_iterations=max_iterations,
        min_score=min_score,
        feedback_field=feedback_field,
    )
    return refine, producer_client, judge_client


# ---------------------------------------------------------------------------
# 1. First attempt passes
# ---------------------------------------------------------------------------


class TestRefine:
    async def test_first_attempt_passes(self) -> None:
        """Producer returns good output on the first try; Refine returns immediately."""

        def producer_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response(
                {"haiku": "autumn leaves drifting / silently to the cold ground / winter is coming"}
            )

        def judge_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"quality_score": 0.95, "reasoning": "great"})

        refine, producer_client, judge_client = _make_refine(producer_fn, judge_fn, min_score=0.8)
        result = await refine(topic="autumn")

        assert isinstance(result, RefineResult)
        assert result.iterations == 1
        assert result.stop_reason == STOP_QUALITY_MET
        assert result.final_score == 0.95
        assert len(result.history) == 1
        assert len(producer_client.call_history) == 1
        assert len(judge_client.call_history) == 1

    # ------------------------------------------------------------------
    # 2. Refines once and passes
    # ------------------------------------------------------------------

    async def test_refines_once_and_passes(self) -> None:
        """Producer iteration 1 = mediocre; iteration 2 = great. iterations==2."""
        producer_count = {"n": 0}
        judge_count = {"n": 0}

        def producer_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            producer_count["n"] += 1
            if producer_count["n"] == 1:
                return _json_response({"haiku": "leaves fall / cold / done"})
            return _json_response(
                {"haiku": "crimson maple leaves / spiraling on autumn wind / silence at twilight"}
            )

        def judge_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            judge_count["n"] += 1
            if judge_count["n"] == 1:
                return _json_response(
                    {"quality_score": 0.6, "reasoning": "lacks imagery and syllable structure"}
                )
            return _json_response({"quality_score": 0.9, "reasoning": "much better"})

        refine, _producer_client, _judge_client = _make_refine(producer_fn, judge_fn, min_score=0.8)
        result = await refine(topic="autumn")

        assert result.iterations == 2
        assert result.stop_reason == STOP_QUALITY_MET
        assert result.final_score == 0.9
        assert len(result.history) == 2
        assert producer_count["n"] == 2
        assert judge_count["n"] == 2

    # ------------------------------------------------------------------
    # 3. Hits max iterations
    # ------------------------------------------------------------------

    async def test_hits_max_iterations(self) -> None:
        """Judge always returns 0.5; Refine exhausts max_iterations and returns best."""

        def producer_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"haiku": "still mediocre"})

        def judge_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"quality_score": 0.5, "reasoning": "meh"})

        refine, producer_client, judge_client = _make_refine(
            producer_fn, judge_fn, max_iterations=3, min_score=0.8
        )
        result = await refine(topic="autumn")

        assert result.iterations == 3
        assert result.stop_reason == STOP_MAX_ITERATIONS
        assert result.final_score == 0.5
        assert len(result.history) == 3
        assert len(producer_client.call_history) == 3
        assert len(judge_client.call_history) == 3

    # ------------------------------------------------------------------
    # 4. Returns best when overrun
    # ------------------------------------------------------------------

    async def test_returns_best_when_overrun(self) -> None:
        """Iteration 2 has the highest score; Refine returns iteration 2's output."""
        producer_count = {"n": 0}
        judge_count = {"n": 0}

        # Three distinct producer outputs we can identify
        producer_outputs = [
            "first attempt — bad",
            "SECOND ATTEMPT — BEST",  # iteration 2
            "third attempt — worse",
        ]
        # Scores: iter1=0.4, iter2=0.7 (highest), iter3=0.5
        judge_scores = [0.4, 0.7, 0.5]

        def producer_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            i = producer_count["n"]
            producer_count["n"] += 1
            return _json_response({"haiku": producer_outputs[i]})

        def judge_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            i = judge_count["n"]
            judge_count["n"] += 1
            return _json_response({"quality_score": judge_scores[i], "reasoning": f"iter {i + 1}"})

        # min_score=0.95 forces all three iterations to run
        refine, _, _ = _make_refine(producer_fn, judge_fn, max_iterations=3, min_score=0.95)
        result = await refine(topic="autumn")

        assert result.stop_reason == STOP_MAX_ITERATIONS
        assert result.iterations == 3
        # Returns iteration 2's output, not iteration 3's
        assert result.outputs.haiku == "SECOND ATTEMPT — BEST"
        assert result.final_score == 0.7
        # And the history still shows all three
        assert [h.score for h in result.history] == [0.4, 0.7, 0.5]

    # ------------------------------------------------------------------
    # 5. result.field forwards to outputs
    # ------------------------------------------------------------------

    async def test_attribute_access_forwards_to_outputs(self) -> None:
        """``result.haiku`` should forward to ``result.outputs.haiku``."""

        def producer_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response(
                {"haiku": "the test passes / quietly without a sound / the assert is green"}
            )

        def judge_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"quality_score": 1.0, "reasoning": "perfect"})

        refine, _, _ = _make_refine(producer_fn, judge_fn)
        result = await refine(topic="testing")

        # Forwarded attribute access
        assert result.haiku == result.outputs.haiku
        assert "test passes" in result.haiku

    # ------------------------------------------------------------------
    # 6. History length matches iterations
    # ------------------------------------------------------------------

    async def test_history_length_matches_iterations(self) -> None:
        """For both early-stop and overrun, len(history) == iterations."""

        # Early-stop case
        def producer_fn_pass(
            messages: list[dict[str, Any]], profile: ModelProfile
        ) -> ProviderResponse:
            return _json_response({"haiku": "ok"})

        def judge_fn_pass(
            messages: list[dict[str, Any]], profile: ModelProfile
        ) -> ProviderResponse:
            return _json_response({"quality_score": 0.95, "reasoning": ""})

        refine_pass, _, _ = _make_refine(producer_fn_pass, judge_fn_pass)
        r1 = await refine_pass(topic="x")
        assert len(r1.history) == r1.iterations == 1

        # Overrun case
        def producer_fn_low(
            messages: list[dict[str, Any]], profile: ModelProfile
        ) -> ProviderResponse:
            return _json_response({"haiku": "ok"})

        def judge_fn_low(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"quality_score": 0.1, "reasoning": ""})

        refine_low, _, _ = _make_refine(producer_fn_low, judge_fn_low, max_iterations=4)
        r2 = await refine_low(topic="x")
        assert len(r2.history) == r2.iterations == 4

    # ------------------------------------------------------------------
    # 7. Trace structure
    # ------------------------------------------------------------------

    async def test_trace_has_one_child_per_iteration(self) -> None:
        """``result.trace.children`` has one entry per iteration; each has 2 children."""
        producer_count = {"n": 0}

        def producer_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            producer_count["n"] += 1
            return _json_response({"haiku": f"iter {producer_count['n']}"})

        def judge_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"quality_score": 0.3, "reasoning": "low"})

        refine, _, _ = _make_refine(producer_fn, judge_fn, max_iterations=3, min_score=0.9)
        invocation = await refine.invoke(topic="x")

        trace = invocation.trace
        assert trace is not None
        assert len(trace.children) == 3, (
            f"Expected one trace child per iteration, got {len(trace.children)}"
        )
        # Each iteration trace should have producer + judge as siblings
        for i, child in enumerate(trace.children, start=1):
            assert child.call_name == f"Refine.iter_{i}"
            assert len(child.children) == 2, (
                f"Iteration {i} expected 2 sub-traces (producer + judge), got {len(child.children)}"
            )

    async def test_trace_quality_met_short_circuit(self) -> None:
        """When QUALITY_MET fires on iteration 1 the trace has exactly one child."""

        def producer_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"haiku": "good"})

        def judge_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"quality_score": 0.99, "reasoning": ""})

        refine, _, _ = _make_refine(producer_fn, judge_fn, max_iterations=5)
        invocation = await refine.invoke(topic="x")

        trace = invocation.trace
        assert trace is not None
        assert len(trace.children) == 1

    # ------------------------------------------------------------------
    # 8. Custom feedback_field
    # ------------------------------------------------------------------

    async def test_custom_feedback_field(self) -> None:
        """``feedback_field='critique'`` should make Refine read judgment.critique."""
        feedback_seen: list[str] = []
        producer_count = {"n": 0}

        def producer_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            producer_count["n"] += 1
            return _json_response({"haiku": f"iter {producer_count['n']}"})

        def judge_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response(
                {
                    "quality_score": 0.2,
                    "critique": "the imagery is weak; add concrete sensory detail",
                    "reasoning": "should never be used",
                }
            )

        # Build a Judge whose judge_call uses CritiqueSig (with `critique` field)
        producer_client = FunctionClient(function=producer_fn)
        judge_client = FunctionClient(function=judge_fn)

        producer = Call(WriteHaiku, model="function-test")
        producer._client = producer_client

        judge = Judge(
            WriteHaiku,
            producer_model="function-test",
            judge_model="function-test",
            criteria="poetic quality",
        )
        # Replace the standard JudgmentSignature judge_call with one using CritiqueSig
        judge.judge_call = Call(CritiqueSig, model="function-test")
        judge.judge_call._client = judge_client

        refine = Refine(
            producer,
            judge,
            max_iterations=2,
            min_score=0.95,
            feedback_field="critique",
        )

        result = await refine(topic="autumn")
        feedback_seen.extend(h.feedback for h in result.history)

        # The feedback recorded in history should be from the `critique` field
        assert all("imagery" in fb for fb in feedback_seen), feedback_seen

    # ------------------------------------------------------------------
    # 9. Feedback is injected into the next producer call
    # ------------------------------------------------------------------

    async def test_feedback_injected_into_next_producer_call(self) -> None:
        """Inspect the second producer call's messages — previous output and
        the judge's feedback should both appear somewhere in the prompt."""
        producer_count = {"n": 0}

        def producer_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            producer_count["n"] += 1
            return _json_response({"haiku": f"PRIOR_OUTPUT_{producer_count['n']}"})

        def judge_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response(
                {
                    "quality_score": 0.3,
                    "reasoning": "VERY_SPECIFIC_CRITIQUE_TOKEN: needs more vivid imagery",
                }
            )

        # The standard JudgmentSignature exposes critique text on the
        # ``reasoning`` field, so configure Refine to read from there.
        refine, producer_client, _ = _make_refine(
            producer_fn,
            judge_fn,
            max_iterations=2,
            min_score=0.9,
            feedback_field="reasoning",
        )
        await refine(topic="autumn")

        assert len(producer_client.call_history) == 2
        # Inspect the SECOND producer call's messages.
        second_messages, _ = producer_client.call_history[1]
        rendered = json.dumps(second_messages, default=str)

        assert "PRIOR_OUTPUT_1" in rendered, (
            "Expected previous output text in second producer call's messages, "
            f"got: {rendered[:500]}"
        )
        assert "VERY_SPECIFIC_CRITIQUE_TOKEN" in rendered, (
            "Expected judge critique text in second producer call's messages, "
            f"got: {rendered[:500]}"
        )

    async def test_default_feedback_field_matches_judgment_signature(self) -> None:
        """Bug 8 regression: ``Refine(producer, Judge(...))`` with NO explicit
        ``feedback_field`` must auto-detect the right field from the judge's
        actual signature, not silently default to a non-existent ``"feedback"``.

        Previous behavior: the default was the literal string ``"feedback"``,
        but ``JudgmentSignature`` exposes ``reasoning`` (not ``feedback``). So
        ``getattr(judgment, "feedback", "")`` always returned ``""`` and Refine
        injected an empty critique on every iteration. The MCP wrapper at
        ``tools.py`` worked around this by explicitly passing
        ``feedback_field="reasoning"``, which confirms the mismatch.

        This test constructs Refine the natural Python-API way (no MCP, no
        explicit feedback_field) and asserts the second producer call's prompt
        contains the judge's critique text.
        """
        producer_count = {"n": 0}

        def producer_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            producer_count["n"] += 1
            return _json_response({"haiku": f"DRAFT_{producer_count['n']}"})

        def judge_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response(
                {
                    "quality_score": 0.3,
                    "reasoning": "AUTO_DETECT_CRITIQUE_TOKEN: be more specific",
                }
            )

        producer_client = FunctionClient(function=producer_fn)
        judge_client = FunctionClient(function=judge_fn)

        producer = Call(WriteHaiku, model="function-test")
        producer._client = producer_client
        judge = Judge(
            WriteHaiku,
            producer_model="function-test",
            judge_model="function-test",
        )
        judge.judge_call._client = judge_client

        # NOTE: NO feedback_field argument here. This is the user-facing default
        # path that was silently broken.
        refine = Refine(producer, judge, max_iterations=2, min_score=0.9)
        result = await refine(topic="autumn")

        # The auto-resolved field should be 'reasoning' (the field
        # JudgmentSignature actually exposes).
        assert refine._feedback_field == "reasoning"
        # The history should carry non-empty feedback for at least the first
        # iteration. (The second iteration is the final one and may not see
        # feedback injected because the loop exits before iteration 3.)
        assert any(h.feedback for h in result.history), (
            "Refine history has no non-empty feedback. The auto-detect of "
            "feedback_field is broken; Refine is silently injecting empty "
            "critiques on every iteration."
        )
        assert "AUTO_DETECT_CRITIQUE_TOKEN" in result.history[0].feedback

        # Inspect the second producer call's messages — the critique token
        # must appear, proving feedback actually flowed into the next prompt.
        assert len(producer_client.call_history) == 2
        second_messages, _ = producer_client.call_history[1]
        rendered = json.dumps(second_messages, default=str)
        assert "AUTO_DETECT_CRITIQUE_TOKEN" in rendered, (
            "Auto-detected feedback never made it into the second producer "
            "call's prompt — the Bug 8 regression is back."
        )

    async def test_producer_instructions_restored_after_run(self) -> None:
        """The producer's instructions must not leak across calls."""
        producer_count = {"n": 0}

        def producer_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            producer_count["n"] += 1
            return _json_response({"haiku": f"iter {producer_count['n']}"})

        def judge_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"quality_score": 0.1, "reasoning": "low"})

        refine, _, _ = _make_refine(producer_fn, judge_fn, max_iterations=3, min_score=0.9)
        original = refine.producer.instructions
        await refine(topic="autumn")
        assert refine.producer.instructions == original


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


class TestRefineConstruction:
    def test_max_iterations_must_be_positive(self) -> None:
        from kaos_llm_core.errors import CallError

        producer = Call(WriteHaiku, model="function-test")
        judge = Judge(WriteHaiku, producer_model="function-test", judge_model="function-test")
        try:
            Refine(producer, judge, max_iterations=0)
        except CallError as e:
            assert "max_iterations" in str(e)
        else:
            raise AssertionError("expected CallError")

    def test_min_score_must_be_in_unit_interval(self) -> None:
        from kaos_llm_core.errors import CallError

        producer = Call(WriteHaiku, model="function-test")
        judge = Judge(WriteHaiku, producer_model="function-test", judge_model="function-test")
        for bad in (-0.1, 1.5):
            try:
                Refine(producer, judge, min_score=bad)
            except CallError as e:
                assert "min_score" in str(e)
            else:
                raise AssertionError(f"expected CallError for min_score={bad}")

    def test_named_calls_exposes_producer_and_judge(self) -> None:
        producer = Call(WriteHaiku, model="function-test")
        judge = Judge(WriteHaiku, producer_model="function-test", judge_model="function-test")
        refine = Refine(producer, judge)
        names = refine.named_calls()
        assert "producer" in names
        assert "judge" in names


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


class TestBuildRefinedInstructions:
    def test_includes_critique_and_previous_output_pydantic(self) -> None:
        """Pydantic outputs render via model_dump()."""
        from pydantic import BaseModel

        class FakeOutput(BaseModel):
            haiku: str

        prev = FakeOutput(haiku="leaves fall")
        text = _build_refined_instructions(
            "Write a haiku.",
            previous_output=prev,
            critique="Add more vivid imagery.",
        )
        assert "Write a haiku." in text
        assert "leaves fall" in text
        assert "vivid imagery" in text
        assert "Refinement context" in text

    def test_includes_critique_and_previous_output_plain_object(self) -> None:
        """Plain Python objects render via vars()."""

        class FakeOutput:
            def __init__(self, haiku: str) -> None:
                self.haiku = haiku

        prev = FakeOutput("leaves fall")
        text = _build_refined_instructions(
            "Write a haiku.",
            previous_output=prev,
            critique="Add more vivid imagery.",
        )
        assert "leaves fall" in text
        assert "vivid imagery" in text

    def test_falls_back_to_repr_when_no_instance_attrs(self) -> None:
        """Slot-only or class-only objects fall back to repr()."""

        class _Slots:
            __slots__ = ()

            def __repr__(self) -> str:
                return "REPR_TOKEN"

        text = _build_refined_instructions(
            "Base.",
            previous_output=_Slots(),
            critique="critique text",
        )
        assert "REPR_TOKEN" in text
        assert "critique text" in text
