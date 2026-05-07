"""Live integration tests for Refine.

One test per provider, skipped when API keys are absent. Each test wires
up a real haiku-writing producer and a real haiku-scoring judge, runs the
Refine loop, and asserts that:

- the loop produced at least one iteration
- ``final_score`` is a sensible float in [0, 1]
- the wrapped output has a non-empty ``haiku`` field

We use the cheapest current-generation model for each provider per
``kaos-llm-client/tests/integration/test_live.py``.
"""

from __future__ import annotations

import pytest

from kaos_llm_core.programs.call import Call
from kaos_llm_core.programs.judge import Judge
from kaos_llm_core.programs.refine import (
    STOP_MAX_ITERATIONS,
    STOP_QUALITY_MET,
    Refine,
)
from kaos_llm_core.signatures import InputField, OutputField, Signature

from .conftest import requires_anthropic, requires_google, requires_openai


class WriteHaiku(Signature):
    """Write a haiku about the given topic. Three lines, 5-7-5 syllables."""

    topic: str = InputField(description="The subject of the haiku")
    haiku: str = OutputField(description="A haiku, three lines separated by newlines")


def _build_refine(producer_model: str, judge_model: str) -> Refine:
    producer = Call(WriteHaiku, model=producer_model)
    judge = Judge(
        WriteHaiku,
        producer_model=judge_model,
        judge_model=judge_model,
        criteria="poetic imagery, syllable structure (5-7-5), evocative language",
    )
    return Refine(producer, judge, max_iterations=2, min_score=0.85)


def _assert_refine_result(result, expected_iters_max: int) -> None:
    assert result is not None
    assert 1 <= result.iterations <= expected_iters_max
    assert 0.0 <= result.final_score <= 1.0
    assert result.stop_reason in (STOP_QUALITY_MET, STOP_MAX_ITERATIONS)
    # Forwarded attribute access
    assert isinstance(result.haiku, str)
    assert len(result.haiku.strip()) > 0
    # History should match iteration count
    assert len(result.history) == result.iterations


@requires_anthropic
class TestRefineAnthropicLive:
    @pytest.mark.integration
    async def test_refine_haiku_anthropic(self) -> None:
        refine = _build_refine(
            producer_model="anthropic:claude-haiku-4-5",
            judge_model="anthropic:claude-haiku-4-5",
        )
        invocation = await refine.invoke(topic="autumn rain on cedar shingles")
        result = invocation.output
        _assert_refine_result(result, expected_iters_max=2)

        # Trace should have one child per iteration
        trace = invocation.trace
        assert trace is not None
        assert len(trace.children) == result.iterations


@requires_openai
class TestRefineOpenAILive:
    @pytest.mark.integration
    async def test_refine_haiku_openai(self) -> None:
        refine = _build_refine(
            producer_model="openai:gpt-5.4-nano",
            judge_model="openai:gpt-5.4-nano",
        )
        result = await refine(topic="midnight tide on a basalt shore")
        _assert_refine_result(result, expected_iters_max=2)


@requires_google
class TestRefineGoogleLive:
    @pytest.mark.integration
    async def test_refine_haiku_google(self) -> None:
        refine = _build_refine(
            producer_model="google:gemini-2.5-flash",
            judge_model="google:gemini-2.5-flash",
        )
        result = await refine(topic="snow falling on a bamboo grove")
        _assert_refine_result(result, expected_iters_max=2)
