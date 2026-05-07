"""Live integration tests for the LLMJudge metric.

One helpfulness-rubric scoring per provider, using the cheapest current
generation model. Skipped automatically when the corresponding API key is
missing.
"""

from __future__ import annotations

import pytest

from kaos_llm_core.metrics import LLMJudge

from .conftest import requires_anthropic, requires_google, requires_openai


@requires_anthropic
@pytest.mark.asyncio
async def test_llm_judge_anthropic() -> None:
    judge = LLMJudge(model="anthropic:claude-haiku-4-5", rubric="helpfulness")
    score = await judge.acall(
        prediction="The capital of France is Paris.",
        gold="What is the capital of France?",
    )
    assert 0.0 <= score <= 1.0
    # A correct, helpful answer should score above the floor
    assert score >= 0.5


@requires_openai
@pytest.mark.asyncio
async def test_llm_judge_openai() -> None:
    judge = LLMJudge(model="openai:gpt-5.4-nano", rubric="helpfulness")
    score = await judge.acall(
        prediction="Paris is the capital of France.",
        gold="What is the capital of France?",
    )
    assert 0.0 <= score <= 1.0
    assert score >= 0.5


@requires_google
@pytest.mark.asyncio
async def test_llm_judge_google() -> None:
    judge = LLMJudge(model="google:gemini-2.5-flash", rubric="helpfulness")
    score = await judge.acall(
        prediction="Paris.",
        gold="What is the capital of France?",
    )
    assert 0.0 <= score <= 1.0
    assert score >= 0.5
