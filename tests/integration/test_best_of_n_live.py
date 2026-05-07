"""Live integration tests for BestOfN.

One creative-writing test per provider. Producer writes a one-sentence
story; a Judge scores creativity; BestOfN samples 3 times and picks the
best. Skipped automatically when the corresponding API key is missing.
"""

from __future__ import annotations

import pytest

from kaos_llm_core.programs.best_of_n import BestOfN, BestOfNResult
from kaos_llm_core.programs.call import Call
from kaos_llm_core.programs.judge import Judge
from kaos_llm_core.signatures import InputField, OutputField, Signature

from .conftest import requires_anthropic, requires_google, requires_openai


class WriteStory(Signature):
    """Write a single creative one-sentence story about the given topic."""

    topic: str = InputField(description="The topic of the story")
    story: str = OutputField(description="A single-sentence creative story")


def _check_result(result: BestOfNResult, n: int) -> None:
    assert isinstance(result, BestOfNResult)
    assert len(result.candidates) == n
    assert len(result.scores) == n
    assert 0 <= result.selected_index < n
    assert result.selection_method == "judge"

    # The selected score should be the maximum of the scores list.
    assert result.scores[result.selected_index] == max(result.scores)

    # The story field should be reachable via attribute forwarding.
    selected_story = result.story
    assert isinstance(selected_story, str)
    assert len(selected_story) > 0

    # All scores should be plausible (Judge returns 0.0..1.0).
    for s in result.scores:
        assert 0.0 <= s <= 1.0, f"Score out of range: {s}"


@requires_anthropic
@pytest.mark.integration
async def test_best_of_n_anthropic_creativity() -> None:
    producer = Call(WriteStory, model="anthropic:claude-haiku-4-5")
    judge = Judge(
        WriteStory,
        producer_model="anthropic:claude-haiku-4-5",
        judge_model="anthropic:claude-haiku-4-5",
        criteria="creativity, originality, and vivid imagery in a one-sentence story",
    )
    program = BestOfN(producer, n=3, selector=judge, seed_strategy="temperature")
    result = await program(topic="a robot learning to paint")
    _check_result(result, n=3)


@requires_openai
@pytest.mark.integration
async def test_best_of_n_openai_creativity() -> None:
    producer = Call(WriteStory, model="openai:gpt-5.4-nano")
    judge = Judge(
        WriteStory,
        producer_model="openai:gpt-5.4-nano",
        judge_model="openai:gpt-5.4-nano",
        criteria="creativity, originality, and vivid imagery in a one-sentence story",
    )
    program = BestOfN(producer, n=3, selector=judge, seed_strategy="temperature")
    result = await program(topic="a lighthouse keeper's last night")
    _check_result(result, n=3)


@requires_google
@pytest.mark.integration
async def test_best_of_n_google_creativity() -> None:
    producer = Call(WriteStory, model="google:gemini-2.5-flash")
    judge = Judge(
        WriteStory,
        producer_model="google:gemini-2.5-flash",
        judge_model="google:gemini-2.5-flash",
        criteria="creativity, originality, and vivid imagery in a one-sentence story",
    )
    program = BestOfN(producer, n=3, selector=judge, seed_strategy="temperature")
    result = await program(topic="a city of clockwork birds")
    _check_result(result, n=3)
