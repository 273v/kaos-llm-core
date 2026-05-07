"""Live integration tests for Call with OpenAI models.

These tests hit the real OpenAI API. They are skipped unless
OPENAI_API_KEY or KAOS_LLM_OPENAI_API_KEY is set.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from kaos_llm_core.programs.call import Call
from kaos_llm_core.signatures import InputField, OutputField, Signature

from .conftest import requires_openai


class Entity(BaseModel):
    name: str
    type: str


class ExtractEntities(Signature):
    """Extract named entities from the given text. Return entity names and their types."""

    text: str = InputField(description="Text to extract entities from")
    entities: list[Entity] = OutputField(description="Extracted named entities")


class SummarizeText(Signature):
    """Summarize the given text in one sentence."""

    text: str = InputField(description="Text to summarize")
    summary: str = OutputField(description="One-sentence summary")


MODEL = "openai:gpt-5.4-nano"


@requires_openai
class TestCallOpenAI:
    @pytest.mark.integration
    async def test_entity_extraction(self) -> None:
        """Extract typed entities via OpenAI."""
        call = Call(ExtractEntities, model=MODEL)
        invocation = await call.invoke(
            text="Microsoft CEO Satya Nadella met with European Commission "
            "President Ursula von der Leyen in Brussels on January 10, 2025."
        )
        result = invocation.output

        assert isinstance(result.entities, list)
        assert len(result.entities) > 0
        for entity in result.entities:
            assert isinstance(entity, Entity)
        names = [e.name for e in result.entities]
        assert any("Microsoft" in n or "Nadella" in n for n in names)

        trace = invocation.trace
        assert trace is not None
        assert trace.input_tokens > 0

    @pytest.mark.integration
    async def test_summarization(self) -> None:
        """Summarize text and verify output is concise."""
        call = Call(SummarizeText, model=MODEL)
        result = await call(
            text="The Federal Reserve announced today that it will maintain the federal "
            "funds rate at its current level of 5.25-5.50 percent. The decision was "
            "unanimous among FOMC members. Chair Powell emphasized that while inflation "
            "has shown signs of moderating, the committee remains vigilant and will "
            "continue to monitor incoming data before making any adjustments."
        )

        assert isinstance(result.summary, str)
        assert len(result.summary) > 10
        summary_lower = result.summary.lower()
        assert any(term in summary_lower for term in ["fed", "rate", "reserve"])
