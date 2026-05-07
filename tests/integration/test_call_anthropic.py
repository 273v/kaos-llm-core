"""Live integration tests for Call with Anthropic Claude models.

These tests hit the real Anthropic API. They are skipped unless
ANTHROPIC_API_KEY or KAOS_LLM_ANTHROPIC_API_KEY is set.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from kaos_llm_core.programs.call import Call
from kaos_llm_core.signatures import InputField, OutputField, Signature

from .conftest import requires_anthropic

# --- Domain Models ---


class Entity(BaseModel):
    """A named entity with its type."""

    name: str
    type: str


# --- Signatures ---


class ExtractEntities(Signature):
    """Extract named entities from the given text. Return entity names and their types."""

    text: str = InputField(description="Text to extract entities from")
    entities: list[Entity] = OutputField(description="Extracted named entities")


class ClassifyDocument(Signature):
    """Classify the type of document based on its content."""

    text: str = InputField(description="Document text (first 500 characters)")
    category: str = OutputField(
        description="Document category",
        json_schema_extra={
            "kaos_field_kind": "output",
            "enum": ["legal", "financial", "medical", "technical", "other"],
        },
    )
    confidence: float = OutputField(description="Classification confidence between 0 and 1")


class AnalyzeSentiment(Signature):
    """Analyze the sentiment of the given text."""

    text: str = InputField(description="Text to analyze")
    sentiment: str = OutputField(description="Sentiment: positive, negative, or neutral")
    score: float = OutputField(description="Sentiment score from -1 (negative) to 1 (positive)")


# --- Tests ---

MODEL = "anthropic:claude-haiku-4-5"


@requires_anthropic
class TestCallAnthropic:
    @pytest.mark.integration
    async def test_entity_extraction(self) -> None:
        """Extract entities from a simple sentence and verify typed output."""
        call = Call(ExtractEntities, model=MODEL)
        invocation = await call.invoke(
            text="The Securities and Exchange Commission filed a complaint against Tesla Inc. "
            "in the Southern District of New York on March 15, 2024."
        )
        result = invocation.output

        entities = result.entities
        assert isinstance(entities, list)
        assert len(entities) > 0

        # Each entity should be a typed Entity object, not a dict
        for entity in entities:
            assert isinstance(entity, Entity), f"Expected Entity, got {type(entity)}"
            assert isinstance(entity.name, str)
            assert isinstance(entity.type, str)

        entity_names = [e.name for e in entities]
        assert any("SEC" in name or "Securities" in name for name in entity_names), (
            f"Expected SEC or Securities and Exchange Commission in entities, got: {entity_names}"
        )

        trace = invocation.trace
        assert trace is not None
        assert trace.model == MODEL
        assert trace.input_tokens > 0

    @pytest.mark.integration
    async def test_document_classification(self) -> None:
        """Classify a legal document and verify enum output."""
        call = Call(ClassifyDocument, model=MODEL)
        result = await call(
            text="UNITED STATES DISTRICT COURT SOUTHERN DISTRICT OF NEW YORK. "
            "Case No. 24-cv-1234. COMPLAINT FOR INJUNCTIVE RELIEF. "
            "Plaintiff Securities and Exchange Commission alleges..."
        )

        assert result.category == "legal", f"Expected 'legal', got '{result.category}'"
        assert 0 <= result.confidence <= 1

    @pytest.mark.integration
    async def test_sentiment_analysis(self) -> None:
        """Analyze sentiment and verify typed output."""
        call = Call(AnalyzeSentiment, model=MODEL)
        result = await call(
            text="The company reported record-breaking quarterly earnings, "
            "exceeding all analyst expectations. Investors are thrilled."
        )

        assert result.sentiment in ("positive", "negative", "neutral")
        assert -1 <= result.score <= 1
        assert result.sentiment == "positive"

    @pytest.mark.integration
    async def test_entity_extraction_with_retry(self) -> None:
        """Verify Call works with typed Pydantic output and retry enabled."""
        call = Call(ExtractEntities, model=MODEL, max_retries=2)
        result = await call(text="Apple Inc and Google LLC announced a partnership.")

        assert len(result.entities) > 0
        names = [e.name for e in result.entities]
        assert any("Apple" in n for n in names), f"Expected Apple in entities: {names}"
