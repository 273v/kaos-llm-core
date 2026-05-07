"""Live integration tests for Call with Google Gemini models.

These tests hit the real Google AI API. They are skipped unless
GOOGLE_API_KEY, GOOGLE_GENERATIVE_AI_API_KEY, or KAOS_LLM_GOOGLE_API_KEY is set.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from kaos_llm_core.programs.call import Call
from kaos_llm_core.signatures import InputField, OutputField, Signature

from .conftest import requires_google


class Entity(BaseModel):
    name: str
    type: str


class ExtractEntities(Signature):
    """Extract named entities from the given text. Return entity names and their types."""

    text: str = InputField(description="Text to extract entities from")
    entities: list[Entity] = OutputField(description="Extracted named entities")


class ClassifyLanguage(Signature):
    """Identify the primary language of the given text."""

    text: str = InputField(description="Text to classify")
    language: str = OutputField(description="ISO 639-1 language code (e.g., 'en', 'fr', 'de')")
    language_name: str = OutputField(description="Full language name (e.g., 'English', 'French')")


MODEL = "google:gemini-2.5-flash"


@requires_google
class TestCallGoogle:
    @pytest.mark.integration
    async def test_entity_extraction(self) -> None:
        """Extract typed entities via Google Gemini."""
        call = Call(ExtractEntities, model=MODEL)
        invocation = await call.invoke(
            text="Amazon founder Jeff Bezos announced a $10 billion donation "
            "to the Bezos Earth Fund to combat climate change."
        )
        result = invocation.output

        assert isinstance(result.entities, list)
        assert len(result.entities) > 0
        for entity in result.entities:
            assert isinstance(entity, Entity)
        names = [e.name for e in result.entities]
        assert any("Bezos" in n or "Amazon" in n for n in names)

        trace = invocation.trace
        assert trace is not None
        assert trace.input_tokens > 0

    @pytest.mark.integration
    async def test_language_classification(self) -> None:
        """Classify language of text and verify ISO code."""
        call = Call(ClassifyLanguage, model=MODEL)
        result = await call(
            text="La Cour européenne des droits de l'homme a rendu un arrêt "
            "important concernant le droit à la vie privée."
        )

        assert result.language == "fr", f"Expected 'fr', got '{result.language}'"
        assert "french" in result.language_name.lower()
