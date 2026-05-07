"""Unit tests for Cited[T] — lightweight extraction citation wrapper."""

from __future__ import annotations

import pytest

from kaos_llm_core.signatures.grounding import (
    Cited,
    Claim,
    MatchStrategy,
    Span,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DELAWARE_TEXT = (
    "The filing fee is $89 and the certificate must include "
    "the corporation's name, its registered office, and the "
    "name of its registered agent."
)


def _span(quote: str, start: int = 0, end: int = 0, uri: str = "doc:test") -> Span:
    """Build a Span with defaults for testing."""
    return Span(source_uri=uri, char_span=(start, end), quote=quote)


def _cited(value: str, quote: str = "The filing fee is $89") -> Cited[str]:
    """Build a simple Cited[str] for testing."""
    return Cited(value=value, spans=[_span(quote)])


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestCitedConstruction:
    def test_basic_construction(self):
        c = _cited("$89")
        assert c.value == "$89"
        assert len(c.spans) == 1
        assert c.confidence == 1.0

    def test_custom_confidence(self):
        c = Cited(value="test", spans=[_span("quote")], confidence=0.8)
        assert c.confidence == 0.8

    def test_multiple_spans(self):
        c = Cited(
            value="multiple sources",
            spans=[_span("first"), _span("second", uri="doc:other")],
        )
        assert len(c.spans) == 2

    def test_requires_at_least_one_span(self):
        with pytest.raises((ValueError, TypeError)):  # Pydantic validation
            Cited(value="no spans", spans=[])

    def test_confidence_clamped(self):
        with pytest.raises((ValueError, TypeError)):
            Cited(value="bad", spans=[_span("q")], confidence=1.5)
        with pytest.raises((ValueError, TypeError)):
            Cited(value="bad", spans=[_span("q")], confidence=-0.1)

    def test_generic_types(self):
        """Cited works with various T types."""
        c_str = Cited[str](value="text", spans=[_span("q")])
        assert c_str.value == "text"

        c_int = Cited[int](value=42, spans=[_span("q")])
        assert c_int.value == 42

        c_bool = Cited[bool](value=True, spans=[_span("q")])
        assert c_bool.value is True

        c_list = Cited[list[str]](value=["a", "b"], spans=[_span("q")])
        assert c_list.value == ["a", "b"]


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------


class TestCitedAccessors:
    def test_sources(self):
        c = Cited(
            value="v",
            spans=[_span("a", uri="doc:1"), _span("b", uri="doc:2"), _span("c", uri="doc:1")],
        )
        assert c.sources == {"doc:1", "doc:2"}

    def test_quote(self):
        c = Cited(value="v", spans=[_span("the first quote")])
        assert c.quote == "the first quote"

    def test_quote_returns_first_span(self):
        c = Cited(value="v", spans=[_span("first"), _span("second")])
        assert c.quote == "first"


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


class TestCitedVerification:
    def test_verify_strict_pass(self):
        span = Span.from_source("doc:d", DELAWARE_TEXT, (0, 23))
        c = Cited(value="$89", spans=[span])
        errors = c.verify({"doc:d": DELAWARE_TEXT})
        assert errors == []

    def test_verify_strict_fail(self):
        span = Span(source_uri="doc:d", char_span=(0, 10), quote="wrong text")
        c = Cited(value="wrong", spans=[span])
        errors = c.verify({"doc:d": DELAWARE_TEXT})
        assert len(errors) == 1
        assert errors[0].reason in ("quote_mismatch", "out_of_range")

    def test_verify_substring_fallback(self):
        span = Span(source_uri="doc:d", char_span=(999, 999), quote="filing fee is $89")
        c = Cited(value="$89", spans=[span])
        errors = c.verify(
            {"doc:d": DELAWARE_TEXT},
            strategies=(MatchStrategy.STRICT, MatchStrategy.SUBSTRING),
        )
        assert errors == []  # substring match succeeds

    def test_verify_source_missing(self):
        c = _cited("v")
        errors = c.verify({"doc:other": "text"})
        assert len(errors) == 1
        assert errors[0].reason == "source_missing"

    def test_verify_multiple_spans(self):
        s1 = Span.from_source("doc:d", DELAWARE_TEXT, (0, 23))
        s2 = Span(source_uri="doc:d", char_span=(999, 999), quote="nonexistent")
        c = Cited(value="mixed", spans=[s1, s2])
        errors = c.verify({"doc:d": DELAWARE_TEXT})
        assert len(errors) == 1
        assert errors[0].span_index == 1

    def test_verify_with_callable_corpus(self):
        span = Span.from_source("doc:d", DELAWARE_TEXT, (0, 23))
        c = Cited(value="$89", spans=[span])

        def lookup(uri: str) -> str:
            if uri == "doc:d":
                return DELAWARE_TEXT
            raise KeyError(uri)

        assert c.verify(lookup) == []


# ---------------------------------------------------------------------------
# Promotion to Claim
# ---------------------------------------------------------------------------


class TestCitedToClaim:
    def test_to_claim_default_statement(self):
        c = _cited("$89 filing fee")
        claim = c.to_claim()
        assert isinstance(claim, Claim)
        assert claim.statement == "$89 filing fee"
        assert claim.claim_type == "factual"
        assert len(claim.supporting_spans) == 1
        assert claim.confidence == 1.0

    def test_to_claim_custom_statement(self):
        c = _cited("$89")
        claim = c.to_claim(statement="The Delaware filing fee is $89")
        assert claim.statement == "The Delaware filing fee is $89"

    def test_to_claim_custom_type(self):
        c = _cited("$89")
        claim = c.to_claim(claim_type="quantitative")
        assert claim.claim_type == "quantitative"

    def test_to_claim_preserves_spans(self):
        s1 = _span("first", uri="doc:1")
        s2 = _span("second", uri="doc:2")
        c = Cited(value="multi", spans=[s1, s2], confidence=0.9)
        claim = c.to_claim()
        assert len(claim.supporting_spans) == 2
        assert claim.confidence == 0.9


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


class TestCitedSerialization:
    def test_json_round_trip(self):
        c = _cited("$89")
        data = c.model_dump()
        restored = Cited[str].model_validate(data)
        assert restored.value == c.value
        assert restored.spans[0].quote == c.spans[0].quote
        assert restored.confidence == c.confidence

    def test_json_schema_generation(self):
        schema = Cited[str].model_json_schema()
        assert "value" in schema["properties"]
        assert "spans" in schema["properties"]
        assert "confidence" in schema["properties"]
