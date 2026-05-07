"""Unit tests for span tagging and citation validation."""

from __future__ import annotations

from pydantic import BaseModel

from kaos_llm_core.signatures.grounding import Cited, MatchStrategy, Span
from kaos_llm_core.signatures.span_tagging import (
    _extract_cited_values,
    tag_paragraphs,
    tag_passages,
    validate_cited_output,
)

CONTRACT = (
    "This Agreement is entered into as of January 1, 2025.\n\n"
    "The Seller shall deliver the goods within 30 days.\n\n"
    "The Buyer shall pay $10,000 upon delivery.\n\n"
    "Governing law: State of Delaware."
)


# ---------------------------------------------------------------------------
# tag_paragraphs
# ---------------------------------------------------------------------------


class TestTagParagraphs:
    def test_basic_tagging(self):
        tagged, span_map = tag_paragraphs(CONTRACT, source_uri="doc:contract")
        assert "[SRC:doc:contract#p0]" in tagged
        assert "[SRC:doc:contract#p1]" in tagged
        assert len(span_map) >= 3

    def test_span_map_offsets(self):
        _tagged, span_map = tag_paragraphs(CONTRACT, source_uri="doc:c")
        for _uri, (start, end) in span_map.items():
            # Offsets should reference the original text
            assert CONTRACT[start:end].strip()

    def test_first_paragraph_content(self):
        _tagged, span_map = tag_paragraphs(CONTRACT, source_uri="doc:c")
        first_uri = "doc:c#p0"
        assert first_uri in span_map
        start, end = span_map[first_uri]
        assert "January 1, 2025" in CONTRACT[start:end]

    def test_min_paragraph_chars(self):
        text = "Hi\n\nThis is a longer paragraph that should be tagged."
        _tagged, span_map = tag_paragraphs(text, source_uri="doc:t", min_paragraph_chars=10)
        # "Hi" is too short, should not be tagged
        assert len(span_map) == 1
        assert "doc:t#p0" in span_map

    def test_empty_text(self):
        _tagged, span_map = tag_paragraphs("", source_uri="doc:e")
        assert span_map == {}

    def test_single_paragraph(self):
        text = "One paragraph with enough content to be tagged."
        tagged, span_map = tag_paragraphs(text, source_uri="doc:s")
        assert len(span_map) == 1
        assert "[SRC:doc:s#p0]" in tagged


# ---------------------------------------------------------------------------
# tag_passages
# ---------------------------------------------------------------------------


class TestTagPassages:
    def test_basic_passages(self):
        passages = {
            "doc:a": "Content of document A.",
            "doc:b": "Content of document B.",
        }
        tagged, source_map = tag_passages(passages)
        assert "=== SOURCE: doc:a ===" in tagged
        assert "=== SOURCE: doc:b ===" in tagged
        assert source_map == passages

    def test_empty_passages(self):
        tagged, source_map = tag_passages({})
        assert tagged == ""
        assert source_map == {}


# ---------------------------------------------------------------------------
# _extract_cited_values
# ---------------------------------------------------------------------------


class TestExtractCitedValues:
    def test_single_cited(self):
        cited = Cited(value="test", spans=[Span(source_uri="u", char_span=(0, 4), quote="test")])
        assert len(_extract_cited_values(cited)) == 1

    def test_pydantic_model_with_cited_fields(self):
        class MyOutput(BaseModel):
            name: Cited[str]
            amount: Cited[int]

        output = MyOutput(
            name=Cited(
                value="Alice", spans=[Span(source_uri="u", char_span=(0, 5), quote="Alice")]
            ),
            amount=Cited(value=100, spans=[Span(source_uri="u", char_span=(10, 13), quote="100")]),
        )
        results = _extract_cited_values(output)
        assert len(results) == 2

    def test_list_of_cited(self):
        items = [
            Cited(value="a", spans=[Span(source_uri="u", char_span=(0, 1), quote="a")]),
            Cited(value="b", spans=[Span(source_uri="u", char_span=(2, 3), quote="b")]),
        ]
        assert len(_extract_cited_values(items)) == 2

    def test_nested_model(self):
        class Inner(BaseModel):
            field: Cited[str]

        class Outer(BaseModel):
            inner: Inner

        output = Outer(
            inner=Inner(
                field=Cited(value="v", spans=[Span(source_uri="u", char_span=(0, 1), quote="v")])
            )
        )
        assert len(_extract_cited_values(output)) == 1

    def test_non_cited_model(self):
        class Plain(BaseModel):
            name: str
            count: int

        output = Plain(name="test", count=5)
        assert len(_extract_cited_values(output)) == 0


# ---------------------------------------------------------------------------
# validate_cited_output
# ---------------------------------------------------------------------------


class TestValidateCitedOutput:
    def test_valid_citations(self):
        span = Span.from_source("doc:c", CONTRACT, (0, 53))
        cited = Cited(value="Jan 1 2025", spans=[span])

        class Output(BaseModel):
            date: Cited[str]

        output = Output(date=cited)
        errors = validate_cited_output(output, {"doc:c": CONTRACT})
        assert errors == []

    def test_invalid_citation(self):
        span = Span(source_uri="doc:c", char_span=(0, 10), quote="WRONG TEXT")
        cited = Cited(value="wrong", spans=[span])

        class Output(BaseModel):
            field: Cited[str]

        output = Output(field=cited)
        errors = validate_cited_output(output, {"doc:c": CONTRACT})
        assert len(errors) == 1

    def test_with_span_map(self):
        """Paragraph URIs should resolve via span_map."""
        _tagged, span_map = tag_paragraphs(CONTRACT, source_uri="doc:c")
        # Create a Cited referencing a paragraph URI
        p0_start, p0_end = span_map["doc:c#p0"]
        p0_text = CONTRACT[p0_start:p0_end]
        span = Span.from_source("doc:c#p0", p0_text, (0, min(20, len(p0_text))))
        cited = Cited(value="date", spans=[span])

        class Output(BaseModel):
            field: Cited[str]

        output = Output(field=cited)
        errors = validate_cited_output(output, {"doc:c": CONTRACT}, span_map=span_map)
        assert errors == []

    def test_missing_source(self):
        span = Span(source_uri="doc:missing", char_span=(0, 5), quote="hello")
        cited = Cited(value="v", spans=[span])
        errors = validate_cited_output(cited, {"doc:other": "text"})
        assert len(errors) == 1
        assert errors[0].reason == "source_missing"

    def test_substring_fallback(self):
        """Substring strategy should recover position-shifted quotes."""
        span = Span(source_uri="doc:c", char_span=(999, 999), quote="30 days")
        cited = Cited(value="30 days", spans=[span])
        errors = validate_cited_output(
            cited,
            {"doc:c": CONTRACT},
            strategies=(MatchStrategy.STRICT, MatchStrategy.SUBSTRING),
        )
        assert errors == []
