"""Tests for :mod:`kaos_llm_core.results` foundation containers."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, ValidationError

from kaos_llm_core.labels import Label
from kaos_llm_core.results import (
    Classification,
    SourceSpan,
    Summary,
    SummaryMethod,
)

# ---------------------------------------------------------------------------
# SourceSpan
# ---------------------------------------------------------------------------


class TestSourceSpan:
    def test_minimal(self) -> None:
        span = SourceSpan(start=0, end=5)
        assert span.start == 0
        assert span.end == 5
        assert span.source_id is None
        assert span.length == 5

    def test_with_source_id(self) -> None:
        span = SourceSpan(start=3, end=10, source_id="doc-1")
        assert span.source_id == "doc-1"
        assert span.length == 7

    def test_zero_length_allowed(self) -> None:
        span = SourceSpan(start=5, end=5)
        assert span.length == 0

    def test_negative_start_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SourceSpan(start=-1, end=5)

    def test_negative_end_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SourceSpan(start=0, end=-1)

    def test_end_before_start_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"end .* must be >= start"):
            SourceSpan(start=10, end=5)

    def test_json_round_trip(self) -> None:
        span = SourceSpan(start=2, end=9, source_id="doc-1")
        round_trip = SourceSpan.model_validate_json(span.model_dump_json())
        assert round_trip == span


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class _ContractPayload(BaseModel):
    parties: list[str]
    effective_date: str


class TestSummaryConstruction:
    def test_minimal(self) -> None:
        s = Summary[str](text="This is a summary.")
        assert s.text == "This is a summary."
        assert s.payload is None
        assert s.method == "abstractive"
        assert s.depth == 0
        assert s.chunks_used == []
        assert s.source_spans == []
        assert s.metadata == {}

    def test_method_tag_accepted(self) -> None:
        for method in ("abstractive", "extractive", "hybrid"):
            s = Summary[str](text="x", method=method)
            assert s.method == method

    def test_invalid_method_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Summary[str](text="x", method="paraphrase")  # ty: ignore[invalid-argument-type]

    def test_with_chunks_and_spans(self) -> None:
        s = Summary[str](
            text="x",
            chunks_used=["c1", "c2"],
            source_spans=[
                SourceSpan(start=0, end=10),
                SourceSpan(start=15, end=20, source_id="doc-2"),
            ],
        )
        assert s.chunks_used == ["c1", "c2"]
        assert len(s.source_spans) == 2

    def test_metadata_arbitrary(self) -> None:
        s = Summary[str](
            text="x",
            metadata={
                "router.model_id": "gpt-5-mini",
                "reducer.branching": 4,
                "cost_usd": 0.012,
            },
        )
        assert s.metadata["router.model_id"] == "gpt-5-mini"

    def test_generic_payload_pydantic(self) -> None:
        payload = _ContractPayload(parties=["Alpha", "Beta"], effective_date="2026-01-01")
        s = Summary[_ContractPayload](text="contract summary", payload=payload)
        assert s.payload is not None
        assert s.payload.parties == ["Alpha", "Beta"]

    def test_summary_method_literal_values(self) -> None:
        # SummaryMethod literal exposes the canonical tag set.
        assert SummaryMethod.__args__ == ("abstractive", "extractive", "hybrid")


class TestSummarySerialization:
    def test_json_round_trip(self) -> None:
        s = Summary[str](
            text="x",
            method="extractive",
            chunks_used=["c1"],
            source_spans=[SourceSpan(start=0, end=5)],
            metadata={"k": "v"},
        )
        round_trip = Summary[str].model_validate_json(s.model_dump_json())
        assert round_trip == s

    def test_dump_is_jsonable(self) -> None:
        s = Summary[str](text="x")
        json.dumps(s.model_dump())

    def test_generic_payload_round_trip(self) -> None:
        payload = _ContractPayload(parties=["A"], effective_date="2026-05-14")
        s = Summary[_ContractPayload](text="x", payload=payload)
        data = s.model_dump_json()
        round_trip = Summary[_ContractPayload].model_validate_json(data)
        assert round_trip.payload is not None
        assert round_trip.payload.parties == ["A"]


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class TestClassificationConstruction:
    def test_minimal_str(self) -> None:
        c = Classification[str]()
        assert c.labels == []
        assert c.scores == {}
        assert c.abstained is False
        assert c.rationale is None

    def test_single_label(self) -> None:
        c = Classification[str](labels=["positive"], scores={"positive": 0.92})
        assert c.labels == ["positive"]
        assert c.scores["positive"] == 0.92
        assert c.top_label == "positive"

    def test_multi_label(self) -> None:
        c = Classification[str](
            labels=["a", "b"],
            scores={"a": 0.8, "b": 0.6, "c": 0.1},
        )
        assert set(c.labels) == {"a", "b"}
        assert c.top_label == "a"

    def test_abstained(self) -> None:
        c = Classification[str](abstained=True)
        assert c.abstained is True
        assert c.top_label is None
        assert c.names == []

    def test_with_label_records(self) -> None:
        labels = [Label(name="x", description="alpha")]
        c = Classification[Label](labels=labels)
        assert c.names == ["x"]
        assert c.top_label == "x"

    def test_top_label_from_scores(self) -> None:
        c = Classification[str](
            labels=["a", "b", "c"],
            scores={"a": 0.2, "b": 0.9, "c": 0.5},
        )
        assert c.top_label == "b"

    def test_top_label_falls_back_to_first_when_no_scores(self) -> None:
        c = Classification[str](labels=["first", "second"])
        assert c.top_label == "first"

    def test_abstained_overrides_top_label(self) -> None:
        c = Classification[str](
            labels=["a"],
            scores={"a": 0.99},
            abstained=True,
        )
        assert c.top_label is None


class TestClassificationSerialization:
    def test_str_round_trip(self) -> None:
        c = Classification[str](
            labels=["a", "b"],
            scores={"a": 0.5, "b": 0.4},
            abstained=False,
            rationale="because.",
            chunks_used=["c1"],
            source_spans=[SourceSpan(start=0, end=3)],
            metadata={"router": "cascade"},
        )
        round_trip = Classification[str].model_validate_json(c.model_dump_json())
        assert round_trip == c

    def test_label_payload_round_trip(self) -> None:
        c = Classification[Label](
            labels=[Label(name="alpha", description="A")],
            scores={"alpha": 0.7},
        )
        round_trip = Classification[Label].model_validate_json(c.model_dump_json())
        assert round_trip.names == ["alpha"]
        assert round_trip.labels[0].description == "A"

    def test_dump_is_jsonable(self) -> None:
        c = Classification[str]()
        json.dumps(c.model_dump())


# ---------------------------------------------------------------------------
# Public API exposure
# ---------------------------------------------------------------------------


def test_results_exposed_from_top_level_module() -> None:
    import kaos_llm_core

    assert kaos_llm_core.Summary is Summary
    assert kaos_llm_core.Classification is Classification
    assert kaos_llm_core.SourceSpan is SourceSpan
    for name in ("Summary", "Classification", "SourceSpan", "SummaryMethod"):
        assert name in kaos_llm_core.__all__
