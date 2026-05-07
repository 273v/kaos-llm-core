"""Unit tests for AlphaDurationExtractor — WS-TR.PR-6f.5.

Covers: arabic-quantity path, written-number path, indefinite-article
path, skipped-modifier behavior (business/calendar/working), every unit
in the gazetteer, span round-tripping.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from kaos_llm_core.extract.alpha.duration import AlphaDurationExtractor, DurationMatch
from kaos_llm_core.extract.base_extractor import ExtractorValueType


@pytest.fixture
def extractor() -> AlphaDurationExtractor:
    return AlphaDurationExtractor()


class TestMetadata:
    def test_name(self, extractor: AlphaDurationExtractor) -> None:
        assert extractor.get_name() == "duration"

    def test_value_type(self, extractor: AlphaDurationExtractor) -> None:
        assert extractor.get_value_type() == ExtractorValueType.DURATION

    def test_unsupported_language_raises(self) -> None:
        with pytest.raises(ValueError, match="does not support language"):
            AlphaDurationExtractor(language="klingon")


class TestBasicUnits:
    @pytest.mark.parametrize(
        ("text", "qty", "unit", "seconds"),
        [
            ("30 seconds", 30, "seconds", 30),
            ("5 minutes", 5, "minutes", 300),
            ("2 hours", 2, "hours", 7200),
            ("90 days", 90, "days", 7776000),
            ("3 weeks", 3, "weeks", 1814400),
            ("6 months", 6, "months", 15552000),
            ("10 years", 10, "years", 315360000),
        ],
    )
    def test_arabic_units(
        self,
        extractor: AlphaDurationExtractor,
        text: str,
        qty: int,
        unit: str,
        seconds: int,
    ) -> None:
        spans = list(extractor.extract_spans(text))
        assert len(spans) == 1
        assert spans[0].value == DurationMatch(
            quantity=Decimal(qty),
            unit=unit,
            total_seconds=Decimal(seconds),
        )


class TestSingularUnits:
    def test_one_day(self, extractor: AlphaDurationExtractor) -> None:
        spans = list(extractor.extract_spans("1 day"))
        assert spans[0].value.unit == "day"
        assert spans[0].value.quantity == Decimal("1")


class TestWrittenQuantity:
    def test_thirteen_months(self, extractor: AlphaDurationExtractor) -> None:
        spans = list(extractor.extract_spans("I worked at Acme for thirteen months."))
        assert spans[0].value == DurationMatch(
            quantity=Decimal("13"),
            unit="months",
            total_seconds=Decimal("33696000"),
        )

    def test_ninety_days(self, extractor: AlphaDurationExtractor) -> None:
        spans = list(extractor.extract_spans("notice of ninety days."))
        assert spans[0].value.quantity == Decimal("90")
        assert spans[0].value.unit == "days"


class TestIndefiniteArticle:
    @pytest.mark.parametrize(
        ("text", "expected_unit"),
        [
            ("a year", "year"),
            ("an hour", "hour"),
            ("the week", "week"),
            ("this day", "day"),
        ],
    )
    def test_article_one(
        self, extractor: AlphaDurationExtractor, text: str, expected_unit: str
    ) -> None:
        spans = list(extractor.extract_spans(text))
        assert len(spans) == 1
        assert spans[0].value.quantity == Decimal("1")
        assert spans[0].value.unit == expected_unit


class TestSkippedModifiers:
    def test_business_days_skipped(self, extractor: AlphaDurationExtractor) -> None:
        """'business' token by itself is a modifier; emission skipped."""
        spans = list(extractor.extract_spans("3 business days."))
        # The 'days' token has 'business' as prior which isn't a number.
        # 'business' is a skip-modifier so no emission on 'business'.
        # Result: no span. (Kelvin-faithful.)
        assert spans == []

    def test_calendar_months_skipped(self, extractor: AlphaDurationExtractor) -> None:
        spans = list(extractor.extract_spans("6 calendar months."))
        assert spans == []

    def test_working_days_skipped(self, extractor: AlphaDurationExtractor) -> None:
        spans = list(extractor.extract_spans("5 working days."))
        assert spans == []


class TestFractionalQuantities:
    def test_fractional_hours(self, extractor: AlphaDurationExtractor) -> None:
        """Unlike kelvin (converts to minutes), we keep fractional
        Decimal and multiply into total_seconds."""
        spans = list(extractor.extract_spans("1.5 hours"))
        assert spans[0].value.quantity == Decimal("1.5")
        assert spans[0].value.total_seconds == Decimal("5400.0")


class TestAnniversary:
    def test_anniversary(self, extractor: AlphaDurationExtractor) -> None:
        """'anniversary' is a year synonym in DURATION_MAP."""
        spans = list(extractor.extract_spans("5 anniversary"))
        assert spans[0].value.unit == "anniversary"
        assert spans[0].value.total_seconds == Decimal("157680000")


class TestEdgeCases:
    def test_empty(self, extractor: AlphaDurationExtractor) -> None:
        assert list(extractor.extract_spans("")) == []

    def test_no_duration(self, extractor: AlphaDurationExtractor) -> None:
        assert list(extractor.extract_spans("The cat sat on the mat.")) == []

    def test_bare_unit_skipped(self, extractor: AlphaDurationExtractor) -> None:
        """'days' alone with no prior token or article → no emission."""
        spans = list(extractor.extract_spans("days"))
        assert spans == []

    def test_span_round_trips(self, extractor: AlphaDurationExtractor) -> None:
        text = "notice period of 90 days is required."
        span = extractor.extract_first_span(text)
        assert span is not None
        assert "90 days" in text[span.start : span.end]
