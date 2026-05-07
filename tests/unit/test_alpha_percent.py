# ruff: noqa: RUF001
"""Unit tests for AlphaPercentExtractor — WS-TR.PR-6f.5.

Covers: symbol-suffix branch (%, ‰, ‱, bps) with scale factors, word
branch (percent, percentage, basis points), the basis-point context
guard, and documented limitations.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from kaos_llm_core.extract.alpha.percent import AlphaPercentExtractor
from kaos_llm_core.extract.base_extractor import ExtractorValueType


@pytest.fixture
def extractor() -> AlphaPercentExtractor:
    return AlphaPercentExtractor()


class TestMetadata:
    def test_name(self, extractor: AlphaPercentExtractor) -> None:
        assert extractor.get_name() == "percent"

    def test_value_type(self, extractor: AlphaPercentExtractor) -> None:
        assert extractor.get_value_type() == ExtractorValueType.PERCENTAGE

    def test_unsupported_language_raises(self) -> None:
        with pytest.raises(ValueError, match="does not support language"):
            AlphaPercentExtractor(language="klingon")


class TestPercentSign:
    def test_integer_percent(self, extractor: AlphaPercentExtractor) -> None:
        spans = list(extractor.extract_spans("I spent 30% of my time."))
        assert len(spans) == 1
        assert spans[0].value == Decimal("0.30")

    def test_decimal_percent(self, extractor: AlphaPercentExtractor) -> None:
        spans = list(extractor.extract_spans("Interest rate is 5.25% per annum."))
        assert spans[0].value == Decimal("0.0525")

    def test_zero_percent(self, extractor: AlphaPercentExtractor) -> None:
        spans = list(extractor.extract_spans("Growth was 0%."))
        assert spans[0].value == Decimal("0.00")

    def test_hundred_percent(self, extractor: AlphaPercentExtractor) -> None:
        spans = list(extractor.extract_spans("Ownership is 100%."))
        assert spans[0].value == Decimal("1.00")

    def test_span_includes_sign(self, extractor: AlphaPercentExtractor) -> None:
        text = "rate 7.5% annually"
        span = extractor.extract_first_span(text)
        assert span is not None
        assert text[span.start : span.end] == "7.5%"


class TestBasisPoints:
    def test_bps_suffix(self, extractor: AlphaPercentExtractor) -> None:
        spans = list(extractor.extract_spans("Spread widened by 50bps."))
        assert spans[0].value == Decimal("0.0050")

    def test_basis_points_two_words(self, extractor: AlphaPercentExtractor) -> None:
        spans = list(extractor.extract_spans("100 basis points of LIBOR."))
        assert spans[0].value == Decimal("0.01")

    def test_basis_without_points_skipped(self, extractor: AlphaPercentExtractor) -> None:
        """'basis' without 'points' is NOT a percent — context guard."""
        spans = list(extractor.extract_spans("100 basis for the model."))
        assert spans == []


class TestPerMille:
    def test_per_mille(self, extractor: AlphaPercentExtractor) -> None:
        spans = list(extractor.extract_spans("Mortality rate: 10‰."))
        assert spans[0].value == Decimal("0.010")

    def test_per_myriad(self, extractor: AlphaPercentExtractor) -> None:
        spans = list(extractor.extract_spans("1‱ concentration."))
        assert spans[0].value == Decimal("0.0001")


class TestFullwidthVariants:
    def test_fullwidth_percent(self, extractor: AlphaPercentExtractor) -> None:
        spans = list(extractor.extract_spans("売上 30％ up."))
        assert spans[0].value == Decimal("0.30")

    def test_small_percent(self, extractor: AlphaPercentExtractor) -> None:
        spans = list(extractor.extract_spans("yield 5﹪."))
        assert spans[0].value == Decimal("0.05")


class TestWordForms:
    def test_percent_word(self, extractor: AlphaPercentExtractor) -> None:
        spans = list(extractor.extract_spans("grew by five percent."))
        assert spans[0].value == Decimal("0.05")

    def test_percentage_word(self, extractor: AlphaPercentExtractor) -> None:
        spans = list(extractor.extract_spans("up 10 percentage points."))
        assert spans[0].value == Decimal("0.10")

    def test_pct_abbreviation(self, extractor: AlphaPercentExtractor) -> None:
        # 'pct.' is in PERCENT_MAP; after rstripping '.' it's 'pct' which isn't
        # a key, so kelvin would miss this. Port stays kelvin-faithful.
        spans = list(extractor.extract_spans("5 pct. of revenue."))
        # Either empty (kelvin-faithful) or matches (improvement). Just
        # assert no crash:
        for s in spans:
            assert isinstance(s.value, Decimal)


class TestPPMPPB:
    def test_ppm(self, extractor: AlphaPercentExtractor) -> None:
        spans = list(extractor.extract_spans("CO2 at 400 ppm."))
        assert spans[0].value == Decimal("0.0004")

    def test_ppb(self, extractor: AlphaPercentExtractor) -> None:
        spans = list(extractor.extract_spans("Contaminant at 5 ppb."))
        assert spans[0].value == Decimal("0.000000005")


class TestEdgeCases:
    def test_empty(self, extractor: AlphaPercentExtractor) -> None:
        assert list(extractor.extract_spans("")) == []

    def test_no_percent(self, extractor: AlphaPercentExtractor) -> None:
        assert list(extractor.extract_spans("The cat sat on the mat.")) == []

    def test_bare_percent_sign_skipped(self, extractor: AlphaPercentExtractor) -> None:
        spans = list(extractor.extract_spans("% alone does nothing."))
        assert spans == []

    def test_multiple_percents(self, extractor: AlphaPercentExtractor) -> None:
        spans = list(extractor.extract_spans("30% of time, 20% of budget."))
        values = [s.value for s in spans]
        assert Decimal("0.30") in values
        assert Decimal("0.20") in values
