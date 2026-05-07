"""Unit tests for AlphaNumberExtractor — WS-TR.PR-6f.4.

Covers: Arabic-number branch (with radix validation), Roman-numeral
branch (with bare-I suppression), written-number branch (including
hyphenated forms), documented limitations, and span round-tripping.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

# Private helpers moved upstream to kaos-nlp-core; the kaos-llm-core
# shim only re-exports the public class. Import the helpers from their
# new home for the unit-level coverage.
from kaos_nlp_core.extract.alpha.number import (
    _is_roman_numeral,
    _parse_arabic_number,
    _parse_written_number,
    _roman_to_decimal,
)

from kaos_llm_core.extract.alpha.number import AlphaNumberExtractor
from kaos_llm_core.extract.base_extractor import ExtractorValueType


@pytest.fixture
def extractor() -> AlphaNumberExtractor:
    return AlphaNumberExtractor()


class TestMetadata:
    def test_name(self, extractor: AlphaNumberExtractor) -> None:
        assert extractor.get_name() == "number"

    def test_value_type(self, extractor: AlphaNumberExtractor) -> None:
        assert extractor.get_value_type() == ExtractorValueType.NUMBER

    def test_languages(self, extractor: AlphaNumberExtractor) -> None:
        assert "en" in extractor.get_languages()


# ----------------------------------------------------------------------
# Arabic numbers
# ----------------------------------------------------------------------


class TestParseArabic:
    @pytest.mark.parametrize(
        ("inp", "expected"),
        [
            ("0", Decimal("0")),
            ("1", Decimal("1")),
            ("100", Decimal("100")),
            ("1234", Decimal("1234")),
            ("1,234", Decimal("1234")),
            ("1,234,567", Decimal("1234567")),
            ("1.5", Decimal("1.5")),
            ("1,234.56", Decimal("1234.56")),
            ("0.01", Decimal("0.01")),
            ("999,999.999", Decimal("999999.999")),
        ],
    )
    def test_valid(self, inp: str, expected: Decimal) -> None:
        assert _parse_arabic_number(inp) == expected

    @pytest.mark.parametrize(
        "inp",
        [
            "",
            "abc",
            "1,2",  # 1-digit block after comma
            "1,23",  # 2-digit block after comma
            "1,2345",  # 4-digit block
            "1..5",  # two periods
            "1.2.3",
            "1,",
            ",",
            ".",
            ",123",
        ],
    )
    def test_invalid(self, inp: str) -> None:
        assert _parse_arabic_number(inp) is None


class TestArabicExtraction:
    def test_single_number(self, extractor: AlphaNumberExtractor) -> None:
        spans = list(extractor.extract_spans("I worked 100 hours."))
        assert len(spans) == 1
        assert spans[0].value == Decimal("100")

    def test_number_span_points_at_digits(self, extractor: AlphaNumberExtractor) -> None:
        text = "I worked 100 hours."
        span = extractor.extract_first_span(text)
        assert span is not None
        assert text[span.start : span.end] == "100"

    def test_decimal(self, extractor: AlphaNumberExtractor) -> None:
        # '$' is NOT in the strip set, so '$13.50' fails Arabic parsing
        # (that's the money extractor's job). Bare decimal works:
        spans = list(extractor.extract_spans("13.50 is the rate."))
        assert spans[0].value == Decimal("13.50")

    def test_thousands(self, extractor: AlphaNumberExtractor) -> None:
        spans = list(extractor.extract_spans("Sold for 1,234,567 dollars."))
        assert len(spans) == 1
        assert spans[0].value == Decimal("1234567")

    def test_multiple_numbers(self, extractor: AlphaNumberExtractor) -> None:
        spans = list(extractor.extract_spans("Chapter 3 has 100 pages and 5 sections."))
        values = [s.value for s in spans]
        assert Decimal("3") in values
        assert Decimal("100") in values
        assert Decimal("5") in values


# ----------------------------------------------------------------------
# Roman numerals
# ----------------------------------------------------------------------


class TestRomanNumerals:
    @pytest.mark.parametrize(
        ("inp", "expected"),
        [
            ("II", Decimal("2")),
            ("IV", Decimal("4")),
            ("V", Decimal("5")),
            ("IX", Decimal("9")),
            ("X", Decimal("10")),
            ("XIV", Decimal("14")),
            ("XL", Decimal("40")),
            ("L", Decimal("50")),
            ("XCIX", Decimal("99")),
            ("C", Decimal("100")),
            ("CD", Decimal("400")),
            ("D", Decimal("500")),
            ("M", Decimal("1000")),
            ("MCMLXIV", Decimal("1964")),
            ("MMXXIV", Decimal("2024")),
        ],
    )
    def test_valid(self, inp: str, expected: Decimal) -> None:
        assert _is_roman_numeral(inp)
        assert _roman_to_decimal(inp) == expected

    def test_mixed_case(self) -> None:
        assert _is_roman_numeral("iv")
        assert _roman_to_decimal("iv") == Decimal("4")

    @pytest.mark.parametrize("inp", ["", "ABC", "MMMMM", "XIIII", "IVX"])
    def test_invalid(self, inp: str) -> None:
        assert not _is_roman_numeral(inp)

    def test_bare_I_suppressed(self, extractor: AlphaNumberExtractor) -> None:
        """Bare ``I`` is too common as a pronoun — must be skipped."""
        spans = list(extractor.extract_spans("I went home."))
        assert spans == []

    def test_roman_in_chapter_heading(self, extractor: AlphaNumberExtractor) -> None:
        spans = list(extractor.extract_spans("See Chapter IV for details."))
        values = [s.value for s in spans]
        assert Decimal("4") in values


# ----------------------------------------------------------------------
# Written numbers
# ----------------------------------------------------------------------


class TestWrittenNumbers:
    @pytest.mark.parametrize(
        ("inp", "expected"),
        [
            ("zero", Decimal("0")),
            ("one", Decimal("1")),
            ("ten", Decimal("10")),
            ("twenty", Decimal("20")),
            ("hundred", Decimal("100")),
            ("million", Decimal("1000000")),
            ("billion", Decimal("1000000000")),
        ],
    )
    def test_single_word(
        self, extractor: AlphaNumberExtractor, inp: str, expected: Decimal
    ) -> None:
        spans = list(extractor.extract_spans(inp))
        assert len(spans) == 1
        assert spans[0].value == expected

    def test_case_insensitive(self, extractor: AlphaNumberExtractor) -> None:
        spans = list(extractor.extract_spans("Ten and MILLION"))
        values = [s.value for s in spans]
        assert Decimal("10") in values
        assert Decimal("1000000") in values

    def test_hyphenated_additive(self, extractor: AlphaNumberExtractor) -> None:
        """Kelvin-faithful: hyphenated written numbers sum components."""
        spans = list(extractor.extract_spans("I saw twenty-three cats."))
        values = [s.value for s in spans]
        assert Decimal("23") in values  # 20 + 3

    def test_hyphenated_two_hundred_limitation(self, extractor: AlphaNumberExtractor) -> None:
        """Documented limitation: 'two-hundred' = 2+100 = 102, not 200."""
        from kaos_nlp_core.locale_data import WRITTEN_NUMBER_MAP

        result = _parse_written_number("two-hundred", WRITTEN_NUMBER_MAP["en"])
        assert result == Decimal("102")

    def test_hyphenated_with_non_number_rejected(self) -> None:
        from kaos_nlp_core.locale_data import WRITTEN_NUMBER_MAP

        assert _parse_written_number("twenty-foo", WRITTEN_NUMBER_MAP["en"]) is None


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------


class TestEdgeCases:
    def test_empty(self, extractor: AlphaNumberExtractor) -> None:
        assert list(extractor.extract_spans("")) == []

    def test_pure_punctuation(self, extractor: AlphaNumberExtractor) -> None:
        assert list(extractor.extract_spans("... !!! ???")) == []

    def test_no_numbers(self, extractor: AlphaNumberExtractor) -> None:
        assert list(extractor.extract_spans("The cat sat on the mat.")) == []

    def test_large_number(self, extractor: AlphaNumberExtractor) -> None:
        spans = list(extractor.extract_spans("The population was 1,234,567,890."))
        assert spans[0].value == Decimal("1234567890")


# ----------------------------------------------------------------------
# Helper methods
# ----------------------------------------------------------------------


class TestHelperMethods:
    def test_extract_first_value(self, extractor: AlphaNumberExtractor) -> None:
        val = extractor.extract_first_value("Chapter 3 of volume IV has 100 pages.")
        assert val == Decimal("3")

    def test_extract_last_value(self, extractor: AlphaNumberExtractor) -> None:
        val = extractor.extract_last_value("Chapter 3 of volume IV has 100 pages.")
        assert val == Decimal("100")

    def test_extract_values(self, extractor: AlphaNumberExtractor) -> None:
        values = list(extractor.extract_values("1 and 2 and three"))
        assert values == [Decimal("1"), Decimal("2"), Decimal("3")]
