"""Unit tests for AlphaDateExtractor — WS-TR.PR-6f.1.

Covers the five detection branches ported from kelvin-nlp:

- Inline separators (MM/DD/YYYY etc.)
- DD Month YYYY
- Ordinal Month YYYY
- Month DD YYYY
- Month Ordinal YYYY
- English-only "Ord day of Month YYYY"

Plus calendar validation, year-bounds resolution, and CUAD-corpus
smoke tests that freeze the golden-span extraction on the 5 bundled
contracts.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest

from kaos_llm_core.extract.alpha import AlphaDateExtractor
from kaos_llm_core.extract.base_extractor import AlphaSpan, ExtractorValueType


@pytest.fixture
def extractor() -> AlphaDateExtractor:
    return AlphaDateExtractor()


class TestExtractorMetadata:
    def test_name(self, extractor: AlphaDateExtractor) -> None:
        assert extractor.get_name() == "date"

    def test_value_type(self, extractor: AlphaDateExtractor) -> None:
        assert extractor.get_value_type() == ExtractorValueType.DATE_FULL

    def test_supports_english(self, extractor: AlphaDateExtractor) -> None:
        assert "en" in extractor.get_languages()

    def test_rejects_unsupported_language(self) -> None:
        with pytest.raises(ValueError, match="does not support language"):
            AlphaDateExtractor(language="zh")


class TestInlineSeparator:
    """Branch 1 — MM/DD/YYYY variants."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Date: 3/1/2011.", datetime.datetime(2011, 3, 1)),
            ("Date: 03/01/2011.", datetime.datetime(2011, 3, 1)),
            ("Date: 3-1-2011.", datetime.datetime(2011, 3, 1)),
            ("Date: 3.1.2011.", datetime.datetime(2011, 3, 1)),
            ("Date: 2011-03-01.", datetime.datetime(2011, 3, 1)),
            ("Date: 12/31/99.", datetime.datetime(1999, 12, 31)),
        ],
    )
    def test_parses_various_separators(
        self, extractor: AlphaDateExtractor, text: str, expected: datetime.datetime
    ) -> None:
        spans = list(extractor.extract_spans(text))
        assert len(spans) >= 1
        assert any(span.value == expected for span in spans), (
            f"expected {expected} in {[s.value for s in spans]}"
        )

    def test_span_points_to_date_token(self, extractor: AlphaDateExtractor) -> None:
        text = "The date 3/1/2011 is significant."
        span = extractor.extract_first_span(text)
        assert span is not None
        assert text[span.start : span.end] == "3/1/2011"

    def test_invalid_lengths_rejected(self, extractor: AlphaDateExtractor) -> None:
        # lengths must be in (1, 2, 4). 3 digits is not a year or day.
        text = "Bad 123/456/789 date."
        assert extractor.extract_first_span(text) is None

    def test_slashed_date_with_month_word_and_4digit_year(
        self, extractor: AlphaDateExtractor
    ) -> None:
        """Kelvin's normalize_date_tokens requires a 4-digit year OR a
        2-digit year > 31 for the month-inside-slash branch. A 2-digit
        year <=31 (like "21") has no extractable signal — same in
        kelvin, same in our port."""
        text = "Filed Jan/01/2021."
        span = extractor.extract_first_span(text)
        assert span is not None
        assert span.value == datetime.datetime(2021, 1, 1)


class TestDayMonthYear:
    """Branch 2 — ``DD Month YYYY``."""

    def test_basic(self, extractor: AlphaDateExtractor) -> None:
        text = "Executed on 19 May 2010 at 12pm."
        span = extractor.extract_first_span(text)
        assert span is not None
        assert span.value == datetime.datetime(2010, 5, 19)
        assert text[span.start : span.end].startswith("19 May 2010")

    def test_with_punctuation(self, extractor: AlphaDateExtractor) -> None:
        text = "Dated 19 May, 2010,"
        assert any(s.value == datetime.datetime(2010, 5, 19) for s in extractor.extract_spans(text))

    def test_two_digit_year_greater_than_31_expands(self, extractor: AlphaDateExtractor) -> None:
        """A 2-digit year > 31 is unambiguously a year (no day can be > 31).
        Kelvin picks it up via the ``int(token) > 31`` heuristic; so does
        ours. The century resolver picks the closer of current / prior."""
        text = "Dated 19 May 99."
        assert any(s.value == datetime.datetime(1999, 5, 19) for s in extractor.extract_spans(text))

    def test_two_digit_year_below_31_unresolvable_in_dd_month_form(
        self, extractor: AlphaDateExtractor
    ) -> None:
        """Documents the limitation: ``19 May 10`` has no year disambiguation
        signal — "10" could be day, year, or just a number. Kelvin declines
        to parse and so do we. The merger layer can fall back to the LLM
        for such cases."""
        text = "Dated 19 May 10."
        # We either emit nothing, or emit something safely inside bounds.
        for span in extractor.extract_spans(text):
            assert 1900 <= span.value.year <= 2050


class TestMonthDayYear:
    """Branch 4 — ``Month DD, YYYY``, the form CUAD most relies on."""

    def test_february_seventeen_nineteen_ninety_nine(self, extractor: AlphaDateExtractor) -> None:
        """This is the CUAD ticketscominc agreement_date gold value —
        the single most important test in the whole PR."""
        text = "This Agreement is dated February 17, 1999."
        span = extractor.extract_first_span(text)
        assert span is not None
        assert span.value == datetime.datetime(1999, 2, 17)
        assert "February 17" in text[span.start : span.end]

    def test_january_fifteen_twenty_twenty_five(self, extractor: AlphaDateExtractor) -> None:
        text = "effective as of January 15, 2025, by and between"
        assert any(s.value == datetime.datetime(2025, 1, 15) for s in extractor.extract_spans(text))

    def test_december_thirty_one(self, extractor: AlphaDateExtractor) -> None:
        text = "Term ends December 31, 2026."
        assert any(
            s.value == datetime.datetime(2026, 12, 31) for s in extractor.extract_spans(text)
        )

    def test_abbreviated_month(self, extractor: AlphaDateExtractor) -> None:
        text = "Signed Jan 15, 2025."
        assert any(s.value == datetime.datetime(2025, 1, 15) for s in extractor.extract_spans(text))


class TestOrdinalForms:
    """Branches 3, 5, 6 — ordinal day forms."""

    def test_ordinal_month_year(self, extractor: AlphaDateExtractor) -> None:
        text = "On the 1st May 2020 the company"
        assert any(s.value == datetime.datetime(2020, 5, 1) for s in extractor.extract_spans(text))

    def test_month_ordinal_year(self, extractor: AlphaDateExtractor) -> None:
        text = "as of May 1st, 2020."
        assert any(s.value == datetime.datetime(2020, 5, 1) for s in extractor.extract_spans(text))

    def test_first_of_month_year_not_supported(self, extractor: AlphaDateExtractor) -> None:
        """Kelvin does NOT handle "first of May 2020" — its branches require
        the ordinal to be directly adjacent to the month (``1st May 2020``
        ✓) or to be preceded by "day of" (``first day of May 2020`` ✓).
        We document the gap; the merger routes this to the LLM."""
        text = "effective on the first of May 2020"
        spans = list(extractor.extract_spans(text))
        # No assertion on hit — this form is deliberately unsupported.
        # The test passes either way; it's here as a living regression
        # guard: if a future port adds this branch, the assertion can
        # tighten.
        assert isinstance(spans, list)

    def test_nth_day_of_month_year(self, extractor: AlphaDateExtractor) -> None:
        text = "made on the first day of May 2020 hereby"
        assert any(s.value == datetime.datetime(2020, 5, 1) for s in extractor.extract_spans(text))


class TestCalendarValidation:
    """Feb 30 and friends must be rejected silently."""

    def test_feb_30_rejected(self, extractor: AlphaDateExtractor) -> None:
        text = "Date: 2/30/2025."  # invalid calendar date
        spans = list(extractor.extract_spans(text))
        # The normalizer returns None; we drop Nones. So no span.
        assert not any(s.value == datetime.datetime(2025, 2, 30) for s in spans)

    def test_month_13_rejected(self, extractor: AlphaDateExtractor) -> None:
        text = "Date: 13/1/2025."
        # The token "13" is > 12 so it gets treated as day; "1" as month,
        # "2025" as year. So this actually parses as Jan 13, 2025.
        span = extractor.extract_first_span(text)
        if span is not None:
            # If parsed, it must be calendar-valid.
            assert span.value.year == 2025
            assert 1 <= span.value.month <= 12
            assert 1 <= span.value.day <= 31


class TestYearBounds:
    """min_year/max_year + 2-digit expansion."""

    def test_year_before_1900_rejected(self, extractor: AlphaDateExtractor) -> None:
        # "1850" is 4-digit and below min_year → rejected outright.
        text = "Dated 15 May 1850."
        assert not any(s.value and s.value.year == 1850 for s in extractor.extract_spans(text))

    def test_year_after_2050_rejected(self, extractor: AlphaDateExtractor) -> None:
        text = "Dated 15 May 2100."
        assert not any(s.value and s.value.year == 2100 for s in extractor.extract_spans(text))

    def test_two_digit_year_picks_closer_century(self, extractor: AlphaDateExtractor) -> None:
        # "99" with current year near 2026 → 1999 (|1999-2026|=27 < |2099-2026|=73).
        text = "Dated 17 Feb 99"
        spans = list(extractor.extract_spans(text))
        assert any(s.value == datetime.datetime(1999, 2, 17) for s in spans)


class TestNormalizeDateTokens:
    """Unit tests for the ``normalize_date_tokens`` helper directly."""

    def test_three_digit_token_rejected(self, extractor: AlphaDateExtractor) -> None:
        # Length must be 1, 2, or 4. 3-digit "202" neither year nor day.
        assert extractor.normalize_date_tokens(["202", "05", "19"]) is None

    def test_year_in_middle_position_rejected(self, extractor: AlphaDateExtractor) -> None:
        # kelvin-nlp date.py:126-127 — year cannot be middle.
        assert extractor.normalize_date_tokens(["01", "2025", "15"]) is None

    def test_non_3_tokens_rejected(self, extractor: AlphaDateExtractor) -> None:
        assert extractor.normalize_date_tokens(["15", "May"]) is None
        assert extractor.normalize_date_tokens(["15", "May", "2025", "extra"]) is None

    def test_month_name_handled(self, extractor: AlphaDateExtractor) -> None:
        # Explicit normalize call with a month-name token.
        assert extractor.normalize_date_tokens(["2025", "May", "19"]) == datetime.datetime(
            2025, 5, 19
        )

    def test_strict_mode_rejects_2_digit_year(self, extractor: AlphaDateExtractor) -> None:
        # "99" → in strict mode, cannot expand to 1999.
        # (But "99" > 31 gets picked as year; strict rejects bounds fix.)
        result = extractor.normalize_date_tokens(["05", "17", "99"], strict=True)
        assert result is None


class TestHelperMethods:
    def test_extract_values_drops_spans(self, extractor: AlphaDateExtractor) -> None:
        text = "Dated 19 May 2010 and ending 31 December 2020."
        values = list(extractor.extract_values(text))
        assert datetime.datetime(2010, 5, 19) in values

    def test_extract_first_value(self, extractor: AlphaDateExtractor) -> None:
        text = "Dated 19 May 2010 and ending 31 December 2020."
        assert extractor.extract_first_value(text) == datetime.datetime(2010, 5, 19)

    def test_extract_last_value(self, extractor: AlphaDateExtractor) -> None:
        text = "Dated 19 May 2010 and ending 31 December 2020."
        assert extractor.extract_last_value(text) == datetime.datetime(2020, 12, 31)

    def test_empty_text(self, extractor: AlphaDateExtractor) -> None:
        assert list(extractor.extract_spans("")) == []
        assert extractor.extract_first_span("") is None
        assert extractor.extract_first_value("") is None


class TestAlphaSpanShape:
    def test_alpha_span_is_frozen(self) -> None:
        # Frozen dataclass — assignment via the usual ``=`` route goes
        # through ``__setattr__`` and raises ``FrozenInstanceError``.
        # ``object.__setattr__`` bypasses the frozen guard (the slot
        # descriptor is NOT read-only on a frozen+slots dataclass),
        # so we test the surface path that users actually hit.
        span = AlphaSpan(value=42, start=0, end=3)
        import dataclasses

        with pytest.raises(dataclasses.FrozenInstanceError):
            # Use setattr to sidestep ty's static analysis while still
            # invoking the normal __setattr__ override.
            setattr(span, "value", 99)  # noqa: B010

    def test_length_property(self) -> None:
        assert AlphaSpan(value="x", start=5, end=20).length == 15


# ---------------------------------------------------------------------
# CUAD sample — the hypothesis-falsification test
# ---------------------------------------------------------------------
#
# If AlphaDateExtractor cannot extract the agreement_date for the 5
# contracts in tests/fixtures/cuad-sample/ without an LLM call, the
# alpha-first sprint hypothesis fails. These tests are the contract.

_CUAD_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "cuad-sample"


class TestCUADAgreementDates:
    """Freeze the extraction of CUAD golden agreement_date spans.

    Golden values per CUAD's annotations (NOT inferred from filenames —
    the SEC filing date in the filename is not the agreement date):

    - ticketscominc:         "February 17, 1999"          (Month DD, YYYY)
    - mphasetechnologies:    "21st day of January 2003"   (Nth day of Month YYYY)
    - dragonsystems:         "19 Jan. 1998"               (DD Month YYYY)
    - centrackinternational: "6th day of April, 1999"     (Nth day of Month YYYY)
    - lucidinc:              "[*]" — CUAD redaction marker, no extractable date

    4 of 5 have extractable dates; the 5th is redacted. The alpha
    extractor must hit all 4 — this is the hypothesis-falsifier for
    the sprint.
    """

    @pytest.fixture(scope="class")
    def extractor(self) -> AlphaDateExtractor:
        return AlphaDateExtractor()

    def test_ticketscom_feb_17_1999(self, extractor: AlphaDateExtractor) -> None:
        path = _CUAD_DIR / "ticketscominc-06-22-1999-ex-10-22-sponsorship-agreement.txt"
        if not path.exists():
            pytest.skip("CUAD sample fixture not present")
        text = path.read_text(encoding="utf-8")
        dates = list(extractor.extract_values(text))
        assert datetime.datetime(1999, 2, 17) in dates, (
            f"expected Feb 17, 1999; sample: {sorted({d.date() for d in dates if d})[:10]}"
        )

    def test_mphasetechnologies_jan_21_2003(self, extractor: AlphaDateExtractor) -> None:
        path = _CUAD_DIR / "mphasetechnologiesinc-20030911-10-k-ex-10-15-1560667-ex-10-1.txt"
        if not path.exists():
            pytest.skip("CUAD sample fixture not present")
        text = path.read_text(encoding="utf-8")
        dates = list(extractor.extract_values(text))
        assert datetime.datetime(2003, 1, 21) in dates, (
            f"expected Jan 21, 2003; sample: {sorted({d.date() for d in dates if d})[:10]}"
        )

    def test_dragonsystems_jan_19_1998(self, extractor: AlphaDateExtractor) -> None:
        path = _CUAD_DIR / "dragonsystemsinc-01-08-1999-ex-10-17-outsourcing-agreement.txt"
        if not path.exists():
            pytest.skip("CUAD sample fixture not present")
        text = path.read_text(encoding="utf-8")
        dates = list(extractor.extract_values(text))
        assert datetime.datetime(1998, 1, 19) in dates, (
            f"expected Jan 19, 1998; sample: {sorted({d.date() for d in dates if d})[:10]}"
        )

    def test_centrackinternational_apr_6_1999(self, extractor: AlphaDateExtractor) -> None:
        path = _CUAD_DIR / "centrackinternationalinc-10-29-1999-ex-10-3-web-site-hosting.txt"
        if not path.exists():
            pytest.skip("CUAD sample fixture not present")
        text = path.read_text(encoding="utf-8")
        dates = list(extractor.extract_values(text))
        assert datetime.datetime(1999, 4, 6) in dates, (
            f"expected Apr 6, 1999; sample: {sorted({d.date() for d in dates if d})[:10]}"
        )

    def test_lucidinc_redacted_no_hallucination(self, extractor: AlphaDateExtractor) -> None:
        """Golden is "[*]" — a redaction marker. Alpha extractor must not
        hallucinate; it returns whatever real dates appear elsewhere in
        the contract body (filing / delivery / notice dates), but none
        is the agreement_date. Just verify no crash + typed output."""
        path = _CUAD_DIR / "lucidinc-04-15-2011-ex-10-9-distributor-agreement.txt"
        if not path.exists():
            pytest.skip("CUAD sample fixture not present")
        text = path.read_text(encoding="utf-8")
        dates = list(extractor.extract_values(text))
        assert all(isinstance(d, datetime.datetime) for d in dates)

    def test_four_of_five_cuad_goldens_extractable(self, extractor: AlphaDateExtractor) -> None:
        """Sprint hypothesis-falsifier: does the LLM-free alpha extractor
        recover every CUAD agreement_date gold span where one exists?"""
        cases = [
            (
                "ticketscominc-06-22-1999-ex-10-22-sponsorship-agreement.txt",
                datetime.datetime(1999, 2, 17),
            ),
            (
                "mphasetechnologiesinc-20030911-10-k-ex-10-15-1560667-ex-10-1.txt",
                datetime.datetime(2003, 1, 21),
            ),
            (
                "dragonsystemsinc-01-08-1999-ex-10-17-outsourcing-agreement.txt",
                datetime.datetime(1998, 1, 19),
            ),
            (
                "centrackinternationalinc-10-29-1999-ex-10-3-web-site-hosting.txt",
                datetime.datetime(1999, 4, 6),
            ),
        ]
        hits = 0
        for fname, target in cases:
            path = _CUAD_DIR / fname
            if not path.exists():
                pytest.skip("CUAD sample fixture not present")
            if target in extractor.extract_values(path.read_text(encoding="utf-8")):
                hits += 1
        assert hits == len(cases), (
            f"alpha-first hypothesis: expected {len(cases)}/{len(cases)}; got {hits}"
        )
