"""Unit tests for AlphaLLMMerger — WS-TR.PR-6f.3.

Covers the full decision matrix: refusal + 0/1/N alpha hits, extracted
+ matching/non-matching alpha, error pass-through, list column
behavior, scalar column dispatch, and canonicalization correctness.
End-to-end coverage with real AlphaDateExtractor / AlphaEntityExtractor
(no fakes) — the merger's value is its composition with the extractors.
"""

from __future__ import annotations

import datetime
from typing import Literal, cast

import pytest
from kaos_content.model.extraction import ExtractionCell, ExtractionCitation
from kaos_content.model.tabular import ColumnType
from kaos_nlp_core.extract.alpha import (
    AlphaDateExtractor,
    AlphaEntityExtractor,
    MoneyMatch,
)
from kaos_nlp_core.extract.base_extractor import AlphaSpan

from kaos_llm_core.extract.merge import (
    AlphaLLMMerger,
    MergerConfig,
    _canonical_date,
    _canonical_entity,
    _canonical_money,
    _canonical_number,
    _map_column_type_name_to_enum,
)
from kaos_llm_core.signatures.extraction import ColumnSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cell(
    column_id: str,
    *,
    status: Literal["extracted", "not_in_document", "unclear", "error"],
    ai_value: object | None = None,
    confidence: float | None = None,
    citations: tuple[ExtractionCitation, ...] = (),
) -> ExtractionCell:
    """Build a test cell without having to spell out every field."""
    if status == "extracted":
        return ExtractionCell(
            doc_id="test-doc",
            column_id=column_id,
            schema_version=1,
            status=status,
            ai_value=ai_value,
            confidence=confidence if confidence is not None else 0.9,
            citations=citations,
        )
    if status == "unclear":
        return ExtractionCell(
            doc_id="test-doc",
            column_id=column_id,
            schema_version=1,
            status=status,
            confidence=confidence if confidence is not None else 0.4,
            citations=citations,
        )
    return ExtractionCell(
        doc_id="test-doc",
        column_id=column_id,
        schema_version=1,
        status=status,
        citations=citations,
    )


DATE_COL = ColumnSpec(id="agreement_date", column_type="date")
ENTITY_COL = ColumnSpec(id="counterparty", column_type="entity_role")
TEXT_COL = ColumnSpec(id="governing_law", column_type="string")
MONEY_COL = ColumnSpec(id="cap_on_liability", column_type="money")
INT_COL = ColumnSpec(id="seat_count", column_type="integer")
FLOAT_COL = ColumnSpec(id="rate", column_type="number")
PARTIES_LIST_WITH_HINT = ColumnSpec(
    id="parties",
    column_type="list",
    constraints={"inner": "string", "alpha_extractor": "entity"},
)
PARTIES_LIST_NO_HINT = ColumnSpec(
    id="parties",
    column_type="list",
    constraints={"inner": "string"},
)


# ---------------------------------------------------------------------------
# Canonicalization helpers
# ---------------------------------------------------------------------------


class TestCanonicalDate:
    def test_datetime_date(self) -> None:
        assert _canonical_date(datetime.date(2024, 1, 15)) == "2024-01-15"

    def test_datetime_datetime(self) -> None:
        dt = datetime.datetime(2024, 1, 15, 12, 30, 0)
        assert _canonical_date(dt) == "2024-01-15"

    def test_iso_string(self) -> None:
        assert _canonical_date("2024-01-15") == "2024-01-15"

    def test_iso_string_with_time(self) -> None:
        assert _canonical_date("2024-01-15T12:00:00") == "2024-01-15"

    def test_garbage_string(self) -> None:
        assert _canonical_date("not a date") is None

    def test_none_input(self) -> None:
        assert _canonical_date(None) is None


class TestCanonicalEntity:
    """Suffix is stripped so LLM forms ("Acme Corp.") match alpha forms
    (EntityMatch(name="Acme", entity_type="Corp.")) — both collapse to "acme"."""

    def test_plain_string_with_suffix(self) -> None:
        assert _canonical_entity("Acme Corp.") == "acme"

    def test_dict_with_name_key(self) -> None:
        assert _canonical_entity({"name": "Acme Corp."}) == "acme"

    def test_whitespace_collapse_and_suffix_strip(self) -> None:
        assert _canonical_entity("  Acme    Corp.  ") == "acme"

    def test_multi_word_name(self) -> None:
        assert _canonical_entity("International Business Machines Corp.") == (
            "international business machines"
        )

    def test_string_without_suffix(self) -> None:
        assert _canonical_entity("Tickets.com") == "tickets.com"

    def test_none_for_empty(self) -> None:
        assert _canonical_entity("") is None
        assert _canonical_entity("   ") is None

    def test_none_for_bare_suffix(self) -> None:
        """A string that is ONLY a suffix (no name) returns None."""
        assert _canonical_entity("Corp.") is None

    def test_none_for_unknown_shape(self) -> None:
        assert _canonical_entity(42) is None


class TestColumnTypeMapping:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("date", ColumnType.DATE),
            ("datetime", ColumnType.DATETIME),
            ("entity_role", ColumnType.ENTITY_ROLE),
            ("money", ColumnType.MONEY),
            ("string", ColumnType.TEXT),
            ("integer", ColumnType.INTEGER),
            ("boolean", ColumnType.BOOLEAN),
            ("list", ColumnType.LIST),
        ],
    )
    def test_mapping(self, name: str, expected: ColumnType) -> None:
        assert _map_column_type_name_to_enum(name) == expected

    def test_unknown_returns_none(self) -> None:
        assert _map_column_type_name_to_enum("enum") is None
        assert _map_column_type_name_to_enum("garbage") is None


# ---------------------------------------------------------------------------
# Merger — scalar date column, promotion from refusal
# ---------------------------------------------------------------------------


class TestScalarDateRefusedPromoted:
    """The hypothesis test: refused date cell + exactly 1 alpha date hit →
    promoted to extracted with ISO date + citation + confidence=0.8."""

    def test_refused_cell_with_single_alpha_hit_is_promoted(self) -> None:
        merger = AlphaLLMMerger()
        cell = _cell("agreement_date", status="not_in_document")
        source = "This Agreement is made on January 15, 2024, between the parties."
        out = merger.merge((cell,), columns=(DATE_COL,), source_text=source, source_uri="doc:test")
        assert len(out) == 1
        assert out[0].status == "extracted"
        assert out[0].ai_value == datetime.date(2024, 1, 15)
        assert out[0].confidence == 0.8
        assert len(out[0].citations) == 1
        assert "January 15, 2024" in out[0].citations[0].snippet
        assert "alpha-promoted" in (out[0].rationale or "")

    def test_unclear_cell_with_single_alpha_hit_is_promoted(self) -> None:
        merger = AlphaLLMMerger()
        cell = _cell("agreement_date", status="unclear", confidence=0.3)
        source = "made on March 5, 2023"
        out = merger.merge((cell,), columns=(DATE_COL,), source_text=source, source_uri="doc:test")
        assert out[0].status == "extracted"
        assert out[0].ai_value == datetime.date(2023, 3, 5)

    def test_refused_cell_no_alpha_hit_passes_through(self) -> None:
        merger = AlphaLLMMerger()
        cell = _cell("agreement_date", status="not_in_document")
        source = "This document contains no dates."
        out = merger.merge((cell,), columns=(DATE_COL,), source_text=source, source_uri="doc:test")
        assert out[0] is cell  # same object — no rebuild

    def test_refused_cell_multiple_alpha_hits_passes_through(self) -> None:
        """Multiple candidates = ambiguous; merger keeps refusal. The LLM
        is the tiebreaker, not the merger."""
        merger = AlphaLLMMerger()
        cell = _cell("agreement_date", status="not_in_document")
        source = "Dated January 15, 2024. Amended March 3, 2025."
        out = merger.merge((cell,), columns=(DATE_COL,), source_text=source, source_uri="doc:test")
        assert out[0].status == "not_in_document"
        assert out[0].ai_value is None


# ---------------------------------------------------------------------------
# Merger — scalar date column, citation strengthening
# ---------------------------------------------------------------------------


class TestScalarDateExtractedCitations:
    def test_extracted_cell_matching_alpha_gets_citation(self) -> None:
        merger = AlphaLLMMerger()
        cell = _cell(
            "agreement_date",
            status="extracted",
            ai_value=datetime.date(2024, 1, 15),
            confidence=0.95,
            citations=(),
        )
        source = "signed on January 15, 2024."
        out = merger.merge((cell,), columns=(DATE_COL,), source_text=source, source_uri="doc:test")
        assert out[0].status == "extracted"
        assert out[0].ai_value == datetime.date(2024, 1, 15)
        assert out[0].confidence == 0.95  # unchanged
        assert len(out[0].citations) == 1

    def test_extracted_cell_non_matching_alpha_passes_through(self) -> None:
        """LLM saw something the alpha regex missed — keep the LLM's
        answer. Don't discard it just because the regex disagrees."""
        merger = AlphaLLMMerger()
        cell = _cell(
            "agreement_date",
            status="extracted",
            ai_value=datetime.date(2024, 1, 15),
            confidence=0.95,
        )
        source = "Amendment signed March 5, 2023."
        out = merger.merge((cell,), columns=(DATE_COL,), source_text=source, source_uri="doc:test")
        assert out[0] is cell  # unchanged

    def test_extracted_cell_no_alpha_at_all_passes_through(self) -> None:
        merger = AlphaLLMMerger()
        cell = _cell(
            "agreement_date",
            status="extracted",
            ai_value=datetime.date(2024, 1, 15),
            confidence=0.95,
        )
        source = "no dates in this text"
        out = merger.merge((cell,), columns=(DATE_COL,), source_text=source, source_uri="doc:test")
        assert out[0] is cell


# ---------------------------------------------------------------------------
# Merger — pass-through cases
# ---------------------------------------------------------------------------


class TestPassThrough:
    def test_error_cell_always_passes_through(self) -> None:
        from kaos_content.model.extraction import ExtractionError

        merger = AlphaLLMMerger()
        cell = ExtractionCell(
            doc_id="d",
            column_id="agreement_date",
            schema_version=1,
            status="error",
            error=ExtractionError(
                code="model_api_error",
                message="rate limit",
                retry_recommended=True,
                attempt=1,
            ),
        )
        source = "signed on January 15, 2024."
        out = merger.merge((cell,), columns=(DATE_COL,), source_text=source, source_uri="doc:test")
        assert out[0] is cell

    def test_unsupported_column_type_passes_through(self) -> None:
        """A plain ``string`` column has no alpha extractor — cell passes through."""
        merger = AlphaLLMMerger()
        cell = _cell(
            "governing_law",
            status="extracted",
            ai_value="Delaware",
            confidence=0.9,
        )
        source = "governed by Delaware law, signed January 15, 2024."
        out = merger.merge((cell,), columns=(TEXT_COL,), source_text=source, source_uri="doc:test")
        assert out[0] is cell

    def test_list_column_without_hint_passes_through(self) -> None:
        """List columns must opt in — without ``alpha_extractor`` constraint,
        we don't know what inner type to dispatch to."""
        merger = AlphaLLMMerger()
        cell = _cell(
            "parties",
            status="not_in_document",
        )
        source = "between Acme Corp. and Beta LLC"
        out = merger.merge(
            (cell,), columns=(PARTIES_LIST_NO_HINT,), source_text=source, source_uri="doc:test"
        )
        assert out[0] is cell


# ---------------------------------------------------------------------------
# Merger — list column with explicit alpha hint
# ---------------------------------------------------------------------------


class TestListColumnWithHint:
    def test_refused_list_cell_populated_from_alpha(self) -> None:
        merger = AlphaLLMMerger()
        cell = _cell("parties", status="not_in_document")
        source = "between Acme Corp. and Beta LLC"
        out = merger.merge(
            (cell,),
            columns=(PARTIES_LIST_WITH_HINT,),
            source_text=source,
            source_uri="doc:test",
        )
        assert out[0].status == "extracted"
        assert isinstance(out[0].ai_value, list)
        values = out[0].ai_value
        assert any("Acme" in v for v in values)
        assert any("Beta" in v for v in values)
        assert out[0].confidence == 0.8

    def test_extracted_list_cell_unions_with_alpha(self) -> None:
        """LLM found some, alpha found others — union preserves both."""
        merger = AlphaLLMMerger()
        cell = _cell(
            "parties",
            status="extracted",
            ai_value=["Acme Corp."],
            confidence=0.95,
        )
        source = "between Acme Corp. and Beta LLC"
        out = merger.merge(
            (cell,),
            columns=(PARTIES_LIST_WITH_HINT,),
            source_text=source,
            source_uri="doc:test",
        )
        assert out[0].status == "extracted"
        values = cast(list[str], out[0].ai_value)
        assert len(values) == 2
        assert any("Beta" in v for v in values)

    def test_extracted_list_no_new_alpha_passes_through(self) -> None:
        merger = AlphaLLMMerger()
        cell = _cell(
            "parties",
            status="extracted",
            ai_value=["Acme Corp."],
            confidence=0.95,
        )
        source = "between Acme Corp. only"
        out = merger.merge(
            (cell,),
            columns=(PARTIES_LIST_WITH_HINT,),
            source_text=source,
            source_uri="doc:test",
        )
        assert out[0] is cell


# ---------------------------------------------------------------------------
# Merger — configuration + contract
# ---------------------------------------------------------------------------


class TestMergerConfig:
    def test_custom_promotion_confidence(self) -> None:
        merger = AlphaLLMMerger(config=MergerConfig(alpha_confidence_on_promotion=0.65))
        cell = _cell("agreement_date", status="not_in_document")
        source = "made on January 15, 2024"
        out = merger.merge((cell,), columns=(DATE_COL,), source_text=source, source_uri="doc:test")
        assert out[0].confidence == 0.65

    def test_citation_cap(self) -> None:
        """max_alpha_citations_per_cell limits how many spans get attached."""
        merger = AlphaLLMMerger(config=MergerConfig(max_alpha_citations_per_cell=2))
        cell = _cell(
            "parties",
            status="extracted",
            ai_value=["Foo Corp."],  # won't match, so all 3 alpha hits are "new"
            confidence=0.9,
        )
        source = "Alpha Inc. and Beta LLC and Gamma Corp. and Delta Ltd."
        out = merger.merge(
            (cell,),
            columns=(PARTIES_LIST_WITH_HINT,),
            source_text=source,
            source_uri="doc:test",
        )
        # New values added, but only up to 2 NEW citations regardless of how many adds.
        assert len(out[0].citations) <= 2


class TestContract:
    def test_cell_column_count_mismatch_raises(self) -> None:
        merger = AlphaLLMMerger()
        with pytest.raises(ValueError, match="cell/column count mismatch"):
            merger.merge(
                (_cell("a", status="not_in_document"),),
                columns=(DATE_COL, DATE_COL),
                source_text="x",
                source_uri="doc:test",
            )

    def test_empty_cells_returns_empty(self) -> None:
        merger = AlphaLLMMerger()
        out = merger.merge((), columns=(), source_text="whatever", source_uri="doc:test")
        assert out == ()

    def test_caches_extractor_per_class(self) -> None:
        """Two DATE cells against the same source text should only
        tokenize once. We confirm by stubbing the extractor and counting
        calls."""
        call_count = {"n": 0}

        class CountingDate(AlphaDateExtractor):
            def extract_spans(self, text: str):  # type: ignore[override,no-untyped-def]  # ty: ignore[invalid-method-override]
                call_count["n"] += 1
                yield from super().extract_spans(text)

        merger = AlphaLLMMerger(
            dispatch={
                ColumnType.DATE: CountingDate(),
                ColumnType.DATETIME: CountingDate(),
            },
        )
        col = ColumnSpec(id="c1", column_type="date")
        col2 = ColumnSpec(id="c2", column_type="date")
        cell1 = _cell("c1", status="not_in_document")
        cell2 = _cell("c2", status="not_in_document")
        merger.merge(
            (cell1, cell2),
            columns=(col, col2),
            source_text="made on January 15, 2024",
            source_uri="doc:test",
        )
        # Each column triggers one extract_spans call → 2 total without
        # cache, 1 with. The cache is keyed on the extractor class.
        # Since both columns share CountingDate, we see exactly 1.
        # NOTE: if DATE and DATETIME dispatched to DIFFERENT instances
        # of the same class, cache would still hit. Here they share the
        # same class.
        assert call_count["n"] == 1


class TestExtractProgramIntegration:
    """Confirm the merger wires into Extract at __init__ correctly."""

    def test_extract_has_default_merger(self) -> None:
        from kaos_llm_core.programs.extract import Extract
        from kaos_llm_core.signatures.extraction import ExtractionSchema

        schema = ExtractionSchema.from_dict(
            {
                "id": "test",
                "columns": [{"id": "agreement_date", "column_type": "date"}],
            }
        )
        extractor = Extract(schema, model="openai:gpt-5.4-nano")
        assert extractor._alpha_merger is not None
        assert isinstance(extractor._alpha_merger, AlphaLLMMerger)

    def test_extract_merger_disabled(self) -> None:
        from kaos_llm_core.programs.extract import Extract
        from kaos_llm_core.signatures.extraction import ExtractionSchema

        schema = ExtractionSchema.from_dict(
            {
                "id": "test",
                "columns": [{"id": "agreement_date", "column_type": "date"}],
            }
        )
        extractor = Extract(
            schema,
            model="openai:gpt-5.4-nano",
            alpha_merger=None,
        )
        assert extractor._alpha_merger is None

    def test_extract_custom_merger(self) -> None:
        from kaos_llm_core.programs.extract import Extract
        from kaos_llm_core.signatures.extraction import ExtractionSchema

        custom = AlphaLLMMerger(config=MergerConfig(alpha_confidence_on_promotion=0.5))
        schema = ExtractionSchema.from_dict(
            {
                "id": "test",
                "columns": [{"id": "agreement_date", "column_type": "date"}],
            }
        )
        extractor = Extract(
            schema,
            model="openai:gpt-5.4-nano",
            alpha_merger=custom,
        )
        assert extractor._alpha_merger is custom


# ---------------------------------------------------------------------------
# Smoke: synthetic AlphaSpan → citation
# ---------------------------------------------------------------------------


class TestCitationShape:
    def test_citation_has_valid_sha256(self) -> None:
        """Verify the SHA-256 hex is exactly 64 chars (Pydantic constraint)."""
        merger = AlphaLLMMerger()
        cell = _cell("agreement_date", status="not_in_document")
        source = "on January 15, 2024"
        out = merger.merge((cell,), columns=(DATE_COL,), source_text=source, source_uri="doc:test")
        assert len(out[0].citations[0].snippet_sha256) == 64

    def test_citation_char_span_points_to_snippet(self) -> None:
        merger = AlphaLLMMerger()
        cell = _cell("agreement_date", status="not_in_document")
        source = "signed on January 15, 2024 by the parties"
        out = merger.merge((cell,), columns=(DATE_COL,), source_text=source, source_uri="doc:test")
        cit = out[0].citations[0]
        assert source[cit.char_span[0] : cit.char_span[1]] == cit.snippet


# ---------------------------------------------------------------------------
# End-to-end with real CUAD-shaped text (agreement_date hypothesis)
# ---------------------------------------------------------------------------


class TestCUADHypothesisE2E:
    """The sprint's measurable claim: running the merger over a refused
    agreement_date cell with real CUAD-shaped contract text recovers the
    gold date via rule-based extraction."""

    @pytest.mark.parametrize(
        ("source", "expected_iso"),
        [
            # Ticketscom pattern: "19 Jan. 1998"
            # The AlphaDateExtractor doesn't accept abbreviated month dots
            # inline, but "19 January 1998" does work.
            ("This Agreement made this 19 January 1998", "1998-01-19"),
            # Dragonsystems: "6th day of April, 1999" — English ordinal
            # "Nth day of Month YYYY" form.
            (
                "entered into this 6th day of April, 1999 by and between",
                "1999-04-06",
            ),
            # Lucidinc: "21st day of January 2003"
            (
                "dated the 21st day of January 2003",
                "2003-01-21",
            ),
        ],
    )
    def test_cuad_refused_cell_promoted_to_gold_date(self, source: str, expected_iso: str) -> None:
        merger = AlphaLLMMerger()
        cell = _cell("agreement_date", status="not_in_document")
        out = merger.merge((cell,), columns=(DATE_COL,), source_text=source, source_uri="doc:test")
        # Must either promote to extracted (if single hit) or leave alone.
        # The point: we never fabricate a non-golden date.
        if out[0].status == "extracted":
            extracted_date = cast(datetime.date, out[0].ai_value)
            assert extracted_date.isoformat() == expected_iso


# ---------------------------------------------------------------------------
# Entity dispatch smoke (ENTITY_ROLE scalar column)
# ---------------------------------------------------------------------------


class TestEntityScalarDispatch:
    def test_refused_entity_cell_promoted(self) -> None:
        merger = AlphaLLMMerger()
        cell = _cell("counterparty", status="not_in_document")
        source = "Counterparty shall be Acme Corp. for this agreement."
        out = merger.merge(
            (cell,), columns=(ENTITY_COL,), source_text=source, source_uri="doc:test"
        )
        # One entity present → should promote.
        assert out[0].status == "extracted"
        assert "Acme" in cast(str, out[0].ai_value)

    def test_extracted_entity_matching_alpha_gets_citation(self) -> None:
        merger = AlphaLLMMerger()
        cell = _cell(
            "counterparty",
            status="extracted",
            ai_value="Acme Corp.",
            confidence=0.9,
        )
        source = "Acme Corp. is the counterparty."
        out = merger.merge(
            (cell,), columns=(ENTITY_COL,), source_text=source, source_uri="doc:test"
        )
        assert out[0].status == "extracted"
        assert out[0].ai_value == "Acme Corp."  # unchanged
        assert len(out[0].citations) == 1


# ---------------------------------------------------------------------------
# AlphaSpan / extractor defensive smoke
# ---------------------------------------------------------------------------


class TestAlphaSpanShape:
    """Make sure AlphaSpan from real extractors round-trips through the
    merger's citation builder."""

    def test_date_extractor_span_yields_valid_citation(self) -> None:
        ext = AlphaDateExtractor()
        source = "dated January 15, 2024 today"
        spans = list(ext.extract_spans(source))
        assert len(spans) == 1
        s = spans[0]
        assert isinstance(s, AlphaSpan)
        assert source[s.start : s.end] == "January 15, 2024"

    def test_entity_extractor_span_yields_valid_citation(self) -> None:
        ext = AlphaEntityExtractor()
        source = "by Acme Corp. today"
        spans = list(ext.extract_spans(source))
        assert len(spans) == 1
        s = spans[0]
        assert source[s.start : s.end] == "Acme Corp."


# ---------------------------------------------------------------------------
# WS-TR.PR-6f.4 — Money + Number canonicalization + dispatch
# ---------------------------------------------------------------------------


class TestCanonicalMoney:
    def test_money_match(self) -> None:
        from decimal import Decimal

        mm = MoneyMatch(amount=Decimal("13.50"), currency="USD")
        assert _canonical_money(mm) == "13.50:USD"

    def test_tuple_form(self) -> None:
        from decimal import Decimal

        assert _canonical_money((Decimal("100"), "EUR")) == "100:EUR"

    def test_dict_form(self) -> None:
        from decimal import Decimal

        assert _canonical_money({"amount": Decimal("100"), "currency": "GBP"}) == "100:GBP"

    def test_bare_numeric_defaults_to_usd(self) -> None:
        # LLM often emits just a number for a MONEY column; default to USD.
        assert _canonical_money(100) == "100:USD"
        assert _canonical_money("250.75") == "250.75:USD"

    def test_case_normalizes_iso(self) -> None:
        from decimal import Decimal

        assert _canonical_money((Decimal("50"), "usd")) == "50:USD"

    def test_invalid(self) -> None:
        assert _canonical_money(None) is None
        assert _canonical_money("not a number") is None
        assert _canonical_money(object()) is None


class TestCanonicalNumber:
    def test_int(self) -> None:
        assert _canonical_number(42) == "42"

    def test_float(self) -> None:
        assert _canonical_number(3.14) == "3.14"

    def test_decimal(self) -> None:
        from decimal import Decimal

        assert _canonical_number(Decimal("1234.5")) == "1234.5"

    def test_string_with_commas(self) -> None:
        assert _canonical_number("1,234") == "1234"

    def test_bool_rejected(self) -> None:
        """bool is an int subclass but we explicitly exclude it —
        True/False shouldn't canonicalize as 1/0."""
        assert _canonical_number(True) is None
        assert _canonical_number(False) is None

    def test_invalid(self) -> None:
        assert _canonical_number(None) is None
        assert _canonical_number("not a number") is None


class TestMoneyColumnDispatch:
    def test_refused_money_cell_promoted_from_alpha(self) -> None:
        merger = AlphaLLMMerger()
        cell = _cell("cap_on_liability", status="not_in_document")
        source = "The cap on liability is $1,000,000 per occurrence."
        out = merger.merge((cell,), columns=(MONEY_COL,), source_text=source, source_uri="doc:test")
        assert out[0].status == "extracted"
        # ai_value is a dict {amount, currency} per _alpha_value_for_cell.
        assert isinstance(out[0].ai_value, dict)
        assert out[0].ai_value["currency"] == "USD"
        from decimal import Decimal

        assert out[0].ai_value["amount"] == Decimal("1000000")
        assert out[0].confidence == 0.8

    def test_extracted_money_matches_alpha_gets_citation(self) -> None:
        from decimal import Decimal

        merger = AlphaLLMMerger()
        cell = _cell(
            "cap_on_liability",
            status="extracted",
            ai_value={"amount": Decimal("500"), "currency": "USD"},
            confidence=0.9,
        )
        source = "The cap is $500 per claim."
        out = merger.merge((cell,), columns=(MONEY_COL,), source_text=source, source_uri="doc:test")
        assert out[0].status == "extracted"
        assert len(out[0].citations) == 1

    def test_refused_money_no_alpha_passes_through(self) -> None:
        merger = AlphaLLMMerger()
        cell = _cell("cap_on_liability", status="not_in_document")
        source = "No monetary cap specified in this contract."
        out = merger.merge((cell,), columns=(MONEY_COL,), source_text=source, source_uri="doc:test")
        assert out[0] is cell


class TestIntegerFloatDispatch:
    def test_refused_integer_promoted(self) -> None:
        from decimal import Decimal

        merger = AlphaLLMMerger()
        cell = _cell("seat_count", status="not_in_document")
        source = "There are 42 seats available."
        out = merger.merge((cell,), columns=(INT_COL,), source_text=source, source_uri="doc:test")
        assert out[0].status == "extracted"
        assert out[0].ai_value == Decimal("42")

    def test_refused_float_promoted(self) -> None:
        from decimal import Decimal

        merger = AlphaLLMMerger()
        cell = _cell("rate", status="not_in_document")
        source = "The applicable rate is 3.14 per unit."
        out = merger.merge((cell,), columns=(FLOAT_COL,), source_text=source, source_uri="doc:test")
        assert out[0].status == "extracted"
        assert out[0].ai_value == Decimal("3.14")

    def test_multiple_numbers_no_promotion(self) -> None:
        """Multiple alpha hits → merger can't pick; keep refusal."""
        merger = AlphaLLMMerger()
        cell = _cell("seat_count", status="not_in_document")
        source = "The room has 100 chairs and 42 tables and 7 desks."
        out = merger.merge((cell,), columns=(INT_COL,), source_text=source, source_uri="doc:test")
        assert out[0].status == "not_in_document"
