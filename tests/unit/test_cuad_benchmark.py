"""Unit tests for the WS-TR.PR-5 CUAD calibration harness.

Covers the scoring logic + fixture loaders. No LLM calls — pure deterministic
tests that freeze the benchmark's judgment heuristics.
"""

from __future__ import annotations

import importlib.util

# Load the script module directly since scripts/ isn't in the package path.
# Register in sys.modules so @dataclass + type hints resolve correctly
# (Python 3.13's dataclass runtime reaches into sys.modules[cls.__module__]).
import sys as _sys
from pathlib import Path

import pytest

_BENCH_PATH = Path(__file__).resolve().parents[2] / "scripts" / "cuad_extraction_benchmark.py"
_spec = importlib.util.spec_from_file_location("cuad_bench", _BENCH_PATH)
assert _spec is not None and _spec.loader is not None
bench = importlib.util.module_from_spec(_spec)
_sys.modules["cuad_bench"] = bench
_spec.loader.exec_module(bench)


class TestCellMatches:
    def test_plain_substring(self) -> None:
        assert bench._cell_matches("Acme Corp", ["Acme Corp"])

    def test_ai_contains_gold(self) -> None:
        """AI value is broader than the gold span — still matches."""
        assert bench._cell_matches("Acme Corporation, a Delaware company", ["Acme Corp"])

    def test_gold_contains_ai(self) -> None:
        """AI value is a subspan of the gold span — still matches."""
        assert bench._cell_matches(
            "State of California", ["...laws of the State of California without..."]
        )

    def test_no_overlap_fails(self) -> None:
        assert not bench._cell_matches("Delaware", ["California"])

    def test_empty_gold_fails(self) -> None:
        assert not bench._cell_matches("anything", [])

    def test_none_ai_fails(self) -> None:
        assert not bench._cell_matches(None, ["something"])

    def test_list_value_fans_out(self) -> None:
        """Parties column is a list — match if ANY element matches ANY gold."""
        assert bench._cell_matches(["Acme Corp", "Beta LLC"], ["Beta LLC is the licensee"])

    def test_list_with_no_match_fails(self) -> None:
        assert not bench._cell_matches(["Alpha", "Beta"], ["Gamma Industries"])

    def test_case_insensitive_match(self) -> None:
        assert bench._cell_matches("acme corp", ["ACME Corp"])

    def test_whitespace_normalized(self) -> None:
        assert bench._cell_matches("Acme    Corp\n", ["Acme Corp"])

    def test_iso_date_matches_natural_language_gold(self) -> None:
        """The whole point of PR-5's date normalization."""
        assert bench._cell_matches("2025-01-15", ["January 15, 2025"])
        assert bench._cell_matches("1999-02-17", ["February 17, 1999"])

    def test_iso_date_with_different_day_does_not_match(self) -> None:
        assert not bench._cell_matches("2025-01-16", ["January 15, 2025"])

    def test_iso_date_with_different_month_does_not_match(self) -> None:
        assert not bench._cell_matches("2025-02-15", ["January 15, 2025"])

    def test_natural_date_falls_through_to_substring(self) -> None:
        """If AI value is natural language, fall back to substring match."""
        assert bench._cell_matches("January 15, 2025", ["15, 2025"])


class TestParseIsoDate:
    def test_valid_iso(self) -> None:
        assert bench._parse_iso_date("2025-01-15") == "2025-01-15"

    def test_natural_language_rejected(self) -> None:
        assert bench._parse_iso_date("January 15, 2025") is None

    def test_short_rejected(self) -> None:
        assert bench._parse_iso_date("2025-01") is None

    def test_extra_text_rejected(self) -> None:
        assert bench._parse_iso_date("2025-01-15 extra") is None


class TestParseNaturalDate:
    def test_month_day_year_comma(self) -> None:
        assert bench._parse_natural_date("February 17, 1999") == "1999-02-17"

    def test_month_day_year_no_comma(self) -> None:
        assert bench._parse_natural_date("February 17 1999") == "1999-02-17"

    def test_day_month_year(self) -> None:
        assert bench._parse_natural_date("17 February 1999") == "1999-02-17"

    def test_abbreviated_month(self) -> None:
        assert bench._parse_natural_date("Feb 17, 1999") == "1999-02-17"

    def test_garbage_returns_none(self) -> None:
        assert bench._parse_natural_date("asdf") is None

    def test_iso_date_not_natural(self) -> None:
        """The natural parser should not handle ISO (that's _parse_iso_date's job)."""
        assert bench._parse_natural_date("2025-01-15") is None

    def test_invalid_day_returns_none(self) -> None:
        assert bench._parse_natural_date("February 30, 2025") is None


class TestFixtureLoaders:
    def test_load_corpus_reads_five_docs(self) -> None:
        corpus = bench._load_corpus()
        assert len(corpus) == 5
        for doc_id, text in corpus.items():
            assert isinstance(doc_id, str)
            assert text  # Non-empty
            # CUAD contracts have real structure — not empty templates.
            assert len(text) >= 4000

    def test_load_golden_shape(self) -> None:
        golden = bench._load_golden()
        assert len(golden) == 5
        for _doc_id, clauses in golden.items():
            # Column-id keys match the CUAD_SCHEMA columns.
            assert set(clauses.keys()) == {
                "parties",
                "agreement_date",
                "governing_law",
                "termination_for_convenience",
                "cap_on_liability",
            }
            # At least the Parties list should have entries.
            assert len(clauses["parties"]) >= 1

    def test_corpus_and_golden_doc_ids_match(self) -> None:
        corpus_ids = set(bench._load_corpus().keys())
        golden_ids = set(bench._load_golden().keys())
        assert corpus_ids == golden_ids


class TestCuadSchemaCompiles:
    def test_schema_to_signature_bare(self) -> None:
        Sig = bench.CUAD_SCHEMA.to_signature(provenance="none")
        # source_text input + 5 output columns.
        assert set(Sig.model_fields.keys()) >= {
            "source_text",
            "parties",
            "agreement_date",
            "governing_law",
            "termination_for_convenience",
            "cap_on_liability",
        }

    def test_schema_to_signature_cited(self) -> None:
        Sig = bench.CUAD_SCHEMA.to_signature(provenance="cited")
        assert Sig is not None
        assert Sig.__name__.startswith("Extract_")


class TestModelReportMetrics:
    def test_precision_on_empty_report(self) -> None:
        report = bench.ModelReport(model="test", skipped=False)
        assert report.precision == 0.0
        assert report.refusal_rate == 0.0

    def test_precision_rounding(self) -> None:
        report = bench.ModelReport(
            model="test",
            skipped=False,
            n_cells=25,
            n_matched=20,
            n_refused=3,
        )
        assert report.precision == pytest.approx(0.80)
        assert report.refusal_rate == pytest.approx(0.12)
