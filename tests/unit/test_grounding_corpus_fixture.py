"""Unit-level sanity for the grounding-corpus fixture used by WS-1.

These tests do not call any LLM provider. They verify that:

- Every doc file in the fixture is ASCII (per ``docs/design/grounding-actual-state.md §4``).
- The JSONL golden set parses and covers exactly 10 answerable + 10 unanswerable.
- Every answerable question's ``expected_doc_id`` names an existing doc.
- Every ``diagnostic_char_span`` lies within its document and points to
  non-empty text.

Lives under ``tests/unit/`` so it runs on every push, independent of live-API
credentials.
"""

from __future__ import annotations

import json
import pathlib

import pytest

_CORPUS_DIR = pathlib.Path(__file__).resolve().parent.parent / "fixtures" / "grounding-corpus"

_URI_TO_FILE = {
    "doc:grounding/delaware-gcl": "delaware-gcl.txt",
    "doc:grounding/first-amendment": "first-amendment.txt",
    "doc:grounding/rfc-2119": "rfc-2119.txt",
    "doc:grounding/fair-use-107": "fair-use-107.txt",
    "doc:grounding/rule-10b-5": "rule-10b-5.txt",
    "doc:grounding/apollo-11": "apollo-11.txt",
    "doc:grounding/gdpr-art-17": "gdpr-art-17.txt",
    "doc:grounding/voyager": "voyager-fact-sheet.txt",
    "doc:grounding/miranda": "miranda-holding.txt",
    "doc:grounding/nist-password": "nist-password-guidance.txt",
}


@pytest.mark.unit
class TestGroundingCorpusFixture:
    def test_ten_docs_present_and_ascii(self) -> None:
        for uri, fname in _URI_TO_FILE.items():
            path = _CORPUS_DIR / fname
            assert path.is_file(), f"missing corpus doc: {path}"
            text = path.read_text()
            assert text.strip(), f"empty corpus doc: {uri}"
            assert text.isascii(), (
                f"corpus doc {uri} ({fname}) contains non-ASCII characters. "
                "Fixtures must stay ASCII: _normalize_unicode only runs for "
                "NORMALIZED_TOKEN strategy, so curly quotes / em-dashes cause "
                "spurious FUZZY_* verification failures."
            )

    def test_questions_jsonl_parses(self) -> None:
        path = _CORPUS_DIR / "grounding-questions.jsonl"
        assert path.is_file(), f"missing questions: {path}"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert len(rows) == 20, f"expected 20 questions, got {len(rows)}"

        answerable = [r for r in rows if r["answerable"]]
        unanswerable = [r for r in rows if not r["answerable"]]
        assert len(answerable) == 10, f"expected 10 answerable, got {len(answerable)}"
        assert len(unanswerable) == 10, f"expected 10 unanswerable, got {len(unanswerable)}"

        ids = {r["id"] for r in rows}
        assert len(ids) == 20, f"duplicate question ids: {ids}"

    def test_answerable_questions_point_at_valid_docs(self) -> None:
        rows = [
            json.loads(line)
            for line in (_CORPUS_DIR / "grounding-questions.jsonl").read_text().splitlines()
            if line.strip()
        ]
        for row in rows:
            if not row["answerable"]:
                continue
            uri = row.get("expected_doc_id")
            assert uri in _URI_TO_FILE, (
                f"question {row['id']} points at unknown doc {uri!r}. "
                f"Known docs: {sorted(_URI_TO_FILE)}"
            )
            text = (_CORPUS_DIR / _URI_TO_FILE[uri]).read_text()
            span = row.get("diagnostic_char_span")
            assert span is not None, f"answerable {row['id']} missing diagnostic_char_span"
            start, end = span
            assert 0 <= start < end <= len(text), (
                f"{row['id']}: char_span {span} out of range for {uri} (len={len(text)})"
            )
            snippet = text[start:end]
            assert snippet.strip(), f"{row['id']}: diagnostic span is whitespace"

    def test_unanswerable_questions_have_reason(self) -> None:
        rows = [
            json.loads(line)
            for line in (_CORPUS_DIR / "grounding-questions.jsonl").read_text().splitlines()
            if line.strip()
        ]
        for row in rows:
            if row["answerable"]:
                continue
            reason = row.get("reason")
            assert reason and reason.strip(), (
                f"unanswerable {row['id']} missing 'reason' field; "
                "unanswerable questions must explain why the corpus can't support them"
            )
