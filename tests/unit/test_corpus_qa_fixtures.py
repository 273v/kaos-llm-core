"""Fixture-shape tests for the FUND-6 expanded corpus.

NOT a live LLM test — just verifies:
1. Each fixture set has the expected number of documents.
2. Each JSONL file parses without errors.
3. Q/A records have the required fields.
4. Total docs across all sets reaches the 30-doc target.
"""

from __future__ import annotations

import json
from pathlib import Path

_FIXTURE_BASE = Path(__file__).parent.parent / "fixtures"


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class TestGroundingCorpus:
    _dir = _FIXTURE_BASE / "grounding-corpus"

    def test_fixture_dir_exists(self) -> None:
        assert self._dir.exists()

    def test_has_minimum_documents(self) -> None:
        txt_files = list(self._dir.glob("*.txt"))
        assert len(txt_files) >= 10, f"Expected ≥10 txt files, got {len(txt_files)}"

    def test_questions_are_valid(self) -> None:
        questions = _load_jsonl(self._dir / "grounding-questions.jsonl")
        assert len(questions) >= 20
        required_keys = {"id", "answerable", "question"}
        for q in questions:
            missing = required_keys - q.keys()
            assert not missing, f"{q.get('id')}: missing {missing}"


class TestMultiformatCorpus:
    _dir = _FIXTURE_BASE / "multiformat-corpus"

    def test_fixture_dir_exists(self) -> None:
        assert self._dir.exists()

    def test_has_minimum_documents(self) -> None:
        content_files = [
            f for f in self._dir.iterdir() if f.suffix in (".txt", ".md", ".html", ".pdf", ".docx")
        ]
        assert len(content_files) >= 10, f"Expected ≥10 content files, got {len(content_files)}"

    def test_questions_are_valid(self) -> None:
        questions = _load_jsonl(self._dir / "multiformat-questions.jsonl")
        assert len(questions) >= 12
        for q in questions:
            assert "id" in q
            assert "answerable" in q


class TestExpandedCorpus:
    _dir = _FIXTURE_BASE / "expanded-corpus"

    def test_fixture_dir_exists(self) -> None:
        assert self._dir.exists()

    def test_has_minimum_documents(self) -> None:
        txt_files = list(self._dir.glob("*.txt"))
        assert len(txt_files) >= 4, f"Expected ≥4 txt files, got {len(txt_files)}"

    def test_questions_are_valid(self) -> None:
        questions = _load_jsonl(self._dir / "expanded-qa-golden.jsonl")
        assert len(questions) >= 16
        required_keys = {"id", "answerable", "question", "expected_doc_id"}
        for q in questions:
            missing = required_keys - q.keys()
            assert not missing, f"{q.get('id')}: missing {missing}"

    def test_answerable_unanswerable_balance(self) -> None:
        questions = _load_jsonl(self._dir / "expanded-qa-golden.jsonl")
        answerable = sum(1 for q in questions if q["answerable"])
        unanswerable = sum(1 for q in questions if not q["answerable"])
        assert answerable >= 12
        assert unanswerable >= 4


class TestCuadSample:
    _dir = _FIXTURE_BASE / "cuad-sample"

    def test_fixture_dir_exists(self) -> None:
        assert self._dir.exists()

    def test_has_contracts(self) -> None:
        txt_files = list(self._dir.glob("*.txt"))
        assert len(txt_files) >= 5


class TestOverallDocCount:
    def test_total_reaches_30(self) -> None:
        """The 30-doc target across all fixture sets."""
        grounding = len(list((_FIXTURE_BASE / "grounding-corpus").glob("*.txt")))
        multiformat = len(
            [
                f
                for f in (_FIXTURE_BASE / "multiformat-corpus").iterdir()
                if f.suffix in (".txt", ".md", ".html", ".pdf", ".docx")
            ]
        )
        expanded = len(list((_FIXTURE_BASE / "expanded-corpus").glob("*.txt")))
        cuad = len(list((_FIXTURE_BASE / "cuad-sample").glob("*.txt")))
        total = grounding + multiformat + expanded + cuad
        assert total >= 30, (
            f"Expected ≥30 total docs, got {total} "
            f"(grounding={grounding}, multiformat={multiformat}, "
            f"expanded={expanded}, cuad={cuad})"
        )

    def test_total_qa_triples_sufficient(self) -> None:
        """Need ≥48 Q/A triples for meaningful regression detection."""
        q_grounding = _load_jsonl(_FIXTURE_BASE / "grounding-corpus" / "grounding-questions.jsonl")
        q_multiformat = _load_jsonl(
            _FIXTURE_BASE / "multiformat-corpus" / "multiformat-questions.jsonl"
        )
        q_expanded = _load_jsonl(_FIXTURE_BASE / "expanded-corpus" / "expanded-qa-golden.jsonl")
        total = len(q_grounding) + len(q_multiformat) + len(q_expanded)
        assert total >= 48, f"Expected ≥48 Q/A triples, got {total}"
