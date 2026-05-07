"""Credential-gating fixtures and shared data for live integration tests.

Tests are skipped by default unless the corresponding API key environment
variable is set. Checks both KAOS_LLM_* and standard env var names.
Mark tests with the appropriate ``requires_*`` marker.

Also provides shared grounding-corpus fixtures used by the WS-1 grounding
end-to-end harness (``test_grounding_e2e.py``) and the calibration script.
"""

from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass

import pytest


def _has_key(*env_vars: str) -> bool:
    """Return True if any of the given env vars is set and non-empty."""
    return any(os.getenv(v) for v in env_vars)


# Skip markers for each provider — check both KAOS_LLM_ and standard names
requires_openai = pytest.mark.skipif(
    not _has_key("KAOS_LLM_OPENAI_API_KEY", "OPENAI_API_KEY"),
    reason="No OpenAI API key",
)
requires_anthropic = pytest.mark.skipif(
    not _has_key("KAOS_LLM_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    reason="No Anthropic API key",
)
requires_google = pytest.mark.skipif(
    not _has_key("KAOS_LLM_GOOGLE_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY"),
    reason="No Google API key",
)


# ---------------------------------------------------------------------------
# Grounding corpus fixture (WS-1 — fundamentals roadmap)
# ---------------------------------------------------------------------------


_CORPUS_DIR = pathlib.Path(__file__).resolve().parent.parent / "fixtures" / "grounding-corpus"
_URI_BY_FILE = {
    "delaware-gcl.txt": "doc:grounding/delaware-gcl",
    "first-amendment.txt": "doc:grounding/first-amendment",
    "rfc-2119.txt": "doc:grounding/rfc-2119",
    "fair-use-107.txt": "doc:grounding/fair-use-107",
    "rule-10b-5.txt": "doc:grounding/rule-10b-5",
    "apollo-11.txt": "doc:grounding/apollo-11",
    "gdpr-art-17.txt": "doc:grounding/gdpr-art-17",
    "voyager-fact-sheet.txt": "doc:grounding/voyager",
    "miranda-holding.txt": "doc:grounding/miranda",
    "nist-password-guidance.txt": "doc:grounding/nist-password",
}


@dataclass(frozen=True, slots=True)
class GroundingQuestion:
    """One labelled Q/A entry from ``grounding-questions.jsonl``."""

    id: str
    answerable: bool
    question: str
    expected_doc_id: str | None = None
    expected_answer_hint: str | None = None
    diagnostic_char_span: tuple[int, int] | None = None
    reason: str | None = None


@pytest.fixture(scope="session")
def grounding_corpus_dir() -> pathlib.Path:
    """Absolute path to the grounding-corpus fixture directory."""
    return _CORPUS_DIR


@pytest.fixture(scope="session")
def grounding_corpus() -> dict[str, str]:
    """Map of ``source_uri -> full document text`` for the grounding fixture.

    Uses stable ``doc:grounding/<slug>`` URIs so tests and the calibration
    script agree on naming regardless of filesystem layout.
    """
    return {uri: (_CORPUS_DIR / fname).read_text() for fname, uri in _URI_BY_FILE.items()}


@pytest.fixture(scope="session")
def grounding_questions() -> list[GroundingQuestion]:
    """All 20 labelled questions (10 answerable + 10 unanswerable)."""
    path = _CORPUS_DIR / "grounding-questions.jsonl"
    questions: list[GroundingQuestion] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        span = raw.get("diagnostic_char_span")
        questions.append(
            GroundingQuestion(
                id=raw["id"],
                answerable=bool(raw["answerable"]),
                question=raw["question"],
                expected_doc_id=raw.get("expected_doc_id"),
                expected_answer_hint=raw.get("expected_answer_hint"),
                diagnostic_char_span=tuple(span) if span is not None else None,
                reason=raw.get("reason"),
            )
        )
    return questions


# ---------------------------------------------------------------------------
# Multi-format corpus fixture (WS-3.7 — fundamentals roadmap)
# ---------------------------------------------------------------------------
#
# Shared across the smoke test, future benchmark script, and the full 3.7
# live harness. The dispatch helper lives here (not in the benchmark script
# alone) so any test module can build a multi-format corpus by importing
# the ``multiformat_index`` session fixture.


_MULTIFORMAT_DIR = (
    pathlib.Path(__file__).resolve().parent.parent / "fixtures" / "multiformat-corpus"
)


def dispatch_parse(path: pathlib.Path):
    """Parse a fixture file by extension, returning a :class:`ContentDocument`.

    Imports each extractor lazily so this conftest works even when only a
    subset of kaos-pdf / kaos-office / kaos-web / kaos-content is installed.
    Missing extractors cause a ``pytest.skip`` at call site — consistent
    with the rest of the harness's "skip when unavailable" pattern.
    """
    ext = path.suffix.lower()
    if ext == ".pdf":
        from kaos_pdf import extract_pdf

        return extract_pdf(path)
    if ext == ".docx":
        from kaos_office import parse_docx

        return parse_docx(path)
    if ext in (".html", ".htm"):
        from kaos_web import html_to_document

        return html_to_document(path.read_text(), url=path.as_uri())
    if ext == ".md":
        from kaos_content.model.attr import SourceRef
        from kaos_content.parsers import parse_markdown

        return parse_markdown(
            path.read_text(),
            source=SourceRef(uri=path.as_uri(), mime_type="text/markdown"),
        )
    if ext == ".txt":
        from kaos_content.model.attr import SourceRef
        from kaos_content.parsers import parse_plain_text

        return parse_plain_text(
            path.read_text(),
            source=SourceRef(uri=path.as_uri(), mime_type="text/plain"),
        )
    msg = (
        f"No dispatch handler for extension {ext!r} (path={path}). "
        "Fix: add a branch to dispatch_parse or drop the file from the fixture. "
        "Alternative: rename the file with a supported extension (.pdf, .docx, .html, .md, .txt)."
    )
    raise ValueError(msg)


@pytest.fixture(scope="session")
def multiformat_dir() -> pathlib.Path:
    """Absolute path to the multiformat-corpus fixture directory."""
    return _MULTIFORMAT_DIR


@pytest.fixture(scope="session")
def multiformat_paths() -> list[pathlib.Path]:
    """All supported fixture files, sorted for determinism.

    ``README.md`` is excluded because it is documentation about the
    fixture, not a corpus document. Any other ``README*`` or
    ``fixtures-provenance.md`` additions are excluded on the same basis.
    """
    if not _MULTIFORMAT_DIR.is_dir():
        return []
    supported = {".pdf", ".docx", ".html", ".htm", ".md", ".txt"}
    excluded_stems = {"README", "fixtures-provenance"}
    return sorted(
        p
        for p in _MULTIFORMAT_DIR.iterdir()
        if p.suffix.lower() in supported and p.stem not in excluded_stems
    )
