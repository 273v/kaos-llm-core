"""Smoke test for WS-3.7 multi-format benchmark (phase 1).

Proves that the full chain works end-to-end on a realistic mixed-format
corpus:

    file paths  →  dispatch_parse (PDF/DOCX/HTML/MD/TXT → ContentDocument)
                →  Corpus.from_documents(docs, level="paragraph")
                →  RAG.query(question, documents=corpus)
                →  Answer with verified citations

Not a calibration run — only validates the composition. The full WS-3.7
acceptance gate with 45-Q golden set + per-format precision breakdown
+ benchmark artifact lands after the shared calibration helpers are
extracted out of ``scripts/grounding_calibration.py``.

Skips cleanly if any extractor package is absent (unit-level parse_plain_text
+ dispatch dry-run tests stay green either way).
"""

from __future__ import annotations

import importlib.util
import time

import pytest

from kaos_llm_core.programs.rag import RAG, RAGResult
from kaos_llm_core.signatures import Answer, MatchStrategy

from .conftest import dispatch_parse, requires_anthropic

_REQUIRED_EXTRACTORS = ("kaos_pdf", "kaos_office", "kaos_web", "kaos_content")


def _all_extractors_available() -> bool:
    return all(importlib.util.find_spec(pkg) is not None for pkg in _REQUIRED_EXTRACTORS)


_HAS_ML_CORE = importlib.util.find_spec("kaos_ml_core") is not None


_LENIENT = (
    MatchStrategy.STRICT,
    MatchStrategy.SUBSTRING,
    MatchStrategy.CASE_INSENSITIVE,
    MatchStrategy.NORMALIZED_TOKEN,
)


@pytest.mark.integration
@pytest.mark.skipif(not _all_extractors_available(), reason="Not all extractors installed")
@pytest.mark.skipif(not _HAS_ML_CORE, reason="kaos-ml-core not installed")
class TestMultiformatCorpusBuilds:
    def test_every_fixture_parses_to_content_document(self, multiformat_paths) -> None:
        """Each of the 10 fixtures parses through its format-specific extractor
        and yields a non-empty ContentDocument with a populated source URI."""
        assert len(multiformat_paths) > 0, (
            "multiformat-corpus fixture is empty; WS-3.7 phase 1 expected at least 10 files"
        )
        for path in multiformat_paths:
            doc = dispatch_parse(path)
            assert doc is not None, f"{path.name}: dispatch_parse returned None"
            assert len(doc.body) > 0, f"{path.name}: parsed document has empty body"
            assert doc.metadata.source is not None, (
                f"{path.name}: metadata.source is None — ContentDocumentCorpus "
                "will fall back to doc:anon-<i> URIs and golden-set lookups break"
            )
            assert doc.metadata.source.uri.endswith(path.name), (
                f"{path.name}: metadata.source.uri does not end with the "
                f"filename (got {doc.metadata.source.uri!r})"
            )

    def test_corpus_builds_across_all_formats(self, multiformat_paths) -> None:
        """Corpus.from_documents ingests every extractor's output without
        format-specific glue. Records build time so phase 2 can set a
        meaningful ceiling assertion."""
        from kaos_ml_core.corpus import Corpus

        start = time.monotonic()
        docs = [dispatch_parse(p) for p in multiformat_paths]
        corpus = Corpus.from_documents(docs, level="paragraph")
        elapsed_ms = (time.monotonic() - start) * 1000.0

        assert corpus.size > 0, "Multi-format corpus must contain at least one passage"
        # Every fixture should have contributed at least one passage.
        doc_uris_seen = {unit.doc_uri for unit in corpus}
        assert len(doc_uris_seen) == len(multiformat_paths), (
            f"Expected {len(multiformat_paths)} distinct doc_uris in the corpus; "
            f"got {len(doc_uris_seen)}. Some extractor failed to set metadata.source "
            f"or produced an empty document."
        )
        # Loose sanity — a 10-doc mix of real PDFs + HTML + DOCX ought to
        # produce at least 20 paragraphs. Exact number depends on the
        # fixture contents; this is a floor, not a ceiling.
        assert corpus.size >= 20, (
            f"Corpus has only {corpus.size} passages — expected ≥ 20 across 10 mixed-format docs"
        )
        # Loose build-time sanity; phase 2 tightens this.
        assert elapsed_ms < 30_000, (
            f"Corpus build took {elapsed_ms:.0f}ms for {len(multiformat_paths)} docs; "
            "expected well under 30s (WS-3.7 ceiling is 60s for 20-50 docs)"
        )


@requires_anthropic
@pytest.mark.integration
@pytest.mark.skipif(not _all_extractors_available(), reason="Not all extractors installed")
@pytest.mark.skipif(not _HAS_ML_CORE, reason="kaos-ml-core not installed")
class TestMultiformatRAGSmoke:
    """One live RAG call over the full 10-doc multi-format corpus. If this
    passes, WS-3.7's plumbing is complete; only the golden Q/A set + per-
    format precision/recall breakdown remain (phase 2)."""

    async def test_can_answer_grounded_question_across_formats(self, multiformat_paths) -> None:
        from kaos_ml_core.corpus import Corpus

        docs = [dispatch_parse(p) for p in multiformat_paths]
        corpus = Corpus.from_documents(docs, level="paragraph")

        rag = RAG(
            model="anthropic:claude-haiku-4-5",
            match_strategies=_LENIENT,
            top_k=8,
        )

        # Question intentionally targets the plain-text fixture — proves
        # the .txt extractor path works end-to-end, not just PDF/HTML.
        result = await rag.query(
            question="What does RFC 2119 say MUST NOT means?",
            documents=corpus,
        )

        assert isinstance(result, RAGResult)
        assert isinstance(result.grounded_answer, Answer), (
            "Expected a grounded Answer on an answerable question sourced "
            "from the RFC 2119 plain-text fixture; got "
            f"{type(result.grounded_answer).__name__}"
        )
        value_lower = str(result.grounded_answer.value).lower()
        assert any(kw in value_lower for kw in ("prohibition", "shall not", "absolute")), (
            f"Answer should describe MUST NOT as an absolute prohibition: "
            f"{result.grounded_answer.value!r}"
        )
