"""Scale E2E tests: RAG pipeline against real Federal Register documents.

Fetches 100 SEC rulemaking documents from the Federal Register API,
downloads the raw text for each, parses any HTML-wrapped responses
via kaos-web html_to_document, chunks each document via SectionChunker,
builds a kaos-ml-core Corpus with paragraph-level AST provenance,
and runs the RAG pipeline with an EmbeddingRetriever built from
the Corpus.

NO TRUNCATION -- the SectionChunker splits each document into 2-8KB
chunks, the retriever returns top-k chunks, and the context budget
handles the rest.

Tests are skipped unless:
- Anthropic API key is available (for LLM calls)
- Internet is available (for Federal Register API calls)

Run with: pytest tests/integration/test_rag_federal_register.py -v
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx
import pytest


def _has_internet() -> bool:
    """Quick check for internet connectivity via Federal Register."""
    try:
        r = httpx.get(
            "https://www.federalregister.gov/api/v1/agencies.json",
            timeout=10,
            follow_redirects=True,
        )
        return r.status_code < 400
    except Exception:
        return False


def _has_anthropic() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("KAOS_LLM_ANTHROPIC_API_KEY"))


requires_fr = pytest.mark.skipif(
    not _has_internet(), reason="No internet / Federal Register unreachable"
)
requires_anthropic = pytest.mark.skipif(not _has_anthropic(), reason="No Anthropic API key")


# --- Corpus building -------------------------------------------------------

_MAX_DOCS = 100
_PER_PAGE = 20
_MAX_PAGES = _MAX_DOCS // _PER_PAGE  # 5 pages of 20


def _parse_fr_text(raw: str, doc_number: str) -> Any:
    """Parse a Federal Register raw text response into a ContentDocument.

    FR's ``raw_text_url`` endpoint wraps plain-text rulemaking bodies
    inside a single ``<pre>`` block. ``html_to_document`` with the
    default ``pre_content_mode="code"`` would map that to a single
    ``CodeBlock`` (correct for real code, wrong for FR's prose), and
    ``Corpus.from_documents(level="paragraph")`` would then find no
    paragraphs. Switching to ``pre_content_mode="prose"`` tells the
    parser this document's ``<pre>`` holds blank-line-separated text
    and emits one ``Paragraph`` per chunk — the shape the corpus
    expects. Works whether the response is HTML-wrapped or raw text
    (the HTML parser silently degrades when there's no ``<html>``
    tag), so a single call handles every current FR shape.
    """
    from kaos_web import html_to_document

    return html_to_document(raw, extract_content=False, pre_content_mode="prose")


@pytest.fixture(scope="module")
def fr_corpus() -> Any:
    """Fetch 100 SEC rulemaking docs, chunk each, and build a Corpus.

    Paginates through the Federal Register API to collect documents,
    fetches the raw text for each, parses via kaos-web html_to_document,
    chunks each document via SectionChunker, and builds a single
    kaos-ml-core Corpus from all chunks with paragraph-level provenance.

    Cached at module scope so all tests share the same corpus.
    """
    from kaos_content.chunking.section_chunker import SectionChunker
    from kaos_ml_core import Corpus
    from kaos_source.connectors.federal_register import (
        FRDocument,
        search_documents,
    )

    async def _fetch_all() -> Corpus:
        # Step 1: Collect document metadata across pages
        all_fr_docs: list[FRDocument] = []
        for page_num in range(1, _MAX_PAGES + 1):
            try:
                result = await search_documents(
                    agencies="securities-and-exchange-commission",
                    doc_type=["RULE", "PRORULE"],
                    per_page=_PER_PAGE,
                    page=page_num,
                    fields=[
                        "document_number",
                        "title",
                        "type",
                        "publication_date",
                        "agency_names",
                        "raw_text_url",
                        "abstract",
                        "citation",
                    ],
                )
                all_fr_docs.extend(result.documents)
            except Exception:
                continue  # skip failed pages

            if len(all_fr_docs) >= _MAX_DOCS:
                break

        all_fr_docs = all_fr_docs[:_MAX_DOCS]

        # Step 2: Fetch raw text, parse, and chunk each document
        chunker = SectionChunker(max_chars=8000, split_depth=2)
        all_chunks = []
        doc_uris: list[str] = []

        async with httpx.AsyncClient(
            timeout=120,
            follow_redirects=True,
            headers={"User-Agent": "kaos-source/0.1 (https://273ventures.com)"},
        ) as client:
            fetched = 0
            for fr_doc in all_fr_docs:
                if not fr_doc.raw_text_url:
                    continue
                try:
                    resp = await client.get(fr_doc.raw_text_url)
                    resp.raise_for_status()
                    raw_text = resp.text

                    if len(raw_text) < 200:
                        continue  # skip stub documents

                    content_doc = _parse_fr_text(raw_text, fr_doc.document_number)
                    if not content_doc.body:
                        continue  # skip documents with no parseable content
                    chunks = chunker.chunk(content_doc)

                    uri = f"fr:{fr_doc.document_number}"
                    for chunk in chunks:
                        if chunk.body:  # only add chunks with content
                            all_chunks.append(chunk)
                            doc_uris.append(uri)
                    fetched += 1

                except Exception:
                    continue  # skip docs that fail to fetch

                if fetched >= _MAX_DOCS:
                    break

        if not all_chunks:
            pytest.skip(
                "No documents with parseable content from FR API. "
                "This can happen when the API returns incomplete document records "
                "or when raw_text_url content changes format."
            )
        return Corpus.from_documents(all_chunks, level="paragraph", doc_uris=doc_uris)

    return asyncio.get_event_loop().run_until_complete(_fetch_all())


@pytest.fixture(scope="module")
def fr_retriever(fr_corpus: Any) -> Any:
    """Build an EmbeddingRetriever from the Corpus.

    Falls back to None if kaos-nlp-transformers is not installed,
    in which case tests use the lazy BM25 path via Corpus.
    """
    try:
        from kaos_nlp_transformers.retrieval import EmbeddingRetriever
    except ImportError:
        return None

    return EmbeddingRetriever.from_corpus(fr_corpus)


# --- Tests -----------------------------------------------------------------


@requires_fr
class TestFRCorpusFetch:
    """Verify we can build a real Corpus from the Federal Register."""

    def test_corpus_size(self, fr_corpus: Any) -> None:
        """Should have many paragraph units from 50+ documents."""
        assert len(fr_corpus) >= 500, f"Only got {len(fr_corpus)} units; expected >= 500"

    def test_corpus_total_chars(self, fr_corpus: Any) -> None:
        """Total text should be 500KB+."""
        total_chars = sum(len(u.text) for u in fr_corpus)
        assert total_chars >= 500_000, f"Only {total_chars} chars; expected >= 500KB"

        print(f"Corpus: {len(fr_corpus)} units, {total_chars:,} chars total")

    def test_units_have_provenance(self, fr_corpus: Any) -> None:
        """Every unit should carry a doc_uri and block_ref."""
        for unit in fr_corpus:
            assert unit.doc_uri, f"Unit row={unit.row} has no doc_uri"
            assert unit.block_ref, f"Unit row={unit.row} has no block_ref"


@requires_fr
@requires_anthropic
class TestRAGPipelineFederalRegister:
    """Run the full RAG pipeline against the real FR corpus."""

    @pytest.mark.integration
    async def test_specific_rule_query(self, fr_corpus: Any, fr_retriever: Any) -> None:
        """Query about SEC disclosure and reporting requirements."""
        from kaos_llm_core.programs.rag import RAG
        from kaos_llm_core.signatures import Answer, MatchStrategy

        rag = RAG(
            model="anthropic:claude-haiku-4-5",
            retriever=fr_retriever,
            top_k=10,
            match_strategies=(
                MatchStrategy.STRICT,
                MatchStrategy.SUBSTRING,
                MatchStrategy.CASE_INSENSITIVE,
                MatchStrategy.NORMALIZED_TOKEN,
            ),
        )
        result = await rag.query(
            question=(
                "What are the SEC's disclosure and reporting requirements for securities offerings?"
            ),
            documents=fr_corpus,
        )
        ga = result.grounded_answer

        # This may return Answer or InsufficientEvidence depending on
        # what documents are in the corpus.  If Answer, verify structure.
        if isinstance(ga, Answer):
            assert len(ga.claims) >= 1
            assert len(ga.sources) >= 1
            answer_preview = str(ga.value)[:200]
            print(
                f"Answer: {answer_preview}...\n  ({len(ga.claims)} claims, sources: {ga.sources})"
            )
        else:
            # InsufficientEvidence is acceptable if the corpus
            # doesn't happen to contain the right rules
            print(f"Refused (acceptable): {ga.reason[:200]}")

    @pytest.mark.integration
    async def test_cross_document_query(self, fr_corpus: Any, fr_retriever: Any) -> None:
        """Cross-document query about SEC reporting requirements."""
        from kaos_llm_core.programs.rag import RAG
        from kaos_llm_core.signatures import Answer, MatchStrategy

        rag = RAG(
            model="anthropic:claude-haiku-4-5",
            retriever=fr_retriever,
            top_k=15,
            match_strategies=(
                MatchStrategy.STRICT,
                MatchStrategy.SUBSTRING,
                MatchStrategy.CASE_INSENSITIVE,
                MatchStrategy.NORMALIZED_TOKEN,
            ),
        )
        result = await rag.query(
            question=(
                "What are the SEC's recent regulations on securities "
                "disclosure and market transparency?"
            ),
            documents=fr_corpus,
        )
        ga = result.grounded_answer

        if isinstance(ga, Answer):
            # Cross-document queries should cite at least one source
            assert len(ga.sources) >= 1, f"Expected at least one citation, got: {ga.sources}"
            print(f"Cross-document answer: {len(ga.claims)} claims, sources: {ga.sources}")
        else:
            print(f"Refused (acceptable): {ga.reason[:200]}")

    @pytest.mark.integration
    async def test_refusal_on_non_sec_topic(self, fr_corpus: Any, fr_retriever: Any) -> None:
        """A question about OSHA (not in SEC corpus) should be refused."""
        from kaos_llm_core.programs.rag import RAG
        from kaos_llm_core.signatures import InsufficientEvidence

        rag = RAG(
            model="anthropic:claude-haiku-4-5",
            retriever=fr_retriever,
            top_k=10,
        )
        result = await rag.query(
            question=("What are OSHA workplace safety requirements for construction sites?"),
            documents=fr_corpus,
        )
        ga = result.grounded_answer

        assert isinstance(ga, InsufficientEvidence), (
            f"Expected InsufficientEvidence (corpus has no OSHA rules), "
            f"got {type(ga).__name__}: {ga}"
        )
        print(f"Correctly refused: {ga.reason[:200]}")

    @pytest.mark.integration
    async def test_verification_integrity(self, fr_corpus: Any, fr_retriever: Any) -> None:
        """Run 5 queries and verify every citation against the actual
        Federal Register document text."""
        from kaos_llm_core.programs.rag import RAG
        from kaos_llm_core.signatures import Answer, MatchStrategy

        questions = [
            "What changes has the SEC made to proxy voting disclosure rules?",
            "What are the SEC's requirements for registration of securities offerings?",
            "How does the SEC regulate short selling and market manipulation?",
            "What are the SEC's rules about broker-dealer registration and compliance?",
            "What reporting requirements does the SEC impose on public companies?",
        ]

        rag = RAG(
            model="anthropic:claude-haiku-4-5",
            retriever=fr_retriever,
            top_k=10,
            match_strategies=(
                MatchStrategy.STRICT,
                MatchStrategy.SUBSTRING,
                MatchStrategy.CASE_INSENSITIVE,
                MatchStrategy.NORMALIZED_TOKEN,
            ),
        )

        total_claims = 0
        total_verified = 0
        total_errors = 0

        for q in questions:
            result = await rag.query(question=q, documents=fr_corpus)
            ga = result.grounded_answer

            if isinstance(ga, Answer):
                total_claims += len(ga.claims)
                errors = result.verification_errors
                total_errors += len(errors)
                total_verified += len(ga.claims) - len(errors)

                print(
                    f"Q: {q[:60]}... -> "
                    f"{len(ga.claims)} claims, "
                    f"{len(errors)} verification errors, "
                    f"sources: {ga.sources}"
                )
                if errors:
                    for e in errors[:3]:
                        print(
                            f"  ERROR: claim {e.claim_index}, "
                            f"span {e.span_index}: {e.reason} "
                            f"(quote: {e.span.quote[:80]}...)"
                        )
            else:
                print(f"Q: {q[:60]}... -> REFUSED: {ga.reason[:100]}")

        print(
            f"\n=== FEDERAL REGISTER VERIFICATION SUMMARY ===\n"
            f"Total claims: {total_claims}\n"
            f"Verified: {total_verified}\n"
            f"Errors: {total_errors}\n"
            f"Verification rate: "
            f"{total_verified / max(total_claims, 1) * 100:.1f}% "
            f"({total_verified}/{total_claims})"
        )

        # At least 90% of claims should verify
        if total_claims > 0:
            rate = total_verified / total_claims
            assert rate >= 0.9, (
                f"Verification rate {rate:.1%} below 90% threshold. "
                f"{total_errors} of {total_claims} claims failed verification."
            )
