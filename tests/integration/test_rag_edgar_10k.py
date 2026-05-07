"""Scale E2E tests: RAG pipeline against a real EDGAR 10-K filing.

Fetches Apple Inc's most recent 10-K (CIK 0000320193) primary document
HTML from EDGAR, converts it to a ContentDocument via kaos-source's
filing_html_to_document (which handles iXBRL stripping), chunks it via
SectionChunker, builds a kaos-ml-core Corpus with paragraph-level AST
provenance, and runs the RAG pipeline with an EmbeddingRetriever built
from the Corpus.

This is a ~220KB single-document test.  The SectionChunker splits the
filing into 2-8KB chunks at heading boundaries, producing passages
that are the right size for retrieval and citation.

Tests are skipped unless:
- Anthropic API key is available (for LLM calls)
- Internet is available (for EDGAR API calls)
- EDGAR User-Agent is configured

Run with: pytest tests/integration/test_rag_edgar_10k.py -v
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx
import pytest


def _has_internet() -> bool:
    """Quick check for internet connectivity via SEC."""
    try:
        r = httpx.get(
            "https://www.sec.gov/",
            timeout=10,
            follow_redirects=True,
            headers={"User-Agent": _edgar_ua()},
        )
        return r.status_code < 400
    except Exception:
        return False


def _has_anthropic() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("KAOS_LLM_ANTHROPIC_API_KEY"))


def _edgar_ua() -> str:
    return os.getenv(
        "KAOS_SOURCE_EDGAR_USER_AGENT",
        "273 Ventures research@273ventures.com",
    )


requires_edgar = pytest.mark.skipif(not _has_internet(), reason="No internet / SEC unreachable")
requires_anthropic = pytest.mark.skipif(not _has_anthropic(), reason="No Anthropic API key")


# --- Corpus building -------------------------------------------------------

# Apple 10-K filing URL (FY2024, filed September 2025)
_AAPL_10K_URL = (
    "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm"
)


@pytest.fixture(scope="module")
def edgar_corpus() -> Any:
    """Fetch Apple's 10-K HTML, convert and chunk via kaos-source + SectionChunker.

    Uses filing_html_to_document (handles iXBRL stripping) to parse
    the filing into a ContentDocument AST, then SectionChunker to split
    it into 2-8KB chunks at heading boundaries, and finally builds a
    kaos-ml-core Corpus with paragraph-level provenance.

    Cached at module scope so all tests share the same corpus.
    """
    from kaos_content.chunking.section_chunker import SectionChunker
    from kaos_ml_core import Corpus
    from kaos_source.connectors.edgar import filing_html_to_document

    async def _fetch() -> Corpus:
        async with httpx.AsyncClient(
            timeout=120,
            follow_redirects=True,
            headers={"User-Agent": _edgar_ua()},
        ) as client:
            resp = await client.get(_AAPL_10K_URL)
            resp.raise_for_status()
            html = resp.text

        doc = filing_html_to_document(html, url=_AAPL_10K_URL)
        chunks = SectionChunker(max_chars=8000, split_depth=2).chunk(doc)

        # Apple's 10-K iXBRL rarely produces ``<h1..h6>``, so indexing at
        # paragraph level surfaces thousands of small fragments (single-line
        # "Item 1A. Risk Factors" etc.) — the LLM then refuses because the
        # retrieved passages look like "only page headers." ``level="document"``
        # indexes each chunk as a single retrieval unit carrying the full
        # serialized chunk text, which is the granularity SectionChunker
        # already produced.
        doc_uris = [f"edgar:aapl-10k#chunk{idx}" for idx in range(len(chunks))]
        return Corpus.from_documents(chunks, level="document", doc_uris=doc_uris)

    return asyncio.get_event_loop().run_until_complete(_fetch())


@pytest.fixture(scope="module")
def edgar_retriever(edgar_corpus: Any) -> Any:
    """Build an EmbeddingRetriever from the Corpus.

    Falls back to None if kaos-nlp-transformers is not installed,
    in which case tests use the lazy BM25 path via Corpus.
    """
    try:
        from kaos_nlp_transformers.retrieval import EmbeddingRetriever
    except ImportError:
        return None

    return EmbeddingRetriever.from_corpus(edgar_corpus)


# --- Tests -----------------------------------------------------------------


@requires_edgar
class TestEDGARCorpusFetch:
    """Verify we can build a real Corpus from EDGAR 10-K."""

    def test_corpus_size(self, edgar_corpus: Any) -> None:
        """Corpus should hold one retrieval unit per ~8KB chunk of the 10-K.

        The 10-K is ~1.5MB of HTML; SectionChunker at max_chars=8000 yields
        ~20-40 chunks. We deliberately index at chunk-level (one Paragraph
        per chunk containing the serialized text) rather than at raw
        paragraph level — see the fixture docstring. Guard against an
        unexpected collapse by checking a reasonable floor.
        """
        assert len(edgar_corpus) >= 10, (
            f"Only {len(edgar_corpus)} units; expected >= 10 chunk-level units for full 10-K"
        )

    def test_corpus_total_chars(self, edgar_corpus: Any) -> None:
        """Total corpus should be 100KB+."""
        total_chars = sum(len(u.text) for u in edgar_corpus)
        assert total_chars >= 100_000, f"Only {total_chars} chars; expected >= 100KB"
        print(f"Corpus: {len(edgar_corpus)} units, {total_chars:,} chars")

    def test_units_have_provenance(self, edgar_corpus: Any) -> None:
        """Every unit should carry a doc_uri and block_ref."""
        for unit in edgar_corpus:
            assert unit.doc_uri, f"Unit row={unit.row} has no doc_uri"
            assert unit.block_ref, f"Unit row={unit.row} has no block_ref"


@requires_edgar
@requires_anthropic
class TestRAGPipelineEDGAR10K:
    """Run the full RAG pipeline against Apple's real 10-K."""

    def _make_rag(self, retriever=None):
        from kaos_llm_core.programs.rag import RAG
        from kaos_llm_core.signatures import MatchStrategy

        return RAG(
            model="anthropic:claude-haiku-4-5",
            retriever=retriever,
            top_k=10,
            match_strategies=(
                MatchStrategy.STRICT,
                MatchStrategy.SUBSTRING,
                MatchStrategy.CASE_INSENSITIVE,
                MatchStrategy.NORMALIZED_TOKEN,
            ),
        )

    @pytest.mark.integration
    async def test_risk_factor_query(self, edgar_corpus: Any, edgar_retriever: Any) -> None:
        """Query about risk factors should cite the filing."""
        from kaos_llm_core.signatures import Answer

        rag = self._make_rag(edgar_retriever)
        result = await rag.query(
            question="What are the main risk factors Apple discloses?",
            documents=edgar_corpus,
        )
        ga = result.grounded_answer

        assert isinstance(ga, Answer), f"Expected Answer, got {type(ga).__name__}: {ga}"

        # Answer should mention risk-related concepts
        v = str(ga.value).lower()
        assert any(kw in v for kw in ("risk", "competition", "supply", "regulatory")), (
            f"Expected risk-related terms, got: {ga.value!r}"
        )

        # Should cite the 10-K filing
        cited_sources = ga.sources
        assert len(cited_sources) >= 1, f"Expected at least one citation, got: {cited_sources}"

        print(f"Answer: {ga.value[:200]}...\n  ({len(ga.claims)} claims, sources: {cited_sources})")

    @pytest.mark.integration
    async def test_financial_query(self, edgar_corpus: Any, edgar_retriever: Any) -> None:
        """Query about financials should cite the filing."""
        from kaos_llm_core.signatures import Answer

        rag = self._make_rag(edgar_retriever)
        result = await rag.query(
            question=(
                "What does Apple report about its revenue and financial performance in this 10-K?"
            ),
            documents=edgar_corpus,
        )
        ga = result.grounded_answer

        assert isinstance(ga, Answer), f"Expected Answer, got {type(ga).__name__}: {ga}"

        # Should mention revenue, earnings, or financial terms
        v = str(ga.value).lower()
        assert any(kw in v for kw in ("revenue", "net income", "earnings", "sales", "financial")), (
            f"Expected financial terms, got: {ga.value!r}"
        )

        # Should cite the filing
        cited = ga.sources
        assert len(cited) >= 1, f"Expected at least one citation, got: {cited}"

        print(f"Financial answer: {len(ga.claims)} claims, sources: {cited}")

    @pytest.mark.integration
    async def test_cross_item_query(self, edgar_corpus: Any, edgar_retriever: Any) -> None:
        """Query requiring information from multiple parts of the filing."""
        from kaos_llm_core.signatures import Answer

        rag = self._make_rag(edgar_retriever)
        result = await rag.query(
            question=(
                "How do Apple's supply chain risks relate to their properties and facilities?"
            ),
            documents=edgar_corpus,
        )
        ga = result.grounded_answer

        assert isinstance(ga, Answer), f"Expected Answer, got {type(ga).__name__}: {ga}"

        # Should cite at least one source
        cited = ga.sources
        assert len(cited) >= 1, f"Expected at least one citation, got: {cited}"

        print(f"Cross-item answer: {len(ga.claims)} claims, sources: {cited}")

    @pytest.mark.integration
    async def test_refusal_on_absent_data(self, edgar_corpus: Any, edgar_retriever: Any) -> None:
        """A question about future data not in the 10-K should be refused."""
        from kaos_llm_core.signatures import InsufficientEvidence

        rag = self._make_rag(edgar_retriever)
        result = await rag.query(
            question="What was Apple's Q3 2026 revenue?",
            documents=edgar_corpus,
        )
        ga = result.grounded_answer

        assert isinstance(ga, InsufficientEvidence), (
            f"Expected InsufficientEvidence (future data not in 10-K), "
            f"got {type(ga).__name__}: {ga}"
        )
        print(f"Correctly refused: {ga.reason[:200]}")

    @pytest.mark.integration
    async def test_verification_integrity(self, edgar_corpus: Any, edgar_retriever: Any) -> None:
        """Run 5 diverse queries and verify every citation."""
        from kaos_llm_core.signatures import Answer

        questions = [
            "What products and services does Apple offer?",
            "What are the main risk factors Apple faces?",
            "What does Apple say about competition in its markets?",
            "What are Apple's key properties and facilities?",
            "What does Apple's management discuss about operating results?",
        ]

        rag = self._make_rag(edgar_retriever)

        total_claims = 0
        total_verified = 0
        total_errors = 0

        for q in questions:
            result = await rag.query(question=q, documents=edgar_corpus)
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
            f"\n=== EDGAR 10-K VERIFICATION SUMMARY ===\n"
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
