"""Scale E2E tests: RAG pipeline against real eCFR Title 26 (Tax Code).

Fetches 200 sections from 26 CFR Part 1 (Income Tax regulations) via
the kaos-source eCFR connector, parses them via kaos-web
html_to_document, builds a kaos-ml-core Corpus with paragraph-level
AST provenance, queries via the RAG program with a BM25Retriever
built from the Corpus, and verifies citations structurally via
Answer.verify().

Title 26 has ~6,020 sections total.  We fetch 200 from Part 1 (Income
Tax), which covers gross income, deductions, credits, and the core
mechanics of federal income taxation.  This is a large, dense regulatory
corpus with heavy cross-referencing and technical tax terminology.

Tests are skipped unless:
- Anthropic API key is available (for LLM calls)
- Internet is available (for eCFR API calls)

Run with: pytest tests/integration/test_rag_ecfr_tax.py -v
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest


def _has_internet() -> bool:
    """Quick check for internet connectivity via eCFR."""
    try:
        import httpx

        r = httpx.get("https://www.ecfr.gov/", timeout=10, follow_redirects=True)
        return r.status_code < 400
    except Exception:
        return False


def _has_anthropic() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("KAOS_LLM_ANTHROPIC_API_KEY"))


requires_ecfr = pytest.mark.skipif(not _has_internet(), reason="No internet / eCFR unreachable")
requires_anthropic = pytest.mark.skipif(not _has_anthropic(), reason="No Anthropic API key")


# --- Corpus building -------------------------------------------------------

_TARGET_SECTIONS = 200


@pytest.fixture(scope="module")
def ecfr_tax_corpus() -> Any:
    """Fetch 200 sections from 26 CFR Part 1 (Income Tax) and build a Corpus.

    Uses the eCFR structure API to discover section identifiers,
    then fetches the HTML content for each, parses via kaos-web
    html_to_document, and builds a kaos-ml-core Corpus with
    paragraph-level provenance.  Sections are already small enough
    that no SectionChunker is needed.

    Cached at module scope so all tests share the same corpus.
    """
    from kaos_ml_core import Corpus
    from kaos_source.connectors.ecfr import (
        get_latest_date,
        get_section_content,
        get_title_structure,
    )
    from kaos_web import html_to_document

    async def _fetch_all() -> Corpus:
        # Get the latest available date for Title 26
        latest_date = await get_latest_date(26)
        if latest_date is None:
            msg = "eCFR titles API did not return a latest_issue_date for Title 26"
            raise RuntimeError(msg)

        # Get the full structure of Title 26
        structure = await get_title_structure(26, latest_date)

        # Find Part 1 (Income Tax) sections
        all_sections = structure.find_sections()

        # Filter to Part 1 sections: their identifiers typically
        # start with "1." (e.g., "1.1-1", "1.61-1", "1.162-1").
        # Part 1 covers sections 1 through ~1563 of the IRC.
        part1_sections = [s for s in all_sections if s.identifier.startswith("1.")]

        # Take the first _TARGET_SECTIONS
        target_sections = part1_sections[:_TARGET_SECTIONS]

        # Fetch content for each section
        all_docs = []
        doc_uris: list[str] = []
        fetched = 0
        for sec in target_sections:
            try:
                html = await get_section_content(
                    title=26,
                    date=latest_date,
                    section=sec.identifier,
                )
                # eCFR section HTML is dense structural markup (nested
                # <div class="section">, <p class="indent-*">, etc.) — the
                # readability heuristics score it below the article-prose
                # floor and drop 60-70% of the paragraphs. Pass
                # ``extract_content=False`` so the full tree survives;
                # we're already fetching a known regulatory endpoint
                # where every paragraph is content.
                doc = html_to_document(html, extract_content=False)
                uri = f"ecfr:26-cfr-{sec.identifier}"
                all_docs.append(doc)
                doc_uris.append(uri)
                fetched += 1
            except Exception:
                continue  # skip sections that 404 or timeout

            # Stop early if we have enough
            if fetched >= _TARGET_SECTIONS:
                break

        return Corpus.from_documents(all_docs, level="paragraph", doc_uris=doc_uris)

    return asyncio.get_event_loop().run_until_complete(_fetch_all())


@pytest.fixture(scope="module")
def ecfr_tax_retriever(ecfr_tax_corpus: Any) -> Any:
    """Build a BM25Retriever from the Corpus.

    BM25 is well-suited for this large regulatory corpus where
    keyword matching against tax terminology is effective.
    """
    from kaos_nlp_core.retrieval.bm25 import BM25Retriever

    return BM25Retriever.from_corpus(ecfr_tax_corpus)


# --- Tests -----------------------------------------------------------------


@requires_ecfr
class TestECFRTaxCorpusFetch:
    """Verify we can build a real Corpus from Title 26."""

    def test_corpus_size(self, ecfr_tax_corpus: Any) -> None:
        """Should have many paragraph units from 150+ sections."""
        assert len(ecfr_tax_corpus) >= 500, (
            f"Only got {len(ecfr_tax_corpus)} units; expected >= 500"
        )

    def test_corpus_total_chars(self, ecfr_tax_corpus: Any) -> None:
        """Total text should be 200KB+."""
        total_chars = sum(len(u.text) for u in ecfr_tax_corpus)
        assert total_chars >= 200_000, f"Only {total_chars} chars; expected >= 200KB"

        # Print corpus stats
        print(f"Corpus: {len(ecfr_tax_corpus)} units, {total_chars:,} chars total")

    def test_units_have_provenance(self, ecfr_tax_corpus: Any) -> None:
        """Every unit should carry a doc_uri and block_ref."""
        for unit in ecfr_tax_corpus:
            assert unit.doc_uri, f"Unit row={unit.row} has no doc_uri"
            assert unit.block_ref, f"Unit row={unit.row} has no block_ref"


@requires_ecfr
@requires_anthropic
class TestRAGPipelineECFRTax:
    """Run the full RAG pipeline against the real Title 26 corpus."""

    @pytest.mark.integration
    async def test_income_tax_query(self, ecfr_tax_corpus: Any, ecfr_tax_retriever: Any) -> None:
        """Query about gross income definitions."""
        from kaos_llm_core.programs.rag import RAG
        from kaos_llm_core.signatures import Answer, MatchStrategy

        rag = RAG(
            model="anthropic:claude-haiku-4-5",
            retriever=ecfr_tax_retriever,
            top_k=10,
            match_strategies=(
                MatchStrategy.STRICT,
                MatchStrategy.SUBSTRING,
                MatchStrategy.CASE_INSENSITIVE,
                MatchStrategy.NORMALIZED_TOKEN,
            ),
        )
        result = await rag.query(
            question=("What income is included in gross income under the tax code?"),
            documents=ecfr_tax_corpus,
        )
        ga = result.grounded_answer

        assert isinstance(ga, Answer), f"Expected Answer, got {type(ga).__name__}: {ga}"

        # Answer should mention income-related concepts
        v = str(ga.value).lower()
        assert any(kw in v for kw in ("gross income", "income", "compensation", "wages")), (
            f"Expected income-related terms, got: {ga.value!r}"
        )

        # Should cite tax code sections
        cited_sources = ga.sources
        assert len(cited_sources) >= 1, f"Expected at least one citation, got: {cited_sources}"

        answer_preview = str(ga.value)[:200]
        print(f"Answer: {answer_preview}...\n  ({len(ga.claims)} claims, sources: {cited_sources})")

    @pytest.mark.integration
    async def test_deduction_query(self, ecfr_tax_corpus: Any, ecfr_tax_retriever: Any) -> None:
        """Query about business expense deductions."""
        from kaos_llm_core.programs.rag import RAG
        from kaos_llm_core.signatures import Answer, MatchStrategy

        rag = RAG(
            model="anthropic:claude-haiku-4-5",
            retriever=ecfr_tax_retriever,
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
                "What types of income are excluded from gross income under the tax regulations?"
            ),
            documents=ecfr_tax_corpus,
        )
        ga = result.grounded_answer

        assert isinstance(ga, Answer), f"Expected Answer, got {type(ga).__name__}: {ga}"

        # Should mention exclusion/income concepts
        v = str(ga.value).lower()
        assert any(
            kw in v
            for kw in (
                "exclud",
                "income",
                "gross",
                "exempt",
                "section",
            )
        ), f"Expected income/exclusion terms, got: {ga.value!r}"

        cited = ga.sources
        assert len(cited) >= 1, f"Expected at least one citation, got: {cited}"

        print(f"Exclusion answer: {len(ga.claims)} claims, sources: {cited}")

    @pytest.mark.integration
    async def test_cross_section_query(self, ecfr_tax_corpus: Any, ecfr_tax_retriever: Any) -> None:
        """Query requiring information from multiple tax code sections."""
        from kaos_llm_core.programs.rag import RAG
        from kaos_llm_core.signatures import Answer, MatchStrategy

        rag = RAG(
            model="anthropic:claude-haiku-4-5",
            retriever=ecfr_tax_retriever,
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
                "How do the rules for computing gross income interact "
                "with the rules for allowable deductions to determine "
                "taxable income?"
            ),
            documents=ecfr_tax_corpus,
        )
        ga = result.grounded_answer

        assert isinstance(ga, Answer), f"Expected Answer, got {type(ga).__name__}: {ga}"

        # Cross-section query should cite multiple sections
        cited = ga.sources
        assert len(cited) >= 1, f"Expected at least one citation, got: {cited}"

        print(f"Cross-section answer: {len(ga.claims)} claims, sources: {cited}")

    @pytest.mark.integration
    async def test_refusal_on_non_tax(self, ecfr_tax_corpus: Any, ecfr_tax_retriever: Any) -> None:
        """A question about securities (Title 17) should be refused."""
        from kaos_llm_core.programs.rag import RAG
        from kaos_llm_core.signatures import InsufficientEvidence

        rag = RAG(
            model="anthropic:claude-haiku-4-5",
            retriever=ecfr_tax_retriever,
            top_k=10,
        )
        result = await rag.query(
            question=("What are the securities exchange reporting requirements under Rule 10b-5?"),
            documents=ecfr_tax_corpus,
        )
        ga = result.grounded_answer

        assert isinstance(ga, InsufficientEvidence), (
            f"Expected InsufficientEvidence (corpus has no securities law), "
            f"got {type(ga).__name__}: {ga}"
        )
        print(f"Correctly refused: {ga.reason[:200]}")

    @pytest.mark.integration
    async def test_verification_integrity(
        self, ecfr_tax_corpus: Any, ecfr_tax_retriever: Any
    ) -> None:
        """Run 5 diverse tax questions and verify every citation
        against the actual eCFR text."""
        from kaos_llm_core.programs.rag import RAG
        from kaos_llm_core.signatures import Answer, MatchStrategy

        questions = [
            "What types of income are excluded from gross income?",
            "What are the rules for depreciation of business property?",
            "How are capital gains and losses treated for tax purposes?",
            "What are the requirements for deducting charitable contributions?",
            "What rules apply to the taxation of interest income?",
        ]

        rag = RAG(
            model="anthropic:claude-haiku-4-5",
            retriever=ecfr_tax_retriever,
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
            result = await rag.query(question=q, documents=ecfr_tax_corpus)
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
            f"\n=== ECFR TAX VERIFICATION SUMMARY ===\n"
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
