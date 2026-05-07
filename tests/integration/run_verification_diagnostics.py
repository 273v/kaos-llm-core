"""Verification failure analysis — runs RAG queries with diagnostic mode.

NOT a pytest test file. Run directly:
    python tests/integration/run_verification_diagnostics.py

Fetches a small eCFR corpus, runs 5 diverse queries, and reports
detailed per-span diagnostic information for every verification failure.
Categorizes failures into: paraphrase, wrong_source, unicode_gap,
out_of_range, or other.

Requires: ANTHROPIC_API_KEY + internet access.
"""

from __future__ import annotations

import asyncio
import sys
from collections import Counter
from pathlib import Path

# Ensure we can import from the package
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


async def main() -> None:
    from kaos_source.connectors.ecfr import get_section_content
    from kaos_web import html_to_document

    from kaos_llm_core.programs.rag import RAG
    from kaos_llm_core.signatures import Answer, InsufficientEvidence, MatchStrategy

    # ── 1. Fetch eCFR corpus (20 sections for analysis) ───────────

    SECTIONS = [
        "240.10b-5",  # Anti-fraud
        "240.10b5-1",  # Insider trading defense
        "240.10b5-2",  # Duties of trust
        "240.13a-1",  # Annual reports
        "240.13a-14",  # Certification
        "240.13d-3",  # Beneficial owner
        "240.14a-8",  # Shareholder proposals
        "240.15c3-1",  # Net capital
        "240.16b-3",  # Employee benefit plans
        "240.17a-4",  # Records preservation
        "240.0-1",  # Definitions
        "240.3b-18",  # Rule 10b5-1 terms
        "240.12b-2",  # Definitions
        "240.14a-9",  # False statements
        "240.21F-2",  # Whistleblower
    ]

    print("Fetching eCFR corpus...")
    documents = []
    doc_uris = []
    for sec_id in SECTIONS:
        try:
            html = await get_section_content(title=17, date="2024-01-01", section=sec_id)
            doc = html_to_document(html)
            if doc.body:
                documents.append(doc)
                doc_uris.append(f"ecfr:17-cfr-{sec_id}")
        except Exception:
            continue

    print(f"  Parsed {len(documents)} documents")

    # ── 2. Build corpus + retriever ────────────────────────────────

    from kaos_ml_core import Corpus

    corpus = Corpus.from_documents(documents, level="paragraph", doc_uris=doc_uris)
    print(f"  Corpus: {len(corpus)} units, {sum(len(u.text) for u in corpus)} chars")

    try:
        retriever = corpus.retriever("embedding")
    except ImportError:
        retriever = corpus.retriever("bm25")

    # ── 3. Run RAG with diagnostic verification ────────────────────

    strategies = (
        MatchStrategy.STRICT,
        MatchStrategy.SUBSTRING,
        MatchStrategy.CASE_INSENSITIVE,
        MatchStrategy.NORMALIZED_TOKEN,
    )

    rag = RAG(
        model="anthropic:claude-haiku-4-5",
        retriever=retriever,
        top_k=10,
        match_strategies=strategies,
    )

    questions = [
        "What does Rule 10b-5 prohibit?",
        "What is the definition of 'beneficial owner' under Rule 13d-3?",
        "What records must broker-dealers preserve under Rule 17a-4?",
        "What exemptions exist for employee benefit plan transactions under Rule 16b-3?",
        "What are the requirements for shareholder proposals under Rule 14a-8?",
    ]

    total_claims = 0
    total_verified = 0
    failure_categories: Counter[str] = Counter()
    all_failures: list[dict] = []

    for q in questions:
        print(f"\n{'=' * 70}")
        print(f"Q: {q}")
        result = await rag.query(question=q, documents=corpus)
        ga = result.grounded_answer

        if isinstance(ga, InsufficientEvidence):
            print(f"  REFUSED: {ga.reason[:120]}")
            continue

        if not isinstance(ga, Answer):
            print(f"  UNEXPECTED: {type(ga).__name__}")
            continue

        print(f"  Answer: {len(ga.claims)} claims, sources: {ga.sources}")

        # Re-verify with diagnostics enabled
        for ci, claim in enumerate(ga.claims):
            errors = claim.verify(
                result.source_map,
                strategies=strategies,
                threshold=0.9,
                diagnostics=True,
            )
            total_claims += 1
            if not errors:
                total_verified += 1
                continue

            for err in errors:
                category = _categorize_failure(err)
                failure_categories[category] += 1
                failure_info = {
                    "question": q[:60],
                    "claim": claim.statement[:80],
                    "span_quote": err.span.quote[:100],
                    "source_uri": err.span.source_uri,
                    "reason": err.reason,
                    "category": category,
                    "strategy_details": [
                        f"{sr.strategy}: {sr.detail[:120]}" for sr in err.strategy_results
                    ],
                }
                all_failures.append(failure_info)

                print(f"\n  FAILURE (claim {ci}, span {err.span_index}): {category}")
                print(f"    Reason: {err.reason}")
                print(f"    Quote: {err.span.quote[:120]}")
                print(f"    Source: {err.span.source_uri}")
                for sr in err.strategy_results:
                    print(f"    {sr.strategy}: {sr.detail[:120]}")

    # ── 4. Summary ─────────────────────────────────────────────────

    print(f"\n{'=' * 70}")
    print("VERIFICATION FAILURE ANALYSIS SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total claims: {total_claims}")
    print(f"Verified: {total_verified}")
    print(f"Failed: {total_claims - total_verified}")
    if total_claims > 0:
        rate = total_verified / total_claims
        print(f"Verification rate: {rate:.1%}")
    print()
    print("Failure categories:")
    for cat, count in failure_categories.most_common():
        print(f"  {cat}: {count}")
    print()

    if all_failures:
        print("Detailed failures:")
        for i, f in enumerate(all_failures):
            print(f"\n  [{i + 1}] Category: {f['category']}")
            print(f"      Question: {f['question']}")
            print(f"      Claim: {f['claim']}")
            print(f"      Quote: {f['span_quote']}")
            print(f"      Source: {f['source_uri']}")
            for d in f["strategy_details"]:
                print(f"      {d}")


def _categorize_failure(err) -> str:
    """Categorize a verification failure based on diagnostics."""
    if err.reason == "source_missing":
        return "wrong_source"
    if err.reason == "out_of_range":
        return "out_of_range"
    # Check if it's a paraphrase (quote not in source at all)
    if err.reason == "quote_mismatch":
        # If SUBSTRING also failed, the quote text is not in the source
        for sr in err.strategy_results:
            if sr.strategy == "substring" and not sr.matched:
                return "paraphrase"
        return "unicode_gap"
    if err.reason == "similarity_below_threshold":
        return "paraphrase"
    return "other"


if __name__ == "__main__":
    asyncio.run(main())
