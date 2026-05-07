"""TIER1-4: Summarization recipe validation on legal documents.

Runs the kaos-llm-core Call program on 5 legal document types and
scores each summary for factual grounding (does the summary contain
key facts from the source?) and length appropriateness.

Document types:
1. Court opinion (Miranda holding excerpt)
2. Contract clause (PPD employment agreement excerpt)
3. Regulation (17 CFR 240.10b-5)
4. Government report (GAO cybersecurity excerpt)
5. SEC filing (Microsoft 10-K excerpt)

Run::

    uv run python scripts/tier1_summarization_e2e.py [--model anthropic:claude-haiku-4-5]
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path
from typing import Any

FIXTURE_BASE = Path(__file__).parent.parent / "tests" / "fixtures"

DOCUMENTS: list[dict[str, Any]] = [
    {
        "name": "court_opinion",
        "path": FIXTURE_BASE / "grounding-corpus" / "miranda-holding.txt",
        "must_contain": ["custodial interrogation", "self-incrimination"],
        "max_summary_chars": 500,
    },
    {
        "name": "contract_clause",
        "path": FIXTURE_BASE / "long-doc-sample" / "ppd-employment-agreement.txt",
        "must_contain": ["pharmaceutical", "employment"],
        "max_summary_chars": 800,
    },
    {
        "name": "regulation",
        "path": FIXTURE_BASE / "grounding-corpus" / "rule-10b-5.txt",
        "must_contain": ["unlawful", "securities"],
        "max_summary_chars": 400,
    },
    {
        "name": "gao_report",
        "path": FIXTURE_BASE / "expanded-corpus" / "gao_report_excerpt.txt",
        "must_contain": ["cybersecurity", "agencies"],
        "max_summary_chars": 500,
    },
    {
        "name": "sec_10k",
        "path": FIXTURE_BASE / "expanded-corpus" / "sec_10k_msft_excerpt.txt",
        "must_contain": ["microsoft", "revenue"],
        "max_summary_chars": 500,
    },
]


async def summarize_doc(text: str, model: str) -> str:
    from kaos_llm_core.programs.call import Call
    from kaos_llm_core.signatures.fields import InputField, OutputField
    from kaos_llm_core.signatures.signature import Signature

    class _Summarize(Signature):
        """Summarize this legal document concisely. Focus on key facts,
        parties, dates, obligations, and holdings. Do not add information
        not present in the source."""

        source_text: str = InputField(description="The document to summarize")
        summary: str = OutputField(description="A concise summary of the document (2-4 paragraphs)")

    call = Call(_Summarize, model=model)
    inv = await call.invoke(source_text=text[:30000])
    return inv.output.summary


async def main_async(model: str) -> None:
    import os

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("KAOS_LLM_ANTHROPIC_API_KEY")):
        print("ERROR: ANTHROPIC_API_KEY not set.")
        return

    print(f"=== TIER1-4 Summarization Validation ({model}) ===\n")

    total_pass = 0
    total_checks = 0

    for doc in DOCUMENTS:
        path = doc["path"]
        if not path.exists():
            print(f"  (skip) {doc['name']}: fixture missing")
            continue

        source = path.read_text()
        t0 = time.perf_counter()
        try:
            summary = await summarize_doc(source, model)
        except Exception as exc:
            print(f"  {doc['name']}: ERROR {type(exc).__name__}: {exc}")
            continue
        elapsed = time.perf_counter() - t0

        summary_lower = summary.lower()
        fact_hits = 0
        for fact in doc["must_contain"]:
            total_checks += 1
            if fact.lower() in summary_lower:
                fact_hits += 1
                total_pass += 1

        fact_score = f"{fact_hits}/{len(doc['must_contain'])}"
        length_ok = len(summary) <= doc["max_summary_chars"]
        length_tag = "OK" if length_ok else f"LONG ({len(summary)} > {doc['max_summary_chars']})"

        print(f"  {doc['name']:20s} | facts: {fact_score} | length: {length_tag} | {elapsed:.1f}s")
        print(f"    preview: {summary[:120]}...")

    accuracy = total_pass / total_checks if total_checks else 0
    print(f"\nFact grounding: {total_pass}/{total_checks} = {accuracy:.0%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TIER1-4 summarization E2E")
    parser.add_argument("--model", default="anthropic:claude-haiku-4-5")
    args = parser.parse_args()
    asyncio.run(main_async(args.model))


if __name__ == "__main__":
    main()
