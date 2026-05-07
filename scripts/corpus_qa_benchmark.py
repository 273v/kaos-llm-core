"""FUND-6 Unified Corpus QA Benchmark.

Loads all three fixture sets (grounding-corpus, multiformat-corpus,
expanded-corpus) as a single 30-document corpus, runs the RAG
pipeline over every Q/A triple, and reports per-set and overall
accuracy.

Two accuracy metrics:
- **Answerable accuracy**: the answer contains the ``expected_answer_hint``
  (case-insensitive substring). Generous by design — the NLI verifier
  (FUND-1) is the precision gate, not this benchmark.
- **Unanswerable accuracy**: the system correctly refuses (returns
  InsufficientEvidence, or the answer says "not in the document" /
  "cannot be determined" / etc.).

Run::

    uv run python scripts/corpus_qa_benchmark.py [--model anthropic:claude-haiku-4-5]

Artifacts:
- ``docs/benchmarks/corpus-qa-30doc-YYYY-MM-DD.json``
- ``docs/benchmarks/corpus-qa-30doc-YYYY-MM-DD.md``
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

FIXTURE_BASE = Path(__file__).parent.parent / "tests" / "fixtures"

SETS = {
    "grounding": {
        "docs_dir": FIXTURE_BASE / "grounding-corpus",
        "questions": FIXTURE_BASE / "grounding-corpus" / "grounding-questions.jsonl",
        "file_exts": {".txt"},
    },
    "multiformat": {
        "docs_dir": FIXTURE_BASE / "multiformat-corpus",
        "questions": FIXTURE_BASE / "multiformat-corpus" / "multiformat-questions.jsonl",
        "file_exts": {".txt", ".md", ".html", ".pdf", ".docx"},
    },
    "expanded": {
        "docs_dir": FIXTURE_BASE / "expanded-corpus",
        "questions": FIXTURE_BASE / "expanded-corpus" / "expanded-qa-golden.jsonl",
        "file_exts": {".txt"},
    },
}


def _load_questions(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _load_text_docs(dir_path: Path, exts: set[str]) -> dict[str, str]:
    """Load plain-text documents; skip non-text formats for now."""
    docs: dict[str, str] = {}
    for f in sorted(dir_path.iterdir()):
        if f.suffix not in exts:
            continue
        if f.suffix in (".pdf", ".docx", ".html"):
            continue
        docs[f.name] = f.read_text(errors="replace")
    return docs


def _check_answerable(answer: str, hint: str) -> bool:
    """Does the answer contain the expected hint (case-insensitive)?"""
    if not hint:
        return True
    return hint.lower() in answer.lower()


def _check_unanswerable(answer: str) -> bool:
    """Does the answer look like a proper refusal?"""
    lower = answer.lower()
    refusal_signals = (
        "insufficient evidence",
        "not mentioned",
        "not found",
        "not in the",
        "cannot be determined",
        "does not mention",
        "no information",
        "not available",
        "cannot answer",
        "not stated",
    )
    return any(signal in lower for signal in refusal_signals)


async def _run_rag_question(
    question: str,
    corpus_texts: dict[str, str],
    model: str,
) -> str:
    """Run a single RAG question against the text corpus.

    Uses the kaos-llm-core starter.text() for simplicity — builds a
    context block from all docs and asks the LLM to answer with
    citations. This is NOT the full RAG retriever pipeline (which would
    use embeddings to select top-k passages); it's the "stuff all docs
    in context" baseline. Appropriate for a 30-doc corpus that fits in
    a single context window for Haiku.
    """
    from kaos_llm_core.programs.call import Call
    from kaos_llm_core.signatures.fields import InputField, OutputField
    from kaos_llm_core.signatures.signature import Signature

    class _QA(Signature):
        """Answer the question based ONLY on the provided documents.
        If the answer is not in the documents, say "Insufficient evidence."
        Quote specific text when possible."""

        context: str = InputField(description="All source documents")
        question: str = InputField(description="The question to answer")
        answer: str = OutputField(
            description="Answer based on the documents, or 'Insufficient evidence'"
        )

    context_parts: list[str] = []
    for name, text in corpus_texts.items():
        context_parts.append(f"=== DOCUMENT: {name} ===\n{text}\n")
    context = "\n".join(context_parts)

    # Truncate to ~100K chars — Haiku's context is 200K but we want
    # headroom for system + question.
    if len(context) > 100_000:
        context = context[:100_000] + "\n[... truncated ...]"

    call = Call(_QA, model=model)
    invocation = await call.invoke(context=context, question=question)
    return invocation.output.answer


async def main_async(model: str) -> None:
    import os

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("KAOS_LLM_ANTHROPIC_API_KEY")):
        print("ERROR: ANTHROPIC_API_KEY not set. Required for RAG benchmark.")
        return

    all_results: list[dict[str, Any]] = []
    set_summaries: dict[str, dict[str, Any]] = {}

    for set_name, cfg in SETS.items():
        docs_dir = cfg["docs_dir"]
        q_path = cfg["questions"]
        if not docs_dir.exists() or not q_path.exists():
            print(f"(skip) {set_name}: missing fixture dir or questions")
            continue

        docs = _load_text_docs(docs_dir, cfg["file_exts"])
        questions = _load_questions(q_path)
        print(f"\n--- {set_name}: {len(docs)} docs, {len(questions)} questions ---")

        correct = 0
        total = 0
        for q in questions:
            total += 1
            t0 = time.perf_counter()
            try:
                answer = await _run_rag_question(q["question"], docs, model)
            except Exception as exc:
                answer = f"[ERROR: {type(exc).__name__}: {exc}]"
            elapsed = time.perf_counter() - t0

            if q["answerable"]:
                is_correct = _check_answerable(answer, q.get("expected_answer_hint", ""))
            else:
                is_correct = _check_unanswerable(answer)

            if is_correct:
                correct += 1

            status = "PASS" if is_correct else "FAIL"
            print(f"  {q['id']}: {status} ({elapsed:.1f}s)")

            all_results.append(
                {
                    "set": set_name,
                    "id": q["id"],
                    "answerable": q["answerable"],
                    "question": q["question"],
                    "expected_hint": q.get("expected_answer_hint", ""),
                    "answer_preview": answer[:200],
                    "correct": is_correct,
                    "elapsed_s": round(elapsed, 2),
                }
            )

        accuracy = correct / total if total else 0.0
        set_summaries[set_name] = {
            "total": total,
            "correct": correct,
            "accuracy": round(accuracy, 3),
        }
        print(f"  {set_name}: {correct}/{total} = {accuracy:.0%}")

    # Overall
    total_all = sum(s["total"] for s in set_summaries.values())
    correct_all = sum(s["correct"] for s in set_summaries.values())
    overall_accuracy = correct_all / total_all if total_all else 0.0

    today = datetime.now().strftime("%Y-%m-%d")
    out_dir = Path(__file__).parent.parent / "docs" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"corpus-qa-30doc-{today}.json"
    md_path = out_dir / f"corpus-qa-30doc-{today}.md"

    payload = {
        "sprint": "FUND-6",
        "model": model,
        "generated_at": datetime.now().isoformat(),
        "total_questions": total_all,
        "total_correct": correct_all,
        "overall_accuracy": round(overall_accuracy, 3),
        "per_set": set_summaries,
        "per_question": all_results,
    }
    json_path.write_text(json.dumps(payload, indent=2))

    lines = [
        f"# FUND-6 Corpus QA Benchmark — {today}",
        "",
        f"Model: **{model}**",
        f"Overall: **{correct_all}/{total_all} = {overall_accuracy:.0%}**",
        "",
        "| Set | Questions | Correct | Accuracy |",
        "|-----|-----------|---------|----------|",
    ]
    for name, s in set_summaries.items():
        lines.append(f"| {name} | {s['total']} | {s['correct']} | {s['accuracy']:.0%} |")
    lines.append("")

    # Per-question detail
    lines.append("## Per-question results")
    lines.append("")
    lines.append("| Set | ID | Answerable | Correct | Time |")
    lines.append("|-----|----|------------|---------|------|")
    for r in all_results:
        mark = "PASS" if r["correct"] else "**FAIL**"
        lines.append(f"| {r['set']} | {r['id']} | {r['answerable']} | {mark} | {r['elapsed_s']}s |")

    md_path.write_text("\n".join(lines))
    print(f"\nOverall: {correct_all}/{total_all} = {overall_accuracy:.0%}")
    print(f"Artifacts: {json_path}, {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FUND-6 Corpus QA benchmark")
    parser.add_argument(
        "--model",
        default="anthropic:claude-haiku-4-5",
        help="Provider:model string (default: anthropic:claude-haiku-4-5)",
    )
    args = parser.parse_args()

    import asyncio

    asyncio.run(main_async(args.model))


if __name__ == "__main__":
    main()
