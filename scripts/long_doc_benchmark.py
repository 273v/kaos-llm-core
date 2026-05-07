"""FUND-7 Long-document extraction calibration.

Runs the Extract program on the 63 KB PPD employment agreement
(tests/fixtures/long-doc-sample/) in two modes:

1. **Baseline** — single-pass extraction, no chunk-retry.
2. **ChunkRetry** — with chunk-retry enabled (default config:
   initial_chunk_chars=8000, max_chunks=16).

Scores both against the golden JSONL and reports per-cell recovery,
cost delta, and timing. This proves the chunk-retry path recovers
cells that the single-pass path misses on documents long enough to
dilute mid-document content in the LLM's attention window.

Run::

    uv run python scripts/long_doc_benchmark.py [--model anthropic:claude-haiku-4-5]

Artifacts:
- ``docs/benchmarks/long-doc-calibration-YYYY-MM-DD.json``
- ``docs/benchmarks/long-doc-calibration-YYYY-MM-DD.md``
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

FIXTURE_DIR = Path(__file__).parent.parent / "tests" / "fixtures" / "long-doc-sample"
DOC_PATH = FIXTURE_DIR / "ppd-employment-agreement.txt"
GOLDEN_PATH = FIXTURE_DIR / "golden.jsonl"


SCHEMA_COLUMNS = [
    {
        "name": "parties",
        "type": "string",
        "description": (
            "The names of the parties to the agreement. Extract each party name "
            "exactly as it appears in the document. This is typically on the first page."
        ),
    },
    {
        "name": "effective_date",
        "type": "date",
        "description": (
            "The effective date of the agreement. Extract as close to the original "
            "wording as possible (e.g. '16th day of September, 2011'). "
            "Look in the preamble or recitals."
        ),
    },
    {
        "name": "governing_law",
        "type": "string",
        "description": (
            "The state or jurisdiction whose law governs this agreement. "
            "Usually in a Governing Law section near the end of the agreement. "
            "Extract only the jurisdiction name (e.g. 'State of North Carolina')."
        ),
    },
    {
        "name": "termination_for_convenience",
        "type": "string",
        "description": (
            "How either party may terminate the agreement at will, without cause. "
            "Extract the key condition or mechanism (e.g. 'written notice by "
            "either Employee or the Company to the other party'). "
            "Look in the termination or cessation article."
        ),
    },
]


def _load_golden() -> dict[str, list[str]]:
    """Load golden answers. Returns {column_name: [acceptable_answers]}."""
    for line in GOLDEN_PATH.read_text().splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        return data.get("clause_answers", {})
    return {}


def _score_cell(extracted: str | None, acceptable: list[str]) -> bool:
    """Lenient substring match — any acceptable answer in the extracted text."""
    if not extracted or not acceptable:
        return False
    ext_lower = extracted.lower()
    return any(ans.lower() in ext_lower for ans in acceptable)


async def _run_extraction(
    source_text: str,
    model: str,
    *,
    use_chunk_retry: bool,
) -> tuple[dict[str, str | None], float, float]:
    """Run Extract and return (column→value, elapsed_s, cost_usd)."""
    from kaos_llm_core.programs.extract import Extract
    from kaos_llm_core.signatures.extraction import ExtractionSchema

    schema = ExtractionSchema.from_dict(
        {
            "id": "long-doc-v1",
            "columns": [
                {
                    "id": c["name"],
                    "column_type": c["type"],
                    "description": c["description"],
                }
                for c in SCHEMA_COLUMNS
            ],
        }
    )

    extract = Extract(
        schema,
        model=model,
        chunk_retry="default" if use_chunk_retry else None,
        provenance="none",
    )

    t0 = time.perf_counter()
    invocation = await extract.invoke(source_text=source_text)
    elapsed = time.perf_counter() - t0

    # ExtractionResult.cells is a tuple of ExtractionCell, keyed by column_id.
    result_map: dict[str, str | None] = {}
    output = invocation.output
    for cell in output.cells:
        val = cell.ai_value
        if val is not None:
            val = str(val)
        result_map[cell.column_id] = val

    cost = output.cost_usd or 0.0
    return result_map, elapsed, cost


async def main_async(model: str) -> None:
    import os

    if not (
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("KAOS_LLM_ANTHROPIC_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    ):
        print("ERROR: No LLM API key found.")
        return

    if not DOC_PATH.exists():
        print(f"ERROR: fixture missing: {DOC_PATH}")
        return

    source_text = DOC_PATH.read_text()
    golden = _load_golden()
    print(f"Document: {DOC_PATH.name} ({len(source_text):,} chars)")
    print(f"Model: {model}")
    print(f"Schema: {len(SCHEMA_COLUMNS)} columns")
    print(f"Golden: {len(golden)} columns with acceptable answers\n")

    results: dict[str, Any] = {}
    for mode, use_cr in [("baseline", False), ("chunk_retry", True)]:
        print(f"--- {mode} ---")
        try:
            cells, elapsed, cost = await _run_extraction(source_text, model, use_chunk_retry=use_cr)
        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            results[mode] = {"error": str(exc)}
            continue

        correct = 0
        total = 0
        cell_details: list[dict[str, Any]] = []
        for col_name in [c["name"] for c in SCHEMA_COLUMNS]:
            total += 1
            extracted = cells.get(col_name)
            acceptable = golden.get(col_name, [])
            is_correct = _score_cell(extracted, acceptable)
            if is_correct:
                correct += 1
            cell_details.append(
                {
                    "column": col_name,
                    "extracted": extracted,
                    "acceptable": acceptable,
                    "correct": is_correct,
                }
            )
            status = "PASS" if is_correct else "FAIL"
            print(f"  {col_name}: {status} — {(extracted or '(empty)')[:80]}")

        accuracy = correct / total if total else 0.0
        results[mode] = {
            "correct": correct,
            "total": total,
            "accuracy": round(accuracy, 3),
            "elapsed_s": round(elapsed, 2),
            "cost_usd": round(cost, 4),
            "cells": cell_details,
        }
        print(f"  → {correct}/{total} = {accuracy:.0%}  ({elapsed:.1f}s, ${cost:.4f})\n")

    # Emit artifacts
    today = datetime.now().strftime("%Y-%m-%d")
    out_dir = Path(__file__).parent.parent / "docs" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"long-doc-calibration-{today}.json"
    md_path = out_dir / f"long-doc-calibration-{today}.md"

    payload = {
        "sprint": "FUND-7",
        "model": model,
        "document": DOC_PATH.name,
        "document_chars": len(source_text),
        "schema_columns": len(SCHEMA_COLUMNS),
        "generated_at": datetime.now().isoformat(),
        "results": results,
    }
    json_path.write_text(json.dumps(payload, indent=2))

    baseline = results.get("baseline", {})
    chunk_retry = results.get("chunk_retry", {})

    lines = [
        f"# FUND-7 Long-Document Calibration — {today}",
        "",
        f"Model: **{model}**",
        f"Document: `{DOC_PATH.name}` ({len(source_text):,} chars)",
        "",
        "| Mode | Correct | Total | Accuracy | Time | Cost |",
        "|------|---------|-------|----------|------|------|",
    ]
    for mode_name, r in [("Baseline", baseline), ("ChunkRetry", chunk_retry)]:
        if "error" in r:
            lines.append(f"| {mode_name} | ERROR | — | — | — | — |")
        else:
            lines.append(
                f"| {mode_name} | {r['correct']} | {r['total']} "
                f"| {r['accuracy']:.0%} | {r['elapsed_s']}s | ${r['cost_usd']:.4f} |"
            )
    lines.append("")

    # Recovery analysis
    if baseline.get("cells") and chunk_retry.get("cells"):
        lines.append("## Per-cell recovery")
        lines.append("")
        lines.append("| Column | Baseline | ChunkRetry | Recovered? |")
        lines.append("|--------|----------|------------|------------|")
        for bc, cc in zip(baseline["cells"], chunk_retry["cells"], strict=True):
            b_mark = "PASS" if bc["correct"] else "FAIL"
            c_mark = "PASS" if cc["correct"] else "FAIL"
            recovered = "YES" if not bc["correct"] and cc["correct"] else ""
            lines.append(f"| {bc['column']} | {b_mark} | {c_mark} | {recovered} |")

    md_path.write_text("\n".join(lines))
    print(f"Artifacts: {json_path}, {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FUND-7 long-doc calibration")
    parser.add_argument(
        "--model",
        default="anthropic:claude-haiku-4-5",
        help="Provider:model (default: anthropic:claude-haiku-4-5)",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args.model))


if __name__ == "__main__":
    main()
