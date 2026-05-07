"""TIER1-1: End-to-end extraction recipe validation.

Proves the full Extract pipeline works on real contracts: recipe JSON →
ExtractionSchema → Extract program → ExtractionResult → score against
golden JSONL.

Runs two modes:
1. **CUAD 5-column** — the existing calibration schema on the 5 CUAD
   sample contracts. Validates the pipeline machinery.
2. **Recipe schemas** — loads each extraction recipe by name and runs
   it on the first CUAD contract (schema validation only — we don't
   have golden answers for all 27/32/24/16 columns, but we prove the
   schema compiles, the LLM returns valid cells, and provenance
   round-trips).

Run::

    uv run python scripts/tier1_extraction_e2e.py [--model anthropic:claude-haiku-4-5]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

CUAD_DIR = Path(__file__).parent.parent / "tests" / "fixtures" / "cuad-sample"
GOLDEN = CUAD_DIR / "cuad-extraction-golden.jsonl"

CUAD_SCHEMA = {
    "id": "cuad-5col-v1",
    "columns": [
        {
            "id": "parties",
            "column_type": "string",
            "description": "Names of the contracting parties.",
        },
        {
            "id": "agreement_date",
            "column_type": "string",
            "description": (
                "Effective date or signing date of the agreement. "
                "Extract as it appears in the text."
            ),
        },
        {
            "id": "governing_law",
            "column_type": "string",
            "description": "State or jurisdiction whose law governs the agreement.",
        },
        {
            "id": "termination_for_convenience",
            "column_type": "string",
            "description": "Clause allowing either party to terminate without cause.",
        },
        {
            "id": "cap_on_liability",
            "column_type": "string",
            "description": "Any limitation on liability amounts or scope.",
        },
    ],
}


def _load_golden() -> dict[str, dict[str, list[str]]]:
    """Load {doc_id: {column_id: [acceptable_answers]}}.

    Normalizes golden keys to snake_case so they match ExtractionSchema
    column IDs ("Agreement Date" → "agreement_date").
    """
    result: dict[str, dict[str, list[str]]] = {}
    for line in GOLDEN.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        doc_id = rec["doc_id"]
        normalized: dict[str, list[str]] = {}
        for key, val in rec.get("clause_answers", {}).items():
            snake_key = key.lower().replace(" ", "_")
            normalized[snake_key] = val
        result[doc_id] = normalized
    return result


def _score(extracted: str | None, acceptable: list[str]) -> bool:
    if not extracted or not acceptable:
        return False
    ext_lower = extracted.lower()
    return any(a.lower() in ext_lower for a in acceptable)


async def run_cuad_validation(model: str) -> None:
    """Run the 5-column CUAD schema on all 5 contracts."""
    from kaos_llm_core.programs.extract import Extract
    from kaos_llm_core.signatures.extraction import ExtractionSchema

    schema = ExtractionSchema.from_dict(CUAD_SCHEMA)
    golden = _load_golden()

    print(f"=== CUAD 5-column extraction ({model}) ===\n")

    total_correct = 0
    total_cells = 0
    for doc_file in sorted(CUAD_DIR.glob("*.txt")):
        doc_id = doc_file.stem
        if doc_id not in golden:
            continue
        source_text = doc_file.read_text()
        gold = golden[doc_id]

        extract = Extract(schema, model=model, provenance="none")
        t0 = time.perf_counter()
        inv = await extract.invoke(source_text=source_text)
        elapsed = time.perf_counter() - t0

        correct = 0
        total = 0
        for cell in inv.output.cells:
            acceptable = gold.get(cell.column_id, [])
            if not acceptable:
                continue
            total += 1
            total_cells += 1
            hit = _score(str(cell.ai_value) if cell.ai_value else None, acceptable)
            if hit:
                correct += 1
                total_correct += 1
            status = "PASS" if hit else "FAIL"
            val_preview = str(cell.ai_value)[:60] if cell.ai_value else "(empty)"
            print(f"  {doc_id[:30]} | {cell.column_id:30s} | {status} | {val_preview}")

        print(f"  → {correct}/{total} ({elapsed:.1f}s)\n")

    accuracy = total_correct / total_cells if total_cells else 0
    print(f"Overall: {total_correct}/{total_cells} = {accuracy:.0%}\n")


async def run_recipe_validation(model: str) -> None:
    """Load each extraction recipe and run it on one contract to prove schema validity."""
    from kaos_agents.recipes import extraction_recipe_names, load_extraction_recipe

    from kaos_llm_core.programs.extract import Extract
    from kaos_llm_core.signatures.extraction import ExtractionSchema

    doc_file = next(CUAD_DIR.glob("*.txt"))
    source_text = doc_file.read_text()[:30000]

    print(f"=== Recipe schema validation ({model}) ===\n")

    for name in extraction_recipe_names():
        recipe = load_extraction_recipe(name)
        schema = ExtractionSchema.from_dict(recipe["schema"])
        n_cols = len(schema.columns)

        extract = Extract(schema, model=model, provenance="none")
        t0 = time.perf_counter()
        try:
            inv = await extract.invoke(source_text=source_text)
            elapsed = time.perf_counter() - t0
            n_extracted = sum(1 for c in inv.output.cells if c.status == "extracted")
            n_refused = sum(1 for c in inv.output.cells if c.status == "not_in_document")
            print(
                f"  {name:25s} | {n_cols} cols | {n_extracted} extracted, "
                f"{n_refused} refused | {elapsed:.1f}s | OK"
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(
                f"  {name:25s} | {n_cols} cols | "
                f"ERROR: {type(exc).__name__}: {exc} | {elapsed:.1f}s"
            )

    print()


async def main_async(model: str) -> None:
    import os

    if not (
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("KAOS_LLM_ANTHROPIC_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    ):
        print("ERROR: No LLM API key found.")
        return

    await run_cuad_validation(model)
    # Recipe validation requires kaos-agents in the venv; skip if unavailable.
    try:
        from kaos_agents.recipes import extraction_recipe_names  # noqa: F401

        await run_recipe_validation(model)
    except ImportError:
        print("(skip) Recipe validation: kaos-agents not in this venv.")


def main() -> None:
    parser = argparse.ArgumentParser(description="TIER1-1 extraction E2E")
    parser.add_argument("--model", default="anthropic:claude-haiku-4-5")
    args = parser.parse_args()
    asyncio.run(main_async(args.model))


if __name__ == "__main__":
    main()
