"""Privilege classification end-to-end validation.

Runs the privilege-classification ExtractionSchema on two test emails:
one clearly privileged (attorney-client communication with Davis Polk),
one clearly non-privileged (internal budget planning email).

Validates that the schema compiles, the LLM returns valid cells, and
the classification is correct on these obvious cases.

Run::

    uv run python scripts/privilege_classification_e2e.py
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent.parent / "tests" / "fixtures" / "privilege-sample"
RECIPE_PATH = (
    Path(__file__).parent.parent.parent
    / "kaos-agents"
    / "kaos_agents"
    / "recipes"
    / "extraction"
    / "privilege-classification.json"
)


async def main() -> None:
    import os

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("KAOS_LLM_ANTHROPIC_API_KEY")):
        print("ERROR: ANTHROPIC_API_KEY not set.")
        return

    from kaos_llm_core.programs.extract import Extract
    from kaos_llm_core.signatures.extraction import ExtractionSchema

    recipe = json.loads(RECIPE_PATH.read_text())
    schema = ExtractionSchema.from_dict(recipe["schema"])
    model = "anthropic:claude-haiku-4-5"

    print(f"=== Privilege Classification E2E ({model}) ===")
    print(f"Schema: {schema.id}, {len(schema.columns)} columns\n")

    for email_file, expected_basis, label in [
        (FIXTURE_DIR / "privileged-email.txt", "attorney_client", "PRIVILEGED"),
        (FIXTURE_DIR / "non-privileged-email.txt", "none", "NOT PRIVILEGED"),
    ]:
        if not email_file.exists():
            print(f"  (skip) {email_file.name}: missing")
            continue

        source_text = email_file.read_text()
        extract = Extract(schema, model=model, provenance="none")

        t0 = time.perf_counter()
        inv = await extract.invoke(source_text=source_text)
        elapsed = time.perf_counter() - t0

        cells = {c.column_id: c for c in inv.output.cells}
        basis = cells.get("privilege_basis")
        basis_val = basis.ai_value if basis else "MISSING"
        correct = (basis_val == expected_basis) or (
            expected_basis == "attorney_client"
            and basis_val in ("attorney_client", "work_product_opinion", "work_product_ordinary")
        )
        status = "PASS" if correct else "FAIL"

        print(f"  {label} ({email_file.name})")
        print(f"    privilege_basis:    {basis_val} (expected: {expected_basis}) [{status}]")

        for col_id in [
            "attorney_identified",
            "client_identified",
            "legal_advice_present",
            "confidentiality_maintained",
            "waiver_risk",
            "privilege_log_description",
        ]:
            cell = cells.get(col_id)
            val = cell.ai_value if cell else "MISSING"
            if isinstance(val, str) and len(val) > 80:
                val = val[:77] + "..."
            print(f"    {col_id:30s}: {val}")

        print(f"    ({elapsed:.1f}s)\n")


if __name__ == "__main__":
    asyncio.run(main())
