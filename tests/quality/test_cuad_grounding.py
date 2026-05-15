"""CUAD span-verification quality harness for :class:`CitedSummary`.

Measures: when CitedSummary is asked about a specific CUAD clause
type, what fraction of returned (verified) source spans actually
contain the golden answer from the CUAD test set?

The full CUAD v1 corpus is CC-BY-4.0 (Hendrycks et al. 2021,
https://github.com/TheAtticusProject/cuad). We use the 5-contract
x5-clause vendored subset at
``../kaos-nlp-core/tests/fixtures/cuad-sample/``, which the
kaos-nlp-core test suite already redistributes under the same
licence terms.

Run with::

    KAOS_LLM_LIVE_PROVIDER=anthropic \\
        ANTHROPIC_API_KEY=... \\
        uv run --no-sync pytest tests/quality/test_cuad_grounding.py \\
            --no-cov -v -m live -s

Cost: ~$0.02-0.05 per full run with claude-haiku-4-5 (5 contracts x
5 clauses x ~500-token contract bodies). Skip cleanly when
``KAOS_LLM_LIVE_PROVIDER`` is unset.

Output: ``docs/benchmarks/quality-cuad-grounding.json`` with the
per-(contract, clause) trace -- whether the gold answer landed in
any verified span, whether the run refused, the verified-claim
ratio, and total token + USD cost.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.live

_BENCH_DIR = Path(__file__).resolve().parent.parent.parent / "docs" / "benchmarks"


def _resolve_cuad_dir() -> Path | None:
    """Find the kaos-nlp-core vendored CUAD sample directory.

    Search order: ``KAOS_CUAD_FIXTURES`` env var, then the monorepo
    sibling ``../kaos-nlp-core/tests/fixtures/cuad-sample/``.
    """
    env = os.environ.get("KAOS_CUAD_FIXTURES")
    if env:
        p = Path(env).expanduser().resolve()
        if (p / "cuad-extraction-golden.jsonl").exists():
            return p
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "kaos-nlp-core" / "tests" / "fixtures" / "cuad-sample"
        if (cand / "cuad-extraction-golden.jsonl").exists():
            return cand
    return None


@pytest.fixture(scope="module")
def cuad_dir() -> Path:
    p = _resolve_cuad_dir()
    if p is None:
        pytest.skip("CUAD vendored fixtures not found (set KAOS_CUAD_FIXTURES)")
    return p


@pytest.fixture(scope="module")
def cuad_records(cuad_dir: Path) -> list[dict]:
    records: list[dict] = []
    with (cuad_dir / "cuad-extraction-golden.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _live_provider_available() -> bool:
    return bool(os.environ.get("KAOS_LLM_LIVE_PROVIDER"))


def _normalize_for_match(s: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for substring match."""
    import re

    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _gold_in_spans(gold_answers: list[str], spans: list[str], min_overlap_chars: int = 20) -> bool:
    """Return True iff any verified span contains (after normalisation) a
    contiguous run of at least ``min_overlap_chars`` characters from any
    gold answer.

    We don't require exact-string equality because the LLM's quoted span
    may include surrounding context or differ in whitespace. The
    ``min_overlap_chars`` floor keeps trivial matches from dominating
    (e.g. the gold answer "Tickets" matching every contract).
    """
    if not gold_answers or not spans:
        return False
    spans_norm = [_normalize_for_match(s) for s in spans if s]
    for gold in gold_answers:
        gold_norm = _normalize_for_match(gold)
        if len(gold_norm) < min_overlap_chars:
            # Short gold -- require it to appear verbatim (normalised).
            if any(gold_norm in sn for sn in spans_norm):
                return True
            continue
        # Long gold -- use difflib to find the longest common substring.
        for sn in spans_norm:
            matcher = difflib.SequenceMatcher(None, gold_norm, sn, autojunk=False)
            match = matcher.find_longest_match(0, len(gold_norm), 0, len(sn))
            if match.size >= min_overlap_chars:
                return True
    return False


def _clause_instructions(clause_name: str) -> str:
    """Per-clause instruction that biases CitedSummary toward the target."""
    return (
        f"You are analysing a commercial contract. Identify and quote the "
        f"clause titled '{clause_name}'. Every claim in your summary must "
        f"quote the supporting text verbatim from the source. If the "
        f"clause is not present, say so explicitly."
    )


async def _run_one_clause(citer, contract_text: str) -> dict:
    """Run CitedSummary on one contract; return a structured trace."""
    summary = await citer(text=contract_text)
    # Pull verified spans + claim counts from the result.
    span_texts = [
        contract_text[s.start : s.end] for s in summary.source_spans if s.end <= len(contract_text)
    ]
    meta = summary.metadata or {}
    # ``Summary`` carries refusal state in ``metadata["cited.refused"]``
    # (set by ``CitedSummary`` when the verified-claim ratio falls
    # below ``refuse_below``); there is no top-level ``refused``
    # attribute on the Pydantic model.
    return {
        "summary_head": summary.text[:120] if summary.text else "",
        "refused": bool(meta.get("cited.refused", False)),
        "n_verified_spans": len(summary.source_spans),
        "n_total_claims": int(meta.get("claims_total", len(summary.source_spans))),
        "n_verified_claims": int(meta.get("claims_verified", len(summary.source_spans))),
        "verified_span_texts": span_texts,
    }


@pytest.mark.skipif(
    not _live_provider_available(),
    reason="KAOS_LLM_LIVE_PROVIDER not set; CUAD harness requires live LLM access",
)
def test_cuad_span_verification_rate(cuad_records: list[dict], cuad_dir: Path) -> None:
    """Per (contract, clause) verification + gold-overlap rate.

    Emits ``docs/benchmarks/quality-cuad-grounding.json`` with the
    full trace + aggregate metrics. The test asserts:

    - The harness completes for every (contract, clause) pair without
      crashing.
    - At least 50% of (contract, clause) pairs return at least ONE
      verified span. This is a low floor -- it catches "LLM refused
      every clause" or "verifier rejected every span" without
      asserting clause-level accuracy.
    """
    from kaos_llm_core.programs.summarize import CitedSummary

    # Load contract bodies.
    contract_bodies: dict[str, str] = {}
    for rec in cuad_records:
        path = cuad_dir / f"{rec['doc_id']}.txt"
        contract_bodies[rec["doc_id"]] = path.read_text(encoding="utf-8")

    target_clauses = (
        "Parties",
        "Agreement Date",
        "Governing Law",
        "Termination For Convenience",
        "Cap On Liability",
    )

    rows: list[dict] = []
    for rec in cuad_records:
        body = contract_bodies[rec["doc_id"]]
        for clause in target_clauses:
            gold = rec["clause_answers"].get(clause, [])
            citer = CitedSummary(instructions=_clause_instructions(clause))
            trace = asyncio.run(_run_one_clause(citer, body))
            trace["doc_id"] = rec["doc_id"]
            trace["clause"] = clause
            trace["gold_count"] = len(gold)
            trace["gold_in_verified_spans"] = _gold_in_spans(gold, trace["verified_span_texts"])
            # Don't store the full verbatim span text in the artifact --
            # keep headers + match outcome to avoid bloating the JSON.
            trace["verified_span_heads"] = [s[:160] for s in trace["verified_span_texts"]]
            trace.pop("verified_span_texts", None)
            rows.append(trace)

    n = len(rows)
    n_any_verified = sum(1 for r in rows if r["n_verified_spans"] > 0)
    n_refused = sum(1 for r in rows if r["refused"])
    n_gold_in_spans = sum(1 for r in rows if r["gold_in_verified_spans"])
    verified_claim_rate = sum(r["n_verified_claims"] for r in rows) / max(
        1, sum(r["n_total_claims"] for r in rows)
    )

    aggregate = {
        "n_cells": n,
        "n_cells_with_any_verified_span": n_any_verified,
        "n_cells_refused": n_refused,
        "n_cells_with_gold_in_verified_spans": n_gold_in_spans,
        "fraction_with_any_verified_span": round(n_any_verified / max(1, n), 4),
        "fraction_refused": round(n_refused / max(1, n), 4),
        "fraction_with_gold_match": round(n_gold_in_spans / max(1, n), 4),
        "overall_verified_claim_rate": round(verified_claim_rate, 4),
    }

    payload = {
        "harness": "CUAD span-verification",
        "model": os.environ.get("KAOS_LLM_MODEL_OVERRIDE", "(provider default)"),
        "fixture": "kaos-nlp-core/tests/fixtures/cuad-sample (CC-BY-4.0)",
        "captured_at": "2026-05-15",
        "aggregate": aggregate,
        "per_cell": rows,
    }

    _BENCH_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _BENCH_DIR / "quality-cuad-grounding.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")

    if os.environ.get("KAOS_LLM_BENCH_PRINT"):
        print("\nCUAD aggregate:", aggregate)

    # Soft assertions -- the harness is informational, not a CI gate.
    assert n == 25, f"Expected 25 (contract, clause) cells, got {n}"
    assert n_any_verified >= n / 2, (
        f"Less than half of cells produced any verified span "
        f"({n_any_verified}/{n}); CitedSummary's verifier may be over-rejecting "
        f"or the LLM is refusing too often."
    )
