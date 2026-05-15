"""BillSum ROUGE harness for :class:`AbstractiveSummary` and the
long-document reducer Programs (:class:`HierarchicalSummary`,
:class:`MapReduceSummary`).

Measures: do our summaries actually approximate human references on a
real summarization benchmark? Reports ROUGE-1, ROUGE-2, and ROUGE-L
(longest common subsequence) F-scores -- the standard summarization
quality metrics from the literature.

The BillSum dataset (Kornraphop/billsum or FiscalNote/billsum on
HuggingFace) ships US Congressional bills paired with human-written
reference summaries, Apache-2.0 licensed. Bills run 5-50k characters
(too long for a single-call AbstractiveSummary on most chat models),
which makes BillSum the right benchmark for our long-document
reducers as well as the single-call path.

Run with::

    KAOS_LLM_LIVE_PROVIDER=anthropic \\
        ANTHROPIC_API_KEY=... \\
        uv run --no-sync pytest tests/quality/test_billsum_rouge.py \\
            --no-cov -v -m live -s

Cost (claude-haiku-4-5, 20 bills, ~10k chars each):
  - AbstractiveSummary single-call: ~$0.10
  - HierarchicalSummary (depth 2): ~$0.30
  - MapReduceSummary (chunk=2k): ~$0.40
Total: ~$0.80 per full run. Skip cleanly when
``KAOS_LLM_LIVE_PROVIDER`` is unset OR ``datasets`` /
``rouge-score`` aren't installed.

Output: ``docs/benchmarks/quality-billsum-rouge.json`` with the
per-bill ROUGE scores + aggregate mean / median.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.live

_BENCH_DIR = Path(__file__).resolve().parent.parent.parent / "docs" / "benchmarks"

# Cost / scale controls -- override via env for the audit cycle.
_DEFAULT_N_BILLS = int(os.environ.get("KAOS_BILLSUM_N", "20"))
_DEFAULT_CHAR_CAP = int(os.environ.get("KAOS_BILLSUM_CHAR_CAP", "15000"))


def _live_provider_available() -> bool:
    return bool(os.environ.get("KAOS_LLM_LIVE_PROVIDER"))


@pytest.fixture(scope="module")
def billsum_pairs() -> list[tuple[str, str]]:
    """Return ``[(bill_text, reference_summary), ...]`` from BillSum test split.

    Skips cleanly if HuggingFace ``datasets`` isn't installed.
    """
    datasets = pytest.importorskip(
        "datasets",
        reason="BillSum harness requires `pip install datasets`",
    )
    # Load the public BillSum test split. The dataset stores
    # ``text`` (bill body) and ``summary`` (human reference).
    try:
        ds = datasets.load_dataset("FiscalNote/billsum", split="test")
    except Exception:
        # Fallback: legacy mirror
        ds = datasets.load_dataset("billsum", split="test")
    rows: list[tuple[str, str]] = []
    for row in ds:
        text = row.get("text") or ""
        ref = row.get("summary") or ""
        if len(text) < 1000 or len(ref) < 100:
            continue
        if len(text) > _DEFAULT_CHAR_CAP:
            text = text[:_DEFAULT_CHAR_CAP]
        rows.append((text, ref))
        if len(rows) >= _DEFAULT_N_BILLS:
            break
    return rows


@pytest.fixture(scope="module")
def rouge_scorer_fixture():
    rouge_score = pytest.importorskip(
        "rouge_score",
        reason="BillSum harness requires `pip install rouge-score`",
    )
    return rouge_score.rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


async def _abstractive(text: str) -> tuple[str, dict]:
    from kaos_llm_core.programs.summarize import AbstractiveSummary

    prog = AbstractiveSummary()
    summary = await prog(text=text)
    return summary.text or "", summary.metadata or {}


async def _hierarchical(text: str) -> tuple[str, dict]:
    from kaos_llm_core.programs.summarize import HierarchicalSummary

    prog = HierarchicalSummary()
    summary = await prog(text=text)
    return summary.text or "", summary.metadata or {}


async def _map_reduce(text: str) -> tuple[str, dict]:
    from kaos_llm_core.programs.summarize import MapReduceSummary

    prog = MapReduceSummary()
    summary = await prog(text=text)
    return summary.text or "", summary.metadata or {}


_PROGRAMS = (
    ("AbstractiveSummary", _abstractive),
    ("HierarchicalSummary", _hierarchical),
    ("MapReduceSummary", _map_reduce),
)


def _score(rouge_scorer, prediction: str, reference: str) -> dict[str, float]:
    if not prediction or not reference:
        return {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0}
    s = rouge_scorer.score(reference, prediction)
    return {
        "rouge1_f": round(s["rouge1"].fmeasure, 4),
        "rouge2_f": round(s["rouge2"].fmeasure, 4),
        "rougeL_f": round(s["rougeL"].fmeasure, 4),
    }


@pytest.mark.skipif(
    not _live_provider_available(),
    reason="KAOS_LLM_LIVE_PROVIDER not set; BillSum harness requires live LLM access",
)
def test_billsum_rouge(billsum_pairs, rouge_scorer_fixture) -> None:
    """Per-bill ROUGE for each summarization Program; aggregate by Program."""
    if not billsum_pairs:
        pytest.skip("BillSum test split returned no usable rows")

    per_cell: list[dict] = []
    for program_name, runner in _PROGRAMS:
        for idx, (text, reference) in enumerate(billsum_pairs):
            try:
                prediction, meta = asyncio.run(runner(text))
            except Exception as exc:
                per_cell.append(
                    {
                        "program": program_name,
                        "bill_index": idx,
                        "error": str(exc),
                        "text_chars": len(text),
                        "reference_chars": len(reference),
                    }
                )
                continue
            scores = _score(rouge_scorer_fixture, prediction, reference)
            per_cell.append(
                {
                    "program": program_name,
                    "bill_index": idx,
                    "text_chars": len(text),
                    "reference_chars": len(reference),
                    "prediction_chars": len(prediction),
                    "rouge1_f": scores["rouge1_f"],
                    "rouge2_f": scores["rouge2_f"],
                    "rougeL_f": scores["rougeL_f"],
                    "claims_total": int(meta.get("claims_total", 0))
                    if isinstance(meta, dict)
                    else 0,
                    "claims_verified": int(meta.get("claims_verified", 0))
                    if isinstance(meta, dict)
                    else 0,
                }
            )

    # Aggregate by Program.
    by_program: dict[str, list[dict]] = {}
    for r in per_cell:
        by_program.setdefault(r["program"], []).append(r)

    aggregate: dict[str, dict] = {}
    for program_name, rows in by_program.items():
        scored = [r for r in rows if "rouge1_f" in r]
        if not scored:
            aggregate[program_name] = {"n": 0, "errors": len(rows)}
            continue

        def _mean(key: str) -> float:
            return round(sum(r[key] for r in scored) / len(scored), 4)

        aggregate[program_name] = {
            "n": len(scored),
            "n_errors": len(rows) - len(scored),
            "rouge1_f_mean": _mean("rouge1_f"),
            "rouge2_f_mean": _mean("rouge2_f"),
            "rougeL_f_mean": _mean("rougeL_f"),
        }

    payload = {
        "harness": "BillSum ROUGE",
        "dataset": "FiscalNote/billsum (Apache-2.0)",
        "n_bills_per_program": len(billsum_pairs),
        "captured_at": "2026-05-15",
        "aggregate": aggregate,
        "per_cell": per_cell,
    }

    _BENCH_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _BENCH_DIR / "quality-billsum-rouge.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")

    if os.environ.get("KAOS_LLM_BENCH_PRINT"):
        print("\nBillSum ROUGE aggregate:")
        for prog, agg in aggregate.items():
            print(f"  {prog:>22s}: {agg}")

    # Soft assertion: at minimum AbstractiveSummary should produce
    # non-trivial ROUGE-L on summarisation it was designed for.
    abs_agg = aggregate.get("AbstractiveSummary", {})
    assert abs_agg.get("n", 0) > 0, "AbstractiveSummary produced no successful rows"
    assert abs_agg.get("rougeL_f_mean", 0.0) > 0.05, (
        f"AbstractiveSummary ROUGE-L mean is suspiciously low "
        f"({abs_agg.get('rougeL_f_mean')}); inspect per_cell for "
        f"refusals or short outputs."
    )
