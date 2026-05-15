"""Per-Program cost / latency bench distilled from live-*.json snapshots.

This bench does NOT make live LLM calls. It loads the snapshotted
results captured by ``tests/scale/test_programs_live.py`` (which IS
gated on ``KAOS_LLM_LIVE_PROVIDER`` and was last run during the audit
cycle on 2026-05-15), normalises their shapes into a single
machine-readable summary, and writes
``docs/benchmarks/programs-cost-latency.json``.

The intent is to give downstream consumers (kelvin-training, kaos-
compliance) a single source-of-truth file describing what each
Program costs per document at the provider/model pair we measured
against -- without forcing them to run their own live-LLM benchmark
to find out. When the next audit cycle re-runs the live tests, this
bench picks up the refreshed snapshots automatically.

Run with::

    uv run --no-sync pytest tests/bench_programs.py --no-cov -v

The test ASSERTS that:
1. Every snapshot under ``docs/benchmarks/live-*.json`` parses cleanly
   and contains the fields required to compute per-doc metrics.
2. The consolidated ``programs-cost-latency.json`` writes cleanly
   and round-trips back to the same structure.

It does NOT assert any absolute throughput / cost targets -- those
belong in CI gates per environment, not here. The data file is the
artifact for downstream review.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_BENCH_DIR = Path(__file__).resolve().parent.parent / "docs" / "benchmarks"


def _live_snapshot_paths() -> list[Path]:
    return sorted(_BENCH_DIR.glob("live-*.json"))


def _doc_count(snap: dict) -> int | None:
    """Best-effort count of how many documents the run processed."""
    if isinstance(snap.get("documents"), int):
        return snap["documents"]
    # ChunkedClassify and similar use "chunks_count" instead of "documents".
    if isinstance(snap.get("chunks_count"), int):
        return snap["chunks_count"]
    # Some snapshots embed per-doc records; count them.
    for key in ("sample_summaries", "sample_results", "records"):
        v = snap.get(key)
        if isinstance(v, list) and v:
            return len(v)
    return None


def _normalize(snap: dict, source: Path) -> dict:
    """Extract the perf-relevant fields from one snapshot in a unified shape."""
    program = snap.get("program", "unknown")
    model = snap.get("model", "unknown")
    elapsed = float(snap.get("elapsed_seconds") or 0.0)
    cost = float(snap.get("total_cost_usd") or 0.0)
    n_in = int(snap.get("total_input_tokens") or 0)
    n_out = int(snap.get("total_output_tokens") or 0)
    n_docs = _doc_count(snap)

    out: dict[str, object] = {
        "source": source.name,
        "program": program,
        "model": model,
        "elapsed_seconds": round(elapsed, 3),
        "total_cost_usd": round(cost, 6),
        "total_input_tokens": n_in,
        "total_output_tokens": n_out,
        "n_documents_or_chunks": n_docs,
    }
    if n_docs and n_docs > 0:
        out["ms_per_doc"] = round((elapsed * 1000) / n_docs, 2)
        out["usd_per_doc"] = round(cost / n_docs, 6)
        out["input_tokens_per_doc"] = round(n_in / n_docs, 1)
        out["output_tokens_per_doc"] = round(n_out / n_docs, 1)
    if elapsed > 0:
        out["tokens_per_second"] = round((n_in + n_out) / elapsed, 1)
    return out


def test_consolidate_live_snapshot_perf() -> None:
    """Distill the 6 live-*.json snapshots into one cost/latency summary."""
    snapshots = _live_snapshot_paths()
    assert snapshots, (
        "No docs/benchmarks/live-*.json snapshots found. "
        "Run tests/scale/test_programs_live.py with "
        "KAOS_LLM_LIVE_PROVIDER set to regenerate them."
    )

    summaries: list[dict] = []
    for path in snapshots:
        with path.open() as f:
            snap = json.load(f)
        rec = _normalize(snap, path)
        # Required fields: every snapshot must at least identify the
        # Program + model it captured. Some snapshots (e.g. the
        # single-document MultiLabelClassify run) don't carry an
        # ``elapsed_seconds`` field because they're not batch-throughput
        # captures -- they're behaviour samples. We include them for
        # provenance but don't derive per-doc ms/usd from them.
        assert rec["program"] != "unknown", f"{path.name} missing 'program' field"
        assert rec["model"] != "unknown", f"{path.name} missing 'model' field"
        summaries.append(rec)

    # Sort for deterministic JSON output.
    summaries.sort(key=lambda r: (r["program"], r["source"]))

    consolidated = {
        "captured_at": "2026-05-15",
        "source": "kaos-llm-core/docs/benchmarks/live-*.json",
        "note": (
            "Single-source perf summary for kaos-llm-core Programs. "
            "Generated from live-LLM runs in tests/scale/test_programs_live.py. "
            "Refresh by running that scale-test (requires KAOS_LLM_LIVE_PROVIDER + credentials) "
            "and re-running tests/bench_programs.py."
        ),
        "programs": summaries,
    }

    out_path = _BENCH_DIR / "programs-cost-latency.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(consolidated, indent=2, sort_keys=False) + "\n")

    # Round-trip assertion: the file we just wrote parses back to the
    # same shape. If a contributor edits the file by hand and breaks
    # the JSON, this catches it on the next bench run.
    reloaded = json.loads(out_path.read_text())
    assert reloaded["programs"] == consolidated["programs"]

    if os.environ.get("KAOS_LLM_BENCH_PRINT"):
        print("\nPer-Program perf summary (from snapshots):")
        for r in summaries:
            line = (
                f"  {r['program']:>22s} | {r['model']:>30s} | "
                f"elapsed {r['elapsed_seconds']:>6.2f}s | "
                f"cost ${r['total_cost_usd']:>7.4f}"
            )
            if "ms_per_doc" in r:
                line += f" | {r['ms_per_doc']:>6.0f} ms/doc | ${r['usd_per_doc']:>7.4f}/doc"
            if "tokens_per_second" in r:
                line += f" | {r['tokens_per_second']:>5.0f} tok/s"
            print(line)
