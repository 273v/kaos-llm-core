"""WS-TR.PR-3 — First calibration benchmark for structured extraction.

Runs :func:`kaos_llm_core.programs.extract.extract_corpus` against the
3-document subset of the WS-3.7 multiformat fixture
(rfc_2119.txt / nist_passwords.txt / federal_register_pfas_summary.md) using
a 3-column schema (document_type / primary_topic / publishing_organization).

Each (model, column) pair is scored for:
- **precision**: fraction of cells with ai_value matching the golden value
  or any acceptable alternative (case-insensitive substring match for text
  fields). This is the "does the model extract the right thing" number.
- **refusal_rate**: fraction of cells where status != "extracted". Ideally
  0 for answerable-on-every-doc schemas like this one.

Per-run cost + latency + token counts are aggregated from batch_run's
BatchResult (authoritative via TrialRunner cost attribution).

Outputs:
- ``docs/benchmarks/extraction-calibration-{YYYY-MM-DD}.json`` (full results)
- ``docs/benchmarks/extraction-calibration-{YYYY-MM-DD}.md``  (human-readable)

Usage::

    uv run python scripts/extraction_benchmark.py
    uv run python scripts/extraction_benchmark.py --models anthropic:claude-haiku-4-5
    uv run python scripts/extraction_benchmark.py --dry-run   # smoke, no LLM calls
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from kaos_llm_core.programs.extract import CorpusExtractionResult, extract_corpus
from kaos_llm_core.signatures.extraction import ExtractionSchema

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "kaos-llm-core" / "tests" / "fixtures" / "multiformat-corpus"
GOLDEN_PATH = FIXTURE_DIR / "extraction-golden.jsonl"
OUT_DIR = REPO_ROOT / "docs" / "benchmarks"

DEFAULT_MODELS: tuple[str, ...] = (
    "anthropic:claude-haiku-4-5",
    "openai:gpt-5.4-nano",
    "google:gemini-2.5-flash",
)

SCHEMA = ExtractionSchema.from_dict(
    {
        "id": "document-metadata-pr3",
        "version": 1,
        "columns": [
            {
                "id": "document_type",
                "column_type": "string",
                "description": (
                    "The general document type (e.g., 'RFC', 'NIST Special Publication', "
                    "'Federal Register notice')."
                ),
            },
            {
                "id": "primary_topic",
                "column_type": "string",
                "description": "One short phrase describing the primary topic.",
            },
            {
                "id": "publishing_organization",
                "column_type": "string",
                "description": (
                    "The organization that authored/published this document "
                    "(e.g., 'IETF', 'NIST', 'US EPA')."
                ),
            },
        ],
    }
)

FIXTURE_FILES: dict[str, str] = {
    "rfc_2119": "rfc_2119.txt",
    "nist_passwords": "nist_passwords.txt",
    "pfas_summary": "federal_register_pfas_summary.md",
}


@dataclass
class CellJudgment:
    doc_id: str
    column_id: str
    ai_value: str | None
    expected: str
    alternatives: tuple[str, ...]
    matched: bool
    status: str


@dataclass
class ModelReport:
    model: str
    skipped: bool
    reason: str | None = None
    n_docs: int = 0
    n_cells: int = 0
    n_matched: int = 0
    n_refused: int = 0
    n_errored: int = 0
    cost_usd: float = 0.0
    tokens_total: int = 0
    latency_ms: float = 0.0
    per_column_precision: dict[str, float] = field(default_factory=dict)
    judgments: list[CellJudgment] = field(default_factory=list)

    @property
    def precision(self) -> float:
        return self.n_matched / self.n_cells if self.n_cells else 0.0

    @property
    def refusal_rate(self) -> float:
        return self.n_refused / self.n_cells if self.n_cells else 0.0


def _load_corpus() -> dict[str, str]:
    return {
        doc_id: (FIXTURE_DIR / filename).read_text(encoding="utf-8")
        for doc_id, filename in FIXTURE_FILES.items()
    }


def _load_golden() -> dict[str, dict[str, Any]]:
    golden: dict[str, dict[str, Any]] = {}
    for line in GOLDEN_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        golden[row["doc_id"]] = row
    return golden


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def _cell_matches(ai_value: Any, expected: str, alternatives: tuple[str, ...]) -> bool:
    """Case-insensitive substring match against expected + alternatives."""
    if ai_value is None:
        return False
    ai_norm = _norm(str(ai_value))
    # Each expected phrase → match if substring of AI value OR AI value is
    # substring of expected (either direction to tolerate model verbosity).
    for target in (expected, *alternatives):
        t_norm = _norm(target)
        if not t_norm:
            continue
        if t_norm in ai_norm or ai_norm in t_norm:
            return True
    return False


async def _run_model(
    model: str,
    corpus: dict[str, str],
    golden: dict[str, dict[str, Any]],
    *,
    dry_run: bool = False,
) -> ModelReport:
    report = ModelReport(model=model, skipped=False, n_docs=len(corpus))

    if dry_run:
        report.skipped = True
        report.reason = "dry-run"
        return report

    with tempfile.TemporaryDirectory(prefix="kaos-extract-bench-") as td:
        try:
            result: CorpusExtractionResult = await extract_corpus(
                SCHEMA,
                corpus,
                output_dir=td,
                model=model,
                provenance="none",  # Bare values — we judge against golden directly.
                max_concurrency=3,
            )
        except Exception as exc:  # Auth, transport, etc.
            report.skipped = True
            report.reason = f"{type(exc).__name__}: {exc}"
            return report

    report.cost_usd = result.cost_usd
    report.tokens_total = result.tokens_total
    # Approx mean per-doc latency (extract_corpus doesn't carry per-row
    # latency in aggregate). Use median of per-doc latencies from the
    # hydrated rows.
    if result.rows:
        latencies = sorted(r.latency_ms for r in result.rows)
        report.latency_ms = latencies[len(latencies) // 2]
    report.n_errored = result.n_docs_errored

    per_col_totals: dict[str, list[int]] = {col.id: [0, 0] for col in SCHEMA.columns}

    # Build a doc_id → ExtractionResult map from the per-row outputs so we
    # assess each cell directly (not via the pivoted table which projects
    # reviewed_value or loses status).
    for doc_row in result.rows:
        doc_golden = golden.get(doc_row.doc_id)
        if doc_golden is None:
            continue
        expected_map: dict[str, str] = doc_golden["expected"]
        alts_map: dict[str, list[str]] = doc_golden.get("acceptable_alternatives", {})

        for cell in doc_row.cells:
            expected = expected_map.get(cell.column_id, "")
            alts = tuple(alts_map.get(cell.column_id, []))
            report.n_cells += 1
            per_col_totals.setdefault(cell.column_id, [0, 0])
            per_col_totals[cell.column_id][1] += 1  # total
            if cell.status != "extracted":
                report.n_refused += 1
                matched = False
            else:
                matched = _cell_matches(cell.ai_value, expected, alts)
            if matched:
                report.n_matched += 1
                per_col_totals[cell.column_id][0] += 1
            report.judgments.append(
                CellJudgment(
                    doc_id=cell.doc_id,
                    column_id=cell.column_id,
                    ai_value=str(cell.ai_value) if cell.ai_value is not None else None,
                    expected=expected,
                    alternatives=alts,
                    matched=matched,
                    status=cell.status,
                )
            )

    report.per_column_precision = {
        col: (matched / total) if total else 0.0 for col, (matched, total) in per_col_totals.items()
    }
    return report


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _write_markdown(path: Path, reports: list[ModelReport], meta: dict[str, Any]) -> None:
    lines: list[str] = [
        "# WS-TR.PR-3 — Structured Extraction Calibration",
        "",
        f"**Date:** {meta['date']}",
        f"**Git commit:** `{meta['git_commit']}`",
        f"**Corpus:** {meta['corpus_dir']}",
        f"**Docs:** {meta['n_docs']}",
        f"**Schema:** `{SCHEMA.id}` v{SCHEMA.version} "
        f"({len(SCHEMA.columns)} columns: "
        f"{', '.join(c.id for c in SCHEMA.columns)})",
        "",
        "## Summary",
        "",
        "| Model | Precision | Refusal rate | $/run | Tokens | Median latency (ms) | Status |",
        "|-------|----------:|-------------:|------:|-------:|--------------------:|--------|",
    ]
    for r in reports:
        status = "skipped" if r.skipped else "ok"
        if r.skipped:
            lines.append(f"| `{r.model}` | — | — | — | — | — | skipped: {r.reason} |")
        else:
            lines.append(
                f"| `{r.model}` | "
                f"{r.precision:.1%} ({r.n_matched}/{r.n_cells}) | "
                f"{r.refusal_rate:.1%} | "
                f"${r.cost_usd:.4f} | "
                f"{r.tokens_total} | "
                f"{r.latency_ms:.0f} | "
                f"{status} |"
            )
    lines.extend(["", "## Per-column precision", ""])
    col_ids = [c.id for c in SCHEMA.columns]
    lines.append("| Model | " + " | ".join(col_ids) + " |")
    lines.append("|-------|" + "|".join(["---:"] * len(col_ids)) + "|")
    for r in reports:
        if r.skipped:
            lines.append(f"| `{r.model}` | " + " | ".join(["—"] * len(col_ids)) + " |")
        else:
            lines.append(
                f"| `{r.model}` | "
                + " | ".join(f"{r.per_column_precision.get(c, 0.0):.1%}" for c in col_ids)
                + " |"
            )

    lines.extend(["", "## Per-cell judgments", ""])
    for r in reports:
        if r.skipped:
            continue
        lines.append(f"### `{r.model}`")
        lines.append("")
        lines.append("| doc_id | column | status | match | ai_value | expected |")
        lines.append("|--------|--------|--------|:-----:|----------|----------|")
        for j in r.judgments:
            ai_trunc = (
                (j.ai_value[:60] + "...")
                if j.ai_value and len(j.ai_value) > 63
                else (j.ai_value or "—")
            )
            exp_trunc = (j.expected[:40] + "...") if len(j.expected) > 43 else j.expected
            match_mark = "✓" if j.matched else "✗"
            lines.append(
                f"| {j.doc_id} | {j.column_id} | {j.status} | {match_mark} | "
                f"{ai_trunc} | {exp_trunc} |"
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


async def _amain(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Provider:model strings to benchmark.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip LLM calls; produce a report with all-skipped rows.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(OUT_DIR),
        help="Where to write the JSON + Markdown artifacts.",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Optional suffix for the output filenames (e.g., '-retry').",
    )
    args = parser.parse_args(argv)

    date_str = datetime.date.today().isoformat()
    out_dir = Path(args.out_dir)
    suffix = args.suffix or ""
    json_path = out_dir / f"extraction-calibration-{date_str}{suffix}.json"
    md_path = out_dir / f"extraction-calibration-{date_str}{suffix}.md"

    corpus = _load_corpus()
    golden = _load_golden()

    reports: list[ModelReport] = []
    for model in args.models:
        # Early-skip if the provider API key is not available — keep the
        # artifact runnable in CI without every provider configured.
        provider = model.split(":", 1)[0]
        env_map = {
            "anthropic": ("KAOS_LLM_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
            "openai": ("KAOS_LLM_OPENAI_API_KEY", "OPENAI_API_KEY"),
            "google": (
                "KAOS_LLM_GOOGLE_API_KEY",
                "GOOGLE_API_KEY",
                "GOOGLE_GENERATIVE_AI_API_KEY",
            ),
        }
        keys = env_map.get(provider, ())
        if keys and not any(os.getenv(k) for k in keys):
            reports.append(
                ModelReport(
                    model=model,
                    skipped=True,
                    reason=f"no API key for provider '{provider}'",
                )
            )
            continue

        print(f"[bench] running {model}...", file=sys.stderr)
        report = await _run_model(model, corpus, golden, dry_run=args.dry_run)
        status = "skipped" if report.skipped else f"precision={report.precision:.1%}"
        print(f"[bench] {model}: {status}", file=sys.stderr)
        reports.append(report)

    meta = {
        "title": "WS-TR.PR-3 Structured Extraction Calibration",
        "date": date_str,
        "git_commit": _git_commit(),
        "corpus_dir": str(FIXTURE_DIR),
        "n_docs": len(corpus),
        "schema_id": SCHEMA.id,
        "schema_version": SCHEMA.version,
        "columns": [c.id for c in SCHEMA.columns],
    }

    payload: dict[str, Any] = {
        **meta,
        "models": [asdict(r) for r in reports],
    }
    _write_json(json_path, payload)
    _write_markdown(md_path, reports, meta)

    print(f"[bench] wrote {json_path}", file=sys.stderr)
    print(f"[bench] wrote {md_path}", file=sys.stderr)
    return 0


def main() -> int:
    return asyncio.run(_amain())


if __name__ == "__main__":
    sys.exit(main())
