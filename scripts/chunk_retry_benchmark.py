"""WS-TR.PR-6b chunk-retry calibration on a long-document fixture.

Runs the same Extract program TWICE per model on the same long
document — once with ``chunk_retry=None`` (baseline) and once with
``chunk_retry="default"`` — and reports per-cell recovery (cells whose
status transitioned from ``not_in_document`` / ``unclear`` /
``error`` on the baseline to ``extracted`` on the retry pass).

The hypothesis being tested: chunk-retry recovers cells that the
single-pass LLM missed because the relevant text was diluted by
context length.

Output: ``docs/benchmarks/chunk-retry-calibration-{YYYY-MM-DD}.{json,md}``

Usage::

    uv run python scripts/chunk_retry_benchmark.py
    uv run python scripts/chunk_retry_benchmark.py --models anthropic:claude-haiku-4-5
    uv run python scripts/chunk_retry_benchmark.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from kaos_llm_core.programs.chunk_retry import ChunkRetryConfig
from kaos_llm_core.programs.extract import Extract
from kaos_llm_core.signatures.extraction import ColumnSpec, ExtractionSchema

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "kaos-llm-core" / "tests" / "fixtures" / "long-doc-sample"
GOLDEN_PATH = FIXTURE_DIR / "golden.jsonl"
OUT_DIR = REPO_ROOT / "docs" / "benchmarks"

# Cheap-tier models per kaos-llm-core/CLAUDE.md guidance — full-tier
# models would be wasteful for a chunk-retry boundary test.
DEFAULT_MODELS: tuple[str, ...] = (
    "anthropic:claude-haiku-4-5",
    "openai:gpt-5.4-nano",
    "google:gemini-2.5-flash",
)

SCHEMA = ExtractionSchema(
    id="long-doc-extraction-v1",
    version=1,
    columns=(
        ColumnSpec(
            id="parties",
            label="Parties",
            column_type="list",
            constraints={"inner": "string", "alpha_extractor": "entity"},
            description=(
                "List the legal entities and named individuals entering "
                "into this agreement (e.g., the company and the employee)."
            ),
        ),
        ColumnSpec(
            id="effective_date",
            label="Effective Date",
            column_type="date",
            description=("The date on which this agreement becomes effective."),
        ),
        ColumnSpec(
            id="governing_law",
            label="Governing Law",
            column_type="string",
            description=(
                "The jurisdiction or governing-law clause "
                "(e.g., 'State of New York'). Verbatim citation preferred."
            ),
        ),
        ColumnSpec(
            id="termination_for_convenience",
            label="Termination for Convenience",
            column_type="string",
            description=(
                "The clause that lets either party terminate the agreement "
                "without cause. Quote the operative sentence."
            ),
        ),
    ),
)


@dataclass
class CellJudgment:
    column_id: str
    baseline_status: str
    retry_status: str
    recovered: bool
    matched_gold: bool
    baseline_value: str | None = None
    retry_value: str | None = None


@dataclass
class ModelReport:
    model: str
    skipped: bool = False
    reason: str | None = None
    n_columns: int = 0
    baseline_cost_usd: float = 0.0
    retry_cost_usd: float = 0.0
    n_recovered: int = 0
    judgments: list[CellJudgment] = field(default_factory=list)


def _load_fixture() -> tuple[str, dict[str, dict[str, list[str]]]]:
    text_path = next(FIXTURE_DIR.glob("*.txt"), None)
    if text_path is None:
        raise FileNotFoundError(f"no .txt fixture in {FIXTURE_DIR}")
    text = text_path.read_text(encoding="utf-8")
    golden: dict[str, dict[str, list[str]]] = {}
    for line in GOLDEN_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        golden[row["doc_id"]] = row.get("clause_answers", {})
    return text, golden


def _norm(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).lower().split())


def _matches_gold(value: Any, gold_spans: list[str]) -> bool:
    """Lenient substring match against any gold span (case- and
    whitespace-insensitive)."""
    if value is None or not gold_spans:
        return False
    if isinstance(value, list):
        return any(_matches_gold(v, gold_spans) for v in value)
    norm_v = _norm(value)
    if not norm_v:
        return False
    for g in gold_spans:
        ng = _norm(g)
        if ng and (ng in norm_v or norm_v in ng):
            return True
    return False


async def _run_model(
    model: str,
    text: str,
    gold_for_doc: dict[str, list[str]],
    *,
    dry_run: bool = False,
) -> ModelReport:
    report = ModelReport(model=model, n_columns=len(SCHEMA.columns))
    if dry_run:
        report.skipped = True
        report.reason = "dry-run"
        return report

    # Baseline: no chunk-retry, no alpha merger (we want a CLEAN
    # baseline for chunk-retry's lift; alpha would obscure it).
    baseline_extractor = Extract(
        SCHEMA,
        model=model,
        provenance="none",
        alpha_merger=None,
        chunk_retry=None,
    )
    try:
        baseline = await baseline_extractor.extract(text=text, doc_id="ppd-employment-agreement")
    except Exception as exc:
        report.skipped = True
        report.reason = f"baseline failed: {type(exc).__name__}: {exc}"
        return report
    report.baseline_cost_usd = baseline.cost_usd

    # Retry: chunk-retry enabled with default config.
    retry_extractor = Extract(
        SCHEMA,
        model=model,
        provenance="none",
        alpha_merger=None,
        chunk_retry="default",
    )
    try:
        retry = await retry_extractor.extract(text=text, doc_id="ppd-employment-agreement")
    except Exception as exc:
        report.skipped = True
        report.reason = f"retry failed: {type(exc).__name__}: {exc}"
        return report
    report.retry_cost_usd = retry.cost_usd

    # Per-cell comparison.
    baseline_by_col = {c.column_id: c for c in baseline.cells}
    retry_by_col = {c.column_id: c for c in retry.cells}
    for col in SCHEMA.columns:
        bcell = baseline_by_col.get(col.id)
        rcell = retry_by_col.get(col.id)
        if bcell is None or rcell is None:
            continue
        gold_spans = gold_for_doc.get(col.id, [])
        recovered = bcell.status != "extracted" and rcell.status == "extracted"
        matched = _matches_gold(rcell.ai_value, gold_spans)
        if recovered:
            report.n_recovered += 1
        report.judgments.append(
            CellJudgment(
                column_id=col.id,
                baseline_status=bcell.status,
                retry_status=rcell.status,
                recovered=recovered,
                matched_gold=matched,
                baseline_value=str(bcell.ai_value)[:120] if bcell.ai_value is not None else None,
                retry_value=str(rcell.ai_value)[:120] if rcell.ai_value is not None else None,
            )
        )

    return report


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_markdown(path: Path, reports: list[ModelReport], meta: dict[str, Any]) -> None:
    lines = [
        f"# {meta['title']}",
        "",
        f"**Date:** {meta['date']}  ",
        f"**Fixture:** `{meta['fixture']}`  ",
        f"**Document size:** {meta['doc_size_chars']:,} chars  ",
        f"**Schema:** `{meta['schema_id']}` v{meta['schema_version']}  ",
        f"**Default chunk size:** "
        f"{ChunkRetryConfig().initial_chunk_chars:,} chars (≈ "
        f"{meta['doc_size_chars'] // ChunkRetryConfig().initial_chunk_chars} "
        f"chunks)",
        "",
        "## Per-model recovery summary",
        "",
        "| Model | Baseline cost | Retry cost | $ delta | Cells recovered | Status |",
        "|-------|---:|---:|---:|---:|--------|",
    ]
    for r in reports:
        if r.skipped:
            lines.append(f"| `{r.model}` | — | — | — | — | skipped: {r.reason} |")
            continue
        delta = r.retry_cost_usd - r.baseline_cost_usd
        lines.append(
            f"| `{r.model}` | ${r.baseline_cost_usd:.4f} | "
            f"${r.retry_cost_usd:.4f} | ${delta:+.4f} | "
            f"{r.n_recovered} of {r.n_columns} | ok |"
        )

    lines.extend(["", "## Per-cell judgments", ""])
    for r in reports:
        if r.skipped:
            continue
        lines.extend([f"### `{r.model}`", ""])
        lines.append("| column | baseline | retry | recovered | matched gold |")
        lines.append("|--------|----------|-------|:---------:|:------------:|")
        for j in r.judgments:
            recov = "✓" if j.recovered else ""
            matched = "✓" if j.matched_gold else "✗"
            lines.append(
                f"| {j.column_id} | {j.baseline_status} | {j.retry_status} | {recov} | {matched} |"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


async def _amain(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Provider:model strings to benchmark.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--suffix", default="")
    args = parser.parse_args(argv)

    text, golden = _load_fixture()
    gold_for_doc = golden.get("ppd-employment-agreement", {})

    date_str = datetime.date.today().isoformat()
    out_dir = Path(args.out_dir)
    suffix = args.suffix or ""
    json_path = out_dir / f"chunk-retry-calibration-{date_str}{suffix}.json"
    md_path = out_dir / f"chunk-retry-calibration-{date_str}{suffix}.md"

    reports: list[ModelReport] = []
    for model in args.models:
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
                    reason=f"no API key for provider {provider!r}",
                )
            )
            continue
        print(f"[bench] running {model}...", file=sys.stderr)
        report = await _run_model(model, text, gold_for_doc, dry_run=args.dry_run)
        if report.skipped:
            print(f"[bench] {model}: skipped ({report.reason})", file=sys.stderr)
        else:
            print(
                f"[bench] {model}: recovered {report.n_recovered}/"
                f"{report.n_columns} cells "
                f"(${report.retry_cost_usd - report.baseline_cost_usd:+.4f})",
                file=sys.stderr,
            )
        reports.append(report)

    meta = {
        "title": "WS-TR.PR-6b chunk-retry calibration",
        "date": date_str,
        "fixture": "kaos-llm-core/tests/fixtures/long-doc-sample/ppd-employment-agreement.txt",
        "doc_size_chars": len(text),
        "schema_id": SCHEMA.id,
        "schema_version": SCHEMA.version,
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
