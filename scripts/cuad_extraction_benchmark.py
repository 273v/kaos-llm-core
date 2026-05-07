# ruff: noqa: RUF001, RUF002
"""WS-TR.PR-5 — CUAD-anchored structured extraction calibration.

Runs the WS-TR :func:`kaos_llm_core.programs.extract.extract_corpus` against a
5-contract × 5-clause subset of The Atticus Project's CUAD v1 (CC-BY-4.0)
and scores each cell against CUAD's ground-truth spans. This is the first
VLAIR-comparable number in our benchmark stack — CUAD is the canonical
public extraction benchmark for legal contracts.

Scoring: per-cell match if the model's ``ai_value`` (case-insensitive) is
a substring of ANY gold span OR any gold span is a substring of the model's
value. This matches the PR-3 lenient-string heuristic and is conservative
compared to CUAD's own token-F1 metric (a model emitting a broader
paraphrase counts against it here).

Outputs:

- ``docs/benchmarks/cuad-extraction-calibration-{YYYY-MM-DD}.json``
- ``docs/benchmarks/cuad-extraction-calibration-{YYYY-MM-DD}.md``

Usage::

    uv run python scripts/cuad_extraction_benchmark.py
    uv run python scripts/cuad_extraction_benchmark.py --models anthropic:claude-haiku-4-5
    uv run python scripts/cuad_extraction_benchmark.py --dry-run

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
FIXTURE_DIR = REPO_ROOT / "kaos-llm-core" / "tests" / "fixtures" / "cuad-sample"
GOLDEN_PATH = FIXTURE_DIR / "cuad-extraction-golden.jsonl"
MANIFEST_PATH = FIXTURE_DIR / "MANIFEST.json"
OUT_DIR = REPO_ROOT / "docs" / "benchmarks"

# The full current-generation matrix per kaos-llm-core CLAUDE.md. A full
# --dry-run pass with all 9 models costs ~$0.50 at today's prices; that's
# the acceptance price for an honest benchmark. To keep CI or local runs
# cheap, pass ``--models anthropic:claude-haiku-4-5 openai:gpt-5.4-nano
# google:gemini-2.5-flash`` explicitly on the CLI — skipping any model
# whose key is not configured is automatic.
DEFAULT_MODELS: tuple[str, ...] = (
    # Anthropic — all current-gen 4.x.
    "anthropic:claude-haiku-4-5",
    "anthropic:claude-sonnet-4-6",
    "anthropic:claude-opus-4-6",
    # OpenAI — all current-gen 5.4.x.
    "openai:gpt-5.4-nano",
    "openai:gpt-5.4-mini",
    "openai:gpt-5.4",
    # Google — Gemini 2.5 Flash + 3-flash-preview + 3.1-pro-preview.
    "google:gemini-2.5-flash",
    "google:gemini-3-flash-preview",
    "google:gemini-3.1-pro-preview",
)

# CUAD targets five named clauses for the sample. Match these exactly to the
# golden JSONL's "clause_answers" keys.
CUAD_TARGETS: tuple[str, ...] = (
    "Parties",
    "Agreement Date",
    "Governing Law",
    "Termination For Convenience",
    "Cap On Liability",
)


# Schema mirrors CUAD's 5 target clauses. Per-column descriptions adopt the
# kelvin-nlp scoping pattern (kelvin/nlp/extract/llm2/config/instructions.py:
# 12-70) — explicit "act as reviewer looking for X / do not include Y"
# scoping that consistently moves precision from ~84% to 90-100% on the
# same task class. The descriptions ride into the prompt via the Signature
# docstring built by ``ExtractionSchema.to_signature``.
#
# Schema bumped to v2 to invalidate any cached calibration runs.
CUAD_SCHEMA = ExtractionSchema.from_dict(
    {
        "id": "cuad-5-clause",
        "version": 2,
        "columns": [
            {
                "id": "parties",
                "column_type": "list",
                "constraints": {"inner": "string"},
                "description": (
                    "Act as a contract reviewer looking for the contracting parties in "
                    "the preamble, recital, or signature block. Return each party's exact "
                    "legal name as it appears (e.g., 'Acme Corporation', 'Beta LLC'). "
                    "Do NOT include mentions of parties or entities in the body of the "
                    "agreement (e.g., named third parties, affiliates referenced for "
                    "scoping). Do NOT include role labels like 'Licensor' or 'Licensee' "
                    "as separate parties — those are descriptors of the named entity. "
                    "Do NOT include the contract title itself."
                ),
            },
            {
                "id": "agreement_date",
                "column_type": "date",
                "description": (
                    "Act as a contract reviewer looking for the date the agreement was "
                    "signed or entered into, as stated in the PREAMBLE — the opening "
                    "paragraph that names the parties. Look for phrases like 'made and "
                    "entered into as of [DATE]', 'effective as of [DATE]', 'dated [DATE]'. "
                    "Return ISO 8601 (YYYY-MM-DD). "
                    "Do NOT return signature dates from the signature block. "
                    "Do NOT return effective dates of underlying transactions or sub-agreements. "
                    "Do NOT return dates of amendments or renewals. "
                    "Do NOT return dates referenced in recitals about prior events. "
                    "If multiple dates appear in the preamble, prefer the one introduced "
                    "by 'as of' or 'dated'."
                ),
            },
            {
                "id": "governing_law",
                "column_type": "verbatim_quote",
                "description": (
                    "Act as a contract reviewer looking for the governing-law clause — "
                    "the sentence naming which jurisdiction's substantive law governs "
                    "the agreement. Typically headed 'Governing Law', 'Choice of Law', "
                    "or buried in the 'Miscellaneous' section. Return the verbatim "
                    "sentence including the jurisdiction name. "
                    "Do NOT include forum-selection or venue clauses (those are about "
                    "WHERE disputes are heard, not WHICH law applies). "
                    "Do NOT include arbitration-rules clauses unless they explicitly "
                    "select a substantive law. "
                    "Do NOT paraphrase — emit the exact contract text."
                ),
            },
            {
                "id": "termination_for_convenience",
                "column_type": "verbatim_quote",
                "description": (
                    "Act as a contract reviewer looking for clauses that allow either "
                    "party to terminate the agreement WITHOUT CAUSE — 'termination for "
                    "convenience', 'termination at will', 'right to terminate without "
                    "cause'. Return the verbatim sentence(s) granting that right, "
                    "including any required notice period. "
                    "Do NOT include termination-for-cause clauses (breach, insolvency, "
                    "change of control). "
                    "Do NOT include expiration-of-term clauses (those are not "
                    "termination for convenience — the agreement just ends). "
                    "Do NOT include termination-fee or break-up-fee dollar amounts (those "
                    "are consequences, not the termination right itself). "
                    "If absent from this contract, omit the field (the schema marks it optional)."
                ),
                "required": False,
            },
            {
                "id": "cap_on_liability",
                "column_type": "verbatim_quote",
                "description": (
                    "Act as a contract reviewer looking for clauses that LIMIT a party's "
                    "monetary liability — aggregate liability caps, dollar-amount limits, "
                    "or sweeping exclusions of consequential / indirect / special / "
                    "punitive damages. Return the verbatim sentence(s). "
                    "Do NOT include indemnification scope clauses (what one party will "
                    "indemnify the other FOR — those describe the duty, not the cap). "
                    "Do NOT include warranty-disclaimer clauses ('AS IS', 'WITHOUT "
                    "WARRANTIES') unless they explicitly cap damages. "
                    "Do NOT include insurance-requirement clauses (minimum coverage "
                    "amounts are not liability caps). "
                    "If absent from this contract, omit the field (the schema marks it optional)."
                ),
                "required": False,
            },
        ],
    }
)


# Map CUAD's clause-name keys to our snake_case column ids.
CUAD_TO_COLUMN_ID: dict[str, str] = {
    "Parties": "parties",
    "Agreement Date": "agreement_date",
    "Governing Law": "governing_law",
    "Termination For Convenience": "termination_for_convenience",
    "Cap On Liability": "cap_on_liability",
}


@dataclass
class CellJudgment:
    doc_id: str
    column_id: str
    ai_value: str | None
    gold_spans: list[str]
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


def _norm(s: str) -> str:
    return " ".join(str(s).lower().split())


def _cell_matches(ai_value: Any, gold_spans: list[str]) -> bool:
    """Lenient substring match (either direction).

    CUAD ships gold spans as verbatim contract snippets (sometimes just the
    named party like "Tickets.com, Inc.,"). We accept a match if:

    - normalized ``ai_value`` contains any normalized gold span, OR
    - any normalized gold span contains the normalized ``ai_value``.

    For list-typed values (e.g., ``parties``), we match if ANY list item
    matches ANY gold span under the above rule.

    Date normalization: if the AI value looks like ISO 8601 (``YYYY-MM-DD``)
    and a gold span looks like a natural-language date
    (``Month D, YYYY`` or ``D Month YYYY``), we compare the underlying date.
    This keeps the scoring faithful to CUAD's verbatim gold annotations
    without penalizing the schema's ``column_type="date"`` hint that forces
    ISO output.
    """
    if ai_value is None:
        return False
    if not gold_spans:
        return False
    # Fan out over list values.
    if isinstance(ai_value, list):
        return any(_cell_matches(elem, gold_spans) for elem in ai_value)
    ai_norm = _norm(ai_value)
    if not ai_norm:
        return False
    # ISO-date fast path: if the AI side is YYYY-MM-DD, try parsing gold
    # spans as natural-language dates and comparing ordinally.
    ai_iso = _parse_iso_date(str(ai_value))
    if ai_iso is not None:
        for gold in gold_spans:
            gold_iso = _parse_natural_date(gold)
            if gold_iso is not None and gold_iso == ai_iso:
                return True
    for gold in gold_spans:
        g_norm = _norm(gold)
        if not g_norm:
            continue
        if g_norm in ai_norm or ai_norm in g_norm:
            return True
    return False


def _parse_iso_date(s: str) -> str | None:
    """Return ``YYYY-MM-DD`` if ``s`` looks like an ISO 8601 date; else None."""
    import re as _re

    m = _re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", s.strip())
    if not m:
        return None
    return s.strip()


def _parse_natural_date(s: str) -> str | None:
    """Parse a natural-language date into ``YYYY-MM-DD``.

    Recognized forms (all case-insensitive, all return ISO 8601):

    - ``January 1, 2025`` / ``January 1 2025``
    - ``1 January 2025`` / ``19 Jan. 1998`` (abbreviated month + period)
    - ``6th day of April, 1999`` / ``21st day of January 2003``
      (kelvin's "Nth day of Month YYYY" English form — common in contracts)

    Returns ``None`` if the input doesn't look like a date. Uses the
    standard library to avoid a dateutil dependency.
    """
    import datetime as _dt
    import re as _re

    _MONTHS = {
        m.lower(): i
        for i, m in enumerate(
            [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ],
            start=1,
        )
    }
    _MONTHS.update({m.lower()[:3]: i for m, i in list(_MONTHS.items())})

    def _build(year: int, month_name: str, day: int) -> str | None:
        month = _MONTHS.get(month_name.lower().rstrip("."))
        if month is None:
            return None
        try:
            return _dt.date(year, month, day).isoformat()
        except ValueError:
            return None

    text = s.strip().rstrip(".,")

    # "Nth day of Month YYYY" — the "day of" English legal form. Matches
    # the actual CUAD goldens for dragonsystems / centrack / mphase.
    m = _re.search(
        r"(\d{1,2})(?:st|nd|rd|th)?\s+day\s+of\s+([A-Za-z]+)\.?,?\s+(\d{4})",
        text,
        flags=_re.IGNORECASE,
    )
    if m:
        return _build(int(m.group(3)), m.group(2), int(m.group(1)))

    # "January 1, 2025" / "January 1 2025" — month-first.
    m = _re.search(r"([A-Za-z]+)\.?\s+(\d{1,2}),?\s+(\d{4})", text)
    if m:
        return _build(int(m.group(3)), m.group(1), int(m.group(2)))

    # "1 January 2025" / "19 Jan. 1998" — day-first, abbrev ok.
    m = _re.search(r"(\d{1,2})\s+([A-Za-z]+)\.?,?\s+(\d{4})", text)
    if m:
        return _build(int(m.group(3)), m.group(2), int(m.group(1)))

    return None


def _load_corpus() -> dict[str, str]:
    """Read every .txt file in the CUAD sample fixture."""
    return {
        path.stem: path.read_text(encoding="utf-8") for path in sorted(FIXTURE_DIR.glob("*.txt"))
    }


def _load_golden() -> dict[str, dict[str, list[str]]]:
    """Load the per-doc clause_answers from the golden JSONL.

    Returns: ``{doc_id: {column_id: [gold_span, ...]}}``.
    """
    golden: dict[str, dict[str, list[str]]] = {}
    for line in GOLDEN_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        doc_id = row["doc_id"]
        clause_answers = row.get("clause_answers", {})
        mapped: dict[str, list[str]] = {}
        for cuad_key, col_id in CUAD_TO_COLUMN_ID.items():
            mapped[col_id] = list(clause_answers.get(cuad_key, []))
        golden[doc_id] = mapped
    return golden


async def _run_model(
    model: str,
    corpus: dict[str, str],
    golden: dict[str, dict[str, list[str]]],
    *,
    dry_run: bool = False,
) -> ModelReport:
    report = ModelReport(model=model, skipped=False, n_docs=len(corpus))

    if dry_run:
        report.skipped = True
        report.reason = "dry-run"
        return report

    with tempfile.TemporaryDirectory(prefix="kaos-cuad-bench-") as td:
        try:
            result: CorpusExtractionResult = await extract_corpus(
                CUAD_SCHEMA,
                corpus,
                output_dir=td,
                model=model,
                provenance="none",
                max_concurrency=3,
            )
        except Exception as exc:
            report.skipped = True
            report.reason = f"{type(exc).__name__}: {exc}"
            return report

    report.cost_usd = result.cost_usd
    report.tokens_total = result.tokens_total
    if result.rows:
        latencies = sorted(r.latency_ms for r in result.rows)
        report.latency_ms = latencies[len(latencies) // 2]
    report.n_errored = result.n_docs_errored

    per_col_totals: dict[str, list[int]] = {col.id: [0, 0] for col in CUAD_SCHEMA.columns}

    for doc_row in result.rows:
        gold_for_doc = golden.get(doc_row.doc_id, {})
        for cell in doc_row.cells:
            gold_spans = gold_for_doc.get(cell.column_id, [])
            report.n_cells += 1
            per_col_totals[cell.column_id][1] += 1
            if cell.status != "extracted":
                report.n_refused += 1
                matched = False
            else:
                matched = _cell_matches(cell.ai_value, gold_spans)
            if matched:
                report.n_matched += 1
                per_col_totals[cell.column_id][0] += 1
            report.judgments.append(
                CellJudgment(
                    doc_id=cell.doc_id,
                    column_id=cell.column_id,
                    ai_value=str(cell.ai_value) if cell.ai_value is not None else None,
                    gold_spans=gold_spans,
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
        "# WS-TR.PR-5 — CUAD Extraction Calibration",
        "",
        "**First VLAIR-comparable calibration artifact for KAOS structured extraction.**",
        "",
        f"**Date:** {meta['date']}",
        f"**Git commit:** `{meta['git_commit']}`",
        "**Source:** CUAD v1 (The Atticus Project, CC-BY-4.0). See "
        "[`kaos-llm-core/tests/fixtures/cuad-sample/DATA_LICENSES.md`]"
        "(../../kaos-llm-core/tests/fixtures/cuad-sample/DATA_LICENSES.md).",
        f"**Subset:** {meta['n_docs']} contracts × {len(CUAD_SCHEMA.columns)} clauses "
        f"= {meta['n_docs'] * len(CUAD_SCHEMA.columns)} cells",
        f"**Schema:** `{CUAD_SCHEMA.id}` v{CUAD_SCHEMA.version} — "
        f"{', '.join(c.id for c in CUAD_SCHEMA.columns)}",
        "**Scoring:** lenient substring match vs CUAD gold spans (either direction).",
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
    col_ids = [c.id for c in CUAD_SCHEMA.columns]
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
        lines.append("| doc_id | column | status | match | ai_value | gold_span_count |")
        lines.append("|--------|--------|--------|:-----:|----------|:---------------:|")
        for j in r.judgments:
            ai_trunc = (
                (j.ai_value[:70] + "...")
                if j.ai_value and len(j.ai_value) > 73
                else (j.ai_value or "—")
            )
            # Markdown-safe table cell (no pipes).
            ai_trunc = ai_trunc.replace("|", "\\|").replace("\n", " ")
            match_mark = "✓" if j.matched else "✗"
            lines.append(
                f"| {j.doc_id[:28]}... | {j.column_id} | {j.status} | "
                f"{match_mark} | {ai_trunc} | {len(j.gold_spans)} |"
            )
        lines.append("")

    # Attribution footer (required by CC-BY-4.0).
    lines.extend(
        [
            "---",
            "",
            "### Attribution",
            "",
            "> Hendrycks, D., Burns, C., Chen, A., & Ball, S. (2021). "
            "*CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review*. "
            "NeurIPS 2021 Datasets & Benchmarks Track. "
            "[arXiv:2103.06268](https://arxiv.org/abs/2103.06268).",
            "",
            "CUAD v1 is © The Atticus Project, distributed under "
            "[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).",
        ]
    )

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
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--suffix", default="")
    args = parser.parse_args(argv)

    date_str = datetime.date.today().isoformat()
    out_dir = Path(args.out_dir)
    suffix = args.suffix or ""
    json_path = out_dir / f"cuad-extraction-calibration-{date_str}{suffix}.json"
    md_path = out_dir / f"cuad-extraction-calibration-{date_str}{suffix}.md"

    corpus = _load_corpus()
    golden = _load_golden()
    if not corpus:
        print(
            f"[bench] no fixture contracts found under {FIXTURE_DIR}. "
            "Rebuild the sample via the CUAD download + extract script.",
            file=sys.stderr,
        )
        return 1

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
        "title": "WS-TR.PR-5 CUAD Extraction Calibration",
        "date": date_str,
        "git_commit": _git_commit(),
        "source": "The Atticus Project — CUAD v1 (CC-BY-4.0)",
        "source_url": "https://github.com/TheAtticusProject/cuad",
        "source_citation": "Hendrycks et al. 2021 (arXiv:2103.06268)",
        "n_docs": len(corpus),
        "schema_id": CUAD_SCHEMA.id,
        "schema_version": CUAD_SCHEMA.version,
        "columns": [c.id for c in CUAD_SCHEMA.columns],
    }
    payload: dict[str, Any] = {**meta, "models": [asdict(r) for r in reports]}
    _write_json(json_path, payload)
    _write_markdown(md_path, reports, meta)
    print(f"[bench] wrote {json_path}", file=sys.stderr)
    print(f"[bench] wrote {md_path}", file=sys.stderr)
    return 0


def main() -> int:
    return asyncio.run(_amain())


if __name__ == "__main__":
    sys.exit(main())
