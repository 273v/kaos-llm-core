"""Grounding calibration report generator (WS-1.6).

Runs the ``tests/fixtures/grounding-corpus`` + ``grounding-questions.jsonl``
harness against one or more live providers and writes:

- ``docs/benchmarks/grounding-calibration-{date}.json`` — per-(model, question)
  results plus per-model aggregates (precision on answerable, refusal recall
  on unanswerable, mean confidence, total cost, mean latency).
- ``docs/benchmarks/grounding-calibration-{date}.md`` — concise markdown
  digest summarising the JSON.

Usage:

  uv run python scripts/grounding_calibration.py --help
  uv run python scripts/grounding_calibration.py                  # all 3 providers
  uv run python scripts/grounding_calibration.py --models anthropic:claude-haiku-4-5
  uv run python scripts/grounding_calibration.py --dry-run        # no API calls
  uv run python scripts/grounding_calibration.py --sample 5       # smoke run

Per CLAUDE.md: use current-gen models only. The script auto-skips any model
whose provider API key is not set in the environment (KAOS_LLM_*_API_KEY or
the legacy OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY names).

As of WS-3.7.2a this script is a thin shell around
:mod:`kaos_llm_core.calibration`, which hosts the shared dataclasses,
per-(model, question) runner, and artifact writers that are reused by
``scripts/multiformat_benchmark.py`` and any future calibration harness.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import pathlib
import sys

from kaos_core.logging import get_logger

from kaos_llm_core.calibration import (
    DEFAULT_LENIENT_STRATEGIES,
    CalibrationReport,
    build_parser,
    dry_run_report,
    git_commit,
    load_questions_jsonl,
    run_model,
    write_json,
    write_markdown,
)

logger = get_logger(__name__)

_CORPUS_DIR = (
    pathlib.Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "grounding-corpus"
)

_URI_TO_FILE: dict[str, str] = {
    "doc:grounding/delaware-gcl": "delaware-gcl.txt",
    "doc:grounding/first-amendment": "first-amendment.txt",
    "doc:grounding/rfc-2119": "rfc-2119.txt",
    "doc:grounding/fair-use-107": "fair-use-107.txt",
    "doc:grounding/rule-10b-5": "rule-10b-5.txt",
    "doc:grounding/apollo-11": "apollo-11.txt",
    "doc:grounding/gdpr-art-17": "gdpr-art-17.txt",
    "doc:grounding/voyager": "voyager-fact-sheet.txt",
    "doc:grounding/miranda": "miranda-holding.txt",
    "doc:grounding/nist-password": "nist-password-guidance.txt",
}


def _load_corpus() -> dict[str, str]:
    """Load the grounding fixture as a ``{uri: text}`` dict for RAG.query."""
    if not _CORPUS_DIR.is_dir():
        msg = (
            f"grounding-corpus missing at {_CORPUS_DIR}. "
            "Fix: regenerate the fixture or run from the kaos-llm-core root."
        )
        raise FileNotFoundError(msg)
    return {uri: (_CORPUS_DIR / fname).read_text() for uri, fname in _URI_TO_FILE.items()}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser(
        description="Run the grounding calibration harness and emit a dated artifact.",
        default_out_dir=pathlib.Path(__file__).resolve().parent.parent.parent
        / "docs"
        / "benchmarks",
    )
    return parser.parse_args(argv)


async def _run_async(args: argparse.Namespace) -> CalibrationReport:
    corpus = _load_corpus()
    questions = load_questions_jsonl(_CORPUS_DIR / "grounding-questions.jsonl")
    if args.sample > 0:
        import random

        rng = random.Random(args.sample)
        questions = rng.sample(questions, min(args.sample, len(questions)))

    date = dt.date.today().isoformat()
    report = CalibrationReport(
        title="Grounding Calibration",
        date=date,
        git_commit=git_commit(),
        corpus_dir=str(_CORPUS_DIR),
        n_docs=len(corpus),
        n_questions=len(questions),
        strategies=[s.value for s in DEFAULT_LENIENT_STRATEGIES],
        models=[],
    )

    for model in args.models:
        if args.dry_run:
            report.models.append(dry_run_report(model, questions))
            continue
        logger.info("Running %s over %d questions", model, len(questions))
        model_report = await run_model(model, corpus, questions)
        report.models.append(model_report)

    return report


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = asyncio.run(_run_async(args))

    stem = f"grounding-calibration-{report.date}"
    if args.suffix:
        stem = f"{stem}-{args.suffix}"
    json_path = args.out_dir / f"{stem}.json"
    md_path = args.out_dir / f"{stem}.md"

    write_json(report, json_path)
    write_markdown(report, md_path)

    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    for model in report.models:
        if model.skipped:
            print(f"  {model.model}: SKIPPED ({model.reason})")
        else:
            print(
                f"  {model.model}: precision={model.precision_on_answerable:.2%} "
                f"refusal_recall={model.refusal_recall_on_unanswerable:.2%} "
                f"cost=${model.total_cost_usd:.4f} "
                f"errors={model.errors}"
            )

    if args.json:
        print(json.dumps(json.loads(json_path.read_text()), indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
