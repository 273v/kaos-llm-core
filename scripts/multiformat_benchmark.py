"""Multi-format corpus-QA calibration (WS-3.7 phase 2).

Composes on top of :mod:`kaos_llm_core.calibration` (extracted from
``scripts/grounding_calibration.py`` in WS-3.7.2a) to run the same
per-(model, question) loop over a realistic mixed-format corpus.

Pipeline:

1. Glob ``tests/fixtures/multiformat-corpus/*.{pdf,docx,html,md,txt}``.
2. Dispatch each file by extension to kaos-pdf / kaos-office / kaos-web /
   kaos-content (markdown + plain text) — each returns a ``ContentDocument``
   with populated ``metadata.source``.
3. Build ``kaos_ml_core.Corpus.from_documents(docs, level="paragraph")``.
4. For each (model, question): ``RAG.query(question, documents=corpus)``
   via the shared ``run_model`` helper; aggregate per-model metrics.
5. Also measure retrieval-layer latency — 100 warm BM25 queries, record
   p50/p95/p99 in ``report.extras`` alongside the LLM metrics.
6. Emit ``docs/benchmarks/corpus-qa-calibration-{date}.{json,md}``.

The ``expected_doc_id`` field in the question set uses stable short
slugs (``file:multiformat/<filename>``); the benchmark resolves them
against the real filesystem URIs before aggregating.

Usage:

  uv run python scripts/multiformat_benchmark.py --help
  uv run python scripts/multiformat_benchmark.py --dry-run
  uv run python scripts/multiformat_benchmark.py --models anthropic:claude-haiku-4-5
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import pathlib
import statistics
import sys
import time
from typing import Any

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

_FIXTURE_DIR = (
    pathlib.Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "multiformat-corpus"
)
_SUPPORTED_EXTS = {".pdf", ".docx", ".html", ".htm", ".md", ".txt"}
_EXCLUDED_STEMS = {"README", "fixtures-provenance"}


# ---------------------------------------------------------------------------
# Corpus construction
# ---------------------------------------------------------------------------


def _fixture_paths() -> list[pathlib.Path]:
    if not _FIXTURE_DIR.is_dir():
        msg = (
            f"multiformat-corpus fixture missing at {_FIXTURE_DIR}. "
            "Fix: ensure the fixture directory exists under kaos-llm-core/tests/fixtures/. "
            "Alternative: run from the kaos-llm-core root."
        )
        raise FileNotFoundError(msg)
    return sorted(
        p
        for p in _FIXTURE_DIR.iterdir()
        if p.suffix.lower() in _SUPPORTED_EXTS and p.stem not in _EXCLUDED_STEMS
    )


def _dispatch_parse(path: pathlib.Path):
    """Parse a fixture file by extension. Mirrors the conftest dispatch."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        from kaos_pdf import extract_pdf

        return extract_pdf(path)
    if ext == ".docx":
        from kaos_office import parse_docx

        return parse_docx(path)
    if ext in (".html", ".htm"):
        from kaos_web import html_to_document

        return html_to_document(path.read_text(), url=path.as_uri())
    if ext == ".md":
        from kaos_content.model.attr import SourceRef
        from kaos_content.parsers import parse_markdown

        return parse_markdown(
            path.read_text(),
            source=SourceRef(uri=path.as_uri(), mime_type="text/markdown"),
        )
    if ext == ".txt":
        from kaos_content.model.attr import SourceRef
        from kaos_content.parsers import parse_plain_text

        return parse_plain_text(
            path.read_text(),
            source=SourceRef(uri=path.as_uri(), mime_type="text/plain"),
        )
    msg = f"No dispatch handler for extension {ext!r}"
    raise ValueError(msg)


def _build_slug_map(paths: list[pathlib.Path]) -> dict[str, str]:
    """Map filesystem file://... URIs to stable ``file:multiformat/<name>`` slugs.

    The question set references the stable slugs so that a CI run on a
    different checkout (or a symlinked fixture) still matches. The map is
    used to fix up ``expected_doc_id`` before aggregating per-format
    precision.
    """
    return {p.as_uri(): f"file:multiformat/{p.name}" for p in paths}


# ---------------------------------------------------------------------------
# Retrieval latency bench
# ---------------------------------------------------------------------------


async def _retrieval_latency_bench(
    corpus: Any,
    queries: list[str],
) -> dict[str, float]:
    """Hit the BM25 retriever ``len(queries)`` times, return p50/p95/p99 ms.

    Warm-up is the first query; measurements exclude it. Returns empty
    if ``Corpus.retriever`` is not available (e.g. dry-run path).
    """
    if not hasattr(corpus, "retriever"):
        return {}
    retriever = corpus.retriever("bm25")
    # Warm-up
    await retriever.retrieve(queries[0], top_k=10)
    latencies: list[float] = []
    for query in queries[1:]:
        start = time.monotonic()
        await retriever.retrieve(query, top_k=10)
        latencies.append((time.monotonic() - start) * 1000.0)
    if not latencies:
        return {}
    return {
        "n_queries": len(latencies),
        "p50_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(statistics.quantiles(latencies, n=20)[18], 2)
        if len(latencies) >= 20
        else round(max(latencies), 2),
        "p99_ms": round(statistics.quantiles(latencies, n=100)[98], 2)
        if len(latencies) >= 100
        else round(max(latencies), 2),
        "mean_ms": round(statistics.mean(latencies), 2),
    }


# ---------------------------------------------------------------------------
# Per-format precision rollup
# ---------------------------------------------------------------------------


def _per_format_precision(
    report: CalibrationReport,
    slug_by_qid: dict[str, str],
) -> dict[str, dict[str, dict[str, int]]]:
    """Compute precision-on-answerable grouped by format (PDF/DOCX/HTML/MD/TXT)."""
    result: dict[str, dict[str, dict[str, int]]] = {}
    for model_report in report.models:
        if model_report.skipped:
            continue
        per_fmt: dict[str, dict[str, int]] = {}
        for r in model_report.results:
            slug = slug_by_qid.get(r.question_id)
            if not slug or not r.answerable:
                continue
            # slug: "file:multiformat/foo.pdf" -> fmt "pdf"
            fmt = slug.rsplit(".", 1)[-1] if "." in slug else "other"
            bucket = per_fmt.setdefault(fmt, {"answerable": 0, "verified": 0})
            bucket["answerable"] += 1
            if r.outcome == "answer" and r.verified:
                bucket["verified"] += 1
        result[model_report.model] = per_fmt
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser(
        description="WS-3.7 multi-format corpus-QA calibration harness.",
        default_out_dir=pathlib.Path(__file__).resolve().parent.parent.parent
        / "docs"
        / "benchmarks",
    )
    return parser.parse_args(argv)


async def _run_async(args: argparse.Namespace) -> CalibrationReport:
    paths = _fixture_paths()
    # slug_map is retained in extras so a future consumer (e.g. the phase-3
    # test harness) can resolve file:// URIs back to the stable slug IDs
    # that the question JSONL references.
    slug_map = _build_slug_map(paths)
    questions_path = _FIXTURE_DIR / "multiformat-questions.jsonl"
    questions = load_questions_jsonl(questions_path)

    if args.sample > 0:
        import random

        rng = random.Random(args.sample)
        questions = rng.sample(questions, min(args.sample, len(questions)))

    date = dt.date.today().isoformat()

    # Build the Corpus once — shared across every model.
    extras: dict[str, Any] = {}
    if not args.dry_run:
        from kaos_ml_core.corpus import Corpus

        start = time.monotonic()
        docs = [_dispatch_parse(p) for p in paths]
        corpus = Corpus.from_documents(docs, level="paragraph")
        build_ms = (time.monotonic() - start) * 1000.0
        extras["n_passages"] = corpus.size
        extras["index_build_ms"] = round(build_ms, 1)
        logger.info(
            "Built corpus: %d docs -> %d passages in %.0fms", len(paths), corpus.size, build_ms
        )

        # Retrieval-layer latency benchmark (reuses the LLM questions as queries).
        latency = await _retrieval_latency_bench(corpus, [q.question for q in questions])
        if latency:
            extras["retrieval_latency"] = latency
    else:
        corpus = None  # type: ignore[assignment]

    report = CalibrationReport(
        title="Multiformat Corpus QA Calibration",
        date=date,
        git_commit=git_commit(),
        corpus_dir=str(_FIXTURE_DIR),
        n_docs=len(paths),
        n_questions=len(questions),
        strategies=[s.value for s in DEFAULT_LENIENT_STRATEGIES],
        models=[],
        extras=extras,
    )

    for model in args.models:
        if args.dry_run:
            report.models.append(dry_run_report(model, questions))
            continue
        logger.info("Running %s over %d questions", model, len(questions))
        # Bump default top_k from 5 → 12 — the multiformat corpus produces
        # ~100 passages/doc (vs ~5 for the synthetic WS-1 grounding corpus),
        # so BM25 top-5 gives <1% recall against the 982-passage fixture.
        # First calibration run with top_k=5 hit 71% precision; failures
        # were retrieval-not-model because the same two questions failed
        # on every provider.
        model_report = await run_model(model, corpus, questions, top_k=12)
        report.models.append(model_report)

    # Per-format precision rollup (keyed by question.id -> slug).
    slug_by_qid = {q.id: q.expected_doc_id for q in questions if q.expected_doc_id}
    report.extras["per_format_precision"] = _per_format_precision(report, slug_by_qid)
    report.extras["slug_map"] = slug_map

    return report


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = asyncio.run(_run_async(args))

    stem = f"corpus-qa-calibration-{report.date}"
    if args.suffix:
        stem = f"{stem}-{args.suffix}"
    json_path = args.out_dir / f"{stem}.json"
    md_path = args.out_dir / f"{stem}.md"

    write_json(report, json_path)
    write_markdown(report, md_path)

    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    if "index_build_ms" in report.extras:
        print(
            f"  corpus: {report.n_docs} docs, {report.extras['n_passages']} passages, "
            f"build {report.extras['index_build_ms']:.0f}ms"
        )
    if "retrieval_latency" in report.extras:
        lat = report.extras["retrieval_latency"]
        print(f"  retrieval latency: p50={lat.get('p50_ms')}ms p95={lat.get('p95_ms')}ms")
    for model in report.models:
        if model.skipped:
            print(f"  {model.model}: SKIPPED ({model.reason})")
        else:
            print(
                f"  {model.model}: precision={model.precision_on_answerable:.2%} "
                f"refusal_recall={model.refusal_recall_on_unanswerable:.2%} "
                f"cost=${model.total_cost_usd:.4f} errors={model.errors}"
            )

    if args.json:
        print(json.dumps(json.loads(json_path.read_text()), indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
