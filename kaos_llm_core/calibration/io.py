"""Calibration I/O: JSONL loader, JSON + markdown artifact writers, git helper, CLI.

Separated from :mod:`kaos_llm_core.calibration.core` so scripts that only
need the runtime loop do not drag in argparse / subprocess.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import subprocess
from typing import Any

from kaos_llm_core.calibration.core import CalibrationReport, Question
from kaos_llm_core.calibration.providers import DEFAULT_MODELS


def load_questions_jsonl(path: pathlib.Path) -> list[Question]:
    """Parse a ``grounding-questions.jsonl``-style file into Questions.

    The schema matches WS-1's fixture — one JSON object per line with
    keys ``id``, ``answerable``, ``question``, and optionally
    ``expected_doc_id``, ``expected_answer_hint``,
    ``diagnostic_char_span``, ``reason``.
    """
    if not path.is_file():
        msg = (
            f"questions JSONL missing at {path}. "
            "Fix: create the file or pass a different path. "
            "Alternative: copy the schema from "
            "kaos-llm-core/tests/fixtures/grounding-corpus/"
            "grounding-questions.jsonl."
        )
        raise FileNotFoundError(msg)
    out: list[Question] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        span = raw.get("diagnostic_char_span")
        out.append(
            Question(
                id=raw["id"],
                answerable=bool(raw["answerable"]),
                question=raw["question"],
                expected_doc_id=raw.get("expected_doc_id"),
                expected_answer_hint=raw.get("expected_answer_hint"),
                diagnostic_char_span=tuple(span) if span is not None else None,
                reason=raw.get("reason"),
            )
        )
    return out


def git_commit() -> str | None:
    """Return the short commit hash of this repo, or None if unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=pathlib.Path(__file__).resolve().parent.parent.parent,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except Exception:
        return None


def write_json(report: CalibrationReport, path: pathlib.Path) -> None:
    """Serialize a :class:`CalibrationReport` to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "title": report.title,
        "date": report.date,
        "git_commit": report.git_commit,
        "corpus_dir": report.corpus_dir,
        "n_docs": report.n_docs,
        "n_questions": report.n_questions,
        "strategies": report.strategies,
        "models": [dataclasses.asdict(m) for m in report.models],
        "extras": report.extras,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=False))


def write_markdown(report: CalibrationReport, path: pathlib.Path) -> None:
    """Emit the human-readable digest alongside the JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(f"# {report.title} — {report.date}")
    lines.append("")
    lines.append(
        f"**Corpus:** `{report.corpus_dir}` ({report.n_docs} docs, {report.n_questions} questions)"
    )
    if report.git_commit:
        lines.append(f"**Git commit:** `{report.git_commit}`")
    lines.append(f"**Strategies:** {', '.join(report.strategies)}")
    if report.extras:
        for key, value in report.extras.items():
            lines.append(f"**{key}:** {value}")
    lines.append("")
    lines.append(
        "| Model | Precision (answerable) | Refusal recall (unanswerable) | "
        "Mean conf | Errors | Total cost | Mean latency |"
    )
    lines.append(
        "|-------|-----------------------:|------------------------------:|"
        "----------:|-------:|-----------:|-------------:|"
    )
    for m in report.models:
        if m.skipped:
            lines.append(f"| `{m.model}` | - | - | - | - | - | SKIP: {m.reason or ''} |")
            continue
        lines.append(
            f"| `{m.model}` | "
            f"{m.precision_on_answerable:.2%} "
            f"({m.answerable_verified_answer}/{m.n_answerable}) | "
            f"{m.refusal_recall_on_unanswerable:.2%} "
            f"({m.unanswerable_refused}/{m.n_unanswerable}) | "
            f"{m.mean_confidence_when_answered:.2f} | "
            f"{m.errors} | "
            f"${m.total_cost_usd:.4f} | "
            f"{m.mean_latency_ms:.0f}ms |"
        )
    lines.append("")
    for m in report.models:
        if m.skipped or not m.results:
            continue
        lines.append(f"## `{m.model}` - per-question")
        lines.append("")
        lines.append("| Q | answerable | outcome | verified | conf | cost | latency |")
        lines.append("|---|-----------:|---------|---------:|-----:|-----:|--------:|")
        for r in m.results:
            error_suffix = f" ({r.error_type})" if r.outcome == "error" and r.error_type else ""
            lines.append(
                f"| {r.question_id} | {r.answerable} | "
                f"{r.outcome}{error_suffix} | "
                f"{r.verified} | {r.confidence:.2f} | "
                f"${r.cost_usd:.6f} | {int(r.latency_ms)}ms |"
            )
        lines.append("")
    path.write_text("\n".join(lines))


def build_parser(description: str, *, default_out_dir: pathlib.Path) -> argparse.ArgumentParser:
    """Build a CLI ArgumentParser with the shared calibration flags.

    Scripts add their own flags (e.g. ``--fixture-dir``) after calling this.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help=(
            "Provider-qualified model ids to evaluate (space-separated). "
            f"Default: {' '.join(DEFAULT_MODELS)}"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=default_out_dir,
        help="Directory for the dated JSON + markdown artifacts.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Optional suffix appended after the date in the artifact filename.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip all API calls; synthesise a plausible artifact for wiring smoke tests.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="If >0, randomly sample N questions instead of running all (cheap smoke).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Echo the final JSON artifact to stdout in addition to writing files.",
    )
    return parser


__all__ = [
    "build_parser",
    "git_commit",
    "load_questions_jsonl",
    "write_json",
    "write_markdown",
]
