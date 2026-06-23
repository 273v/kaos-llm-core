"""Calibration runtime: dataclasses + per-(model, question) runner.

Corpus-agnostic — ``run_model`` accepts whatever ``documents`` shape
:class:`kaos_llm_core.programs.rag.RAG.invoke` accepts. Callers
(scripts/grounding_calibration.py, scripts/multiformat_benchmark.py)
supply the corpus and question set; this module handles the loop, the
per-outcome counters, the cost/latency rollups, and the dry-run variant.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from kaos_core.logging import get_logger

from kaos_llm_core.calibration.providers import PROVIDER_ENV, has_api_key_for, provider_of
from kaos_llm_core.calibration.strategies import DEFAULT_LENIENT_STRATEGIES
from kaos_llm_core.programs.rag import RAG, RAGResult
from kaos_llm_core.signatures import Answer, MatchStrategy

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Question:
    """One labelled entry from a calibration golden set."""

    id: str
    answerable: bool
    question: str
    expected_doc_id: str | None
    expected_answer_hint: str | None
    diagnostic_char_span: tuple[int, int] | None
    reason: str | None


# ``QuestionResult`` is fully populated at construction in ``run_question``;
# the defaults below are only reached on the error branch. Kept mutable to
# match the builder pattern used by the aggregate containers below.
@dataclass(slots=True)
class QuestionResult:
    """Per-(model, question) outcome captured during a calibration run."""

    question_id: str
    answerable: bool
    outcome: str  # "answer" | "insufficient_evidence" | "error"
    verified: bool
    confidence: float
    value: str | None
    verification_errors: int
    cost_usd: float
    tokens_total: int
    latency_ms: float
    error_type: str | None = None
    error_message: str | None = None


# Accumulator — ``run_model`` increments counters and appends results as it
# iterates. Kept mutable (not frozen) intentionally.
@dataclass(slots=True)
class ModelReport:
    """Per-model aggregates across a calibration run."""

    model: str
    skipped: bool
    reason: str | None = None
    n_questions: int = 0
    results: list[QuestionResult] = field(default_factory=list)
    # Counts
    n_answerable: int = 0
    n_unanswerable: int = 0
    answerable_verified_answer: int = 0  # TP
    answerable_refused: int = 0
    answerable_unverified_answer: int = 0  # answered but failed verify
    unanswerable_refused: int = 0  # TN
    unanswerable_answered: int = 0  # FP
    # Derived metrics
    precision_on_answerable: float = 0.0
    refusal_recall_on_unanswerable: float = 0.0
    mean_confidence_when_answered: float = 0.0
    total_cost_usd: float = 0.0
    mean_latency_ms: float = 0.0
    errors: int = 0


# Accumulator — the top-level report grows one ModelReport per model.
@dataclass(slots=True)
class CalibrationReport:
    """Top-level artifact — what gets serialized to JSON + markdown."""

    title: str
    date: str
    git_commit: str | None
    corpus_dir: str
    n_docs: int
    n_questions: int
    strategies: list[str]
    models: list[ModelReport]
    # Optional extras specific to certain calibrations (e.g. per-format
    # precision breakdown in the multiformat benchmark, retrieval latency
    # percentiles). Kept opaque so scripts can attach without adding a
    # field per use case.
    extras: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


async def run_question(
    rag: RAG,
    question: Question,
    documents: Any,
) -> QuestionResult:
    """Invoke ``rag`` once, translate the RAGResult into a ``QuestionResult``.

    ``documents`` is forwarded to ``RAG.invoke`` unchanged — can be a
    dict, a list of SearchableDocument, a Corpus Protocol instance, or
    anything else RAG accepts.
    """
    start = time.monotonic()
    try:
        invocation = await rag.invoke(question=question.question, documents=documents)
    except Exception as exc:
        latency_ms = (time.monotonic() - start) * 1000.0
        logger.warning(
            "rag.invoke failed q=%s err=%s: %s",
            question.id,
            type(exc).__name__,
            exc,
        )
        return QuestionResult(
            question_id=question.id,
            answerable=question.answerable,
            outcome="error",
            verified=False,
            confidence=0.0,
            value=None,
            verification_errors=0,
            cost_usd=0.0,
            tokens_total=0,
            latency_ms=latency_ms,
            error_type=type(exc).__name__,
            error_message=str(exc)[:200],
        )
    latency_ms = (time.monotonic() - start) * 1000.0
    result = invocation.output
    assert isinstance(result, RAGResult), f"unexpected result type {type(result)!r}"
    outcome_obj = result.grounded_answer

    if isinstance(outcome_obj, Answer):
        outcome = "answer"
        value = str(outcome_obj.value)[:500]
        verified = result.is_verified
        confidence = float(outcome_obj.confidence)
        verification_errors = len(result.verification_errors)
    else:
        outcome = "insufficient_evidence"
        value = None
        verified = False
        confidence = 0.0
        verification_errors = 0

    return QuestionResult(
        question_id=question.id,
        answerable=question.answerable,
        outcome=outcome,
        verified=verified,
        confidence=confidence,
        value=value,
        verification_errors=verification_errors,
        cost_usd=float(invocation.usage.cost_usd),
        tokens_total=int(invocation.usage.total_tokens),
        latency_ms=latency_ms,
    )


async def run_model(
    model: str,
    documents: Any,
    questions: list[Question],
    *,
    match_strategies: tuple[MatchStrategy, ...] = DEFAULT_LENIENT_STRATEGIES,
    top_k: int = 5,
) -> ModelReport:
    """Loop every question through one model, accumulate aggregates."""
    if not has_api_key_for(model):
        env_names = ", ".join(PROVIDER_ENV.get(provider_of(model), ()))
        return ModelReport(
            model=model,
            skipped=True,
            reason=f"No API key in environment; set one of: {env_names}",
            n_questions=len(questions),
        )

    report = ModelReport(model=model, skipped=False, n_questions=len(questions))
    report.n_answerable = sum(1 for q in questions if q.answerable)
    report.n_unanswerable = len(questions) - report.n_answerable

    rag = RAG(model=model, match_strategies=match_strategies, top_k=top_k)

    answered_confidences: list[float] = []
    latencies: list[float] = []
    for question in questions:
        result = await run_question(rag, question, documents)
        report.results.append(result)
        latencies.append(result.latency_ms)
        report.total_cost_usd += result.cost_usd

        if result.outcome == "error":
            report.errors += 1
            continue

        if result.outcome == "answer":
            answered_confidences.append(result.confidence)

        if question.answerable:
            if result.outcome == "answer" and result.verified:
                report.answerable_verified_answer += 1
            elif result.outcome == "insufficient_evidence":
                report.answerable_refused += 1
            elif result.outcome == "answer" and not result.verified:
                report.answerable_unverified_answer += 1
        else:
            if result.outcome == "insufficient_evidence":
                report.unanswerable_refused += 1
            elif result.outcome == "answer":
                report.unanswerable_answered += 1

        logger.debug(
            "[%s] %s (answerable=%s) -> %s verified=%s conf=%.2f cost=$%.6f %dms",
            model,
            question.id,
            question.answerable,
            result.outcome,
            result.verified,
            result.confidence,
            result.cost_usd,
            int(result.latency_ms),
        )

    if report.n_answerable > 0:
        report.precision_on_answerable = report.answerable_verified_answer / report.n_answerable
    if report.n_unanswerable > 0:
        report.refusal_recall_on_unanswerable = report.unanswerable_refused / report.n_unanswerable
    if answered_confidences:
        report.mean_confidence_when_answered = statistics.mean(answered_confidences)
    if latencies:
        report.mean_latency_ms = statistics.mean(latencies)

    return report


def dry_run_report(model: str, questions: list[Question]) -> ModelReport:
    """Synthesise a ModelReport without hitting any API.

    For wiring-smoke tests: proves the CLI + artifact pipeline works
    without burning tokens. Populates "perfect" outcomes (every
    answerable answered + verified; every unanswerable refused) so the
    emitted markdown is structurally identical to a real run.
    """
    report = ModelReport(
        model=model,
        skipped=False,
        n_questions=len(questions),
        n_answerable=sum(1 for q in questions if q.answerable),
    )
    report.n_unanswerable = len(questions) - report.n_answerable
    for question in questions:
        outcome = "answer" if question.answerable else "insufficient_evidence"
        verified = question.answerable
        confidence = 0.9 if question.answerable else 0.0
        result = QuestionResult(
            question_id=question.id,
            answerable=question.answerable,
            outcome=outcome,
            verified=verified,
            confidence=confidence,
            value=question.expected_answer_hint if question.answerable else None,
            verification_errors=0,
            cost_usd=0.0,
            tokens_total=0,
            latency_ms=0.0,
        )
        report.results.append(result)
        if question.answerable:
            report.answerable_verified_answer += 1
        else:
            report.unanswerable_refused += 1
    report.precision_on_answerable = 1.0
    report.refusal_recall_on_unanswerable = 1.0
    report.mean_confidence_when_answered = 0.9
    return report


__all__ = [
    "CalibrationReport",
    "ModelReport",
    "Question",
    "QuestionResult",
    "dry_run_report",
    "run_model",
    "run_question",
]
