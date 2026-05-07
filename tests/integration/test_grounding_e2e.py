"""End-to-end grounding harness (WS-1).

Drives the full RAG pipeline over the 10-document ``grounding-corpus``
fixture and the 20-question golden set, one run per (provider, question).

Acceptance thresholds are calibrated from
``docs/benchmarks/grounding-calibration-2026-04-14.json`` (the WS-1.6
artifact). Per-provider floors are deliberately set ≥ 10% below observed
values to tolerate LLM sample variance without masking regressions:

======================  measured 2026-04-14  acceptance floor
anthropic/haiku-4-5     prec 100%, refusal 90%     prec ≥0.85, refusal ≥0.70
openai/gpt-5.4-nano     prec 100%, refusal 80%     prec ≥0.85, refusal ≥0.60
google/gemini-2.5-flash prec 100%, refusal 90%     prec ≥0.85, refusal ≥0.70

When the calibration script (``scripts/grounding_calibration.py``) runs
against a newer model and shows a sustained uplift, bump the floors here
and commit the new artifact in the same PR — the floors are a living
contract with the measured landscape, not an arbitrary target.

Run with ``pytest -m integration`` and the appropriate provider API key
set. Tests auto-skip when a provider key is missing.
"""

from __future__ import annotations

import logging

import pytest

from kaos_llm_core.programs.rag import RAG, RAGResult
from kaos_llm_core.signatures import Answer, InsufficientEvidence, MatchStrategy

from .conftest import (
    GroundingQuestion,
    requires_anthropic,
    requires_google,
    requires_openai,
)

logger = logging.getLogger("kaos_llm_core.tests.grounding_e2e")


# Lenient cascade — LLMs drift on whitespace, case, punctuation. Fuzzy
# strategies are intentionally excluded from the default cascade; see
# ``docs/design/grounding-actual-state.md §4`` on the Unicode-normalization
# asymmetry. The fixture corpus is ASCII-only so NORMALIZED_TOKEN is safe.
_LENIENT: tuple[MatchStrategy, ...] = (
    MatchStrategy.STRICT,
    MatchStrategy.SUBSTRING,
    MatchStrategy.CASE_INSENSITIVE,
    MatchStrategy.NORMALIZED_TOKEN,
)

# Single retry budget per sample — LLM variance, not a retry loop.
_MAX_ATTEMPTS = 2


async def _run_one(
    rag: RAG,
    question: str,
    corpus: dict[str, str],
) -> RAGResult | None:
    """Run one RAG query with a small retry budget for provider flakiness."""
    last_exc: Exception | None = None
    for _ in range(_MAX_ATTEMPTS):
        try:
            return await rag.query(question=question, documents=corpus)
        except Exception as exc:
            last_exc = exc
            logger.warning("rag.query raised %s: %s", type(exc).__name__, exc)
    if last_exc is not None:
        logger.error("rag.query failed all %d attempts: %s", _MAX_ATTEMPTS, last_exc)
    return None


async def _run_provider(
    model: str,
    corpus: dict[str, str],
    questions: list[GroundingQuestion],
) -> tuple[float, float, int]:
    """Run the full harness for one model.

    Returns ``(precision_on_answerable, refusal_recall_on_unanswerable, errors)``.
    """
    rag = RAG(model=model, match_strategies=_LENIENT, top_k=5)

    n_answerable = sum(1 for q in questions if q.answerable)
    n_unanswerable = len(questions) - n_answerable
    answerable_verified = 0
    unanswerable_refused = 0
    errors = 0

    for q in questions:
        result = await _run_one(rag, q.question, corpus)
        if result is None:
            errors += 1
            continue
        assert isinstance(result, RAGResult), (
            f"[{model}] {q.id}: expected RAGResult, got {type(result)!r}"
        )
        outcome = result.grounded_answer
        assert isinstance(outcome, (Answer, InsufficientEvidence)), (
            f"[{model}] {q.id}: grounded_answer is neither Answer nor "
            f"InsufficientEvidence: {type(outcome)!r}"
        )

        if q.answerable:
            if isinstance(outcome, Answer) and result.is_verified:
                answerable_verified += 1
        else:
            if isinstance(outcome, InsufficientEvidence):
                unanswerable_refused += 1

        logger.info(
            "[%s] %s (answerable=%s) -> %s (verified=%s, conf=%.2f)",
            model,
            q.id,
            q.answerable,
            type(outcome).__name__,
            result.is_verified,
            result.confidence,
        )

    precision = answerable_verified / n_answerable if n_answerable else 0.0
    refusal_recall = unanswerable_refused / n_unanswerable if n_unanswerable else 0.0
    return precision, refusal_recall, errors


# ---------------------------------------------------------------------------
# Per-provider classes — acceptance floors calibrated from the
# 2026-04-14 benchmark. See module docstring.
# ---------------------------------------------------------------------------


@requires_anthropic
@pytest.mark.integration
class TestGroundingE2EAnthropic:
    MODEL = "anthropic:claude-haiku-4-5"
    MIN_PRECISION = 0.85
    MIN_REFUSAL_RECALL = 0.70

    async def test_grounding_harness(
        self,
        grounding_corpus: dict[str, str],
        grounding_questions: list[GroundingQuestion],
    ) -> None:
        precision, refusal_recall, errors = await _run_provider(
            self.MODEL, grounding_corpus, grounding_questions
        )
        logger.info(
            "[%s] precision=%.2f refusal_recall=%.2f errors=%d",
            self.MODEL,
            precision,
            refusal_recall,
            errors,
        )
        assert errors == 0, f"[{self.MODEL}] {errors} RAG errors"
        assert precision >= self.MIN_PRECISION, (
            f"[{self.MODEL}] precision {precision:.2%} below floor "
            f"{self.MIN_PRECISION:.2%} — either the model regressed or the "
            "fixture needs an audit; re-run scripts/grounding_calibration.py"
        )
        assert refusal_recall >= self.MIN_REFUSAL_RECALL, (
            f"[{self.MODEL}] refusal_recall {refusal_recall:.2%} below floor "
            f"{self.MIN_REFUSAL_RECALL:.2%}"
        )


@requires_openai
@pytest.mark.integration
class TestGroundingE2EOpenAI:
    MODEL = "openai:gpt-5.4-nano"
    MIN_PRECISION = 0.85
    MIN_REFUSAL_RECALL = 0.60

    async def test_grounding_harness(
        self,
        grounding_corpus: dict[str, str],
        grounding_questions: list[GroundingQuestion],
    ) -> None:
        precision, refusal_recall, errors = await _run_provider(
            self.MODEL, grounding_corpus, grounding_questions
        )
        logger.info(
            "[%s] precision=%.2f refusal_recall=%.2f errors=%d",
            self.MODEL,
            precision,
            refusal_recall,
            errors,
        )
        assert errors == 0
        assert precision >= self.MIN_PRECISION, (
            f"[{self.MODEL}] precision {precision:.2%} below floor {self.MIN_PRECISION:.2%}"
        )
        assert refusal_recall >= self.MIN_REFUSAL_RECALL, (
            f"[{self.MODEL}] refusal_recall {refusal_recall:.2%} below floor "
            f"{self.MIN_REFUSAL_RECALL:.2%}"
        )


@requires_google
@pytest.mark.integration
class TestGroundingE2EGoogle:
    MODEL = "google:gemini-2.5-flash"
    MIN_PRECISION = 0.85
    MIN_REFUSAL_RECALL = 0.70

    async def test_grounding_harness(
        self,
        grounding_corpus: dict[str, str],
        grounding_questions: list[GroundingQuestion],
    ) -> None:
        precision, refusal_recall, errors = await _run_provider(
            self.MODEL, grounding_corpus, grounding_questions
        )
        logger.info(
            "[%s] precision=%.2f refusal_recall=%.2f errors=%d",
            self.MODEL,
            precision,
            refusal_recall,
            errors,
        )
        assert errors == 0
        assert precision >= self.MIN_PRECISION, (
            f"[{self.MODEL}] precision {precision:.2%} below floor {self.MIN_PRECISION:.2%}"
        )
        assert refusal_recall >= self.MIN_REFUSAL_RECALL, (
            f"[{self.MODEL}] refusal_recall {refusal_recall:.2%} below floor "
            f"{self.MIN_REFUSAL_RECALL:.2%}"
        )
