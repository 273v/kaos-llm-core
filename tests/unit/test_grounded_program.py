"""Unit tests for the Grounded program wrapper."""

from __future__ import annotations

from unittest.mock import patch

from pydantic import BaseModel

from kaos_llm_core.programs.grounded import Grounded, GroundedResult
from kaos_llm_core.signatures.grounding import (
    Answer,
    Cited,
    Claim,
    InsufficientEvidence,
    MatchStrategy,
    Span,
    SpanError,
)

SOURCE_TEXT = "The filing fee is $89 and the certificate must include the corporation's name."
CORPUS = {"doc:test": SOURCE_TEXT}


def _good_span(quote: str = "filing fee is $89") -> Span:
    return Span.from_source("doc:test", SOURCE_TEXT, (4, 23))


def _bad_span() -> Span:
    return Span(source_uri="doc:test", char_span=(0, 10), quote="WRONG TEXT HERE")


# ---------------------------------------------------------------------------
# Mock producer
# ---------------------------------------------------------------------------


class _MockCall:
    """Minimal mock of a Call that returns configurable outputs."""

    def __init__(self, outputs: list):
        self._outputs = list(outputs)
        self._call_count = 0
        self.instructions = ""

    async def __call__(self, **kwargs):
        idx = min(self._call_count, len(self._outputs) - 1)
        self._call_count += 1
        return self._outputs[idx]

    # Minimal attributes for clone_call compatibility
    _kwargs: dict = {}  # noqa: RUF012
    examples: list = []  # noqa: RUF012
    model = "test"
    _codec = None
    _client = None


# ---------------------------------------------------------------------------
# GroundedResult
# ---------------------------------------------------------------------------


class TestGroundedResult:
    def test_verified_result(self):
        r = GroundedResult(output="test", is_verified=True, attempts=1)
        assert r.is_verified
        assert r.attempts == 1

    def test_unverified_result(self):
        r = GroundedResult(
            output="test",
            verification_errors=(
                SpanError(span_index=0, span=_bad_span(), reason="quote_mismatch"),
            ),
            is_verified=False,
            attempts=3,
        )
        assert not r.is_verified
        assert r.attempts == 3


# ---------------------------------------------------------------------------
# Grounded with Cited[T] output
# ---------------------------------------------------------------------------


class TestGroundedCited:
    async def test_verified_cited_passes(self):
        """Cited[T] with correct span should pass verification."""

        class Output(BaseModel):
            fee: Cited[str]

        good_output = Output(fee=Cited(value="$89", spans=[_good_span()]))
        producer = _MockCall([good_output])

        grounded = Grounded(producer, corpus=CORPUS)

        with patch("kaos_llm_core.programs.grounded.clone_call", return_value=producer):
            result = await grounded()

        assert result.is_verified
        assert result.attempts == 1
        assert result.output.fee.value == "$89"

    async def test_bad_cited_triggers_retry(self):
        """Cited[T] with wrong span should retry and accept the fix."""

        class Output(BaseModel):
            fee: Cited[str]

        bad_output = Output(fee=Cited(value="$89", spans=[_bad_span()]))
        good_output = Output(fee=Cited(value="$89", spans=[_good_span()]))
        producer = _MockCall([bad_output, good_output])

        grounded = Grounded(producer, corpus=CORPUS, max_retries=2)

        with patch(
            "kaos_llm_core.programs.grounded.clone_call",
            return_value=_MockCall([good_output]),
        ):
            result = await grounded()

        assert result.is_verified
        assert result.attempts == 2

    async def test_all_retries_exhausted(self):
        """When all retries fail, return best attempt."""

        class Output(BaseModel):
            fee: Cited[str]

        bad_output = Output(fee=Cited(value="$89", spans=[_bad_span()]))
        producer = _MockCall([bad_output, bad_output, bad_output])

        grounded = Grounded(producer, corpus=CORPUS, max_retries=2)

        with patch(
            "kaos_llm_core.programs.grounded.clone_call",
            return_value=_MockCall([bad_output]),
        ):
            result = await grounded()

        assert not result.is_verified
        assert result.attempts == 3
        assert len(result.verification_errors) > 0


# ---------------------------------------------------------------------------
# Grounded with GroundedAnswer output
# ---------------------------------------------------------------------------


class TestGroundedAnswer:
    async def test_verified_answer_passes(self):
        """Answer[T] with verified claims should pass."""
        answer = Answer(
            value="The fee is $89",
            claims=[
                Claim(
                    statement="Filing fee is $89",
                    supporting_spans=[_good_span()],
                )
            ],
            confidence=0.95,
        )
        producer = _MockCall([answer])
        grounded = Grounded(producer, corpus=CORPUS)

        with patch("kaos_llm_core.programs.grounded.clone_call", return_value=producer):
            result = await grounded()

        assert result.is_verified
        assert result.output.value == "The fee is $89"

    async def test_insufficient_evidence_passes(self):
        """InsufficientEvidence is a valid output — no verification needed."""
        refusal = InsufficientEvidence(reason="Not in corpus")
        producer = _MockCall([refusal])
        grounded = Grounded(producer, corpus=CORPUS)

        with patch("kaos_llm_core.programs.grounded.clone_call", return_value=producer):
            result = await grounded()

        assert result.is_verified
        assert isinstance(result.output, InsufficientEvidence)


# ---------------------------------------------------------------------------
# Feedback building
# ---------------------------------------------------------------------------


class TestBuildFeedback:
    def test_feedback_format(self):
        from kaos_llm_core.signatures.grounding import SpanError

        errors = [
            SpanError(span_index=0, span=_bad_span(), reason="quote_mismatch"),
        ]
        feedback = Grounded._build_feedback(errors)
        assert "citation verification errors" in feedback
        assert "quote_mismatch" in feedback
        assert "WRONG TEXT" in feedback

    def test_feedback_default_inlines_every_error(self):
        """Default (no cap) MUST inline every error so the retry-prompt
        model sees every citation it's asked to fix."""
        from kaos_llm_core.signatures.grounding import SpanError

        errors = [
            SpanError(span_index=i, span=_bad_span(), reason="quote_mismatch") for i in range(15)
        ]
        feedback = Grounded._build_feedback(errors)
        # All 15 spans should appear; no "and N more" summary line.
        for i in range(15):
            assert f"Span {i}:" in feedback, f"missing span {i} in feedback"
        assert "more errors" not in feedback

    def test_feedback_opt_in_max_errors_caps_with_summary(self):
        """When ``max_errors`` is set explicitly, the cap fires with a
        summary line counting the dropped errors."""
        from kaos_llm_core.signatures.grounding import SpanError

        errors = [
            SpanError(span_index=i, span=_bad_span(), reason="quote_mismatch") for i in range(15)
        ]
        feedback = Grounded._build_feedback(errors, max_errors=10)
        assert "Span 9:" in feedback
        assert "Span 10:" not in feedback
        assert "5 more errors" in feedback

    def test_feedback_default_does_not_truncate_quotes(self):
        """Default (no per_quote cap) MUST inline the full quote text."""
        from kaos_llm_core.signatures.grounding import Span, SpanError

        long_quote = "X" * 500
        span = Span(source_uri="doc", quote=long_quote, char_span=(0, 500))
        errors = [SpanError(span_index=0, span=span, reason="quote_mismatch")]
        feedback = Grounded._build_feedback(errors)
        assert long_quote in feedback
        assert "truncated" not in feedback

    def test_feedback_opt_in_per_quote_budget_truncates_with_marker(self):
        """When ``per_quote_char_budget`` is set, quotes are truncated
        with an explicit marker telling the model how many chars were
        dropped."""
        from kaos_llm_core.signatures.grounding import Span, SpanError

        long_quote = "X" * 500
        span = Span(source_uri="doc", quote=long_quote, char_span=(0, 500))
        errors = [SpanError(span_index=0, span=span, reason="quote_mismatch")]
        feedback = Grounded._build_feedback(errors, per_quote_char_budget=80)
        assert "truncated 420 more chars" in feedback


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestGroundedConfig:
    async def test_custom_strategies(self):
        """Custom match strategies should be used for verification."""

        class Output(BaseModel):
            fee: Cited[str]

        # Bad position but good quote — SUBSTRING should save it
        span = Span(source_uri="doc:test", char_span=(999, 999), quote="filing fee is $89")
        output = Output(fee=Cited(value="$89", spans=[span]))
        producer = _MockCall([output])

        grounded = Grounded(
            producer,
            corpus=CORPUS,
            match_strategies=(MatchStrategy.STRICT, MatchStrategy.SUBSTRING),
        )

        with patch("kaos_llm_core.programs.grounded.clone_call", return_value=producer):
            result = await grounded()

        assert result.is_verified

    async def test_zero_retries(self):
        """max_retries=0 should not retry."""

        class Output(BaseModel):
            fee: Cited[str]

        bad_output = Output(fee=Cited(value="$89", spans=[_bad_span()]))
        producer = _MockCall([bad_output])

        grounded = Grounded(producer, corpus=CORPUS, max_retries=0)

        with patch("kaos_llm_core.programs.grounded.clone_call", return_value=producer):
            result = await grounded()

        assert not result.is_verified
        assert result.attempts == 1
