"""Grounded — a Program wrapper that verifies citations after generation.

Composes any Call that produces ``Cited[T]`` or ``GroundedAnswer[T]``
output with post-decode citation verification and automatic retry
on verification failure.

Pipeline:
    1. Run the inner Call (producer)
    2. Extract all ``Cited[T]`` values from the output
    3. Verify every span against the source corpus
    4. On failure: clone the Call, inject feedback about failed spans, retry
    5. On exhausted retries: return the best attempt (fewest errors)

Usage::

    from kaos_llm_core import Call, Signature, InputField, OutputField
    from kaos_llm_core.signatures import Cited
    from kaos_llm_core.programs.grounded import Grounded

    class DocumentExtraction(Signature):
        '''Extract key fields from the document.'''
        document: str = InputField()
        effective_date: Cited[str] = OutputField()
        author: Cited[str] = OutputField()

    producer = Call(DocumentExtraction, model="anthropic:claude-haiku-4-5")
    grounded = Grounded(producer, corpus={"doc:report": document_text})

    result = await grounded(document=tagged_document)
    # result.effective_date is a Cited[str] with verified spans
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from kaos_core.logging import get_logger

from kaos_llm_core.programs.base import Program
from kaos_llm_core.programs.cloning import clone_call
from kaos_llm_core.signatures.grounding import (
    DEFAULT_MATCH_STRATEGIES,
    Answer,
    InsufficientEvidence,
    MatchStrategy,
    SpanError,
)
from kaos_llm_core.signatures.span_tagging import validate_cited_output

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class GroundedResult:
    """Result of a Grounded program execution.

    Attributes:
        output: The decoded output from the producer Call.
        verification_errors: Span errors from the best attempt.
        is_verified: True if zero verification errors.
        attempts: Number of generation attempts (1 = no retries).
    """

    output: Any
    verification_errors: tuple[SpanError, ...] = ()
    is_verified: bool = True
    attempts: int = 1


class Grounded(Program):
    """Wrap a Call with post-decode citation verification and retry.

    After the inner Call produces output containing ``Cited[T]`` or
    ``GroundedAnswer[T]`` fields, the Grounded wrapper verifies every
    cited span against the provided corpus.  On verification failure,
    the Call is retried with feedback about which spans failed.

    If all retries are exhausted, returns the best attempt (fewest
    verification errors).

    Args:
        producer: The Call to wrap.  Must produce output with
            ``Cited[T]`` fields or ``GroundedAnswer[T]`` output.
        corpus: Source text lookup — ``dict[str, str]`` mapping
            ``source_uri`` to source text.
        max_retries: Maximum retry attempts on verification failure.
            Default 2 (same as RAG).
        match_strategies: Verification strategies (tried in order).
        threshold: Similarity threshold for fuzzy strategies.
        span_map: Optional paragraph-level span map from
            :func:`tag_paragraphs` for resolving ``#pN`` URIs.
    """

    def __init__(
        self,
        producer: Any,  # Call — typed as Any to avoid circular import
        corpus: dict[str, str],
        *,
        max_retries: int = 2,
        match_strategies: Iterable[MatchStrategy] = DEFAULT_MATCH_STRATEGIES,
        threshold: float = 0.9,
        span_map: dict[str, tuple[int, int]] | None = None,
    ) -> None:
        super().__init__()
        self.producer = producer  # auto-registered as child
        self._corpus = corpus
        self._max_retries = max_retries
        self._match_strategies = tuple(match_strategies)
        self._threshold = threshold
        self._span_map = span_map

    async def forward(self, **kwargs: Any) -> GroundedResult:
        """Execute the grounded pipeline: produce → verify → retry.

        All keyword arguments are forwarded to the producer Call.

        Returns:
            :class:`GroundedResult` with the best output and
            verification status.
        """
        # First attempt
        output = await self.producer(**kwargs)
        errors = self._verify(output)

        if not errors:
            return GroundedResult(
                output=output,
                verification_errors=(),
                is_verified=True,
                attempts=1,
            )

        # Track best attempt
        best_output = output
        best_errors = errors
        attempts = 1

        # Retry with feedback
        for retry_idx in range(self._max_retries):
            feedback = self._build_feedback(best_errors)
            retry_producer = clone_call(self.producer, deep=False)

            # Inject feedback into instructions
            original_instructions = getattr(retry_producer, "instructions", "") or ""
            retry_producer.instructions = f"{original_instructions}\n\n{feedback}"

            try:
                retry_output = await retry_producer(**kwargs)
            except Exception as exc:
                logger.debug("Grounded: retry %d failed: %s", retry_idx + 1, exc, exc_info=True)
                break

            retry_errors = self._verify(retry_output)
            attempts += 1

            if not retry_errors:
                return GroundedResult(
                    output=retry_output,
                    verification_errors=(),
                    is_verified=True,
                    attempts=attempts,
                )

            # Keep the better attempt
            if len(retry_errors) < len(best_errors):
                best_output = retry_output
                best_errors = retry_errors

            logger.debug(
                "Grounded: retry %d — %d errors (best: %d)",
                retry_idx + 1,
                len(retry_errors),
                len(best_errors),
            )

        # Exhausted retries — return best attempt with errors
        logger.debug(
            "Grounded: exhausted %d retries, %d verification error(s) remain",
            self._max_retries,
            len(best_errors),
        )

        return GroundedResult(
            output=best_output,
            verification_errors=tuple(best_errors),
            is_verified=False,
            attempts=attempts,
        )

    def _verify(self, output: Any) -> list[SpanError]:
        """Verify all cited spans in the output."""
        # Check for Answer[T] (from GroundedAnswer output)
        if isinstance(output, Answer):
            return output.verify(
                self._corpus,
                strategies=self._match_strategies,
                threshold=self._threshold,
            )

        # Check for InsufficientEvidence (no verification needed)
        if isinstance(output, InsufficientEvidence):
            return []

        # Walk the output for Cited[T] fields
        return validate_cited_output(
            output,
            self._corpus,
            strategies=self._match_strategies,
            threshold=self._threshold,
            span_map=self._span_map,
        )

    @staticmethod
    def _build_feedback(
        errors: list[SpanError],
        *,
        max_errors: int | None = None,
        per_quote_char_budget: int | None = None,
    ) -> str:
        """Build retry feedback from verification errors.

        Args:
            errors: Verification errors from the previous attempt.
            max_errors: When set, only the first ``max_errors`` entries
                are inlined; the remainder are summarized as
                ``... and N more errors``. **Default ``None`` — every
                error is inlined.** The model is being asked to fix
                these citations; silently dropping errors past index N
                means the model never sees the failure it's supposed
                to repair. Pass an explicit cap only when the caller
                has a measured prompt-bloat issue.
            per_quote_char_budget: When set, truncates each error's
                ``quote`` excerpt to this many characters with a
                ``...`` marker. **Default ``None`` — the full quote is
                inlined.** The model is asked to fix a quote it must
                see verbatim; chopping at 80 chars means the model
                fixes the prefix and ignores the trailing context.
                Pass an explicit cap only when the operator has
                measured a specific context-window pressure.
        """
        lines = [
            "Your previous answer had citation verification errors. "
            "Please re-answer, ensuring every cited quote is an EXACT "
            "verbatim substring from the source text.\n\n"
            "Failed citations:"
        ]
        kept_errors = errors if max_errors is None else errors[:max_errors]
        for err in kept_errors:
            quote = err.span.quote
            if per_quote_char_budget is not None and len(quote) > per_quote_char_budget:
                dropped = len(quote) - per_quote_char_budget
                logger.warning(
                    "grounded: retry-feedback quote truncated span_index=%s "
                    "kept=%d dropped=%d budget=%d",
                    err.span_index,
                    per_quote_char_budget,
                    dropped,
                    per_quote_char_budget,
                )
                quote = quote[:per_quote_char_budget] + f"... (truncated {dropped} more chars)"
            lines.append(f"- Span {err.span_index}: quote={quote!r} reason={err.reason}")
        if max_errors is not None and len(errors) > max_errors:
            dropped_errors = len(errors) - max_errors
            logger.warning(
                "grounded: retry-feedback errors truncated kept=%d dropped=%d budget=%d",
                max_errors,
                dropped_errors,
                max_errors,
            )
            lines.append(f"  ... and {dropped_errors} more errors")
        return "\n".join(lines)
