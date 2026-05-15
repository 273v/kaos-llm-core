"""Single-shot abstractive summarization Programs."""

from __future__ import annotations

from typing import Any

from kaos_llm_client import BaseProviderClient
from kaos_llm_client.settings import KaosLLMSettings

from kaos_llm_core.codecs.base import Codec
from kaos_llm_core.programs.base import Program
from kaos_llm_core.programs.call import Call
from kaos_llm_core.results import Summary
from kaos_llm_core.settings import KaosLLMCoreSettings
from kaos_llm_core.signatures import InputField, OutputField, Signature
from kaos_llm_core.signatures.grounding import (
    DEFAULT_MATCH_STRATEGIES,
    Answer,
    GroundedAnswer,
    MatchStrategy,
)
from kaos_llm_core.types import Example


class AbstractiveSummarySignature(Signature):
    """Produce a concise abstractive summary of the input text.

    The summary should preserve the key facts, dates, parties, and
    obligations from the source. Do not introduce information that is
    not in the source.
    """

    text: str = InputField(description="Source text to summarize.")
    summary: str = OutputField(
        description=(
            "A concise abstractive summary of the source text. "
            "Preserves key facts; introduces no new content."
        ),
    )


class _CitedSummarySignature(Signature):
    """Produce a cited abstractive summary of the input text.

    Each claim in the summary must be supported by one or more
    verbatim quotes from the source. Return an ``Answer`` whose
    ``claims`` enumerate the supporting evidence.
    """

    text: str = InputField(description="Source text to summarize.")
    summary: GroundedAnswer[str] = OutputField(
        description=(
            "Cited abstractive summary. The ``value`` is the natural-language "
            "summary; each ``claim`` quotes the substring of ``text`` that "
            "supports it. Refuse via InsufficientEvidence if the source "
            "cannot be summarized faithfully."
        ),
    )


class AbstractiveSummary(Program):
    """Single-shot abstractive summarization.

    Wraps a single :class:`Call` over
    :class:`AbstractiveSummarySignature`. Returns a
    :class:`~kaos_llm_core.results.Summary[str]` whose ``text`` is the
    summary string.

    The Program-vs-Call distinction matters here: callers always get
    back a :class:`~kaos_llm_core.results.Summary` regardless of how
    the underlying LLM was prompted, so swapping
    :class:`AbstractiveSummary` for any long-doc Program is a
    type-stable substitution.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        codec: Codec | None = None,
        client: BaseProviderClient | None = None,
        settings: KaosLLMSettings | None = None,
        core_settings: KaosLLMCoreSettings | None = None,
        examples: list[Example] | None = None,
        instructions: str | None = None,
        max_retries: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.summarizer = Call(
            AbstractiveSummarySignature,
            model=model,
            codec=codec,
            client=client,
            settings=settings,
            core_settings=core_settings,
            examples=examples,
            instructions=instructions,
            max_retries=max_retries,
            **kwargs,
        )

    async def forward(  # ty: ignore[invalid-method-override]
        self, *, text: str, parent_id: str | None = None
    ) -> Summary[str]:
        result = await self.summarizer(text=text)
        return Summary[str](
            text=result.summary,
            method="abstractive",
            chunks_used=[parent_id] if parent_id else [],
            metadata={
                "program": "AbstractiveSummary",
            },
        )


class CitedSummary(Program):
    """Abstractive summary with cited claims that are **verified at runtime**.

    Uses :class:`~kaos_llm_core.signatures.grounding.GroundedAnswer` to
    force the LLM to ground each claim in verbatim source text. After
    the LLM returns, every claim's supporting spans are checked against
    the actual source via :meth:`Answer.verify`; only spans from
    fully-verified claims land in :attr:`Summary.source_spans`. This
    makes a hallucinated quote distinguishable at the type boundary —
    the rich payload is preserved unmodified for caller inspection,
    but the curated ``source_spans`` list reflects only what survived
    verification.

    Args:
        verify_strategies: Match strategies to try when verifying spans
            against the source. Defaults to
            :data:`DEFAULT_MATCH_STRATEGIES` (``STRICT`` then
            ``SUBSTRING``). Pass a longer chain for fuzzy tolerance.
        verify_threshold: Similarity threshold for the ``FUZZY_*``
            strategies. Default ``0.9``.
        refuse_below: Floor on verified-claim ratio. When the fraction
            of claims fully verified is strictly less than
            ``refuse_below``, the summary text is replaced with the
            empty string, ``source_spans`` is empty, and the metadata
            records ``"cited.refused": True``. Default ``0.0``
            (off — preserve current behavior, never refuse). Set to
            e.g. ``0.5`` for "at least half the claims must verify."

    The full ``GroundedAnswer`` payload (including unverified claims)
    remains on :attr:`Summary.payload`; downstream callers that want
    to inspect what failed read ``payload`` and re-run
    ``payload.verify(...)`` themselves. The curated, verified-only
    view lives on :attr:`Summary.source_spans` plus
    :attr:`Summary.metadata`.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        codec: Codec | None = None,
        client: BaseProviderClient | None = None,
        settings: KaosLLMSettings | None = None,
        core_settings: KaosLLMCoreSettings | None = None,
        examples: list[Example] | None = None,
        instructions: str | None = None,
        max_retries: int | None = None,
        verify_strategies: tuple[MatchStrategy, ...] = DEFAULT_MATCH_STRATEGIES,
        verify_threshold: float = 0.9,
        refuse_below: float = 0.0,
        **kwargs: Any,
    ) -> None:
        if not (0.0 <= refuse_below <= 1.0):
            raise ValueError(f"refuse_below must be in [0.0, 1.0], got {refuse_below}")
        self._verify_strategies = tuple(verify_strategies)
        self._verify_threshold = verify_threshold
        self._refuse_below = refuse_below
        self.cited_summarizer = Call(
            _CitedSummarySignature,
            model=model,
            codec=codec,
            client=client,
            settings=settings,
            core_settings=core_settings,
            examples=examples,
            instructions=instructions,
            max_retries=max_retries,
            **kwargs,
        )

    async def forward(  # ty: ignore[invalid-method-override]
        self,
        *,
        text: str,
        parent_id: str | None = None,
    ) -> Summary[GroundedAnswer[str]]:
        result = await self.cited_summarizer(text=text)
        grounded: GroundedAnswer[str] = result.summary
        # ``Answer.value`` is generic over ``T``; cast to ``str`` for the
        # ``Summary.text`` field. The Signature constrains ``T = str``
        # so the value is always a string in practice.
        summary_text = str(grounded.value) if isinstance(grounded, Answer) else ""
        # Pool span coordinates from verified claims only.
        from kaos_llm_core.results import SourceSpan

        spans: list[SourceSpan] = []
        verified_claim_count = 0
        unverified_claim_count = 0
        error_reasons: list[str] = []
        is_answer = isinstance(grounded, Answer)
        total_claims = len(grounded.claims) if is_answer else 0

        if is_answer and total_claims > 0:
            # Use a callable corpus: every supporting span on a single-doc
            # CitedSummary resolves to the same ``text``, regardless of
            # whatever ``source_uri`` the LLM picked for the Span.
            def _corpus(_source_uri: str) -> str:
                return text

            errors = grounded.verify(
                corpus=_corpus,
                strategies=self._verify_strategies,
                threshold=self._verify_threshold,
            )
            # Group errors by claim index so we can include all-verified
            # claims' spans and exclude any claim with at least one bad
            # span. ``errors`` is empty iff every span verifies.
            bad_claim_indices: set[int] = {err.claim_index for err in errors}
            error_reasons = [err.reason for err in errors]
            for claim_index, claim in enumerate(grounded.claims):
                if claim_index in bad_claim_indices:
                    unverified_claim_count += 1
                    continue
                verified_claim_count += 1
                for span in claim.supporting_spans:
                    start, end = span.char_span
                    spans.append(
                        SourceSpan(
                            start=start,
                            end=end,
                            source_id=parent_id or span.source_uri or None,
                        )
                    )

        # Refusal floor: if the verified fraction is below the threshold,
        # collapse the summary text + spans so a downstream caller cannot
        # mistake the hallucinated payload for a verified result. The full
        # ``payload`` is preserved on the Summary for diagnosis.
        verified_ratio = verified_claim_count / total_claims if total_claims > 0 else 0.0
        refused = total_claims > 0 and verified_ratio < self._refuse_below
        if refused:
            summary_text = ""
            spans = []

        return Summary[GroundedAnswer[str]](
            text=summary_text,
            payload=grounded,
            method="abstractive",
            chunks_used=[parent_id] if parent_id else [],
            source_spans=spans,
            metadata={
                "program": "CitedSummary",
                "cited.kind": getattr(grounded, "kind", "unknown"),
                "cited.claim_count": total_claims,
                "cited.verified_claim_count": verified_claim_count,
                "cited.unverified_claim_count": unverified_claim_count,
                "cited.verified_ratio": round(verified_ratio, 4),
                "cited.refused": refused,
                "cited.verify_strategies": [s.value for s in self._verify_strategies],
                "cited.error_reasons": error_reasons[:5],  # truncate for noise
            },
        )


__all__ = [
    "AbstractiveSummary",
    "AbstractiveSummarySignature",
    "CitedSummary",
]
