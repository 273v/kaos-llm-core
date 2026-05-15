"""No-LLM extractive summarization.

:class:`ExtractiveSummary` wraps an external ranker (the canonical
implementation is
:class:`kaos_nlp_transformers.extraction.ExtractiveRanker`, but any
object that conforms to the :class:`Ranker` protocol works) and packages
the top-k sentences into a :class:`~kaos_llm_core.results.Summary`.

No LLM call is made; the Program runs entirely on embeddings or
cross-encoder scores supplied by the ranker. The returned
:class:`Summary` carries ``method="extractive"``, ``payload=None``, and
per-pick scores in the metadata so downstream code can distinguish a
purely extractive summary from an abstractive one at the type
boundary.

The picks are returned in **source order** (sorted by ``start``
offset) so the summary reads as a coherent narrative, even though the
ranker selected them by salience. The ranker-assigned rank and score
are preserved in ``metadata["picks"]`` for callers that need the
score-ordered view.

This Program is the no-LLM half of the §6.1 summarization taxonomy
and the foundation for :class:`HybridSummary` (Phase 6), which calls
:class:`ExtractiveSummary` for the pre-filter step before sending the
survivors through :class:`CitedSummary`.

.. warning::
   Extractive summaries are deterministic given the ranker, but the
   ranker itself depends on embedding-model output. Treat extractive
   summaries as triage signal: they reproduce sentences verbatim
   from the source, but the *selection* may miss material content.
   Do not rely on extractive summaries for legal, financial, medical,
   or compliance decisions without human review.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from kaos_nlp_core.segmentation import segment_sentences
from kaos_nlp_core.types import Segment

from kaos_llm_core.programs.base import Program
from kaos_llm_core.results import SourceSpan, Summary


@runtime_checkable
class RankedSegment(Protocol):
    """Structural type for objects returned by a :class:`Ranker`.

    Matches :class:`kaos_nlp_transformers.extraction.ScoredSegment` and
    any caller-supplied stub used in offline tests. Only the attributes
    listed here are consumed; additional fields on the concrete object
    are ignored.
    """

    text: str
    start: int
    end: int
    score: float
    rank: int


@runtime_checkable
class Ranker(Protocol):
    """Structural type for a sentence-level salience ranker.

    Mirrors :meth:`kaos_nlp_transformers.extraction.ExtractiveRanker.rank`
    so the canonical implementation in ``kaos-nlp-transformers``
    satisfies the protocol without an explicit ``Ranker`` import. Stubs
    used in offline tests conform to the same shape.

    ``query``, ``top_k``, and ``diversify`` are forwarded as kwargs.
    """

    def rank(
        self,
        sentences: Sequence[Segment],
        *,
        query: str | None = ...,
        top_k: int | None = ...,
        diversify: float = ...,
    ) -> Sequence[RankedSegment]:  # pragma: no cover - protocol
        ...


class ExtractiveSummary(Program):
    """No-LLM extractive summary built on top of a :class:`Ranker`.

    Workflow:

    1. Segment the input via
       :func:`kaos_nlp_core.segmentation.segment_sentences`.
    2. Score the sentences via ``ranker.rank(...)``.
    3. Re-sort the top-k picks by source ``start`` offset.
    4. Join the picks with ``joiner`` and wrap in
       :class:`~kaos_llm_core.results.Summary` with
       ``method="extractive"``.

    Args:
        ranker: Object conforming to :class:`Ranker`. The canonical
            implementation is
            :class:`kaos_nlp_transformers.extraction.ExtractiveRanker`.
        top_k: Number of sentences to pick. Defaults to ``5``. Capped
            at the number of available sentences at runtime.
        diversify: MMR diversity parameter in ``[0, 1]`` passed through
            to the ranker. ``0`` (default) is pure salience; higher
            values penalise redundancy.
        joiner: String inserted between picks when materialising the
            summary text. Defaults to ``" "`` (single space) so the
            output reads as flowing prose; pass ``"\\n\\n"`` for a
            bullet-style summary.

    The Program holds no :class:`~kaos_llm_core.programs.call.Call`
    children — :meth:`Program.invoke` still builds a (childless) trace
    and returns an :class:`~kaos_llm_core.programs._invocation.Invocation`
    with zero usage, so the surface is type-stable with the LLM-backed
    summarizers.
    """

    def __init__(
        self,
        *,
        ranker: Ranker,
        top_k: int = 5,
        diversify: float = 0.0,
        joiner: str = " ",
    ) -> None:
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k}")
        if not (0.0 <= diversify <= 1.0):
            raise ValueError(f"diversify must be in [0, 1], got {diversify}")
        self._ranker = ranker
        self._top_k = top_k
        self._diversify = diversify
        self._joiner = joiner

    async def forward(  # ty: ignore[invalid-method-override]
        self,
        *,
        text: str,
        query: str | None = None,
        top_k: int | None = None,
        parent_id: str | None = None,
    ) -> Summary[str]:
        """Produce an extractive summary of ``text``.

        Args:
            text: Source text to summarise. Empty input returns an
                empty :class:`Summary` with no picks.
            query: Optional natural-language query. When supplied, the
                ranker uses query-focused scoring; otherwise centroid
                scoring.
            top_k: Per-call override of the constructor's ``top_k``.
                ``None`` keeps the constructor default.
            parent_id: Optional parent-chunk identifier, recorded in
                :attr:`Summary.chunks_used`.

        Returns:
            :class:`Summary` whose ``text`` is the joined picks in
            source order. ``method="extractive"``. ``source_spans``
            carries the half-open ``(start, end)`` offsets. Metadata:

            - ``program``: ``"ExtractiveSummary"``
            - ``ranker``: class name of the supplied ranker
            - ``query``: the query (or ``None``)
            - ``top_k_requested``: the resolved top-k for this call
            - ``picks``: list of ``{"rank", "score", "start", "end"}``
              in score order (highest first)
            - ``n_sentences``: number of input sentences after
              segmentation
        """
        resolved_top_k = top_k if top_k is not None else self._top_k
        if resolved_top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {resolved_top_k}")

        if not text:
            return Summary[str](
                text="",
                method="extractive",
                chunks_used=[parent_id] if parent_id else [],
                metadata={
                    "program": "ExtractiveSummary",
                    "ranker": type(self._ranker).__name__,
                    "query": query,
                    "top_k_requested": resolved_top_k,
                    "picks": [],
                    "n_sentences": 0,
                },
            )

        sentences: list[Segment] = list(segment_sentences(text))
        if not sentences:
            return Summary[str](
                text="",
                method="extractive",
                chunks_used=[parent_id] if parent_id else [],
                metadata={
                    "program": "ExtractiveSummary",
                    "ranker": type(self._ranker).__name__,
                    "query": query,
                    "top_k_requested": resolved_top_k,
                    "picks": [],
                    "n_sentences": 0,
                },
            )

        ranked: Sequence[RankedSegment] = self._ranker.rank(
            sentences,
            query=query,
            top_k=resolved_top_k,
            diversify=self._diversify,
        )
        # `rank()` already truncates to top_k; defensive cap mirrors
        # the protocol contract for stub rankers that might return more.
        picks = list(ranked)[:resolved_top_k]

        # Score-order metadata before re-sorting for narrative output.
        picks_meta: list[dict[str, Any]] = [
            {
                "rank": int(p.rank),
                "score": float(p.score),
                "start": int(p.start),
                "end": int(p.end),
            }
            for p in picks
        ]

        # Re-sort by source order so the summary reads as a coherent
        # narrative. The score-ordered view is preserved in metadata.
        picks_by_source = sorted(picks, key=lambda p: (p.start, p.end))
        summary_text = self._joiner.join(p.text for p in picks_by_source)
        source_spans = [
            SourceSpan(start=int(p.start), end=int(p.end), source_id=parent_id)
            for p in picks_by_source
        ]

        return Summary[str](
            text=summary_text,
            method="extractive",
            chunks_used=[parent_id] if parent_id else [],
            source_spans=source_spans,
            metadata={
                "program": "ExtractiveSummary",
                "ranker": type(self._ranker).__name__,
                "query": query,
                "top_k_requested": resolved_top_k,
                "picks": picks_meta,
                "n_sentences": len(sentences),
            },
        )


__all__ = [
    "ExtractiveSummary",
    "RankedSegment",
    "Ranker",
]
