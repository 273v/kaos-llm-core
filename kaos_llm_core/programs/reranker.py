"""JudgeReranker -- LLM-as-reranker using a relevance-scoring Signature.

Implements the ``Reranker`` protocol from ``kaos_nlp_core.retrieval`` by
scoring each passage's relevance to the query via an LLM call.  Uses a
dedicated ``RelevanceSignature`` (simpler and cheaper than the full
``JudgmentSignature``) and runs candidates concurrently with a semaphore
to control parallelism.

This is the "zero new dependencies" reranker: it uses only existing
kaos-llm-core + kaos-llm-client infrastructure.  For cross-encoder
reranking (faster, free, deterministic), see the planned
``CrossEncoderReranker`` in kaos-nlp-transformers.
"""

from __future__ import annotations

import asyncio
from typing import Any

from kaos_core.logging import get_logger
from kaos_nlp_core.retrieval.protocol import RetrievalResult
from kaos_nlp_core.retrieval.reranker import RankedResult

from kaos_llm_core.programs.call import Call
from kaos_llm_core.signatures.fields import InputField, OutputField
from kaos_llm_core.signatures.signature import Signature

logger = get_logger(__name__)


# ── Signature ────────────────────────────────────────────────────────────────


class RelevanceSignature(Signature):
    """Score how relevant a passage is to a search query.

    Return a relevance_score between 0.0 (completely irrelevant) and
    1.0 (perfectly relevant).  A score of 0.5 means the passage is
    tangentially related but does not directly answer the query.
    """

    query: str = InputField(description="The search query")
    passage: str = InputField(description="The candidate passage to score for relevance")
    relevance_score: float = OutputField(
        description="Relevance score from 0.0 (irrelevant) to 1.0 (perfectly relevant)"
    )


# ── JudgeReranker ────────────────────────────────────────────────────────────


class JudgeReranker:
    """Reranker that uses an LLM to score query-passage relevance.

    Implements the ``Reranker`` protocol from ``kaos_nlp_core.retrieval``.

    Each candidate passage is scored independently via a lightweight
    ``RelevanceSignature`` call.  Scoring runs concurrently (up to
    ``max_concurrent`` in-flight calls) using ``asyncio.gather`` with
    a semaphore.  Results are sorted by score descending and optionally
    truncated to ``top_k``.

    Example::

        reranker = JudgeReranker("anthropic:claude-haiku-4-5")
        ranked = await reranker.rerank("renewable energy policy", results, top_k=5)
        for r in ranked:
            print(r.rerank_score, r.result.text[:80])
    """

    def __init__(
        self,
        model: str,
        *,
        max_concurrent: int = 5,
        core_settings: Any = None,
    ) -> None:
        self._call = Call(
            RelevanceSignature,
            model=model,
            core_settings=core_settings,
        )
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def _score_one(self, query: str, passage: str) -> float:
        """Score a single passage against the query.

        Returns a float clamped to [0.0, 1.0].
        """
        async with self._semaphore:
            result = await self._call(query=query, passage=passage)
            score = float(result.relevance_score)
            return max(0.0, min(1.0, score))

    async def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RankedResult]:
        """Score each result's relevance to the query using the LLM.

        Args:
            query: The search query.
            results: Retrieval results to rerank.
            top_k: Return only the top K results.  ``None`` returns all,
                reordered by rerank score.

        Returns:
            ``list[RankedResult]`` sorted by ``rerank_score`` descending,
            truncated to ``top_k`` if specified.
        """
        if not results:
            return []

        # Score all passages concurrently (bounded by the semaphore).
        scores = await asyncio.gather(
            *[self._score_one(query, r.text) for r in results],
            return_exceptions=True,
        )

        # Pair results with scores, treating failures as 0.0.
        ranked: list[RankedResult] = []
        for result, score in zip(results, scores, strict=True):
            if isinstance(score, BaseException):
                logger.warning(
                    "JudgeReranker: scoring failed for passage (doc_id=%s): %s",
                    result.doc_id,
                    score,
                )
                score_value = 0.0
            else:
                score_value = score
            ranked.append(RankedResult(result=result, rerank_score=score_value))

        # Sort by score descending.
        ranked.sort(key=lambda r: r.rerank_score, reverse=True)

        # Truncate to top_k if specified.
        if top_k is not None:
            ranked = ranked[:top_k]

        return ranked


__all__ = [
    "JudgeReranker",
    "RelevanceSignature",
]
