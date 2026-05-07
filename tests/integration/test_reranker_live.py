"""Integration test for JudgeReranker with a real LLM call.

Requires an Anthropic API key.  Skipped if no key is available.
"""

from __future__ import annotations

import pytest
from kaos_nlp_core.retrieval import RetrievalResult

from kaos_llm_core.programs.reranker import JudgeReranker

from .conftest import requires_anthropic


def _make_test_results() -> list[RetrievalResult]:
    """Build 5 passages: 2 relevant to "breach of contract" + 3 irrelevant."""
    return [
        # Relevant
        RetrievalResult(
            text=(
                "A breach of contract occurs when one party fails to fulfill their "
                "obligations under the agreement without a lawful excuse. The non-breaching "
                "party may seek damages, specific performance, or rescission of the contract."
            ),
            score=2.0,
            doc_id="relevant-1",
            metadata={"source": "legal-textbook"},
        ),
        # Irrelevant
        RetrievalResult(
            text=(
                "The Great Barrier Reef is the world's largest coral reef system, "
                "composed of over 2,900 individual reef systems and hundreds of islands "
                "made of over 600 types of hard and soft coral."
            ),
            score=1.5,
            doc_id="irrelevant-1",
        ),
        # Relevant
        RetrievalResult(
            text=(
                "Material breach of contract entitles the non-breaching party to terminate "
                "the agreement and sue for expectation damages. Courts consider the severity "
                "of the breach, whether the breaching party acted in good faith, and the "
                "extent to which the injured party received the benefit of the bargain."
            ),
            score=1.0,
            doc_id="relevant-2",
            metadata={"source": "case-law"},
        ),
        # Irrelevant
        RetrievalResult(
            text=(
                "Photosynthesis is the process by which green plants and certain other "
                "organisms transform light energy into chemical energy. During this process, "
                "carbon dioxide and water are converted into glucose and oxygen."
            ),
            score=0.8,
            doc_id="irrelevant-2",
        ),
        # Irrelevant
        RetrievalResult(
            text=(
                "The International Space Station orbits Earth at an altitude of approximately "
                "420 kilometers and travels at about 28,000 kilometers per hour, completing "
                "one orbit roughly every 90 minutes."
            ),
            score=0.5,
            doc_id="irrelevant-3",
        ),
    ]


@requires_anthropic
class TestJudgeRerankerLive:
    async def test_relevant_passages_ranked_on_top(self) -> None:
        """The two contract-law passages should be ranked above the three irrelevant ones."""
        reranker = JudgeReranker(
            "anthropic:claude-haiku-4-5",
            max_concurrent=5,
        )
        results = _make_test_results()
        ranked = await reranker.rerank("breach of contract", results, top_k=5)

        assert len(ranked) == 5

        # The top 2 results should be the relevant passages.
        top_two_ids = {ranked[0].result.doc_id, ranked[1].result.doc_id}
        assert top_two_ids == {"relevant-1", "relevant-2"}, (
            f"Expected relevant passages on top, got: "
            f"{[(r.result.doc_id, r.rerank_score) for r in ranked]}"
        )

        # Relevant scores should be significantly higher than irrelevant.
        relevant_scores = [r.rerank_score for r in ranked if "relevant" in r.result.doc_id]
        irrelevant_scores = [r.rerank_score for r in ranked if "irrelevant" in r.result.doc_id]
        assert min(relevant_scores) > max(irrelevant_scores), (
            f"Relevant scores {relevant_scores} should all exceed "
            f"irrelevant scores {irrelevant_scores}"
        )

    @pytest.mark.parametrize("top_k", [1, 2, 3])
    async def test_top_k_truncation(self, top_k: int) -> None:
        """top_k should be respected with a real LLM."""
        reranker = JudgeReranker(
            "anthropic:claude-haiku-4-5",
            max_concurrent=3,
        )
        results = _make_test_results()
        ranked = await reranker.rerank("breach of contract", results, top_k=top_k)
        assert len(ranked) == top_k

    async def test_scores_in_valid_range(self) -> None:
        """All scores from a real LLM should be in [0.0, 1.0]."""
        reranker = JudgeReranker(
            "anthropic:claude-haiku-4-5",
            max_concurrent=5,
        )
        results = _make_test_results()
        ranked = await reranker.rerank("breach of contract", results)

        for r in ranked:
            assert 0.0 <= r.rerank_score <= 1.0, (
                f"Score {r.rerank_score} out of range for doc {r.result.doc_id}"
            )
