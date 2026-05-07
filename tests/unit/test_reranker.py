"""Unit tests for JudgeReranker with mocked LLM responses."""

from __future__ import annotations

import json
from typing import Any

from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart
from kaos_nlp_core.retrieval import RankedResult, Reranker, RetrievalResult

from kaos_llm_core.programs.reranker import JudgeReranker, RelevanceSignature


def _json_response(data: dict[str, Any]) -> ProviderResponse:
    return ProviderResponse.model_construct(
        provider="function",
        model="function-test",
        raw={},
        parts=[ContentPart.model_construct(type="text", text=json.dumps(data))],
        usage=UsageInfo.model_construct(input_tokens=20, output_tokens=10, total_tokens=30),
        stop_reason="end_turn",
        status_code=200,
        response_headers={},
    )


def _make_results() -> list[RetrievalResult]:
    """Build 5 retrieval results: 2 relevant (contract/breach) + 3 irrelevant."""
    return [
        RetrievalResult(
            text="The defendant breached the contract by failing to deliver goods on time.",
            score=2.5,
            doc_id="0",
            metadata={"page": 1},
        ),
        RetrievalResult(
            text="Chocolate cake recipes require flour, sugar, and cocoa powder.",
            score=2.0,
            doc_id="1",
        ),
        RetrievalResult(
            text="The court found that the breach of contract was material and intentional.",
            score=1.5,
            doc_id="2",
            metadata={"page": 3},
        ),
        RetrievalResult(
            text="The weather forecast calls for sunny skies and mild temperatures.",
            score=1.0,
            doc_id="3",
        ),
        RetrievalResult(
            text="Python programming uses indentation to define code blocks.",
            score=0.5,
            doc_id="4",
        ),
    ]


# Map doc_id to a predetermined relevance score for deterministic testing.
_MOCK_SCORES: dict[str, float] = {
    "0": 0.92,  # highly relevant (breach + contract)
    "1": 0.05,  # irrelevant (cooking)
    "2": 0.95,  # highly relevant (breach of contract)
    "3": 0.03,  # irrelevant (weather)
    "4": 0.02,  # irrelevant (programming)
}


def _scoring_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
    """Mock LLM that returns predetermined relevance scores based on passage content."""
    # Extract the passage from the messages to figure out which doc this is.
    blob = " ".join(
        m.get("content", "") if isinstance(m.get("content"), str) else "" for m in messages
    )
    # Determine score based on passage content.
    if "breached the contract" in blob:
        score = _MOCK_SCORES["0"]
    elif "chocolate cake" in blob.lower():
        score = _MOCK_SCORES["1"]
    elif "breach of contract was material" in blob:
        score = _MOCK_SCORES["2"]
    elif "weather forecast" in blob:
        score = _MOCK_SCORES["3"]
    elif "Python programming" in blob:
        score = _MOCK_SCORES["4"]
    else:
        score = 0.5
    return _json_response({"relevance_score": score})


class TestJudgeRerankerProtocol:
    def test_implements_reranker_protocol(self) -> None:
        """JudgeReranker should be recognized as a Reranker by isinstance."""
        reranker = JudgeReranker("function-test")
        assert isinstance(reranker, Reranker)


class TestJudgeRerankerUnit:
    async def test_rerank_sorts_by_relevance(self) -> None:
        """Relevant passages should be ranked above irrelevant ones."""
        client = FunctionClient(function=_scoring_fn)
        reranker = JudgeReranker("function-test", max_concurrent=5)
        reranker._call._client = client

        results = _make_results()
        ranked = await reranker.rerank("breach of contract", results)

        assert len(ranked) == 5
        # The two relevant docs (doc_id 0 and 2) should be on top.
        top_two_ids = {ranked[0].result.doc_id, ranked[1].result.doc_id}
        assert top_two_ids == {"0", "2"}

    async def test_rerank_top_k(self) -> None:
        """top_k should truncate results."""
        client = FunctionClient(function=_scoring_fn)
        reranker = JudgeReranker("function-test", max_concurrent=5)
        reranker._call._client = client

        results = _make_results()
        ranked = await reranker.rerank("breach of contract", results, top_k=2)

        assert len(ranked) == 2
        assert ranked[0].result.doc_id in ("0", "2")
        assert ranked[1].result.doc_id in ("0", "2")

    async def test_rerank_empty_input(self) -> None:
        """Empty input should return empty output."""
        reranker = JudgeReranker("function-test")
        ranked = await reranker.rerank("query", [])
        assert ranked == []

    async def test_scores_in_range(self) -> None:
        """All rerank scores should be clamped to [0.0, 1.0]."""
        client = FunctionClient(function=_scoring_fn)
        reranker = JudgeReranker("function-test", max_concurrent=5)
        reranker._call._client = client

        results = _make_results()
        ranked = await reranker.rerank("breach of contract", results)

        for r in ranked:
            assert 0.0 <= r.rerank_score <= 1.0

    async def test_metadata_preserved(self) -> None:
        """Original RetrievalResult metadata should survive reranking."""
        client = FunctionClient(function=_scoring_fn)
        reranker = JudgeReranker("function-test", max_concurrent=5)
        reranker._call._client = client

        results = _make_results()
        ranked = await reranker.rerank("breach of contract", results)

        # Find the result with doc_id "0" and verify its metadata.
        doc0 = next(r for r in ranked if r.result.doc_id == "0")
        assert doc0.result.metadata["page"] == 1

    async def test_sorted_descending(self) -> None:
        """Results should be sorted by rerank_score descending."""
        client = FunctionClient(function=_scoring_fn)
        reranker = JudgeReranker("function-test", max_concurrent=5)
        reranker._call._client = client

        results = _make_results()
        ranked = await reranker.rerank("breach of contract", results)

        scores = [r.rerank_score for r in ranked]
        assert scores == sorted(scores, reverse=True)

    async def test_returns_ranked_result_type(self) -> None:
        """Each element should be a RankedResult wrapping a RetrievalResult."""
        client = FunctionClient(function=_scoring_fn)
        reranker = JudgeReranker("function-test")
        reranker._call._client = client

        results = _make_results()[:1]
        ranked = await reranker.rerank("query", results)

        assert len(ranked) == 1
        assert isinstance(ranked[0], RankedResult)
        assert isinstance(ranked[0].result, RetrievalResult)

    async def test_score_clamped_above_one(self) -> None:
        """Scores > 1.0 from the LLM should be clamped to 1.0."""

        def over_score_fn(
            messages: list[dict[str, Any]], profile: ModelProfile
        ) -> ProviderResponse:
            return _json_response({"relevance_score": 1.5})

        client = FunctionClient(function=over_score_fn)
        reranker = JudgeReranker("function-test")
        reranker._call._client = client

        results = [RetrievalResult(text="test", score=1.0, doc_id="0")]
        ranked = await reranker.rerank("query", results)
        assert ranked[0].rerank_score == 1.0

    async def test_score_clamped_below_zero(self) -> None:
        """Scores < 0.0 from the LLM should be clamped to 0.0."""

        def under_score_fn(
            messages: list[dict[str, Any]], profile: ModelProfile
        ) -> ProviderResponse:
            return _json_response({"relevance_score": -0.5})

        client = FunctionClient(function=under_score_fn)
        reranker = JudgeReranker("function-test")
        reranker._call._client = client

        results = [RetrievalResult(text="test", score=1.0, doc_id="0")]
        ranked = await reranker.rerank("query", results)
        assert ranked[0].rerank_score == 0.0


class TestRelevanceSignature:
    def test_signature_fields(self) -> None:
        """RelevanceSignature should have the expected input/output fields."""
        from kaos_llm_core.signatures.introspection import get_input_fields, get_output_fields

        inputs = get_input_fields(RelevanceSignature)
        outputs = get_output_fields(RelevanceSignature)

        assert "query" in inputs
        assert "passage" in inputs
        assert "relevance_score" in outputs
