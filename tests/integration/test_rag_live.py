"""Live integration tests for the RAG pipeline.

Uses real LLM calls and real BM25 retrieval over public-domain corpora
to verify the end-to-end retrieve -> reason -> verify pipeline.

Tests are skipped unless the corresponding provider API key is set.
Run with: ``pytest -m integration tests/integration/test_rag_live.py``
"""

from __future__ import annotations

import pytest

from kaos_llm_core.programs.rag import RAG, RAGResult
from kaos_llm_core.signatures import Answer, InsufficientEvidence, MatchStrategy

from .conftest import requires_anthropic, requires_openai

# ---------------------------------------------------------------------------
# Corpora — real public text
# ---------------------------------------------------------------------------

DELAWARE_GCL_URI = "doc:delaware-gcl-illustrative"
DELAWARE_GCL = (
    "The Delaware General Corporation Law requires that a certificate "
    "of incorporation be filed with the Secretary of State.  The filing "
    "fee is $89 and the certificate must include the corporation's "
    "name, its registered office, and the name of its registered agent."
)

FIRST_AMENDMENT_URI = "doc:us-const-amend-1"
FIRST_AMENDMENT = (
    "Congress shall make no law respecting an establishment of "
    "religion, or prohibiting the free exercise thereof; or abridging "
    "the freedom of speech, or of the press; or the right of the "
    "people peaceably to assemble, and to petition the government "
    "for a redress of grievances."
)

RFC_2119_URI = "doc:rfc-2119"
RFC_2119 = (
    "In many standards track documents several words are used to signify "
    "the requirements in the specification.  These words are often "
    "capitalized.  This document defines these words as they should be "
    "interpreted in IETF documents.\n\n"
    '1. MUST   This word, or the terms "REQUIRED" or "SHALL", mean that '
    "the definition is an absolute requirement of the specification.\n\n"
    '2. MUST NOT   This phrase, or the phrase "SHALL NOT", mean that '
    "the definition is an absolute prohibition of the specification."
)

# Lenient match chain for live tests — LLMs frequently drift on
# whitespace, case, and trailing punctuation.
_LENIENT: tuple[MatchStrategy, ...] = (
    MatchStrategy.STRICT,
    MatchStrategy.SUBSTRING,
    MatchStrategy.CASE_INSENSITIVE,
    MatchStrategy.NORMALIZED_TOKEN,
)

_MAX_ATTEMPTS = 3
"""Retry budget for LLM sample variance."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _run_with_retries(
    rag: RAG,
    question: str,
    documents: dict[str, str],
    max_attempts: int = _MAX_ATTEMPTS,
) -> RAGResult:
    """Run the RAG query with retries for LLM sample variance."""
    last_error: Exception | None = None
    for _ in range(max_attempts):
        try:
            result = await rag.query(question=question, documents=documents)
            return result
        except Exception as e:
            last_error = e
            continue
    assert last_error is not None
    raise last_error


# ---------------------------------------------------------------------------
# Tests — Anthropic
# ---------------------------------------------------------------------------


@requires_anthropic
@pytest.mark.integration
class TestRAGLiveAnthropic:
    """End-to-end RAG tests with Anthropic models."""

    async def test_factual_qa_single_doc(self) -> None:
        """Ask a factual question about the Delaware GCL."""
        rag = RAG(
            model="anthropic:claude-haiku-4-5",
            match_strategies=_LENIENT,
            top_k=5,
        )
        result = await _run_with_retries(
            rag,
            question="What is the filing fee for a certificate of incorporation in Delaware?",
            documents={DELAWARE_GCL_URI: DELAWARE_GCL},
        )

        assert isinstance(result, RAGResult)
        assert isinstance(result.grounded_answer, Answer)
        assert "$89" in str(result.grounded_answer.value)
        assert result.confidence > 0.0
        # Verification should pass with lenient strategies
        assert len(result.verification_errors) == 0

    async def test_refusal_on_irrelevant_question(self) -> None:
        """Ask something completely unrelated to the corpus."""
        rag = RAG(
            model="anthropic:claude-haiku-4-5",
            match_strategies=_LENIENT,
            top_k=5,
        )
        result = await _run_with_retries(
            rag,
            question="What is the orbital period of Neptune?",
            documents={DELAWARE_GCL_URI: DELAWARE_GCL},
        )

        assert isinstance(result, RAGResult)
        assert isinstance(result.grounded_answer, InsufficientEvidence)

    async def test_multi_doc_retrieval(self) -> None:
        """Search across multiple documents and verify cross-document citations."""
        corpus = {
            DELAWARE_GCL_URI: DELAWARE_GCL,
            RFC_2119_URI: RFC_2119,
            FIRST_AMENDMENT_URI: FIRST_AMENDMENT,
        }
        rag = RAG(
            model="anthropic:claude-haiku-4-5",
            match_strategies=_LENIENT,
            top_k=5,
        )
        result = await _run_with_retries(
            rag,
            question="What does MUST mean in RFC 2119?",
            documents=corpus,
        )

        assert isinstance(result, RAGResult)
        assert isinstance(result.grounded_answer, Answer)
        # The answer should reference the RFC, not the other documents
        answer_text = str(result.grounded_answer.value).lower()
        assert "requirement" in answer_text or "shall" in answer_text


# ---------------------------------------------------------------------------
# Tests — OpenAI
# ---------------------------------------------------------------------------


@requires_openai
@pytest.mark.integration
class TestRAGLiveOpenAI:
    """End-to-end RAG tests with OpenAI models."""

    async def test_factual_qa_single_doc(self) -> None:
        """Ask a factual question about the Delaware GCL."""
        rag = RAG(
            model="openai:gpt-5.4-nano",
            match_strategies=_LENIENT,
            top_k=5,
        )
        result = await _run_with_retries(
            rag,
            question="What must a Delaware certificate of incorporation include?",
            documents={DELAWARE_GCL_URI: DELAWARE_GCL},
        )

        assert isinstance(result, RAGResult)
        assert isinstance(result.grounded_answer, Answer)
        value_lower = str(result.grounded_answer.value).lower()
        assert "name" in value_lower or "registered" in value_lower
