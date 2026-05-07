"""Unit tests for the RAG program — Retrieve -> Reason -> Verify pipeline.

Tests use ``FunctionClient`` from kaos-llm-client to deterministically
simulate LLM responses without network access. The retrieval layer is
exercised through ``search_corpus`` with real text and BM25 scoring.

Tests cover:
1. ``build_passage_context`` — passage-level URI generation and context
   assembly
2. ``CorpusQA`` — Signature schema validity
3. ``RAG`` pipeline — answer path, insufficient evidence path,
   verification failure with retry, and empty corpus handling
"""

from __future__ import annotations

import json
from typing import Any, cast
from unittest.mock import AsyncMock, patch

from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart

from kaos_llm_core.programs.call import Call
from kaos_llm_core.programs.rag import (
    RAG,
    CorpusQA,
    RAGResult,
    build_passage_context,
)
from kaos_llm_core.signatures import (
    Answer,
    InsufficientEvidence,
)
from kaos_llm_core.signatures.introspection import create_output_model

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DELAWARE_GCL_URI = "doc:delaware-gcl"
DELAWARE_GCL = (
    "The Delaware General Corporation Law requires that a certificate "
    "of incorporation be filed with the Secretary of State.  The filing "
    "fee is $89 and the certificate must include the corporation's name, "
    "its registered office, and the name of its registered agent."
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

CORPUS = {DELAWARE_GCL_URI: DELAWARE_GCL, RFC_2119_URI: RFC_2119}


def _json_response(data: dict[str, Any]) -> ProviderResponse:
    """Build a ProviderResponse from a dict (JSON-encoded text part)."""
    return ProviderResponse.model_construct(
        provider="function",
        model="function-test",
        raw={},
        parts=[ContentPart.model_construct(type="text", text=json.dumps(data))],
        usage=UsageInfo.model_construct(input_tokens=10, output_tokens=5, total_tokens=15),
        stop_reason="end_turn",
        status_code=200,
        response_headers={},
    )


def _make_answer_response(
    value: str,
    statement: str,
    source_uri: str,
    quote: str,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Build a valid Answer response dict for the CorpusQA signature.

    char_span is set to [0, 0] — the RAG Signature tells the LLM
    not to compute offsets (they're tracked server-side via the
    passage URI and retrieval metadata).
    """
    return {
        "result": {
            "kind": "answer",
            "value": value,
            "confidence": confidence,
            "claims": [
                {
                    "statement": statement,
                    "claim_type": "factual",
                    "supporting_spans": [
                        {
                            "source_uri": source_uri,
                            "char_span": [0, 0],
                            "quote": quote,
                        }
                    ],
                    "confidence": confidence,
                }
            ],
        }
    }


def _make_insufficient_evidence_response(reason: str) -> dict[str, Any]:
    """Build an InsufficientEvidence response dict."""
    return {
        "result": {
            "kind": "insufficient_evidence",
            "reason": reason,
            "missing": ["relevant source text"],
        }
    }


def _validated_corpus_qa_result(data: dict[str, Any]) -> Answer[str] | InsufficientEvidence:
    model = create_output_model(CorpusQA)
    validated = cast(Any, model.model_validate(data))
    result = validated.result
    assert isinstance(result, (Answer, InsufficientEvidence))
    return result


# ---------------------------------------------------------------------------
# 1. build_passage_context tests
# ---------------------------------------------------------------------------


class TestBuildPassageContext:
    def test_empty_results(self) -> None:
        ctx, source_map = build_passage_context([], {})
        assert ctx == ""
        assert source_map == {}

    def test_single_passage_gets_passage_uri(self) -> None:
        """Each passage gets its own URI, not a document-level URI."""
        from kaos_nlp_core.retrieval.protocol import RetrievalResult

        results = [
            RetrievalResult(
                text="The filing fee is $89",
                score=5.0,
                doc_id=DELAWARE_GCL_URI,
                metadata={"block_ref": "#/body/3"},
                char_start=100,
                char_end=121,
            )
        ]
        ctx, source_map = build_passage_context(results, CORPUS)

        # Passage URI includes block_ref
        passage_uri = f"{DELAWARE_GCL_URI}#/body/3"
        assert f"=== SOURCE: {passage_uri} ===" in ctx
        assert "The filing fee is $89" in ctx
        # source_map maps passage URI → passage text (NOT full doc)
        assert passage_uri in source_map
        assert source_map[passage_uri] == "The filing fee is $89"
        # Full doc URI should NOT be in source_map
        assert DELAWARE_GCL_URI not in source_map

    def test_multi_source_ordering(self) -> None:
        """Passages from different docs each get their own URI."""
        from kaos_nlp_core.retrieval.protocol import RetrievalResult

        results = [
            RetrievalResult(
                text="The filing fee is $89",
                score=5.0,
                doc_id=DELAWARE_GCL_URI,
                metadata={"block_ref": "#/body/2"},
            ),
            RetrievalResult(
                text="MUST means absolute requirement",
                score=4.0,
                doc_id=RFC_2119_URI,
                metadata={"block_ref": "#/body/1"},
            ),
        ]
        ctx, source_map = build_passage_context(results, CORPUS)

        gcl_uri = f"{DELAWARE_GCL_URI}#/body/2"
        rfc_uri = f"{RFC_2119_URI}#/body/1"
        assert f"=== SOURCE: {gcl_uri} ===" in ctx
        assert f"=== SOURCE: {rfc_uri} ===" in ctx
        # Both passage URIs in source_map
        assert gcl_uri in source_map
        assert rfc_uri in source_map
        # Values are passage text, not full document text
        assert source_map[gcl_uri] == "The filing fee is $89"
        assert source_map[rfc_uri] == "MUST means absolute requirement"

    def test_deterministic_output(self) -> None:
        """Same inputs produce identical output."""
        from kaos_nlp_core.retrieval.protocol import RetrievalResult

        results = [
            RetrievalResult(text="foo", score=2.0, doc_id="a", metadata={"block_ref": "#/body/0"}),
            RetrievalResult(text="bar", score=1.0, doc_id="b", metadata={"block_ref": "#/body/0"}),
        ]
        corpus = {"a": "foo full text", "b": "bar full text"}

        ctx1, map1 = build_passage_context(results, corpus)
        ctx2, map2 = build_passage_context(results, corpus)

        assert ctx1 == ctx2
        assert map1 == map2

    def test_passages_from_same_doc_get_separate_uris(self) -> None:
        """Each passage from the same doc gets its own URI + header."""
        from kaos_nlp_core.retrieval.protocol import RetrievalResult

        results = [
            RetrievalResult(
                text="Passage one",
                score=5.0,
                doc_id="doc:x",
                metadata={"block_ref": "#/body/3"},
                char_start=0,
                char_end=11,
            ),
            RetrievalResult(
                text="Passage two",
                score=3.0,
                doc_id="doc:x",
                metadata={"block_ref": "#/body/1"},
                char_start=0,
                char_end=31,
            ),
        ]
        corpus = {"doc:x": "Passage one ... Passage two"}
        ctx, source_map = build_passage_context(results, corpus)

        # Two separate SOURCE headers (passage-level, not document-level)
        assert "=== SOURCE: doc:x#/body/3 ===" in ctx
        assert "=== SOURCE: doc:x#/body/1 ===" in ctx
        # Both passages present
        assert "Passage one" in ctx
        assert "Passage two" in ctx
        # source_map has passage-level entries
        assert source_map["doc:x#/body/3"] == "Passage one"
        assert source_map["doc:x#/body/1"] == "Passage two"

    def test_fallback_to_char_start_uri(self) -> None:
        """Without block_ref, passage URI falls back to char offset."""
        from kaos_nlp_core.retrieval.protocol import RetrievalResult

        results = [
            RetrievalResult(
                text="some text",
                score=1.0,
                doc_id="doc:y",
                char_start=42,
            )
        ]
        ctx, source_map = build_passage_context(results, {"doc:y": "..."})
        assert "=== SOURCE: doc:y#c42 ===" in ctx
        assert "doc:y#c42" in source_map

    def test_fallback_to_hash_uri(self) -> None:
        """Without block_ref or char_start, passage URI uses text hash."""
        from kaos_nlp_core.retrieval.protocol import RetrievalResult

        results = [RetrievalResult(text="unique text", score=1.0, doc_id="doc:z")]
        ctx, source_map = build_passage_context(results, {"doc:z": "..."})
        # Should have a hash-based URI
        assert "=== SOURCE: doc:z#h" in ctx
        assert len(source_map) == 1

    def test_explicit_passage_uri_disambiguates_shared_block_ref(self) -> None:
        """Sentence-level results can supply distinct URIs for one paragraph block_ref."""
        from kaos_nlp_core.retrieval.protocol import RetrievalResult

        results = [
            RetrievalResult(
                text="Sentence one.",
                score=2.0,
                doc_id="doc:sent",
                metadata={"block_ref": "#/body/7", "passage_uri": "doc:sent#c0"},
                char_start=0,
                char_end=13,
            ),
            RetrievalResult(
                text="Sentence two.",
                score=1.0,
                doc_id="doc:sent",
                metadata={"block_ref": "#/body/7", "passage_uri": "doc:sent#c20"},
                char_start=20,
                char_end=33,
            ),
        ]

        ctx, source_map = build_passage_context(results, {})

        assert "=== SOURCE: doc:sent#c0 ===" in ctx
        assert "=== SOURCE: doc:sent#c20 ===" in ctx
        assert source_map["doc:sent#c0"] == "Sentence one."
        assert source_map["doc:sent#c20"] == "Sentence two."


# ---------------------------------------------------------------------------
# 2. CorpusQA Signature tests
# ---------------------------------------------------------------------------


class TestCorpusQASignature:
    def test_output_model_creates_successfully(self) -> None:
        """The CorpusQA Signature produces a valid output model."""
        model = create_output_model(CorpusQA)
        assert model is not None
        assert "result" in model.model_fields

    def test_signature_has_required_fields(self) -> None:
        """CorpusQA has question and corpus as input fields."""
        from kaos_llm_core.signatures.introspection import get_input_fields

        fields = get_input_fields(CorpusQA)
        assert "question" in fields
        assert "corpus" in fields


# ---------------------------------------------------------------------------
# 3. RAG pipeline tests
# ---------------------------------------------------------------------------


class TestRAGPipeline:
    """Tests for the RAG program using mocked retrieval + LLM."""

    async def test_answer_path(self) -> None:
        """RAG returns a verified answer when the LLM cites the passage correctly."""
        quote = "The filing fee is $89"
        # The passage URI that build_passage_context will generate
        passage_uri = f"{DELAWARE_GCL_URI}#/body/3"

        answer_data = _make_answer_response(
            value="The filing fee is $89.",
            statement="The Delaware GCL filing fee is $89.",
            source_uri=passage_uri,
            quote=quote,
        )

        def llm_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response(answer_data)

        rag = RAG(model="function-test", top_k=5)
        client = FunctionClient(function=llm_fn)
        rag.call._client = client

        from kaos_nlp_core.retrieval.protocol import RetrievalResult

        mock_results = [
            RetrievalResult(
                text=DELAWARE_GCL,
                score=5.0,
                doc_id=DELAWARE_GCL_URI,
                metadata={"block_ref": "#/body/3"},
            )
        ]

        with patch(
            "kaos_content.search.search_corpus",
            new_callable=AsyncMock,
            return_value=mock_results,
        ):
            result = await rag.query(
                question="What is the filing fee?",
                documents={DELAWARE_GCL_URI: DELAWARE_GCL},
            )

        assert isinstance(result, RAGResult)
        assert isinstance(result.grounded_answer, Answer)
        # Verification checks against passage text — quote IS in the passage
        assert result.is_verified
        assert result.confidence > 0.0
        assert len(result.verification_errors) == 0
        assert len(result.retrieved_passages) == 1

    async def test_insufficient_evidence_path(self) -> None:
        """RAG returns InsufficientEvidence when LLM can't answer."""
        ie_data = _make_insufficient_evidence_response(
            "The corpus does not contain information about Mars."
        )

        def llm_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response(ie_data)

        rag = RAG(model="function-test", top_k=5)
        client = FunctionClient(function=llm_fn)
        rag.call._client = client

        from kaos_nlp_core.retrieval.protocol import RetrievalResult

        mock_results = [
            RetrievalResult(
                text=DELAWARE_GCL,
                score=1.0,
                doc_id=DELAWARE_GCL_URI,
                metadata={"block_ref": "#/body/3"},
            )
        ]

        with patch(
            "kaos_content.search.search_corpus",
            new_callable=AsyncMock,
            return_value=mock_results,
        ):
            result = await rag.query(
                question="What is the capital of Mars?",
                documents={DELAWARE_GCL_URI: DELAWARE_GCL},
            )

        assert isinstance(result, RAGResult)
        assert isinstance(result.grounded_answer, InsufficientEvidence)
        assert not result.is_verified
        assert result.confidence == 0.0

    async def test_verification_failure_triggers_retry(self) -> None:
        """When verification fails, RAG retries with feedback."""
        passage_uri = f"{DELAWARE_GCL_URI}#/body/3"

        # First call: fabricated quote that won't verify
        bad_answer = _make_answer_response(
            value="The fee is $89.",
            statement="The fee is $89.",
            source_uri=passage_uri,
            quote="THIS IS A FABRICATED QUOTE THAT DOES NOT EXIST",
        )
        # Second call (retry): good quote
        good_quote = "The filing fee is $89"
        good_answer = _make_answer_response(
            value="The fee is $89.",
            statement="The fee is $89.",
            source_uri=passage_uri,
            quote=good_quote,
        )

        call_count = {"n": 0}

        def llm_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _json_response(bad_answer)
            return _json_response(good_answer)

        rag = RAG(model="function-test", top_k=5, max_retries=2)
        client = FunctionClient(function=llm_fn)
        rag.call._client = client

        from kaos_llm_core.programs.cloning import clone_call

        original_clone = clone_call

        def patched_clone(call: Call) -> Call:
            cloned = original_clone(call)
            cloned._client = client
            return cloned

        from kaos_nlp_core.retrieval.protocol import RetrievalResult

        mock_results = [
            RetrievalResult(
                text=DELAWARE_GCL,
                score=5.0,
                doc_id=DELAWARE_GCL_URI,
                metadata={"block_ref": "#/body/3"},
            )
        ]

        with (
            patch(
                "kaos_content.search.search_corpus",
                new_callable=AsyncMock,
                return_value=mock_results,
            ),
            patch(
                "kaos_llm_core.programs.cloning.clone_call",
                side_effect=patched_clone,
            ),
        ):
            result = await rag.query(
                question="What is the filing fee?",
                documents={DELAWARE_GCL_URI: DELAWARE_GCL},
            )

        # Should have retried and got the good answer
        assert call_count["n"] >= 2
        assert isinstance(result.grounded_answer, Answer)
        # After retry, the good quote verifies against the passage text
        assert result.is_verified

    async def test_empty_corpus(self) -> None:
        """RAG with empty corpus produces InsufficientEvidence."""
        ie_data = _make_insufficient_evidence_response("No documents provided.")

        def llm_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response(ie_data)

        rag = RAG(model="function-test", top_k=5)
        client = FunctionClient(function=llm_fn)
        rag.call._client = client

        with patch(
            "kaos_content.search.search_corpus",
            new_callable=AsyncMock,
            return_value=[],
        ):
            result = await rag.query(question="Anything?", documents={})

        assert isinstance(result.grounded_answer, InsufficientEvidence)

    async def test_named_calls_exposed(self) -> None:
        """RAG exposes inner Call via named_calls for optimizer discovery."""
        rag = RAG(model="function-test")
        assert hasattr(rag, "call")
        assert isinstance(rag.call, Call)

    async def test_query_expansion_path(self) -> None:
        """RAG with a query_expander retrieves for multiple queries and merges."""
        quote = "The filing fee is $89"
        passage_uri = f"{DELAWARE_GCL_URI}#/body/3"

        answer_data = _make_answer_response(
            value="The filing fee is $89.",
            statement="The Delaware GCL filing fee is $89.",
            source_uri=passage_uri,
            quote=quote,
        )

        def llm_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response(answer_data)

        # Create a mock expander that returns known queries
        class MockExpander:
            async def expand(self, question: str) -> list[str]:
                return [question, "filing fee amount", "certificate fee cost dollars"]

        from kaos_nlp_core.retrieval.protocol import RetrievalResult

        mock_results_q1 = [
            RetrievalResult(
                text=DELAWARE_GCL,
                score=5.0,
                doc_id=DELAWARE_GCL_URI,
                metadata={"block_ref": "#/body/3"},
            )
        ]
        mock_results_q2 = [
            RetrievalResult(
                text=DELAWARE_GCL,
                score=3.0,
                doc_id=DELAWARE_GCL_URI,
                metadata={"block_ref": "#/body/3"},
            )
        ]
        mock_results_q3 = [
            RetrievalResult(
                text=RFC_2119,
                score=1.0,
                doc_id=RFC_2119_URI,
                metadata={"block_ref": "#/body/1"},
            )
        ]

        call_count = {"n": 0}

        async def mock_search(docs: Any, query: str, top_k: int = 10) -> list[Any]:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return mock_results_q1
            elif call_count["n"] == 2:
                return mock_results_q2
            else:
                return mock_results_q3

        rag = RAG(
            model="function-test",
            top_k=5,
            query_expander=MockExpander(),
        )
        client = FunctionClient(function=llm_fn)
        rag.call._client = client

        with patch(
            "kaos_content.search.search_corpus",
            new_callable=AsyncMock,
            side_effect=mock_search,
        ):
            result = await rag.query(
                question="What is the filing fee?",
                documents=CORPUS,
            )

        # search_corpus called once per expanded query (3 queries)
        assert call_count["n"] == 3
        assert isinstance(result, RAGResult)
        assert isinstance(result.grounded_answer, Answer)
        # Merged results: same passage from q1/q2 deduplicated (max score=5.0),
        # plus the RFC passage from q3 — should have 2 unique passages.
        assert len(result.retrieved_passages) == 2
        # Should be verified since quote exists in passage text
        assert result.is_verified


# ---------------------------------------------------------------------------
# 4. Codec round-trip tests
# ---------------------------------------------------------------------------


class TestCorpusQACodec:
    def test_answer_round_trip(self) -> None:
        """A valid Answer dict validates through the output model."""
        data = _make_answer_response(
            value="$89",
            statement="fee is $89",
            source_uri="doc:test#/body/3",
            quote="The filing fee is $89",
        )
        validated = _validated_corpus_qa_result(data)
        assert isinstance(validated, Answer)

    def test_insufficient_evidence_round_trip(self) -> None:
        """A valid InsufficientEvidence dict validates."""
        data = _make_insufficient_evidence_response("No info")
        validated = _validated_corpus_qa_result(data)
        assert isinstance(validated, InsufficientEvidence)
