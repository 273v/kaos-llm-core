"""Verify that three-tier GroundedAnswer round-trips through JSONCodec.

The codec's ``decode`` returns a raw dict.  The caller (``Call``)
then runs ``output_model.model_validate(output_dict)`` — that is where
the discriminated-union dispatch actually happens.  These tests
exercise both steps end-to-end without any network:

1. ``JSONCodec.decode(signature, response)`` parses LLM JSON text.
2. ``create_output_model(signature).model_validate(...)`` validates
   the dict into concrete ``Answer[T]`` / ``InsufficientEvidence``
   instances, including the nested ``Claim`` / ``Span`` types.

We also confirm the *encoded* system prompt carries the discriminator
schema so the LLM is actually told about all three tiers.
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest
from kaos_llm_client import ProviderResponse, UsageInfo
from kaos_llm_client.types import ContentPart
from pydantic import ValidationError

from kaos_llm_core.codecs.json_codec import JSONCodec
from kaos_llm_core.signatures import (
    Answer,
    GroundedAnswer,
    InputField,
    InsufficientEvidence,
    OutputField,
    Signature,
)
from kaos_llm_core.signatures.grounding import _hash_quote
from kaos_llm_core.signatures.introspection import create_output_model

CORPUS_URI = "doc:delaware-gcl"
CORPUS_TEXT = (
    "The Delaware General Corporation Law requires that a certificate "
    "of incorporation be filed with the Secretary of State.  The filing "
    "fee is $89 and the certificate must include the corporation's name, "
    "its registered office, and the name of its registered agent."
)


class CorpusQA(Signature):
    """Answer the question from the corpus, or refuse.

    Return ``kind: answer`` with a value, claims, and confidence,
    or ``kind: insufficient_evidence`` with a reason.  Each claim
    must cite at least one span from the corpus with exact quote
    and char positions.
    """

    question: str = InputField(description="the question")
    corpus: str = InputField(description="source text")
    result: GroundedAnswer[str] = OutputField(description="cited answer or refusal")


def _make_response(text: str) -> ProviderResponse:
    return ProviderResponse.model_construct(
        provider="test",
        model="test-model",
        raw={},
        parts=[ContentPart.model_construct(type="text", text=text)],
        usage=UsageInfo.model_construct(input_tokens=10, output_tokens=5, total_tokens=15),
        stop_reason="end_turn",
        status_code=200,
        response_headers={},
    )


def _span_payload(uri: str, char_span: tuple[int, int], source_text: str) -> dict:
    start, end = char_span
    quote = source_text[start:end]
    return {
        "source_uri": uri,
        "char_span": [start, end],
        "quote": quote,
        "quote_hash": _hash_quote(quote),
    }


def _validated_result(decoded: dict[str, Any]) -> GroundedAnswer[str]:
    model = create_output_model(CorpusQA)
    validated = cast(Any, model.model_validate(decoded))
    result = validated.result
    assert isinstance(result, (Answer, InsufficientEvidence))
    return cast(GroundedAnswer[str], result)


class TestEncodedPromptCarriesAllThreeTiers:
    """The system prompt must describe Span, Claim, and Answer shapes."""

    def test_system_prompt_contains_three_tier_schema(self) -> None:
        codec = JSONCodec()
        messages = codec.encode(CorpusQA, {"question": "What is X?", "corpus": "some text"})
        system = messages[0]
        content = system["content"]
        # Discriminator between Answer and InsufficientEvidence
        assert "discriminator" in content
        assert '"answer"' in content
        assert '"insufficient_evidence"' in content
        # Claim-level fields
        assert "claims" in content
        assert "claim_type" in content
        assert "supporting_spans" in content
        # Span-level fields
        assert "quote" in content
        assert "quote_hash" in content
        assert "char_span" in content


class TestDecodeAnswerVariant:
    def test_single_claim_answer(self) -> None:
        codec = JSONCodec()
        payload = {
            "result": {
                "kind": "answer",
                "value": "$89",
                "confidence": 0.95,
                "claims": [
                    {
                        "statement": "The Delaware filing fee is $89",
                        "claim_type": "factual",
                        "supporting_spans": [_span_payload(CORPUS_URI, (139, 142), CORPUS_TEXT)],
                        "relevance": "source explicitly names the fee",
                    }
                ],
            }
        }
        response = _make_response(json.dumps(payload))
        decoded = codec.decode(CorpusQA, response)
        validated = _validated_result(decoded)

        assert isinstance(validated, Answer)
        assert validated.value == "$89"
        assert len(validated.claims) == 1
        assert validated.claims[0].claim_type == "factual"

        # And critically: the citation actually verifies against the corpus.
        errors = validated.verify({CORPUS_URI: CORPUS_TEXT})
        assert errors == []

    def test_multi_claim_answer(self) -> None:
        codec = JSONCodec()
        requirements_span_text = (
            "the corporation's name, its registered office, and the name of its registered agent"
        )
        requirements_start = CORPUS_TEXT.find(requirements_span_text)
        assert requirements_start != -1

        payload = {
            "result": {
                "kind": "answer",
                "value": "name, office, agent",
                "confidence": 0.9,
                "claims": [
                    {
                        "statement": "The certificate must include the name",
                        "claim_type": "factual",
                        "supporting_spans": [
                            _span_payload(
                                CORPUS_URI,
                                (requirements_start, requirements_start + 23),
                                CORPUS_TEXT,
                            )
                        ],
                    },
                    {
                        "statement": "The certificate must include the office",
                        "claim_type": "factual",
                        "supporting_spans": [
                            _span_payload(
                                CORPUS_URI,
                                (requirements_start + 25, requirements_start + 48),
                                CORPUS_TEXT,
                            )
                        ],
                    },
                ],
            }
        }
        response = _make_response(json.dumps(payload))
        decoded = codec.decode(CorpusQA, response)
        validated = _validated_result(decoded)

        assert isinstance(validated, Answer)
        assert len(validated.claims) == 2
        # Both citations should verify byte-for-byte.
        errors = validated.verify({CORPUS_URI: CORPUS_TEXT})
        assert errors == []

    def test_temporal_claim(self) -> None:
        codec = JSONCodec()
        payload = {
            "result": {
                "kind": "answer",
                "value": "The fee was $89",
                "confidence": 0.9,
                "claims": [
                    {
                        "statement": "Fee was $89 as of the source date",
                        "claim_type": "temporal",
                        "supporting_spans": [_span_payload(CORPUS_URI, (139, 142), CORPUS_TEXT)],
                        "valid_from": "2024-01-01T00:00:00",
                    }
                ],
            }
        }
        response = _make_response(json.dumps(payload))
        decoded = codec.decode(CorpusQA, response)
        validated = _validated_result(decoded)

        assert isinstance(validated, Answer)
        assert validated.claims[0].claim_type == "temporal"
        assert validated.claims[0].valid_from is not None


class TestDecodeInsufficientVariant:
    def test_refusal_with_attempted_claims_and_missing(self) -> None:
        codec = JSONCodec()
        payload = {
            "result": {
                "kind": "insufficient_evidence",
                "reason": "corpus does not mention LLC franchise tax",
                "missing": ["Delaware LLC franchise tax schedule"],
                "what_would_resolve": "Include Title 6 Chapter 18 of Delaware Code",
            }
        }
        response = _make_response(json.dumps(payload))
        decoded = codec.decode(CorpusQA, response)
        validated = _validated_result(decoded)

        assert isinstance(validated, InsufficientEvidence)
        assert "LLC" in validated.missing[0]
        assert validated.what_would_resolve is not None

    def test_refusal_minimal(self) -> None:
        codec = JSONCodec()
        payload = {"result": {"kind": "insufficient_evidence", "reason": "no context"}}
        response = _make_response(json.dumps(payload))
        decoded = codec.decode(CorpusQA, response)
        validated = _validated_result(decoded)

        assert isinstance(validated, InsufficientEvidence)
        assert validated.missing == []


class TestDecodeMalformed:
    """Malformed answers must raise ValidationError so Call's retry loop catches them."""

    def test_answer_without_claims_rejected(self) -> None:
        codec = JSONCodec()
        payload = {
            "result": {
                "kind": "answer",
                "value": "x",
                "confidence": 0.9,
                "claims": [],
            }
        }
        response = _make_response(json.dumps(payload))
        decoded = codec.decode(CorpusQA, response)
        model = create_output_model(CorpusQA)
        with pytest.raises(ValidationError):
            model.model_validate(decoded)

    def test_claim_without_supporting_spans_rejected(self) -> None:
        codec = JSONCodec()
        payload = {
            "result": {
                "kind": "answer",
                "value": "x",
                "confidence": 0.9,
                "claims": [
                    {
                        "statement": "uncited claim",
                        "supporting_spans": [],
                    }
                ],
            }
        }
        response = _make_response(json.dumps(payload))
        decoded = codec.decode(CorpusQA, response)
        model = create_output_model(CorpusQA)
        with pytest.raises(ValidationError):
            model.model_validate(decoded)

    def test_span_without_quote_rejected(self) -> None:
        codec = JSONCodec()
        payload = {
            "result": {
                "kind": "answer",
                "value": "x",
                "confidence": 0.9,
                "claims": [
                    {
                        "statement": "x",
                        "supporting_spans": [
                            {
                                "source_uri": "u",
                                "char_span": [0, 5],
                                "quote": "",  # empty quote invalid
                                "quote_hash": "x" * 16,
                            }
                        ],
                    }
                ],
            }
        }
        response = _make_response(json.dumps(payload))
        decoded = codec.decode(CorpusQA, response)
        model = create_output_model(CorpusQA)
        with pytest.raises(ValidationError):
            model.model_validate(decoded)

    def test_invalid_claim_type_rejected(self) -> None:
        codec = JSONCodec()
        payload = {
            "result": {
                "kind": "answer",
                "value": "x",
                "confidence": 0.9,
                "claims": [
                    {
                        "statement": "x",
                        "claim_type": "bogus",
                        "supporting_spans": [_span_payload(CORPUS_URI, (0, 10), CORPUS_TEXT)],
                    }
                ],
            }
        }
        response = _make_response(json.dumps(payload))
        decoded = codec.decode(CorpusQA, response)
        model = create_output_model(CorpusQA)
        with pytest.raises(ValidationError):
            model.model_validate(decoded)

    def test_unknown_kind_rejected(self) -> None:
        codec = JSONCodec()
        payload = {"result": {"kind": "maybe"}}
        response = _make_response(json.dumps(payload))
        decoded = codec.decode(CorpusQA, response)
        model = create_output_model(CorpusQA)
        with pytest.raises(ValidationError):
            model.model_validate(decoded)
