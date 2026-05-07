"""Unit tests for the three-tier grounding primitives.

Covers ``kaos_llm_core.signatures.grounding``:

- Tier 1: Span construction, verification, byte-level checks,
  position fallback, hash tamper detection
- Tier 2: Claim construction, typed evaluation dispatch, temporal
  validity, per-span verification
- Tier 3: Answer[T] with generic payloads, claim-level verification,
  convenience accessors
- InsufficientEvidence with attempted_claims and missing hints
- GroundedAnswer[T] discriminator dispatch via TypeAdapter
- Integration with Signature / create_output_model
- Schema generation with discriminator + min_length constraints
- RefusalPolicy configuration

No codec, no provider, no network.  Codec tests live in
``test_codec_json_grounding.py``; live tests live in
``tests/integration/test_grounding_live.py``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, cast

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from kaos_llm_core.signatures import (
    Answer,
    Claim,
    ClaimError,
    GroundedAnswer,
    InputField,
    InsufficientEvidence,
    OutputField,
    RefusalPolicy,
    Signature,
    Span,
)
from kaos_llm_core.signatures.grounding import _hash_quote
from kaos_llm_core.signatures.introspection import (
    create_output_model,
    signature_to_json_schema,
)

# --- Fixtures ---


CORPUS_TEXT = (
    "The Delaware General Corporation Law requires that a certificate "
    "of incorporation be filed with the Secretary of State.  The filing "
    "fee is $89 and the certificate must include the corporation's name, "
    "its registered office, and the name of its registered agent."
)
CORPUS_URI = "doc:delaware-gcl"
CORPUS = {CORPUS_URI: CORPUS_TEXT}


# --- Tier 1: Span ---


class TestSpanHashUtility:
    def test_hash_is_stable(self) -> None:
        h1 = _hash_quote("hello world")
        h2 = _hash_quote("hello world")
        assert h1 == h2
        assert len(h1) == 16

    def test_hash_differs_on_change(self) -> None:
        assert _hash_quote("hello") != _hash_quote("hello ")


class TestSpanConstruction:
    def test_from_source_computes_quote_and_hash(self) -> None:
        span = Span.from_source(CORPUS_URI, CORPUS_TEXT, (139, 142))
        assert span.quote == "$89"
        assert span.quote_hash == _hash_quote("$89")

    def test_from_source_with_page(self) -> None:
        span = Span.from_source(CORPUS_URI, CORPUS_TEXT, (0, 10), page=3)
        assert span.page == 3

    def test_from_source_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            Span.from_source(CORPUS_URI, "short", (0, 100))

    def test_from_source_rejects_empty_quote(self) -> None:
        with pytest.raises(ValueError):
            Span.from_source(CORPUS_URI, CORPUS_TEXT, (5, 5))

    def test_from_source_rejects_inverted_range(self) -> None:
        # (start, end) with start > end yields empty/invalid quote
        with pytest.raises((ValueError, ValidationError)):
            Span.from_source(CORPUS_URI, CORPUS_TEXT, (10, 5))

    def test_direct_construction(self) -> None:
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 5),
            quote="The D",
            quote_hash=_hash_quote("The D"),
        )
        assert span.quote == "The D"

    def test_frozen(self) -> None:
        span = Span.from_source(CORPUS_URI, CORPUS_TEXT, (0, 5))
        with pytest.raises(ValidationError):
            span.__setattr__("quote", "hacked")

    def test_empty_source_uri_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Span(source_uri="", char_span=(0, 5), quote="x", quote_hash="0" * 16)

    def test_empty_quote_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Span(source_uri="u", char_span=(0, 0), quote="", quote_hash="0" * 16)


class TestSpanVerification:
    def test_verify_success(self) -> None:
        span = Span.from_source(CORPUS_URI, CORPUS_TEXT, (139, 142))
        assert span.verify(CORPUS_TEXT) is True

    def test_verify_wrong_source_fails(self) -> None:
        span = Span.from_source(CORPUS_URI, CORPUS_TEXT, (139, 142))
        assert span.verify("a completely different document") is False

    def test_verify_out_of_range_fails(self) -> None:
        span = Span.from_source(CORPUS_URI, CORPUS_TEXT, (139, 142))
        # Truncate source so the span no longer fits.
        assert span.verify(CORPUS_TEXT[:50]) is False

    def test_quote_hash_is_always_recomputed(self) -> None:
        """Any quote_hash passed in is silently replaced with the true hash.

        This is the LLM-fabrication defense: the model validator
        ignores whatever the caller provides and derives the hash
        from quote.
        """
        bogus = Span(
            source_uri=CORPUS_URI,
            char_span=(139, 142),
            quote="$89",
            quote_hash="deadbeefdeadbeef",  # bogus input
        )
        assert bogus.quote_hash == _hash_quote("$89")
        # And because the hash was recomputed, verify still succeeds.
        assert bogus.verify(CORPUS_TEXT) is True

    def test_substring_strategy_tolerates_position_shift(self) -> None:
        """When positions shift (e.g., source re-parsed), fall back
        via MatchStrategy.SUBSTRING."""
        from kaos_llm_core.signatures import MatchStrategy

        span = Span.from_source(CORPUS_URI, CORPUS_TEXT, (139, 142))  # "$89"
        # Prepend noise to source — positions shift but substring still exists.
        shifted = "Introductory text.  " + CORPUS_TEXT
        assert span.verify(shifted, strategy=MatchStrategy.STRICT) is False
        assert span.verify(shifted, strategy=MatchStrategy.SUBSTRING) is True

    def test_substring_strategy_wrong_source_fails(self) -> None:
        from kaos_llm_core.signatures import MatchStrategy

        span = Span.from_source(CORPUS_URI, CORPUS_TEXT, (139, 142))
        assert span.verify("totally different text", strategy=MatchStrategy.SUBSTRING) is False


# --- Tier 2: Claim ---


class TestClaimConstruction:
    def _span(self) -> Span:
        return Span.from_source(CORPUS_URI, CORPUS_TEXT, (139, 142))

    def test_minimal(self) -> None:
        claim = Claim(
            statement="The filing fee is $89",
            supporting_spans=[self._span()],
        )
        assert claim.claim_type == "factual"  # default
        assert claim.confidence == 1.0  # default
        assert claim.valid_from is None

    def test_all_claim_types(self) -> None:
        for ct in ("factual", "temporal", "normative", "counterfactual", "modal", "quantitative"):
            c = Claim(statement="x", claim_type=ct, supporting_spans=[self._span()])
            assert c.claim_type == ct

    def test_invalid_claim_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Claim(
                statement="x",
                claim_type=cast(Any, "bogus"),
                supporting_spans=[self._span()],
            )

    def test_empty_statement_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Claim(statement="", supporting_spans=[self._span()])

    def test_zero_spans_rejected(self) -> None:
        """No supporting spans → no claim.  This is the core guarantee."""
        with pytest.raises(ValidationError):
            Claim(statement="x", supporting_spans=[])

    def test_temporal_validity(self) -> None:
        claim = Claim(
            statement="X was in effect",
            claim_type="temporal",
            supporting_spans=[self._span()],
            valid_from=datetime(2024, 1, 1),
            valid_to=datetime(2024, 12, 31),
        )
        assert claim.valid_from == datetime(2024, 1, 1)
        assert claim.valid_to == datetime(2024, 12, 31)

    def test_relevance_rationale(self) -> None:
        claim = Claim(
            statement="x",
            supporting_spans=[self._span()],
            relevance="The source explicitly names the fee.",
        )
        assert claim.relevance is not None


class TestClaimVerification:
    def test_verify_success_with_dict_corpus(self) -> None:
        claim = Claim(
            statement="The filing fee is $89",
            supporting_spans=[Span.from_source(CORPUS_URI, CORPUS_TEXT, (139, 142))],
        )
        errors = claim.verify(CORPUS)
        assert errors == []

    def test_verify_success_with_callable_corpus(self) -> None:
        claim = Claim(
            statement="x",
            supporting_spans=[Span.from_source(CORPUS_URI, CORPUS_TEXT, (139, 142))],
        )
        errors = claim.verify(lambda uri: CORPUS[uri])
        assert errors == []

    def test_verify_missing_source(self) -> None:
        claim = Claim(
            statement="x",
            supporting_spans=[Span.from_source("doc:nowhere", "x" * 200, (0, 10))],
        )
        errors = claim.verify(CORPUS)  # doc:nowhere not in corpus
        assert len(errors) == 1
        assert errors[0].reason == "source_missing"

    def test_verify_position_shifted_but_substring_tolerated(self) -> None:
        """When char_span is wrong but the quote still appears in the
        source, verification falls back to substring search and
        reports NO error."""
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 3),  # wrong position — real "$89" is at 139
            quote="$89",
        )
        claim = Claim(statement="x", supporting_spans=[span])
        errors = claim.verify(CORPUS)
        assert errors == []

    def test_verify_quote_mismatch(self) -> None:
        # Build a span with a quote that doesn't match position in source.
        # Hash correct for the quote, but position points to different text.
        fake_quote = "$99"
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(139, 142),  # in CORPUS this is "$89"
            quote=fake_quote,
            quote_hash=_hash_quote(fake_quote),
        )
        claim = Claim(statement="x", supporting_spans=[span])
        errors = claim.verify(CORPUS)
        assert len(errors) == 1
        assert errors[0].reason == "quote_mismatch"

    def test_verify_out_of_range(self) -> None:
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(5000, 5010),
            quote="x" * 10,
            quote_hash=_hash_quote("x" * 10),
        )
        claim = Claim(statement="x", supporting_spans=[span])
        errors = claim.verify(CORPUS)
        assert len(errors) == 1
        assert errors[0].reason == "out_of_range"

    def test_verify_reports_all_failures(self) -> None:
        good = Span.from_source(CORPUS_URI, CORPUS_TEXT, (0, 10))
        bad1 = Span.from_source("doc:nowhere", "y" * 20, (0, 5))
        bad2 = Span(
            source_uri=CORPUS_URI,
            char_span=(139, 142),
            quote="wrong",
            quote_hash=_hash_quote("wrong"),
        )
        claim = Claim(statement="x", supporting_spans=[good, bad1, bad2])
        errors = claim.verify(CORPUS)
        assert len(errors) == 2
        reasons = {e.reason for e in errors}
        assert "source_missing" in reasons
        assert "quote_mismatch" in reasons


# --- Tier 3: Answer[T] ---


def _simple_claim(statement: str = "x", span_range: tuple[int, int] = (0, 10)) -> Claim:
    return Claim(
        statement=statement,
        supporting_spans=[Span.from_source(CORPUS_URI, CORPUS_TEXT, span_range)],
    )


class TestAnswerConstruction:
    def test_answer_str(self) -> None:
        ans = Answer[str](
            value="Paris",
            claims=[_simple_claim("Paris is the capital of France")],
            confidence=0.95,
        )
        assert ans.kind == "answer"
        assert ans.value == "Paris"
        assert len(ans.claims) == 1

    def test_answer_int(self) -> None:
        ans = Answer[int](value=89, claims=[_simple_claim()], confidence=0.9)
        assert ans.value == 89

    def test_answer_list_payload(self) -> None:
        ans = Answer[list[str]](
            value=["name", "office", "agent"],
            claims=[
                _simple_claim("name is required"),
                _simple_claim("office is required"),
                _simple_claim("agent is required"),
            ],
            confidence=0.92,
        )
        assert len(ans.value) == 3
        assert len(ans.claims) == 3

    def test_answer_custom_model(self) -> None:
        class Fee(BaseModel):
            amount: int
            currency: str

        ans = Answer[Fee](
            value=Fee(amount=89, currency="USD"),
            claims=[_simple_claim()],
            confidence=0.95,
        )
        assert ans.value.amount == 89

    def test_zero_claims_rejected(self) -> None:
        """An Answer without claims is not grounded."""
        with pytest.raises(ValidationError):
            Answer[str](value="x", claims=[], confidence=0.9)

    def test_confidence_bounded(self) -> None:
        with pytest.raises(ValidationError):
            Answer[str](value="x", claims=[_simple_claim()], confidence=1.1)
        with pytest.raises(ValidationError):
            Answer[str](value="x", claims=[_simple_claim()], confidence=-0.1)


class TestAnswerAccessors:
    def test_spans_flattens(self) -> None:
        ans = Answer[str](
            value="x",
            claims=[
                _simple_claim("c1", (0, 10)),
                _simple_claim("c2", (10, 20)),
                _simple_claim("c3", (20, 30)),
            ],
            confidence=0.9,
        )
        assert len(ans.spans) == 3

    def test_sources_deduplicates(self) -> None:
        ans = Answer[str](
            value="x",
            claims=[_simple_claim("c1"), _simple_claim("c2")],
            confidence=0.9,
        )
        # Both claims cite the same source URI
        assert ans.sources == {CORPUS_URI}


class TestAnswerVerification:
    def test_verify_all_good(self) -> None:
        ans = Answer[str](
            value="$89",
            claims=[
                Claim(
                    statement="filing fee",
                    supporting_spans=[Span.from_source(CORPUS_URI, CORPUS_TEXT, (139, 142))],
                )
            ],
            confidence=0.95,
        )
        assert ans.verify(CORPUS) == []

    def test_verify_catches_fabricated_quote(self) -> None:
        """The key regression test: a Span whose quote doesn't match
        its char_span in the corpus is caught by verify().

        An LLM trying to fabricate a citation either fabricates the
        quote (caught here) or the char_span (caught the same way
        since source_text[span] != quote).  Either failure mode
        produces a quote_mismatch error.
        """
        fabricated_span = Span(
            source_uri=CORPUS_URI,
            char_span=(139, 142),  # real text at this position is "$89"
            quote="$1000",  # but we claim it says $1000
        )
        ans = Answer[str](
            value="$1000",
            claims=[Claim(statement="fee", supporting_spans=[fabricated_span])],
            confidence=0.95,
        )
        errors = ans.verify(CORPUS)
        assert len(errors) == 1
        assert errors[0].reason == "quote_mismatch"
        assert errors[0].claim_index == 0
        assert errors[0].span_index == 0

    def test_verify_reports_errors_per_claim(self) -> None:
        good_claim = _simple_claim("good", (0, 10))
        bad_claim = Claim(
            statement="bad",
            supporting_spans=[
                Span(
                    source_uri=CORPUS_URI,
                    char_span=(0, 5),
                    quote="wrong",
                    quote_hash=_hash_quote("wrong"),
                )
            ],
        )
        ans = Answer[str](
            value="x",
            claims=[good_claim, bad_claim],
            confidence=0.9,
        )
        errors = ans.verify(CORPUS)
        assert len(errors) == 1
        assert errors[0].claim_index == 1  # the bad claim
        assert isinstance(errors[0], ClaimError)


# --- InsufficientEvidence ---


class TestInsufficientEvidence:
    def test_minimal(self) -> None:
        ie = InsufficientEvidence(reason="no corpus")
        assert ie.kind == "insufficient_evidence"
        assert ie.attempted_claims == []
        assert ie.missing == []
        assert ie.what_would_resolve is None

    def test_with_attempted_claims(self) -> None:
        ie = InsufficientEvidence(
            reason="corpus lacks LLC franchise tax info",
            attempted_claims=[_simple_claim("LLC franchise tax is $X")],
            missing=["Delaware LLC franchise tax schedule", "effective date"],
            what_would_resolve="Provide the Delaware LLC Act Title 6 Chapter 18.",
        )
        assert len(ie.attempted_claims) == 1
        assert len(ie.missing) == 2
        assert ie.what_would_resolve is not None

    def test_empty_reason_rejected(self) -> None:
        with pytest.raises(ValidationError):
            InsufficientEvidence(reason="")


# --- Discriminator dispatch ---


class TestDiscriminatorDispatch:
    def _adapter(self) -> TypeAdapter:
        return TypeAdapter(GroundedAnswer[str])

    def test_dispatch_to_answer(self) -> None:
        adapter = self._adapter()
        result = adapter.validate_python(
            {
                "kind": "answer",
                "value": "Paris",
                "confidence": 0.9,
                "claims": [
                    {
                        "statement": "Paris is the capital",
                        "supporting_spans": [
                            {
                                "source_uri": CORPUS_URI,
                                "char_span": (0, 10),
                                "quote": "The Delawa",
                                "quote_hash": _hash_quote("The Delawa"),
                            }
                        ],
                    }
                ],
            }
        )
        assert isinstance(result, Answer)
        assert result.value == "Paris"

    def test_dispatch_to_insufficient(self) -> None:
        adapter = self._adapter()
        result = adapter.validate_python({"kind": "insufficient_evidence", "reason": "no context"})
        assert isinstance(result, InsufficientEvidence)

    def test_unknown_kind_rejected(self) -> None:
        adapter = self._adapter()
        with pytest.raises(ValidationError):
            adapter.validate_python({"kind": "maybe"})

    def test_missing_kind_rejected(self) -> None:
        adapter = self._adapter()
        with pytest.raises(ValidationError):
            adapter.validate_python({"value": "x", "claims": [], "confidence": 0.5})

    def test_generic_param_enforces_value_type(self) -> None:
        adapter = TypeAdapter(GroundedAnswer[int])
        with pytest.raises(ValidationError):
            adapter.validate_python(
                {
                    "kind": "answer",
                    "value": "not an int",
                    "confidence": 0.9,
                    "claims": [
                        {
                            "statement": "x",
                            "supporting_spans": [
                                {
                                    "source_uri": "u",
                                    "char_span": (0, 1),
                                    "quote": "x",
                                    "quote_hash": _hash_quote("x"),
                                }
                            ],
                        }
                    ],
                }
            )


# --- JSON round-trip ---


class TestJSONRoundTrip:
    def test_answer_round_trip(self) -> None:
        original = Answer[str](
            value="$89",
            claims=[
                Claim(
                    statement="filing fee",
                    claim_type="factual",
                    supporting_spans=[Span.from_source(CORPUS_URI, CORPUS_TEXT, (139, 142))],
                    relevance="source states the amount",
                )
            ],
            confidence=0.95,
        )
        dumped = original.model_dump_json()
        restored = Answer[str].model_validate_json(dumped)
        assert restored == original

    def test_temporal_claim_round_trip(self) -> None:
        ts = datetime(2024, 3, 1)
        claim = Claim(
            statement="X was in effect",
            claim_type="temporal",
            supporting_spans=[Span.from_source(CORPUS_URI, CORPUS_TEXT, (0, 50))],
            valid_from=ts,
        )
        dumped = claim.model_dump_json()
        restored = Claim.model_validate_json(dumped)
        assert restored.valid_from == ts
        assert restored.claim_type == "temporal"

    def test_union_round_trip_via_adapter(self) -> None:
        adapter = TypeAdapter(GroundedAnswer[str])
        original = Answer[str](
            value="x",
            claims=[_simple_claim()],
            confidence=0.8,
        )
        dumped = adapter.dump_json(original)
        restored = adapter.validate_json(dumped)
        assert isinstance(restored, Answer)


# --- Signature integration ---


class SampleQA(Signature):
    """Answer the question from the corpus, or refuse."""

    question: str = InputField(description="user's question")
    corpus: str = InputField(description="source text")
    result: GroundedAnswer[str] = OutputField(description="cited answer or refusal")


def _validated_sample_result(data: dict[str, Any]) -> GroundedAnswer[str]:
    model = create_output_model(SampleQA)
    validated = cast(Any, model.model_validate(data))
    result = validated.result
    assert isinstance(result, (Answer, InsufficientEvidence))
    return cast(GroundedAnswer[str], result)


class TestSignatureIntegration:
    def test_create_output_model(self) -> None:
        model = create_output_model(SampleQA)
        assert model.__name__ == "SampleQAOutput"

    def test_validate_answer_variant(self) -> None:
        validated = _validated_sample_result(
            {
                "result": {
                    "kind": "answer",
                    "value": "$89",
                    "confidence": 0.95,
                    "claims": [
                        {
                            "statement": "filing fee is $89",
                            "claim_type": "factual",
                            "supporting_spans": [
                                {
                                    "source_uri": CORPUS_URI,
                                    "char_span": (139, 142),
                                    "quote": "$89",
                                    "quote_hash": _hash_quote("$89"),
                                }
                            ],
                            "relevance": "source explicitly names the fee",
                        }
                    ],
                }
            }
        )
        assert isinstance(validated, Answer)
        assert validated.claims[0].claim_type == "factual"

    def test_validate_insufficient_variant(self) -> None:
        validated = _validated_sample_result(
            {
                "result": {
                    "kind": "insufficient_evidence",
                    "reason": "corpus silent on LLC tax",
                    "missing": ["LLC franchise tax schedule"],
                }
            }
        )
        assert isinstance(validated, InsufficientEvidence)
        assert "LLC" in validated.missing[0]

    def test_schema_has_discriminator_and_span_constraints(self) -> None:
        schema = signature_to_json_schema(SampleQA)
        defs = schema.get("$defs", {})

        # Find GroundedAnswer alias
        grounded = next((b for n, b in defs.items() if n.startswith("GroundedAnswer")), None)
        assert grounded is not None
        assert grounded.get("discriminator", {}).get("propertyName") == "kind"

        # Answer variant should require min 1 claim
        answer_def = next((b for n, b in defs.items() if n.startswith("Answer")), None)
        assert answer_def is not None
        assert "claims" in answer_def.get("required", [])
        claims_prop = answer_def["properties"]["claims"]
        assert claims_prop.get("minItems") == 1

        # Claim should require min 1 supporting span
        claim_def = defs.get("Claim")
        assert claim_def is not None
        spans_prop = claim_def["properties"]["supporting_spans"]
        assert spans_prop.get("minItems") == 1


# --- RefusalPolicy ---


class TestRefusalPolicy:
    def test_defaults(self) -> None:
        p = RefusalPolicy()
        assert p.min_confidence == 0.7
        assert p.require_verification is False
        assert p.min_spans_per_claim == 1
        assert p.calibration == "self_report"

    def test_override(self) -> None:
        p = RefusalPolicy(
            min_confidence=0.85,
            require_verification=True,
            min_spans_per_claim=2,
            calibration="ensemble",
        )
        assert p.require_verification is True
        assert p.min_spans_per_claim == 2

    def test_invalid_calibration(self) -> None:
        with pytest.raises(ValidationError):
            RefusalPolicy(calibration=cast(Any, "guess"))

    def test_min_spans_at_least_one(self) -> None:
        with pytest.raises(ValidationError):
            RefusalPolicy(min_spans_per_claim=0)
