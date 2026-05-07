"""Unit tests for the kaos_llm_core.metrics.retrieval module.

Tests cover all five retrieval metrics:
- faithfulness (structural and fallback paths)
- context_precision
- context_recall
- answer_relevancy (mocked embeddings)
- ragas_composite
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from kaos_llm_core.metrics.retrieval import (
    DEFAULT_RAGAS_WEIGHTS,
    _answer_text,
    _extract_refs,
    answer_relevancy,
    context_precision,
    context_recall,
    expansion_lift,
    faithfulness,
    ragas_composite,
    refusal_rate,
)
from kaos_llm_core.signatures.grounding import (
    Answer,
    Claim,
    InsufficientEvidence,
    MatchStrategy,
    Span,
)

# ---------------------------------------------------------------------------
# Fixtures: reusable corpus and grounded answer objects
# ---------------------------------------------------------------------------

SOURCE_URI = "doc://test/source-1"
SOURCE_TEXT = (
    "The Supreme Court held in Marbury v. Madison (1803) that the judiciary "
    "has the power of judicial review. This landmark decision established "
    "the principle that federal courts can declare legislative and executive "
    "acts unconstitutional."
)

CORPUS: dict[str, str] = {SOURCE_URI: SOURCE_TEXT}


def _callable_name(value: object) -> str:
    name = getattr(value, "__name__", type(value).__name__)
    return name if isinstance(name, str) else type(value).__name__


def _make_span(quote: str, source_uri: str = SOURCE_URI) -> Span:
    """Build a Span with correct char_span from the source text."""
    start = SOURCE_TEXT.index(quote)
    return Span(
        source_uri=source_uri,
        char_span=(start, start + len(quote)),
        quote=quote,
    )


def _make_answer(claims: list[Claim], value: str = "test answer") -> Answer[str]:
    """Build an Answer with the given claims."""
    return Answer[str](
        kind="answer",
        value=value,
        claims=claims,
        confidence=0.95,
    )


def _make_claim(quote: str, statement: str | None = None) -> Claim:
    """Build a single-span factual Claim."""
    return Claim(
        statement=statement or f"The source says: {quote[:40]}...",
        claim_type="factual",
        supporting_spans=[_make_span(quote)],
        confidence=1.0,
    )


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


class TestExtractRefs:
    def test_from_dict(self) -> None:
        assert _extract_refs({"refs": ["a", "b"]}, "refs") == {"a", "b"}

    def test_from_dict_set(self) -> None:
        assert _extract_refs({"refs": {"a", "b"}}, "refs") == {"a", "b"}

    def test_from_dict_missing(self) -> None:
        assert _extract_refs({"other": [1]}, "refs") is None

    def test_from_none(self) -> None:
        assert _extract_refs(None, "refs") is None

    def test_from_attr_object(self) -> None:
        class Obj:
            def __init__(self) -> None:
                self.refs = ["x", "y"]

        assert _extract_refs(Obj(), "refs") == {"x", "y"}

    def test_single_string(self) -> None:
        assert _extract_refs({"refs": "single"}, "refs") == {"single"}


class TestAnswerText:
    def test_string(self) -> None:
        assert _answer_text("hello") == "hello"

    def test_none(self) -> None:
        assert _answer_text(None) is None

    def test_dict_answer_key(self) -> None:
        assert _answer_text({"answer": "test"}) == "test"

    def test_dict_value_key(self) -> None:
        assert _answer_text({"value": "test"}) == "test"

    def test_answer_object(self) -> None:
        claim = _make_claim("judicial review")
        answer = _make_answer([claim], value="the answer")
        assert _answer_text(answer) == "the answer"

    def test_insufficient_evidence(self) -> None:
        ie = InsufficientEvidence(reason="no data found")
        assert _answer_text(ie) == "no data found"


# ---------------------------------------------------------------------------
# Faithfulness tests
# ---------------------------------------------------------------------------


class TestFaithfulnessStructural:
    """Test the structural (zero-LLM-cost) faithfulness path."""

    def test_all_claims_verified(self) -> None:
        """All spans match the corpus -- faithfulness = 1.0."""
        claim1 = _make_claim("Marbury v. Madison (1803)")
        claim2 = _make_claim("power of judicial review")
        answer = _make_answer([claim1, claim2])

        metric = faithfulness(CORPUS)
        score = metric(answer, {})
        assert score == 1.0

    def test_one_claim_fails(self) -> None:
        """One of two claims has a bad span -- faithfulness = 0.5."""
        good_claim = _make_claim("Marbury v. Madison (1803)")
        bad_span = Span(
            source_uri=SOURCE_URI,
            char_span=(0, 10),
            quote="FABRICATED QUOTE THAT DOES NOT EXIST",
        )
        bad_claim = Claim(
            statement="A fabricated assertion",
            claim_type="factual",
            supporting_spans=[bad_span],
            confidence=1.0,
        )
        answer = _make_answer([good_claim, bad_claim])

        metric = faithfulness(CORPUS)
        score = metric(answer, {})
        assert score == pytest.approx(0.5)

    def test_all_claims_fail(self) -> None:
        """All spans are fabricated -- faithfulness = 0.0."""
        bad_span = Span(
            source_uri=SOURCE_URI,
            char_span=(0, 5),
            quote="NOPE NOT IN SOURCE",
        )
        bad_claim = Claim(
            statement="Bad claim",
            claim_type="factual",
            supporting_spans=[bad_span],
            confidence=1.0,
        )
        answer = _make_answer([bad_claim])

        metric = faithfulness(CORPUS)
        score = metric(answer, {})
        assert score == 0.0

    def test_source_missing(self) -> None:
        """Span references a source_uri not in the corpus -- claim fails."""
        missing_span = Span(
            source_uri="doc://nonexistent",
            char_span=(0, 5),
            quote="hello",
        )
        claim = Claim(
            statement="test",
            claim_type="factual",
            supporting_spans=[missing_span],
            confidence=1.0,
        )
        answer = _make_answer([claim])

        metric = faithfulness(CORPUS)
        score = metric(answer, {})
        assert score == 0.0

    def test_single_claim_single_span(self) -> None:
        """Simple case: one claim, one span, verifies correctly."""
        claim = _make_claim("landmark decision")
        answer = _make_answer([claim])

        metric = faithfulness(CORPUS)
        score = metric(answer, {})
        assert score == 1.0

    def test_multiple_spans_per_claim(self) -> None:
        """A claim with multiple spans; all must verify for the claim to pass."""
        span1 = _make_span("Marbury v. Madison (1803)")
        span2 = _make_span("power of judicial review")
        claim = Claim(
            statement="Multi-span claim about judicial review",
            claim_type="factual",
            supporting_spans=[span1, span2],
            confidence=1.0,
        )
        answer = _make_answer([claim])

        metric = faithfulness(CORPUS)
        score = metric(answer, {})
        assert score == 1.0

    def test_multiple_spans_partial_fail(self) -> None:
        """A claim with 2 spans where one fails -- the claim is considered failed."""
        good_span = _make_span("Marbury v. Madison (1803)")
        bad_span = Span(
            source_uri=SOURCE_URI,
            char_span=(0, 3),
            quote="THIS IS NOT IN THE SOURCE",
        )
        claim = Claim(
            statement="Partially supported claim",
            claim_type="factual",
            supporting_spans=[good_span, bad_span],
            confidence=1.0,
        )
        answer = _make_answer([claim])

        metric = faithfulness(CORPUS)
        score = metric(answer, {})
        assert score == 0.0

    def test_three_claims_two_verified(self) -> None:
        """3 claims, 2 good, 1 bad -- faithfulness = 2/3."""
        good1 = _make_claim("Marbury v. Madison (1803)")
        good2 = _make_claim("landmark decision")
        bad_span = Span(
            source_uri=SOURCE_URI,
            char_span=(0, 5),
            quote="FABRICATED",
        )
        bad = Claim(
            statement="Bad",
            claim_type="factual",
            supporting_spans=[bad_span],
            confidence=1.0,
        )
        answer = _make_answer([good1, good2, bad])

        metric = faithfulness(CORPUS)
        score = metric(answer, {})
        assert score == pytest.approx(2.0 / 3.0)


class TestFaithfulnessStrategies:
    """Test that strategy escalation works in the faithfulness metric."""

    def test_substring_strategy(self) -> None:
        """Span has wrong char_span but correct quote -- SUBSTRING matches."""
        span = Span(
            source_uri=SOURCE_URI,
            char_span=(999, 999 + len("judicial review")),  # wrong position
            quote="judicial review",
        )
        claim = Claim(
            statement="test",
            claim_type="factual",
            supporting_spans=[span],
            confidence=1.0,
        )
        answer = _make_answer([claim])

        # With only STRICT, this should fail
        metric_strict = faithfulness(CORPUS, strategies=(MatchStrategy.STRICT,))
        assert metric_strict(answer, {}) == 0.0

        # With STRICT + SUBSTRING, it should pass
        metric_escalate = faithfulness(
            CORPUS, strategies=(MatchStrategy.STRICT, MatchStrategy.SUBSTRING)
        )
        assert metric_escalate(answer, {}) == 1.0

    def test_case_insensitive_strategy(self) -> None:
        """Span has case mismatch -- CASE_INSENSITIVE matches."""
        span = Span(
            source_uri=SOURCE_URI,
            char_span=(0, 100),
            quote="MARBURY V. MADISON",
        )
        claim = Claim(
            statement="test",
            claim_type="factual",
            supporting_spans=[span],
            confidence=1.0,
        )
        answer = _make_answer([claim])

        metric = faithfulness(
            CORPUS,
            strategies=(
                MatchStrategy.STRICT,
                MatchStrategy.SUBSTRING,
                MatchStrategy.CASE_INSENSITIVE,
            ),
        )
        assert metric(answer, {}) == 1.0


class TestFaithfulnessEdgeCases:
    """Edge cases for the faithfulness metric."""

    def test_insufficient_evidence_is_faithful(self) -> None:
        """InsufficientEvidence should return 1.0 (refusal is faithful)."""
        ie = InsufficientEvidence(reason="Cannot determine from given context")
        metric = faithfulness(CORPUS)
        assert metric(ie, {}) == 1.0

    def test_plain_string_no_judge_model(self) -> None:
        """Plain string without LLM judge model returns 0.0 with warning."""
        metric = faithfulness(CORPUS)
        score = metric("some plain text answer", {})
        assert score == 0.0

    def test_none_prediction(self) -> None:
        """None prediction returns 0.0."""
        metric = faithfulness(CORPUS)
        assert metric(None, {}) == 0.0

    def test_no_corpus_with_answer(self) -> None:
        """Answer prediction without corpus returns 0.0 with warning."""
        claim = _make_claim("judicial review")
        answer = _make_answer([claim])
        metric = faithfulness(None)  # no corpus
        assert metric(answer, {}) == 0.0

    def test_callable_corpus(self) -> None:
        """Corpus can be a callable instead of a dict."""

        def lookup(uri: str) -> str:
            if uri == SOURCE_URI:
                return SOURCE_TEXT
            raise KeyError(uri)

        claim = _make_claim("Marbury v. Madison (1803)")
        answer = _make_answer([claim])

        metric = faithfulness(lookup)
        assert metric(answer, {}) == 1.0

    def test_metric_name(self) -> None:
        """Metric function should have a descriptive __name__."""
        metric = faithfulness(CORPUS)
        assert _callable_name(metric) == "faithfulness"


# ---------------------------------------------------------------------------
# Context Precision tests
# ---------------------------------------------------------------------------


class TestContextPrecision:
    def test_perfect_precision(self) -> None:
        """All retrieved refs are relevant -- precision = 1.0."""
        metric = context_precision()
        pred = {"retrieved_refs": ["a", "b", "c"]}
        gold = {"relevant_refs": ["a", "b", "c", "d"]}
        assert metric(pred, gold) == 1.0

    def test_half_precision(self) -> None:
        """2 of 4 retrieved are relevant -- precision = 0.5."""
        metric = context_precision()
        pred = {"retrieved_refs": ["a", "b", "x", "y"]}
        gold = {"relevant_refs": ["a", "b", "c"]}
        assert metric(pred, gold) == 0.5

    def test_zero_precision(self) -> None:
        """No retrieved refs are relevant -- precision = 0.0."""
        metric = context_precision()
        pred = {"retrieved_refs": ["x", "y"]}
        gold = {"relevant_refs": ["a", "b"]}
        assert metric(pred, gold) == 0.0

    def test_empty_retrieved(self) -> None:
        """Empty retrieved set -- precision = 0.0."""
        metric = context_precision()
        pred = {"retrieved_refs": []}
        gold = {"relevant_refs": ["a"]}
        assert metric(pred, gold) == 0.0

    def test_missing_retrieved(self) -> None:
        """No retrieved_refs on prediction -- returns 0.0."""
        metric = context_precision()
        assert metric({}, {"relevant_refs": ["a"]}) == 0.0

    def test_missing_relevant(self) -> None:
        """No relevant_refs in gold -- returns 0.0."""
        metric = context_precision()
        pred = {"retrieved_refs": ["a"]}
        assert metric(pred, {}) == 0.0

    def test_retrieved_from_gold(self) -> None:
        """retrieved_refs can be in gold (for eval setups that store both)."""
        metric = context_precision()
        gold = {
            "retrieved_refs": ["a", "b", "x"],
            "relevant_refs": ["a", "b"],
        }
        # prediction has no retrieved_refs, falls back to gold
        assert metric({}, gold) == pytest.approx(2.0 / 3.0)

    def test_attr_object_prediction(self) -> None:
        """Prediction can be an attribute object (e.g. a Pydantic model)."""

        class Pred:
            def __init__(self) -> None:
                self.retrieved_refs = ["a", "b"]

        metric = context_precision()
        gold = {"relevant_refs": ["a", "c"]}
        assert metric(Pred(), gold) == 0.5

    def test_custom_field_names(self) -> None:
        """Custom field names work."""
        metric = context_precision(
            ref_field="gold_blocks",
            retrieved_field="pred_blocks",
        )
        pred = {"pred_blocks": ["a", "b"]}
        gold = {"gold_blocks": ["a"]}
        assert metric(pred, gold) == 0.5

    def test_metric_name(self) -> None:
        metric = context_precision()
        assert "context_precision" in _callable_name(metric)


# ---------------------------------------------------------------------------
# Context Recall tests
# ---------------------------------------------------------------------------


class TestContextRecall:
    def test_perfect_recall(self) -> None:
        """All relevant refs are retrieved -- recall = 1.0."""
        metric = context_recall()
        pred = {"retrieved_refs": ["a", "b", "c", "x"]}
        gold = {"relevant_refs": ["a", "b", "c"]}
        assert metric(pred, gold) == 1.0

    def test_half_recall(self) -> None:
        """1 of 2 relevant refs retrieved -- recall = 0.5."""
        metric = context_recall()
        pred = {"retrieved_refs": ["a", "x"]}
        gold = {"relevant_refs": ["a", "b"]}
        assert metric(pred, gold) == 0.5

    def test_zero_recall(self) -> None:
        """No relevant refs retrieved -- recall = 0.0."""
        metric = context_recall()
        pred = {"retrieved_refs": ["x", "y"]}
        gold = {"relevant_refs": ["a", "b"]}
        assert metric(pred, gold) == 0.0

    def test_empty_relevant(self) -> None:
        """Empty relevant set -- recall = 0.0 (degenerate case)."""
        metric = context_recall()
        pred = {"retrieved_refs": ["a"]}
        gold = {"relevant_refs": []}
        assert metric(pred, gold) == 0.0

    def test_missing_fields(self) -> None:
        metric = context_recall()
        assert metric({}, {}) == 0.0

    def test_retrieved_from_gold(self) -> None:
        """Fallback: retrieved_refs from gold when absent from prediction."""
        metric = context_recall()
        gold = {
            "retrieved_refs": ["a"],
            "relevant_refs": ["a", "b"],
        }
        assert metric({}, gold) == 0.5

    def test_metric_name(self) -> None:
        metric = context_recall()
        assert "context_recall" in _callable_name(metric)


# ---------------------------------------------------------------------------
# Answer Relevancy tests (mocked embeddings)
# ---------------------------------------------------------------------------


class TestAnswerRelevancy:
    def _mock_sim_factory(self, return_value: float):
        """Return a mock semantic_similarity factory."""

        def factory(model: str = "", **kwargs: Any):
            def _sim(prediction: Any, gold: Any) -> float:
                return return_value

            _sim.__name__ = "mock_semantic_similarity"
            return _sim

        return factory

    def test_happy_path(self) -> None:
        """Answer relevancy returns the similarity score."""
        with patch(
            "kaos_llm_core.metrics.text.semantic_similarity",
            self._mock_sim_factory(0.85),
        ):
            metric = answer_relevancy()
            gold = {"question": "What is judicial review?"}
            score = metric("The power of courts to review laws", gold)
            assert score == 0.85

    def test_missing_question(self) -> None:
        """No question in gold -- returns 0.0."""
        with patch(
            "kaos_llm_core.metrics.text.semantic_similarity",
            self._mock_sim_factory(0.85),
        ):
            metric = answer_relevancy()
            assert metric("an answer", {}) == 0.0

    def test_none_prediction(self) -> None:
        """None prediction -- returns 0.0."""
        with patch(
            "kaos_llm_core.metrics.text.semantic_similarity",
            self._mock_sim_factory(0.85),
        ):
            metric = answer_relevancy()
            assert metric(None, {"question": "test"}) == 0.0

    def test_answer_object_prediction(self) -> None:
        """Answer object should have its value extracted."""
        with patch(
            "kaos_llm_core.metrics.text.semantic_similarity",
            self._mock_sim_factory(0.9),
        ):
            metric = answer_relevancy()
            claim = _make_claim("judicial review")
            answer = _make_answer([claim], value="Courts can review laws")
            gold = {"question": "What is judicial review?"}
            score = metric(answer, gold)
            assert score == 0.9

    def test_custom_question_field(self) -> None:
        """Custom question field name works."""
        with patch(
            "kaos_llm_core.metrics.text.semantic_similarity",
            self._mock_sim_factory(0.7),
        ):
            metric = answer_relevancy(question_field="query")
            gold = {"query": "What is judicial review?"}
            score = metric("test answer", gold)
            assert score == 0.7

    def test_metric_name(self) -> None:
        with patch(
            "kaos_llm_core.metrics.text.semantic_similarity",
            self._mock_sim_factory(0.0),
        ):
            metric = answer_relevancy()
            assert "answer_relevancy" in _callable_name(metric)


# ---------------------------------------------------------------------------
# RAGAS Composite tests
# ---------------------------------------------------------------------------


class TestRagasComposite:
    def test_equal_weights(self) -> None:
        """All sub-metrics return 1.0 -- composite = 1.0."""
        # Build an answer that verifies perfectly
        claim = _make_claim("Marbury v. Madison (1803)")
        answer = _make_answer([claim], value="Marbury v. Madison")

        gold = {
            "question": "What case established judicial review?",
            "relevant_refs": ["a", "b"],
            "retrieved_refs": ["a", "b"],
        }

        with patch(
            "kaos_llm_core.metrics.text.semantic_similarity",
        ) as mock_sim:
            mock_sim.return_value = lambda p, g: 1.0
            mock_sim.return_value.__name__ = "mock"
            metric = ragas_composite(CORPUS)
            score = metric(answer, gold)
            # faithfulness=1.0, ctx_prec=1.0, ctx_recall=1.0, ans_rel=1.0
            assert score == pytest.approx(1.0)

    def test_partial_scores(self) -> None:
        """Mix of scores produces correct weighted average."""
        # Faithfulness: 1 good claim, 1 bad claim => 0.5
        good_claim = _make_claim("Marbury v. Madison (1803)")
        bad_span = Span(
            source_uri=SOURCE_URI,
            char_span=(0, 5),
            quote="NOPE",
        )
        bad_claim = Claim(
            statement="Bad",
            claim_type="factual",
            supporting_spans=[bad_span],
            confidence=1.0,
        )
        answer = _make_answer([good_claim, bad_claim])

        gold = {
            "question": "What case?",
            "relevant_refs": ["a", "b"],
            "retrieved_refs": ["a", "c"],  # precision=0.5, recall=0.5
        }

        with patch(
            "kaos_llm_core.metrics.text.semantic_similarity",
        ) as mock_sim:
            mock_sim.return_value = lambda p, g: 0.8
            mock_sim.return_value.__name__ = "mock"
            metric = ragas_composite(CORPUS)
            score = metric(answer, gold)
            # faithfulness=0.5, ctx_prec=0.5, ctx_recall=0.5, ans_rel=0.8
            expected = 0.25 * 0.5 + 0.25 * 0.5 + 0.25 * 0.5 + 0.25 * 0.8
            assert score == pytest.approx(expected)

    def test_custom_weights(self) -> None:
        """Custom weights are respected and normalized."""
        claim = _make_claim("Marbury v. Madison (1803)")
        answer = _make_answer([claim])

        gold = {
            "question": "test",
            "relevant_refs": ["a"],
            "retrieved_refs": ["a"],
        }

        with patch(
            "kaos_llm_core.metrics.text.semantic_similarity",
        ) as mock_sim:
            mock_sim.return_value = lambda p, g: 0.5
            mock_sim.return_value.__name__ = "mock"
            # Weight faithfulness at 100%, others at 0%
            weights = {
                "faithfulness": 1.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "answer_relevancy": 0.0,
            }
            metric = ragas_composite(CORPUS, weights=weights)
            score = metric(answer, gold)
            # faithfulness=1.0, everything else irrelevant
            assert score == pytest.approx(1.0)

    def test_invalid_weight_keys(self) -> None:
        """Wrong weight keys raise ValueError."""
        with pytest.raises(ValueError, match="weights must have keys"):
            ragas_composite(CORPUS, weights={"bad_key": 1.0})

    def test_zero_total_weight(self) -> None:
        """Weights summing to zero raise ValueError."""
        with pytest.raises(ValueError, match="sum to a positive"):
            ragas_composite(
                CORPUS,
                weights={
                    "faithfulness": 0.0,
                    "context_precision": 0.0,
                    "context_recall": 0.0,
                    "answer_relevancy": 0.0,
                },
            )

    def test_metric_name(self) -> None:
        with patch(
            "kaos_llm_core.metrics.text.semantic_similarity",
        ) as mock_sim:
            mock_sim.return_value = lambda p, g: 0.0
            mock_sim.return_value.__name__ = "mock"
            metric = ragas_composite(CORPUS)
            assert _callable_name(metric) == "ragas_composite"


class TestDefaultRagasWeights:
    def test_sum_to_one(self) -> None:
        assert sum(DEFAULT_RAGAS_WEIGHTS.values()) == pytest.approx(1.0)

    def test_four_keys(self) -> None:
        assert set(DEFAULT_RAGAS_WEIGHTS.keys()) == {
            "faithfulness",
            "context_precision",
            "context_recall",
            "answer_relevancy",
        }


# ---------------------------------------------------------------------------
# Integration: metrics module exports
# ---------------------------------------------------------------------------


class TestModuleExports:
    """Verify all retrieval metrics are exported from the metrics package."""

    def test_faithfulness_importable(self) -> None:
        from kaos_llm_core.metrics import faithfulness as f

        assert callable(f)

    def test_context_precision_importable(self) -> None:
        from kaos_llm_core.metrics import context_precision as f

        assert callable(f)

    def test_context_recall_importable(self) -> None:
        from kaos_llm_core.metrics import context_recall as f

        assert callable(f)

    def test_answer_relevancy_importable(self) -> None:
        from kaos_llm_core.metrics import answer_relevancy as f

        assert callable(f)

    def test_ragas_composite_importable(self) -> None:
        from kaos_llm_core.metrics import ragas_composite as f

        assert callable(f)

    def test_default_weights_importable(self) -> None:
        from kaos_llm_core.metrics import DEFAULT_RAGAS_WEIGHTS as w

        assert isinstance(w, dict)


# ---------------------------------------------------------------------------
# Refusal rate tests
# ---------------------------------------------------------------------------


class TestRefusalRate:
    """Tests for the refusal_rate metric."""

    def test_refusal_returns_one(self) -> None:
        ie = InsufficientEvidence(reason="Not enough context")
        metric = refusal_rate()
        assert metric(ie, {}) == 1.0

    def test_answer_returns_zero(self) -> None:
        claim = _make_claim("judicial review")
        answer = _make_answer([claim])
        metric = refusal_rate()
        assert metric(answer, {}) == 0.0

    def test_string_returns_zero(self) -> None:
        metric = refusal_rate()
        assert metric("some text", {}) == 0.0

    def test_none_returns_zero(self) -> None:
        metric = refusal_rate()
        assert metric(None, {}) == 0.0

    def test_metric_name(self) -> None:
        metric = refusal_rate()
        assert _callable_name(metric) == "refusal_rate"


# ---------------------------------------------------------------------------
# Expansion lift tests
# ---------------------------------------------------------------------------


class TestExpansionLift:
    """Tests for the expansion_lift metric."""

    def test_expansion_helps(self) -> None:
        """Expanded retrieval finds more relevant refs → positive lift."""
        metric = expansion_lift()
        gold = {
            "relevant_refs": ["#/body/3", "#/body/7"],
            "expanded_refs": ["#/body/3", "#/body/7"],  # 2/2 = 1.0
            "raw_refs": ["#/body/12"],  # 0/1 = 0.0
        }
        assert metric(None, gold) == pytest.approx(1.0)

    def test_expansion_hurts(self) -> None:
        """Expanded retrieval has lower precision → negative lift."""
        metric = expansion_lift()
        gold = {
            "relevant_refs": ["#/body/3"],
            "expanded_refs": ["#/body/12", "#/body/15"],  # 0/2 = 0.0
            "raw_refs": ["#/body/3"],  # 1/1 = 1.0
        }
        assert metric(None, gold) == pytest.approx(-1.0)

    def test_no_change(self) -> None:
        """Same precision → zero lift."""
        metric = expansion_lift()
        gold = {
            "relevant_refs": ["#/body/3"],
            "expanded_refs": ["#/body/3"],
            "raw_refs": ["#/body/3"],
        }
        assert metric(None, gold) == pytest.approx(0.0)

    def test_empty_relevant(self) -> None:
        """No gold refs → 0.0."""
        metric = expansion_lift()
        gold = {
            "relevant_refs": [],
            "expanded_refs": ["#/body/3"],
            "raw_refs": ["#/body/3"],
        }
        assert metric(None, gold) == 0.0

    def test_empty_expanded(self) -> None:
        """Empty expanded refs → precision 0.0."""
        metric = expansion_lift()
        gold = {
            "relevant_refs": ["#/body/3"],
            "expanded_refs": [],
            "raw_refs": ["#/body/3"],  # 1/1 = 1.0
        }
        assert metric(None, gold) == pytest.approx(-1.0)

    def test_metric_name(self) -> None:
        metric = expansion_lift()
        assert _callable_name(metric) == "expansion_lift"


# ---------------------------------------------------------------------------
# Partial span failure (edge case from code review)
# ---------------------------------------------------------------------------


class TestPartialSpanFailure:
    """Tests for claims where some spans verify and others don't."""

    def test_one_verified_one_failed(self) -> None:
        """Claim with one good span and one bad span: claim fails."""
        good_span = Span(
            source_uri=SOURCE_URI,
            char_span=(0, 23),
            quote="Marbury v. Madison (1803)",
        )
        bad_span = Span(
            source_uri=SOURCE_URI,
            char_span=(0, 10),
            quote="THIS IS NOT IN THE SOURCE AT ALL",
        )
        claim = Claim(
            statement="test claim",
            supporting_spans=[good_span, bad_span],
        )
        errors = claim.verify(CORPUS)
        # One span fails → claim has errors
        assert len(errors) == 1
        assert errors[0].span_index == 1

    def test_all_verified(self) -> None:
        """All spans verify → no errors."""
        span1 = Span(source_uri=SOURCE_URI, char_span=(0, 23), quote="Marbury v. Madison (1803)")
        span2 = Span(source_uri=SOURCE_URI, char_span=(0, 23), quote="Marbury v. Madison (1803)")
        claim = Claim(statement="test", supporting_spans=[span1, span2])
        errors = claim.verify(CORPUS)
        assert len(errors) == 0

    def test_diagnostics_on_partial(self) -> None:
        """Diagnostics populated only for failed spans."""
        good_span = Span(
            source_uri=SOURCE_URI, char_span=(0, 23), quote="Marbury v. Madison (1803)"
        )
        bad_span = Span(source_uri=SOURCE_URI, char_span=(0, 10), quote="NONEXISTENT TEXT")
        claim = Claim(statement="test", supporting_spans=[good_span, bad_span])
        errors = claim.verify(CORPUS, diagnostics=True)
        assert len(errors) == 1
        # Diagnostics should be populated
        assert len(errors[0].strategy_results) > 0
