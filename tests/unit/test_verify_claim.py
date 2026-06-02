"""Tests for the NLI claim-verification helpers.

Offline + deterministic: a stub :class:`NLIScorer` returns hand-crafted
``(entailment, neutral, contradiction)`` triples per hypothesis (claim).
The real ``kaos_nlp_transformers.NliModel`` satisfies the same Protocol.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from kaos_llm_core.errors import CallError
from kaos_llm_core.programs.classify import ClaimVerdict, verify_claim, verify_claims


@dataclass(frozen=True, slots=True)
class _StubScore:
    entailment: float
    neutral: float
    contradiction: float


class _StubScorer:
    """Returns pre-baked NLI triples keyed by hypothesis (claim) text."""

    def __init__(self, table: dict[str, tuple[float, float, float]]) -> None:
        self._table = table
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, premise: str, hypotheses: Sequence[str]) -> Sequence[_StubScore]:
        self.calls.append((premise, list(hypotheses)))
        return [_StubScore(*self._table.get(h, (0.1, 0.8, 0.1))) for h in hypotheses]


class _MiscountScorer:
    """Returns the wrong number of scores, to exercise the guard."""

    def score(self, premise: str, hypotheses: Sequence[str]) -> Sequence[_StubScore]:
        return [_StubScore(0.5, 0.3, 0.2)]


def test_verify_claim_entailment_supported() -> None:
    scorer = _StubScorer({"the sky is blue": (0.9, 0.07, 0.03)})
    v = verify_claim(scorer, "the sky is blue", evidence="a clear daytime sky is blue")
    assert isinstance(v, ClaimVerdict)
    assert v.label == "entailment"
    assert v.confidence == pytest.approx(0.9, abs=1e-6)
    assert v.supported is True
    assert v.refuted is False
    assert v.claim == "the sky is blue"
    assert (v.entailment, v.neutral, v.contradiction) == pytest.approx((0.9, 0.07, 0.03))


def test_verify_claim_contradiction_refuted() -> None:
    scorer = _StubScorer({"the sky is green": (0.02, 0.08, 0.90)})
    v = verify_claim(scorer, "the sky is green", evidence="the sky is blue")
    assert v.label == "contradiction"
    assert v.refuted is True
    assert v.supported is False


def test_verify_claim_neutral() -> None:
    scorer = _StubScorer({"it will rain tomorrow": (0.1, 0.8, 0.1)})
    v = verify_claim(scorer, "it will rain tomorrow", evidence="the sky is blue")
    assert v.label == "neutral"
    assert v.supported is False
    assert v.refuted is False


def test_verify_claims_order_and_single_forward_pass() -> None:
    table = {
        "claim a": (0.8, 0.1, 0.1),
        "claim b": (0.1, 0.1, 0.8),
        "claim c": (0.2, 0.7, 0.1),
    }
    scorer = _StubScorer(table)
    claims = ["claim a", "claim b", "claim c"]
    verdicts = verify_claims(scorer, claims, evidence="some evidence text")
    assert [v.claim for v in verdicts] == claims
    assert [v.label for v in verdicts] == ["entailment", "contradiction", "neutral"]
    # One scorer call for all claims; evidence is the premise, claims are
    # the hypotheses.
    assert len(scorer.calls) == 1
    premise, hypotheses = scorer.calls[0]
    assert premise == "some evidence text"
    assert hypotheses == claims


def test_verify_claims_empty_skips_scorer() -> None:
    scorer = _StubScorer({})
    assert verify_claims(scorer, [], evidence="anything") == []
    assert scorer.calls == []


def test_verify_claim_tie_resolves_to_entailment() -> None:
    scorer = _StubScorer({"ambiguous": (1 / 3, 1 / 3, 1 / 3)})
    v = verify_claim(scorer, "ambiguous", evidence="ambiguous evidence")
    assert v.label == "entailment"


def test_verify_claims_score_count_mismatch_raises() -> None:
    with pytest.raises(CallError, match="one score per hypothesis"):
        verify_claims(_MiscountScorer(), ["a", "b"], evidence="e")


def test_claim_verdict_is_frozen() -> None:
    v = ClaimVerdict(
        claim="c",
        label="neutral",
        confidence=0.5,
        entailment=0.25,
        neutral=0.5,
        contradiction=0.25,
    )
    with pytest.raises((AttributeError, TypeError)):
        v.label = "entailment"  # ty: ignore[invalid-assignment]  # frozen by design
