"""Tests for the NLI claim-verification helpers.

Offline + deterministic: a stub :class:`NLIScorer` returns hand-crafted
``(entailment, neutral, contradiction)`` triples per hypothesis (claim).
The real ``kaos_nlp_transformers.NliModel`` satisfies the same Protocol.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pytest

from kaos_llm_core.errors import CallError
from kaos_llm_core.programs.classify import (
    ClaimJudge,
    ClaimVerdict,
    Embedder,
    JudgeVerdict,
    verify_claim,
    verify_claims,
)


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


# --------------------------------------------------------------------------
# Hybrid fallback: NLI-neutral-band escalation to an injected ClaimJudge,
# cost-gated by an optional Embedder. Offline fakes; the judge call-count is
# the cost contract and is asserted precisely.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _StubJudgeVerdict:
    label: str
    confidence: float
    rationale: str | None = None


class _CountingJudge:
    """Records every judge call; returns a pre-baked verdict per claim."""

    def __init__(self, table: dict[str, tuple[str, float]]) -> None:
        self._table = table
        self.calls: list[tuple[str, str]] = []

    def judge(self, claim: str, evidence: str) -> JudgeVerdict:
        self.calls.append((claim, evidence))
        label, conf = self._table.get(claim, ("neutral", 0.5))
        return _StubJudgeVerdict(label=label, confidence=conf, rationale="because")


class _StubEmbedder:
    """Deterministic embedder: maps each text to a pre-baked unit vector."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors
        self.calls = 0

    def embed(self, texts: Iterable[str], *, batch_size: int = 32) -> np.ndarray:
        self.calls += 1
        rows = [self._vectors[t] for t in texts]
        arr = np.asarray(rows, dtype=np.float32)
        # L2-normalise rows to honour the Embedder contract.
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return arr / norms


def test_protocols_are_runtime_checkable() -> None:
    judge = _CountingJudge({})
    embedder = _StubEmbedder({})
    assert isinstance(judge, ClaimJudge)
    assert isinstance(embedder, Embedder)
    assert isinstance(_StubJudgeVerdict("neutral", 0.5), JudgeVerdict)


def test_confident_entail_does_not_escalate() -> None:
    scorer = _StubScorer({"supported claim": (0.95, 0.04, 0.01)})
    judge = _CountingJudge({"supported claim": ("refuted", 0.9)})
    v = verify_claim(scorer, "supported claim", evidence="evidence", judge=judge)
    assert v.label == "entailment"
    assert v.method == "nli"
    assert judge.calls == []  # confident NLI is trusted; judge NOT called


def test_confident_contradiction_does_not_escalate() -> None:
    scorer = _StubScorer({"refuted claim": (0.02, 0.03, 0.95)})
    judge = _CountingJudge({"refuted claim": ("supported", 0.9)})
    v = verify_claim(scorer, "refuted claim", evidence="evidence", judge=judge)
    assert v.label == "contradiction"
    assert v.method == "nli"
    assert judge.calls == []


def test_neutral_below_gate_keeps_nli_and_skips_judge() -> None:
    scorer = _StubScorer({"mismatch claim": (0.1, 0.85, 0.05)})
    judge = _CountingJudge({"mismatch claim": ("supported", 0.9)})
    # Orthogonal vectors → cosine 0.0, below the default gate of 0.5.
    embedder = _StubEmbedder(
        {"mismatch claim": [1.0, 0.0], "evidence": [0.0, 1.0]},
    )
    v = verify_claim(scorer, "mismatch claim", evidence="evidence", judge=judge, embedder=embedder)
    assert v.label == "neutral"  # NLI neutral kept
    assert v.method == "nli"
    assert v.gate_cosine == pytest.approx(0.0, abs=1e-6)
    assert judge.calls == []  # below gate → judge NOT called (the cost contract)
    assert embedder.calls == 1  # one batched embed of [claim, evidence]


def test_neutral_above_gate_escalates_and_verdict_flips() -> None:
    scorer = _StubScorer({"paraphrased claim": (0.1, 0.85, 0.05)})
    judge = _CountingJudge({"paraphrased claim": ("supported", 0.88)})
    # Identical vectors → cosine 1.0, above the default gate of 0.5.
    embedder = _StubEmbedder(
        {"paraphrased claim": [1.0, 1.0], "evidence": [1.0, 1.0]},
    )
    v = verify_claim(
        scorer, "paraphrased claim", evidence="evidence", judge=judge, embedder=embedder
    )
    assert v.label == "entailment"  # flipped neutral → supported by the judge
    assert v.supported is True
    assert v.method == "llm_judge"
    assert v.confidence == pytest.approx(0.88)
    assert v.gate_cosine == pytest.approx(1.0, abs=1e-6)
    assert v.rationale == "because"
    assert len(judge.calls) == 1
    assert judge.calls[0] == ("paraphrased claim", "evidence")


def test_neutral_with_judge_no_embedder_escalates_directly() -> None:
    scorer = _StubScorer({"claim": (0.1, 0.85, 0.05)})
    judge = _CountingJudge({"claim": ("refuted", 0.7)})
    v = verify_claim(scorer, "claim", evidence="evidence", judge=judge)
    assert v.label == "contradiction"
    assert v.method == "llm_judge"
    assert v.gate_cosine is None  # no embedder → no cosine computed
    assert len(judge.calls) == 1


def test_no_judge_supplied_preserves_pure_nli() -> None:
    scorer = _StubScorer({"claim": (0.1, 0.85, 0.05)})
    embedder = _StubEmbedder({"claim": [1.0, 1.0], "evidence": [1.0, 1.0]})
    # Embedder but no judge: pure NLI, no embed call (nothing to gate).
    v = verify_claim(scorer, "claim", evidence="evidence", embedder=embedder)
    assert v.label == "neutral"
    assert v.method == "nli"
    assert v.gate_cosine is None
    assert embedder.calls == 0


def test_batch_one_nli_call_only_neutral_band_escalates() -> None:
    table = {
        "entail": (0.9, 0.05, 0.05),  # confident entail → no escalation
        "contra": (0.05, 0.05, 0.9),  # confident contra → no escalation
        "neutral_close": (0.1, 0.85, 0.05),  # neutral, cosine high → escalate
        "neutral_far": (0.1, 0.85, 0.05),  # neutral, cosine low → keep NLI
    }
    scorer = _StubScorer(table)
    judge = _CountingJudge({"neutral_close": ("supported", 0.8)})
    embedder = _StubEmbedder(
        {
            "neutral_close": [1.0, 0.0],
            "neutral_far": [0.0, 1.0],
            "evidence": [1.0, 0.0],
        }
    )
    claims = ["entail", "contra", "neutral_close", "neutral_far"]
    verdicts = verify_claims(scorer, claims, evidence="evidence", judge=judge, embedder=embedder)
    # ONE NLI forward pass for all four claims.
    assert len(scorer.calls) == 1
    assert scorer.calls[0] == ("evidence", claims)
    assert [v.label for v in verdicts] == [
        "entailment",
        "contradiction",
        "entailment",  # neutral_close flipped by judge
        "neutral",  # neutral_far kept (below gate)
    ]
    assert [v.method for v in verdicts] == ["nli", "nli", "llm_judge", "nli"]
    # Judge called exactly once: only neutral_close cleared the gate.
    assert judge.calls == [("neutral_close", "evidence")]


def test_low_confidence_entail_escalates() -> None:
    # Entailment the NLI head is not confident about (below ``confident``).
    scorer = _StubScorer({"weak claim": (0.4, 0.35, 0.25)})
    judge = _CountingJudge({"weak claim": ("refuted", 0.9)})
    v = verify_claim(scorer, "weak claim", evidence="evidence", judge=judge, confident=0.5)
    assert v.label == "contradiction"
    assert v.method == "llm_judge"
    assert len(judge.calls) == 1


def test_unrecognised_judge_label_raises() -> None:
    scorer = _StubScorer({"claim": (0.1, 0.85, 0.05)})
    judge = _CountingJudge({"claim": ("maybe", 0.5)})
    with pytest.raises(CallError, match="unrecognised label"):
        verify_claim(scorer, "claim", evidence="evidence", judge=judge)


def test_threshold_validation() -> None:
    scorer = _StubScorer({})
    with pytest.raises(ValueError, match="confident"):
        verify_claim(scorer, "c", evidence="e", confident=1.5)
    with pytest.raises(ValueError, match="neutral_floor"):
        verify_claim(scorer, "c", evidence="e", neutral_floor=-0.1)
    with pytest.raises(ValueError, match="gate"):
        verify_claim(scorer, "c", evidence="e", gate=2.0)
