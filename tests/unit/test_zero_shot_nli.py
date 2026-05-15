"""Tests for :class:`~kaos_llm_core.programs.classify.ZeroShotNLIClassifier`.

Offline + deterministic: a stub :class:`NLIScorer` returns hand-crafted
``(entailment, neutral, contradiction)`` triples per hypothesis. The
real ``kaos_nlp_transformers.NliModel`` will satisfy the same Protocol
once it ships (plan §4.2.3, §8).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from kaos_llm_core.errors import CallError
from kaos_llm_core.labels import ABSTAIN_LABEL, Label, LabelSet
from kaos_llm_core.programs.classify import ZeroShotNLIClassifier
from kaos_llm_core.results import Classification


@dataclass(frozen=True, slots=True)
class _StubScore:
    """Minimal :class:`NLIScore` implementation."""

    entailment: float
    neutral: float
    contradiction: float


class _StubScorer:
    """Returns pre-baked entailment scores keyed by hypothesis."""

    def __init__(self, table: dict[str, tuple[float, float, float]]) -> None:
        self._table = table
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, premise: str, hypotheses: Sequence[str]) -> Sequence[_StubScore]:
        self.calls.append((premise, list(hypotheses)))
        out: list[_StubScore] = []
        for h in hypotheses:
            ent, neut, con = self._table.get(h, (0.1, 0.8, 0.1))
            out.append(_StubScore(entailment=ent, neutral=neut, contradiction=con))
        return out


def _labels(*names: str, allow_abstain: bool = True) -> LabelSet:
    return LabelSet(
        labels=[Label(name=n, description=f"description of {n}") for n in names],
        exclusive=True,
        allow_abstain=allow_abstain,
    )


class TestZeroShotNLI:
    @pytest.mark.asyncio
    async def test_argmax_picks_highest_entailment(self) -> None:
        labels = _labels("contract", "memo", "letter")
        scorer = _StubScorer(
            {
                "This text is about description of contract.": (0.9, 0.05, 0.05),
                "This text is about description of memo.": (0.2, 0.7, 0.1),
                "This text is about description of letter.": (0.1, 0.8, 0.1),
            }
        )
        program = ZeroShotNLIClassifier(labels=labels, scorer=scorer)
        result = await program(text="Acme Corp hereby agrees to deliver…")
        assert isinstance(result, Classification)
        assert result.top_label == "contract"
        # Scores map carries entailment probabilities in LabelSet order.
        assert pytest.approx(result.scores["contract"]) == 0.9
        # nli metadata records the full distribution.
        nli = result.metadata["nli"]
        contract_row = next(row for row in nli if row["label"] == "contract")
        assert pytest.approx(contract_row["entailment"]) == 0.9
        assert pytest.approx(contract_row["contradiction"]) == 0.05

    @pytest.mark.asyncio
    async def test_min_score_triggers_abstain(self) -> None:
        labels = _labels("contract", "memo")
        scorer = _StubScorer(
            {
                "This text is about description of contract.": (0.3, 0.4, 0.3),
                "This text is about description of memo.": (0.2, 0.6, 0.2),
            }
        )
        program = ZeroShotNLIClassifier(labels=labels, scorer=scorer, min_score=0.5)
        result = await program(text="unclear text")
        assert result.abstained is True
        assert result.metadata["abstain_reason"] == "below_min_score"

    @pytest.mark.asyncio
    async def test_min_score_not_triggered_when_above(self) -> None:
        labels = _labels("contract", "memo")
        scorer = _StubScorer(
            {
                "This text is about description of contract.": (0.8, 0.1, 0.1),
                "This text is about description of memo.": (0.2, 0.6, 0.2),
            }
        )
        program = ZeroShotNLIClassifier(labels=labels, scorer=scorer, min_score=0.5)
        result = await program(text="x")
        assert result.abstained is False
        assert result.top_label == "contract"

    @pytest.mark.asyncio
    async def test_custom_hypothesis_template(self) -> None:
        labels = _labels("billing", "support")
        scorer = _StubScorer(
            {
                "This email is about description of billing.": (0.9, 0.05, 0.05),
                "This email is about description of support.": (0.1, 0.8, 0.1),
            }
        )
        program = ZeroShotNLIClassifier(
            labels=labels,
            scorer=scorer,
            hypothesis_template="This email is about {}.",
        )
        result = await program(text="invoice attached")
        assert result.top_label == "billing"
        # Confirm the scorer saw the templated hypotheses.
        last_premise, last_hypotheses = scorer.calls[-1]
        assert last_premise == "invoice attached"
        assert "This email is about description of billing." in last_hypotheses

    @pytest.mark.asyncio
    async def test_empty_input_abstains(self) -> None:
        labels = _labels("a", "b", allow_abstain=True)
        scorer = _StubScorer({})
        program = ZeroShotNLIClassifier(labels=labels, scorer=scorer)
        result = await program(text="")
        assert result.abstained is True
        assert result.scores == {ABSTAIN_LABEL: 0.0}

    @pytest.mark.asyncio
    async def test_zero_token_usage(self) -> None:
        labels = _labels("a", "b")
        scorer = _StubScorer(
            {
                "This text is about description of a.": (0.9, 0.05, 0.05),
                "This text is about description of b.": (0.1, 0.8, 0.1),
            }
        )
        program = ZeroShotNLIClassifier(labels=labels, scorer=scorer)
        invocation = await program.invoke(text="x")
        # NLI runs entirely outside the LLM provider path.
        assert invocation.usage.input_tokens == 0
        assert invocation.usage.output_tokens == 0

    def test_invalid_template_rejected(self) -> None:
        labels = _labels("a", "b")
        scorer = _StubScorer({})
        with pytest.raises(CallError, match="placeholder"):
            ZeroShotNLIClassifier(
                labels=labels,
                scorer=scorer,
                hypothesis_template="No placeholder here.",
            )

    def test_multi_label_set_rejected(self) -> None:
        labels = LabelSet(
            labels=[Label(name="a"), Label(name="b")],
            exclusive=False,
        )
        scorer = _StubScorer({})
        with pytest.raises(CallError, match="exclusive LabelSet"):
            ZeroShotNLIClassifier(labels=labels, scorer=scorer)

    def test_min_score_out_of_range(self) -> None:
        labels = _labels("a", "b")
        scorer = _StubScorer({})
        with pytest.raises(ValueError, match=r"min_score must be in \[0, 1\]"):
            ZeroShotNLIClassifier(labels=labels, scorer=scorer, min_score=1.5)

    @pytest.mark.asyncio
    async def test_scorer_returning_wrong_count_raises(self) -> None:
        class _BadScorer:
            def score(self, premise: str, hypotheses: Sequence[str]) -> Sequence[_StubScore]:
                return [_StubScore(entailment=0.5, neutral=0.3, contradiction=0.2)]

        labels = _labels("a", "b")
        program = ZeroShotNLIClassifier(labels=labels, scorer=_BadScorer())
        with pytest.raises(CallError, match="one score per hypothesis"):
            await program(text="x")
