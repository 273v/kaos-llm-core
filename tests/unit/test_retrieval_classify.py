"""Tests for :class:`~kaos_llm_core.programs.classify.RetrievalClassify`."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pytest

from kaos_llm_core.errors import CallError
from kaos_llm_core.labels import Label, LabelSet
from kaos_llm_core.programs.base import Program
from kaos_llm_core.programs.classify import RetrievalClassify
from kaos_llm_core.results import Classification


class _StubEmbedder:
    """Embedder stub: pre-baked unit vectors for known inputs.

    Unknown inputs fall back to a small deterministic vector.
    """

    def __init__(self, table: dict[str, list[float]]) -> None:
        self._table = {k: np.asarray(v, dtype=np.float32) for k, v in table.items()}
        # Normalise rows.
        for k, v in self._table.items():
            norm = float(np.linalg.norm(v))
            if norm > 0:
                self._table[k] = v / norm

    def embed(self, texts: Iterable[str], *, batch_size: int = 32) -> np.ndarray:
        out: list[np.ndarray] = []
        for t in texts:
            out.append(self._table.get(t, np.array([0.1, 0.1], dtype=np.float32)))
        return np.stack(out, axis=0)


def _labels(*names: str, exclusive: bool = True, allow_abstain: bool = True) -> LabelSet:
    return LabelSet(
        labels=[Label(name=n, description=f"about {n}") for n in names],
        exclusive=exclusive,
        allow_abstain=allow_abstain,
    )


# Two-axis embedding: positive examples on axis 0, negative on axis 1.
_TABLE = {
    "I love it": [1.0, 0.0],
    "Excellent product": [0.9, 0.1],
    "Best purchase ever": [0.8, 0.2],
    "Terrible quality": [0.0, 1.0],
    "Worst purchase": [0.1, 0.9],
    "Hate this": [0.2, 0.8],
}


class _StubLLMTieBreak(Program):
    """Always returns the same Label — used to verify tie-break wiring."""

    def __init__(self, *, label_set: LabelSet, picked_name: str) -> None:
        self._label_set = label_set
        self._picked_name = picked_name

    async def forward(  # ty: ignore[invalid-method-override]
        self, *, text: str, parent_id: str | None = None
    ) -> Classification:
        return Classification(
            labels=[self._label_set.by_name(self._picked_name)],
            scores={self._picked_name: 1.0},
            abstained=False,
            rationale="tie-break stub",
            chunks_used=[],
            metadata={"program": "_StubLLMTieBreak"},
        )


class TestRetrievalClassify:
    @pytest.mark.asyncio
    async def test_picks_label_with_majority_neighbours(self) -> None:
        labels = _labels("positive", "negative")
        corpus = [
            ("I love it", "positive"),
            ("Excellent product", "positive"),
            ("Best purchase ever", "positive"),
            ("Terrible quality", "negative"),
            ("Worst purchase", "negative"),
            ("Hate this", "negative"),
        ]
        program = RetrievalClassify(
            labels=labels,
            embedder=_StubEmbedder(_TABLE),
            corpus=corpus,
            k=3,
        )
        # Input lands on the positive axis.
        result = await program(text="I love it")
        assert isinstance(result, Classification)
        assert result.top_label == "positive"
        assert result.metadata["program"] == "RetrievalClassify"
        assert result.metadata["close_call"] is False
        # Scores normalised: positive >> negative.
        assert result.scores["positive"] > result.scores["negative"]

    @pytest.mark.asyncio
    async def test_close_call_triggers_tie_break(self) -> None:
        labels = _labels("positive", "negative")
        corpus = [
            ("I love it", "positive"),
            ("Terrible quality", "negative"),
        ]
        tie_break = _StubLLMTieBreak(label_set=labels, picked_name="negative")
        # Input is exactly between the two axes → close call.
        program = RetrievalClassify(
            labels=labels,
            embedder=_StubEmbedder(_TABLE | {"mixed": [0.7, 0.7]}),
            corpus=corpus,
            k=2,
            tie_break=tie_break,
            tie_break_margin=0.5,  # very wide margin so the tie-break fires
        )
        result = await program(text="mixed")
        # Tie-break flipped to negative. ``top_label`` is a property
        # over ``scores`` which still has the tied weighted-vote
        # distribution; the actual picked label landed in
        # ``result.labels[0]``.
        assert result.labels[0].name == "negative"
        assert result.metadata["tie_break.used"] is True
        assert result.metadata["tie_break.label"] == "negative"

    @pytest.mark.asyncio
    async def test_empty_corpus_abstains(self) -> None:
        labels = _labels("a", "b")
        program = RetrievalClassify(
            labels=labels,
            embedder=_StubEmbedder({}),
            corpus=[],
            k=3,
        )
        result = await program(text="anything")
        assert result.abstained is True
        assert result.metadata["abstain_reason"] == "empty_corpus"

    @pytest.mark.asyncio
    async def test_empty_input_abstains(self) -> None:
        labels = _labels("a", "b", allow_abstain=True)
        program = RetrievalClassify(
            labels=labels,
            embedder=_StubEmbedder({"x": [1.0, 0.0]}),
            corpus=[("x", "a")],
            k=1,
        )
        result = await program(text="")
        assert result.abstained is True
        assert result.metadata["abstain_reason"] == "empty_input"

    def test_unknown_label_in_corpus_rejected(self) -> None:
        labels = _labels("a", "b")
        with pytest.raises(CallError, match="unknown labels"):
            RetrievalClassify(
                labels=labels,
                embedder=_StubEmbedder({}),
                corpus=[("hello", "c")],
                k=1,
            )

    def test_multi_label_set_rejected(self) -> None:
        labels = _labels("a", "b", exclusive=False)
        with pytest.raises(CallError, match="exclusive LabelSet"):
            RetrievalClassify(
                labels=labels,
                embedder=_StubEmbedder({}),
                corpus=[],
                k=1,
            )

    def test_invalid_k_rejected(self) -> None:
        labels = _labels("a", "b")
        with pytest.raises(ValueError, match="k must be"):
            RetrievalClassify(
                labels=labels,
                embedder=_StubEmbedder({}),
                corpus=[],
                k=0,
            )
