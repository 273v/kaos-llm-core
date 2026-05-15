"""Tests for :class:`~kaos_llm_core.programs.classify.PrototypeClassify`.

Offline-deterministic: the :class:`Embedder` protocol lets us pass a
stub that returns hand-crafted unit-norm vectors. Live coverage against
a real ``EmbeddingModel`` lives in ``tests/quality`` (Phase 5 leftover
live harness, behind ``@pytest.mark.live``).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pytest

from kaos_llm_core.labels import ABSTAIN_LABEL, Label, LabelSet
from kaos_llm_core.programs.classify import PrototypeClassify
from kaos_llm_core.results import Classification


class _StubEmbedder:
    """Embedder stub that maps an exact text to a pre-baked unit vector.

    Unknown inputs fall back to a deterministic hash-based unit vector
    so the embedder is total: every callable input gets a row out, but
    seeded inputs land on the prototypes we choose for the test.
    """

    def __init__(self, table: dict[str, np.ndarray]) -> None:
        # Normalise rows up front so the unit-norm contract holds.
        self._table: dict[str, np.ndarray] = {}
        for key, vec in table.items():
            arr = np.asarray(vec, dtype=np.float32)
            norm = float(np.linalg.norm(arr))
            self._table[key] = arr / (norm if norm > 0 else 1.0)
        # Track inputs for assertion purposes.
        self.calls: list[list[str]] = []

    def _unknown_vector(self, text: str, dim: int) -> np.ndarray:
        rng = np.random.default_rng(seed=abs(hash(text)) % (2**32))
        v = rng.standard_normal(dim).astype(np.float32)
        norm = float(np.linalg.norm(v))
        return v / (norm if norm > 0 else 1.0)

    def embed(self, texts: Iterable[str], *, batch_size: int = 32) -> np.ndarray:
        del batch_size  # unused
        text_list = list(texts)
        self.calls.append(text_list)
        # Probe an arbitrary stored row to learn the dim. If the table
        # is empty, default to 4 (the test vectors below all use 4-d).
        dim = next(iter(self._table.values())).shape[0] if self._table else 4
        rows = [
            self._table[t] if t in self._table else self._unknown_vector(t, dim) for t in text_list
        ]
        return np.stack(rows, axis=0)


def _label_set(names: tuple[str, ...], *, exclusive: bool, allow_abstain: bool = True) -> LabelSet:
    return LabelSet(
        labels=[Label(name=n, description=f"description of {n}") for n in names],
        exclusive=exclusive,
        allow_abstain=allow_abstain,
    )


# Four-dimensional one-hot vectors so prototype/argmax assertions are
# trivial: an input that aligns with prototype A has cosine 1 against A
# and 0 against the others.
_LABEL_VECTORS = {
    "description of A": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "description of B": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    "description of C": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    "description of D": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
}

# Inputs that exactly align with a single label.
_INPUT_VECTORS = {
    "ay": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "bee": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    "ab": np.array([0.6, 0.8, 0.0, 0.0], dtype=np.float32),  # closer to B
    "low": np.array([0.3, 0.0, 0.0, 0.0], dtype=np.float32),  # weak A
    "mid": np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32),  # half A, half B
}


def _embedder() -> _StubEmbedder:
    return _StubEmbedder({**_LABEL_VECTORS, **_INPUT_VECTORS})


class TestPrototypeClassifyExclusive:
    @pytest.mark.asyncio
    async def test_argmax_picks_aligned_label(self) -> None:
        labels = _label_set(("A", "B", "C", "D"), exclusive=True)
        program = PrototypeClassify(labels=labels, embedder=_embedder())
        result = await program(text="ay")
        assert isinstance(result, Classification)
        assert result.abstained is False
        assert result.top_label == "A"
        assert result.metadata["program"] == "PrototypeClassify"
        # Every label gets a score, in declaration order.
        assert set(result.scores.keys()) == {"A", "B", "C", "D"}
        # A is exactly 1.0, B/C/D exactly 0.0.
        assert result.scores["A"] == pytest.approx(1.0)
        assert result.scores["B"] == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.asyncio
    async def test_min_score_triggers_abstain(self) -> None:
        labels = _label_set(("A", "B", "C", "D"), exclusive=True, allow_abstain=True)
        program = PrototypeClassify(labels=labels, embedder=_embedder(), min_score=0.95)
        # Input "low" has cosine ≈ 1.0 against A but only after
        # normalization (it's not unit-norm in the table). Pick "mid"
        # which scores ~0.707 against both A and B — below 0.95.
        result = await program(text="mid")
        assert result.abstained is True
        assert result.labels == []
        assert result.metadata["abstain_reason"] == "below_min_score"

    @pytest.mark.asyncio
    async def test_min_score_not_triggered_when_above_threshold(self) -> None:
        labels = _label_set(("A", "B", "C", "D"), exclusive=True)
        program = PrototypeClassify(labels=labels, embedder=_embedder(), min_score=0.5)
        result = await program(text="ay")
        assert result.abstained is False
        assert result.top_label == "A"

    @pytest.mark.asyncio
    async def test_label_prototypes_cached(self) -> None:
        labels = _label_set(("A", "B"), exclusive=True)
        embedder = _embedder()
        program = PrototypeClassify(labels=labels, embedder=embedder)
        await program(text="ay")
        await program(text="bee")
        # First call embeds labels (one batch) + the input ("ay" batch).
        # Second call only embeds the new input ("bee" batch).
        # Total: 3 embed() calls.
        assert len(embedder.calls) == 3
        # The first call carried the label prototype texts.
        assert embedder.calls[0] == ["description of A", "description of B"]
        # The second and third carried just the inputs.
        assert embedder.calls[1] == ["ay"]
        assert embedder.calls[2] == ["bee"]


class TestPrototypeClassifyMultiLabel:
    @pytest.mark.asyncio
    async def test_threshold_picks_all_passing(self) -> None:
        labels = _label_set(("A", "B", "C", "D"), exclusive=False)
        # "mid" gives cosine ~0.707 against A and B; threshold=0.5 keeps
        # both, drops C and D.
        program = PrototypeClassify(labels=labels, embedder=_embedder(), threshold=0.5)
        result = await program(text="mid")
        assert result.abstained is False
        names = [label.name for label in result.labels]
        assert names == ["A", "B"]

    @pytest.mark.asyncio
    async def test_high_threshold_abstains_when_allowed(self) -> None:
        labels = _label_set(("A", "B", "C", "D"), exclusive=False, allow_abstain=True)
        program = PrototypeClassify(labels=labels, embedder=_embedder(), threshold=0.99)
        result = await program(text="mid")
        # cosine ~0.707 against A and B; below 0.99.
        assert result.abstained is True
        assert result.metadata["abstain_reason"] == "no_label_above_threshold"

    @pytest.mark.asyncio
    async def test_high_threshold_falls_back_to_argmax_when_no_abstain(self) -> None:
        labels = _label_set(("A", "B", "C", "D"), exclusive=False, allow_abstain=False)
        program = PrototypeClassify(labels=labels, embedder=_embedder(), threshold=0.99)
        result = await program(text="ab")
        # "ab" scores ~0.8 against B, ~0.6 against A; no label clears
        # 0.99, so we fall back to argmax = B.
        assert result.abstained is False
        names = [label.name for label in result.labels]
        assert names == ["B"]


class TestPrototypeClassifyEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_input_abstains(self) -> None:
        labels = _label_set(("A", "B"), exclusive=True, allow_abstain=True)
        program = PrototypeClassify(labels=labels, embedder=_embedder())
        result = await program(text="")
        assert result.abstained is True
        assert result.scores == {ABSTAIN_LABEL: 0.0}

    @pytest.mark.asyncio
    async def test_empty_input_falls_back_when_no_abstain(self) -> None:
        labels = _label_set(("A", "B"), exclusive=True, allow_abstain=False)
        program = PrototypeClassify(labels=labels, embedder=_embedder())
        result = await program(text="")
        assert result.abstained is False
        # First-label deterministic fallback.
        assert result.labels[0].name == "A"
        assert result.metadata["fallback_reason"] == "empty_input_no_abstain"

    @pytest.mark.asyncio
    async def test_parent_id_recorded_in_chunks_used(self) -> None:
        labels = _label_set(("A", "B"), exclusive=True)
        program = PrototypeClassify(labels=labels, embedder=_embedder())
        result = await program(text="ay", parent_id="doc-7")
        assert "doc-7" in result.chunks_used

    @pytest.mark.asyncio
    async def test_invocation_zero_usage(self) -> None:
        labels = _label_set(("A", "B"), exclusive=True)
        program = PrototypeClassify(labels=labels, embedder=_embedder())
        invocation = await program.invoke(text="ay")
        # No LLM call -> zero usage.
        assert invocation.usage.input_tokens == 0
        assert invocation.usage.output_tokens == 0

    def test_threshold_must_be_in_range(self) -> None:
        labels = _label_set(("A", "B"), exclusive=False)
        with pytest.raises(ValueError, match=r"threshold must be in \[-1, 1\]"):
            PrototypeClassify(labels=labels, embedder=_embedder(), threshold=1.5)

    def test_min_score_must_be_in_range(self) -> None:
        labels = _label_set(("A", "B"), exclusive=True)
        with pytest.raises(ValueError, match=r"min_score must be in \[-1, 1\]"):
            PrototypeClassify(labels=labels, embedder=_embedder(), min_score=2.0)

    @pytest.mark.asyncio
    async def test_label_with_no_description_uses_name(self) -> None:
        # Construct a LabelSet where labels have no description; prompt_text
        # falls back to the bare name. The embedder must therefore see
        # plain "A" / "B" as prototype inputs.
        labels = LabelSet(
            labels=[Label(name="A"), Label(name="B")],
            exclusive=True,
        )
        embedder = _StubEmbedder(
            {
                "A": _LABEL_VECTORS["description of A"],
                "B": _LABEL_VECTORS["description of B"],
                "ay": _INPUT_VECTORS["ay"],
            }
        )
        program = PrototypeClassify(labels=labels, embedder=embedder)
        result = await program(text="ay")
        assert result.top_label == "A"
        # Confirm the embedder saw the bare names, not the descriptions.
        assert embedder.calls[0] == ["A", "B"]
