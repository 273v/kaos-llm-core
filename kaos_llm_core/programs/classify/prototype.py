"""No-LLM prototype classification via embedding cosine.

:class:`PrototypeClassify` embeds the input plus each label's prototype
text (``Label.description`` falling back to ``Label.name``) and scores
each label by cosine similarity against the input. No LLM call is
made; the Program runs entirely on the embedder supplied by the
caller.

The canonical embedder is
:class:`kaos_nlp_transformers.EmbeddingModel`, but any object that
conforms to the :class:`Embedder` protocol works (offline tests
substitute a deterministic stub). Cosine routing goes through the
Rust-backed
:func:`kaos_nlp_core.similarity.cosine_one_to_many_normalized` fast
path — the embedder contract guarantees unit-norm rows on the canonical
implementation, so the per-row norm computation is skipped.

Behaviour by :attr:`LabelSet.exclusive`:

- ``exclusive=True`` (default): single-label mode. The highest-cosine
  label wins. An optional ``min_score`` floor abstains when the top
  cosine falls below the threshold and the LabelSet permits
  abstention.
- ``exclusive=False``: multi-label mode. Every label whose cosine is
  ``>= threshold`` is included; when no label clears the threshold,
  the Program abstains (when permitted) or returns the argmax.

This Program is the no-LLM half of the §6.2 classification taxonomy
and the foundation for :class:`RetrievalClassify` (Phase 6).

.. warning::
   Prototype classification is deterministic given the embedder, but
   the embedder itself depends on a pinned model revision. Treat
   prototype output as triage signal: it's fast, cheap, and
   reproducible, but its accuracy depends on how well the
   ``description`` strings reflect the actual decision boundary. Do
   not rely on prototype classifications for legal, financial,
   medical, or compliance decisions without human expert review.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

import numpy as np
from kaos_nlp_core.similarity import (
    cosine_one_to_many_normalized as _cosine_one_to_many_normalized,
)
from kaos_nlp_core.similarity import (
    l2_normalize_in_place as _l2_normalize_in_place,
)

from kaos_llm_core.labels import ABSTAIN_LABEL, Label, LabelSet
from kaos_llm_core.programs.base import Program
from kaos_llm_core.results import Classification


@runtime_checkable
class Embedder(Protocol):
    """Structural type for an embedding backend.

    Mirrors :class:`kaos_nlp_transformers.EmbeddingModel` so the
    canonical implementation satisfies the protocol without an
    explicit ``Embedder`` import. Stubs used in offline tests conform
    to the same shape.

    The :meth:`embed` method must return a ``(N, dim)`` ``float32``
    numpy array. **Rows must be L2-normalised** — the cosine fast path
    in :class:`PrototypeClassify` requires unit-norm inputs. When
    :class:`PrototypeClassify` is constructed with ``normalize=True``
    (default), it defends against drift by L2-normalising the returned
    rows in place; pass ``normalize=False`` to skip the defensive
    pass when the embedder's contract is trusted.
    """

    def embed(
        self, texts: Iterable[str], *, batch_size: int = 32
    ) -> np.ndarray:  # pragma: no cover - protocol
        ...


def _to_contiguous_f32(arr: np.ndarray) -> np.ndarray:
    """Coerce ``arr`` to a C-contiguous ``float32`` view.

    The Rust similarity bindings require contiguous ``f32`` buffers;
    this helper makes the boundary explicit and minimises hidden
    copies.
    """
    out = arr if arr.dtype == np.float32 else arr.astype(np.float32, copy=False)
    if not out.flags["C_CONTIGUOUS"]:
        out = np.ascontiguousarray(out)
    return out


def _normalize_rows_in_place(matrix: np.ndarray) -> None:
    """L2-normalise every row of a 2-D float32 matrix in place.

    The Rust :func:`l2_normalize_in_place` works on 1-D vectors;
    apply it row-by-row. Zero-norm rows are left untouched (the
    primitive yields ``0.0`` cosine for them).
    """
    for row in matrix:
        _l2_normalize_in_place(row)


class PrototypeClassify(Program):
    """No-LLM classifier driven by embedding cosine against label prototypes.

    Workflow on the first :meth:`forward` call:

    1. Embed every label's :attr:`Label.prompt_text` once and cache
       the resulting unit-norm prototype matrix.

    Workflow on every :meth:`forward` call:

    2. Embed the input text.
    3. Score via
       :func:`kaos_nlp_core.similarity.cosine_one_to_many_normalized`.
    4. Apply the picking rule (argmax + ``min_score``, or threshold
       + abstention).

    Args:
        labels: Label space. Exclusive and multi-label sets are both
            supported; the picking rule branches on
            :attr:`LabelSet.exclusive`.
        embedder: Object conforming to :class:`Embedder`. The canonical
            implementation is
            :class:`kaos_nlp_transformers.EmbeddingModel`.
        threshold: Multi-label cosine threshold in ``[-1, 1]``. Labels
            with cosine ``>= threshold`` are picked. Ignored when
            ``labels.exclusive`` is ``True``. Default ``0.5``.
        min_score: Exclusive-mode abstention floor in ``[-1, 1]``. When
            the top cosine is strictly less than ``min_score`` and the
            LabelSet allows abstention, the Program abstains. ``None``
            (default) disables the floor — the argmax always wins.
        normalize: When ``True`` (default), defensively L2-normalise
            every embedded row in place before the cosine call. Set
            ``False`` to trust the embedder's unit-norm contract.

    The Program holds no :class:`~kaos_llm_core.programs.call.Call`
    children — :meth:`Program.invoke` still builds a (childless) trace
    and returns an :class:`~kaos_llm_core.programs._invocation.Invocation`
    with zero usage, so the surface is type-stable with the LLM-backed
    classifiers.
    """

    def __init__(
        self,
        *,
        labels: LabelSet,
        embedder: Embedder,
        threshold: float = 0.5,
        min_score: float | None = None,
        normalize: bool = True,
    ) -> None:
        if not (-1.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [-1, 1], got {threshold}")
        if min_score is not None and not (-1.0 <= min_score <= 1.0):
            raise ValueError(f"min_score must be in [-1, 1] or None, got {min_score}")
        self._label_set = labels
        self._embedder = embedder
        self._threshold = float(threshold)
        self._min_score = float(min_score) if min_score is not None else None
        self._normalize = normalize
        # Lazy prototype embedding cache; built on first forward().
        self._prototypes: np.ndarray | None = None

    def _ensure_prototypes(self) -> np.ndarray:
        if self._prototypes is not None:
            return self._prototypes
        prompts = [label.prompt_text for label in self._label_set]
        vectors = self._embedder.embed(prompts)
        vectors = _to_contiguous_f32(vectors)
        if self._normalize:
            _normalize_rows_in_place(vectors)
        self._prototypes = vectors
        return vectors

    def _abstain(self, *, parent_id: str | None, reason: str) -> Classification[Label]:
        return Classification[Label](
            labels=[],
            scores={ABSTAIN_LABEL: 0.0},
            abstained=True,
            chunks_used=[parent_id] if parent_id else [],
            metadata={
                "program": "PrototypeClassify",
                "embedder": type(self._embedder).__name__,
                "abstain_reason": reason,
            },
        )

    async def forward(  # ty: ignore[invalid-method-override]
        self,
        *,
        text: str,
        parent_id: str | None = None,
    ) -> Classification[Label]:
        """Classify ``text`` against the cached label prototypes.

        Args:
            text: Source text to classify. Empty input abstains (when
                the LabelSet permits) or falls back to the lowest-index
                label.
            parent_id: Optional parent-chunk identifier, recorded in
                :attr:`Classification.chunks_used`.

        Returns:
            :class:`Classification` whose ``scores`` map carries one
            cosine value per label (using the canonical label-name
            keys). ``labels`` carries the picked :class:`Label`
            instance(s) — one for exclusive, the threshold-passing
            subset for multi-label. Metadata:

            - ``program``: ``"PrototypeClassify"``
            - ``embedder``: class name of the supplied embedder
            - ``threshold``: the multi-label threshold (always recorded)
            - ``min_score``: the exclusive-mode abstention floor or
              ``None``
            - ``top_score``: the highest cosine seen this call
        """
        if not text:
            if self._label_set.allow_abstain:
                return self._abstain(parent_id=parent_id, reason="empty_input")
            # No abstention allowed; cannot pick a label without an
            # input. Pick the first label as a deterministic fallback.
            first = self._label_set.labels[0]
            return Classification[Label](
                labels=[first],
                scores=dict.fromkeys(self._label_set.names, 0.0),
                abstained=False,
                chunks_used=[parent_id] if parent_id else [],
                metadata={
                    "program": "PrototypeClassify",
                    "embedder": type(self._embedder).__name__,
                    "threshold": self._threshold,
                    "min_score": self._min_score,
                    "top_score": 0.0,
                    "fallback_reason": "empty_input_no_abstain",
                },
            )

        prototypes = self._ensure_prototypes()
        # Embed exactly one row, take the first.
        query_matrix = self._embedder.embed([text])
        query_matrix = _to_contiguous_f32(query_matrix)
        query = query_matrix[0]
        if self._normalize:
            _l2_normalize_in_place(query)

        scores_arr = _cosine_one_to_many_normalized(query, prototypes)
        scores_arr = np.asarray(scores_arr, dtype=np.float32)

        score_map = {name: float(scores_arr[i]) for i, name in enumerate(self._label_set.names)}
        top_idx = int(np.argmax(scores_arr))
        top_score = float(scores_arr[top_idx])
        base_meta: dict[str, Any] = {
            "program": "PrototypeClassify",
            "embedder": type(self._embedder).__name__,
            "threshold": self._threshold,
            "min_score": self._min_score,
            "top_score": top_score,
        }

        if self._label_set.exclusive:
            if (
                self._min_score is not None
                and top_score < self._min_score
                and self._label_set.allow_abstain
            ):
                return Classification[Label](
                    labels=[],
                    scores=score_map,
                    abstained=True,
                    chunks_used=[parent_id] if parent_id else [],
                    metadata={
                        **base_meta,
                        "abstain_reason": "below_min_score",
                    },
                )
            picked_label = self._label_set.by_name(self._label_set.names[top_idx])
            return Classification[Label](
                labels=[picked_label],
                scores=score_map,
                abstained=False,
                chunks_used=[parent_id] if parent_id else [],
                metadata=base_meta,
            )

        # Multi-label: every label whose cosine clears the threshold.
        # Preserve LabelSet declaration order in the returned ``labels``
        # list so the result is deterministic across runs.
        picked_names = [
            name for i, name in enumerate(self._label_set.names) if scores_arr[i] >= self._threshold
        ]
        if not picked_names:
            if self._label_set.allow_abstain:
                return Classification[Label](
                    labels=[],
                    scores=score_map,
                    abstained=True,
                    chunks_used=[parent_id] if parent_id else [],
                    metadata={
                        **base_meta,
                        "abstain_reason": "no_label_above_threshold",
                    },
                )
            picked_names = [self._label_set.names[top_idx]]
        picked = [self._label_set.by_name(n) for n in picked_names]
        return Classification[Label](
            labels=picked,
            scores=score_map,
            abstained=False,
            chunks_used=[parent_id] if parent_id else [],
            metadata=base_meta,
        )


__all__ = [
    "Embedder",
    "PrototypeClassify",
]
