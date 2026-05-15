"""Retrieval-augmented classification (plan §6.2).

:class:`RetrievalClassify` is the many-shot embedding classifier:
given a labeled corpus ``[(example_text, label_name), …]`` and an
input, embed the input, retrieve the k nearest neighbours by cosine,
and vote weighted by similarity. Compared to
:class:`~kaos_llm_core.programs.classify.PrototypeClassify` (which
embeds one prototype per label), this Program embeds *many* examples
per label and lets the corpus structure drive the decision boundary.

Optional LLM tie-break: when the top label's weighted score is
within ``tie_break_margin`` of the runner-up, the Program defers to
a supplied tie-break classifier (typically
:class:`ZeroShotClassify`) and reports its decision. When no
tie-break is configured, the Program returns its argmax and tags
``metadata["close_call"] = True``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from kaos_nlp_core.similarity import (
    cosine_one_to_many_normalized as _cosine_one_to_many_normalized,
)
from kaos_nlp_core.similarity import (
    l2_normalize_in_place as _l2_normalize_in_place,
)

from kaos_llm_core.errors import CallError
from kaos_llm_core.labels import ABSTAIN_LABEL, Label, LabelSet
from kaos_llm_core.programs.base import Program
from kaos_llm_core.programs.classify.prototype import Embedder
from kaos_llm_core.results import Classification


def _to_contiguous_f32(arr: np.ndarray) -> np.ndarray:
    out = arr if arr.dtype == np.float32 else arr.astype(np.float32, copy=False)
    if not out.flags["C_CONTIGUOUS"]:
        out = np.ascontiguousarray(out)
    return out


class RetrievalClassify(Program):
    """kNN classifier over a labeled corpus, optional LLM tie-break.

    Args:
        labels: Label space. Must be ``exclusive=True``; multi-label
            kNN with thresholding is a future extension.
        embedder: A :class:`Embedder` (canonical:
            ``kaos_nlp_transformers.EmbeddingModel``).
        corpus: Labeled examples as ``(text, label_name)`` pairs. Every
            ``label_name`` must be in ``labels``. The corpus is
            embedded once on the first :meth:`forward` call and
            cached.
        k: Number of nearest neighbours to consult. Default ``5``.
            Capped at ``len(corpus)`` at runtime.
        tie_break: Optional :class:`Program` whose ``forward`` accepts
            ``text=`` (and ``parent_id=``) and returns a
            :class:`Classification`. Called when the weighted top-1
            score is within ``tie_break_margin`` of the runner-up.
            Typical wiring:
            ``tie_break=ZeroShotClassify(labels=labels, model=…)``.
        tie_break_margin: Threshold for triggering the tie-break.
            Default ``0.05``. Set to ``0.0`` to disable the
            tie-break path entirely.
        normalize: Defensively L2-normalise the embedded rows. Default
            ``True``.

    Returns:
        :class:`Classification` whose ``labels`` is the picked
        ``Label`` (or empty + ``abstained=True`` when the corpus is
        empty), ``scores`` is the per-label weighted-vote score
        normalised to sum to ``1``, ``rationale`` carries a brief
        nearest-neighbour summary, and ``metadata`` records
        ``top_score``, ``runner_up_score``, ``close_call``, and
        ``tie_break.used`` when applicable.
    """

    def __init__(
        self,
        *,
        labels: LabelSet,
        embedder: Embedder,
        corpus: list[tuple[str, str]],
        k: int = 5,
        tie_break: Program | None = None,
        tie_break_margin: float = 0.05,
        normalize: bool = True,
    ) -> None:
        if not labels.exclusive:
            raise CallError(
                "RetrievalClassify requires an exclusive LabelSet. "
                "Multi-label kNN is not yet supported."
            )
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if not (0.0 <= tie_break_margin <= 2.0):
            raise ValueError(f"tie_break_margin must be in [0, 2], got {tie_break_margin}")
        # Validate that every example label is known.
        known_names = set(labels.names)
        unknown = sorted({label for _, label in corpus} - known_names)
        if unknown:
            raise CallError(
                f"RetrievalClassify corpus references unknown labels: {unknown!r}. "
                f"Allowed: {sorted(known_names)}"
            )
        self._label_set = labels
        self._embedder = embedder
        self._corpus = list(corpus)
        self._k = k
        self.tie_break = tie_break  # public attr → auto-registers as Program child
        self._tie_break_margin = float(tie_break_margin)
        self._normalize = normalize
        self._corpus_vecs: np.ndarray | None = None
        self._corpus_labels: list[str] = [label for _, label in corpus]

    def _ensure_corpus(self) -> np.ndarray:
        if self._corpus_vecs is not None:
            return self._corpus_vecs
        if not self._corpus:
            # Empty corpus is allowed at construction; abstention is
            # handled in :meth:`forward`.
            self._corpus_vecs = np.zeros((0, 0), dtype=np.float32)
            return self._corpus_vecs
        texts = [text for text, _ in self._corpus]
        vecs = _to_contiguous_f32(self._embedder.embed(texts))
        if self._normalize:
            for row in vecs:
                _l2_normalize_in_place(row)
        self._corpus_vecs = vecs
        return vecs

    async def forward(  # ty: ignore[invalid-method-override]
        self,
        *,
        text: str,
        parent_id: str | None = None,
    ) -> Classification[Label]:
        corpus_vecs = self._ensure_corpus()
        base_meta: dict[str, Any] = {
            "program": "RetrievalClassify",
            "embedder": type(self._embedder).__name__,
            "k_requested": self._k,
            "corpus.size": len(self._corpus),
        }

        if corpus_vecs.shape[0] == 0:
            return Classification[Label](
                labels=[],
                scores={ABSTAIN_LABEL: 0.0},
                abstained=True,
                chunks_used=[parent_id] if parent_id else [],
                metadata={**base_meta, "abstain_reason": "empty_corpus"},
            )

        if not text:
            if self._label_set.allow_abstain:
                return Classification[Label](
                    labels=[],
                    scores={ABSTAIN_LABEL: 0.0},
                    abstained=True,
                    chunks_used=[parent_id] if parent_id else [],
                    metadata={**base_meta, "abstain_reason": "empty_input"},
                )
            first = self._label_set.labels[0]
            return Classification[Label](
                labels=[first],
                scores=dict.fromkeys(self._label_set.names, 0.0),
                abstained=False,
                chunks_used=[parent_id] if parent_id else [],
                metadata={**base_meta, "fallback_reason": "empty_input_no_abstain"},
            )

        # Embed the input and score against the corpus.
        query_matrix = _to_contiguous_f32(self._embedder.embed([text]))
        query = query_matrix[0]
        if self._normalize:
            _l2_normalize_in_place(query)
        sims = np.asarray(
            _cosine_one_to_many_normalized(query, corpus_vecs),
            dtype=np.float32,
        )

        # Pick the top-k neighbours (cap at corpus size); preserve
        # cosine order for the rationale.
        cap = min(self._k, sims.shape[0])
        if cap == sims.shape[0]:
            top_idx = np.argsort(-sims)
        else:
            partition = np.argpartition(-sims, cap - 1)[:cap]
            top_idx = partition[np.argsort(-sims[partition])]
        top_idx = top_idx[:cap]

        # Weighted vote: each neighbour contributes ``max(sim, 0)`` to
        # its label's tally. (Negative cosines, which can occur on
        # opposing topics, contribute zero so they don't subtract
        # from on-axis votes.)
        weighted_scores: dict[str, float] = dict.fromkeys(self._label_set.names, 0.0)
        for idx in top_idx:
            sim = float(sims[idx])
            if sim < 0:
                sim = 0.0
            weighted_scores[self._corpus_labels[int(idx)]] += sim
        total = sum(weighted_scores.values())
        if total > 0:
            normalised = {k: v / total for k, v in weighted_scores.items()}
        else:
            # Every neighbour had non-positive cosine → vote-by-count
            # fallback so the result remains deterministic.
            counts: dict[str, int] = dict.fromkeys(self._label_set.names, 0)
            for idx in top_idx:
                counts[self._corpus_labels[int(idx)]] += 1
            total_count = sum(counts.values()) or 1
            normalised = {k: v / total_count for k, v in counts.items()}

        # Argmax with deterministic tie-break by LabelSet order.
        ordered = sorted(
            normalised.items(),
            key=lambda kv: (-kv[1], self._label_set.names.index(kv[0])),
        )
        top_name, top_score = ordered[0]
        runner_up_score = ordered[1][1] if len(ordered) > 1 else 0.0
        close_call = (top_score - runner_up_score) < self._tie_break_margin

        rationale_lines = [
            f"{self._corpus_labels[int(idx)]}  cos={float(sims[idx]):.3f}: "
            f"{self._corpus[int(idx)][0][:60]}{'…' if len(self._corpus[int(idx)][0]) > 60 else ''}"
            for idx in top_idx
        ]
        rationale = "kNN neighbours (rank desc):\n" + "\n".join(rationale_lines)
        meta: dict[str, Any] = {
            **base_meta,
            "top_label": top_name,
            "top_score": top_score,
            "runner_up_score": runner_up_score,
            "close_call": close_call,
            "neighbour_count": int(cap),
        }

        # Optional LLM tie-break.
        if close_call and self.tie_break is not None and self._tie_break_margin > 0:
            tie_result: Classification = await self.tie_break(text=text, parent_id=parent_id)
            tie_top = tie_result.top_label
            meta["tie_break.used"] = True
            meta["tie_break.program"] = type(self.tie_break).__name__
            meta["tie_break.label"] = tie_top
            if tie_top is None:
                # Tie-break abstained — fall through to the argmax.
                pass
            else:
                if tie_top in self._label_set:
                    return Classification[Label](
                        labels=[self._label_set.by_name(tie_top)],
                        scores=normalised,
                        abstained=False,
                        rationale=(tie_result.rationale or "") + "\n\n" + rationale,
                        chunks_used=[parent_id] if parent_id else [],
                        metadata=meta,
                    )

        picked = self._label_set.by_name(top_name)
        return Classification[Label](
            labels=[picked],
            scores=normalised,
            abstained=False,
            rationale=rationale,
            chunks_used=[parent_id] if parent_id else [],
            metadata=meta,
        )


__all__ = ["RetrievalClassify"]
