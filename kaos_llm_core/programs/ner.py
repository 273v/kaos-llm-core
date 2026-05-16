"""Zero-shot named-entity recognition via GLiNER-style span extractors.

:class:`GLiNERExtract` is the lightweight NER counterpart to
:class:`~kaos_llm_core.programs.classify.ZeroShotNLIClassifier`:
LLM-free, deterministic, runs locally through an injected
:class:`NerExtractor` Protocol implementation. Plan §4.2.4 / Phase 8.

The NER model itself is **not** bundled in :mod:`kaos_llm_core` — the
canonical implementation lives in
:class:`kaos_nlp_transformers.GLiNERExtractor`, whose
``.extract(texts, labels, *, threshold, max_width, ...)`` method
satisfies the :class:`NerExtractor` Protocol declared below at
runtime. Until that ships, callers wire any object that produces the
expected span dicts — including offline stubs and existing local
GLiNER-style libraries.

Decision rule:

- Call the extractor with ``([source_text], labels)`` plus the
  per-Program tuning (threshold / max_width / flat_ner / dup_label /
  multi_label).
- Marshal the returned objects into
  :class:`~kaos_llm_core.results.EntitySpan` records preserving byte
  offsets, label, and confidence.
- Sort spans by ``(start, end)`` ascending so downstream code sees a
  deterministic order.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from kaos_llm_core.programs.base import Program
from kaos_llm_core.results import Entities, EntitySpan


@runtime_checkable
class EntityResult(Protocol):
    """Structural type for one entity span returned by an NER backend.

    Mirrors :class:`kaos_nlp_transformers.Entity`: byte-offset half-open
    ``[start, end)`` span into the source text, decoded substring,
    label name, and a ``[0, 1]`` confidence score.
    """

    start: int
    end: int
    text: str
    label: str
    score: float


@runtime_checkable
class NerExtractor(Protocol):
    """Structural type for a zero-shot NER extractor.

    Implementations score a batch of input texts against a label list
    in one call so the model can amortise its forward pass;
    :class:`GLiNERExtract` issues exactly one :meth:`extract` call per
    :meth:`forward` (batch size = 1).
    """

    def extract(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
        *,
        threshold: float = 0.5,
        max_width: int = 12,
        flat_ner: bool = True,
        dup_label: bool = False,
        multi_label: bool = False,
    ) -> Sequence[Sequence[EntityResult]]:  # pragma: no cover - protocol
        ...


class GLiNERExtract(Program):
    """Zero-shot named-entity recognition via a GLiNER-style extractor.

    Args:
        extractor: Object conforming to :class:`NerExtractor`. The
            canonical implementation is
            :class:`kaos_nlp_transformers.GLiNERExtractor`.
        labels: Entity-class labels to look for (e.g.
            ``["person", "organization", "place"]``). Custom domain
            labels work because GLiNER is zero-shot —
            ``["medical condition", "drug name", "dosage"]`` is fine.
        threshold: Minimum confidence (sigmoid-normalized) to accept a
            span. Default 0.5 — matches the upstream Python
            ``GLiNER.predict_entities`` default.
        max_width: Maximum span width in words.
        flat_ner: If ``True`` (default), no two output spans overlap.
            If ``False``, ``dup_label`` and ``multi_label`` decide
            whether same-label / different-label overlaps are kept.
        dup_label: Permit overlapping spans with the SAME label (only
            effective when ``flat_ner=False``).
        multi_label: Permit overlapping spans with DIFFERENT labels
            (only effective when ``flat_ner=False``).

    The Program holds no :class:`~kaos_llm_core.programs.call.Call`
    children — it runs entirely on the supplied extractor.
    :meth:`Program.invoke` builds a childless trace with zero token
    usage so the surface is type-stable with the LLM-backed
    extractors.

    Returns:
        :class:`~kaos_llm_core.results.Entities` carrying a list of
        :class:`~kaos_llm_core.results.EntitySpan` records sorted by
        source-text byte offset. ``metadata["program"]`` is
        ``"GLiNERExtract"`` and ``metadata["extractor"]`` is the
        backend class name.
    """

    def __init__(
        self,
        *,
        extractor: NerExtractor,
        labels: Sequence[str],
        threshold: float = 0.5,
        max_width: int = 12,
        flat_ner: bool = True,
        dup_label: bool = False,
        multi_label: bool = False,
    ) -> None:
        if not labels:
            raise ValueError("`labels` must contain at least one entry")
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        if max_width < 1:
            raise ValueError(f"max_width must be >= 1, got {max_width}")
        self._extractor = extractor
        self._labels = tuple(labels)
        self._threshold = float(threshold)
        self._max_width = int(max_width)
        self._flat_ner = bool(flat_ner)
        self._dup_label = bool(dup_label)
        self._multi_label = bool(multi_label)

    async def forward(  # ty: ignore[invalid-method-override]
        self,
        *,
        text: str,
        parent_id: str | None = None,
    ) -> Entities:
        if not text:
            return Entities(
                spans=[],
                labels=list(self._labels),
                metadata={
                    "program": "GLiNERExtract",
                    "extractor": type(self._extractor).__name__,
                    "skip_reason": "empty_input",
                    "parent_id": parent_id,
                },
            )

        raw = self._extractor.extract(
            [text],
            list(self._labels),
            threshold=self._threshold,
            max_width=self._max_width,
            flat_ner=self._flat_ner,
            dup_label=self._dup_label,
            multi_label=self._multi_label,
        )
        # Backend contract: list of length 1 (one batch element).
        per_text = list(raw[0]) if raw else []

        spans: list[EntitySpan] = []
        for ent in per_text:
            spans.append(
                EntitySpan(
                    start=int(ent.start),
                    end=int(ent.end),
                    text=str(ent.text),
                    label=str(ent.label),
                    score=float(ent.score),
                )
            )
        spans.sort(key=lambda s: (s.start, s.end))

        meta: dict[str, Any] = {
            "program": "GLiNERExtract",
            "extractor": type(self._extractor).__name__,
            "threshold": self._threshold,
            "max_width": self._max_width,
            "flat_ner": self._flat_ner,
            "dup_label": self._dup_label,
            "multi_label": self._multi_label,
            "n_spans": len(spans),
        }
        if parent_id is not None:
            meta["parent_id"] = parent_id
        return Entities(
            spans=spans,
            labels=list(self._labels),
            metadata=meta,
        )


__all__ = ["EntityResult", "GLiNERExtract", "NerExtractor"]
