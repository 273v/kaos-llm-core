"""Aggregator strategy classes for chunked classification.

Wraps the pure aggregation functions in
:mod:`kaos_nlp_core.aggregation` with a typed strategy interface that
operates on :class:`~kaos_llm_core.results.Classification` instances
and a :class:`~kaos_llm_core.labels.LabelSet`.

Why a strategy layer:
    Programs like ``ChunkedClassify`` and ``EnsembleClassify`` are
    parameterized over the aggregation rule. Passing an
    :class:`Aggregator` instance instead of a string ("majority",
    "union", …) gives type-safe configuration, lets aggregators carry
    parameters (thresholds, weights), and keeps the program code
    free of large if/elif chains.

What's deliberately not here:
    The aggregation math itself. All counting happens in
    ``kaos_nlp_core.aggregation``; this module only translates
    :class:`Classification`/:class:`LabelSet` into the lighter inputs
    those functions expect (sequences of label-name iterables and/or
    score maps) and translates the result back into a
    :class:`Classification`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

from kaos_nlp_core.aggregation import (
    intersection,
    majority,
    max_score,
    union,
    vote,
    weighted,
)

from kaos_llm_core.labels import Label, LabelSet
from kaos_llm_core.results import Classification, SourceSpan


def _names_from_classification(c: Classification[Label] | Classification[str]) -> list[str]:
    """Extract label names from a :class:`Classification` regardless of payload type."""
    return c.names


def _merge_provenance(
    per_chunk: Sequence[Classification[Label] | Classification[str]],
) -> tuple[list[str], list[SourceSpan], dict[str, object]]:
    """Pool ``chunks_used``, ``source_spans``, and ``metadata`` across chunks.

    Returns
        ``(chunks_used, source_spans, metadata)``. ``metadata``
        carries the count of contributing chunks and a histogram of
        per-chunk picks for debuggability.
    """
    chunks_used: list[str] = []
    source_spans: list[SourceSpan] = []
    picks_histogram: dict[str, int] = {}
    for c in per_chunk:
        chunks_used.extend(c.chunks_used)
        source_spans.extend(c.source_spans)
        for name in c.names:
            picks_histogram[name] = picks_histogram.get(name, 0) + 1
    metadata: dict[str, object] = {
        "aggregator.input_chunks": len(per_chunk),
        "aggregator.label_histogram": dict(picks_histogram),
    }
    return chunks_used, source_spans, metadata


def _labels_from_names(label_set: LabelSet, picks: Sequence[str]) -> list[Label]:
    """Return :class:`Label` instances for the given names."""
    return [label_set.by_name(name) for name in picks]


@runtime_checkable
class Aggregator(Protocol):
    """Strategy for combining per-chunk classifications.

    Implementations operate on :class:`Classification` instances whose
    label payload is either :class:`Label` (recommended) or :class:`str`.
    The returned :class:`Classification` carries the same payload type
    as the inputs.

    The protocol is :func:`runtime_checkable` so callers can use
    ``isinstance(obj, Aggregator)`` to gate runtime hook-in.
    """

    def combine(
        self,
        per_chunk: Sequence[Classification[Label]],
        label_set: LabelSet,
    ) -> Classification[Label]:
        """Combine ``per_chunk`` into a single document-level result.

        Args:
            per_chunk: Classifications produced per chunk, in source
                order.
            label_set: The label space the underlying classifier was
                configured with. Used to (a) resolve names back to
                :class:`Label` records and (b) honor the
                ``exclusive`` / ``allow_abstain`` policy flags.

        Returns:
            A single :class:`Classification` with provenance pooled
            from the inputs.
        """
        ...


class _ExclusiveAggregator:
    """Helper shared by exclusive (single-label) strategies."""

    def _build(
        self,
        winner: str | None,
        per_chunk: Sequence[Classification[Label]],
        label_set: LabelSet,
        scores: dict[str, float] | None = None,
    ) -> Classification[Label]:
        chunks_used, source_spans, metadata = _merge_provenance(per_chunk)
        if winner is None or winner not in label_set:
            return Classification[Label](
                labels=[],
                scores=scores or {},
                abstained=True,
                chunks_used=chunks_used,
                source_spans=source_spans,
                metadata=metadata,
            )
        return Classification[Label](
            labels=[label_set.by_name(winner)],
            scores=scores or {},
            abstained=False,
            chunks_used=chunks_used,
            source_spans=source_spans,
            metadata=metadata,
        )


class VoteAggregator(_ExclusiveAggregator):
    """Plurality voting: pick the label appearing in the most chunks.

    Ties are broken by first appearance. Returns an abstained
    :class:`Classification` when ``per_chunk`` is empty or when the
    plurality label is not in ``label_set`` (defensive).
    """

    def combine(
        self,
        per_chunk: Sequence[Classification[Label]],
        label_set: LabelSet,
    ) -> Classification[Label]:
        per_chunk_names = [_names_from_classification(c) for c in per_chunk]
        winner = vote(per_chunk_names)
        return self._build(winner, per_chunk, label_set)


class MajorityAggregator(_ExclusiveAggregator):
    """Threshold-gated majority voting.

    Returns an abstained classification when no label meets
    ``threshold``.
    """

    def __init__(self, *, threshold: float = 0.5) -> None:
        if not (0.0 < threshold <= 1.0):
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        self.threshold = threshold

    def combine(
        self,
        per_chunk: Sequence[Classification[Label]],
        label_set: LabelSet,
    ) -> Classification[Label]:
        per_chunk_names = [_names_from_classification(c) for c in per_chunk]
        winner = majority(per_chunk_names, threshold=self.threshold)
        return self._build(winner, per_chunk, label_set)


class _MultiLabelAggregator:
    """Helper shared by multi-label strategies."""

    def _build(
        self,
        picks: frozenset[str],
        per_chunk: Sequence[Classification[Label]],
        label_set: LabelSet,
        scores: dict[str, float] | None = None,
    ) -> Classification[Label]:
        # Preserve order of first appearance in ``label_set`` so the
        # returned classification is stable across runs.
        ordered = [name for name in label_set.names if name in picks]
        chunks_used, source_spans, metadata = _merge_provenance(per_chunk)
        return Classification[Label](
            labels=_labels_from_names(label_set, ordered),
            scores=scores or {},
            abstained=not ordered,
            chunks_used=chunks_used,
            source_spans=source_spans,
            metadata=metadata,
        )


class UnionAggregator(_MultiLabelAggregator):
    """Multi-label union: any label appearing in any chunk wins.

    High recall, low precision. Use when ``label_set.exclusive=False``
    and the document may contain content for multiple topics.
    """

    def combine(
        self,
        per_chunk: Sequence[Classification[Label]],
        label_set: LabelSet,
    ) -> Classification[Label]:
        per_chunk_names = [_names_from_classification(c) for c in per_chunk]
        picks = union(per_chunk_names)
        return self._build(picks, per_chunk, label_set)


class IntersectionAggregator(_MultiLabelAggregator):
    """Multi-label intersection: a label must appear in *every* chunk.

    High precision, low recall. Use when chunks are essentially
    redundant views of the same content and disagreement should be
    treated as uncertainty.
    """

    def combine(
        self,
        per_chunk: Sequence[Classification[Label]],
        label_set: LabelSet,
    ) -> Classification[Label]:
        per_chunk_names = [_names_from_classification(c) for c in per_chunk]
        picks = intersection(per_chunk_names)
        return self._build(picks, per_chunk, label_set)


class WeightedAggregator(_MultiLabelAggregator, _ExclusiveAggregator):
    """Weighted aggregation honoring per-chunk weights.

    The weight associated with each chunk defaults to ``1.0`` per
    chunk. Callers can also pass a per-chunk weight extractor that
    derives a weight from the chunk's classification (e.g., its
    top-label confidence).

    Operates in single-label or multi-label mode depending on the
    ``label_set.exclusive`` flag.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        chunk_weight: Callable[[Classification[Label]], float] | None = None,
    ) -> None:
        if not (0.0 < threshold <= 1.0):
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        self.threshold = threshold
        self.chunk_weight = chunk_weight

    def combine(
        self,
        per_chunk: Sequence[Classification[Label]],
        label_set: LabelSet,
    ) -> Classification[Label]:
        per_chunk_names = [_names_from_classification(c) for c in per_chunk]
        if self.chunk_weight is None:
            weights: list[float] | None = None
        else:
            weights = [self.chunk_weight(c) for c in per_chunk]
        result = weighted(
            per_chunk_names,
            weights=weights,
            threshold=self.threshold,
            multi=not label_set.exclusive,
        )
        if label_set.exclusive:
            winner = result if isinstance(result, str) else None
            return _ExclusiveAggregator._build(self, winner, per_chunk, label_set)
        picks = result if isinstance(result, frozenset) else frozenset()
        return _MultiLabelAggregator._build(self, picks, per_chunk, label_set)


class MaxScoreAggregator(_MultiLabelAggregator, _ExclusiveAggregator):
    """Pool per-chunk score maps by taking the max score per label.

    Useful when underlying classifiers produce calibrated scores and
    the document-level decision should be the label with the highest
    single-chunk confidence.

    Operates in single-label or multi-label mode depending on the
    ``label_set.exclusive`` flag.
    """

    def __init__(self, *, threshold: float | None = None) -> None:
        self.threshold = threshold

    def combine(
        self,
        per_chunk: Sequence[Classification[Label]],
        label_set: LabelSet,
    ) -> Classification[Label]:
        score_maps = [dict(c.scores) for c in per_chunk]
        pooled: dict[str, float] = {}
        for score_map in score_maps:
            for name, value in score_map.items():
                if name not in pooled or value > pooled[name]:
                    pooled[name] = value
        result = max_score(
            score_maps,
            multi=not label_set.exclusive,
            threshold=self.threshold,
        )
        if label_set.exclusive:
            winner = result if isinstance(result, str) else None
            return _ExclusiveAggregator._build(self, winner, per_chunk, label_set, pooled)
        picks = result if isinstance(result, frozenset) else frozenset()
        return _MultiLabelAggregator._build(self, picks, per_chunk, label_set, pooled)


__all__ = [
    "Aggregator",
    "IntersectionAggregator",
    "MajorityAggregator",
    "MaxScoreAggregator",
    "UnionAggregator",
    "VoteAggregator",
    "WeightedAggregator",
]
