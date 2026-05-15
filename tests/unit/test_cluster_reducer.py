"""Tests for :class:`~kaos_llm_core.composition.reduce.Cluster`.

Offline + deterministic. A stub :class:`ClusterEmbedder` returns
hand-crafted unit vectors so the spherical-k-means assignment is
predictable; an async stub ``merge_fn`` records the per-call groups so
we can assert how leaves were partitioned.

This module also covers ``_spherical_kmeans`` directly via a
public-API black-box test (one cluster per orthogonal direction).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pytest

from kaos_llm_core.composition import Cluster
from kaos_llm_core.results import Summary


class _StubEmbedder:
    """ClusterEmbedder stub that maps an exact text to a pre-baked vector.

    Unknown inputs raise — the test always seeds every input it cares
    about.
    """

    def __init__(self, table: dict[str, np.ndarray]) -> None:
        self._table = {k: np.asarray(v, dtype=np.float32) for k, v in table.items()}
        self.calls: list[list[str]] = []

    def embed(self, texts: Iterable[str], *, batch_size: int = 32) -> np.ndarray:
        del batch_size  # unused
        out: list[np.ndarray] = []
        text_list = list(texts)
        self.calls.append(text_list)
        for t in text_list:
            if t not in self._table:
                raise KeyError(t)
            out.append(self._table[t])
        return np.stack(out, axis=0)


class _RecordingMerge:
    """Async ``merge_fn`` stub that records every call and returns a stub Summary."""

    def __init__(self) -> None:
        self.calls: list[list[Summary]] = []

    async def __call__(self, group: Sequence[Summary]) -> Summary:
        group_list = list(group)
        self.calls.append(group_list)
        # Concatenate the input texts so we can spot-check assignment.
        return Summary[str](
            text=" | ".join(s.text for s in group_list),
            method="abstractive",
            chunks_used=[c for s in group_list for c in s.chunks_used],
            metadata={"merge.input_count": len(group_list)},
        )


def _summary(text: str, chunk_id: str | None = None) -> Summary[str]:
    return Summary[str](
        text=text,
        method="abstractive",
        chunks_used=[chunk_id] if chunk_id else [],
    )


# Four canonical inputs, each pointing along a different axis. Spherical
# k-means with k=2 should pull (A1, A2) into one cluster and (B1, B2)
# into another since the within-axis cosine is 1.0 and across-axis 0.0.
_VECTORS = {
    "A1": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "A2": np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32),
    "B1": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    "B2": np.array([0.0, 0.0, 0.9, 0.1], dtype=np.float32),
}


class TestClusterReducer:
    @pytest.mark.asyncio
    async def test_two_clusters_partition_inputs(self) -> None:
        embedder = _StubEmbedder(_VECTORS)
        reducer = Cluster(embedder=embedder, k=2)
        leaves = [
            _summary("A1", "c1"),
            _summary("A2", "c2"),
            _summary("B1", "c3"),
            _summary("B2", "c4"),
        ]
        merge = _RecordingMerge()
        result = await reducer.reduce(leaves, merge)

        # Two cluster-merges + one final cross-cluster merge.
        assert len(merge.calls) == 3
        cluster_groups = merge.calls[:2]
        sizes = sorted(len(g) for g in cluster_groups)
        assert sizes == [2, 2]
        # Each cluster groups same-axis leaves.
        for group in cluster_groups:
            texts = {s.text for s in group}
            assert texts in ({"A1", "A2"}, {"B1", "B2"})
        # Final merge takes the cluster summaries.
        assert len(merge.calls[2]) == 2

        # Metadata records the clustering.
        assert result.metadata["reducer"] == "Cluster"
        assert result.metadata["reducer.k"] == 2
        assert sorted(result.metadata["reducer.cluster_sizes"]) == [2, 2]
        # Provenance pooled across leaves.
        assert sorted(result.chunks_used) == ["c1", "c2", "c3", "c4"]

    @pytest.mark.asyncio
    async def test_auto_k_resolves_to_sqrt_heuristic(self) -> None:
        # 4 leaves → sqrt(4) = 2 → k = 2.
        embedder = _StubEmbedder(_VECTORS)
        reducer = Cluster(embedder=embedder, k="auto")
        leaves = [_summary(name, f"c{i}") for i, name in enumerate(["A1", "A2", "B1", "B2"])]
        merge = _RecordingMerge()
        result = await reducer.reduce(leaves, merge)
        assert result.metadata["reducer.k_requested"] == "auto"
        # k=2 produced 2 cluster merges + final merge = 3 calls.
        assert len(merge.calls) == 3

    @pytest.mark.asyncio
    async def test_single_leaf_passthrough(self) -> None:
        # Cluster with one leaf returns it directly without merging.
        embedder = _StubEmbedder(_VECTORS)
        reducer = Cluster(embedder=embedder, k=2)
        only = _summary("A1", "c1")
        merge = _RecordingMerge()
        result = await reducer.reduce([only], merge)
        assert merge.calls == []
        assert result.text == "A1"
        assert result.metadata["reducer"] == "Cluster"
        assert result.metadata["reducer.k"] == 1

    @pytest.mark.asyncio
    async def test_k_capped_at_n(self) -> None:
        # Caller requested k=10 but only 4 inputs available.
        embedder = _StubEmbedder(_VECTORS)
        reducer = Cluster(embedder=embedder, k=10)
        leaves = [_summary(name, f"c{i}") for i, name in enumerate(["A1", "A2", "B1", "B2"])]
        merge = _RecordingMerge()
        result = await reducer.reduce(leaves, merge)
        # At most 4 clusters (one per leaf); k_requested reflects
        # the original ask, k reflects the realised non-empty count.
        assert result.metadata["reducer.k"] <= 4
        assert result.metadata["reducer.k_requested"] == 10

    @pytest.mark.asyncio
    async def test_empty_leaves_degenerate_to_single_merge(self) -> None:
        # All-empty leaves can't be clustered; the reducer should
        # fall back to a single merge across them.
        embedder = _StubEmbedder({})
        reducer = Cluster(embedder=embedder, k=3)
        leaves = [_summary("", f"c{i}") for i in range(3)]
        merge = _RecordingMerge()
        result = await reducer.reduce(leaves, merge)
        assert len(merge.calls) == 1
        assert len(merge.calls[0]) == 3
        assert result.metadata["reducer.degenerate"] == "all_empty"

    @pytest.mark.asyncio
    async def test_empty_input_raises(self) -> None:
        embedder = _StubEmbedder({})
        reducer = Cluster(embedder=embedder, k=2)
        with pytest.raises(ValueError, match="Cluster requires"):
            merge = _RecordingMerge()
            await reducer.reduce([], merge)

    def test_invalid_k_rejected(self) -> None:
        embedder = _StubEmbedder({})
        with pytest.raises(ValueError, match="k must be"):
            Cluster(embedder=embedder, k=0)

    def test_invalid_max_iter_rejected(self) -> None:
        embedder = _StubEmbedder({})
        with pytest.raises(ValueError, match="max_iter must be"):
            Cluster(embedder=embedder, k=2, max_iter=0)
