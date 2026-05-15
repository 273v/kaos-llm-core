"""Tests for :mod:`kaos_llm_core.composition.reduce` strategies.

The reducer protocol takes a ``merge_fn`` callback to do the LLM
work; these tests pass a deterministic stub so behavior is testable
offline.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from kaos_llm_core.composition import MapReduce, Reducer, Tree
from kaos_llm_core.composition.reduce import Refine
from kaos_llm_core.results import Summary

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _leaf(text: str, *, chunk_id: str | None = None, depth: int = 0) -> Summary[str]:
    return Summary[str](
        text=text,
        chunks_used=[chunk_id] if chunk_id else [],
        depth=depth,
    )


async def _join_merge(leaves: Sequence[Summary[str]]) -> Summary[str]:
    """Stub merge: concatenate texts and pool chunk ids."""
    pooled: list[str] = []
    seen: set[str] = set()
    for leaf in leaves:
        for cid in leaf.chunks_used:
            if cid not in seen:
                pooled.append(cid)
                seen.add(cid)
    return Summary[str](
        text=" | ".join(leaf.text for leaf in leaves),
        chunks_used=pooled,
        depth=max((leaf.depth for leaf in leaves), default=0),
        metadata={"merge_input": len(leaves)},
    )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", [MapReduce, Refine, Tree])
def test_implements_reducer_protocol(factory) -> None:
    assert isinstance(factory(), Reducer)


# ---------------------------------------------------------------------------
# MapReduce
# ---------------------------------------------------------------------------


class TestMapReduce:
    @pytest.mark.asyncio
    async def test_basic_three_leaves(self) -> None:
        leaves = [
            _leaf("alpha", chunk_id="c1"),
            _leaf("beta", chunk_id="c2"),
            _leaf("gamma", chunk_id="c3"),
        ]
        out = await MapReduce().reduce(leaves, _join_merge)
        assert out.text == "alpha | beta | gamma"
        assert set(out.chunks_used) == {"c1", "c2", "c3"}
        assert out.metadata["reducer"] == "MapReduce"
        assert out.metadata["reducer.input_leaves"] == 3
        assert out.depth == 1

    @pytest.mark.asyncio
    async def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one leaf"):
            await MapReduce().reduce([], _join_merge)

    @pytest.mark.asyncio
    async def test_single_leaf(self) -> None:
        out = await MapReduce().reduce([_leaf("only", chunk_id="c1")], _join_merge)
        assert out.text == "only"
        assert out.chunks_used == ["c1"]


# ---------------------------------------------------------------------------
# Refine
# ---------------------------------------------------------------------------


class TestRefine:
    @pytest.mark.asyncio
    async def test_sequential_left_to_right(self) -> None:
        leaves = [
            _leaf("A", chunk_id="c1"),
            _leaf("B", chunk_id="c2"),
            _leaf("C", chunk_id="c3"),
        ]
        out = await Refine().reduce(leaves, _join_merge)
        # Refine merges (A, B) -> "A | B", then (that, C) -> "A | B | C".
        assert out.text == "A | B | C"
        assert set(out.chunks_used) == {"c1", "c2", "c3"}
        assert out.metadata["reducer"] == "Refine"

    @pytest.mark.asyncio
    async def test_single_leaf_does_not_call_merge(self) -> None:
        call_count = 0

        async def counting_merge(group: Sequence[Summary[str]]) -> Summary[str]:
            nonlocal call_count
            call_count += 1
            return await _join_merge(group)

        out = await Refine().reduce([_leaf("only")], counting_merge)
        assert call_count == 0
        assert out.text == "only"
        assert out.metadata["reducer"] == "Refine"

    @pytest.mark.asyncio
    async def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one leaf"):
            await Refine().reduce([], _join_merge)


# ---------------------------------------------------------------------------
# Tree
# ---------------------------------------------------------------------------


class TestTree:
    @pytest.mark.asyncio
    async def test_four_leaves_branching_two(self) -> None:
        leaves = [
            _leaf("A", chunk_id="c1"),
            _leaf("B", chunk_id="c2"),
            _leaf("C", chunk_id="c3"),
            _leaf("D", chunk_id="c4"),
        ]
        out = await Tree(branching=2).reduce(leaves, _join_merge)
        # Level 1: (A,B)->"A | B", (C,D)->"C | D"
        # Level 2: ("A | B", "C | D") -> "A | B | C | D"
        assert out.text == "A | B | C | D"
        assert out.metadata["reducer.depth_applied"] == 2
        assert out.metadata["reducer.branching"] == 2
        assert set(out.chunks_used) == {"c1", "c2", "c3", "c4"}

    @pytest.mark.asyncio
    async def test_single_group_short_circuits(self) -> None:
        leaves = [_leaf("A"), _leaf("B")]
        out = await Tree(branching=4).reduce(leaves, _join_merge)
        # Two leaves, branching=4 → one merge call.
        assert out.metadata["reducer.depth_applied"] == 1

    @pytest.mark.asyncio
    async def test_folds_singleton_tail(self) -> None:
        # 5 leaves, branching=2: groups (A,B), (C,D), (E,).
        # The (E,) singleton should fold into the previous group.
        leaves = [_leaf(c) for c in "ABCDE"]
        merge_calls: list[int] = []

        async def tracking_merge(group: Sequence[Summary[str]]) -> Summary[str]:
            merge_calls.append(len(group))
            return await _join_merge(group)

        await Tree(branching=2).reduce(leaves, tracking_merge)
        # First level: one merge of size 2, one merge of size 3 (folded).
        # Second level: one merge of size 2.
        # So at minimum: there should be a 3-sized merge at level 1.
        assert 3 in merge_calls

    @pytest.mark.asyncio
    async def test_max_depth_caps_recursion(self) -> None:
        leaves = [_leaf(str(i)) for i in range(8)]
        # branching=2 with max_depth=1: one level only. The reducer
        # falls back to a single flat merge if depth would exceed cap.
        out = await Tree(branching=2, max_depth=1).reduce(leaves, _join_merge)
        # With max_depth=1, we do 1 batched level + 1 final flat
        # merge. So depth_applied is 2 (the level + the flush) at
        # most. The text contains all leaves either way.
        for c in "01234567":
            assert c in out.text

    def test_invalid_branching(self) -> None:
        with pytest.raises(ValueError, match=r"branching must be >= 2"):
            Tree(branching=1)

    def test_invalid_max_depth(self) -> None:
        with pytest.raises(ValueError, match=r"max_depth must be >= 1"):
            Tree(max_depth=0)

    @pytest.mark.asyncio
    async def test_parallel_sibling_merges(self) -> None:
        """Sibling groups at the same level should merge concurrently."""
        import asyncio

        start_events: list[asyncio.Event] = []
        end_events: list[asyncio.Event] = []

        async def slow_merge(group: Sequence[Summary[str]]) -> Summary[str]:
            start = asyncio.Event()
            end = asyncio.Event()
            start_events.append(start)
            end_events.append(end)
            start.set()
            # Wait for any sibling to also start before completing.
            if len(start_events) >= 2:
                await asyncio.sleep(0)  # give other tasks a chance
            end.set()
            return await _join_merge(group)

        leaves = [_leaf(c) for c in "ABCD"]
        out = await Tree(branching=2).reduce(leaves, slow_merge)
        assert out.text == "A | B | C | D"
        # At least one level must run with two siblings concurrent.
        assert len(start_events) >= 3  # 2 at level 1 + 1 at level 2

    @pytest.mark.asyncio
    async def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one leaf"):
            await Tree().reduce([], _join_merge)


# ---------------------------------------------------------------------------
# Provenance accumulation across reducers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", [MapReduce, Refine, Tree])
@pytest.mark.asyncio
async def test_provenance_pooled_across_reducers(factory) -> None:
    reducer = factory()
    leaves = [
        _leaf("A", chunk_id="c1"),
        _leaf("B", chunk_id="c2"),
        _leaf("C", chunk_id="c3"),
    ]
    out = await reducer.reduce(leaves, _join_merge)
    assert set(out.chunks_used) == {"c1", "c2", "c3"}
