"""Tests for :class:`~kaos_llm_core.cache.chunk.InMemoryChunkCache`."""

from __future__ import annotations

import pytest

from kaos_llm_core.cache import ChunkCacheKey, InMemoryChunkCache


class TestInMemoryChunkCache:
    @pytest.mark.asyncio
    async def test_miss_returns_none(self) -> None:
        cache = InMemoryChunkCache()
        key = ChunkCacheKey(chunk_id="c1", program_name="AbstractiveSummary")
        assert await cache.get(key) is None
        assert cache.hit_count == 0
        assert cache.miss_count == 1

    @pytest.mark.asyncio
    async def test_set_then_get_hits(self) -> None:
        cache = InMemoryChunkCache()
        key = ChunkCacheKey(chunk_id="c1", program_name="AbstractiveSummary")
        await cache.set(key, {"value": 42})
        got = await cache.get(key)
        assert got == {"value": 42}
        assert cache.hit_count == 1

    @pytest.mark.asyncio
    async def test_program_name_discriminates(self) -> None:
        cache = InMemoryChunkCache()
        a = ChunkCacheKey(chunk_id="c1", program_name="AbstractiveSummary")
        b = ChunkCacheKey(chunk_id="c1", program_name="StructuredSummary")
        await cache.set(a, "from-A")
        assert await cache.get(b) is None
        assert await cache.get(a) == "from-A"

    @pytest.mark.asyncio
    async def test_model_hint_discriminates(self) -> None:
        cache = InMemoryChunkCache()
        haiku = ChunkCacheKey(
            chunk_id="c1", program_name="AbstractiveSummary", model_hint="anthropic:claude-haiku"
        )
        sonnet = ChunkCacheKey(
            chunk_id="c1", program_name="AbstractiveSummary", model_hint="anthropic:claude-sonnet"
        )
        await cache.set(haiku, "haiku-output")
        assert await cache.get(sonnet) is None
        assert await cache.get(haiku) == "haiku-output"

    @pytest.mark.asyncio
    async def test_fifo_eviction(self) -> None:
        cache = InMemoryChunkCache(max_entries=2)
        keys = [ChunkCacheKey(chunk_id=f"c{i}", program_name="P") for i in range(3)]
        await cache.set(keys[0], "a")
        await cache.set(keys[1], "b")
        await cache.set(keys[2], "c")
        # Oldest dropped.
        assert await cache.get(keys[0]) is None
        assert await cache.get(keys[1]) == "b"
        assert await cache.get(keys[2]) == "c"

    @pytest.mark.asyncio
    async def test_overwrite_refreshes_fifo_position(self) -> None:
        cache = InMemoryChunkCache(max_entries=2)
        a = ChunkCacheKey(chunk_id="c0", program_name="P")
        b = ChunkCacheKey(chunk_id="c1", program_name="P")
        c = ChunkCacheKey(chunk_id="c2", program_name="P")
        await cache.set(a, "1")
        await cache.set(b, "2")
        # Re-set a -> a becomes the newest; b is now the oldest.
        await cache.set(a, "1-fresh")
        await cache.set(c, "3")
        # b was evicted as the oldest after the re-set.
        assert await cache.get(b) is None
        assert await cache.get(a) == "1-fresh"
        assert await cache.get(c) == "3"

    def test_max_entries_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="max_entries must be > 0"):
            InMemoryChunkCache(max_entries=0)
