"""Tests for SemanticCache — embedding-based response dedup."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart

from kaos_llm_core.cache.semantic import SemanticCache, _cosine_similarity
from kaos_llm_core.programs.call import Call
from kaos_llm_core.signatures import InputField, OutputField, Signature


class ExtractSig(Signature):
    """Extract entities."""

    text: str = InputField(description="Input text")
    entities: list[str] = OutputField(description="Entity names")


def _json_response(data: dict[str, Any]) -> ProviderResponse:
    return ProviderResponse.model_construct(
        provider="function",
        model="function-test",
        raw={},
        parts=[ContentPart.model_construct(type="text", text=json.dumps(data))],
        usage=UsageInfo.model_construct(input_tokens=10, output_tokens=5, total_tokens=15),
        stop_reason="end_turn",
        status_code=200,
        response_headers={},
    )


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_similar_vectors(self) -> None:
        sim = _cosine_similarity([1.0, 0.1], [1.0, 0.0])
        assert sim > 0.99

    def test_zero_vector(self) -> None:
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


class TestSemanticCache:
    def test_initial_state(self) -> None:
        cache = SemanticCache()
        assert cache.size == 0
        assert cache.total_hits == 0

    def test_clear(self) -> None:
        cache = SemanticCache()
        cache._entries.append(None)  # ty: ignore[invalid-argument-type]
        cache.clear()
        assert cache.size == 0

    async def test_cache_miss_calls_llm(self) -> None:
        """On cache miss, the Call should be invoked."""
        call_count = 0

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            nonlocal call_count
            call_count += 1
            return _json_response({"entities": ["X"]})

        client = FunctionClient(function=fn)
        call = Call(ExtractSig, model="function-test", client=client)

        cache = SemanticCache(similarity_threshold=0.99)
        # Mock the embedding to avoid needing a real embedding API
        cache._embed = AsyncMock(return_value=[1.0, 0.0, 0.0])  # ty: ignore[invalid-assignment]

        result = await cache.call(call, text="hello world")
        assert result.entities == ["X"]
        assert call_count == 1
        assert cache.size == 1

    async def test_cache_hit_skips_llm(self) -> None:
        """On cache hit, the Call should NOT be invoked again."""
        call_count = 0

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            nonlocal call_count
            call_count += 1
            return _json_response({"entities": ["X"]})

        client = FunctionClient(function=fn)
        call = Call(ExtractSig, model="function-test", client=client)

        cache = SemanticCache(similarity_threshold=0.95)
        # First call: embedding [1.0, 0.0]
        cache._embed = AsyncMock(return_value=[1.0, 0.0])  # ty: ignore[invalid-assignment]
        await cache.call(call, text="hello world")
        assert call_count == 1

        # Second call: very similar embedding → cache hit
        cache._embed = AsyncMock(return_value=[0.99, 0.01])
        result = await cache.call(call, text="hello there world")
        assert result.entities == ["X"]
        assert call_count == 1  # NOT called again
        assert cache.total_hits == 1

    async def test_cache_miss_on_low_similarity(self) -> None:
        """Dissimilar inputs should NOT hit cache."""
        call_count = 0

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            nonlocal call_count
            call_count += 1
            return _json_response({"entities": [f"result_{call_count}"]})

        client = FunctionClient(function=fn)
        call = Call(ExtractSig, model="function-test", client=client)

        cache = SemanticCache(similarity_threshold=0.95)
        cache._embed = AsyncMock(return_value=[1.0, 0.0])  # ty: ignore[invalid-assignment]
        await cache.call(call, text="first query")

        # Very different embedding → cache miss
        cache._embed = AsyncMock(return_value=[0.0, 1.0])  # ty: ignore[invalid-assignment]
        await cache.call(call, text="completely different")
        assert call_count == 2
        assert cache.size == 2

    async def test_eviction_on_max_entries(self) -> None:
        """Cache should evict oldest entries when over max_entries."""

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"entities": ["X"]})

        client = FunctionClient(function=fn)
        call = Call(ExtractSig, model="function-test", client=client)

        cache = SemanticCache(similarity_threshold=0.99, max_entries=3)
        embed_counter = 0

        async def unique_embed(text: str) -> list[float]:
            nonlocal embed_counter
            embed_counter += 1
            # Each embedding is orthogonal → cosine similarity = 0 → no cache hits
            vec = [0.0] * 10
            vec[embed_counter % 10] = 1.0
            return vec

        cache._embed = unique_embed  # ty: ignore[invalid-assignment]

        for i in range(5):
            await cache.call(call, text=f"query {i}")

        assert cache.size == 3  # evicted oldest 2


# ---------------------------------------------------------------------------
# Phase 16.1: exact-input fast path (closes the same-input duplicate-miss bug)
# ---------------------------------------------------------------------------


class TestExactInputFastPath:
    async def test_byte_identical_input_skips_embedding(self) -> None:
        """The Phase 16.1 fix: byte-identical inputs should hit the cache
        without making any embedding API call. Before the fix, an
        embedding call was paid on every lookup which could miss on
        identical text when the embedding API returned slightly different
        vectors or threshold edge cases bit."""
        call_count = 0

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            nonlocal call_count
            call_count += 1
            return _json_response({"entities": ["X"]})

        client = FunctionClient(function=fn)
        call = Call(ExtractSig, model="function-test", client=client)

        cache = SemanticCache(similarity_threshold=0.99)

        embed_call_count = 0

        async def counting_embed(text: str) -> list[float]:
            nonlocal embed_call_count
            embed_call_count += 1
            return [1.0, 0.0, 0.0]

        cache._embed = counting_embed  # ty: ignore[invalid-assignment]

        # First call: cold cache → embeds + LLM call
        await cache.call(call, text="byte identical input")
        assert embed_call_count == 1
        assert call_count == 1

        # Second call with byte-identical input: fast path, NO embedding
        await cache.call(call, text="byte identical input")
        assert embed_call_count == 1, "exact-input fast path should skip embed"
        assert call_count == 1, "exact-input fast path should skip LLM"

        # Third call with same input: fast path again
        await cache.call(call, text="byte identical input")
        assert embed_call_count == 1
        assert call_count == 1
        assert cache.total_hits == 2

    async def test_exact_path_respects_signature_isolation(self) -> None:
        """Two Calls with the same input text but different signatures
        must NOT share an exact-path entry."""

        class OtherSig(Signature):
            """Different sig with same field names."""

            text: str = InputField(description="Input")
            entities: list[str] = OutputField(description="Output")

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"entities": ["X"]})

        client = FunctionClient(function=fn)
        call1 = Call(ExtractSig, model="function-test", client=client)
        call2 = Call(OtherSig, model="function-test", client=client)

        cache = SemanticCache(similarity_threshold=0.99)
        cache._embed = AsyncMock(return_value=[1.0, 0.0, 0.0])  # ty: ignore[invalid-assignment]

        await cache.call(call1, text="same text")
        # Different signature → different exact-path key → cache miss + new entry
        await cache.call(call2, text="same text")
        assert cache.size == 2

    async def test_exact_path_respects_instructions(self) -> None:
        """Two Calls with the same signature + input but different
        instructions must NOT share an exact-path entry (audit finding F
        applies to the fast path too)."""

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"entities": ["X"]})

        client = FunctionClient(function=fn)
        call1 = Call(ExtractSig, model="function-test", client=client, instructions="Be brief.")
        call2 = Call(ExtractSig, model="function-test", client=client, instructions="Be verbose.")

        cache = SemanticCache(similarity_threshold=0.99)
        cache._embed = AsyncMock(return_value=[1.0, 0.0, 0.0])  # ty: ignore[invalid-assignment]

        await cache.call(call1, text="same text")
        await cache.call(call2, text="same text")
        # Different config_fingerprint (different instructions) → different
        # exact-path key → 2 entries.
        assert cache.size == 2


# ---------------------------------------------------------------------------
# Phase 16.1: disk persistence layer
# ---------------------------------------------------------------------------


class TestDiskPersistence:
    async def test_disk_path_persists_entries_across_instances(self, tmp_path) -> None:
        """Two SemanticCache instances against the same disk_path should
        see each other's entries via the JSONL replay."""

        disk = tmp_path / "cache.jsonl"

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"entities": ["X"]})

        client = FunctionClient(function=fn)
        call = Call(ExtractSig, model="function-test", client=client)

        # Cache 1: cold, write one entry to disk
        cache1 = SemanticCache(disk_path=disk, similarity_threshold=0.99)
        cache1._embed = AsyncMock(return_value=[1.0, 0.0, 0.0])  # ty: ignore[invalid-assignment]
        await cache1.call(call, text="persistent input")
        assert disk.exists()
        assert cache1.size == 1

        # Cache 2: same disk path, should replay the prior entry
        cache2 = SemanticCache(disk_path=disk, similarity_threshold=0.99)
        cache2._embed = AsyncMock(return_value=[1.0, 0.0, 0.0])  # ty: ignore[invalid-assignment]
        assert cache2.size == 1, "replay should rehydrate the entry"

        # Lookup against cache2 should hit the exact-path index
        # without firing the LLM (call_count tracked via fn closure).
        call_count = {"n": 0}

        def fn2(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            call_count["n"] += 1
            return _json_response({"entities": ["new"]})

        call2 = Call(ExtractSig, model="function-test", client=FunctionClient(function=fn2))
        # Use the SAME instructions/config so the config_fp matches
        await cache2.call(call2, text="persistent input")
        assert call_count["n"] == 0, "replayed entry should produce a fast-path hit"

    async def test_disk_path_tolerates_corrupt_last_line(self, tmp_path) -> None:
        """A partial last line in the JSONL log (kill-9 between write
        and fsync) must be tolerated by the replay scan."""
        disk = tmp_path / "corrupt.jsonl"

        # Hand-write a valid entry plus a corrupt trailing line
        disk.write_text(
            json.dumps(
                {
                    "input_key": "abcd1234",
                    "embedding": [1.0, 0.0],
                    "result_json": '{"entities":["X"]}',
                    "signature_name": "ExtractSig",
                    "model": "function-test",
                    "config_fingerprint": "fp",
                    "hit_count": 0,
                    "input_sha256": "deadbeef" * 8,
                }
            )
            + "\n"
            + '{"input_key":"partial","embedding":[1.0',  # corrupt
        )
        cache = SemanticCache(disk_path=disk, similarity_threshold=0.99)
        # The valid entry should be loaded; the corrupt one dropped
        assert cache.size == 1

    async def test_clear_disk_truncates_log(self, tmp_path) -> None:
        disk = tmp_path / "to-clear.jsonl"

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"entities": ["X"]})

        client = FunctionClient(function=fn)
        call = Call(ExtractSig, model="function-test", client=client)

        cache = SemanticCache(disk_path=disk, similarity_threshold=0.99)
        cache._embed = AsyncMock(return_value=[1.0, 0.0])  # ty: ignore[invalid-assignment]
        await cache.call(call, text="will be cleared")
        assert disk.exists()
        cache.clear_disk()
        assert not disk.exists()

    async def test_clear_in_memory_does_not_truncate_disk(self, tmp_path) -> None:
        """clear() drops in-memory state only; the disk log persists."""
        disk = tmp_path / "preserve.jsonl"

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"entities": ["X"]})

        client = FunctionClient(function=fn)
        call = Call(ExtractSig, model="function-test", client=client)

        cache = SemanticCache(disk_path=disk, similarity_threshold=0.99)
        cache._embed = AsyncMock(return_value=[1.0, 0.0])  # ty: ignore[invalid-assignment]
        await cache.call(call, text="survives clear")

        cache.clear()
        assert cache.size == 0
        assert disk.exists()

        # New instance should replay the on-disk entry
        cache2 = SemanticCache(disk_path=disk, similarity_threshold=0.99)
        assert cache2.size == 1

    async def test_max_entries_honored_on_replay(self, tmp_path) -> None:
        """If the disk log has more entries than max_entries, replay
        should keep only the most recent N (FIFO)."""
        disk = tmp_path / "big.jsonl"
        # Hand-write 5 entries in the new (Phase 16.5) format.
        with disk.open("w", encoding="utf-8") as f:
            for i in range(5):
                f.write(
                    json.dumps(
                        {
                            "embedding": [float(i), 0.0],
                            "result_json": f'{{"entities":["e{i}"]}}',
                            "signature_name": "ExtractSig",
                            "model": "function-test",
                            "config_fingerprint": "fp",
                            "input_sha256": f"{i:064d}",
                            "hit_count": 0,
                        }
                    )
                    + "\n"
                )

        cache = SemanticCache(disk_path=disk, max_entries=3, similarity_threshold=0.99)
        assert cache.size == 3
        # Should be the last 3 (entries 2, 3, 4)
        assert cache._entries[0].input_sha256 == f"{2:064d}"
        assert cache._entries[2].input_sha256 == f"{4:064d}"

    async def test_legacy_disk_log_migration(self, tmp_path) -> None:
        """Phase 16.5: pre-Phase-16.5 disk logs stored a separate
        ``input_key`` (16-char prefix) on the entry plus a top-level
        ``input_sha256``. The replay must migrate them onto the new
        CacheEntry shape (full sha256 on the entry, no input_key)."""
        disk = tmp_path / "legacy.jsonl"
        disk.write_text(
            json.dumps(
                {
                    # Legacy shape — this is what pre-Phase-16.5 wrote
                    "input_key": "abcd1234",  # the now-removed 16-char prefix
                    "embedding": [1.0, 0.0],
                    "result_json": '{"entities":["X"]}',
                    "signature_name": "ExtractSig",
                    "model": "function-test",
                    "config_fingerprint": "fp",
                    "hit_count": 0,
                    "input_sha256": "deadbeef" * 8,  # was top-level
                }
            )
            + "\n"
        )
        cache = SemanticCache(disk_path=disk, similarity_threshold=0.99)
        assert cache.size == 1
        # The migrated entry should have the full sha lifted onto it.
        assert cache._entries[0].input_sha256 == "deadbeef" * 8
        # And the exact-input index should have rebuilt with that key.
        assert (
            "ExtractSig",
            "function-test",
            "fp",
            "deadbeef" * 8,
        ) in cache._exact_index

    async def test_eviction_preserves_exact_index_for_survivors(self, tmp_path) -> None:
        """Phase 16.5 fix: after FIFO eviction, the exact-input index
        must still resolve every surviving entry. Pre-fix this test
        would have shown an empty exact_index for evicted entries.
        """

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"entities": ["X"]})

        client = FunctionClient(function=fn)
        call = Call(ExtractSig, model="function-test", client=client)

        # max_entries=2 → after the third write, the first entry is
        # evicted. The exact index should still cover the surviving
        # second and third entries.
        cache = SemanticCache(similarity_threshold=0.99, max_entries=2)
        embed_counter = 0

        async def unique_embed(text: str) -> list[float]:
            nonlocal embed_counter
            embed_counter += 1
            vec = [0.0] * 10
            vec[embed_counter % 10] = 1.0
            return vec

        cache._embed = unique_embed  # ty: ignore[invalid-assignment]

        await cache.call(call, text="alpha")
        await cache.call(call, text="bravo")
        await cache.call(call, text="charlie")  # triggers eviction of "alpha"

        assert cache.size == 2
        # Exact index should cover BOTH survivors (bravo + charlie).
        # Pre-Phase-16.5 this assertion would fail because
        # _rebuild_exact_index left the index empty.
        assert len(cache._exact_index) == 2
