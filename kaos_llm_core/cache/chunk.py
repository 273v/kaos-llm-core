"""Chunk-result cache — reuse per-chunk Program output across runs.

Companion to :mod:`kaos_llm_core.cache.semantic`. Where
:class:`~kaos_llm_core.cache.semantic.SemanticCache` keys on
``(signature, model, config_fingerprint, embedding)`` for a single
:class:`~kaos_llm_core.programs.call.Call`, this cache keys on
``(chunk_id, program_name, model_hint)`` for an entire
:class:`~kaos_llm_core.programs.base.Program` invocation over one
chunk.

``chunk_id`` is a stable, content-derived blake3 (see
:class:`kaos_nlp_core.chunking.Chunk`), so two documents that share an
identical chunk hash share the same cache entry. The plan's §5.3
"CTPH gives us 'this is the same chunk' cheaply" pattern lands here:
when a document is re-summarised after a small edit, every chunk
whose content didn't change replays from this cache without an LLM
call.

The cache is async-safe under :class:`asyncio.Lock`. The default
in-memory implementation is process-local; callers needing
durability supply their own implementation conforming to
:class:`ChunkCache`.
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol, runtime_checkable

from kaos_core.types.content import KaosModel

__all__ = [
    "ChunkCache",
    "ChunkCacheKey",
    "InMemoryChunkCache",
]


class ChunkCacheKey(KaosModel):
    """Stable cache key for a per-chunk Program result.

    Three components, all required:

    - ``chunk_id``: the
      :attr:`kaos_nlp_core.chunking.Chunk.chunk_id` of the input
      chunk. Already a stable hash over the chunk's text + offsets,
      so identical chunks across documents resolve to the same key.
    - ``program_name``: the class name of the per-chunk Program
      (``"AbstractiveSummary"``, ``"ZeroShotClassify"``, etc.). Two
      Programs operating on the same chunk should not share a cache
      entry.
    - ``model_hint``: an opaque string identifying the underlying
      LLM (model id when known, ``""`` when the per-chunk Program is
      no-LLM and the cache key needs no model discriminator).

    The fields are part of the public on-disk shape; renaming them is
    a breaking change.
    """

    chunk_id: str
    program_name: str
    model_hint: str = ""


@runtime_checkable
class ChunkCache(Protocol):
    """Async cache protocol for per-chunk Program output.

    Two methods, both async. The expected value is JSON-serialisable
    via Pydantic — :class:`~kaos_llm_core.results.Summary` and
    :class:`~kaos_llm_core.results.Classification` instances both
    qualify.

    Implementations must be safe under concurrent ``get`` /
    ``set`` calls from the long-doc Programs that run leaves with
    bounded concurrency.
    """

    async def get(self, key: ChunkCacheKey) -> Any | None:  # pragma: no cover - protocol
        """Return the cached value for ``key`` or ``None`` on miss."""
        ...

    async def set(self, key: ChunkCacheKey, value: Any) -> None:  # pragma: no cover - protocol
        """Store ``value`` under ``key``."""
        ...


class InMemoryChunkCache:
    """Default :class:`ChunkCache` implementation — process-local dict.

    Keys are :class:`ChunkCacheKey` instances; values are stored as-is
    (no JSON round-trip). The cache is bounded by ``max_entries`` with
    FIFO eviction.

    Args:
        max_entries: Maximum number of cached entries. When the cache
            grows beyond this, the oldest entries are dropped in
            insertion order. Default ``10_000``.
    """

    def __init__(self, *, max_entries: int = 10_000) -> None:
        if max_entries <= 0:
            raise ValueError(f"max_entries must be > 0, got {max_entries}")
        self._max_entries = max_entries
        self._entries: dict[tuple[str, str, str], Any] = {}
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _flatten(key: ChunkCacheKey) -> tuple[str, str, str]:
        return (key.chunk_id, key.program_name, key.model_hint)

    async def get(self, key: ChunkCacheKey) -> Any | None:
        async with self._lock:
            value = self._entries.get(self._flatten(key))
            if value is None:
                self._misses += 1
            else:
                self._hits += 1
            return value

    async def set(self, key: ChunkCacheKey, value: Any) -> None:
        async with self._lock:
            flat = self._flatten(key)
            if flat in self._entries:
                # Reinsert at the tail to update FIFO order.
                del self._entries[flat]
            self._entries[flat] = value
            if len(self._entries) > self._max_entries:
                # Drop the oldest until we're back at the cap. ``dict``
                # preserves insertion order, so the iterator gives us
                # the keys in age order.
                overflow = len(self._entries) - self._max_entries
                for stale_key in list(self._entries.keys())[:overflow]:
                    del self._entries[stale_key]

    @property
    def hit_count(self) -> int:
        """Number of successful :meth:`get` lookups since construction."""
        return self._hits

    @property
    def miss_count(self) -> int:
        """Number of unsuccessful :meth:`get` lookups since construction."""
        return self._misses

    def __len__(self) -> int:
        return len(self._entries)
