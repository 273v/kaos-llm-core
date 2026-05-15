"""Response caches for LLM Programs.

Two complementary cache shapes:

- :class:`SemanticCache` — keys on
  ``(signature, model, config_fingerprint, embedding)`` for a single
  :class:`~kaos_llm_core.programs.call.Call`. Two tiers: exact-input
  hash fast path, embedding-similarity slow path.

- :class:`ChunkCache` / :class:`InMemoryChunkCache` — keys on
  ``(chunk_id, program_name, model_hint)`` for an entire
  :class:`~kaos_llm_core.programs.base.Program` invocation over one
  chunk. ``chunk_id`` is a content-derived hash; identical chunks
  across documents share entries. Consumed by the long-document
  Programs (``HierarchicalSummary`` / ``MapReduceSummary`` /
  ``RefineSummary`` / ``ChunkedClassify``) via their ``cache=``
  constructor argument.
"""

from kaos_llm_core.cache.chunk import ChunkCache, ChunkCacheKey, InMemoryChunkCache
from kaos_llm_core.cache.semantic import SemanticCache

__all__ = [
    "ChunkCache",
    "ChunkCacheKey",
    "InMemoryChunkCache",
    "SemanticCache",
]
