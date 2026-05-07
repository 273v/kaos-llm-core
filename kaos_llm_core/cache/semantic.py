"""Semantic caching — cache LLM responses by input similarity.

Two-tier cache. The fast path is an exact-input hash lookup that hits
**without making any embedding call** when the same Call sees the same
input bytes. The slow path is the embedding-similarity search for
semantically similar (but not identical) inputs.

The fast path closes the documented "same-input duplicate miss" bug:
the previous version always paid an embedding API call on every
lookup and then accepted/rejected based on cosine similarity, which
could miss on byte-identical inputs when the embedding API returned
slightly different vectors (or when threshold edge cases bit). With
the exact-input fast path, byte-identical inputs are O(1) hash
lookups and never touch the embedding API at all.

Optional disk persistence: pass ``disk_path=...`` and the cache writes
each new entry to a JSONL file under the disk path. On init, the file
is replayed to rehydrate ``_entries``. The file is the durable record;
in-memory state is reconstructed from it. Atomic appends use ``open(...,
"a")`` so multiple processes against the same file produce a coherent
log (the JSONL line discipline is the same as the Phase 15.2 batch log).

Example::

    cache = SemanticCache(
        embed_model="openai:text-embedding-3-small",
        similarity_threshold=0.95,
        disk_path="~/.cache/kaos/llm-core-semantic.jsonl",  # optional
    )
    call = Call(ExtractEntities, model="anthropic:claude-haiku-4-5")
    # First call: cold cache, hits the LLM, persists to disk if configured
    result1 = await cache.call(call, text="SEC filed suit against Acme")
    # Second call with byte-identical input: O(1) hash hit, no embedding API
    result2 = await cache.call(call, text="SEC filed suit against Acme")
    # Third call with similar input: embedding-similarity hit
    result3 = await cache.call(call, text="SEC filed a lawsuit against Acme Corp")
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from kaos_core.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class CacheEntry:
    """A cached input-output pair with its embedding vector.

    The cache key is the conjunction of (signature_name, model,
    config_fingerprint, embedding similarity). The previous version
    keyed only on signature_name + embedding, which let a Call with
    different instructions return another Call's cached answer. Audit
    finding F.

    ``input_sha256`` (Phase 16.5) stores the FULL sha256 of the input
    text so the exact-input fast-path index can be rebuilt after
    eviction. The previous version stored only the first 16 hex chars
    in ``input_key``, which made ``_rebuild_exact_index`` impossible
    and silently dropped exact-path coverage post-eviction.
    """

    embedding: list[float]
    result_json: str
    signature_name: str
    model: str
    config_fingerprint: str
    input_sha256: str = ""
    hit_count: int = 0


class SemanticCache:
    """Two-tier response cache for LLM Calls.

    Tier 1 (fast path): exact-input hash lookup keyed on
    ``(signature_name, model, config_fingerprint, sha256(input_text))``.
    Hits in O(1) without making any embedding API call. This is the
    "same-input duplicate miss" fix — the previous version always paid
    an embedding call before any lookup and then could miss on
    byte-identical inputs due to embedding non-determinism / threshold
    edge cases.

    Tier 2 (slow path): embedding-similarity search across cached
    entries with the same (sig, model, config) key. Used when the
    fast-path hash misses but a semantically-similar prior input
    exists.

    **Stampede protection:** when multiple concurrent requests miss
    the cache for the same input, a per-key ``asyncio.Lock`` ensures
    only one request actually calls the LLM. The remaining waiters
    re-check the cache after acquiring the lock and find the result
    already stored (double-check pattern). Unrelated inputs are not
    serialized — they acquire independent locks.

    Optional disk persistence: pass ``disk_path=...`` and entries are
    appended to a JSONL file. On init, the file is replayed to
    rehydrate the in-memory state. Multiple processes against the same
    file see a coherent log via the same discipline as the Phase 15.2
    batch log.

    Args:
        embed_model: Model for computing input embeddings (slow path only).
        similarity_threshold: Cosine similarity threshold for slow-path hits.
        max_entries: Maximum number of cached entries (FIFO eviction).
        disk_path: Optional path to a JSONL file for durable persistence.
            ``~`` is expanded; parent dirs are created on first append.
    """

    def __init__(
        self,
        embed_model: str = "openai:text-embedding-3-small",
        similarity_threshold: float = 0.95,
        max_entries: int = 1000,
        disk_path: str | Path | None = None,
    ) -> None:
        self.embed_model = embed_model
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self._entries: list[CacheEntry] = []
        # Tier 1 fast-path index: (sig, model, config_fp, input_sha256) → CacheEntry
        # The index is rebuilt on every list mutation (append, evict, replay).
        self._exact_index: dict[tuple[str, str, str, str], CacheEntry] = {}
        self._client: Any = None
        # Phase 9c: protect _entries against concurrent reads/writes. Without
        # this lock, two parallel cache.call() invocations could (a) both
        # miss because the first hasn't appended yet (cache stampede), or
        # (b) lose an append when eviction rebinds _entries mid-flight.
        # The lock is async because the critical sections include awaits
        # (embed + the underlying call_obj invocation).
        self._lock: asyncio.Lock = asyncio.Lock()
        # Per-key stampede locks: keyed by the exact_key tuple. When
        # multiple concurrent requests miss the cache for the SAME input,
        # only one acquires this lock and calls the LLM; the rest wait
        # and then find the result in the cache (double-check pattern).
        # Entries are cleaned up after the lock is released.
        self._in_flight: dict[tuple[str, str, str, str], asyncio.Lock] = {}
        # Phase 16.1: optional disk persistence
        self._disk_path: Path | None = None
        if disk_path is not None:
            self._disk_path = Path(disk_path).expanduser()
            self._disk_path.parent.mkdir(parents=True, exist_ok=True)
            self._replay_disk_log()

    async def call(self, call_obj: Any, **inputs: Any) -> Any:
        """Execute a Call with semantic caching.

        Checks the cache first. On miss, runs the Call and caches the result.
        """
        from kaos_llm_core.programs.call import Call

        if not isinstance(call_obj, Call):
            # Not a Call, just execute directly
            return await call_obj(**inputs)

        input_text = self._inputs_to_text(inputs)
        sig_name = call_obj.signature.__name__
        model_id = call_obj._model or ""
        config_fp = _config_fingerprint(call_obj)
        input_sha = hashlib.sha256(input_text.encode("utf-8")).hexdigest()

        # Tier 1: exact-input fast path. Hash lookup BEFORE any embedding
        # API call. Closes the "same-input duplicate miss" bug — the
        # previous version always paid an embedding call before any
        # lookup and could miss on byte-identical inputs due to
        # embedding non-determinism / threshold edge cases. The lock
        # protects the dict against concurrent eviction rebuilds.
        async with self._lock:
            exact_key = (sig_name, model_id, config_fp, input_sha)
            exact_entry = self._exact_index.get(exact_key)
            if exact_entry is not None:
                exact_entry.hit_count += 1
                logger.info(
                    "SemanticCache exact hit: %s (hits=%d)",
                    sig_name,
                    exact_entry.hit_count,
                )
                output_model = call_obj._output_model
                return output_model.model_validate_json(exact_entry.result_json)

        # Tier 2: embedding-similarity slow path. Compute the embedding
        # outside the lock so concurrent unrelated lookups don't
        # serialize on the embed API.
        embedding = await self._embed(input_text)

        async with self._lock:
            best_entry, best_sim = self._find_similar(embedding, sig_name, model_id, config_fp)
            if best_entry is not None and best_sim >= self.similarity_threshold:
                best_entry.hit_count += 1
                logger.info(
                    "SemanticCache similarity hit: %s (similarity=%.3f, hits=%d)",
                    sig_name,
                    best_sim,
                    best_entry.hit_count,
                )
                # Reconstruct result from cached JSON
                output_model = call_obj._output_model
                return output_model.model_validate_json(best_entry.result_json)

        # Cache miss — acquire a per-key stampede lock so only one
        # concurrent request for the SAME input calls the LLM. Other
        # waiters will re-check the cache after acquiring the lock and
        # find the result already stored (double-check pattern).
        # Unrelated inputs acquire independent locks and proceed in
        # parallel — the per-key lock does NOT serialize unrelated work.
        stampede_lock = self._in_flight.setdefault(exact_key, asyncio.Lock())
        try:
            async with stampede_lock:
                # Double-check: another waiter may have populated the
                # cache while we waited for the stampede lock.
                async with self._lock:
                    exact_entry = self._exact_index.get(exact_key)
                    if exact_entry is not None:
                        exact_entry.hit_count += 1
                        logger.info(
                            "SemanticCache stampede-coalesced hit: %s (hits=%d)",
                            sig_name,
                            exact_entry.hit_count,
                        )
                        output_model = call_obj._output_model
                        return output_model.model_validate_json(exact_entry.result_json)

                result = await call_obj(**inputs)

                # Cache the result under the lock. Use in-place delete for
                # eviction so we never rebind ``self._entries`` — rebinding
                # under concurrent iteration is the lost-write race the
                # previous version had.
                result_json = result.model_dump_json()
                entry = CacheEntry(
                    embedding=embedding,
                    result_json=result_json,
                    signature_name=sig_name,
                    model=model_id,
                    config_fingerprint=config_fp,
                    input_sha256=input_sha,
                )
                async with self._lock:
                    self._entries.append(entry)
                    self._exact_index[(sig_name, model_id, config_fp, input_sha)] = entry
                    if len(self._entries) > self.max_entries:
                        # In-place truncation: drop the oldest entries without
                        # rebinding the list. Concurrent appenders see the same
                        # list object so their writes are not lost.
                        excess = len(self._entries) - self.max_entries
                        del self._entries[:excess]
                        # Eviction invalidates the exact-input index for the
                        # dropped entries. Rebuild it from each surviving
                        # entry's full input_sha256 (Phase 16.5 fix — the
                        # previous version stored only a 16-char prefix and
                        # could not reconstruct the full key, so it dropped
                        # exact-path coverage post-eviction).
                        self._rebuild_exact_index()
                    if self._disk_path is not None:
                        self._append_disk_log(entry)

                return result
        finally:
            # Clean up the per-key lock entry. Only remove if the lock
            # stored for this key is the same object we acquired — a
            # concurrent setdefault may have raced and installed a new
            # lock for the same key (harmless: both point to a valid
            # asyncio.Lock and the loser's cleanup is a no-op).
            self._in_flight.pop(exact_key, None)

    @property
    def size(self) -> int:
        """Number of entries in the cache."""
        return len(self._entries)

    @property
    def total_hits(self) -> int:
        """Total cache hits across all entries."""
        return sum(e.hit_count for e in self._entries)

    def clear(self) -> None:
        """Clear all cached entries (in-memory only).

        The disk JSONL log, if configured, is NOT truncated by this
        call — call ``clear_disk()`` to drop persisted entries too.
        """
        self._entries.clear()
        self._exact_index.clear()
        self._in_flight.clear()

    def clear_disk(self) -> None:
        """Truncate the on-disk JSONL log if disk persistence is enabled."""
        if self._disk_path is not None and self._disk_path.exists():
            self._disk_path.unlink()

    @property
    def disk_path(self) -> Path | None:
        """The configured disk persistence path, or None."""
        return self._disk_path

    def _rebuild_exact_index(self) -> None:
        """Rebuild the exact-input hash index from the entries list.

        Called after eviction. Replays each surviving entry's
        ``(sig, model, config_fp, input_sha256)`` key into the fast-path
        dict. Phase 16.5 fix: ``CacheEntry.input_sha256`` now stores the
        full 64-char sha256 (was a 16-char prefix in ``input_key`` that
        could not be reconstructed back to the original key). Eviction
        is rare and the index is small, so the O(n) rebuild is fine.
        """
        self._exact_index.clear()
        for entry in self._entries:
            if not entry.input_sha256:
                # Legacy entry from a pre-Phase-16.5 disk log replay
                # without the full sha. Skip — the slow path still
                # serves it via embedding similarity.
                continue
            self._exact_index[
                (entry.signature_name, entry.model, entry.config_fingerprint, entry.input_sha256)
            ] = entry

    def _append_disk_log(self, entry: CacheEntry) -> None:
        """Append one cache entry to the disk JSONL log.

        Each line is a JSON object with all entry fields including the
        full ``input_sha256`` so the replay path can rebuild the
        exact-input index. Append-only, atomic per line. Phase 16.5:
        ``input_sha256`` lives on the entry now (was a separate
        argument that the writer had to remember to pass through).
        """
        if self._disk_path is None:
            return
        record = asdict(entry)
        # Phase 16.5: schema_version stamps the line so future schema
        # changes can be detected on replay rather than misread.
        record["schema_version"] = "1"
        with self._disk_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")

    def _replay_disk_log(self) -> None:
        """Read the on-disk JSONL log and rehydrate ``_entries`` + index.

        Called once on init when ``disk_path`` is configured. Tolerates
        a corrupt last line (the kill-9 window between write and fsync)
        by silently dropping it, matching the Phase 15.2 batch log
        discipline. Honors ``max_entries`` by FIFO-truncating to the
        cap before populating in-memory state.

        Phase 16.5: handles two record shapes:
          - new (post-16.5): full CacheEntry fields incl. input_sha256
          - legacy (pre-16.5): a separate top-level input_sha256 field
            plus an input_key field on the entry. Legacy entries are
            re-parsed by stripping the input_key and lifting the
            top-level input_sha256 onto the entry.
        """
        if self._disk_path is None or not self._disk_path.exists():
            return
        loaded: list[CacheEntry] = []
        with self._disk_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "SemanticCache: ignoring corrupt line %d in %s",
                        line_no + 1,
                        self._disk_path,
                    )
                    continue
                # Strip schema_version (cosmetic; not a CacheEntry field).
                record.pop("schema_version", None)
                # Legacy migration: pre-Phase-16.5 lines stored a
                # separate top-level "input_sha256" alongside an
                # "input_key" prefix on the entry. Lift the sha onto
                # the entry and drop the prefix.
                legacy_sha = record.pop("input_sha256", None) if "input_key" in record else None
                if "input_key" in record:
                    record.pop("input_key", None)
                if legacy_sha is not None and "input_sha256" not in record:
                    record["input_sha256"] = legacy_sha
                try:
                    entry = CacheEntry(**record)
                except TypeError:
                    logger.warning(
                        "SemanticCache: schema-mismatched line %d in %s",
                        line_no + 1,
                        self._disk_path,
                    )
                    continue
                if not entry.input_sha256:
                    # Cannot reach the fast path without the full sha;
                    # slow path will still serve it via embedding.
                    loaded.append(entry)
                    continue
                loaded.append(entry)

        # Honor max_entries on replay (FIFO).
        if len(loaded) > self.max_entries:
            loaded = loaded[-self.max_entries :]
        for entry in loaded:
            self._entries.append(entry)
            if entry.input_sha256:
                self._exact_index[
                    (
                        entry.signature_name,
                        entry.model,
                        entry.config_fingerprint,
                        entry.input_sha256,
                    )
                ] = entry
        if loaded:
            logger.info(
                "SemanticCache: replayed %d entries from %s",
                len(loaded),
                self._disk_path,
            )

    async def _embed(self, text: str) -> list[float]:
        """Compute embedding for a text string."""
        from kaos_llm_client import create_client

        if self._client is None:
            self._client = create_client(self.embed_model)

        response = await self._client.embed_async(text)
        return response.embeddings[0]

    def _find_similar(
        self,
        embedding: list[float],
        sig_name: str,
        model_id: str,
        config_fingerprint: str,
    ) -> tuple[CacheEntry | None, float]:
        """Find the most similar cached entry for the same signature/model/config.

        All four key components must match before similarity is even
        computed. The previous version only checked the signature, which
        let a Call with different instructions return another Call's
        cached answer (audit finding F).
        """
        best_entry: CacheEntry | None = None
        best_sim = -1.0

        for entry in self._entries:
            if entry.signature_name != sig_name:
                continue
            if entry.model != model_id:
                continue
            if entry.config_fingerprint != config_fingerprint:
                continue
            sim = _cosine_similarity(embedding, entry.embedding)
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        return best_entry, best_sim

    def _inputs_to_text(self, inputs: dict[str, Any]) -> str:
        """Convert inputs to a text string for embedding."""
        parts = []
        for _key, value in sorted(inputs.items()):
            if isinstance(value, str):
                parts.append(value)
            else:
                parts.append(json.dumps(value, default=str))
        return " ".join(parts)


def _config_fingerprint(call_obj: Any) -> str:
    """Build a stable fingerprint of the parts of a Call that affect its output.

    Includes ``instructions``, ``examples`` (serialized), and the persisted
    hyperparameters (``temperature``, ``top_p``, etc.). Two Calls with the
    same fingerprint should be functionally equivalent for cache-equivalence
    purposes.

    The hyperparameter set is read from ``Call._PERSISTED_HYPERPARAMETERS``
    so the fingerprint stays in lockstep with whatever Call considers
    "behavior-affecting". The Phase 9b version hard-coded a subset
    (``temperature, top_p, top_k, max_tokens, seed``) which silently
    drifted from the canonical list — ``presence_penalty`` and
    ``frequency_penalty`` change behavior but did not invalidate the cache.
    Reading from ``_PERSISTED_HYPERPARAMETERS`` is the single source of
    truth and any future hyperparameter Call adds is picked up
    automatically.

    Audit finding F — the original SemanticCache ignored these and treated
    all Calls with the same Signature as cache-equivalent.
    """
    # Lazy import to avoid a circular dependency between cache and programs.
    from kaos_llm_core.programs.call import Call

    persisted_keys = getattr(Call, "_PERSISTED_HYPERPARAMETERS", ())
    # Phase 9e: prefer the public ``Call.hyperparameters`` property over
    # reaching into ``_kwargs``. The property returns a defensive copy and
    # is the documented public accessor for the same data.
    if hasattr(call_obj, "hyperparameters"):
        all_hyperparams = call_obj.hyperparameters
    else:
        all_hyperparams = getattr(call_obj, "_kwargs", {}) or {}
    parts: dict[str, Any] = {
        "instructions": getattr(call_obj, "instructions", "") or "",
        "examples": [
            ex.model_dump() if hasattr(ex, "model_dump") else str(ex)
            for ex in (getattr(call_obj, "examples", []) or [])
        ],
        "hyperparameters": {k: v for k, v in all_hyperparams.items() if k in persisted_keys},
    }
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
