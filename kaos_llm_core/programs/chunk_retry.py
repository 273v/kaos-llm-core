# ruff: noqa: RUF001, RUF002
"""ChunkRetry — recursive per-cell extraction recovery on long documents.

The 6th and final layer of kelvin's extraction architecture (per
``docs/design/ws-tr-alpha-extractor-sprint.md`` §1). When the LLM refuses
to extract a value because it couldn't find one in the input — either
because the relevant content was outside its context window (long
contracts, multi-100-page filings) or because attention diluted across
a long document — chunk-retry re-invokes the LLM on smaller pieces of
the source text and promotes any value that surfaces.

**Composition** with the rest of the WS-TR stack:

  raw text
     │
     ▼
  Extract.call           ← layer 2: all-at-once LLM extraction
     │
     ▼
  AlphaLLMMerger         ← layer 4 + 5: rule-based candidates ∪ LLM
     │
     ▼
  ChunkRetry             ← THIS LAYER (layer 6): recover refused cells
     │                     by re-running the LLM on bounded chunks
     ▼
  ExtractionResult

**Per-cell, not per-document.** Only cells with ``status="not_in_document"``
or ``status="unclear"`` after the merger has run are candidates for
retry. Cells already in ``status="extracted"`` are left alone (the LLM
or alpha already produced a value); cells in ``status="error"`` are also
left alone (those are transient API failures, not content-availability
problems — the kaos-llm-client retry policy already handles them).

**Cost-bounded by design.** A pathological 200-page filing with 5
columns where 3 columns refuse would otherwise spawn ~50 LLM calls. The
defaults cap this:

- ``initial_chunk_chars`` = 8000 — initial chunk size (~2k tokens at
  English-text density). Roughly 4–8 chunks for a 50-page contract.
- ``max_chunks_per_doc`` = 16 — global safety valve. Documents that
  would exceed this are not retried at all.
- ``max_total_cost_usd`` = 0.50 — per-extract budget. Once cumulative
  retry cost exceeds this, remaining retries are short-circuited.
- ``confidence_on_recovery`` = 0.7 — sandwiched between alpha
  promotion (0.8) and a confident LLM extraction (0.9–0.95) so
  consumers can sort by trust level.

**Composition with AlphaLLMMerger.** The merger has already exhausted
rule-based candidates BEFORE chunk-retry runs. So a cell still in
``not_in_document`` after the merger means: (a) the LLM said "not in
the document" AND (b) alpha extractors produced no candidate or
multiple ambiguous candidates. Chunk-retry is the last-resort recovery
path before the cell is marked permanently refused.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from kaos_content.model.extraction import ExtractionCell
from kaos_core.logging import get_logger
from pydantic import BaseModel

if TYPE_CHECKING:
    from kaos_llm_core.signatures.extraction import ColumnSpec

logger = get_logger(__name__)

# Statuses that signal a content-availability refusal. ``error`` is
# excluded — those are transient API failures handled at the
# kaos-llm-client retry layer, not by chunk-retry.
_RETRY_STATUSES: frozenset[str] = frozenset({"not_in_document", "unclear"})


@dataclass(frozen=True, slots=True)
class ChunkRetryConfig:
    """Tunable policy for :class:`ChunkRetry`.

    All defaults chosen for the typical CUAD / 10-K extraction profile:
    documents in the 5–50 page range with 3–10 columns. Aggressive
    truncation defaults — better to leave a cell refused than spend
    $5 per document chasing a long tail.
    """

    enabled: bool = True
    """Master switch. ``Extract`` skips chunk-retry entirely when False."""

    initial_chunk_chars: int = 8000
    """Initial chunk size in characters. Roughly equivalent to
    ~2 000 tokens of English prose at typical density. Chosen so that
    ~4–8 chunks cover a 50-page contract without flooding the LLM."""

    min_chunk_chars: int = 1000
    """Smallest chunk size we'll retry at. Below this the chunk loses
    enough surrounding context that extraction quality drops faster
    than retry can recover."""

    backoff_divisor: int = 2
    """Recursive backoff factor — each retry depth shrinks the chunk
    size by this divisor (binary backoff at 2, ternary at 3, etc.).
    Matches kelvin's ``recursive.py:240-263`` halving pattern."""

    max_retry_depth: int = 2
    """Maximum recursion depth. With ``initial_chunk_chars=8000`` and
    ``backoff_divisor=2``, depth 2 means smallest chunk is 2000 chars."""

    max_chunks_per_doc: int = 16
    """Hard upper bound on chunks per document at any depth. Documents
    that would exceed this skip retry entirely — the heuristic is that
    retry cost scales with chunk count, so very long documents are
    better served by a different pipeline (see PR-6b future work:
    document segmentation before extraction)."""

    max_total_cost_usd: float = 0.50
    """Total per-extract retry budget. Once exceeded, remaining cells
    keep their refused status — the alternative is unbounded spend on
    pathological inputs."""

    max_candidates_per_cell: int = 5
    """Cap on candidate values collected across chunks for a single
    cell. Higher means more options for ambiguity resolution but more
    citations on the final cell."""

    confidence_on_recovery: float = 0.7
    """Confidence assigned to a cell that chunk-retry promoted.
    Sandwiched between alpha promotion (0.8) and a confident LLM
    extraction (0.9–0.95) so consumers can sort by trust level. Lower
    than alpha because alpha is deterministic; chunk-retry depends on
    a second LLM call that might also be wrong."""


@dataclass(frozen=True, slots=True)
class ChunkSpan:
    """One contiguous chunk of source text plus its position."""

    text: str
    start: int
    end: int


def _chunk_text(
    text: str,
    *,
    max_chars: int,
    min_chars: int,
) -> list[ChunkSpan]:
    """Split text into character-bounded chunks at sentence boundaries.

    Uses ``kaos_nlp_core.segmentation.segment_sentences`` (the bundled
    legal Punkt model — our nupunkt equivalent) when available; falls
    back to paragraph splitting (``segment_paragraphs``) when sentence
    segmentation is unavailable, and to a crude double-newline split
    when kaos-nlp-core itself is missing. Final fallback is a hard
    byte-cut for pathological single-sentence inputs that exceed
    ``max_chars`` (mirrors kelvin's BYTES level in the
    SECTIONS→PARAGRAPHS→SENTENCES→LINES→TOKENS→BYTES cascade at
    ``kelvin/nlp/segments/recursive_splitter.py:88-95``).

    Greedy accumulation: walk the segments in order, keep adding to the
    current chunk while ``(seg_end - cur_start)`` stays under
    ``max_chars``, emit when full. Slower than kelvin's prefix-sum +
    binary-search variant (``recursive_splitter.py:191-216``) but
    O(n) which is fine for the document sizes chunk-retry actually
    sees (5-50 KB legal contracts).
    """
    if len(text) <= max_chars:
        return [ChunkSpan(text=text, start=0, end=len(text))]

    try:
        from kaos_nlp_core.segmentation import segment_sentences

        segments = segment_sentences(text)
    except ImportError:
        # kaos-nlp-core is technically optional; a reasonable
        # fallback for the test environment is paragraph splitting.
        segments = []

    if not segments:
        # Crude paragraph fallback when no segmenter is available.
        segments = _paragraph_fallback(text)

    # Track only span boundaries during accumulation; the chunk text is
    # sliced from the original source so that
    # ``source_text[chunk.start:chunk.end] == chunk.text`` holds — alpha
    # citations + span verification depend on this round-trip.
    chunks: list[ChunkSpan] = []
    cur_start = -1
    cur_end = -1

    for seg in segments:
        seg_text = seg.text if hasattr(seg, "text") else seg["text"]
        seg_start = seg.start if hasattr(seg, "start") else seg["start"]
        seg_end = seg.end if hasattr(seg, "end") else seg["end"]
        seg_len = seg_end - seg_start

        # Single segment too big: hard-split by byte cut. This is rare
        # (a multi-paragraph block with no sentence boundaries) but
        # important for robustness on programmatic inputs (CSV-as-text,
        # minified XML).
        if seg_len > max_chars:
            if cur_start != -1:
                chunks.append(ChunkSpan(text=text[cur_start:cur_end], start=cur_start, end=cur_end))
                cur_start = -1
            for hard_chunk in _hard_split(seg_text, max_chars, base_offset=seg_start):
                chunks.append(hard_chunk)
            continue

        # Would adding this segment overflow? Emit current chunk first.
        # Span length includes inter-sentence whitespace because we
        # measure from cur_start to seg_end.
        if cur_start != -1 and (seg_end - cur_start) > max_chars:
            chunks.append(ChunkSpan(text=text[cur_start:cur_end], start=cur_start, end=cur_end))
            cur_start = -1

        if cur_start == -1:
            cur_start = seg_start
        cur_end = seg_end

    if cur_start != -1:
        chunks.append(ChunkSpan(text=text[cur_start:cur_end], start=cur_start, end=cur_end))

    # Drop any empty/too-small trailing chunk that would just waste a
    # call. The min_chars guard is applied here, not during
    # accumulation, because we want to keep small chunks if they're the
    # only way to cover a region.
    if len(chunks) > 1 and chunks[-1].end - chunks[-1].start < min_chars:
        # Merge the tiny tail into the prior chunk.
        prior = chunks[-2]
        merged = ChunkSpan(
            text=text[prior.start : chunks[-1].end],
            start=prior.start,
            end=chunks[-1].end,
        )
        chunks = [*chunks[:-2], merged]

    return chunks


def _paragraph_fallback(text: str) -> list[Any]:
    """Crude paragraph splitter for environments without kaos-nlp-core."""
    out = []
    pos = 0
    for para in text.split("\n\n"):
        if not para:
            pos += 2
            continue
        end = pos + len(para)
        out.append({"text": para, "start": pos, "end": end})
        pos = end + 2
    return out


def _hard_split(text: str, max_chars: int, *, base_offset: int) -> list[ChunkSpan]:
    """Last-resort byte-cut for sentences that exceed ``max_chars``."""
    chunks: list[ChunkSpan] = []
    for i in range(0, len(text), max_chars):
        chunks.append(
            ChunkSpan(
                text=text[i : i + max_chars],
                start=base_offset + i,
                end=base_offset + min(i + max_chars, len(text)),
            )
        )
    return chunks


# Type alias for the per-chunk extraction callable. Takes (chunk_text,
# doc_id) and returns a fresh ExtractionResult-like value. Caller wires
# this to Extract.extract() with the merger and chunk-retry both
# disabled to avoid recursion.
ChunkExtractFn = Callable[[str, str], Awaitable["ChunkExtractResult"]]


class ChunkExtractResult(BaseModel):
    """Minimal result shape that ChunkRetry needs from a per-chunk
    extraction call. Subset of :class:`ExtractionResult`."""

    cells: tuple[ExtractionCell, ...]
    cost_usd: float = 0.0


@dataclass(frozen=True, slots=True)
class ChunkRetry:
    """Per-cell extraction recovery on long documents.

    Composes after :class:`AlphaLLMMerger` in the Extract pipeline.
    See module docstring for the architectural rationale.

    Use as a class attribute on :class:`Extract`; the actual
    per-chunk extraction is provided by the caller as a closure over
    ``Extract`` itself (with merger + chunk_retry both disabled to
    avoid infinite recursion).
    """

    config: ChunkRetryConfig = field(default_factory=ChunkRetryConfig)

    async def retry_refused_cells(
        self,
        cells: tuple[ExtractionCell, ...],
        *,
        columns: tuple[ColumnSpec, ...],
        source_text: str,
        doc_id: str,
        extract_chunk: ChunkExtractFn,
    ) -> tuple[tuple[ExtractionCell, ...], float]:
        """Re-extract refused cells from chunked source text.

        Returns ``(new_cells, total_retry_cost_usd)``. ``new_cells`` is
        the same cells with refused entries replaced where chunk-retry
        recovered a value. ``total_retry_cost_usd`` is the cumulative
        LLM cost incurred across all chunk-extract calls.

        Args:
            cells: Output of the alpha-merged LLM extraction.
            columns: Schema column specs (same length as ``cells``).
            source_text: The original full document text.
            doc_id: Stable doc identifier; stamped onto recovered cells.
            extract_chunk: Async callable that runs LLM extraction on a
                chunk of text. Must NOT itself invoke chunk-retry —
                infinite recursion otherwise.
        """
        if not self.config.enabled:
            return cells, 0.0

        # Identify refused cells — only these are candidates for retry.
        refused_indices = [i for i, cell in enumerate(cells) if cell.status in _RETRY_STATUSES]
        if not refused_indices:
            return cells, 0.0

        # Skip retry for documents that would exceed the chunk count cap
        # — pathological inputs are better served by a real document
        # segmentation pipeline, not by chunk-retry.
        chunks = _chunk_text(
            source_text,
            max_chars=self.config.initial_chunk_chars,
            min_chars=self.config.min_chunk_chars,
        )
        if len(chunks) > self.config.max_chunks_per_doc:
            logger.info(
                "ChunkRetry skipped: document would yield %d chunks (cap=%d)",
                len(chunks),
                self.config.max_chunks_per_doc,
            )
            return cells, 0.0

        if len(chunks) <= 1:
            # The full document already fits in one chunk — chunking
            # wouldn't change what the LLM saw on the original call.
            # Skip retry entirely.
            return cells, 0.0

        logger.info(
            "ChunkRetry firing: doc=%s chunks=%d refused_cells=%d budget=$%.2f",
            doc_id,
            len(chunks),
            len(refused_indices),
            self.config.max_total_cost_usd,
        )

        # Per-column accumulator: collect candidate cells from each chunk
        # for any column that's still refused. Indexed by column_id.
        refused_col_ids = {cells[i].column_id for i in refused_indices}
        candidates: dict[str, list[ExtractionCell]] = {col_id: [] for col_id in refused_col_ids}

        cumulative_cost = 0.0
        for chunk_index, chunk in enumerate(chunks):
            if cumulative_cost >= self.config.max_total_cost_usd:
                logger.warning(
                    "ChunkRetry budget exhausted at chunk %d/%d (cost=$%.4f)",
                    chunk_index,
                    len(chunks),
                    cumulative_cost,
                )
                break
            try:
                chunk_result = await extract_chunk(chunk.text, doc_id)
            except Exception as exc:  # log and continue
                # A chunk failure shouldn't kill the rest of retry. The
                # other chunks may still recover the value.
                logger.warning(
                    "ChunkRetry chunk %d/%d failed: %s",
                    chunk_index + 1,
                    len(chunks),
                    exc,
                )
                continue

            cumulative_cost += chunk_result.cost_usd
            for cell in chunk_result.cells:
                if cell.column_id in candidates and cell.status == "extracted":
                    candidates[cell.column_id].append(cell)

        if not any(candidates.values()):
            return cells, cumulative_cost

        # Promote successful candidates back into the original cells.
        new_cells = list(cells)
        for i in refused_indices:
            cell = cells[i]
            cell_candidates = candidates.get(cell.column_id, [])
            if not cell_candidates:
                continue
            promoted = self._promote_candidates(cell, cell_candidates)
            new_cells[i] = promoted
            logger.info(
                "ChunkRetry recovered cell column=%s doc=%s candidates=%d",
                cell.column_id,
                doc_id,
                len(cell_candidates),
            )

        return tuple(new_cells), cumulative_cost

    # ---- Internals ---------------------------------------------------

    def _promote_candidates(
        self,
        original: ExtractionCell,
        candidates: list[ExtractionCell],
    ) -> ExtractionCell:
        """Pick the best candidate value across chunks and promote it.

        Strategy:
          - If exactly one candidate value (modulo string-equal
            comparison of ``ai_value``), promote it directly.
          - If multiple distinct values, take the FIRST candidate
            (leftmost-in-document wins — typically the most
            authoritative occurrence in legal documents) and append
            the others' citations.
          - If a candidate has more citations than the original,
            inherit those citations.

        Confidence is set to ``config.confidence_on_recovery``
        regardless — chunk-retry doesn't have ground truth on which
        chunk's answer is "more right".
        """
        # Dedup by string-form of ai_value. Preserve order (first occurrence wins).
        seen: dict[str, ExtractionCell] = {}
        for cand in candidates:
            key = str(cand.ai_value)
            if key not in seen:
                seen[key] = cand

        unique = list(seen.values())[: self.config.max_candidates_per_cell]
        winner = unique[0]

        # Union all citations from all candidates (capped).
        all_citations = ()
        for cand in unique:
            all_citations = all_citations + cand.citations
        all_citations = all_citations[: self.config.max_candidates_per_cell]

        rationale_suffix = " [chunk-retry recovered: extracted from sub-document]"
        if len(unique) > 1:
            rationale_suffix += f" ({len(unique)} candidates merged)"

        return original.model_copy(
            update={
                "status": "extracted",
                "ai_value": winner.ai_value,
                "confidence": self.config.confidence_on_recovery,
                "citations": all_citations,
                "rationale": (winner.rationale or original.rationale or "") + rationale_suffix,
            }
        )


def iter_chunks(
    text: str,
    *,
    max_chars: int = 8000,
    min_chars: int = 1000,
) -> Iterator[ChunkSpan]:
    """Public chunking helper.

    Convenience entry point for callers that want to use the same
    chunking algorithm chunk-retry uses (e.g., for benchmark scripts
    or other downstream tools). Returns the same :class:`ChunkSpan`
    objects produced internally.
    """
    yield from _chunk_text(text, max_chars=max_chars, min_chars=min_chars)


__all__ = [
    "ChunkExtractFn",
    "ChunkExtractResult",
    "ChunkRetry",
    "ChunkRetryConfig",
    "ChunkSpan",
    "iter_chunks",
]
