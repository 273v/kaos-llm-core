"""FUND-10 — Cross-document dedup via content-hash normalization.

When extraction fans out across 30 docs, the same counterparty name,
governing law, or dollar amount shows up in many documents with minor
variation (whitespace drift, suffix form, date format). Without dedup,
downstream aggregates double-count: "Acme Corp" appears 17 times in
a 30-row table when there's really one entity.

This module provides a post-batch dedup layer that operates on a
sequence of :class:`~kaos_content.model.extraction.ExtractionCell`
objects (the output shape of ``extract_corpus``). It:

1. Computes a canonical form per cell using the column's
   :class:`~kaos_content.model.tabular.ColumnType` (entity → strip
   suffix + lowercase; date → ISO 8601; money → decimal + currency;
   text → whitespace-collapse + lowercase).
2. Hashes the canonical form (SHA-256).
3. Groups cells by ``(column_id, content_hash)`` and keeps the
   first-seen cell as the canonical representative.
4. Emits a ``DedupResult`` carrying the deduped cells + a lineage
   map from each ``(column_id, hash)`` → ``list[doc_id]`` so
   consumers can recover which documents contributed each unique
   value.

The caller (``extract_corpus`` or a post-batch aggregation step)
feeds the deduped cells into ``TabularDocument.from_cells()`` to
produce the final aggregate table.

Usage::

    from kaos_llm_core.dedup import dedup_cells

    result = dedup_cells(all_cells)
    print(result.removed_count)          # how many dups
    print(result.lineage["parties"])      # {hash: [doc_ids]}
    tabular = TabularDocument.from_cells(result.cells, ...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kaos_content.normalize import (
    canonical_hash as _canonical_hash,
)
from kaos_content.normalize import (
    normalize_date,
    normalize_entity,
    normalize_number,
    normalize_text,
)


@dataclass(frozen=True, slots=True)
class DedupResult:
    """Output of :func:`dedup_cells`.

    Attributes:
        cells: Deduped cell sequence in original order (first-seen wins).
        removed_count: How many duplicate cells were dropped.
        lineage: ``{column_id: {content_hash: [doc_ids]}}`` — which
            documents contributed each unique value. Consumers use
            this to report "this entity appears in 17/30 documents."
    """

    cells: tuple[Any, ...]
    removed_count: int
    lineage: dict[str, dict[str, list[str]]]


def dedup_cells(
    cells: list[Any] | tuple[Any, ...],
    *,
    column_types: dict[str, str] | None = None,
) -> DedupResult:
    """Deduplicate extraction cells by normalized content hash.

    Args:
        cells: Sequence of ``ExtractionCell`` (or any object with
            ``doc_id``, ``column_id``, ``ai_value``, ``status``).
        column_types: Optional mapping from ``column_id`` to
            ``ColumnType.value`` string (e.g. ``{"parties": "text",
            "effective_date": "date"}``). When provided, the canonical
            form respects the type (date normalization, entity suffix
            stripping, etc.). When absent, all columns use the text
            normalizer (whitespace-collapse + lowercase).

    Returns:
        :class:`DedupResult` with deduped cells + lineage map.
    """
    col_types = column_types or {}
    seen: dict[tuple[str, str], Any] = {}
    lineage: dict[str, dict[str, list[str]]] = {}
    kept: list[Any] = []
    removed = 0

    for cell in cells:
        status = getattr(cell, "status", None)
        if status != "extracted":
            kept.append(cell)
            continue

        col_id = getattr(cell, "column_id", "")
        doc_id = getattr(cell, "doc_id", "")
        ai_value = getattr(cell, "ai_value", None)

        if ai_value is None:
            kept.append(cell)
            continue

        col_type = col_types.get(col_id, "text")
        canonical = _canonicalize(str(ai_value), col_type)
        content_hash = _canonical_hash(canonical)

        key = (col_id, content_hash)
        lineage.setdefault(col_id, {}).setdefault(content_hash, []).append(doc_id)

        if key in seen:
            removed += 1
            continue

        seen[key] = cell
        kept.append(cell)

    return DedupResult(
        cells=tuple(kept),
        removed_count=removed,
        lineage=lineage,
    )


# ------------------------------------------------------------------
# Canonicalization — delegates to kaos_content.normalize (single
# source of truth for the entire platform). This module previously
# had its own 31-entry entity suffix set and float-based number
# normalization; both diverged from the canonical implementations.
# ------------------------------------------------------------------


def _canonicalize(value: str, col_type: str) -> str:
    """Normalize a cell value to a canonical hash-friendly form.

    Delegates to the shared :mod:`kaos_content.normalize` module so
    dedup hashes match across extraction cells, document-level dedup,
    and alpha-LLM merger comparisons.
    """
    if col_type in ("entity_role", "string", "text", "verbatim_quote"):
        return normalize_entity(value)
    if col_type == "date":
        return normalize_date(value)
    if col_type in ("money", "number", "integer", "float", "decimal"):
        return normalize_number(value)
    return normalize_text(value)


__all__ = ["DedupResult", "dedup_cells"]
