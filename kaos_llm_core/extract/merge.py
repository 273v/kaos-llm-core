"""AlphaLLMMerger — union rule-based alpha extractions with LLM output.

The composition layer of the WS-TR alpha-extractor sprint. Each
``ExtractionCell`` produced by the LLM pipeline is augmented with the
candidates the alpha extractors enumerate for that column's
``ColumnType``. The LLM remains the decider; alpha exhaustively
enumerates.

Merge policy (applied per cell):

1. **LLM returned a value AND an alpha hit matches it** (case- and
   whitespace-normalized, with type-specific canonicalization for
   dates / money): keep the LLM's value; APPEND the alpha span to
   ``citations`` so the cell carries both sources. The confidence
   stays at the LLM's level.

2. **LLM returned a value but no alpha hit matches it**: keep the LLM
   cell UNCHANGED. The LLM saw something the rule layer missed (the
   gazetteer has documented gaps — hyphenated names, lowercase
   suffixes, etc.). Do not discard the LLM's answer just because the
   regex didn't find it.

3. **LLM refused (status != ``"extracted"``) AND alpha has exactly
   ONE hit**: PROMOTE the alpha hit to the cell's ``ai_value`` with
   ``status="extracted"`` and ``confidence=0.8`` (configurable on the
   merger). Attach the alpha span as the sole citation. This is the
   primary precision lever for the cheap-tier models that often
   refuse on clauses the regex can clearly identify.

4. **LLM refused AND alpha has multiple hits**: keep the refusal. The
   LLM is the tiebreaker for ambiguous cases; we will not pick one
   alpha hit over another heuristically.

5. **LLM errored**: pass through — merger never alters ``status="error"``
   cells.

Dispatch is keyed by :class:`ColumnType`:

- ``DATE`` / ``DATETIME``  → :class:`AlphaDateExtractor`
- ``ENTITY_ROLE``          → :class:`AlphaEntityExtractor`

Additional column types (MONEY, SCORE, etc.) land in PR-6f.4 and PR-6f.5.
Columns whose type is not dispatchable receive no augmentation — the
LLM cell passes through as-is.

Disable via ``provenance="none"`` on :class:`Extract` or by passing
``merger=None`` explicitly. Both paths skip the merger entirely.
"""

from __future__ import annotations

import datetime
import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

from kaos_content.model.extraction import ExtractionCell, ExtractionCitation
from kaos_content.model.tabular import ColumnType
from kaos_nlp_core.extract.alpha import (
    AlphaDateExtractor,
    AlphaDurationExtractor,
    AlphaEntityExtractor,
    AlphaMoneyExtractor,
    AlphaNumberExtractor,
    AlphaPercentExtractor,
    DurationMatch,
    EntityMatch,
    MoneyMatch,
)
from kaos_nlp_core.extract.base_extractor import AlphaSpan, BaseAlphaExtractor

if TYPE_CHECKING:
    from kaos_llm_core.signatures.extraction import ColumnSpec


@dataclass(frozen=True, slots=True)
class MergerConfig:
    """Tunable policy for :class:`AlphaLLMMerger`.

    - ``alpha_confidence_on_promotion`` — confidence assigned to a cell
      when an alpha hit is promoted into an LLM refusal. Defaults to
      ``0.8`` (high but below the LLM's typical 0.9-0.95 so that
      consumers can prioritize LLM-confirmed values).
    - ``max_alpha_citations_per_cell`` — cap on alpha spans attached
      to a single cell's citations. Default 5 (enough for a
      multi-reference contract without bloating the artifact).
    """

    alpha_confidence_on_promotion: float = 0.8
    max_alpha_citations_per_cell: int = 5


# ----------------------------------------------------------------------
# Dispatch table — ColumnType → alpha extractor class.
# ----------------------------------------------------------------------
#
# Keyed by ColumnType rather than ColumnTypeName so the merger can be
# dispatched directly from kaos-content's typed enum. The mapping is
# built lazily inside _build_dispatch_table() to avoid instantiating
# extractors until the merger itself is constructed.


def _build_dispatch_table() -> dict[ColumnType, BaseAlphaExtractor]:
    return {
        ColumnType.DATE: AlphaDateExtractor(),
        ColumnType.DATETIME: AlphaDateExtractor(),
        ColumnType.ENTITY_ROLE: AlphaEntityExtractor(),
        ColumnType.MONEY: AlphaMoneyExtractor(),
        ColumnType.INTEGER: AlphaNumberExtractor(),
        ColumnType.FLOAT: AlphaNumberExtractor(),
        ColumnType.DURATION: AlphaDurationExtractor(),
        # SCORE columns (per-cell confidence / risk ratings) benefit from
        # the number / percent extractors. The merger uses the FLOAT-
        # canonicalizer for SCORE — the caller is expected to emit a
        # fractional [0,1] score regardless of surface form.
        ColumnType.SCORE: AlphaPercentExtractor(),
    }


# ----------------------------------------------------------------------
# Value-canonicalization helpers.
# ----------------------------------------------------------------------
#
# Each entry normalizes both sides (LLM value vs. alpha value) to a
# hashable key so the merger can ask "did alpha and the LLM agree?".


def _canonical_date(value: Any) -> str | None:
    """ISO 8601 string or ``None``. Handles ``datetime.date`` / ``datetime.datetime``
    / ISO-string / ``YYYY-MM-DD``-shaped inputs uniformly."""
    if isinstance(value, datetime.datetime):
        return value.date().isoformat()
    if isinstance(value, datetime.date):
        return value.isoformat()
    if isinstance(value, str):
        s = value.strip()
        # ISO 8601 fast path.
        try:
            return datetime.date.fromisoformat(s[:10]).isoformat()
        except (ValueError, IndexError):
            return None
    return None


def _canonical_entity(value: Any) -> str | None:
    """Canonical entity name: lowercased + whitespace-collapsed + trailing
    punctuation stripped + entity suffix stripped.

    Why strip the suffix: the LLM typically emits a full party string
    like ``"Acme Corp."``, while the alpha extractor splits name and
    suffix into separate fields (``EntityMatch(name="Acme",
    entity_type="Corp.")``). A naive canonicalization would make these
    compare unequal. We normalize by keeping only the name portion,
    lowercased — so ``"Acme Corp."`` and ``EntityMatch("Acme", "Corp.")``
    both collapse to ``"acme"``.

    This matches the original kelvin-nlp behavior where the gazetteer
    returned name-only strings.
    """
    raw: str
    if isinstance(value, EntityMatch):
        raw = value.name
    elif isinstance(value, dict):
        raw_val = value.get("name") or value.get("value") or ""
        raw = raw_val if isinstance(raw_val, str) else ""
    elif isinstance(value, str):
        raw = value
    else:
        return None
    raw = raw.strip(" ,.;:")
    if not raw:
        return None
    # Strip a trailing corporate-entity suffix so LLM-form matches
    # alpha-form. Use the extractor's gazetteer as the suffix source.
    tokens = raw.split()
    while tokens:
        last = tokens[-1].rstrip(",;:")
        if last in AlphaEntityExtractor.entity_set:
            tokens.pop()
        else:
            break
    if not tokens:
        return None
    return " ".join(tokens).lower()


def _canonical_money(value: Any) -> str | None:
    """Canonical money key: ``"<decimal>:<ISO>"``.

    Accepts :class:`MoneyMatch`, ``(Decimal, str)`` / ``(str, str)``
    tuples, dicts with ``amount``/``currency`` keys, and bare Decimals
    / numeric strings (treated as USD by default — common LLM output
    shape). Returns ``None`` for unparseable inputs.
    """
    amount: Decimal | None = None
    currency: str | None = None

    if isinstance(value, MoneyMatch):
        amount = value.amount
        currency = value.currency
    elif isinstance(value, dict):
        amount_raw = value.get("amount") or value.get("value")
        currency = value.get("currency") or value.get("iso")
        if isinstance(amount_raw, Decimal):
            amount = amount_raw
        elif isinstance(amount_raw, (int, float, str)):
            try:
                amount = Decimal(str(amount_raw))
            except (InvalidOperation, ValueError):
                return None
    elif isinstance(value, tuple) and len(value) == 2:
        amount_raw, currency = value
        if isinstance(amount_raw, Decimal):
            amount = amount_raw
        elif isinstance(amount_raw, (int, float, str)):
            try:
                amount = Decimal(str(amount_raw))
            except (InvalidOperation, ValueError):
                return None
    elif isinstance(value, (int, float, Decimal, str)):
        try:
            amount = value if isinstance(value, Decimal) else Decimal(str(value))
        except (InvalidOperation, ValueError):
            return None
        currency = "USD"

    if amount is None or not isinstance(currency, str):
        return None
    # Normalize the Decimal to its canonical form (strip trailing zeros
    # but preserve scale for sub-cent precision). str(Decimal("13.50")) ==
    # "13.50"; str(Decimal("13.50").normalize()) == "1.35E+1" — we want
    # the former so equal amounts canonicalize identically.
    return f"{amount}:{currency.upper()}"


def _canonical_number(value: Any) -> str | None:
    """Canonical number key: string form of Decimal(value). Accepts
    bare numerics, numeric strings, and :class:`Decimal`."""
    if isinstance(value, bool):
        # bool is a subclass of int; exclude it explicitly so True != 1.
        return None
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (int, float)):
        return str(Decimal(str(value)))
    if isinstance(value, str):
        stripped = value.strip().replace(",", "")
        try:
            return str(Decimal(stripped))
        except (InvalidOperation, ValueError):
            return None
    return None


def _canonical_duration(value: Any) -> str | None:
    """Canonical duration key: ``"<total_seconds>"`` as string.

    Accepts :class:`DurationMatch`, ``(Decimal, str)`` / ``(str, str)``
    tuples, dicts with ``quantity``/``unit``/``total_seconds`` keys, or
    bare numeric values (treated as seconds).
    """
    from kaos_nlp_core.locale_data import DURATION_MAP as _DM

    unit_to_seconds = _DM.get("en", {})

    if isinstance(value, DurationMatch):
        return str(value.total_seconds)
    if isinstance(value, dict):
        if "total_seconds" in value:
            try:
                return str(Decimal(str(value["total_seconds"])))
            except (InvalidOperation, ValueError):
                return None
        qty_raw = value.get("quantity") or value.get("value")
        unit = value.get("unit")
        if qty_raw is None or not isinstance(unit, str):
            return None
        try:
            qty = qty_raw if isinstance(qty_raw, Decimal) else Decimal(str(qty_raw))
        except (InvalidOperation, ValueError):
            return None
        seconds_per = unit_to_seconds.get(unit.lower().rstrip("s"))
        if seconds_per is None:
            seconds_per = unit_to_seconds.get(unit.lower())
        if seconds_per is None:
            return None
        return str(qty * Decimal(seconds_per))
    if isinstance(value, tuple) and len(value) == 2:
        qty_raw, unit = value
        if not isinstance(unit, str):
            return None
        try:
            qty = qty_raw if isinstance(qty_raw, Decimal) else Decimal(str(qty_raw))
        except (InvalidOperation, ValueError):
            return None
        seconds_per = unit_to_seconds.get(unit.lower().rstrip("s"))
        if seconds_per is None:
            seconds_per = unit_to_seconds.get(unit.lower())
        if seconds_per is None:
            return None
        return str(qty * Decimal(seconds_per))
    if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        try:
            return str(value if isinstance(value, Decimal) else Decimal(str(value)))
        except (InvalidOperation, ValueError):
            return None
    return None


def _canonical(value: Any, column_type: ColumnType) -> str | None:
    """Pick the right canonicalizer for a column type."""
    if column_type in (ColumnType.DATE, ColumnType.DATETIME):
        return _canonical_date(value)
    if column_type == ColumnType.ENTITY_ROLE:
        return _canonical_entity(value)
    if column_type == ColumnType.MONEY:
        return _canonical_money(value)
    if column_type in (ColumnType.INTEGER, ColumnType.FLOAT, ColumnType.SCORE):
        return _canonical_number(value)
    if column_type == ColumnType.DURATION:
        return _canonical_duration(value)
    return None


def _alpha_values_for_column(
    spans: Iterable[AlphaSpan[Any]], column_type: ColumnType
) -> list[tuple[str, AlphaSpan[Any]]]:
    """Pair each alpha span with its canonical key for the column type."""
    pairs: list[tuple[str, AlphaSpan[Any]]] = []
    for span in spans:
        key = _canonical(span.value, column_type)
        if key is not None:
            pairs.append((key, span))
    return pairs


def _citation_from_span(
    span: AlphaSpan[Any], source_text: str, source_uri: str
) -> ExtractionCitation:
    """Build a kaos-content :class:`ExtractionCitation` from an alpha span.

    The alpha extractor gives us char_span + value; we synthesize the
    remaining citation fields from the source text slice. The
    ``block_ref`` / ``page`` / ``parent_block_ref`` defaults mirror
    what the Cited[T] → ExtractionCitation adapter uses when no
    ContentDocument AST is attached.
    """
    snippet = source_text[span.start : span.end]
    snippet_hash = hashlib.sha256(snippet.encode("utf-8")).hexdigest()
    return ExtractionCitation(
        block_ref=f"#/span/{source_uri}",
        page=1,
        char_span=(span.start, span.end),
        snippet=snippet,
        snippet_sha256=snippet_hash,
        confidence=1.0,  # rule-based — deterministic
    )


# ----------------------------------------------------------------------
# The merger.
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AlphaLLMMerger:
    """Compose alpha extractor hits with LLM-projected cells.

    Construct with default dispatch::

        merger = AlphaLLMMerger()

    Or inject a custom dispatch table (e.g., to add a caller-specific
    extractor) via the frozen ``dispatch`` field on the dataclass::

        merger = AlphaLLMMerger(dispatch={ColumnType.DATE: my_extractor})

    Then call :meth:`merge` inside ``Extract.extract`` to augment cells.
    """

    config: MergerConfig = field(default_factory=MergerConfig)
    dispatch: dict[ColumnType, BaseAlphaExtractor] = field(default_factory=_build_dispatch_table)

    def merge(
        self,
        cells: tuple[ExtractionCell, ...],
        *,
        columns: tuple[ColumnSpec, ...],
        source_text: str,
        source_uri: str = "doc:inline",
    ) -> tuple[ExtractionCell, ...]:
        """Return a new tuple of cells with alpha augmentation applied.

        ``columns`` must be aligned with ``cells`` positionally — each
        ``cells[i]`` corresponds to ``columns[i]``. This matches the
        ``Extract._project_cells`` ordering guarantee.
        """
        if len(cells) != len(columns):
            msg = f"cell/column count mismatch: {len(cells)} cells vs. {len(columns)} columns"
            raise ValueError(msg)

        new_cells: list[ExtractionCell] = []
        alpha_cache: dict[type[BaseAlphaExtractor], list[AlphaSpan[Any]]] = {}

        for cell, col in zip(cells, columns, strict=True):
            new_cells.append(
                self._merge_one(
                    cell=cell,
                    col=col,
                    source_text=source_text,
                    source_uri=source_uri,
                    alpha_cache=alpha_cache,
                )
            )
        return tuple(new_cells)

    # -- internals ----------------------------------------------------

    def _merge_one(
        self,
        *,
        cell: ExtractionCell,
        col: ColumnSpec,
        source_text: str,
        source_uri: str,
        alpha_cache: dict[type[BaseAlphaExtractor], list[AlphaSpan[Any]]],
    ) -> ExtractionCell:
        """Apply the merge policy to one cell.

        Dispatch order:

        1. Errors always pass through unchanged.
        2. List-typed columns: consult ``col.constraints["alpha_extractor"]``
           for an opt-in hint (``"entity"``, ``"date"``, ``"money"``).
           Without a hint, list columns are NOT auto-augmented — we
           don't know whether the caller wants the list to hold
           entities, dates, dollar amounts, or something else.
        3. Scalar columns: map ``col.column_type`` to a :class:`ColumnType`
           and dispatch to the registered alpha extractor for that type.
        """
        if cell.status == "error":
            return cell

        # Path A: list columns with an explicit alpha hint.
        if col.column_type == "list":
            hint = col.constraints.get("alpha_extractor")
            if not hint:
                return cell
            extractor = self._dispatch_by_hint(hint)
            if extractor is None:
                return cell
            alpha_spans = self._get_cached_alpha_spans(extractor, source_text, alpha_cache)
            col_type = _hint_to_column_type(hint)
            if col_type is None:
                return cell
            return self._merge_list_cell(
                cell=cell,
                col=col,
                col_type=col_type,
                alpha_spans=alpha_spans,
                source_text=source_text,
                source_uri=source_uri,
            )

        # Path B: scalar columns — type-driven dispatch.
        col_type = _map_column_type_name_to_enum(col.column_type)
        if col_type is None or col_type not in self.dispatch:
            return cell
        extractor = self.dispatch[col_type]
        alpha_spans = self._get_cached_alpha_spans(extractor, source_text, alpha_cache)
        return self._merge_scalar_cell(
            cell=cell,
            col_type=col_type,
            alpha_spans=alpha_spans,
            source_text=source_text,
            source_uri=source_uri,
        )

    def _dispatch_by_hint(self, hint: str) -> BaseAlphaExtractor | None:
        """Resolve an ``alpha_extractor`` hint to a concrete extractor."""
        hint_map = {
            "entity": ColumnType.ENTITY_ROLE,
            "date": ColumnType.DATE,
            "datetime": ColumnType.DATETIME,
        }
        key = hint_map.get(hint)
        if key is None:
            return None
        return self.dispatch.get(key)

    def _get_cached_alpha_spans(
        self,
        extractor: BaseAlphaExtractor,
        source_text: str,
        cache: dict[type[BaseAlphaExtractor], list[AlphaSpan[Any]]],
    ) -> list[AlphaSpan[Any]]:
        """Cache alpha hits per extractor-class so if two columns share
        the same extractor (e.g., two DATE columns) we tokenize once."""
        cls = type(extractor)
        if cls not in cache:
            cache[cls] = list(extractor.extract_spans(source_text))
        return cache[cls]

    def _merge_scalar_cell(
        self,
        *,
        cell: ExtractionCell,
        col_type: ColumnType,
        alpha_spans: list[AlphaSpan[Any]],
        source_text: str,
        source_uri: str,
    ) -> ExtractionCell:
        alpha_pairs = _alpha_values_for_column(alpha_spans, col_type)

        if cell.status == "extracted":
            # LLM has a value. Strengthen citations if alpha agrees.
            llm_canonical = _canonical(cell.ai_value, col_type)
            matched_spans: list[AlphaSpan[Any]] = []
            if llm_canonical is not None:
                matched_spans = [s for k, s in alpha_pairs if k == llm_canonical]
            if not matched_spans:
                # LLM saw something alpha missed — pass through.
                return cell

            extra_citations = tuple(
                _citation_from_span(span, source_text, source_uri)
                for span in matched_spans[: self.config.max_alpha_citations_per_cell]
            )
            return cell.model_copy(update={"citations": cell.citations + extra_citations})

        # Refused or unclear — promote if alpha has exactly one hit.
        if len(alpha_pairs) == 1:
            _key, span = alpha_pairs[0]
            value = _alpha_value_for_cell(span.value, col_type)
            citation = _citation_from_span(span, source_text, source_uri)
            return cell.model_copy(
                update={
                    "status": "extracted",
                    "ai_value": value,
                    "confidence": self.config.alpha_confidence_on_promotion,
                    "rationale": (cell.rationale or "")
                    + " [alpha-promoted: rule-based extraction]",
                    "citations": (*cell.citations, citation),
                }
            )

        # Multiple alpha candidates for a refused cell — don't pick one
        # heuristically; keep the refusal. The LLM is the tiebreaker.
        return cell

    def _merge_list_cell(
        self,
        *,
        cell: ExtractionCell,
        col: ColumnSpec,
        col_type: ColumnType,
        alpha_spans: list[AlphaSpan[Any]],
        source_text: str,
        source_uri: str,
    ) -> ExtractionCell:
        """For list-typed columns, union the LLM list with alpha names."""
        alpha_pairs = _alpha_values_for_column(alpha_spans, col_type)

        if cell.status == "extracted" and isinstance(cell.ai_value, list):
            llm_values = list(cell.ai_value)
            llm_keys = {_canonical(v, col_type) for v in llm_values}
            llm_keys.discard(None)

            # Add alpha hits that don't collide with LLM values.
            added_citations: list[ExtractionCitation] = []
            for key, span in alpha_pairs:
                if key not in llm_keys:
                    llm_values.append(_alpha_value_for_cell(span.value, col_type))
                    llm_keys.add(key)
                    if len(added_citations) < self.config.max_alpha_citations_per_cell:
                        added_citations.append(_citation_from_span(span, source_text, source_uri))
            if added_citations:
                return cell.model_copy(
                    update={
                        "ai_value": llm_values,
                        "citations": cell.citations + tuple(added_citations),
                    }
                )
            return cell

        # Refused / empty list — populate from alpha if we have at least
        # one hit.
        if alpha_pairs:
            values = [_alpha_value_for_cell(s.value, col_type) for _, s in alpha_pairs]
            # Dedup by canonical key, preserving first-seen order.
            seen: set[str] = set()
            unique_values: list[Any] = []
            unique_citations: list[ExtractionCitation] = []
            for (key, span), val in zip(alpha_pairs, values, strict=True):
                if key in seen:
                    continue
                seen.add(key)
                unique_values.append(val)
                if len(unique_citations) < self.config.max_alpha_citations_per_cell:
                    unique_citations.append(_citation_from_span(span, source_text, source_uri))
            return cell.model_copy(
                update={
                    "status": "extracted",
                    "ai_value": unique_values,
                    "confidence": self.config.alpha_confidence_on_promotion,
                    "rationale": (cell.rationale or "")
                    + " [alpha-promoted: rule-based enumeration]",
                    "citations": (*cell.citations, *unique_citations),
                }
            )

        return cell


def _alpha_value_for_cell(alpha_value: Any, col_type: ColumnType) -> Any:
    """Convert an alpha extractor's native value into the cell-level
    shape.

    - ``EntityMatch`` → ``"Acme Corp."`` string
    - ``MoneyMatch`` → ``{"amount": Decimal, "currency": str}``
    - ``DurationMatch`` → ``{"quantity": Decimal, "unit": str, "total_seconds": Decimal}``
    - ``datetime.datetime`` → ``datetime.date``
    - everything else passes through unchanged
    """
    if isinstance(alpha_value, EntityMatch):
        # Cell-friendly form: "name + suffix" reconstructed.
        return f"{alpha_value.name} {alpha_value.entity_type}".strip()
    if isinstance(alpha_value, MoneyMatch):
        # Dict form serializes through JSON (ExtractionCell is Pydantic
        # and round-trips through batch_run's JSONL log). Decimals dump
        # as strings by default.
        return {"amount": alpha_value.amount, "currency": alpha_value.currency}
    if isinstance(alpha_value, DurationMatch):
        return {
            "quantity": alpha_value.quantity,
            "unit": alpha_value.unit,
            "total_seconds": alpha_value.total_seconds,
        }
    if isinstance(alpha_value, datetime.datetime):
        return alpha_value.date()
    return alpha_value


def _hint_to_column_type(hint: str) -> ColumnType | None:
    """Map a ``constraints["alpha_extractor"]`` opt-in hint to a
    :class:`ColumnType`. Used to canonicalize/dispatch alpha spans for
    list-typed columns (where the string-level ``column_type`` is
    ``"list"`` and doesn't carry inner-type info)."""
    return {
        "entity": ColumnType.ENTITY_ROLE,
        "date": ColumnType.DATE,
        "datetime": ColumnType.DATETIME,
    }.get(hint)


def _map_column_type_name_to_enum(name: str) -> ColumnType | None:
    """Map ColumnSpec.column_type (a string) to the kaos-content enum.

    Returns ``None`` for names that don't map (e.g., ``"enum"``, ``"object"``).
    """
    mapping = {
        "date": ColumnType.DATE,
        "datetime": ColumnType.DATETIME,
        "entity_role": ColumnType.ENTITY_ROLE,
        "money": ColumnType.MONEY,
        "score": ColumnType.SCORE,
        "verbatim_quote": ColumnType.VERBATIM_QUOTE,
        "string": ColumnType.TEXT,
        "integer": ColumnType.INTEGER,
        "number": ColumnType.FLOAT,
        "boolean": ColumnType.BOOLEAN,
        "list": ColumnType.LIST,
        "object": ColumnType.STRUCT,
    }
    return mapping.get(name)


__all__ = [
    "AlphaLLMMerger",
    "MergerConfig",
]
