"""Structured metrics — accuracy, JSON field match, numeric comparisons, P/R/F1.

These metrics target structured outputs (dicts or attribute objects) and
numeric outputs. They follow the same conventions as
:mod:`kaos_llm_core.metrics.text`: ``None`` returns ``0.0``; non-coercible
inputs raise ``TypeError`` with a recovery hint.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from kaos_core.logging import get_logger

from kaos_llm_core.metrics.text import exact_match

logger = get_logger(__name__)


def accuracy(prediction: Any, gold: Any) -> float:
    """Alias for :func:`exact_match` — semantic clarity for classification."""
    return exact_match(prediction, gold)


def _get_field(obj: Any, field: str) -> Any:
    """Extract ``field`` from a dict-like or attr-like object. Returns sentinel
    ``_MISSING`` when the field is absent."""
    if obj is None:
        return _MISSING
    if isinstance(obj, dict):
        return obj.get(field, _MISSING)
    if hasattr(obj, field):
        try:
            return getattr(obj, field)
        except Exception as exc:
            logger.debug("_get_field: getattr(%r, %r) failed: %s", type(obj).__name__, field, exc)
            return _MISSING
    return _MISSING


_MISSING = object()


def json_field_match(field: str) -> Callable[[Any, Any], float]:
    """Factory: extract ``field`` from prediction and gold and compare.

    Returns 0.0 if either side is missing the field. Comparison is by
    equality (``==``). Use for "did the model populate this specific field
    correctly" checks where other fields don't matter.
    """
    if not isinstance(field, str) or not field:
        raise TypeError(
            "json_field_match() field must be a non-empty string. "
            "Alternative: use exact_match for whole-object equality."
        )

    def _metric(prediction: Any, gold: Any) -> float:
        p = _get_field(prediction, field)
        g = _get_field(gold, field)
        if p is _MISSING or g is _MISSING:
            return 0.0
        return 1.0 if p == g else 0.0

    _metric.__name__ = f"json_field_match[{field}]"
    return _metric


def _to_float(value: Any) -> float | None:
    """Coerce ``value`` to a float. Returns ``None`` if not numeric."""
    if value is None:
        return None
    if isinstance(value, bool):
        # bool is a subclass of int — treat explicitly so True/False don't slip through
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, dict):
        # Try common numeric field names
        for key in ("answer", "value", "score", "result"):
            if key in value:
                return _to_float(value[key])
        return None
    if hasattr(value, "answer"):
        try:
            return _to_float(value.answer)  # type: ignore[attr-defined]
        except Exception as exc:
            logger.debug("_to_float: access to .answer failed: %s", exc)
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def numeric_close(tolerance: float = 1e-6) -> Callable[[Any, Any], float]:
    """Factory: 1.0 if ``abs(pred - gold) <= tolerance``, else 0.0.

    Returns 0.0 if either input cannot be coerced to a float.
    """
    if tolerance < 0:
        raise ValueError(
            "numeric_close() tolerance must be non-negative. "
            "Alternative: use numeric_ratio for a continuous score."
        )

    def _metric(prediction: Any, gold: Any) -> float:
        p = _to_float(prediction)
        g = _to_float(gold)
        if p is None or g is None:
            return 0.0
        return 1.0 if abs(p - g) <= tolerance else 0.0

    _metric.__name__ = f"numeric_close[{tolerance}]"
    return _metric


def numeric_ratio(prediction: Any, gold: Any) -> float:
    """Continuous numeric score in ``[0, 1]``.

    Returns ``1.0 - min(|pred - gold| / max(|gold|, 1), 1.0)``. Returns 0.0 if
    either input cannot be coerced to a float.
    """
    p = _to_float(prediction)
    g = _to_float(gold)
    if p is None or g is None:
        return 0.0
    denom = max(abs(g), 1.0)
    raw = abs(p - g) / denom
    return float(1.0 - min(raw, 1.0))


def _coerce_iterable(value: Any) -> list[Any] | None:
    """Coerce ``value`` to a list of items for set/multiset metrics."""
    if value is None:
        return None
    if isinstance(value, str | bytes):
        # A bare string is a single item, not an iterable of characters
        return [value]
    if isinstance(value, dict):
        # Pull out a sensible field if present
        for key in ("answer", "items", "values", "results"):
            if key in value:
                return _coerce_iterable(value[key])
        return list(value.keys())
    if hasattr(value, "answer"):
        try:
            return _coerce_iterable(value.answer)  # type: ignore[attr-defined]
        except Exception as exc:
            logger.debug("_coerce_iterable: access to .answer failed: %s", exc)
            return None
    try:
        return list(value)
    except TypeError:
        return None


def precision_recall_f1(
    mode: Literal["set", "multiset"] = "set",
) -> Callable[[Any, Any], dict[str, float]]:
    """Factory: precision/recall/F1 over iterables.

    ``mode="set"`` deduplicates inputs (set semantics). ``mode="multiset"``
    counts repetitions (multiset semantics).

    Returns a dict ``{"precision": ..., "recall": ..., "f1": ...}``. To plug
    into a single-float optimizer loop, wrap with::

        prf = precision_recall_f1()
        f1_only = lambda p, g: prf(p, g)["f1"]
    """
    if mode not in ("set", "multiset"):
        raise ValueError(
            f"precision_recall_f1 mode must be 'set' or 'multiset', got {mode!r}. "
            "Alternative: pass mode='set' for the typical use case."
        )

    def _metric(prediction: Any, gold: Any) -> dict[str, float]:
        p = _coerce_iterable(prediction)
        g = _coerce_iterable(gold)
        if p is None or g is None:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        if mode == "set":
            p_set = set(map(_hashable, p))
            g_set = set(map(_hashable, g))
            tp = len(p_set & g_set)
            fp = len(p_set - g_set)
            fn = len(g_set - p_set)
        else:
            from collections import Counter

            p_counter = Counter(_hashable(x) for x in p)
            g_counter = Counter(_hashable(x) for x in g)
            tp = sum((p_counter & g_counter).values())
            fp = sum(p_counter.values()) - tp
            fn = sum(g_counter.values()) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}

    _metric.__name__ = f"precision_recall_f1[{mode}]"
    return _metric


def _hashable(value: Any) -> Any:
    """Make a value hashable for set/Counter operations."""
    try:
        hash(value)
        return value
    except TypeError:
        return repr(value)
