"""Text metrics — string equality, regex, contains, semantic similarity.

All metrics accept ``Any`` for prediction and gold and coerce to ``str``.
``None`` inputs return ``0.0`` (never raise). The ``regex_match``,
``contains``, and ``semantic_similarity`` helpers are factories — call them
once with their parameters and pass the returned callable to an optimizer.

For metrics that score against ``Example.outputs`` dicts (see
:func:`kaos_llm_core.optimization.evaluation.evaluate`), the gold argument
may be a ``dict``. By convention, metrics first try ``gold["answer"]`` if
gold is a dict; otherwise they coerce gold directly. This matches the
``DynamicPhase6Signature`` shape used by the MCP optimizer tools while
remaining flexible for direct Python use.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from typing import Any

from kaos_core.logging import get_logger

logger = get_logger(__name__)


def _coerce(value: Any) -> str | None:
    """Coerce a metric input to a string. ``None`` stays ``None``."""
    if value is None:
        return None
    if isinstance(value, dict):
        # Heuristic: prefer common output field names; fall back to str(dict).
        for key in ("answer", "output", "value", "text"):
            if key in value:
                inner = value[key]
                return None if inner is None else str(inner)
        return str(value)
    if hasattr(value, "answer"):
        try:
            inner = value.answer  # type: ignore[attr-defined]
            return None if inner is None else str(inner)
        except Exception as exc:
            logger.debug("_coerce: access to .answer failed: %s", exc)
    return str(value)


def exact_match(prediction: Any, gold: Any) -> float:
    """1.0 if str(prediction) == str(gold), else 0.0.

    Returns 0.0 if either side is ``None``.
    """
    p = _coerce(prediction)
    g = _coerce(gold)
    if p is None or g is None:
        return 0.0
    return 1.0 if p == g else 0.0


def case_insensitive_match(prediction: Any, gold: Any) -> float:
    """1.0 if ``str(prediction).lower() == str(gold).lower()``, else 0.0."""
    p = _coerce(prediction)
    g = _coerce(gold)
    if p is None or g is None:
        return 0.0
    return 1.0 if p.lower() == g.lower() else 0.0


_WS_RE = re.compile(r"\s+")
_TRAILING_PUNCT_RE = re.compile(r"[.,;:!?\s]+$")


def _normalize(text: str) -> str:
    """Normalize text for lenient label/free-text comparison.

    The normalization is intentionally aggressive — it's the right default
    for label-style classification tasks where the model commonly returns
    ``"Termination"`` or ``"termination."`` or ``'"termination"'`` instead
    of bare ``"termination"`` (the F2 finding from Phase 6 E2E).

    Steps:
      1. ``strip()`` outer whitespace
      2. lowercase
      3. strip surrounding ASCII quotes (``"``, ``'``, `` ` ``)
      4. strip trailing punctuation (``.,;:!?`` and trailing whitespace)
      5. collapse runs of internal whitespace to a single space
    """
    s = text.strip().lower().strip("\"'`")
    s = _TRAILING_PUNCT_RE.sub("", s)
    return _WS_RE.sub(" ", s)


def normalized_match(prediction: Any, gold: Any) -> float:
    """1.0 if normalized(prediction) == normalized(gold), else 0.0.

    Uses :func:`_normalize` which strips outer whitespace, lowercases,
    strips surrounding quotes, strips trailing punctuation, and collapses
    internal whitespace. Designed to be the right default for free-text
    label tasks where models routinely return "Termination" /
    "termination." / '"termination"' instead of bare "termination".
    """
    p = _coerce(prediction)
    g = _coerce(gold)
    if p is None or g is None:
        return 0.0
    return 1.0 if _normalize(p) == _normalize(g) else 0.0


def regex_match(pattern: str) -> Callable[[Any, Any], float]:
    """Factory: returns a metric that checks ``re.fullmatch(pattern, prediction)``.

    The gold argument is ignored — this is a predicate metric. Useful for
    "did the model produce a well-formed answer" checks regardless of the
    gold answer.

    Raises ``ValueError`` if the pattern does not compile.
    """
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise ValueError(
            f"regex_match pattern is not a valid regex: {exc}. "
            "Fix the pattern syntax. Alternative: use exact_match for literal comparison."
        ) from exc

    def _metric(prediction: Any, gold: Any) -> float:
        p = _coerce(prediction)
        if p is None:
            return 0.0
        return 1.0 if compiled.fullmatch(p) is not None else 0.0

    _metric.__name__ = f"regex_match[{pattern}]"
    return _metric


def contains(substring: str, *, case_insensitive: bool = False) -> Callable[[Any, Any], float]:
    """Factory: returns a metric that checks if ``str(prediction)`` contains substring.

    Gold is ignored. Set ``case_insensitive=True`` for a case-insensitive search.
    """
    if not isinstance(substring, str):
        raise TypeError(
            "contains() substring must be a string. "
            "Alternative: use exact_match for non-string equality."
        )
    needle = substring.lower() if case_insensitive else substring

    def _metric(prediction: Any, gold: Any) -> float:
        p = _coerce(prediction)
        if p is None:
            return 0.0
        haystack = p.lower() if case_insensitive else p
        return 1.0 if needle in haystack else 0.0

    _metric.__name__ = f"contains[{substring!r}]"
    return _metric


def semantic_similarity(
    model: str = "openai:text-embedding-3-small",
    *,
    threshold: float = 0.0,
) -> Callable[[Any, Any], float]:
    """Factory: returns a metric that embeds prediction and gold and compares.

    Returns the cosine similarity in ``[0, 1]`` (clamped). If ``threshold > 0``,
    returns ``1.0`` when similarity >= threshold and ``0.0`` otherwise (binary
    mode).

    Requires kaos-llm-client embeddings; gracefully returns ``0.0`` and logs a
    warning if the embedding provider is unavailable. Build the metric once and
    reuse it across an optimization run — each call performs two embedding
    requests.
    """

    def _metric(prediction: Any, gold: Any) -> float:
        p = _coerce(prediction)
        g = _coerce(gold)
        if p is None or g is None:
            return 0.0
        try:
            from kaos_llm_client import create_client
        except ImportError:
            logger.warning(
                "semantic_similarity metric requires kaos-llm-client; returning 0.0. "
                "Install kaos-llm-client or use a string metric like normalized_match."
            )
            return 0.0
        try:
            client = create_client(model)
            response = client.embed([p, g])
            vectors = response.embeddings
        except Exception as exc:
            logger.warning(
                "semantic_similarity embedding call failed: %s. "
                "Check API key and model id. Returning 0.0.",
                exc,
            )
            return 0.0
        if not vectors or len(vectors) < 2:
            return 0.0
        v1, v2 = vectors[0], vectors[1]
        score = _cosine(v1, v2)
        # Clamp to [0, 1]; raw cosine in [-1, 1].
        clamped = max(0.0, min(1.0, score))
        if threshold > 0.0:
            return 1.0 if clamped >= threshold else 0.0
        return clamped

    _metric.__name__ = f"semantic_similarity[{model}]"
    return _metric


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
