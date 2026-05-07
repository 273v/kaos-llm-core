"""Shared helpers for the 6 WS-TR.PR-6f.7 alpha extractor MCP tools.

All six tools share the same shape:

- Input: a single ``text`` string (the source to extract from).
- Output: a JSON-serializable list of span dicts with at least
  ``start`` / ``end`` offsets plus the per-extractor value field.
- Annotations: ``readOnlyHint=True``, ``destructiveHint=False``,
  ``idempotentHint=True``, ``openWorldHint=False`` (pure-local Rust+Python
  with no provider calls and no external state).
- No retries, no provider keys, no settings — these are deterministic
  rule-based functions.

This module centralizes the annotation block, the ``text`` parameter
schema, and a tiny serialization helper so the per-tool files stay
focused on their extractor + output shape.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from kaos_core.types.annotations import ToolAnnotations
from kaos_core.types.parameters import ParameterSchema

# Alpha extractors are pure-local, deterministic, side-effect-free.
# Claude Code auto-approves readOnlyHint=True tools (per
# docs/guides/tool-design.md), which is exactly what we want for
# extraction primitives that the LLM should be able to call freely.
ALPHA_ANNOTATIONS: ToolAnnotations = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)

# All six tools take exactly one ``text`` input.
ALPHA_TEXT_PARAM: ParameterSchema = ParameterSchema(
    name="text",
    type="string",
    description=(
        "Source text to extract from. Pass the raw document body or a "
        "specific passage. Empty string returns an empty list."
    ),
    required=True,
)

# A reasonable upper bound on how many spans a single tool call should
# inline-return. Past this, callers should chunk the text or filter
# downstream — alpha extractors are O(n) tokens but a single contract
# can yield hundreds of dates / entities. Honors the 16 KB inline
# threshold loosely (a typical span dict is ~80 bytes; 256 spans ≈ 20 KB).
ALPHA_MAX_RESULTS: int = 256


def coerce_decimal(value: Decimal) -> str:
    """Convert a :class:`Decimal` to a JSON-safe string preserving scale.

    Pydantic's default JSON encoder emits Decimal as a plain string,
    which is what we want — preserves precision (no IEEE-754 rounding)
    and round-trips cleanly through the MCP wire.
    """
    return str(value)


def common_text_input(inputs: dict[str, Any]) -> str | None:
    """Pull the ``text`` field out of the inputs dict; return ``None`` if
    absent / wrong type. Caller decides how to handle the failure."""
    raw = inputs.get("text")
    if not isinstance(raw, str):
        return None
    return raw


__all__ = [
    "ALPHA_ANNOTATIONS",
    "ALPHA_MAX_RESULTS",
    "ALPHA_TEXT_PARAM",
    "coerce_decimal",
    "common_text_input",
]
