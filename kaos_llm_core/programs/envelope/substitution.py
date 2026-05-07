"""JSON-pointer substitution for envelope step inputs and outputs.

The ``$.inputs.X`` / ``$.steps.id.output.field`` pointer syntax is
the only reference mechanism v3 envelopes need. Resolution is a
shallow walk over the live execution context dict.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from kaos_llm_core.programs.envelope._errors import ProgramEnvelopeError


def _resolve_pointer(pointer: str, ctx: Mapping[str, Any]) -> Any:
    """Resolve a '$.inputs.X' or '$.steps.id.output.field' pointer."""
    if not pointer.startswith("$."):
        msg = f"Pointer {pointer!r} must start with '$.'"
        raise ProgramEnvelopeError(msg)
    parts = pointer[2:].split(".")
    cursor: Any = ctx
    for segment in parts:
        if isinstance(cursor, Mapping):
            if segment not in cursor:
                msg = (
                    f"Pointer {pointer!r}: segment {segment!r} not found in "
                    f"context (available keys: {sorted(cursor.keys())[:10]})."
                )
                raise ProgramEnvelopeError(msg)
            cursor = cursor[segment]
        elif hasattr(cursor, segment):
            cursor = getattr(cursor, segment)
        else:
            msg = (
                f"Pointer {pointer!r}: cannot navigate segment {segment!r} on "
                f"{type(cursor).__name__}."
            )
            raise ProgramEnvelopeError(msg)
    return cursor


def _resolve_step_inputs(
    step_inputs: dict[str, str],
    ctx: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve every JSON-pointer in a step's inputs to its actual value."""
    resolved: dict[str, Any] = {}
    for field_name, pointer in step_inputs.items():
        resolved[field_name] = _resolve_pointer(pointer, ctx)
    return resolved


def _resolve_output_mapping(
    output_mapping: dict[str, str],
    ctx: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve the top-level output mapping into a result dict."""
    return {name: _resolve_pointer(p, ctx) for name, p in output_mapping.items()}
