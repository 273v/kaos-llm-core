"""Cross-step validation helpers for the Program v3 envelope.

These helpers are called from ``ProgramEnvelope._validate_envelope``
(via lazy import to break the cycle).

Phase 16.5: ``_validate_step_kind_fields`` was removed because the
discriminated-union sub-models in :mod:`.models` enforce per-kind
required fields at parse time. Only ``_validate_pointer`` (for
JSON-pointer references) and ``_validate_type_refs`` (for
``$ref`` resolution into the hoisted ``types`` block) remain.
"""

from __future__ import annotations

from typing import Any

from kaos_llm_core.programs.envelope.models import InputSpec


def _validate_pointer(
    pointer: str,
    program_inputs: dict[str, InputSpec],
    seen_step_ids: set[str],
    *,
    where: str,
) -> None:
    """Validate a JSON-pointer reference. Raises ValueError on failure."""
    if not pointer.startswith("$."):
        msg = f"{where}: pointer {pointer!r} must start with '$.'"
        raise ValueError(msg)
    parts = pointer[2:].split(".")
    if not parts:
        msg = f"{where}: pointer {pointer!r} is malformed"
        raise ValueError(msg)
    source = parts[0]
    if source == "inputs":
        if len(parts) < 2:
            msg = f"{where}: pointer {pointer!r} must reference a specific input field"
            raise ValueError(msg)
        if parts[1] not in program_inputs:
            msg = (
                f"{where}: pointer {pointer!r} references program input "
                f"{parts[1]!r} which is not declared. Declared inputs: "
                f"{sorted(program_inputs)}."
            )
            raise ValueError(msg)
    elif source == "steps":
        if len(parts) < 4:
            msg = (
                f"{where}: pointer {pointer!r} must reference a step output "
                f"in the form '$.steps.<id>.output.<field>' "
                "(or '$.steps.<id>.output' for the whole output dict)."
            )
            raise ValueError(msg)
        if parts[1] not in seen_step_ids:
            msg = (
                f"{where}: pointer {pointer!r} references step {parts[1]!r} "
                f"which is not declared earlier in the steps list "
                "(forward references are not allowed; the DAG is linear). "
                f"Declared so far: {sorted(seen_step_ids)}."
            )
            raise ValueError(msg)
        if parts[2] != "output":
            msg = (
                f"{where}: pointer {pointer!r} must reference '.output' as the "
                f"second segment, got {parts[2]!r}."
            )
            raise ValueError(msg)
    else:
        msg = (
            f"{where}: pointer {pointer!r} must start with '$.inputs.' or "
            f"'$.steps.', got source {source!r}."
        )
        raise ValueError(msg)


def _validate_type_refs(
    type_spec: dict[str, Any],
    types_block: dict[str, dict[str, Any]],
    *,
    where: str,
) -> None:
    """Walk a type spec recursively; check every $ref resolves."""
    if not isinstance(type_spec, dict):
        return
    if "$ref" in type_spec:
        ref = type_spec["$ref"]
        if not isinstance(ref, str) or not ref.startswith("#/types/"):
            msg = f"{where}: $ref {ref!r} must be of the form '#/types/<name>'"
            raise ValueError(msg)
        type_name = ref[len("#/types/") :]
        if type_name not in types_block:
            msg = (
                f"{where}: $ref {ref!r} references type {type_name!r} which is "
                f"not declared in the envelope's 'types' block. Declared types: "
                f"{sorted(types_block)}."
            )
            raise ValueError(msg)
        return
    # Recurse into nested type fragments (array items, object properties)
    if "array" in type_spec and isinstance(type_spec["array"], dict):
        _validate_type_refs(type_spec["array"], types_block, where=f"{where}.array")
    if "items" in type_spec and isinstance(type_spec["items"], dict):
        _validate_type_refs(type_spec["items"], types_block, where=f"{where}.items")
    if "properties" in type_spec and isinstance(type_spec["properties"], dict):
        for prop_name, prop_type in type_spec["properties"].items():
            if isinstance(prop_type, dict):
                _validate_type_refs(
                    prop_type, types_block, where=f"{where}.properties[{prop_name!r}]"
                )
