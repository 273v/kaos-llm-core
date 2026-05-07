"""Build a Signature class from one envelope step's declared inputs/outputs.

Phase 15.1 supports the common JSON-schema-ish type cases. Complex
features (anyOf / oneOf / allOf) fall back to ``Any``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kaos_llm_core.integrations.common.signatures import build_signature
from kaos_llm_core.programs.envelope.models import EnvelopeStep
from kaos_llm_core.signatures.fields import InputField, OutputField

if TYPE_CHECKING:
    from kaos_llm_core.signatures.signature import Signature


def _python_type_for_field(
    type_spec: dict[str, Any],
    types_block: dict[str, dict[str, Any]],
) -> Any:
    """Map a JSON-schema-ish type spec to a Python annotation.

    Phase 15.1 supports the common cases: ``str``, ``int``, ``float``,
    ``bool``, ``list[X]``, ``dict[str, Any]``, plus ``$ref`` into
    the hoisted types block (resolved to a recursive call). Complex
    JSON Schema features (``anyOf``, ``oneOf``, ``allOf``) are out
    of scope for v3; the executor falls back to ``Any`` for unknown
    shapes.
    """
    if "$ref" in type_spec:
        ref = type_spec["$ref"]
        type_name = ref[len("#/types/") :]
        return _python_type_for_field(types_block[type_name], types_block)
    if "array" in type_spec and isinstance(type_spec["array"], dict):
        item_type = _python_type_for_field(type_spec["array"], types_block)
        return list[item_type]  # type: ignore[valid-type]
    json_type = type_spec.get("type")
    if json_type == "string":
        if "enum" in type_spec:
            return str  # Phase 15.1 represents enums as plain strings.
        return str
    if json_type == "integer":
        return int
    if json_type == "number":
        return float
    if json_type == "boolean":
        return bool
    if json_type == "array":
        if "items" in type_spec and isinstance(type_spec["items"], dict):
            item_type = _python_type_for_field(type_spec["items"], types_block)
            return list[item_type]  # type: ignore[valid-type]
        return list[Any]
    if json_type == "object":
        return dict[str, Any]
    return Any  # type: ignore[return-value]


def _build_signature_for_step(
    step: EnvelopeStep,
    types_block: dict[str, dict[str, Any]],
) -> type[Signature]:
    """Build a Pydantic Signature class for one envelope step."""
    fields: dict[str, tuple[Any, Any]] = {}
    # Inputs to the underlying Call are the keys of step.inputs (the field
    # names declared in the envelope, post-resolution).
    for field_name in step.inputs:
        fields[field_name] = (Any, InputField(description=field_name))
    # Outputs are step.output_fields, with their declared types.
    for field_name, output_field in step.output_fields.items():
        py_type = _python_type_for_field(output_field.type, types_block)
        fields[field_name] = (py_type, OutputField(description=output_field.description))

    sig_name = f"Step_{step.id}_Sig"
    return build_signature(sig_name, fields, step.instruction)
