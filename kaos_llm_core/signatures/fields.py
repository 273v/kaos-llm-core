"""InputField and OutputField descriptors for Signatures.

These are thin wrappers around ``pydantic.Field`` that carry a ``kind`` marker
(``"input"`` or ``"output"``) in the field's ``json_schema_extra`` metadata.
Codecs and introspection utilities use this marker to distinguish inputs from outputs.
"""

from __future__ import annotations

from typing import Any

from pydantic.fields import FieldInfo

_FIELD_KIND_KEY = "kaos_field_kind"


def InputField(
    default: Any = ...,
    *,
    description: str = "",
    examples: list[Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Declare an input field on a Signature.

    Args:
        default: Default value. Use ``...`` (Ellipsis) for required fields.
        description: Human-readable description used in prompt formatting.
        examples: Example values for documentation and codec formatting.
        **kwargs: Additional Pydantic Field arguments (e.g., ``ge``, ``le``).

    Returns:
        A Pydantic FieldInfo with ``kaos_field_kind="input"`` in json_schema_extra.
    """
    extra = kwargs.pop("json_schema_extra", None) or {}
    extra[_FIELD_KIND_KEY] = "input"
    if examples is not None:
        extra["examples"] = examples
    from pydantic import Field

    return Field(default, description=description, json_schema_extra=extra, **kwargs)


def OutputField(
    default: Any = ...,
    *,
    description: str = "",
    examples: list[Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Declare an output field on a Signature.

    Args:
        default: Default value. Use ``...`` (Ellipsis) for required fields.
        description: Human-readable description used in prompt formatting.
        examples: Example values for documentation and codec formatting.
        **kwargs: Additional Pydantic Field arguments (e.g., ``json_schema_extra``).

    Returns:
        A Pydantic FieldInfo with ``kaos_field_kind="output"`` in json_schema_extra.
    """
    extra = kwargs.pop("json_schema_extra", None) or {}
    extra[_FIELD_KIND_KEY] = "output"
    if examples is not None:
        extra["examples"] = examples
    from pydantic import Field

    return Field(default, description=description, json_schema_extra=extra, **kwargs)


def _get_field_kind(field_info: FieldInfo) -> str | None:
    """Extract the kaos field kind from a FieldInfo's json_schema_extra.

    Pydantic's ``json_schema_extra`` has a complex union type that ty cannot
    fully narrow through ``isinstance(extra, dict)``. We access ``.get()``
    after the isinstance guard, which is safe at runtime.
    """
    extra = field_info.json_schema_extra
    if not isinstance(extra, dict):
        return None
    return extra.get(_FIELD_KIND_KEY)  # ty: ignore[invalid-argument-type]


def is_input_field(field_info: FieldInfo) -> bool:
    """Return True if this field is an InputField."""
    return _get_field_kind(field_info) == "input"


def is_output_field(field_info: FieldInfo) -> bool:
    """Return True if this field is an OutputField."""
    return _get_field_kind(field_info) == "output"
