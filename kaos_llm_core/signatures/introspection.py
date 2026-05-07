"""Signature introspection utilities.

Functions to inspect a Signature class and extract its input/output fields,
instruction text, JSON Schema, and dynamic output Pydantic model.
"""

from __future__ import annotations

import textwrap
from typing import Any

from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo

from kaos_llm_core.errors import SignatureError
from kaos_llm_core.signatures.fields import is_input_field, is_output_field
from kaos_llm_core.signatures.signature import Signature


def get_input_fields(sig: type[Signature]) -> dict[str, FieldInfo]:
    """Return all InputField entries from a Signature class.

    Args:
        sig: A Signature subclass.

    Returns:
        Dict mapping field name to FieldInfo for all input fields.
    """
    return {name: info for name, info in sig.model_fields.items() if is_input_field(info)}


def get_output_fields(sig: type[Signature]) -> dict[str, FieldInfo]:
    """Return all OutputField entries from a Signature class.

    Args:
        sig: A Signature subclass.

    Returns:
        Dict mapping field name to FieldInfo for all output fields.

    Raises:
        SignatureError: If the Signature has no output fields.
    """
    outputs = {name: info for name, info in sig.model_fields.items() if is_output_field(info)}
    if not outputs:
        raise SignatureError(
            f"Signature {sig.__name__} has no OutputFields. "
            "Add at least one field with OutputField(). "
            "Example: entities: list[str] = OutputField(description='Extracted entities')"
        )
    return outputs


def get_instruction(sig: type[Signature]) -> str:
    """Extract the instruction from a Signature's docstring.

    Args:
        sig: A Signature subclass.

    Returns:
        The cleaned docstring text, or an empty string if no docstring.
    """
    doc = sig.__doc__
    if not doc:
        return ""
    return textwrap.dedent(doc).strip()


def signature_to_json_schema(sig: type[Signature]) -> dict[str, Any]:
    """Generate a JSON Schema for the output fields of a Signature.

    This schema is used by JSONCodec to instruct the LLM on the expected
    output format, and for response validation.

    Args:
        sig: A Signature subclass.

    Returns:
        A JSON Schema dict describing the output fields.
    """
    output_model = create_output_model(sig)
    return output_model.model_json_schema()


def create_output_model(sig: type[Signature]) -> type[BaseModel]:
    """Create a dynamic Pydantic model from a Signature's output fields.

    The resulting model can be used for:
    - Passing to ``kaos-llm-client``'s ``pydantic_async(output_type=...)``
    - Validating decoded responses
    - Generating JSON Schema

    Args:
        sig: A Signature subclass.

    Returns:
        A Pydantic BaseModel subclass with the output fields.
    """
    output_fields = get_output_fields(sig)

    # Build field definitions for create_model: (type, FieldInfo)
    field_defs: dict[str, Any] = {}
    for name, field_info in output_fields.items():
        field_defs[name] = (field_info.annotation, field_info)

    model_name = f"{sig.__name__}Output"
    return create_model(model_name, **field_defs)
