"""build_signature — single source of truth for dynamic Signature construction.

Phase 14A: replaces three near-identical implementations that lived
inside ``starter._make_signature``, ``decorator._function_to_signature``,
and a handful of ad-hoc constructors inside ``tools.py``. Every caller
that needs to build a Signature subclass at runtime — from a dict of
fields, a Python function's type hints, or a Pydantic model — now
calls one of the helpers exported here.

Two entry points:

- :func:`build_signature` — the dict-of-fields path. Used by the
  starter API (``text``, ``extract``, ``classify``, ``summarize``)
  and by MCP tool wrappers that build a Signature from JSON-like
  parameter specs.
- :func:`build_signature_from_function` — the function-introspection
  path. Used by the ``@llm_call`` decorator. Walks ``get_type_hints``,
  separates input parameters from the return type, and treats
  Pydantic return types as flat output fields.

Both functions return a :class:`Signature` subclass that the rest of
the package consumes through the standard ``Call(signature, ...)``
constructor — there is no special handoff for dynamically-built
Signatures.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, get_type_hints

from pydantic import BaseModel, create_model

from kaos_llm_core.signatures.fields import InputField, OutputField
from kaos_llm_core.signatures.signature import Signature

__all__ = ["build_signature", "build_signature_from_function"]


def build_signature(
    name: str,
    fields: dict[str, tuple[Any, Any]],
    doc: str,
) -> type[Signature]:
    """Build a :class:`Signature` subclass from a dict of fields.

    ``fields`` maps field names to ``(annotation, FieldInfo)`` tuples
    where ``FieldInfo`` is one of :class:`InputField` /
    :class:`OutputField`. ``doc`` becomes the class docstring (which
    the codecs read as the system instruction).

    The dynamic ``Signature`` is built via ``pydantic.create_model``
    with ``__base__=Signature``. ``ty`` cannot fully narrow the
    overload set so the ignore directive stays — the call shape is
    correct at runtime per the pydantic API.
    """
    sig_cls: Any = create_model(  # ty: ignore[no-matching-overload]
        name,
        __base__=Signature,
        **fields,
    )
    sig_cls.__doc__ = doc
    return sig_cls


def build_signature_from_function(fn: Callable[..., Any]) -> type[Signature]:
    """Build a :class:`Signature` subclass from a Python function's hints.

    Parameters become :class:`InputField` slots; the return type
    becomes either a single ``result`` :class:`OutputField` or — when
    the return type is a Pydantic ``BaseModel`` subclass — one
    :class:`OutputField` per model field. The function's docstring
    becomes the Signature's docstring (which codecs read as the
    instruction).

    Used by the ``@llm_call`` decorator. Phase 14C deletes the
    in-line ``decorator._function_to_signature`` implementation and
    routes through this helper instead.
    """
    hints = get_type_hints(fn)
    doc = inspect.getdoc(fn) or ""
    fn_name: str = getattr(fn, "__name__", "anonymous")

    return_type = hints.pop("return", None)
    sig = inspect.signature(fn)

    field_defs: dict[str, Any] = {}
    annotations: dict[str, Any] = {}

    # Input fields from function parameters
    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        ann = hints.get(param_name, str)
        annotations[param_name] = ann
        if param.default is not inspect.Parameter.empty:
            field_defs[param_name] = InputField(default=param.default, description=param_name)
        else:
            field_defs[param_name] = InputField(description=param_name)

    # Output fields from return type
    if return_type is not None and _is_pydantic_model(return_type):
        for name, field_info in return_type.model_fields.items():
            annotations[name] = field_info.annotation
            desc = field_info.description or name
            if field_info.is_required():
                field_defs[name] = OutputField(description=desc)
            else:
                field_defs[name] = OutputField(default=field_info.default, description=desc)
    elif return_type is not None:
        annotations["result"] = return_type
        field_defs["result"] = OutputField(description="The result")

    namespace = {"__annotations__": annotations, "__doc__": doc, **field_defs}
    return type(fn_name + "Signature", (Signature,), namespace)


def _is_pydantic_model(tp: Any) -> bool:
    """Check if a type is a Pydantic ``BaseModel`` subclass."""
    try:
        return isinstance(tp, type) and issubclass(tp, BaseModel)
    except TypeError:
        return False
