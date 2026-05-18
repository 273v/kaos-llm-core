"""@llm_call decorator — syntactic sugar for simple LLM functions.

Builds a Signature from type hints, wraps in a Call, returns an async callable.

Example::

    @llm_call(model="anthropic:claude-sonnet-4-6")
    async def extract_entities(text: str) -> list[Entity]:
        '''Extract named entities from the text.'''
        ...

    # Use it
    result = await extract_entities(text="Acme Corp announced...")

Single-field unwrap
-------------------

When the decorated function's return type produces a Signature with
**exactly one** OutputField (the common case for `-> str`, `-> int`,
`-> list[Foo]` etc.), the wrapper returns the value of that single
field directly. This honors the function's declared return-type
annotation: a function written as ``async def f(...) -> str`` returns
a ``str``, not a synthesized SignatureOutput wrapper.

Multi-field Signatures (the function is annotated to return a
multi-field Pydantic model, or `build_signature_from_function`
inferred multiple outputs) still return the SignatureOutput object so
callers can read each field by name. The behavior is decided once at
decoration time by inspecting the generated Signature class.

This unwrap closes the gap that bit the kaos-ui SPA's
``summarize_session_title`` Program: callers wrote
``(await fn(...)).strip()`` against the `-> str` annotation; the
decorator had been returning a generated ``fnSignatureOutput`` object
with no ``.strip`` attribute, silently breaking every auto-titler
call. See ``kaos-modules/docs/plans/persona-matrix-followups.md`` §5.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from kaos_llm_core.codecs.base import Codec
from kaos_llm_core.integrations.common.signatures import build_signature_from_function
from kaos_llm_core.programs.call import Call
from kaos_llm_core.signatures.introspection import get_output_fields


def llm_call(
    model: str | None = None,
    *,
    codec: Codec | None = None,
    max_retries: int | None = None,
    **kwargs: Any,
) -> Callable:
    """Decorator that turns a typed function into an LLM Call.

    The function's type hints become InputFields and OutputFields.
    The docstring becomes the instruction. The return type becomes the output.

    When the generated Signature has exactly one OutputField, the wrapper
    auto-unwraps the result and returns that field's value directly; the
    function's `-> ReturnType` annotation then reflects what callers
    actually receive. Multi-field Signatures return the full
    SignatureOutput object unchanged.

    Args:
        model: Model identifier (e.g., "anthropic:claude-sonnet-4-6").
        codec: Optional codec override.
        max_retries: Max validation retries.
        **kwargs: Extra params passed to the LLM client (temperature, etc.).

    Returns:
        A decorator that wraps the function.
    """

    def decorator(fn: Callable) -> Callable:
        # Phase 14C: route through the shared
        # ``integrations.common.signatures`` helper instead of the
        # in-line ``_function_to_signature`` implementation.
        sig_cls = build_signature_from_function(fn)

        # Create a Call with that Signature
        call = Call(
            sig_cls,
            model=model,
            codec=codec,
            max_retries=max_retries,
            **kwargs,
        )

        # Decide at decoration time whether to auto-unwrap the result.
        # ``get_output_fields`` raises SignatureError if the Signature
        # has no outputs (Call construction would have already failed
        # for that case, so this is purely defensive).
        output_field_names = tuple(get_output_fields(sig_cls).keys())
        _unwrap_field: str | None = output_field_names[0] if len(output_field_names) == 1 else None

        @functools.wraps(fn)
        async def wrapper(**call_inputs: Any) -> Any:
            result = await call(**call_inputs)
            if _unwrap_field is None:
                return result
            # Single-output signature: return the field value directly
            # so the function's `-> T` annotation is honest.
            return getattr(result, _unwrap_field)

        # Expose the underlying Call for introspection.
        # functools.wraps returns a function; ty cannot see dynamic attrs on it.
        wrapper._call = call  # ty: ignore[unresolved-attribute]
        wrapper._signature_class = sig_cls  # ty: ignore[unresolved-attribute]
        # ``None`` when the signature is multi-output. Useful for tests
        # and introspection — never read by the runtime path.
        wrapper._unwrap_field = _unwrap_field  # ty: ignore[unresolved-attribute]

        return wrapper

    return decorator


# Phase 14C: ``_function_to_signature`` and ``_is_pydantic_model``
# deleted. The function-introspection signature builder lives in
# :mod:`kaos_llm_core.integrations.common.signatures`
# (``build_signature_from_function``). The decorator imports it
# directly.
