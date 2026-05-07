"""@llm_call decorator — syntactic sugar for simple LLM functions.

Builds a Signature from type hints, wraps in a Call, returns an async callable.

Example::

    @llm_call(model="anthropic:claude-sonnet-4-6")
    async def extract_entities(text: str) -> list[Entity]:
        '''Extract named entities from the text.'''
        ...

    # Use it
    result = await extract_entities(text="Acme Corp announced...")
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from kaos_llm_core.codecs.base import Codec
from kaos_llm_core.integrations.common.signatures import build_signature_from_function
from kaos_llm_core.programs.call import Call


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

        @functools.wraps(fn)
        async def wrapper(**call_inputs: Any) -> Any:
            return await call(**call_inputs)

        # Expose the underlying Call for introspection.
        # functools.wraps returns a function; ty cannot see dynamic attrs on it.
        wrapper._call = call  # ty: ignore[unresolved-attribute]
        wrapper._signature_class = sig_cls  # ty: ignore[unresolved-attribute]

        return wrapper

    return decorator


# Phase 14C: ``_function_to_signature`` and ``_is_pydantic_model``
# deleted. The function-introspection signature builder lives in
# :mod:`kaos_llm_core.integrations.common.signatures`
# (``build_signature_from_function``). The decorator imports it
# directly.
