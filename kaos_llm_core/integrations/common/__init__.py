"""Shared helpers consumed by starter, the decorator, and MCP tool wrappers.

Phase 14A: every place in the package that builds a dynamic
:class:`Signature` from a dict of fields, a function's type hints, or
a Pydantic model now goes through :func:`signatures.build_signature`.
The previous arrangement had three near-identical implementations
(``starter._make_signature``, ``decorator._function_to_signature``, and
the per-tool ad-hoc constructors inside ``tools.py``) — moving them
into one helper kills the duplication and gives test coverage one
target.
"""

from kaos_llm_core.integrations.common.signatures import (
    build_signature,
    build_signature_from_function,
)

__all__ = ["build_signature", "build_signature_from_function"]
