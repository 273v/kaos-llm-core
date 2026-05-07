"""Shared attribute-forwarding mixin + ``HasOutputs`` typing Protocol.

Two related surfaces:

  - ``OutputForwardingMixin`` — runtime behavior. The composed-program
    result types (``ReActResult`` / ``RefineResult`` / ``BestOfNResult``
    / ``MultiChainComparisonResult`` / ``ProgramOfThoughtResult``)
    inherit this to get ``result.field_name`` forwarding to whatever
    is in ``self.outputs`` (a dict or a Pydantic model). The mixin
    has no fields, no constructor, just the ``__getattr__`` method —
    safe to mix into ``@dataclass(slots=True)`` subclasses.
  - ``HasOutputs`` — a ``runtime_checkable`` ``typing.Protocol`` that
    formalizes the "this object has an ``outputs`` attribute" contract
    so ty (and downstream callers) can ``isinstance``-narrow against
    it without depending on the mixin specifically. Phase 16.5 fix:
    the envelope executor's result-unwrap path used to call
    ``hasattr(output, "outputs")``, which gives ty no information.
    Switching to ``isinstance(output, HasOutputs)`` makes the unwrap
    type-checked AND catches future result types that forget to
    declare an ``outputs`` slot.

History: before this module the three original result types each
implemented ``__getattr__`` differently — ReActResult only handled
the dict case, RefineResult only handled the object case, BestOfNResult
handled both. This mixin uses the BestOfN-style permissive lookup as
the canonical behavior.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = ["HasOutputs", "OutputForwardingMixin"]


@runtime_checkable
class HasOutputs(Protocol):
    """A typing Protocol for any result wrapper that exposes an ``outputs``.

    The envelope executor's normalization in
    :meth:`_EnvelopeProgram.forward` uses ``isinstance(value, HasOutputs)``
    to detect result wrappers (BestOfNResult, MultiChainComparisonResult,
    ProgramOfThoughtResult, and any future analog). Type-checked, so a
    new result wrapper that forgets to declare an ``outputs`` slot is
    caught at definition time rather than at runtime.

    The Protocol is ``runtime_checkable`` so it works at runtime as
    well as in static type narrowing.
    """

    outputs: Any


class OutputForwardingMixin:
    """Forward unknown attribute access to the wrapped ``outputs`` field.

    Subclasses must declare an ``outputs`` field via dataclass / slots.
    The forwarding rules:

    1. Concrete dataclass fields take precedence (Python checks the slot
       descriptors before falling through to ``__getattr__``).
    2. If ``outputs`` is a dict and the requested name is a key, return
       the value. Dicts take precedence over objects so a Pydantic model
       containing a key named ``"keys"`` does not collide with the
       ``dict.keys`` method.
    3. If ``outputs`` is an object with the requested attribute, return
       that attribute. ``getattr`` is used because it correctly handles
       Pydantic models, dataclasses, and plain ``__init__`` instances.
    4. Otherwise raise ``AttributeError`` with the requested name. The
       error message is intentionally bare so it matches Python's
       built-in ``AttributeError`` shape — no extra context that would
       confuse downstream try/except logic.
    """

    # Subclasses (dataclasses with slots) declare ``outputs`` themselves.
    # The annotation here is informational only — there is no shared slot
    # because mixins cannot extend a slotted dataclass's __slots__ tuple.
    outputs: Any

    def __getattr__(self, name: str) -> Any:
        # ``__getattr__`` is only called when normal lookup fails, so the
        # dataclass slots above are reached first. Use object.__getattribute__
        # to avoid triggering this same __getattr__ recursively if ``outputs``
        # itself is not yet set (e.g. during dataclass __init__).
        try:
            outputs = object.__getattribute__(self, "outputs")
        except AttributeError as exc:
            raise AttributeError(name) from exc
        if isinstance(outputs, dict):
            if name in outputs:
                return outputs[name]
            raise AttributeError(name)
        if outputs is None:
            raise AttributeError(name)
        # Object path: defer to standard attribute lookup. ``getattr`` raises
        # AttributeError on its own if the attribute does not exist, which is
        # exactly the contract __getattr__ wants.
        return getattr(outputs, name)
