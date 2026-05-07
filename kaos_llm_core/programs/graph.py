"""ProgramGraph — explicit child registry for composed programs.

Phase 11: Programs no longer discover their children by walking
``vars(self)``. Instead, :class:`Program.__setattr__` auto-registers any
public ``Call`` or sub-:class:`Program` assigned as an instance
attribute into a per-instance ``_child_registry`` dict. The registry is
the single source of truth for ``named_calls()``, ``get_learnable_state``,
and the optimizer ``primary_call`` heuristic.

Why an explicit registry instead of ``vars(self)`` reflection?

1. **Determinism.** Two programs with the same children declared in
   different order used to optimize against different "primary" Calls
   because the previous heuristic walked ``vars(self)`` (insertion
   order) and stopped at the first one. The registry preserves
   declaration order explicitly so the rule is "the first Call you
   assigned wins" — visible in code, not implicit in dict iteration.

2. **No private fallback shim.** Programs that don't auto-register
   children simply have no children. No silent fallback through
   ``vars(self)``. If a program stores a Call under a private name
   (``self._inner_call``) it must explicitly alias the value to a
   public name (``self.call = self._inner_call``) so the registry
   sees it. This makes the optimizer-target story visible at the
   __init__ site instead of hidden behind a name-mangling rule.

3. **Composability.** Programs that store children in a list (e.g.
   :class:`Ensemble`'s voters) override :meth:`Program.named_calls`
   to add the list members. Auto-registration handles the common
   case; the override stays for the structural exception.

The graph object itself is a thin wrapper around the registry — it
exists so optimizer code reads ``program.graph.primary()`` instead of
poking at ``_child_registry`` directly. Future expansion (e.g. typed
edges, dependency walks) lives on this class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kaos_llm_core.programs.base import Program
    from kaos_llm_core.programs.call import Call

__all__ = ["ProgramGraph"]


class ProgramGraph:
    """Read-only view over a Program's child registry.

    Constructed lazily via :attr:`Program.graph`. Wraps the per-instance
    ``_child_registry`` dict. The wrapper exists so callers don't reach
    into the private dict; future graph operations (typed edges,
    optimizer targeting) live on this class.
    """

    __slots__ = ("_owner", "_registry")

    def __init__(self, owner: Program, registry: dict[str, Call | Program]) -> None:
        self._owner = owner
        self._registry = registry

    def children(self) -> dict[str, Call | Program]:
        """Return a snapshot of all registered children, in declaration order."""
        return dict(self._registry)

    def primary(self) -> Call:
        """Return the first registered :class:`Call` in declaration order.

        Used by the codec / model / hyperparameter optimizers to pick
        the Call they will mutate when the user passes a Program.
        Skips sub-:class:`Program` instances and walks the registry
        in declaration order — so a Program declaring ``self.zeta``
        before ``self.alpha`` is optimized against ``zeta``, which
        matches the user's intent (``zeta`` is the work-horse).

        Raises :class:`TypeError` when no registered child is a Call.
        Programs whose children are all sub-Programs need to be
        optimized through their inner Calls explicitly.
        """
        # Local import to break the import cycle (graph -> call -> base -> graph).
        from kaos_llm_core.programs.call import Call as _Call

        for value in self._registry.values():
            if isinstance(value, _Call):
                return value
        raise TypeError(
            f"ProgramGraph.primary(): no Call is registered on "
            f"{type(self._owner).__name__}. Optimizers that target a 'primary' "
            "Call require at least one Call attribute. Either pass a Call "
            "directly to the optimizer, or assign your Call as a public "
            "attribute (``self.extract = Call(...)``) so it auto-registers."
        )
