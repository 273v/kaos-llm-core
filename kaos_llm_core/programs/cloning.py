"""clone_call — produce an isolated copy of a Call for loop iterations.

Phase 12A: extracts the deepcopy-with-shallow-fallback clone pattern that
:class:`BestOfN._clone_producer` and :func:`Refine.forward` were each
implementing. The shared utility:

- Tries ``copy.deepcopy`` first so mutable internals (``_kwargs``,
  ``examples`` list, codec instance, router state) are isolated from the
  original. The original ``BestOfN`` audit (#7) showed that a shallow
  copy shared the examples list and any router/scratch state, which
  races on concurrent samples.
- Falls back to ``copy.copy`` if deepcopy fails — some transport clients
  hold non-pickleable resources (sockets, locks). In that case the
  fallback still rebuilds the ``examples`` list so concurrent samples
  do not race on it, but it accepts that the codec/router may be
  shared.

There is no per-execution scratch to reset — Phase 10 moved everything
to the task-isolated Invocation ContextVar.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from kaos_core.logging import get_logger

if TYPE_CHECKING:
    from kaos_llm_core.programs.call import Call

logger = get_logger(__name__)

__all__ = ["clone_call"]


def clone_call(producer: Call, *, deep: bool = True) -> Call:
    """Return an isolated clone of ``producer`` suitable for loop iterations.

    Args:
        producer: The :class:`Call` to clone. Not mutated.
        deep: When ``True`` (default) attempt :func:`copy.deepcopy` first
            and fall back to :func:`copy.copy` on failure. Pass ``False``
            to skip the deepcopy attempt entirely (cheaper for callers
            that know their producer holds non-pickleable resources).

    Returns:
        A new :class:`Call` instance whose mutable internals are
        isolated from ``producer``. Both the deepcopy path and the
        shallow-fallback path guarantee isolation of the
        ``examples`` list AND the ``_kwargs`` dict so concurrent
        per-iteration mutations do not leak into the original.
        Other mutable shared state (codec instance, router state)
        is isolated only on the deepcopy path.

    Note:
        ``deep=True`` succeeding does **not** guarantee the cloned
        provider client is functionally usable — some clients hold
        sockets or locks that survive deepcopy structurally but
        produce broken behavior when invoked. Callers that know
        their client is not deepcopy-safe should pass
        ``deep=False`` and rely on the shallow path's explicit
        ``examples``/``_kwargs`` isolation.
    """
    if deep:
        try:
            return copy.deepcopy(producer)
        except Exception as exc:
            logger.debug(
                "clone_call: deepcopy of %s failed (%s); falling back to "
                "shallow copy. Per-execution scratch is already isolated "
                "via the Invocation contextvar; the fallback explicitly "
                "rebuilds ``examples`` and ``_kwargs`` but cannot rebuild "
                "the codec or router state.",
                type(producer).__name__,
                exc,
            )
    clone = copy.copy(producer)
    # Shallow path: rebuild mutable per-iteration containers so
    # concurrent samples do not race on them. ``examples`` and
    # ``_kwargs`` are the two slots that producer-cloning callers
    # (Refine, BestOfN) actually mutate.
    clone.examples = list(producer.examples)
    clone._kwargs = dict(producer._kwargs)
    return clone
