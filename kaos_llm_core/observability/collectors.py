"""Trace collectors — contextvar-based child trace collection.

The base ``Program.__call__`` used to collect child traces by snapshotting
``last_trace.trace_id`` on each named sub-call before ``forward()`` and then
diffing after. That approach has two structural problems:

1. **Repeated invocations of the same sub-Call collapse to one entry.** The
   sub-Call's ``last_trace`` slot only holds the most recent execution; any
   earlier invocations in a loop are silently lost. A program that calls
   ``self.producer`` three times records exactly one child trace.

2. **Transient sub-Calls created inside ``forward()`` are invisible.** They
   are not in ``named_calls()`` so they cannot be enumerated by the snapshot
   pass. ``BestOfN`` is the canonical victim — its per-sample producer
   clones are constructed on the fly and discarded, so the parent trace
   has zero children and zero tokens.

The fix is a **push-based collector**: ``Program.__call__`` (and any other
caller that wants to capture children) opens a :class:`TraceCollector`
context for the duration of ``forward()``. Every successful ``Call._execute``
appends its trace into the active collector. The order of appended traces
matches actual execution order, repeated invocations are preserved as
distinct entries, and transient clones contribute traces too.

Collectors nest. Inner collectors capture their own children and the
*parent* collector receives whatever the inner collector chose to surface
(typically the parent ``ExecutionTrace`` the inner program built around its
own children). This is what lets a program-of-programs build a clean
two-level trace tree without each layer duplicating bookkeeping.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kaos_llm_core.observability.traces import ExecutionTrace


# Active collector for the current async task. ``None`` means no collector
# is active and ``Call._execute`` should not push (Python-only callers that
# do not wrap their work in a Program get the legacy behavior — the trace
# lives only on ``call.last_trace``).
_active_collector: contextvars.ContextVar[list[ExecutionTrace] | None] = contextvars.ContextVar(
    "kaos_llm_core_trace_collector",
    default=None,
)


@contextmanager
def collect_traces() -> Iterator[list[ExecutionTrace]]:
    """Open a fresh trace collector for the duration of the with-block.

    Yields the list that will receive traces. Nested ``collect_traces``
    contexts are independent — each captures only the work performed
    while it is the innermost active collector. The outer collector
    receives whatever the inner block decides to push back to it after
    closing (typically a single parent ``ExecutionTrace`` constructed
    from the inner children).

    Usage::

        with collect_traces() as children:
            await some_program.forward(...)
        # children now holds every Call.last_trace produced inside the block,
        # in execution order, with repeated invocations preserved.
    """
    collected: list[ExecutionTrace] = []
    token = _active_collector.set(collected)
    try:
        yield collected
    finally:
        _active_collector.reset(token)


def push_trace(trace: ExecutionTrace) -> None:
    """Append a trace to the active collector if one is in scope.

    Called by ``Call._execute`` after a successful execution and by
    program-level ``__call__`` overrides that want their own parent
    trace to surface to the surrounding collector.

    No-op when no collector is active. This preserves backward compat
    for callers that invoke a Call directly without wrapping it in a
    Program — ``call.last_trace`` still holds the result, the collector
    just isn't there to receive a copy.
    """
    bucket = _active_collector.get()
    if bucket is None:
        return
    bucket.append(trace)


def active_collector() -> list[ExecutionTrace] | None:
    """Return the active collector list, or ``None`` if no collector is in scope.

    Exposed for advanced callers (e.g., a custom Program that wants to peek
    at sibling traces emitted earlier in the same forward()). Most callers
    should use :func:`collect_traces` and ``push_trace`` instead.
    """
    return _active_collector.get()
