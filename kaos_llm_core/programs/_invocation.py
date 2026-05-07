"""Invocation — the canonical runtime contract for one Call execution.

The :class:`Invocation` is the single source of truth for everything that
happens during one ``Call.invoke()`` (or ``Program.invoke()``): the resolved
provider client and model, the per-execution scratch a subclass might need
(``ChainOfThought``'s ``native_thinking`` flag, future subclass extras),
the cumulative ExecutionTrace, the token+cost summary, the final output,
the error if one occurred, and the per-task ``KaosContext``.

It replaces every legacy carrier the package had before:

- ``Call._last_trace`` instance attribute
- ``Program._last_trace`` instance attribute
- ``object.__setattr__(result, "_last_trace", ...)`` mutation pattern
- ``Call._current_client`` / ``Call._current_model`` instance scratch
- ``ChainOfThought._native_thinking`` instance flag
- ``_ExecState`` dataclass + ``_exec_state_var`` ContextVar
- the ``prediction._last_trace`` legacy fallback in ``estimate_eval_cost``

One type, one lifecycle:

1. **Construction** — top of ``Call._run_pipeline``. The Invocation is built
   with the resolved ``client``, ``model``, freshly-allocated
   ``ExecutionTrace``, and any subclass extras populated by
   :meth:`Call._build_invocation_extras`. The new Invocation is set as the
   active value of :data:`_active_invocation_var` for the duration of the
   pipeline.

2. **Mutation during execution** — step methods may read the active
   Invocation via :func:`current_invocation` to look up the resolved
   client / model / extras. Token accumulators on ``invocation.usage``
   and trace fields on ``invocation.trace`` are updated in place across
   the validation-retry loop.

3. **Finalization** — bottom of ``_run_pipeline``. On success, ``output``
   is assigned. On exception, ``error`` is assigned, the exception is
   tagged with ``exc.invocation = invocation`` so the caller can retrieve
   the partial Invocation, and the exception is re-raised. ``trace.error``
   and cost estimates are populated either way.

4. **Consumption** — ``Call.invoke()`` returns the Invocation directly.
   ``Call.__call__()`` unwraps to ``invocation.output`` for the ergonomic
   bare-output return. The eval harness reads ``invocation.trace`` and
   ``invocation.usage`` directly. Optimizers consume Invocations via
   the TrialRunner sink (Phase 13).

ContextVar isolation: :data:`_active_invocation_var` is task-isolated, so
two parallel ``await call(...)`` invocations on the same Call instance
each see their own Invocation. There is no instance-level scratch, so
there is nothing to race on.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from kaos_llm_core.observability.traces import ExecutionTrace


@dataclass(slots=True)
class TokenUsage:
    """Per-Invocation token + cost summary.

    Mutable during pipeline execution (the validation-retry loop
    accumulates tokens across attempts) and effectively frozen after
    Invocation finalization. Consumers (eval harness, budget tracker,
    cost reports) read directly from this object instead of walking
    the trace tree, which is O(1) versus O(children).
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


@dataclass(slots=True)
class Invocation:
    """Complete record of one ``Call.invoke()`` execution.

    Constructed at the top of ``Call._run_pipeline``, mutated through the
    pipeline (token accumulators, trace fields), finalized at the bottom
    by setting ``output`` (success) or ``error`` (failure). Returned by
    ``Call.invoke()`` and ``Program.invoke()``; unwrapped to ``output``
    by ``Call.__call__`` and ``Program.__call__``.

    Step methods that need the resolved ``client`` / ``model`` / subclass
    ``extras`` read them from the active Invocation via
    :func:`current_invocation`. This replaces the Phase 9c
    ``_ExecState`` ContextVar with one type that owns both the in-flight
    scratch and the final result record.

    Attributes:
        id: A short hex UUID for deduplication. Used by the Phase 13
            ``TrialRunner`` sink to ensure the same Invocation is not
            charged to a trial twice (e.g. once via the publication
            sink and again via ``ExampleResult.invocation``).
        client: The resolved ``BaseProviderClient`` (typed Any to avoid
            an import cycle here). Set at construction.
        model: The provider-prefixed model id (e.g.
            ``"anthropic:claude-haiku-4-5"``). Set at construction.
        context: The per-task ``KaosContext`` snapshot from
            ``_call_context_var.get()``. Set at construction so hook
            callbacks that receive an Invocation can correlate by
            session/trace id without re-reading the contextvar.
        extras: Subclass-specific scratch (``ChainOfThought`` populates
            ``"native_thinking"``, future subclasses add their own keys).
            Populated at construction by
            :meth:`Call._build_invocation_extras`.
        output: The validated, post-processed result. ``None`` until
            success is reached or on the error path. The bare-output
            return value of ``Call.__call__`` is exactly this field.
        trace: The cumulative ``ExecutionTrace`` for this Invocation.
            ``None`` only when ``trace_enabled=False``. Mutated in place
            during the retry loop; finalized with cost/latency/error
            before the Invocation is returned.
        usage: Per-invocation token + cost summary. Accumulated across
            retry attempts. Mirrors ``trace.input_tokens`` etc but is
            cheaper to read for budget consumers.
        error: The exception that ended this Invocation (if any). On
            failure the same exception is also tagged with
            ``exc.invocation = self`` and re-raised; ``error`` is set
            so the partial Invocation remains useful for telemetry.
    """

    id: str = field(default_factory=lambda: uuid4().hex[:16])
    client: Any = None
    model: str = ""
    context: Any = None
    extras: dict[str, Any] = field(default_factory=dict)
    output: Any = None
    trace: ExecutionTrace | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    error: BaseException | None = None


# Per-task active Invocation. Set at the top of ``Call._run_pipeline``,
# reset at the bottom. Step methods read it via :func:`current_invocation`
# to access the resolved client / model / extras without depending on
# instance state. ContextVar is task-isolated, so concurrent invocations
# on the same Call see their own Invocation.
_active_invocation_var: contextvars.ContextVar[Invocation | None] = contextvars.ContextVar(
    "kaos_llm_core_active_invocation",
    default=None,
)


def current_invocation() -> Invocation | None:
    """Return the active Invocation for the current task.

    Returns ``None`` outside an active ``Call._run_pipeline`` execution.
    Step methods that need the resolved client / model / extras call
    this and treat ``None`` as "not in execution" (a defensive default
    used only if a step method is invoked directly by user code).
    """
    return _active_invocation_var.get()
