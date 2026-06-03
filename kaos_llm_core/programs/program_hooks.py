"""ProgramHooks — lifecycle hooks for composed programs (ReAct/Refine/BestOfN).

Counterpart to :class:`kaos_llm_core.programs.hooks.CallHooks`. ``CallHooks``
fires around a single ``Call._execute`` (one logical LLM invocation with
optional validation retries). ``ProgramHooks`` fires around a *composed*
program — a ReAct loop, a Refine cycle, a BestOfN sample fan-out — that
makes multiple Call invocations and needs its own observer surface.

Why a separate hook type instead of reusing CallHooks?

1. ``CallHooks.on_call_start/on_call_end`` are scoped to a *logical Call*.
   A ReAct execution is **not** a single logical Call: it owns 1..N turns
   each of which may be a Call-shaped operation. Reusing CallHooks would
   force users to write callbacks that branch on "is this an iteration or
   the whole program?" which is clumsy.

2. Programs have iteration semantics that ``CallHooks`` does not capture:
   ReAct has tool calls + observations per turn, Refine has scores +
   feedback per iteration, BestOfN has parallel samples. ``ProgramHooks``
   exposes ``on_iteration`` that programs invoke with a structured payload
   so observers can record per-iteration cost / score / latency without
   reverse-engineering the trace tree.

3. Lifecycle hooks for a *Program* still want a final ``on_program_end``
   with the assembled parent ``ExecutionTrace`` so a single observer can
   build a per-run telemetry record without inspecting state on the
   Program instance.

Hook contract:

- All hooks are optional. A hook that raises is logged at WARNING and
  swallowed — exactly the same policy as ``CallHooks``. Observability
  must never affect control flow.
- Hooks always receive a ``context=`` keyword argument carrying an
  optional :class:`KaosContext` (set by the MCP tool wrapper or any
  caller that wants session/trace correlation). ``None`` is allowed.
- Backward-compat fallback: callbacks that do not accept ``context``
  are detected by :func:`fire_hook` and invoked without the kwarg.

Programs that fire ProgramHooks:

- :class:`~kaos_llm_core.programs.react.ReAct` — once around the loop
  plus ``on_iteration`` per turn.
- :class:`~kaos_llm_core.programs.refine.Refine` — once around the
  refine cycle plus ``on_iteration`` per produce-and-judge round.
- :class:`~kaos_llm_core.programs.best_of_n.BestOfN` — once around the
  fan-out and selection.

CallHooks attached to the *inner* Call/producer/judge of any of these
programs continue to fire normally — those hooks fire at the
``Call._execute`` boundary and ProgramHooks fires at the program
boundary. The two layers compose without conflict.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from kaos_core.logging import get_logger

from kaos_llm_core.programs.hooks import fire_hook

logger = get_logger(__name__)

__all__ = ["ProgramHook", "ProgramHooks", "fire_program_hook"]


ProgramHook = Callable[..., Any]
"""A program-lifecycle callback. Signature varies by which slot it fills."""


@dataclass
class ProgramHooks:
    """Optional lifecycle callbacks fired around composed programs.

    A hook that raises is caught, logged at WARNING, and ignored — hooks
    never affect program control flow.

    Hook signatures:

    - ``on_program_start(program, inputs, *, context=None)`` — fires once
      at the top of ``program.__call__``, before any iteration runs.
    - ``on_iteration(program, iteration, payload, *, context=None)`` —
      fires per iteration. ``iteration`` is 0-indexed. ``payload`` is a
      program-specific dict — see each program's docstring for the keys.
    - ``on_tool_start(program, tool_call, *, context=None)`` — fires
      immediately *before* a single tool call is dispatched, once per
      tool call, with the :class:`~kaos_llm_core.programs.tool.ToolCall`
      about to run (``.id`` / ``.name`` / ``.arguments``). This is a
      finer grain than ``on_iteration`` (which fires once per loop turn,
      after every tool in that turn has already executed): it lets an
      observer surface "tool X is running now" *while* the call is in
      flight, which is what a live UI needs. Only ReAct fires it.
    - ``on_tool_end(program, observation, *, context=None)`` — fires
      immediately *after* each tool call returns, once per tool call,
      with the :class:`~kaos_llm_core.programs.react.ToolObservation`
      (carrying ``tool_call_id`` / ``tool_name`` / ``arguments`` /
      ``result`` / ``is_error``). Pairs with ``on_tool_start`` to bracket
      one tool's execution. Only ReAct fires it.
    - ``on_program_end(program, inputs, invocation, *, context=None)``
      — fires once after the program completes successfully. ``trace`` is
      the assembled parent :class:`ExecutionTrace`.
    - ``on_program_error(program, inputs, exception, *, context=None)``
      — fires once if the program raises. The exception is re-raised
      after the hook returns.
    """

    on_program_start: ProgramHook | None = None
    on_iteration: ProgramHook | None = None
    on_tool_start: ProgramHook | None = None
    on_tool_end: ProgramHook | None = None
    on_program_end: ProgramHook | None = None
    on_program_error: ProgramHook | None = None


def fire_program_hook(
    hook: ProgramHook | None,
    *args: Any,
    context: Any = None,
    **kwargs: Any,
) -> None:
    """Invoke a ProgramHook callback, swallowing exceptions.

    Thin re-export of :func:`kaos_llm_core.programs.hooks.fire_hook` so
    callers can read program code without an extra import. The retry-on-
    TypeError fallback for context-unaware callbacks is identical to
    ``CallHooks``.
    """
    fire_hook(hook, *args, context=context, **kwargs)
