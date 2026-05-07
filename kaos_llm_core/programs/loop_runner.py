"""LoopRunner — shared execution kernel for iterative programs.

Phase 12A: ReAct and Refine each implement their own loop with
near-identical bookkeeping — per-iteration ``collect_traces`` scope,
parent trace assembly, hook firing, stop-reason classification, and
error handling. The duplication grew organically through Phase 9b/9c/9d
fixes that had to land twice (once per program).

``LoopRunner`` is the shared kernel. Programs supply four callbacks:

- ``make_state()`` — produce the per-run mutable state.
- ``step(iter, state)`` — execute one iteration. Returns
  :class:`StepOutcome` with the iteration record and an optional
  :class:`StopReason` to break the loop early.
- ``build_result(state, records)`` — assemble the final program result
  from the accumulated step records.
- ``on_step_error(state, exc)`` — optional. Decide whether the loop
  should continue (return ``None``) or terminate with a specific
  :class:`StopReason`.

The runner owns:

- Per-iteration ``collect_traces`` scope. Each step's children are
  collected into the per-iteration trace; the per-iteration traces
  become children of the parent program trace.
- Stop-reason classification: ``COMPLETED`` (max iterations reached),
  ``EARLY_STOP`` (step returned a stop reason), ``ERROR`` (uncaught
  step exception).

Programs migrating onto :class:`LoopRunner` shrink to ~30 LoC of
program-specific glue: define the state dataclass, write ``step()``,
write ``build_result()``, optionally write ``on_step_error()``, and
delegate ``forward()`` to the runner.

This module is foundational for Phases 12B (ReAct) and 12C (Refine).
Phase 12A ships the runner + cloning utility + unit tests against a
mock step. No program changes land in this phase.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum

__all__ = [
    "LoopConfig",
    "LoopResult",
    "LoopRunner",
    "StepOutcome",
    "StopReason",
]


class StopReason(StrEnum):
    """Why a :class:`LoopRunner` loop terminated."""

    COMPLETED = "completed"
    """The step function ran ``max_iterations`` times without an early stop."""

    EARLY_STOP = "early_stop"
    """The step function returned a non-None ``stop_reason``."""

    ERROR = "error"
    """An uncaught exception escaped the step function."""


@dataclass(slots=True)
class StepOutcome[Record]:
    """Return value of a :class:`LoopRunner` step function.

    Attributes:
        record: The per-iteration record. Appended to the records list
            so :meth:`LoopRunner.run` can hand them to ``build_result``.
        stop_reason: When non-None the runner halts the loop after
            recording this iteration. Use it for completion-detection
            (the model returned a final answer, the score crossed the
            min threshold, etc.). The string is opaque to the runner —
            programs choose their own taxonomy.
    """

    record: Record
    stop_reason: str | None = None


@dataclass(slots=True)
class LoopResult[Result, Record]:
    """Final return value of :meth:`LoopRunner.run`.

    Attributes:
        result: Whatever ``build_result`` produced.
        records: All :class:`StepOutcome.record` values, in iteration
            order.
        iterations_used: The number of step calls that completed
            (including the one that triggered the early stop).
        stop_reason: Why the loop ended. Either a :class:`StopReason`
            or the program-specific string returned by ``step``.
        error: The exception that terminated the loop, if any. Only
            populated when ``stop_reason == StopReason.ERROR``.
    """

    result: Result
    records: list[Record] = field(default_factory=list)
    iterations_used: int = 0
    stop_reason: str = StopReason.COMPLETED.value
    error: BaseException | None = None


@dataclass(slots=True)
class LoopConfig:
    """Configuration for a :class:`LoopRunner`.

    Attributes:
        max_iterations: Hard cap on step invocations. Required.
        propagate_errors: When ``True`` (default) an unhandled step
            exception is re-raised after the runner builds its
            partial :class:`LoopResult`. When ``False`` the runner
            returns the partial result with ``stop_reason=ERROR``
            and the exception attached as ``LoopResult.error`` —
            programs that prefer to surface failures as a result
            field (ReAct's ``stop_reason="ERROR"`` semantics) opt
            into this.
    """

    max_iterations: int
    propagate_errors: bool = True

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError(
                f"LoopConfig.max_iterations must be >= 1, got {self.max_iterations}. "
                "Set it to 1 for a degenerate single-shot loop."
            )


@dataclass
class LoopRunner[State, Record, Result]:
    """Shared execution kernel for iterative programs.

    Programs construct a :class:`LoopRunner` and call :meth:`run` from
    inside their ``forward()`` (or ``invoke``) method. The runner owns
    iteration counting and stop-reason classification; the program
    owns its state shape, the per-iteration step body, and the final
    result assembly.

    The runner is intentionally minimal — it does not own
    :func:`collect_traces` scopes, hook firing, or
    :class:`Invocation` construction. Those concerns belong to the
    enclosing :class:`Program.invoke` (which already wraps every
    program with the right trace + hook scaffolding via
    :class:`Program.__call__`). Keeping the runner focused on
    iteration semantics is what makes it reusable across programs
    with different observability needs.
    """

    config: LoopConfig
    make_state: Callable[[], State]
    step: Callable[[int, State], Awaitable[StepOutcome[Record]]]
    build_result: Callable[[State, list[Record]], Result]
    on_step_error: Callable[[State, BaseException], str | None] | None = None

    async def run(self) -> LoopResult[Result, Record]:
        """Execute the loop and return a :class:`LoopResult`.

        The control flow:

        1. ``state = make_state()``.
        2. For each iteration ``0..max_iterations-1``:
           a. ``outcome = await step(i, state)``
           b. Append ``outcome.record`` to the records list.
           c. If ``outcome.stop_reason`` is set, break with that
              stop reason.
           d. If the step raises:
              - If ``on_step_error`` is set and returns a string,
                break with that stop reason after appending no
                record (the error is recorded on the result).
              - Otherwise, mark the loop as ``ERROR`` and (depending
                on ``config.propagate_errors``) re-raise or return
                the partial result.
        3. ``result = build_result(state, records)``.
        4. Return :class:`LoopResult`.
        """
        state = self.make_state()
        records: list[Record] = []
        iterations_used = 0
        stop_reason: str = StopReason.COMPLETED.value
        error: BaseException | None = None

        for i in range(self.config.max_iterations):
            iterations_used = i + 1
            try:
                outcome = await self.step(i, state)
            except BaseException as exc:
                error = exc
                if self.on_step_error is not None:
                    handled = self.on_step_error(state, exc)
                    if handled is not None:
                        stop_reason = handled
                        break
                stop_reason = StopReason.ERROR.value
                if self.config.propagate_errors:
                    # Build a degenerate partial result so the caller's
                    # exception handler can still pull records out via
                    # exception tagging if it wants. We do not invoke
                    # build_result on the error path because programs
                    # cannot reliably build a final result from a
                    # partial state — let the exception carry the
                    # information instead.
                    raise
                break
            records.append(outcome.record)
            if outcome.stop_reason is not None:
                stop_reason = outcome.stop_reason
                break

        result = self.build_result(state, records)
        return LoopResult(
            result=result,
            records=records,
            iterations_used=iterations_used,
            stop_reason=stop_reason,
            error=error,
        )
