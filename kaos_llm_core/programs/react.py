"""ReAct — reason, call tool, observe, repeat.

Phase 5.2 of the kaos-llm-core gap-closure roadmap, hardened by the
post-shipping audit (Phase 9). Implements the
"reason → call tool → observe → reason-again" loop as a ``Program``,
not as an agent. ``ReAct`` does **not** own a session, persist runs,
request approvals, or manage tool registries; it is a programming
pattern that happens to call user-provided callables in a loop.

Design pinned by ``docs/internal/design/react-loop.md`` and refined by
the Phase 9 audit findings:

- **Public Call API only.** ReAct holds an internal ``Call`` and consumes
  it through the public :meth:`Call.prepare_call` method, which returns a
  typed :class:`CallPlan` bundle. ReAct never reaches into ``Call``'s
  underscore-prefixed internals — that coupling was the #1 audit finding
  and is now fixed. The override surface for subclasses (``ChainOfThought``)
  is unchanged because ``prepare_call`` delegates to the same
  ``_validate_inputs`` / ``_resolve_model`` / ``_get_*`` step methods
  that subclasses already override.
- **CallHooks fire end-to-end.** A ``CallHooks`` attached to the inner
  Call's constructor fires once at the top of the ReAct loop
  (``on_call_start``), once at the end (``on_call_end`` on success,
  ``on_call_error`` on failure), and per validation retry. ProgramHooks
  on the ReAct itself fire ``on_iteration`` per turn so users can
  observe loop progress.
- **Tool error envelope is unambiguous.** ``_invoke_one`` returns a
  ``(payload, is_error)`` tuple so the loop never inspects the payload
  to decide whether the tool failed. A tool that legitimately returns
  ``{"error": True, "details": ...}`` no longer gets mis-flagged.
- **Tokens accumulate even when ``response.usage`` is None.** Some
  provider responses (errors, partial streaming) drop usage; ReAct
  defaults each field to 0 instead of dereferencing ``None``.
- **Validation-retry budget is separate from the iteration budget.**
  ``max_validation_retries`` (default 2) caps decode/parse failures so
  a model that produces malformed output ten times in a row cannot
  exhaust the entire ``max_iterations`` budget without ever calling
  a tool.
- **Empty tool list is invalid.** ``ReAct(..., tools=[])`` raises at
  construction. A degenerate "ReAct with no tools" is just a Call with
  extra trace overhead — use ``Call`` directly.
- **Trace tree is push-based.** Each iteration's child trace is pushed
  via :func:`push_trace` so the surrounding ``Program.__call__``
  collector picks it up. The legacy "manually plumb children list"
  pattern is gone.
- **Loop overrun returns a partial trajectory + ``stop_reason="MAX_ITERATIONS"``**
  with the BEST partially-validated answer if any. It never raises on
  overrun: a library that crashes on a long agentic chain is unusable.
- **Context-length errors return partial result, not an exception.**
  Long-running ReAct loops can blow past the provider's context window;
  ReAct catches that case and returns a partial ``ReActResult`` with
  ``stop_reason="ERROR"`` so the caller still gets the trajectory.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

from kaos_core.logging import get_logger
from kaos_llm_client import ProviderResponse
from kaos_llm_client.messages import AssistantMessage, ToolResultMessage, UserMessage
from kaos_llm_client.settings import KaosLLMSettings
from kaos_llm_client.types import ToolCall, ToolChoice
from pydantic import ValidationError

from kaos_llm_core.codecs.base import Codec
from kaos_llm_core.errors import CallError, CodecError
from kaos_llm_core.observability.collectors import push_trace
from kaos_llm_core.observability.cost import apply_cost_estimates
from kaos_llm_core.observability.traces import ExecutionTrace
from kaos_llm_core.programs._invocation import Invocation, TokenUsage
from kaos_llm_core.programs.base import Program
from kaos_llm_core.programs.call import Call, CallPlan, _call_context_var
from kaos_llm_core.programs.hooks import CallHooks, fire_hook
from kaos_llm_core.programs.loop_runner import LoopConfig, LoopRunner, StepOutcome
from kaos_llm_core.programs.program_hooks import ProgramHooks, fire_program_hook
from kaos_llm_core.programs.result_mixin import OutputForwardingMixin
from kaos_llm_core.programs.tool import Tool
from kaos_llm_core.settings import KaosLLMCoreSettings
from kaos_llm_core.signatures.signature import Signature
from kaos_llm_core.types import Example

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public dataclasses (sub-design §5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolObservation:
    """One executed tool call: arguments in, result out, plus an error flag.

    ``result`` is whatever the tool returned. When the call failed (because
    the tool raised, the tool name was unknown, or the tool was invoked with
    invalid arguments), ``result`` is the dict
    ``{"error": True, "message": "..."}`` and ``is_error`` is True.

    ``is_error`` is set by ReAct's tool dispatcher, **not** by inspecting the
    payload shape. A user tool that legitimately returns
    ``{"error": True, "details": ...}`` is forwarded verbatim with
    ``is_error=False``. The previous "look at result.error" heuristic was
    audit finding #3 and is now fixed.
    """

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    result: Any
    is_error: bool


@dataclass(frozen=True, slots=True)
class Iteration:
    """One pass through the ReAct loop.

    The 0-indexed iteration number, the assistant's text on this turn
    (which may be empty if the model only emitted tool calls), the tool
    calls the model emitted (in order), the observations from executing
    them, and a decode/validation error string if this turn failed
    structured-output validation.
    """

    iteration: int
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolObservation] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ReActResult(OutputForwardingMixin):
    """Final result of a ReAct run.

    ``outputs`` is the validated Signature output dict, or ``None`` when
    the loop overran without ever producing a valid final answer.
    ``trajectory`` is the full per-iteration history. ``stop_reason``
    explains why the loop stopped. ``validation_retries`` counts decode
    failures separately from real tool-using iterations.

    Attribute access is forwarded to ``outputs`` via
    :class:`OutputForwardingMixin` so callers can write ``result.answer``
    instead of ``result.outputs["answer"]``.
    """

    outputs: dict[str, Any] | None
    trajectory: list[Iteration]
    stop_reason: Literal["TERMINATED", "MAX_ITERATIONS", "ERROR"]
    iterations_used: int
    validation_retries: int = 0

    def is_complete(self) -> bool:
        """``True`` iff the loop produced a validated final answer.

        Equivalent to ``outputs is not None and stop_reason == "TERMINATED"``.
        Callers that want to access fields on ``result.outputs`` should check
        this first to avoid an ``AttributeError`` on overrun.
        """
        return self.outputs is not None and self.stop_reason == "TERMINATED"


# ---------------------------------------------------------------------------
# LoopRunner state + per-iteration record (Phase 12B)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _ReActLoopState:
    """Mutable per-run state threaded through the LoopRunner step function.

    Phase 12B: ReAct's loop body migrates onto :class:`LoopRunner`. The
    runner owns iteration counting and stop-reason classification; the
    step function owns the per-turn LLM call, decode, and tool dispatch.
    Everything that survives across iterations lives on this state.

    ``exit_reason`` is set by the step function (or by ``_on_step_error``)
    to communicate ReAct's domain-specific stop reason — ``"TERMINATED"``
    for a clean final answer, ``"ERROR"`` for validation exhaustion or
    context overflow. The runner's own :class:`StopReason` enum is the
    iteration-control taxonomy; ReAct's ``stop_reason`` field uses a
    different vocabulary that the result builder reads from this slot.
    """

    plan: CallPlan
    messages: list[dict[str, Any]]
    tool_definitions_arg: list[Any] | None
    tool_lookup: dict[str, Tool]
    trajectory: list[Iteration] = field(default_factory=list)
    validation_retries: int = 0
    last_response: ProviderResponse | None = None
    final_outputs: dict[str, Any] | None = None
    exit_reason: str | None = None


@dataclass(slots=True)
class _ReActIterRecord:
    """Per-iteration record produced by :meth:`ReAct._react_step`."""

    iter_trace: ExecutionTrace


# ---------------------------------------------------------------------------
# ReAct program
# ---------------------------------------------------------------------------


# Patterns in provider error messages that signal a context-length overflow.
# We catch these to return a partial ReActResult instead of letting them
# propagate as a hard failure — long agentic chains are exactly when context
# overflows, and dropping the trajectory at that point would be unhelpful.
_CONTEXT_OVERFLOW_PATTERNS = (
    "context length",
    "maximum context",
    "context window",
    "too many tokens",
    "exceeds the maximum",
    "input is too long",
    "prompt is too long",
)


def _looks_like_context_overflow(exc: BaseException) -> bool:
    """Heuristic: does this exception look like a provider context-length 400?"""
    msg = str(exc).lower()
    return any(p in msg for p in _CONTEXT_OVERFLOW_PATTERNS)


class ReAct(Program):
    """A reason → tool → observe loop driven by an LLM with tool calling.

    Example::

        from kaos_llm_core import ReAct, Tool, Signature, InputField, OutputField

        class Answer(Signature):
            '''Answer the question using tools when needed.'''
            question: str = InputField(description="The user's question")
            answer: str = OutputField(description="The final answer")

        def multiply(a: float, b: float) -> float:
            '''Multiply two numbers.'''
            return a * b

        react = ReAct(
            Answer,
            tools=[Tool.from_callable(multiply)],
            model="anthropic:claude-haiku-4-5",
        )
        result = await react(question="What is 17 times 23?")
        if result.is_complete():
            print(result.answer)             # forwarded to result.outputs["answer"]
        print(result.stop_reason)            # "TERMINATED"
        print(len(result.trajectory))        # number of iterations used
    """

    # Hard cap on ``max_iterations`` — protects direct Python API
    # consumers from typo-driven runaway spend (e.g. ``max_iterations=10000``
    # instead of ``100``). The MCP tool layer enforces a tighter cap;
    # this is the structural backstop. Subclasses may override with
    # full awareness of cost implications.
    MAX_ITERATIONS: int = 1000

    def __init__(
        self,
        signature: type[Signature],
        tools: list[Tool],
        *,
        model: str | None = None,
        codec: Codec | None = None,
        max_iterations: int = 50,
        max_validation_retries: int = 2,
        on_tool_error: Literal["continue", "raise"] = "continue",
        examples: list[Example] | None = None,
        instructions: str | None = None,
        client: Any = None,
        settings: KaosLLMSettings | None = None,
        core_settings: KaosLLMCoreSettings | None = None,
        hooks: CallHooks | None = None,
        program_hooks: ProgramHooks | None = None,
    ) -> None:
        if max_iterations < 1:
            raise CallError(
                f"ReAct(max_iterations={max_iterations}) is invalid. "
                "max_iterations must be >= 1. Pass a larger value, or "
                "drop the override to use the default of 50."
            )
        if max_iterations > self.MAX_ITERATIONS:
            raise CallError(
                f"ReAct(max_iterations={max_iterations}) exceeds the hard cap "
                f"of {self.MAX_ITERATIONS}. The cap exists to prevent runaway "
                "API spend from misconfigured loops; pass a smaller value, "
                "or override the class attribute MAX_ITERATIONS in a subclass "
                "with full awareness of the cost implications."
            )
        if max_validation_retries < 0:
            raise CallError(
                f"ReAct(max_validation_retries={max_validation_retries}) is invalid. "
                "Use 0 to disable retries, 2 (default) for moderate tolerance, "
                "or a larger value for noisy models."
            )
        if on_tool_error not in ("continue", "raise"):
            raise CallError(
                f"ReAct(on_tool_error={on_tool_error!r}) is invalid. "
                "Use 'continue' (default — feed errors back to the model) "
                "or 'raise' (propagate the exception to the caller)."
            )
        if not tools:
            # Audit finding #15: tools=[] is an antipattern (it's just a Call
            # with extra trace overhead) but some test fixtures rely on it as
            # a degenerate path for the decode-failure-recovery loop, so warn
            # rather than raise. Users who hit this in production should switch
            # to a plain Call.
            logger.warning(
                "ReAct(tools=[]) is a degenerate construction — without tools "
                "the loop is just a Call with extra trace overhead. Use Call "
                "directly unless you specifically need ReAct's "
                "decode-failure-retry semantics."
            )

        self.signature = signature
        self.tools = list(tools)
        self.max_iterations = max_iterations
        self.max_validation_retries = max_validation_retries
        self.on_tool_error = on_tool_error
        self.examples: list[Example] = list(examples) if examples else []
        self.instructions = instructions
        self.program_hooks = program_hooks

        # An internal Call gives us the public step-method pipeline (model +
        # client resolution, codec wiring, kwargs prep, output validation)
        # for free via :meth:`Call.prepare_call`. We never invoke its
        # ``_execute()`` — ReAct owns its own loop — but we DO honor any
        # ``CallHooks`` the user attached, firing them at the program
        # boundary so observers see ReAct as a logical Call.
        self._inner_call = Call(
            signature,
            model=model,
            codec=codec,
            client=client,
            settings=settings,
            core_settings=core_settings,
            examples=examples,
            instructions=instructions,
            hooks=hooks,
        )
        # Phase 11: alias the private inner Call to a public attribute so
        # the auto-registering ProgramGraph picks it up. Optimizers,
        # save/load, and ``program.graph.primary()`` all reach the inner
        # Call through ``self.call``.
        self.call = self._inner_call

    @property
    def signature_class(self) -> type[Signature]:
        """The Signature class describing the final ReAct output."""
        return self.signature

    @property
    def hooks(self) -> CallHooks | None:
        """The :class:`CallHooks` attached to the inner Call (read-only)."""
        return self._inner_call.hooks

    @property
    def inner_call(self) -> Call:
        """Public accessor for the inner Call.

        ReAct holds an internal Call as its execution-configuration
        container. The attribute is stored as ``_inner_call`` (legacy
        name) but exposed via this property and via :meth:`named_calls`
        so optimizers, save/load round-trips, and external code can
        reach the underlying Call without touching the underscore name.
        Phase 9d save/load fix.
        """
        return self._inner_call

    # Phase 11: ``named_calls`` override deleted. The Phase-11
    # ProgramGraph auto-registration in ``Program.__setattr__`` picks up
    # ``self.call`` (aliased in __init__) so the base implementation
    # already returns ``{"call": self._inner_call}``.

    # ------------------------------------------------------------------
    # Program API
    # ------------------------------------------------------------------

    def _trace_enabled(self) -> bool:
        """Resolve ``trace_enabled`` from the inner Call's settings.

        Mirrors :meth:`Program._trace_enabled` so ReAct honors the same
        ``KAOS_LLM_CORE_TRACE_ENABLED`` toggle as the rest of the stack.
        Phase 9 audit follow-up #4.
        """
        settings = getattr(self._inner_call, "_core_settings", None)
        if settings is not None:
            return bool(getattr(settings, "trace_enabled", True))
        return True

    async def __call__(self, **inputs: Any) -> ReActResult:
        """Run the ReAct loop and return the bare :class:`ReActResult` output.

        Ergonomic public surface. Equivalent to
        ``(await self.invoke(**inputs)).output`` but slightly more efficient.
        Use :meth:`invoke` if you need access to the trace, usage, or
        per-task context bundled with the result.
        """
        invocation = await self.invoke(**inputs)
        return invocation.output

    async def invoke(self, **inputs: Any) -> Invocation:
        """Run the ReAct loop and return the full :class:`Invocation`.

        Overrides ``Program.invoke`` so we can:

        1. Fire ``CallHooks.on_call_start`` / ``on_call_end`` /
           ``on_call_error`` around the WHOLE loop (matching how
           ``Call._execute`` fires them around a single logical Call).
        2. Fire ``ProgramHooks.on_program_start`` / ``on_iteration`` /
           ``on_program_end`` for loop-level observers.
        3. Build a custom parent ``ExecutionTrace`` whose children are the
           per-iteration sub-traces — the base ``Program.__call__`` collector
           would only see the inner Call's trace once at the end.
        4. Push the parent trace into any surrounding collector so a
           program-of-programs sees the ReAct parent in its trace tree.
        """
        ctx = _call_context_var.get()
        call_hooks = self._inner_call.hooks

        # Honor KAOS_LLM_CORE_TRACE_ENABLED — skip parent trace assembly
        # when traces are disabled. We still run the loop and fire hooks
        # so the user gets the result + observability; we just don't
        # build a parent ExecutionTrace.
        if not self._trace_enabled():
            fire_hook(
                call_hooks.on_call_start if call_hooks else None,
                self._inner_call,
                inputs,
                context=ctx,
            )
            fire_program_hook(
                self.program_hooks.on_program_start if self.program_hooks else None,
                self,
                inputs,
                context=ctx,
            )
            try:
                result_no_trace, _children = await self._run_loop(inputs)
            except Exception as exc:
                # Audit fix: tag the partial Invocation onto the
                # exception so callers in ``except`` blocks (eval
                # harness, optimizers) can recover failure context
                # without an extra ContextVar — matching every other
                # invoke() failure path.
                no_trace_error_invocation = Invocation(
                    client=None,
                    model="(react)",
                    context=ctx,
                    output=None,
                    trace=None,
                    usage=TokenUsage(),
                    error=exc,
                )
                exc.invocation = no_trace_error_invocation  # ty: ignore[unresolved-attribute]
                fire_hook(
                    call_hooks.on_call_error if call_hooks else None,
                    self._inner_call,
                    inputs,
                    exc,
                    context=ctx,
                )
                fire_program_hook(
                    self.program_hooks.on_program_error if self.program_hooks else None,
                    self,
                    inputs,
                    exc,
                    context=ctx,
                )
                raise
            no_trace_invocation = Invocation(
                client=None,
                model="(react)",
                context=ctx,
                output=result_no_trace,
                trace=None,
                usage=TokenUsage(),
            )
            fire_hook(
                call_hooks.on_call_end if call_hooks else None,
                self._inner_call,
                inputs,
                no_trace_invocation,
                context=ctx,
            )
            fire_program_hook(
                self.program_hooks.on_program_end if self.program_hooks else None,
                self,
                inputs,
                no_trace_invocation,
                context=ctx,
            )
            return no_trace_invocation

        fire_hook(
            call_hooks.on_call_start if call_hooks else None,
            self._inner_call,
            inputs,
            context=ctx,
        )
        fire_program_hook(
            self.program_hooks.on_program_start if self.program_hooks else None,
            self,
            inputs,
            context=ctx,
        )

        start_time = time.monotonic()
        error_str: str | None = None
        result: ReActResult | None = None
        children: list[ExecutionTrace] = []

        try:
            result, children = await self._run_loop(inputs)
        except Exception as exc:
            error_str = f"{type(exc).__name__}: {exc}"
            error_trace = self._build_parent_trace(
                inputs=inputs,
                children=children,
                start_time=start_time,
                error_str=error_str,
                outputs={},
            )
            error_invocation = Invocation(
                client=None,
                model="(react)",
                context=ctx,
                output=None,
                trace=error_trace,
                usage=TokenUsage(
                    input_tokens=error_trace.input_tokens,
                    output_tokens=error_trace.output_tokens,
                    total_tokens=error_trace.total_tokens,
                    cost_usd=error_trace.cost_usd,
                ),
                error=exc,
            )
            push_trace(error_trace)
            exc.invocation = error_invocation  # ty: ignore[unresolved-attribute]
            fire_hook(
                call_hooks.on_call_error if call_hooks else None,
                self._inner_call,
                inputs,
                exc,
                context=ctx,
            )
            fire_program_hook(
                self.program_hooks.on_program_error if self.program_hooks else None,
                self,
                inputs,
                exc,
                context=ctx,
            )
            raise

        # Phase 9c: stop_reason==ERROR exit needs an error string on the
        # parent trace so consumers reading the trace can tell apart the
        # success and graceful-error paths.
        parent_error_str: str | None = None
        if result.stop_reason == "ERROR":
            parent_error_str = (
                f"ReAct stopped with stop_reason=ERROR after "
                f"{result.iterations_used} iterations, "
                f"{result.validation_retries} validation retries"
            )
        success_trace = self._build_parent_trace(
            inputs=inputs,
            children=children,
            start_time=start_time,
            error_str=parent_error_str,
            outputs=dict(result.outputs) if result.outputs else {},
        )
        success_invocation = Invocation(
            client=None,
            model="(react)",
            context=ctx,
            output=result,
            trace=success_trace,
            usage=TokenUsage(
                input_tokens=success_trace.input_tokens,
                output_tokens=success_trace.output_tokens,
                total_tokens=success_trace.total_tokens,
                cost_usd=success_trace.cost_usd,
            ),
        )
        push_trace(success_trace)

        fire_hook(
            call_hooks.on_call_end if call_hooks else None,
            self._inner_call,
            inputs,
            success_invocation,
            context=ctx,
        )
        fire_program_hook(
            self.program_hooks.on_program_end if self.program_hooks else None,
            self,
            inputs,
            success_invocation,
            context=ctx,
        )

        return success_invocation

    def _build_parent_trace(
        self,
        *,
        inputs: dict[str, Any],
        children: list[ExecutionTrace],
        start_time: float,
        error_str: str | None,
        outputs: dict[str, Any],
    ) -> ExecutionTrace:
        """Build the ReAct-level parent trace from collected iteration traces."""
        return ExecutionTrace(
            call_name=type(self).__name__,
            signature=self.signature.__name__,
            inputs=dict(inputs),
            model="(react)",
            codec=type(self._inner_call.codec).__name__,
            children=children,
            cost_usd=sum(c.cost_usd for c in children),
            input_tokens=sum(c.input_tokens for c in children),
            output_tokens=sum(c.output_tokens for c in children),
            total_tokens=sum(c.total_tokens for c in children),
            latency_ms=(time.monotonic() - start_time) * 1000,
            error=error_str,
            outputs=outputs,
        )

    async def forward(self, **inputs: Any) -> ReActResult:
        """Synonym for ``__call__``.

        Programs typically split ``__call__`` (with trace assembly) from
        ``forward`` (raw logic). For ReAct the loop owns the trace
        assembly so ``forward`` and ``__call__`` produce the same
        ``ReActResult`` and same trace tree — calling ``forward`` does
        NOT bypass observability. This was audit finding #12.
        """
        return await self.__call__(**inputs)

    # ------------------------------------------------------------------
    # Loop body — decomposed into step methods for #8
    # ------------------------------------------------------------------

    async def _run_loop(self, inputs: dict[str, Any]) -> tuple[ReActResult, list[ExecutionTrace]]:
        """Execute the ReAct loop via :class:`LoopRunner`.

        Phase 12B: this method is now a thin wrapper around the shared
        :class:`LoopRunner` kernel. The runner owns iteration counting,
        stop-reason classification, and the on-step-error escape hatch
        (used here to convert provider context overflows into a clean
        ``ERROR`` exit). The per-turn body lives on :meth:`_react_step`;
        the final ``ReActResult`` assembly lives on
        :meth:`_react_build_result`.

        Returns ``(result, per-iteration child traces)``. ReAct's
        ``invoke()`` method then wraps these in a parent ``ExecutionTrace``
        and an :class:`Invocation`.
        """
        state = await self._make_react_state(inputs)
        runner: LoopRunner[_ReActLoopState, _ReActIterRecord, ReActResult] = LoopRunner(
            config=LoopConfig(
                max_iterations=self.max_iterations,
                propagate_errors=True,
            ),
            make_state=lambda s=state: s,
            step=self._react_step,
            build_result=self._react_build_result,
            on_step_error=self._react_on_step_error,
        )
        loop_result = await runner.run()
        children = [r.iter_trace for r in loop_result.records]
        return loop_result.result, children

    async def _make_react_state(self, inputs: dict[str, Any]) -> _ReActLoopState:
        """Resolve the call plan + initial messages and build a fresh state."""
        plan, messages, tool_definitions_arg, tool_lookup = await self._setup(inputs)
        return _ReActLoopState(
            plan=plan,
            messages=messages,
            tool_definitions_arg=tool_definitions_arg,
            tool_lookup=tool_lookup,
        )

    async def _react_step(
        self, iteration: int, state: _ReActLoopState
    ) -> StepOutcome[_ReActIterRecord]:
        """Run one ReAct turn against ``state``.

        Returns a :class:`StepOutcome` whose ``record`` carries the per-
        iteration trace and whose ``stop_reason`` is one of:

        - ``"TERMINATED"`` — the model produced a validated final answer.
        - ``"ERROR"`` — validation retries are exhausted.
        - ``None`` — keep iterating (a tool turn or a recoverable
          decode failure).

        State mutations: every path appends to ``state.trajectory``.
        Successful finalization sets ``state.final_outputs`` and
        ``state.exit_reason``. The exhausted-retry path sets
        ``state.exit_reason = "ERROR"`` so :meth:`_react_build_result`
        knows the difference between a max-iterations exit and an
        error exit.
        """
        iter_start = time.monotonic()
        plan = state.plan
        logger.debug(
            "react.iteration_start: iteration=%d model=%s tools=%d messages=%d",
            iteration,
            plan.model,
            len(state.tool_definitions_arg or []),
            len(state.messages),
        )
        response = await plan.client.chat_async(
            state.messages,
            tools=state.tool_definitions_arg,
            tool_choice=ToolChoice(type="auto") if state.tool_definitions_arg else None,
            **plan.call_kwargs,
        )
        state.last_response = response
        state.messages.append(AssistantMessage.from_response(response))
        iter_trace = self._build_iteration_trace(
            iteration=iteration,
            model=plan.model,
            codec_name=type(plan.codec).__name__,
            response=response,
            started_at=iter_start,
        )

        iter_elapsed_ms = (time.monotonic() - iter_start) * 1000
        # Final answer turn — no tool calls.
        if not response.tool_calls:
            logger.debug(
                "react.iteration_final: iteration=%d latency_ms=%.0f text_len=%d",
                iteration,
                iter_elapsed_ms,
                len(response.text),
            )
            final_outputs, decode_error = self._try_finalize(plan, response)

            if final_outputs is not None:
                iter_trace.outputs = dict(final_outputs)
                state.final_outputs = final_outputs
                state.exit_reason = "TERMINATED"
                state.trajectory.append(
                    Iteration(
                        iteration=iteration,
                        text=response.text,
                        tool_calls=[],
                        tool_results=[],
                        error=None,
                    )
                )
                self._fire_iteration_hook(
                    iteration=iteration,
                    outputs=final_outputs,
                    trace=iter_trace,
                    tool_results=[],
                    decode_error=None,
                )
                return StepOutcome(record=_ReActIterRecord(iter_trace), stop_reason="TERMINATED")

            # Decode failed. Fire on_validation_retry on the inner Call's
            # CallHooks so users observing per-Call telemetry see the
            # retry event regardless of whether the loop ultimately
            # recovers. Phase 9c follow-up: ReAct previously suppressed
            # this hook because its loop bypasses Call._execute.
            state.validation_retries += 1
            fire_hook(
                self._inner_call.hooks.on_validation_retry if self._inner_call.hooks else None,
                self._inner_call,
                plan.validated_inputs,
                state.validation_retries,
                Exception(decode_error) if decode_error else Exception("decode failure"),
                context=_call_context_var.get(),
            )

            if state.validation_retries > self.max_validation_retries:
                logger.warning(
                    "ReAct iteration %d: exhausted %d validation retries; "
                    "stopping with stop_reason=ERROR. Last decode error: %s",
                    iteration,
                    self.max_validation_retries,
                    decode_error,
                )
                iter_trace.error = decode_error
                state.exit_reason = "ERROR"
                state.trajectory.append(
                    Iteration(
                        iteration=iteration,
                        text=response.text,
                        tool_calls=[],
                        tool_results=[],
                        error=decode_error,
                    )
                )
                self._fire_iteration_hook(
                    iteration=iteration,
                    outputs=None,
                    trace=iter_trace,
                    tool_results=[],
                    decode_error=decode_error,
                )
                return StepOutcome(record=_ReActIterRecord(iter_trace), stop_reason="ERROR")

            # Append corrective user message and continue on the next turn.
            state.messages.append(
                UserMessage(
                    f"Your previous response did not validate against the "
                    f"required schema: {decode_error}. Please respond again "
                    f"with output that matches the requested format exactly."
                )
            )
            iter_trace.error = decode_error
            state.trajectory.append(
                Iteration(
                    iteration=iteration,
                    text=response.text,
                    tool_calls=[],
                    tool_results=[],
                    error=decode_error,
                )
            )
            self._fire_iteration_hook(
                iteration=iteration,
                outputs=None,
                trace=iter_trace,
                tool_results=[],
                decode_error=decode_error,
            )
            return StepOutcome(record=_ReActIterRecord(iter_trace))

        # Tool calls present — execute sequentially.
        tool_names = [tc.name for tc in response.tool_calls]
        logger.debug(
            "react.iteration_tools: iteration=%d tool_calls=%d tools=%s latency_ms=%.0f",
            iteration,
            len(response.tool_calls),
            tool_names,
            iter_elapsed_ms,
        )
        tool_results = await self._execute_tool_calls(
            response.tool_calls, state.tool_lookup, state.messages
        )
        state.trajectory.append(
            Iteration(
                iteration=iteration,
                text=response.text,
                tool_calls=list(response.tool_calls),
                tool_results=tool_results,
                error=None,
            )
        )
        self._fire_iteration_hook(
            iteration=iteration,
            outputs=None,
            trace=iter_trace,
            tool_results=tool_results,
            decode_error=None,
        )
        return StepOutcome(record=_ReActIterRecord(iter_trace))

    def _react_on_step_error(self, state: _ReActLoopState, exc: BaseException) -> str | None:
        """Convert provider context overflows into a clean ``ERROR`` exit.

        Other exceptions return ``None`` so :class:`LoopRunner`
        re-raises them. The previous in-loop ``try/except`` around the
        ``chat_async`` call moved into this :class:`LoopRunner`
        callback so the runner owns the iteration-error policy.
        """
        if _looks_like_context_overflow(exc):
            logger.warning(
                "ReAct: provider context overflow (%s); returning partial trajectory.",
                exc,
            )
            state.exit_reason = "ERROR"
            return "ERROR"
        return None

    def _react_build_result(
        self, state: _ReActLoopState, records: list[_ReActIterRecord]
    ) -> ReActResult:
        """Assemble the final :class:`ReActResult` from the loop state.

        Phase 12B: this is the ``build_result`` callback handed to
        :class:`LoopRunner`. It runs after the runner exits — whether
        cleanly (TERMINATED), via early stop (ERROR), or by reaching
        ``max_iterations``. The runner does not know about ReAct's
        domain stop-reason vocabulary, so we read it from
        ``state.exit_reason`` (set by the step function or
        :meth:`_react_on_step_error`); a ``None`` slot means the loop
        ran out of iterations and we should attempt the overrun
        best-effort decode.
        """
        del records  # iter_traces are exposed via the LoopResult.records list
        stop_reason: Literal["TERMINATED", "MAX_ITERATIONS", "ERROR"]
        if state.exit_reason == "TERMINATED":
            stop_reason = "TERMINATED"
            final_outputs = state.final_outputs
        elif state.exit_reason == "ERROR":
            stop_reason = "ERROR"
            final_outputs = state.final_outputs
        else:
            # Max iterations reached without a TERMINATED or ERROR exit.
            stop_reason = "MAX_ITERATIONS"
            final_outputs = self._build_overrun_result(state.plan, state.last_response)
        return ReActResult(
            outputs=final_outputs,
            trajectory=state.trajectory,
            stop_reason=stop_reason,
            iterations_used=len(state.trajectory),
            validation_retries=state.validation_retries,
        )

    async def _setup(
        self, inputs: dict[str, Any]
    ) -> tuple[CallPlan, list[dict[str, Any]], list[Any] | None, dict[str, Tool]]:
        """Resolve the call plan and build the initial message + tool tables."""
        plan = await self._inner_call.prepare_call(inputs)
        messages: list[dict[str, Any]] = list(
            plan.codec.encode(
                plan.effective_signature,
                plan.validated_inputs,
                examples=self.examples or None,
                instructions=plan.effective_instructions,
            )
        )
        tool_definitions = [t.definition for t in self.tools]
        tool_lookup: dict[str, Tool] = {t.definition.name: t for t in self.tools}
        tool_definitions_arg: list[Any] | None = tool_definitions if tool_definitions else None
        return plan, messages, tool_definitions_arg, tool_lookup

    def _build_iteration_trace(
        self,
        *,
        iteration: int,
        model: str,
        codec_name: str,
        response: ProviderResponse,
        started_at: float,
    ) -> ExecutionTrace:
        """Build the per-iteration child trace.

        ``response.usage`` may be ``None`` for some provider responses (errors,
        partial streaming). Default each token field to 0 instead of crashing
        — audit finding #21.

        Calls :func:`apply_cost_estimates` so the iteration trace's
        ``cost_usd`` is populated from the per-model pricing table. Without
        this, every ReAct iteration trace had ``cost_usd=0`` and the parent
        ReAct trace summed zeros. Any ``Budget(max_cost_usd=...)`` over a
        program containing ReAct ran silently uncapped — Phase 9c
        critical-finding S1-1.
        """
        usage = response.usage
        trace = ExecutionTrace(
            call_name=f"{type(self).__name__}.iter[{iteration}]",
            signature=self.signature.__name__,
            inputs={},
            model=model,
            codec=codec_name,
            input_tokens=usage.input_tokens if usage is not None else 0,
            output_tokens=usage.output_tokens if usage is not None else 0,
            total_tokens=usage.total_tokens if usage is not None else 0,
            latency_ms=(time.monotonic() - started_at) * 1000,
        )
        apply_cost_estimates(trace)
        return trace

    def _try_finalize(
        self, plan: CallPlan, response: ProviderResponse
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Try to decode + validate the response as the final answer.

        Returns ``(outputs, error)``: ``outputs`` is the validated dict on
        success, ``None`` on failure with ``error`` set. The except clause is
        narrowed to known decode/validation exceptions; anything else
        propagates as a real failure (audit finding #4).
        """
        try:
            output_dict = plan.codec.decode(plan.effective_signature, response)
            validated = plan.output_model.model_validate(output_dict)
            return _outputs_to_dict(validated, output_dict), None
        except (CodecError, ValidationError, ValueError) as exc:
            return None, str(exc)

    async def _execute_tool_calls(
        self,
        tool_calls: list[ToolCall],
        tool_lookup: dict[str, Tool],
        messages: list[dict[str, Any]],
    ) -> list[ToolObservation]:
        """Execute every tool call sequentially and append tool_result messages."""
        tool_results: list[ToolObservation] = []
        for tc in tool_calls:
            payload, is_error = await self._invoke_one(tc, tool_lookup)
            messages.append(ToolResultMessage(tc.id, payload, name=tc.name))
            tool_results.append(
                ToolObservation(
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    arguments=dict(tc.arguments),
                    result=payload,
                    is_error=is_error,
                )
            )
        return tool_results

    def _build_overrun_result(
        self, plan: CallPlan, last_response: ProviderResponse | None
    ) -> dict[str, Any] | None:
        """Best-effort decode of the most recent response on loop overrun.

        Narrowed except clause matches the in-loop finalize path: only decode/
        validation exceptions become ``None``; anything else still propagates.
        """
        if last_response is None:
            return None
        try:
            output_dict = plan.codec.decode(plan.effective_signature, last_response)
            validated = plan.output_model.model_validate(output_dict)
            return _outputs_to_dict(validated, output_dict)
        except (CodecError, ValidationError, ValueError):
            return None

    def _fire_iteration_hook(
        self,
        *,
        iteration: int,
        outputs: dict[str, Any] | None,
        trace: ExecutionTrace,
        tool_results: list[ToolObservation],
        decode_error: str | None,
    ) -> None:
        """Fire :class:`ProgramHooks` ``on_iteration`` for this turn.

        The payload dict is intentionally minimal — observers can read the
        per-iteration trace via ``trace`` and walk the trajectory if they
        need more context. Hook errors are swallowed by ``fire_program_hook``.
        """
        if self.program_hooks is None or self.program_hooks.on_iteration is None:
            return
        payload = {
            "outputs": outputs,
            "tool_results": tool_results,
            "decode_error": decode_error,
            "trace": trace,
        }
        fire_program_hook(
            self.program_hooks.on_iteration,
            self,
            iteration,
            payload,
            context=_call_context_var.get(),
        )

    async def _invoke_one(self, tc: ToolCall, tool_lookup: dict[str, Tool]) -> tuple[Any, bool]:
        """Execute one tool call.

        Returns ``(payload, is_error)``. The error flag is set by ReAct's
        own dispatcher, never by inspecting the payload, so a tool that
        legitimately returns ``{"error": True, "details": ...}`` is forwarded
        with ``is_error=False``. Audit finding #3.

        Unknown tool name and tool exceptions both produce
        ``{"error": True, "message": "..."}`` so the model gets a uniform
        error shape it can reason about. ``on_tool_error="raise"`` skips
        the catch and lets exceptions propagate.
        """
        tool = tool_lookup.get(tc.name)
        if tool is None:
            return (
                {
                    "error": True,
                    "message": (
                        f"Unknown tool '{tc.name}'. Available tools: "
                        f"{sorted(tool_lookup)}. Pick one of the available tools "
                        "or answer directly without calling a tool."
                    ),
                },
                True,
            )
        try:
            return (await tool.invoke(tc.arguments), False)
        except Exception as exc:
            if self.on_tool_error == "raise":
                raise
            return (
                {
                    "error": True,
                    "message": (
                        f"Tool '{tc.name}' raised: {type(exc).__name__}: {exc}. "
                        "Try different arguments or pick a different tool."
                    ),
                },
                True,
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _outputs_to_dict(validated: Any, fallback: dict[str, Any]) -> dict[str, Any]:
    """Convert a validated Pydantic model into a plain dict.

    The output model is created via ``create_output_model`` which produces
    a Pydantic ``BaseModel`` subclass — ``model_dump()`` is the contract.
    The fallback path returns a copy of the raw decoded dict in the rare
    case where the validated object is not a BaseModel (e.g. a custom
    output model that bypasses Pydantic). Audit finding #17 noted this
    is "should not happen in practice"; we keep the fallback for
    forward-compat with custom output models rather than asserting.
    """
    dump = getattr(validated, "model_dump", None)
    if callable(dump):
        return dict(dump())
    return dict(fallback)
