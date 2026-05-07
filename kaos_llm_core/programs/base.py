"""Program — composed LLM function base class.

A Program is a callable that takes typed inputs and produces typed outputs,
potentially making one or more LLM Calls. Programs are the composition unit
of kaos-llm-core: they define how Calls are wired together.

Sub-Calls are discovered via :class:`ProgramGraph` — every public Call
or sub-Program assigned as an instance attribute auto-registers into
``self._child_registry`` through :meth:`Program.__setattr__`. There is no
``vars(self)`` fallback. Programs that store children in a list (e.g.
:class:`Ensemble`'s voters) override :meth:`named_calls` to expose the
list members. **Trace collection** during
``forward()`` is push-based via :func:`collect_traces`: every successful
``Call._execute`` appends its trace to the active collector. This means
loops, repeated invocations, and transient sub-Calls are all captured in
execution order.

Phase 10 redesign: ``Program.invoke()`` returns the canonical
:class:`Invocation` bundling output, trace, usage, error, and context.
``Program.__call__`` unwraps to ``invocation.output`` for the ergonomic
bare-output return. There is no ``_last_trace`` instance attribute and
no ``object.__setattr__(result, "_last_trace", ...)`` mutation. Failed
program executions still propagate their trace upward to the parent
collector via the same Invocation contract.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from kaos_core.logging import get_logger

from kaos_llm_core.observability.collectors import collect_traces, push_trace
from kaos_llm_core.observability.traces import ExecutionTrace
from kaos_llm_core.programs._invocation import Invocation, TokenUsage
from kaos_llm_core.programs.call import Call, _call_context_var
from kaos_llm_core.programs.graph import ProgramGraph
from kaos_llm_core.programs.program_hooks import ProgramHooks, fire_program_hook
from kaos_llm_core.settings import KaosLLMCoreSettings

logger = get_logger(__name__)


class Program:
    """Base class for composed LLM functions.

    Subclass and implement ``forward()`` with your composition logic.
    Assign Call instances as attributes in ``__init__`` for optimizer discovery.

    Example::

        class Analyzer(Program):
            def __init__(self):
                self.extract = Call(ExtractEntities, model="anthropic:claude-sonnet-4-6")
                self.classify = Call(ClassifyRisk, model="anthropic:claude-haiku-4-5")

            async def forward(self, text: str) -> Analysis:
                entities = await self.extract(text=text)
                risk = await self.classify(text=text, entities=entities.entities)
                return Analysis(entities=entities.entities, risk=risk.level)
    """

    # Optional ProgramHooks attached to this Program. Subclasses may set
    # this in __init__; the base class fires the hooks around forward().
    program_hooks: ProgramHooks | None = None

    def __setattr__(self, name: str, value: Any) -> None:
        """Auto-register public Call/Program children into the graph.

        Phase 11: every assignment of a ``Call`` or sub-:class:`Program`
        to a public attribute name (no leading underscore) populates
        the per-instance ``_child_registry`` dict in declaration order.
        Reassigning the attribute to a non-Call/Program value (e.g.
        ``None``) un-registers it. There is no ``vars(self)`` fallback
        — the registry is the single source of truth for
        :meth:`named_calls`, :meth:`get_learnable_state`, and
        :meth:`graph`.

        Programs that store children under private names
        (``self._inner_call``) or in a container (``self.voters``
        list) must explicitly alias the value to a public attribute
        OR override :meth:`named_calls` for the structural exception.
        """
        super().__setattr__(name, value)
        if name.startswith("_") or name == "program_hooks":
            return
        if value is self:
            return
        registry: dict[str, Call | Program] = self.__dict__.setdefault("_child_registry", {})
        if isinstance(value, (Call, Program)):
            registry[name] = value
        elif name in registry:
            # Reassignment to a non-child value un-registers.
            del registry[name]

    @property
    def graph(self) -> ProgramGraph:
        """Return the :class:`ProgramGraph` view over this program's children."""
        return ProgramGraph(self, self.__dict__.setdefault("_child_registry", {}))

    async def forward(self, **kwargs: Any) -> Any:
        """Execute the program logic. Override in subclasses.

        Args:
            **kwargs: Input arguments for the program.

        Returns:
            The program's output.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.forward() is not implemented. "
            f"Override forward() with your composition logic."
        )

    def _trace_enabled(self) -> bool:
        """Resolve ``trace_enabled`` for this Program.

        Returns ``True`` if **any** sub-Call has tracing enabled (or no
        sub-Call has explicit settings, in which case we read the env via
        a fresh ``KaosLLMCoreSettings()``).
        """
        any_settings_seen = False
        for sub in self.named_calls().values():
            settings = getattr(sub, "_core_settings", None)
            if settings is None:
                continue
            any_settings_seen = True
            if bool(getattr(settings, "trace_enabled", True)):
                return True
        if any_settings_seen:
            return False
        return bool(KaosLLMCoreSettings().trace_enabled)

    async def __call__(self, **kwargs: Any) -> Any:
        """Execute the program and return the bare output.

        Ergonomic public surface. Equivalent to
        ``(await self.invoke(**kwargs)).output`` but slightly more efficient.
        For callers that need the full Invocation (trace, usage, error
        context), use :meth:`invoke` directly.
        """
        invocation = await self.invoke(**kwargs)
        return invocation.output

    async def invoke(self, **kwargs: Any) -> Invocation:
        """Execute the program with push-based hierarchical trace collection.

        Returns the finalized :class:`Invocation` bundling output, trace,
        usage, error, and context. ``Program.__call__`` is the
        bare-output unwrap; this method is the runtime contract for
        callers that need access to the trace tree, the rolled-up token
        + cost summary, or the per-task ``KaosContext``.

        Implementation:

        - Opens a :func:`collect_traces` context for the duration of
          ``forward()``. Every successful ``Call._execute`` inside the
          program appends its trace into the collector. Repeated
          invocations of the same sub-Call produce distinct entries.
          Transient sub-Calls created inside ``forward()`` (e.g.
          ``BestOfN``'s per-sample clones) are captured even though
          they are not in :meth:`named_calls`.

        - Builds the parent ``ExecutionTrace`` from the collected
          children, sums child cost/tokens for the parent rollup, and
          packs everything into the returned Invocation.

        - On exception, the partial Invocation is attached to the
          raised exception as ``exc.invocation`` and the exception is
          re-raised. The eval harness reads it from the except block to
          capture failure traces. The parent trace is also pushed into
          the surrounding collector (if any) so a program-of-programs
          sees failed sub-programs in its trace tree.
        """
        ctx = _call_context_var.get()
        prog_hooks = self.program_hooks

        fire_program_hook(
            prog_hooks.on_program_start if prog_hooks else None,
            self,
            kwargs,
            context=ctx,
        )

        # Honor KAOS_LLM_CORE_TRACE_ENABLED — when traces are disabled at
        # the leaf-Call level the parent Program also stops building its
        # parent trace tree.
        if not self._trace_enabled():
            try:
                no_trace_result = await self.forward(**kwargs)
            except Exception as exc:
                fire_program_hook(
                    prog_hooks.on_program_error if prog_hooks else None,
                    self,
                    kwargs,
                    exc,
                    context=ctx,
                )
                raise
            no_trace_invocation = Invocation(
                client=None,
                model="(program)",
                context=ctx,
                output=no_trace_result,
                trace=None,
                usage=TokenUsage(),
            )
            fire_program_hook(
                prog_hooks.on_program_end if prog_hooks else None,
                self,
                kwargs,
                no_trace_invocation,
                context=ctx,
            )
            return no_trace_invocation

        start_time = time.monotonic()
        error_str: str | None = None
        forward_result: Any = None
        children: list[ExecutionTrace] = []
        invocation: Invocation | None = None

        try:
            try:
                with collect_traces() as children:
                    try:
                        forward_result = await self.forward(**kwargs)
                    except Exception as e:
                        error_str = f"{type(e).__name__}: {e}"
                        raise
            finally:
                # Build the parent trace whether or not forward() raised.
                # The children list reflects whatever sub-traces were
                # pushed before the exception, which is exactly what an
                # observer wants on the error path.
                trace = ExecutionTrace(
                    call_name=type(self).__name__,
                    signature=type(self).__name__,
                    inputs=kwargs,
                    model="(program)",
                    codec="(program)",
                    children=list(children),
                    cost_usd=sum(c.cost_usd for c in children),
                    input_tokens=sum(c.input_tokens for c in children),
                    output_tokens=sum(c.output_tokens for c in children),
                    total_tokens=sum(c.total_tokens for c in children),
                    latency_ms=(time.monotonic() - start_time) * 1000,
                    error=error_str,
                )
                invocation = Invocation(
                    client=None,
                    model="(program)",
                    context=ctx,
                    output=forward_result,
                    trace=trace,
                    usage=TokenUsage(
                        input_tokens=trace.input_tokens,
                        output_tokens=trace.output_tokens,
                        total_tokens=trace.total_tokens,
                        cost_usd=trace.cost_usd,
                    ),
                )
                # Push the parent trace into any outer collector. Audit
                # finding F: this MUST run on both the success and the
                # error path so a program-of-programs sees failed
                # sub-programs in its trace tree.
                push_trace(trace)
        except Exception as exc:
            # Tag the partial invocation onto the exception so callers
            # (e.g. the eval harness) can read it from an except block.
            if invocation is not None:
                invocation.error = exc
                exc.invocation = invocation  # ty: ignore[unresolved-attribute]
            fire_program_hook(
                prog_hooks.on_program_error if prog_hooks else None,
                self,
                kwargs,
                exc,
                context=ctx,
            )
            raise

        # Success path
        assert invocation is not None  # set in the finally above
        fire_program_hook(
            prog_hooks.on_program_end if prog_hooks else None,
            self,
            kwargs,
            invocation,
            context=ctx,
        )
        return invocation

    def named_calls(self) -> dict[str, Call | Program]:
        """Return all auto-registered Call/Program children in declaration order.

        Phase 11: reads from the per-instance ``_child_registry`` populated
        by :meth:`__setattr__`. There is no ``vars(self)`` fallback — a
        program with no public Call/Program attributes (or that stores
        them under private names without aliasing to public ones) returns
        an empty dict. Override this method for the structural exception
        of children stored in a list (e.g. :class:`Ensemble`'s voters).
        """
        return dict(self.__dict__.get("_child_registry", {}))

    def get_learnable_state(self) -> dict[str, Any]:
        """Return all learnable state from this Program and its sub-Calls."""
        state: dict[str, Any] = {}
        for name, call in self.named_calls().items():
            state[name] = call.get_learnable_state()
        return state

    def set_learnable_state(self, state: dict[str, Any]) -> None:
        """Restore learnable state from a dict."""
        calls = self.named_calls()
        for name, call_state in state.items():
            if name in calls:
                calls[name].set_learnable_state(call_state)

    SCHEMA_VERSION = 2

    def save(self, path: str | Path) -> None:
        """Save learnable state to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        envelope = {
            "program": type(self).__name__,
            "version": self.SCHEMA_VERSION,
            "state": self.get_learnable_state(),
        }
        path.write_text(json.dumps(envelope, indent=2, default=str), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        """Load learnable state from a JSON file."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if "state" not in data:
            raise ValueError(
                f"Saved program at {path} is missing the 'state' key. "
                f"This file may be corrupted. Re-save the program with Program.save()."
            )
        version = data.get("version", 1)
        if version > self.SCHEMA_VERSION:
            raise ValueError(
                f"Saved program at {path} has version {version}, but this "
                f"kaos-llm-core only supports up to version {self.SCHEMA_VERSION}. "
                f"Upgrade kaos-llm-core to load this file."
            )
        if version < self.SCHEMA_VERSION:
            logger.warning(
                "Loading kaos-llm-core program from %s with v%d schema; current is v%d. "
                "Hyperparameters, codec, and model are not present in v1 files and will "
                "remain at construction-time values. Re-save with Program.save() to "
                "upgrade to v%d.",
                path,
                version,
                self.SCHEMA_VERSION,
                self.SCHEMA_VERSION,
            )
        self.set_learnable_state(data["state"])
