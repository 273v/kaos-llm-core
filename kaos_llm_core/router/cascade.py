"""CascadeRouter — try cheap model first, escalate on quality check failure.

Each cascade step creates a temporary Call clone and invokes its pipeline,
getting the full retry/trace/validation flow for free. No pipeline duplication.

Concurrency
-----------

Phase 9c hardening: ``execute_cascade`` keeps per-execution scratch state
on the **call stack**, not on the router instance. The instance-level
``last_traces`` / ``model_used`` properties return a snapshot of the most
recent completion, but concurrent ``execute_cascade`` invocations on the
same router no longer corrupt each other's lists. Each concurrent caller
gets its own local ``traces: list[ExecutionTrace]`` accumulator.

The instance properties race on the final atomic write (whichever cascade
finishes last wins), but the values are always consistent — there are no
torn lists or ``IndexError`` reads against a half-cleared list. Callers
that need per-execution traces should consume the return tuple from
``execute_cascade`` directly instead of reading the instance properties.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kaos_core.logging import get_logger

from kaos_llm_core.errors import CallError
from kaos_llm_core.observability.traces import ExecutionTrace
from kaos_llm_core.signatures.signature import Signature

logger = get_logger(__name__)


# Module-level sentinel for the default escalation check. Identity-checked
# by router/serialization.py to detect "user did not pass a custom callable"
# without depending on fragile bytecode comparison.
def _default_escalation_check(_result: Any) -> bool:
    """Default escalation check: always accept the first model's result."""
    return True


class CascadeRouter:
    """Try models in order (cheapest first), escalate on quality check failure.

    Example::

        router = CascadeRouter(
            models=["anthropic:claude-haiku-4-5", "anthropic:claude-sonnet-4-6"],
            escalation_check=lambda result: getattr(result, 'confidence', 0) >= 0.8,
        )
        call = Call(ExtractEntities, router=router)
        result = await call(text="...")
    """

    def __init__(
        self,
        models: list[str],
        escalation_check: Callable[[Any], bool] | None = None,
    ) -> None:
        if not models:
            raise CallError(
                "CascadeRouter requires at least one model. "
                "Pass models=['anthropic:claude-haiku-4-5', 'anthropic:claude-sonnet-4-6']."
            )
        self.models = models
        self.escalation_check = escalation_check or _default_escalation_check
        self._last_traces: list[ExecutionTrace] = []
        self._model_used: str | None = None

    @property
    def last_traces(self) -> list[ExecutionTrace]:
        """Traces from all cascade attempts (including failed ones)."""
        return self._last_traces

    @property
    def model_used(self) -> str | None:
        """The model that produced the final accepted result."""
        return self._model_used

    async def execute_cascade(
        self,
        parent_call: Any,
        inputs: dict[str, Any],
    ) -> Any:
        """Execute the cascade using the parent Call's full pipeline.

        For each model, creates a temporary Call clone (preserving the parent's
        codec, examples, instructions, settings, kwargs) and invokes its
        ``invoke()`` method. Returns the winning :class:`Invocation` directly
        — the caller (``Call._execute_inner``) returns it up the stack.

        Phase 10: returns ``Invocation`` instead of bare output. The trace
        accumulation reads ``invocation.trace`` from each step's Invocation
        instead of the deleted ``step_call.last_trace`` property.

        Args:
            parent_call: The Call instance that owns this router.
            inputs: Already-validated input field values.

        Returns:
            The :class:`Invocation` from the first model that passes the
            quality check, or the last successful Invocation if no model
            passes.
        """
        from kaos_llm_core.programs._invocation import Invocation
        from kaos_llm_core.programs.call import Call

        # Per-execution scratch on the call stack — concurrent invocations
        # don't share these.
        traces: list[ExecutionTrace] = []
        model_used: str | None = None

        last_invocation: Invocation | None = None
        last_error: Exception | None = None

        for i, model in enumerate(self.models):
            try:
                # Create a temporary Call with this model, reusing parent's config.
                step_call = Call(
                    parent_call.signature,
                    model=model,
                    codec=parent_call._codec,
                    client=parent_call._client,
                    settings=parent_call._settings,
                    core_settings=parent_call._core_settings,
                    examples=parent_call.examples,
                    instructions=parent_call.instructions,
                    max_retries=parent_call._max_retries,
                    hooks=parent_call.hooks,
                    **parent_call._kwargs,
                )

                # Execute through the standard pipeline (retry, trace, etc.).
                # invoke() returns the full Invocation; the result the
                # escalation_check sees is invocation.output.
                step_invocation = await step_call.invoke(**inputs)

                if step_invocation.trace is not None:
                    traces.append(step_invocation.trace)

                last_invocation = step_invocation

                if self.escalation_check(step_invocation.output):
                    model_used = model
                    logger.info(
                        "CascadeRouter: %s accepted from %s (model %d/%d)",
                        parent_call.signature.__name__,
                        model,
                        i + 1,
                        len(self.models),
                    )
                    # Atomic snapshot publish for inspectable router state.
                    self._last_traces = traces
                    self._model_used = model_used
                    return step_invocation

                logger.info(
                    "CascadeRouter: %s escalating from %s (model %d/%d)",
                    parent_call.signature.__name__,
                    model,
                    i + 1,
                    len(self.models),
                )

            except Exception as e:
                # Phase 10: try to capture the partial Invocation tagged on
                # the exception so we can record its trace.
                attached = getattr(e, "invocation", None)
                if attached is not None and getattr(attached, "trace", None) is not None:
                    traces.append(attached.trace)
                else:
                    trace = ExecutionTrace(
                        call_name=parent_call.signature.__name__,
                        model=model,
                        error=str(e),
                    )
                    traces.append(trace)
                last_error = e
                logger.warning(
                    "CascadeRouter: %s failed on %s: %s",
                    parent_call.signature.__name__,
                    model,
                    str(e),
                )

        if last_invocation is not None:
            model_used = self.models[-1]
            self._last_traces = traces
            self._model_used = model_used
            return last_invocation

        # All-fail path: still publish the snapshot so callers can introspect
        # what was attempted before raising.
        self._last_traces = traces
        self._model_used = None
        raise CallError(
            f"CascadeRouter: all {len(self.models)} models failed "
            f"for {parent_call.signature.__name__}. "
            f"Last error: {last_error}. Check model names and API keys.",
        )

    async def select_model(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
        context: Any = None,
    ) -> str:
        """Return the first model (Router protocol compatibility)."""
        return self.models[0]
