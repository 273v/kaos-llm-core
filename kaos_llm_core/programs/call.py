"""Call — single LLM invocation with a Signature.

The atomic unit of kaos-llm-core. Wraps one LLM request with typed I/O,
codec-based encoding/decoding, validation retry, and execution tracing.

The execution pipeline is decomposed into overridable step methods so that
subclasses (e.g., ChainOfThought) can customize individual steps without
duplicating the orchestration logic (retry, trace, error classification).

Phase 10 redesign: every Call execution returns an :class:`Invocation` —
the canonical runtime contract bundling output, trace, usage, error, and
context. ``Call.__call__`` unwraps to ``invocation.output`` for the
ergonomic bare-output return; ``Call.invoke`` returns the full Invocation
for callers (eval harness, optimizers, hooks) that need trace/usage
access. There is no longer any ``_last_trace`` instance attribute, no
``_current_client`` / ``_current_model`` scratch, no ``_ExecState``
ContextVar, and no ``object.__setattr__(result, "_last_trace", ...)``
mutation. Step methods read execution context via
:func:`current_invocation` against an :data:`_active_invocation_var`
ContextVar that holds the in-flight Invocation for the duration of
``_run_pipeline``.
"""

from __future__ import annotations

import asyncio
import contextvars
import time
from dataclasses import dataclass
from typing import Any

from kaos_core.logging import get_logger
from kaos_llm_client import (
    BaseProviderClient,
    ProviderResponse,
    create_client,
)
from kaos_llm_client.settings import KaosLLMSettings
from pydantic import BaseModel

from kaos_llm_core.codecs.base import Codec
from kaos_llm_core.codecs.json_codec import JSONCodec
from kaos_llm_core.errors import CallError, ValidationRetryExhaustedError
from kaos_llm_core.observability.collectors import push_trace
from kaos_llm_core.observability.cost import apply_cost_estimates
from kaos_llm_core.observability.traces import ExecutionTrace
from kaos_llm_core.optimization.trial_runner import publish_invocation
from kaos_llm_core.programs._invocation import (
    Invocation,
    TokenUsage,
    _active_invocation_var,
    current_invocation,
)
from kaos_llm_core.programs.hooks import CallHooks, fire_hook
from kaos_llm_core.settings import KaosLLMCoreSettings
from kaos_llm_core.signatures.introspection import (
    create_output_model,
    get_input_fields,
    get_instruction,
)
from kaos_llm_core.signatures.signature import Signature
from kaos_llm_core.types import Example

logger = get_logger(__name__)


# Per-task KaosContext for hook correlation. Set by callers (typically the
# MCP tool wrapper at execution time) so :func:`fire_hook` can pass it to
# user callbacks via the ``context=`` kwarg. Defaults to ``None`` so existing
# Python-only callers see the same behavior they always have.
_call_context_var: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "kaos_llm_core_call_context",
    default=None,
)


@dataclass(frozen=True, slots=True)
class CallPlan:
    """The fully-resolved bundle a Call needs to issue one provider request.

    Returned by :meth:`Call.prepare_call`. Composing programs (``ReAct``,
    custom routers, etc.) consume this instead of reaching into Call's
    private step methods. ``CallPlan`` is a stable, public contract — the
    private ``_resolve_model`` / ``_get_effective_signature`` / etc. methods
    remain the *override* surface for subclasses (``ChainOfThought``), while
    ``prepare_call`` is the *consumption* surface for callers.

    Attributes:
        validated_inputs: Inputs after type-checking and default materialization.
        model: The provider-prefixed model id (e.g. ``"anthropic:claude-haiku-4-5"``).
        client: A live :class:`BaseProviderClient` ready to call ``chat_async``.
        codec: The codec used to encode/decode this Call's wire format.
        effective_signature: The :class:`Signature` to encode for this turn
            (may differ from ``call.signature`` for subclasses like
            ``ChainOfThought`` that flip between native-thinking and
            prompt-based modes).
        effective_instructions: The system instruction text the codec will plumb
            into the prompt.
        output_model: The Pydantic model used to validate decoded outputs.
        call_kwargs: Extra kwargs forwarded to ``client.chat_async`` (e.g.
            ``temperature``, ``thinking``).
    """

    validated_inputs: dict[str, Any]
    model: str
    client: BaseProviderClient
    codec: Codec
    effective_signature: type[Signature]
    effective_instructions: str
    output_model: type[BaseModel]
    call_kwargs: dict[str, Any]


def set_call_context(context: Any) -> contextvars.Token[Any]:
    """Set the KaosContext seen by hook callbacks during the next Call.

    Returns a ``Token`` that the caller is expected to ``reset`` once the
    Call completes. Tool wrappers (kaos-llm-core ``KaosTool.execute``)
    use this to make ``KaosContext`` available to hook callbacks for
    session/trace correlation.
    """
    return _call_context_var.set(context)


class Call:
    """Single LLM invocation with a typed Signature.

    The execution pipeline:

    1. ``_validate_inputs()`` — check types, materialize defaults
    2. ``_resolve_model()`` / ``_resolve_client()`` — pick model + client
    3. ``_prepare_call_kwargs()`` — build kwargs for chat_async
    4. ``_get_effective_signature()`` / ``_get_effective_instructions()`` — what to encode
    5. ``codec.encode(...)`` — build messages
    6. ``client.chat_async(...)`` — call LLM (provider failure → immediate CallError)
    7. ``codec.decode(...)`` + ``output_model.model_validate(...)`` —
       parse/validate (retry on failure)
    8. ``_post_process()`` — subclass hook to modify result

    Subclasses override step methods, not ``_run_pipeline()``.
    """

    def __init__(
        self,
        signature: type[Signature],
        *,
        model: str | None = None,
        codec: Codec | None = None,
        client: BaseProviderClient | None = None,
        router: Any = None,
        settings: KaosLLMSettings | None = None,
        core_settings: KaosLLMCoreSettings | None = None,
        examples: list[Example] | None = None,
        instructions: str | None = None,
        max_retries: int | None = None,
        hooks: CallHooks | None = None,
        **kwargs: Any,
    ) -> None:
        self.signature = signature
        self.hooks = hooks
        self._model = model
        self._codec = codec or JSONCodec()
        self._client = client
        self._router = router
        self._settings = settings
        self._core_settings = core_settings or KaosLLMCoreSettings()
        self.examples: list[Example] = list(examples) if examples else []
        self.instructions: str = instructions or get_instruction(signature)
        self._max_retries = (
            max_retries if max_retries is not None else self._core_settings.default_max_retries
        )
        self._kwargs = kwargs
        self._output_model = create_output_model(signature)

    @property
    def codec(self) -> Codec:
        """Read-only accessor for the codec.

        Public so composing programs (``ReAct``, custom routers) can name the
        codec in their own trace metadata without reaching into ``_codec``.
        """
        return self._codec

    @property
    def hyperparameters(self) -> dict[str, Any]:
        """Read-only snapshot of the generation kwargs (``temperature`` etc.).

        Returns a copy so callers cannot mutate the underlying dict by
        accident. ``BestOfN`` and other samplers consume this through
        :meth:`prepare_call` rather than touching ``_kwargs`` directly.
        """
        return dict(self._kwargs)

    # ---------------------------------------------------------------
    # Per-execution scratch — read by step methods via current_invocation()
    # ---------------------------------------------------------------

    def _build_invocation_extras(self, *, client: Any, model: str) -> dict[str, Any]:
        """Subclass hook: populate ``Invocation.extras`` for this execution.

        Override to add subclass-specific keys that step methods need.
        ``ChainOfThought`` populates ``"native_thinking"`` so its
        ``_prepare_call_kwargs`` / ``_get_effective_signature`` can switch
        behavior without writing instance state. Returns a fresh dict on
        every call (no shared mutable default).
        """
        return {}

    async def prepare_call(self, inputs: dict[str, Any]) -> CallPlan:
        """Resolve everything needed to issue one provider request.

        Composing programs that need to drive their own loop (``ReAct`` is the
        canonical example) call this once at the top of execution and read the
        returned :class:`CallPlan`. This replaces eight separate underscore
        reach-throughs with one public, typed contract.

        Subclasses (``ChainOfThought``) keep customizing individual steps by
        overriding the underscore step methods this delegates to. Subclasses
        with **per-execution scratch** (e.g. ``ChainOfThought``'s
        native-thinking detection) override :meth:`_build_invocation_extras`
        to populate ``Invocation.extras``; their step methods then read
        from :func:`current_invocation` instead of instance attributes.

        ``prepare_call`` constructs a transient Invocation around the step
        methods so a step method that reads ``current_invocation()`` sees
        consistent state during plan building. The Invocation is discarded
        when ``prepare_call`` returns — only the ``CallPlan`` is kept.

        Args:
            inputs: Raw input dict, as the user would pass to ``__call__``.

        Returns:
            A :class:`CallPlan` with the resolved model, client, codec,
            effective signature, instructions, output model, and call kwargs.
        """
        validated = self._validate_inputs(inputs)
        model = await self._resolve_model(validated)
        client = self._resolve_client(model)
        plan_invocation = Invocation(
            client=client,
            model=model,
            context=_call_context_var.get(),
            extras=self._build_invocation_extras(client=client, model=model),
        )
        token = _active_invocation_var.set(plan_invocation)
        try:
            return CallPlan(
                validated_inputs=validated,
                model=model,
                client=client,
                codec=self._codec,
                effective_signature=self._get_effective_signature(),
                effective_instructions=self._get_effective_instructions(),
                output_model=self._get_output_model(),
                call_kwargs=self._prepare_call_kwargs(),
            )
        finally:
            _active_invocation_var.reset(token)

    async def __call__(self, **inputs: Any) -> Any:
        """Execute the Call and return the bare output.

        Ergonomic public surface. Equivalent to
        ``(await self.invoke(**inputs)).output`` but slightly more
        efficient.

        **When to choose this vs :meth:`invoke`:**

        * Use ``await call(**inputs)`` for read-only call sites where
          the caller doesn't need the trace, usage, or per-call cost
          at the call site itself. Cost rollups still flow through
          the active :class:`Invocation` ContextVar (set by an
          enclosing ``Program`` / ``Refine`` / ``BestOfN`` /
          ``TrialRunner``); they just aren't visible at this exact
          line.
        * Use ``await call.invoke(**inputs)`` when the caller needs
          to *attribute* the cost / tokens / trace at this site —
          e.g., to plumb into ``TurnComplete.usage``, a
          ``--max-cost`` ceiling, or a ``PlanBudget``. The
          architecture audit (``kaos-agents/docs/design/architecture-
          audit-raw-llm-calls.md``) found 5 sites in kaos-agents
          where ``__call__`` was used at a site that NEEDED
          attribution — those got migrated to ``.invoke()``.

        If a caller cannot decide which they need, default to
        ``.invoke()``. Throwing away a typed Invocation is cheap;
        recovering one after the fact isn't possible.
        """
        invocation = await self._execute(inputs)
        return invocation.output

    async def invoke(self, **inputs: Any) -> Invocation:
        """Execute the Call and return the full :class:`Invocation` record.

        The runtime contract for callers that need access to ``trace``,
        ``usage``, ``error``, or ``context`` alongside the output. Eval
        harness, optimizers, hooks, and the cascade router all consume
        Invocations directly via this method.

        Raises the same exceptions ``__call__`` does (``CallError``,
        ``ValidationRetryExhaustedError``); the partial Invocation is
        attached to the raised exception as ``exc.invocation`` so the
        caller can retrieve it from an ``except`` block.
        """
        return await self._execute(inputs)

    def __call_sync__(self, **inputs: Any) -> Any:
        """Synchronous execution wrapper. Returns bare output."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            async def _runner() -> Any:
                inv = await self._execute(inputs)
                return inv.output

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, _runner()).result()

        async def _runner_top() -> Any:
            inv = await self._execute(inputs)
            return inv.output

        return asyncio.run(_runner_top())

    # ---------------------------------------------------------------
    # Overridable step methods
    # ---------------------------------------------------------------

    def _validate_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Validate inputs and materialize defaults. Override for custom validation."""
        input_fields = get_input_fields(self.signature)
        required = {n for n, f in input_fields.items() if f.is_required()}
        allowed = set(input_fields.keys())

        missing = required - set(inputs.keys())
        if missing:
            raise CallError(
                f"Call({self.signature.__name__}) missing required inputs: {sorted(missing)}. "
                f"Expected: {sorted(allowed)}. Got: {sorted(inputs.keys())}."
            )

        extra = set(inputs.keys()) - allowed
        if extra:
            raise CallError(
                f"Call({self.signature.__name__}) received unexpected inputs: {sorted(extra)}. "
                f"Allowed inputs: {sorted(allowed)}."
            )

        # Materialize defaults for omitted optional fields
        validated = dict(inputs)
        for name, field_info in input_fields.items():
            if name not in validated and not field_info.is_required():
                validated[name] = field_info.get_default(call_default_factory=True)

        # Type-check via a Pydantic model built from input fields only
        from pydantic import create_model

        input_model_fields: dict[str, Any] = {}
        for name, field_info in input_fields.items():
            input_model_fields[name] = (field_info.annotation, field_info)
        input_model = create_model(f"{self.signature.__name__}InputValidator", **input_model_fields)
        try:
            input_model.model_validate(validated)
        except Exception as e:
            raise CallError(
                f"Call({self.signature.__name__}) input validation failed: {e}. "
                f"Check that input types match the Signature's field annotations."
            ) from e

        return validated

    def _prepare_call_kwargs(self) -> dict[str, Any]:
        """Build extra kwargs for ``client.chat_async()``. Override to inject params."""
        return dict(self._kwargs)

    def _get_effective_signature(self) -> type[Signature]:
        """The signature to use for encoding/decoding. Override for dynamic selection."""
        return self.signature

    def _get_effective_instructions(self) -> str:
        """The instruction text to use for encoding. Override for dynamic selection."""
        return self.instructions

    def _get_output_model(self) -> type[BaseModel]:
        """The Pydantic model for output validation. Override for dynamic selection."""
        return self._output_model

    def _post_process(
        self, result: Any, response: ProviderResponse, output_dict: dict[str, Any]
    ) -> Any:
        """Hook called after successful decode+validate. Override to modify result."""
        return result

    # ---------------------------------------------------------------
    # Orchestration — NOT meant to be overridden
    # ---------------------------------------------------------------

    async def _execute(self, inputs: dict[str, Any]) -> Invocation:
        """Core execution wrapper. Owns hook firing + active-collector push.

        Returns the finalized :class:`Invocation`. On exception, the partial
        Invocation is attached to ``exc.invocation`` and the exception is
        re-raised; this is the path the eval harness uses to capture failure
        traces from an ``except`` block without an extra ContextVar.

        Hook integration: ``self.hooks`` (a :class:`CallHooks`) wraps this
        method. ``on_call_start`` fires at the top, ``on_call_end`` fires on
        success with the final Invocation, ``on_call_error`` fires on
        exception. Hooks that raise are logged and ignored.
        """
        ctx = _call_context_var.get()
        fire_hook(
            self.hooks.on_call_start if self.hooks else None,
            self,
            inputs,
            context=ctx,
        )
        invocation: Invocation | None = None
        try:
            try:
                invocation = await self._execute_inner(inputs)
            except Exception as exc:
                # Tag the exception with the partial Invocation if we got far
                # enough to construct one. The eval harness reads this from
                # its except block to capture failure traces.
                if invocation is None:
                    invocation = getattr(exc, "invocation", None)
                fire_hook(
                    self.hooks.on_call_error if self.hooks else None,
                    self,
                    inputs,
                    exc,
                    context=ctx,
                )
                raise
            fire_hook(
                self.hooks.on_call_end if self.hooks else None,
                self,
                inputs,
                invocation,
                context=ctx,
            )
            return invocation
        finally:
            # Push the just-completed trace into the active collector (if any).
            # MUST run on both success and failure so a failing leaf Call
            # appears as a child of the parent Program's trace tree.
            if invocation is not None and invocation.trace is not None:
                push_trace(invocation.trace)

    async def _execute_inner(self, inputs: dict[str, Any]) -> Invocation:
        """Inner execution body. Hooks fire in :meth:`_execute` around this."""
        validated_inputs = self._validate_inputs(inputs)

        # CascadeRouter delegation. The router runs its own per-step Calls
        # via Call.invoke() and returns the winning Invocation directly.
        from kaos_llm_core.router.cascade import CascadeRouter

        if isinstance(self._router, CascadeRouter):
            return await self._router.execute_cascade(self, validated_inputs)

        model = await self._resolve_model(validated_inputs)
        client = self._resolve_client(model)
        return await self._run_pipeline(
            validated_inputs=validated_inputs,
            model=model,
            client=client,
        )

    async def _run_pipeline(
        self,
        *,
        validated_inputs: dict[str, Any],
        model: str,
        client: BaseProviderClient,
    ) -> Invocation:
        """Run the encode → request → decode → validate → retry pipeline.

        Constructs the in-flight Invocation, sets it as the active value of
        ``_active_invocation_var``, runs the pipeline, finalizes the
        Invocation, and returns it. Step methods (and any subclass overrides)
        read execution context via :func:`current_invocation`.

        On failure, the exception is tagged with ``exc.invocation = invocation``
        before being re-raised so the caller's ``except`` block can read the
        partial Invocation.
        """
        ctx = _call_context_var.get()
        invocation = Invocation(
            client=client,
            model=model,
            context=ctx,
            extras=self._build_invocation_extras(client=client, model=model),
        )
        invocation_token = _active_invocation_var.set(invocation)

        try:
            trace_enabled = self._core_settings.trace_enabled
            effective_sig = self._get_effective_signature()
            output_model = self._get_output_model()
            call_kwargs = self._prepare_call_kwargs()

            if trace_enabled:
                invocation.trace = ExecutionTrace(
                    call_name=self.signature.__name__,
                    signature=self.signature.__name__,
                    inputs=validated_inputs,
                    model=model,
                    codec=type(self._codec).__name__,
                    examples_used=len(self.examples),
                )

            start_time = time.monotonic()
            last_validation_error: Exception | None = None
            attempts = 0

            messages = self._codec.encode(
                effective_sig,
                validated_inputs,
                examples=self.examples or None,
                instructions=self._get_effective_instructions(),
            )

            for attempt in range(self._max_retries + 1):
                attempts = attempt + 1

                # Provider call — NOT retried on failure
                try:
                    response: ProviderResponse = await client.chat_async(messages, **call_kwargs)
                except Exception as e:
                    self._finalize_trace_on_provider_error(
                        invocation=invocation,
                        exc=e,
                        start_time=start_time,
                    )
                    err = CallError(
                        f"Call to {self.signature.__name__} failed: {e}. "
                        f"Check model ({model}) and API key configuration.",
                    )
                    invocation.error = err
                    err.invocation = invocation  # ty: ignore[unresolved-attribute]
                    # Phase 13A: charge any tokens already consumed before
                    # the provider raised. Many providers bill on attempt,
                    # not on success.
                    publish_invocation(invocation)
                    raise err from e

                # Real provider work happened — accumulate usage regardless of
                # whether decode/validate succeeds, so retry-attempt tokens
                # are always counted.
                invocation.usage.input_tokens += response.usage.input_tokens
                invocation.usage.output_tokens += response.usage.output_tokens
                invocation.usage.total_tokens += response.usage.total_tokens

                # Decode + validate — IS retried
                try:
                    output_dict = self._codec.decode(effective_sig, response)
                    result = output_model.model_validate(output_dict)
                    result = self._post_process(result, response, output_dict)

                    self._finalize_trace_on_success(
                        invocation=invocation,
                        output_dict=output_dict,
                        attempts=attempt,
                        start_time=start_time,
                    )
                    invocation.output = result
                    # Phase 13A: charge this completed Invocation to the
                    # active TrialRunner trial (no-op outside any trial
                    # scope). The publish must run AFTER
                    # ``_finalize_trace_on_success`` because that helper
                    # populates ``invocation.usage.cost_usd`` from the
                    # per-model pricing table.
                    publish_invocation(invocation)
                    return invocation

                except Exception as e:
                    last_validation_error = e
                    if attempt < self._max_retries:
                        messages.append({"role": "assistant", "content": response.text})
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"Your previous response could not be parsed: {e}. "
                                    f"Please try again and ensure your output matches "
                                    f"the requested format exactly."
                                ),
                            }
                        )
                        logger.warning(
                            "Validation retry %d/%d for %s: %s",
                            attempt + 1,
                            self._max_retries,
                            self.signature.__name__,
                            str(e),
                        )
                        fire_hook(
                            self.hooks.on_validation_retry if self.hooks else None,
                            self,
                            validated_inputs,
                            attempt + 1,
                            e,
                            context=ctx,
                        )

            # All validation retries exhausted
            self._finalize_trace_on_validation_exhaustion(
                invocation=invocation,
                last_validation_error=last_validation_error,
                attempts=attempts,
                start_time=start_time,
            )
            err = ValidationRetryExhaustedError(
                f"All {attempts} validation attempts failed for {self.signature.__name__}. "
                f"Last error: {last_validation_error}. "
                f"Try using a model that supports native structured output, "
                f"or simplify the output schema.",
                attempts=attempts,
                last_error=last_validation_error,
            )
            invocation.error = err
            err.invocation = invocation  # ty: ignore[unresolved-attribute]
            # Phase 13A: even on the failure path the trial still pays
            # for the tokens the provider already consumed. Publish
            # before raising so a Reflective optimizer that catches the
            # error sees the partial cost.
            publish_invocation(invocation)
            raise err
        finally:
            _active_invocation_var.reset(invocation_token)

    @staticmethod
    def _finalize_trace_on_success(
        *,
        invocation: Invocation,
        output_dict: dict[str, Any],
        attempts: int,
        start_time: float,
    ) -> None:
        """Populate trace + cost on the success path."""
        if invocation.trace is None:
            return
        invocation.trace.outputs = output_dict
        invocation.trace.input_tokens = invocation.usage.input_tokens
        invocation.trace.output_tokens = invocation.usage.output_tokens
        invocation.trace.total_tokens = invocation.usage.total_tokens
        invocation.trace.retries = attempts
        invocation.trace.latency_ms = (time.monotonic() - start_time) * 1000
        apply_cost_estimates(invocation.trace)
        invocation.usage.cost_usd = invocation.trace.cost_usd

    @staticmethod
    def _finalize_trace_on_provider_error(
        *,
        invocation: Invocation,
        exc: Exception,
        start_time: float,
    ) -> None:
        """Populate trace + cost on the provider-error path."""
        if invocation.trace is None:
            return
        invocation.trace.input_tokens = invocation.usage.input_tokens
        invocation.trace.output_tokens = invocation.usage.output_tokens
        invocation.trace.total_tokens = invocation.usage.total_tokens
        invocation.trace.error = str(exc)
        invocation.trace.latency_ms = (time.monotonic() - start_time) * 1000
        apply_cost_estimates(invocation.trace)
        invocation.usage.cost_usd = invocation.trace.cost_usd

    @staticmethod
    def _finalize_trace_on_validation_exhaustion(
        *,
        invocation: Invocation,
        last_validation_error: Exception | None,
        attempts: int,
        start_time: float,
    ) -> None:
        """Populate trace + cost on the validation-retry-exhausted path."""
        if invocation.trace is None:
            return
        invocation.trace.input_tokens = invocation.usage.input_tokens
        invocation.trace.output_tokens = invocation.usage.output_tokens
        invocation.trace.total_tokens = invocation.usage.total_tokens
        invocation.trace.error = str(last_validation_error)
        invocation.trace.retries = attempts
        invocation.trace.latency_ms = (time.monotonic() - start_time) * 1000
        apply_cost_estimates(invocation.trace)
        invocation.usage.cost_usd = invocation.trace.cost_usd

    async def _resolve_model(self, inputs: dict[str, Any] | None = None) -> str:
        """Resolve the model to use. Priority: explicit → router → default."""
        if self._model:
            return self._model
        if self._router is not None:
            return await self._router.select_model(self.signature, inputs or {}, None)
        if self._core_settings.default_model:
            return self._core_settings.default_model
        raise CallError(
            f"No model specified for Call({self.signature.__name__}). "
            f"Set model= on the Call, set router=, or set KAOS_LLM_CORE_DEFAULT_MODEL env var."
        )

    def _resolve_client(self, model: str) -> BaseProviderClient:
        """Resolve or create the LLM client."""
        if self._client is not None:
            return self._client
        return create_client(model, settings=self._settings)

    # ---------------------------------------------------------------
    # Learnable state
    # ---------------------------------------------------------------

    # Hyperparameter keys persisted via learnable state. Anything else in
    # ``self._kwargs`` is ignored — only these well-known generation params
    # round-trip through save()/load(). This list intentionally matches the
    # parameters routinely tuned by HyperparameterOptimizer.
    _PERSISTED_HYPERPARAMETERS = (
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "presence_penalty",
        "frequency_penalty",
        "seed",
    )

    def get_learnable_state(self) -> dict[str, Any]:
        """Return the learnable state (for optimizer save/load).

        v2 schema:

        - ``instructions``: system instruction text
        - ``examples``: few-shot examples
        - ``hyperparameters``: dict of generation params from ``self._kwargs``,
          filtered to ``_PERSISTED_HYPERPARAMETERS``.
        - ``codec``: fully-qualified class name of ``self._codec``.
        - ``model``: explicit model id if one was set on the Call.
        - ``router``: serialized router state.
        """
        from kaos_llm_core.router.serialization import router_to_state

        state: dict[str, Any] = {
            "instructions": self.instructions,
            "examples": [ex.model_dump() for ex in self.examples],
            "hyperparameters": {
                k: self._kwargs[k] for k in self._PERSISTED_HYPERPARAMETERS if k in self._kwargs
            },
            "codec": f"{type(self._codec).__module__}.{type(self._codec).__name__}",
            "model": self._model,
            "router": router_to_state(self._router),
        }
        return state

    def set_learnable_state(self, state: dict[str, Any]) -> None:
        """Restore learnable state from a dict."""
        if "instructions" in state:
            self.instructions = state["instructions"]
        if "examples" in state:
            self.examples = [Example.model_validate(ex) for ex in state["examples"]]
        if "hyperparameters" in state:
            for key in self._PERSISTED_HYPERPARAMETERS:
                self._kwargs.pop(key, None)
            self._kwargs.update(state["hyperparameters"])
        if state.get("codec"):
            new_codec = _resolve_codec_class(state["codec"])
            if new_codec is not None:
                self._codec = new_codec()
            else:
                logger.warning(
                    "Codec '%s' from saved state could not be resolved; "
                    "keeping current codec %s. To restore the original codec, "
                    "ensure the codec class is importable.",
                    state["codec"],
                    type(self._codec).__name__,
                )
        if "model" in state and state["model"] is not None:
            self._model = state["model"]
        if state.get("router"):
            from kaos_llm_core.router.serialization import router_from_state

            restored = router_from_state(state["router"])
            if restored is not None:
                self._router = restored


# Codec resolution is restricted to first-party modules so that
# ``Call.set_learnable_state`` (reached via ``Program.load`` from a JSON
# envelope) cannot trigger ``importlib.import_module`` of arbitrary
# modules supplied by an attacker. Module-level code runs at import
# time, so the allowlist check must happen *before* the import. Any
# new official codec module belongs under ``kaos_llm_core.codecs.``;
# the prefix check is sufficient to guarantee that.
_ALLOWED_CODEC_MODULE_PREFIXES: tuple[str, ...] = ("kaos_llm_core.codecs.",)


def _resolve_codec_class(qualified_name: str) -> type[Codec] | None:
    """Resolve a fully-qualified codec class name to its class object.

    Only modules under ``kaos_llm_core.codecs.`` are resolvable; this
    prevents ``set_learnable_state`` from importing arbitrary modules
    via a saved-state JSON payload.
    """
    module_path, _, class_name = qualified_name.rpartition(".")
    if not module_path or not class_name:
        return None
    if not any(module_path.startswith(prefix) for prefix in _ALLOWED_CODEC_MODULE_PREFIXES):
        logger.warning(
            "_resolve_codec_class: refusing to import codec from non-allowlisted "
            "module %r (allowed prefixes: %s)",
            module_path,
            ", ".join(_ALLOWED_CODEC_MODULE_PREFIXES),
        )
        return None
    try:
        import importlib

        module = importlib.import_module(module_path)
        cls = getattr(module, class_name, None)
        if cls is None or not isinstance(cls, type):
            return None
        return cls  # type: ignore[return-value]
    except Exception as exc:
        logger.debug("_resolve_codec_class: failed to resolve %r: %s", qualified_name, exc)
        return None


# Re-exports for callers that import from this module
__all__ = [
    "Call",
    "CallPlan",
    "Invocation",
    "TokenUsage",
    "current_invocation",
    "set_call_context",
]
