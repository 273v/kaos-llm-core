"""CallHooks — lifecycle hooks fired around :meth:`Call._execute`.

Mirrors :class:`kaos_llm_client.types.RequestHooks` at the Call (logical)
layer. ``RequestHooks`` fires around each individual HTTP request;
``CallHooks`` fires around the entire logical call (one or more HTTP requests
plus validation retries).

Per the roadmap §5.5 nesting order::

    CallHooks.on_call_start         # logical start
      Call._validate_inputs
      Call._resolve_model / _resolve_client
      Call._prepare_call_kwargs
      codec.encode
        RequestHooks.on_request     # HTTP boundary (in kaos-llm-client)
        client.chat_async
        RequestHooks.on_response
      codec.decode + validate
      Call._post_process
    CallHooks.on_call_end           # logical end
    CallHooks.on_call_error         # if Call._execute raises

Cost and token telemetry: ``RequestHooks`` reports per-HTTP-request usage;
``CallHooks`` reports the **sum** across the validation-retry loop. Users
who want raw transport telemetry use :class:`RequestHooks`; users who want
logical-call telemetry use :class:`CallHooks`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from kaos_core.logging import get_logger

logger = get_logger(__name__)


HookCallback = Callable[..., Any]


@dataclass
class CallHooks:
    """Optional lifecycle callbacks fired around ``Call._execute()``.

    All hooks are optional. A hook that raises does **not** crash the Call —
    the exception is caught, logged at WARNING level, and execution continues.

    Phase 10 hook signatures (the on_call_end hook receives the full
    :class:`Invocation` instead of separate result/trace positional args):

    - ``on_call_start(call, inputs, *, context=None)``
    - ``on_call_end(call, inputs, invocation, *, context=None)``
    - ``on_call_error(call, inputs, exception, *, context=None)``
    - ``on_validation_retry(call, inputs, attempt, error, *, context=None)``

    The ``context`` kwarg is always passed by the firing site (it may be
    ``None`` when no context is in scope). :func:`fire_hook` retries the
    invocation without ``context=`` on ``TypeError`` to support callbacks
    that don't yet accept it — that's a one-line ergonomic shim, not a
    backwards-compat layer.
    """

    on_call_start: HookCallback | None = None
    on_call_end: HookCallback | None = None
    on_call_error: HookCallback | None = None
    on_validation_retry: HookCallback | None = None


def fire_hook(
    hook: HookCallback | None,
    *args: Any,
    context: Any = None,
    **kwargs: Any,
) -> None:
    """Invoke a hook callback, swallowing exceptions.

    A hook that raises is logged and ignored — hooks must never affect the
    normal control flow of a Call.

    The ``context`` kwarg carries an optional :class:`KaosContext` so
    observers can correlate hook events with ``session_id`` / ``trace_id``.
    Older callbacks that don't accept ``context`` are detected by trying
    once with the kwarg and falling back to the no-context invocation on
    ``TypeError``. This preserves the §5.5 design decision while keeping
    backward compatibility for callbacks already in the wild.
    """
    if hook is None:
        return
    try:
        try:
            hook(*args, context=context, **kwargs)
        except TypeError as te:
            # Distinguish "callback rejects context kwarg" from "callback's
            # body raised TypeError". The former matches by message contents.
            msg = str(te)
            if "context" in msg and (
                "unexpected keyword argument" in msg or "got multiple values" in msg
            ):
                hook(*args, **kwargs)
            else:
                raise
    except Exception as exc:
        logger.warning(
            "CallHooks callback %r raised %s; continuing. "
            "Hooks must not raise; wrap your callback in a try/except.",
            getattr(hook, "__name__", repr(hook)),
            exc,
        )
