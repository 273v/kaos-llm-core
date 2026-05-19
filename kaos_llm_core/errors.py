"""Exception hierarchy for kaos-llm-core.

All exceptions inherit ``KaosCoreError(message, **details)`` so that structured
context flows through for agent-friendly error messages.
"""

from __future__ import annotations

from typing import Any

from kaos_core.exceptions import KaosCoreError


class KaosLLMCoreError(KaosCoreError):
    """Base for all kaos-llm-core errors."""


class SignatureError(KaosLLMCoreError):
    """Invalid Signature definition.

    Raised when:
    - A Signature has no OutputFields
    - Field types are unsupported
    - Docstring is missing (no instruction)
    """


class CodecError(KaosLLMCoreError):
    """Codec encode or decode failure.

    Raised when:
    - Response cannot be parsed into expected output format
    - JSON extraction fails
    - Field markers not found in response
    """


class CallError(KaosLLMCoreError):
    """Call execution failure.

    Raised when:
    - Model resolution fails (no model specified and no default)
    - Client creation fails
    - Provider returns an error after retries
    """


class ValidationRetryExhaustedError(CallError):
    """All validation retries exhausted.

    The LLM returned responses that failed output validation on every attempt.
    """

    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        last_error: Exception | None = None,
        **details: Any,
    ) -> None:
        super().__init__(message, attempts=attempts, **details)
        self.attempts = attempts
        if last_error is not None:
            self.__cause__ = last_error


class ToolReportedError(KaosLLMCoreError):
    """A tool reported a structured error and wants ReAct to propagate it.

    Raised by a tool's executor (e.g. the kaos-agents tool bridge wrapping a
    ``ToolResult`` with ``isError=True``) when it wants ReAct's dispatcher to
    record ``is_error=True`` on the resulting :class:`ToolObservation` AND
    preserve the tool's own structured payload as the observation result —
    not the generic ``"Tool 'X' raised: ..."`` wrapper applied to unhandled
    exceptions.

    The difference between this and a bare ``Exception``:

    - Bare ``Exception`` → ReAct reports ``is_error=True`` AND replaces the
      payload with a generic ``"Tool 'X' raised: ..."`` wrapper. The model
      loses the tool's remediation hint.
    - ``ToolReportedError`` → ReAct reports ``is_error=True`` AND preserves
      the tool's own ``payload`` (typically ``{"error": True, "message":
      "<remediation>"}``). The model sees the actionable error AND
      downstream observability (kaos-agents memory, audit CLI, UI chips,
      critics) sees ``is_error=True``.

    This closes the audit-finding-#3 gap that previously forced a binary
    choice between "preserve payload (lose error flag)" and "raise exception
    (lose payload)". With this exception both signals propagate.

    Args:
        payload: The structured error payload to forward to the model. Most
            commonly ``{"error": True, "message": "<text>"}``; may be any
            JSON-serializable value the tool wants the model to reason over.
        message: Optional human-readable message used by the default
            ``str(exc)`` rendering. Defaults to the stringified payload.
    """

    def __init__(self, payload: Any, *, message: str | None = None) -> None:
        super().__init__(message or str(payload))
        self.payload = payload
