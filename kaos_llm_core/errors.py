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
