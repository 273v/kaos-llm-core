"""Router protocol — interface for model selection strategies."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from kaos_llm_core.signatures.signature import Signature


@runtime_checkable
class Router(Protocol):
    """Protocol for model selection strategies.

    A Router decides which model should handle a given Call based on
    the Signature, inputs, and optional context.
    """

    async def select_model(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
        context: Any = None,
    ) -> str:
        """Select a model identifier for the given call.

        Args:
            signature: The Signature class being called.
            inputs: The input field values.
            context: Optional context (e.g., KaosContext).

        Returns:
            A model identifier string (e.g., "anthropic:claude-sonnet-4-6").
        """
        ...
