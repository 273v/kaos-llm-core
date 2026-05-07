"""Codec protocol — the interface for bidirectional format translation.

A Codec encodes (Signature + inputs → messages) and decodes (response → typed outputs).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from kaos_llm_client import ProviderResponse

from kaos_llm_core.signatures.signature import Signature
from kaos_llm_core.types import Example


@runtime_checkable
class Codec(Protocol):
    """Protocol for bidirectional LLM message format translation.

    Encode direction:
        Signature + inputs + examples + instructions → provider messages

    Decode direction:
        ProviderResponse → typed output dict matching Signature output fields
    """

    def encode(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
        examples: list[Example] | None = None,
        instructions: str | None = None,
    ) -> list[dict[str, Any]]:
        """Encode a Signature call into provider messages.

        Args:
            signature: The Signature class defining I/O contract.
            inputs: Input field values.
            examples: Optional few-shot examples.
            instructions: Optional instruction override (default: Signature docstring).

        Returns:
            A list of message dicts compatible with kaos-llm-client.
        """
        ...

    def decode(
        self,
        signature: type[Signature],
        response: ProviderResponse,
    ) -> dict[str, Any]:
        """Decode a provider response into typed output values.

        Args:
            signature: The Signature class defining expected outputs.
            response: The raw provider response.

        Returns:
            A dict mapping output field names to their parsed values.

        Raises:
            CodecError: If the response cannot be decoded.
        """
        ...
