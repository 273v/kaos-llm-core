"""Codec helpers for multimodal (binary) signature fields.

Phase 8.2 / §5.2: when a Signature declares an input whose type is a
``BinaryData`` subclass (e.g., :class:`kaos_llm_core.signatures.multimodal.Image`),
the codec:

1. Separates the binary fields from the text/JSON fields.
2. Encodes the text fields as usual (JSON object, field markers, or XML).
3. Emits the final user message as a *multipart* content list, where the
   first part is the text encoding and the remaining parts are the binary
   payloads in ``kaos-llm-client`` content-part format.

When there are zero binary fields, the codec output must be byte-identical
to the pre-Phase-8 behavior — callers verify this with dedicated tests.
The :func:`extract_binary_inputs` helper therefore returns the *same* dict
instance (via a shallow copy when needed) only when binary fields are
actually present.

``kaos-llm-client`` providers consume the ``image_url`` / ``input_audio`` /
``document`` content parts natively and map them to provider-specific
attachment shapes. We never reformat these into provider-specific dicts in
this module; we produce the canonical ``kaos_llm_client.multimodal``
content parts only.
"""

from __future__ import annotations

from typing import Any

from kaos_llm_client.multimodal import (
    audio_input,
    document_from_bytes,
    image_from_bytes,
)
from kaos_llm_client.types import BinaryData

from kaos_llm_core.signatures.multimodal import is_binary_field
from kaos_llm_core.signatures.signature import Signature


def extract_binary_inputs(
    signature: type[Signature],
    inputs: dict[str, Any],
) -> tuple[dict[str, Any], list[tuple[str, BinaryData]]]:
    """Separate binary-typed inputs from the rest.

    Returns ``(text_inputs, binary_inputs)`` where:

    - ``text_inputs`` contains inputs whose Signature field type is NOT a
      ``BinaryData`` subclass. When no binary fields exist, this is the
      original ``inputs`` dict unchanged (identity-preserving so regression
      tests are byte-identical).
    - ``binary_inputs`` is a list of ``(field_name, BinaryData)`` tuples,
      in Signature field-declaration order.

    Non-``BinaryData`` runtime values assigned to a binary field are left
    in ``text_inputs`` (defensive: the validator will fail earlier in a
    well-typed call, but this keeps encode robust).
    """
    binary_field_names: list[str] = []
    for name, info in signature.model_fields.items():
        if is_binary_field(info.annotation):
            binary_field_names.append(name)

    if not binary_field_names:
        return inputs, []

    text_inputs = dict(inputs)
    binary_inputs: list[tuple[str, BinaryData]] = []
    for name in binary_field_names:
        if name not in text_inputs:
            continue
        value = text_inputs[name]
        if isinstance(value, BinaryData):
            binary_inputs.append((name, value))
            text_inputs.pop(name)
        # Otherwise leave it in text_inputs; downstream validation / encoding
        # will surface a clear error.

    return text_inputs, binary_inputs


def binary_to_content_part(bd: BinaryData) -> dict[str, Any]:
    """Convert a :class:`BinaryData` instance to a kaos-llm-client content part.

    Uses the canonical :mod:`kaos_llm_client.multimodal` helpers so the
    emitted shape matches exactly what the provider layer expects (and what
    ``image_from_path`` etc. produce). Audio format is inferred from the
    media subtype.
    """
    raw = bd.to_bytes()
    if bd.is_image:
        return image_from_bytes(raw, media_type=bd.media_type)
    if bd.is_audio:
        # media_type is e.g. "audio/wav" -> format "wav"
        fmt = bd.media_type.split("/", 1)[1] if "/" in bd.media_type else "wav"
        return audio_input(raw, format=fmt)
    # Fall back to document for everything else (PDF, text/*, etc.)
    return document_from_bytes(raw, media_type=bd.media_type)


def attach_binaries_to_user_message(
    messages: list[dict[str, Any]],
    binary_inputs: list[tuple[str, BinaryData]],
) -> list[dict[str, Any]]:
    """Attach binary inputs to the final user message as multipart content.

    The final element of ``messages`` must be a ``{"role": "user", ...}``
    message whose ``content`` is currently a plain string (the text
    encoding of the non-binary fields). This function replaces that
    content with a list of content parts:

    - ``[{"type": "text", "text": <original text>}, <binary parts...>]``

    When ``binary_inputs`` is empty this is a no-op and the messages list
    is returned unchanged (byte-identical regression guarantee).

    **Empty-text edge case (caught by live verification 2026-04-07):** when
    the text encoding of the non-binary inputs is empty (e.g. the Signature
    declares only binary fields), Anthropic rejects the request with
    ``messages: text content blocks must be non-empty``. Other providers
    likely behave similarly. We omit the text part entirely in that case
    so the user message is binary-only, which every provider handles fine.
    """
    if not binary_inputs:
        return messages
    if not messages or messages[-1].get("role") != "user":
        return messages

    user = messages[-1]
    original_content = user.get("content", "")
    parts: list[dict[str, Any]] = []
    # Only emit a text part when the text encoding is non-empty. Some
    # providers (notably Anthropic) reject zero-length text content blocks.
    if isinstance(original_content, str) and original_content.strip():
        parts.append({"type": "text", "text": original_content})
    elif not isinstance(original_content, str) and original_content:
        # Already-multipart content (rare) — keep as-is.
        parts.append({"type": "text", "text": str(original_content)})
    for _name, bd in binary_inputs:
        parts.append(binary_to_content_part(bd))
    user["content"] = parts
    return messages
