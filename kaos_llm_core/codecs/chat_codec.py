"""ChatCodec — natural language encoding and decoding with field markers.

Better for open-ended generation where strict JSON is too constraining.
Uses field markers (``[field_name]``) to delineate output sections.
"""

from __future__ import annotations

import re
from typing import Any

from kaos_llm_client import ProviderResponse

from kaos_llm_core.codecs.multimodal import (
    attach_binaries_to_user_message,
    extract_binary_inputs,
)
from kaos_llm_core.errors import CodecError
from kaos_llm_core.signatures.introspection import (
    get_input_fields,
    get_instruction,
    get_output_fields,
)
from kaos_llm_core.signatures.signature import Signature
from kaos_llm_core.types import Example


class ChatCodec:
    """Codec that uses natural language with field markers.

    Encode: builds system prompt with instruction + field descriptions,
    then a user message with input values. Output fields are requested
    using ``[field_name]`` markers.

    Decode: extracts output values from between ``[field_name]`` markers
    in the response text.
    """

    def encode(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
        examples: list[Example] | None = None,
        instructions: str | None = None,
    ) -> list[dict[str, Any]]:
        """Encode into provider messages with field markers.

        Phase 8.2: binary fields (``BinaryData`` subclasses) become
        additional parts of the final user message. Byte-identical to
        pre-Phase-8 when no binary fields are present.
        """
        text_inputs, binary_inputs = extract_binary_inputs(signature, inputs)
        instruction = instructions or get_instruction(signature)
        input_fields = get_input_fields(signature)
        output_fields = get_output_fields(signature)

        # Build system prompt
        system_parts: list[str] = []
        if instruction:
            system_parts.append(instruction)

        system_parts.append("\n## Output Format")
        system_parts.append(
            "Structure your response using these markers. "
            "Each marker MUST appear at the start of a new line on its own. "
            "Place the output value on the lines after the marker:"
        )
        for name, field_info in output_fields.items():
            desc = field_info.description or name
            system_parts.append(f"[{name}] {desc}")

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "\n".join(system_parts)},
        ]

        # Add few-shot examples
        if examples:
            for ex in examples:
                user_content = _format_inputs(ex.inputs, input_fields)
                messages.append({"role": "user", "content": user_content})
                assistant_content = _format_example_output(ex.outputs, output_fields)
                messages.append({"role": "assistant", "content": assistant_content})

        # User message with actual inputs
        user_content = _format_inputs(text_inputs, input_fields)
        messages.append({"role": "user", "content": user_content})

        messages = attach_binaries_to_user_message(messages, binary_inputs)

        return messages

    def decode(
        self,
        signature: type[Signature],
        response: ProviderResponse,
    ) -> dict[str, Any]:
        """Decode field-marker-delimited output from the response.

        Handles markers in any order by scanning for all markers first,
        sorting by position, then extracting content between consecutive
        markers. Attempts JSON parsing for non-string field annotations.
        """
        import json

        text = response.text
        if not text:
            raise CodecError(
                "Empty response from LLM. Check that the model and API key are correct."
            )

        output_fields = get_output_fields(signature)
        result: dict[str, Any] = {}

        # Find all marker positions using line-anchored matching.
        # Markers must appear at the start of a line to avoid collisions
        # with literal [field_name] text inside another field's content.
        marker_positions: list[tuple[int, str]] = []
        for name in output_fields:
            pattern = re.compile(rf"^\[{re.escape(name)}\]", re.MULTILINE)
            match = pattern.search(text)
            if match:
                marker_positions.append((match.start(), name))

        # Sort by position in text
        marker_positions.sort()

        # Extract content between consecutive markers
        for i, (pos, name) in enumerate(marker_positions):
            content_start = pos + len(f"[{name}]")
            content_end = marker_positions[i + 1][0] if i + 1 < len(marker_positions) else len(text)
            raw_value = text[content_start:content_end].strip()

            # Attempt type coercion for non-string annotations
            ann = output_fields[name].annotation
            if ann is not str and ann is not None:
                try:
                    result[name] = json.loads(raw_value)
                    continue
                except (json.JSONDecodeError, ValueError):
                    pass
                # Phase 16.5: list-typed fallback. Models often return
                # list[str] / list[int] / list[float] fields as a
                # comma- or newline-separated string instead of a JSON
                # array. Recover by splitting on common delimiters
                # before falling through to the raw-string assignment.
                coerced = _coerce_to_list(raw_value, ann)
                if coerced is not None:
                    result[name] = coerced
                    continue

            result[name] = raw_value

        # Check for missing required fields
        for name, field_info in output_fields.items():
            if name not in result and field_info.is_required():
                raise CodecError(
                    f"Required output field [{name}] not found in response. "
                    f"Response: {text[:500]!r}"
                )

        return result


def _format_inputs(inputs: dict[str, Any], input_fields: dict[str, Any]) -> str:
    """Format input values as a readable user message."""
    parts: list[str] = []
    for name, value in inputs.items():
        parts.append(f"**{name}**: {value}")
    return "\n\n".join(parts)


def _format_example_output(
    outputs: dict[str, Any],
    output_fields: dict[str, Any],
) -> str:
    """Format example outputs using field markers."""
    parts: list[str] = []
    for name in output_fields:
        if name in outputs:
            parts.append(f"[{name}]\n{outputs[name]}")
    return "\n\n".join(parts)


def _coerce_to_list(raw: str, annotation: Any) -> list[Any] | None:
    """Phase 16.5: list-typed-output fallback for ChatCodec/XMLCodec.

    Models frequently return ``list[str]`` / ``list[int]`` /
    ``list[float]`` fields as a comma- or newline-separated string
    instead of a JSON array. The Phase 16.3 codec regression matrix
    surfaced this as the dominant failure mode for ChatCodec and
    XMLCodec on extract-list / multi-field signatures.

    Returns the coerced list on success, or ``None`` if the annotation
    is not a list type or no recognizable delimiter could be found
    (in which case the caller should fall back to the raw value).

    Element type coercion: ``int`` and ``float`` items are parsed
    individually; non-parseable items poison the whole result and
    return ``None`` (so the caller falls back).
    """
    import typing

    origin = typing.get_origin(annotation)
    if origin not in (list, tuple) and annotation is not list:
        return None

    raw = raw.strip()
    if not raw:
        return []
    # Strip surrounding brackets if the model wrote ``[a, b, c]`` but
    # without proper JSON quoting.
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1].strip()
    if not raw:
        return []

    # Try the most common delimiters in order of specificity. Newlines
    # are tried first because models often emit a one-per-line list
    # under a [field] marker.
    candidates: list[str] = []
    if "\n" in raw:
        candidates = [line.strip() for line in raw.split("\n")]
    elif ";" in raw:
        candidates = [piece.strip() for piece in raw.split(";")]
    elif "," in raw:
        candidates = [piece.strip() for piece in raw.split(",")]
    else:
        # Single-element list, no delimiter
        candidates = [raw]

    # Drop empty pieces and surrounding quotes
    cleaned: list[str] = []
    for piece in candidates:
        piece = piece.strip().strip("\"'")
        if piece:
            cleaned.append(piece)

    if not cleaned:
        return []

    # Element-type coercion based on the inner annotation
    args = typing.get_args(annotation)
    inner = args[0] if args else str
    if inner is int:
        try:
            return [int(p) for p in cleaned]
        except ValueError:
            return None
    if inner is float:
        try:
            return [float(p) for p in cleaned]
        except ValueError:
            return None
    # str (or unknown — pass strings through)
    return cleaned
