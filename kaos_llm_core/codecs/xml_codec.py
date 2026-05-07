"""XMLCodec — XML tag-based encoding and decoding.

Some models (especially Claude) parse XML tags more reliably than JSON
or field markers. XMLCodec wraps each output field in ``<field_name>`` tags
and extracts values from the response using the same tags.

Best for structured extraction tasks where JSON is too rigid and
field markers are too fragile.
"""

from __future__ import annotations

import json
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


class XMLCodec:
    """Codec that uses XML tags for field delineation.

    Encode: system prompt with instruction + field descriptions in XML format,
    user message with input values. Output fields are requested as XML tags.

    Decode: extracts values from between ``<field>...</field>`` tags in the
    response text. Handles CDATA, nested XML, and attempts JSON parsing
    for non-string field annotations.
    """

    def encode(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
        examples: list[Example] | None = None,
        instructions: str | None = None,
    ) -> list[dict[str, Any]]:
        """Encode into provider messages with XML tag instructions.

        Phase 8.2: binary fields (``BinaryData`` subclasses) are routed as
        multipart user-message content. Byte-identical to pre-Phase-8 when
        no binary fields are present.
        """
        text_inputs, binary_inputs = extract_binary_inputs(signature, inputs)
        instruction = instructions or get_instruction(signature)
        input_fields = get_input_fields(signature)
        output_fields = get_output_fields(signature)

        system_parts: list[str] = []
        if instruction:
            system_parts.append(instruction)

        system_parts.append("\n## Output Format")
        system_parts.append(
            "Respond using XML tags. Each output field must be wrapped in its own tag. "
            "Use exactly these tags:"
        )
        for name, field_info in output_fields.items():
            desc = field_info.description or name
            system_parts.append(f"<{name}> {desc} </{name}>")

        system_parts.append(
            "\nProvide your response with each field in its XML tag. "
            "Do not include any text outside the XML tags."
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "\n".join(system_parts)},
        ]

        if examples:
            for ex in examples:
                user_content = _format_inputs(ex.inputs, input_fields)
                messages.append({"role": "user", "content": user_content})
                assistant_content = _format_xml_output(ex.outputs, output_fields)
                messages.append({"role": "assistant", "content": assistant_content})

        user_content = _format_inputs(text_inputs, input_fields)
        messages.append({"role": "user", "content": user_content})

        messages = attach_binaries_to_user_message(messages, binary_inputs)

        return messages

    def decode(
        self,
        signature: type[Signature],
        response: ProviderResponse,
    ) -> dict[str, Any]:
        """Decode XML-tagged output from the response."""
        text = response.text
        if not text:
            raise CodecError(
                "Empty response from LLM. Check that the model and API key are correct."
            )

        output_fields = get_output_fields(signature)
        result: dict[str, Any] = {}

        for name, field_info in output_fields.items():
            pattern = re.compile(
                rf"<{re.escape(name)}>\s*(.*?)\s*</{re.escape(name)}>",
                re.DOTALL,
            )
            match = pattern.search(text)
            if match:
                raw_value = match.group(1).strip()

                # Attempt type coercion for non-string annotations
                ann = field_info.annotation
                if ann is not str and ann is not None:
                    try:
                        result[name] = json.loads(raw_value)
                        continue
                    except (json.JSONDecodeError, ValueError):
                        pass
                    # Phase 16.5: list-typed fallback. See ChatCodec for
                    # the rationale — models often return list[str] /
                    # list[int] / list[float] fields as a comma- or
                    # newline-separated string instead of a JSON array.
                    from kaos_llm_core.codecs.chat_codec import _coerce_to_list

                    coerced = _coerce_to_list(raw_value, ann)
                    if coerced is not None:
                        result[name] = coerced
                        continue

                result[name] = raw_value
            elif field_info.is_required():
                raise CodecError(
                    f"Required output field <{name}> not found in response. "
                    f"Response: {text[:500]!r}. "
                    f"Ensure the model wraps each output in XML tags."
                )

        return result


def _format_inputs(inputs: dict[str, Any], input_fields: dict[str, Any]) -> str:
    """Format input values as a readable user message."""
    parts: list[str] = []
    for name, value in inputs.items():
        parts.append(f"<{name}>{value}</{name}>")
    return "\n".join(parts)


def _format_xml_output(outputs: dict[str, Any], output_fields: dict[str, Any]) -> str:
    """Format example outputs using XML tags."""
    parts: list[str] = []
    for name in output_fields:
        if name in outputs:
            value = outputs[name]
            if isinstance(value, (list, dict)):
                value = json.dumps(value, default=str)
            parts.append(f"<{name}>{value}</{name}>")
    return "\n".join(parts)
