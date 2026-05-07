"""JSONCodec — structured JSON output encoding and decoding.

Encodes Signature inputs into messages with JSON schema instructions.
Decodes LLM responses by extracting and validating JSON output.
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
    signature_to_json_schema,
)
from kaos_llm_core.signatures.signature import Signature
from kaos_llm_core.types import Example

# Pattern to extract JSON from markdown code fences
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


class JSONCodec:
    """Codec that uses JSON structured output.

    Encode: builds system prompt with instruction + field descriptions + JSON schema,
    then a user message with input values.

    Decode: extracts JSON from the response text (plain or code-fenced) and validates
    against the Signature's output fields.
    """

    def encode(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
        examples: list[Example] | None = None,
        instructions: str | None = None,
    ) -> list[dict[str, Any]]:
        """Encode into provider messages with JSON schema.

        Phase 8.2: fields whose declared type is a ``BinaryData`` subclass
        (e.g., :class:`kaos_llm_core.signatures.multimodal.Image`) are
        routed as multipart user-message content. When there are no
        binary fields the output is byte-identical to the pre-Phase-8
        behavior.
        """
        text_inputs, binary_inputs = extract_binary_inputs(signature, inputs)
        instruction = instructions or get_instruction(signature)
        input_fields = get_input_fields(signature)
        output_fields = get_output_fields(signature)
        schema = signature_to_json_schema(signature)

        # Build system prompt
        system_parts: list[str] = []
        if instruction:
            system_parts.append(instruction)

        # Describe output fields
        system_parts.append("\n## Output Format")
        system_parts.append("Respond with a JSON object containing these fields:")
        for name, field_info in output_fields.items():
            desc = field_info.description or name
            system_parts.append(f"- **{name}**: {desc}")

        system_parts.append("\n## JSON Schema")
        system_parts.append(f"```json\n{json.dumps(schema, indent=2)}\n```")
        system_parts.append(
            "\nRespond ONLY with valid JSON matching this schema. "
            "No additional text before or after the JSON."
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "\n".join(system_parts)},
        ]

        # Add few-shot examples
        if examples:
            for ex in examples:
                # User message with example inputs
                user_content = _format_inputs(ex.inputs, input_fields)
                messages.append({"role": "user", "content": user_content})
                # Assistant message with example outputs
                messages.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(ex.outputs, default=str),
                    }
                )

        # User message with actual inputs
        user_content = _format_inputs(text_inputs, input_fields)
        messages.append({"role": "user", "content": user_content})

        # Attach binary inputs (images/audio/documents) as multipart content.
        # No-op when no binary fields are present (byte-identical regression).
        messages = attach_binaries_to_user_message(messages, binary_inputs)

        return messages

    def decode(
        self,
        signature: type[Signature],
        response: ProviderResponse,
    ) -> dict[str, Any]:
        """Decode JSON from the provider response."""
        # Try response.output_json first (provider already parsed)
        if response.output_json is not None:
            parsed = response.output_json
            if isinstance(parsed, dict):
                return _validate_outputs(parsed, signature)

        text = response.text
        if not text:
            raise CodecError(
                "Empty response from LLM. "
                "Check that the model and API key are correct. "
                "Try a different model or increase max_tokens."
            )

        # Try to extract JSON
        parsed = _extract_json(text)
        return _validate_outputs(parsed, signature)


def _format_inputs(
    inputs: dict[str, Any],
    input_fields: dict[str, Any],
) -> str:
    """Format input values as a readable user message."""
    parts: list[str] = []
    for name, value in inputs.items():
        field_info = input_fields.get(name)
        desc = ""
        if field_info and field_info.description:
            desc = f" ({field_info.description})"
        if isinstance(value, str) and len(value) > 200:
            parts.append(f"**{name}**{desc}:\n{value}")
        else:
            parts.append(f"**{name}**{desc}: {_serialize_value(value)}")
    return "\n\n".join(parts)


def _serialize_value(value: Any) -> str:
    """Serialize a value for display in a prompt."""
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


def _compute_truncation_closure(text: str) -> str:
    """Compute the suffix that would close a JSON document truncated mid-token.

    Walks ``text`` once tracking string / array / object nesting, then
    returns the characters needed to finish any open structure. Returns
    ``""`` when the input is already balanced. Used to salvage responses
    from models that hit ``max_tokens`` inside an output field — the
    common failure mode that otherwise yields
    ``'{"summary":"<long prose, cut off mid-sentence'``.

    The walk honours ``\\"`` escapes but makes no attempt to guess at
    *what* the truncated value should have been — it only produces
    syntactically-valid closure so ``json.loads`` can parse whatever
    prefix is intact.
    """
    stack: list[str] = []
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if in_string:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]" and stack and stack[-1] == ch:
            stack.pop()
    suffix_parts: list[str] = []
    if in_string:
        suffix_parts.append('"')
    suffix_parts.extend(reversed(stack))
    return "".join(suffix_parts)


def _try_loads_dict(candidate: str) -> dict[str, Any] | None:
    """Parse ``candidate`` as JSON; return it only if it's a top-level dict."""
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from text, handling code fences."""
    # Try code-fenced JSON first
    match = _JSON_FENCE_RE.search(text)
    if match:
        result = _try_loads_dict(match.group(1).strip())
        if result is not None:
            return result

    # Try the full text as JSON
    text = text.strip()
    result = _try_loads_dict(text)
    if result is not None:
        return result

    # Try to find JSON object boundaries
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        result = _try_loads_dict(text[start : end + 1])
        if result is not None:
            return result

    # Salvage truncated responses: models that hit max_tokens mid-string
    # produce ``'{"summary":"..." <cut>'`` with no closing ``"`` or ``}``.
    # Compute the structural closure and try one more parse.
    if start != -1:
        tail = text[start:]
        closure = _compute_truncation_closure(tail)
        if closure:
            result = _try_loads_dict(tail + closure)
            if result is not None:
                return result

    raise CodecError(
        f"Could not extract JSON from response. "
        f"Response length: {len(text)} chars. "
        f"Head (first 200): {text[:200]!r}. "
        f"Tail (last 200): {text[-200:]!r}. "
        f"Common cause: the model hit max_tokens before closing the JSON. "
        f"Try increasing max_tokens or using a model that supports native "
        f"structured output."
    )


def _validate_outputs(parsed: dict[str, Any], signature: type[Signature]) -> dict[str, Any]:
    """Validate parsed JSON against the Signature's output fields."""
    output_fields = get_output_fields(signature)

    # Check required fields are present
    missing = []
    for name, field_info in output_fields.items():
        if field_info.is_required() and name not in parsed:
            missing.append(name)

    if missing:
        raise CodecError(
            f"Response JSON missing required output fields: {missing}. "
            f"Got keys: {list(parsed.keys())}. "
            f"Expected: {list(output_fields.keys())}."
        )

    # Filter to only output fields (ignore extra keys from the LLM)
    return {name: parsed[name] for name in output_fields if name in parsed}
