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
        """Decode JSON from the provider response.

        Reconciles two candidate parses and prefers the more complete one:

        1. ``response.output_json`` — the provider/client convenience parse.
           This can be a *partial* recovery (``pydantic_core`` with
           ``allow_partial``) that silently drops trailing fields when a string
           value contains an inline unescaped quote.
        2. ``_extract_json(response.text)`` — this codec's own fence-aware,
           inline-quote-repairing parse of the full text.

        Historically (1) was returned unconditionally whenever non-``None``,
        so a truncated fragment hid the complete output. We now validate both
        and keep whichever satisfies the Signature's required fields, breaking
        ties toward the candidate with more keys. This guarantees a complete
        object with quoted text round-trips without content loss while
        preserving the genuine-truncation salvage path.
        """
        text = response.text

        # Candidate A: this codec's own parse of the full text (fence-aware,
        # inline-quote repair, truncation closure).
        text_parsed: dict[str, Any] | None = None
        text_error: CodecError | None = None
        if text:
            try:
                text_parsed = _extract_json(text)
            except CodecError as exc:
                text_error = exc

        # Candidate B: the client-side convenience parse.
        output_json = response.output_json
        oj_parsed = output_json if isinstance(output_json, dict) else None

        # Prefer the more complete VALID candidate. A candidate is "valid"
        # when it carries every required output field.
        chosen = _pick_more_complete(text_parsed, oj_parsed, signature)
        if chosen is not None:
            return _validate_outputs(chosen, signature)

        if not text:
            raise CodecError(
                "Empty response from LLM. "
                "Check that the model and API key are correct. "
                "Try a different model or increase max_tokens."
            )

        # Neither candidate validated. Surface the schema-aware error from the
        # full-text parse when we have a dict (e.g. missing required field),
        # otherwise the extraction error.
        if text_parsed is not None:
            return _validate_outputs(text_parsed, signature)
        if oj_parsed is not None:
            return _validate_outputs(oj_parsed, signature)
        if text_error is not None:
            raise text_error
        # No dict from either path: run extraction once more to raise the
        # canonical "could not extract JSON" error.
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


def _repair_inline_quotes(text: str) -> str | None:
    """Escape unescaped ``"`` that appear *inside* JSON string values.

    Single forward pass. A ``"`` encountered while inside a string is treated
    as that string's closing quote only when the next non-whitespace character
    is structural (``,``, ``}``, ``]``, ``:``) or end-of-input. Any other ``"``
    inside a string is an inline quote and is escaped to ``\\"``. Returns the
    repaired text, or ``None`` when no change was needed (already balanced),
    so callers can skip a redundant reparse.

    This salvages the common legal-product failure mode where a model quotes
    document text verbatim inside an output field, producing a COMPLETE object
    that strict ``json.loads`` rejects at the inline quote.
    """
    out: list[str] = []
    in_string = False
    escape = False
    changed = False
    n = len(text)
    for i, ch in enumerate(text):
        if escape:
            out.append(ch)
            escape = False
            continue
        if in_string:
            if ch == "\\":
                out.append(ch)
                escape = True
                continue
            if ch == '"':
                j = i + 1
                while j < n and text[j] in " \t\r\n":
                    j += 1
                if j >= n or text[j] in ",}]:":
                    out.append(ch)
                    in_string = False
                else:
                    out.append('\\"')
                    changed = True
                continue
            out.append(ch)
            continue
        if ch == '"':
            in_string = True
        out.append(ch)
    if not changed:
        return None
    return "".join(out)


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
        candidate = text[start : end + 1]
        result = _try_loads_dict(candidate)
        if result is not None:
            return result
        # Salvage a COMPLETE object whose string value contains an inline
        # unescaped double-quote (e.g. a memo quoting document text verbatim).
        # Without this, strict json.loads breaks at the quote and the whole
        # object is lost / silently truncated downstream.
        repaired = _repair_inline_quotes(candidate)
        if repaired is not None:
            result = _try_loads_dict(repaired)
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


def _has_required_fields(parsed: dict[str, Any], signature: type[Signature]) -> bool:
    """Whether ``parsed`` carries every required output field of ``signature``."""
    output_fields = get_output_fields(signature)
    return all(name in parsed for name, fi in output_fields.items() if fi.is_required())


def _pick_more_complete(
    text_parsed: dict[str, Any] | None,
    oj_parsed: dict[str, Any] | None,
    signature: type[Signature],
) -> dict[str, Any] | None:
    """Choose the more complete VALID parse between the two candidates.

    A candidate is preferred only when it satisfies the Signature's required
    fields. When both are valid, the one with more keys wins (the partial
    recovery that dropped trailing fields therefore loses to the full parse).
    Returns ``None`` when neither candidate validates, so the caller can raise
    a schema-aware error.
    """
    text_ok = text_parsed is not None and _has_required_fields(text_parsed, signature)
    oj_ok = oj_parsed is not None and _has_required_fields(oj_parsed, signature)
    if text_ok and oj_ok:
        assert text_parsed is not None and oj_parsed is not None
        return text_parsed if len(text_parsed) >= len(oj_parsed) else oj_parsed
    if text_ok:
        return text_parsed
    if oj_ok:
        return oj_parsed
    return None


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
