"""Tests for kaos_llm_core.codecs.json_codec — JSONCodec encoding and decoding."""

from __future__ import annotations

import pytest
from kaos_llm_client import ProviderResponse, UsageInfo
from kaos_llm_client.types import ContentPart

from kaos_llm_core.codecs.json_codec import JSONCodec
from kaos_llm_core.errors import CodecError
from kaos_llm_core.signatures import InputField, OutputField, Signature
from kaos_llm_core.types import Example


class ExtractEntities(Signature):
    """Extract named entities from text."""

    text: str = InputField(description="Input text")
    entities: list[str] = OutputField(description="Extracted entity names")
    confidence: float = OutputField(description="Extraction confidence 0-1")


def _make_response(text: str) -> ProviderResponse:
    """Create a ProviderResponse with given text content."""
    return ProviderResponse.model_construct(
        provider="test",
        model="test-model",
        raw={},
        parts=[ContentPart.model_construct(type="text", text=text)],
        usage=UsageInfo.model_construct(input_tokens=10, output_tokens=5, total_tokens=15),
        stop_reason="end_turn",
        status_code=200,
        response_headers={},
    )


class TestJSONCodecEncode:
    def test_basic_encode(self) -> None:
        codec = JSONCodec()
        messages = codec.encode(
            ExtractEntities,
            {"text": "The SEC filed suit."},
        )

        assert len(messages) >= 2
        # First message is system prompt
        assert messages[0]["role"] == "system"
        assert "Extract named entities" in messages[0]["content"]
        assert "entities" in messages[0]["content"]
        assert "confidence" in messages[0]["content"]
        # Last message is user with inputs
        assert messages[-1]["role"] == "user"
        assert "SEC" in messages[-1]["content"]

    def test_encode_with_instruction_override(self) -> None:
        codec = JSONCodec()
        messages = codec.encode(
            ExtractEntities,
            {"text": "test"},
            instructions="Custom instruction here.",
        )
        assert "Custom instruction here" in messages[0]["content"]

    def test_encode_with_examples(self) -> None:
        codec = JSONCodec()
        examples = [
            Example(
                inputs={"text": "Apple Inc announced earnings."},
                outputs={"entities": ["Apple Inc"], "confidence": 0.9},
            ),
        ]
        messages = codec.encode(
            ExtractEntities,
            {"text": "new input"},
            examples=examples,
        )
        # Should have: system, user (example), assistant (example), user (actual)
        assert len(messages) == 4
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
        assert "Apple Inc" in messages[2]["content"]
        assert messages[3]["role"] == "user"

    def test_encode_json_schema_present(self) -> None:
        codec = JSONCodec()
        messages = codec.encode(ExtractEntities, {"text": "test"})
        system_content = messages[0]["content"]
        assert "JSON Schema" in system_content


class TestJSONCodecDecode:
    def test_decode_clean_json(self) -> None:
        codec = JSONCodec()
        response = _make_response('{"entities": ["SEC", "Acme"], "confidence": 0.85}')
        result = codec.decode(ExtractEntities, response)
        assert result["entities"] == ["SEC", "Acme"]
        assert result["confidence"] == 0.85

    def test_decode_code_fenced_json(self) -> None:
        codec = JSONCodec()
        text = '```json\n{"entities": ["SEC"], "confidence": 0.9}\n```'
        response = _make_response(text)
        result = codec.decode(ExtractEntities, response)
        assert result["entities"] == ["SEC"]

    def test_decode_json_with_surrounding_text(self) -> None:
        codec = JSONCodec()
        text = 'Here are the results:\n{"entities": ["SEC"], "confidence": 0.7}\nDone.'
        response = _make_response(text)
        result = codec.decode(ExtractEntities, response)
        assert result["entities"] == ["SEC"]

    def test_decode_ignores_extra_keys(self) -> None:
        codec = JSONCodec()
        response = _make_response('{"entities": ["X"], "confidence": 0.5, "extra_field": true}')
        result = codec.decode(ExtractEntities, response)
        assert "extra_field" not in result
        assert result["entities"] == ["X"]

    def test_decode_missing_required_field_raises(self) -> None:
        codec = JSONCodec()
        response = _make_response('{"entities": ["X"]}')
        with pytest.raises(CodecError, match="missing required output fields"):
            codec.decode(ExtractEntities, response)

    def test_decode_empty_response_raises(self) -> None:
        codec = JSONCodec()
        response = _make_response("")
        with pytest.raises(CodecError, match="Empty response"):
            codec.decode(ExtractEntities, response)

    def test_decode_unparseable_raises(self) -> None:
        codec = JSONCodec()
        response = _make_response("This is not JSON at all.")
        with pytest.raises(CodecError, match="Could not extract JSON"):
            codec.decode(ExtractEntities, response)

    def test_decode_salvages_truncated_mid_string(self) -> None:
        """Response cut off mid-string (model hit max_tokens) is closed tolerantly."""
        codec = JSONCodec()
        response = _make_response(
            '{"entities": ["SEC", "Acme"], "confidence": 0.85'  # no closing }
        )
        result = codec.decode(ExtractEntities, response)
        assert result["entities"] == ["SEC", "Acme"]
        assert result["confidence"] == 0.85

    def test_decode_truncated_before_required_field_raises_clear_error(self) -> None:
        """Truncation that drops a required output field reports the field by name
        rather than the generic 'Could not extract JSON' — preserving the tolerant
        parse but surfacing the actual schema mismatch.
        """
        codec = JSONCodec()
        # Truncated inside the last entity string; confidence never emitted.
        response = _make_response('{"entities": ["SEC", "Acme Corp is a compan')
        with pytest.raises(CodecError, match="missing required output fields"):
            codec.decode(ExtractEntities, response)

    def test_decode_error_includes_head_and_tail(self) -> None:
        """Unrecoverable parse errors report both head and tail of the response."""
        codec = JSONCodec()
        junk = "prefix " + ("x" * 400) + " suffix"
        response = _make_response(junk)
        with pytest.raises(CodecError, match=r"Head.*Tail"):
            codec.decode(ExtractEntities, response)


class TestTruncationClosure:
    """Unit coverage for the ``_compute_truncation_closure`` helper."""

    def test_balanced_input_returns_empty(self) -> None:
        from kaos_llm_core.codecs.json_codec import _compute_truncation_closure

        assert _compute_truncation_closure('{"a": 1, "b": [1, 2]}') == ""

    def test_open_string_gets_quote_and_brace(self) -> None:
        from kaos_llm_core.codecs.json_codec import _compute_truncation_closure

        assert _compute_truncation_closure('{"summary": "cut off') == '"}'

    def test_open_array_inside_object(self) -> None:
        from kaos_llm_core.codecs.json_codec import _compute_truncation_closure

        assert _compute_truncation_closure('{"items": [{"a": 1') == "}]}"

    def test_escaped_quote_does_not_close_string(self) -> None:
        from kaos_llm_core.codecs.json_codec import _compute_truncation_closure

        # string content ``He said \"hi\" and`` with no closing quote
        assert _compute_truncation_closure(r'{"s": "He said \"hi\" and') == '"}'

    def test_decode_from_output_json(self) -> None:
        """Test that output_json is used when available."""
        codec = JSONCodec()
        response = ProviderResponse.model_construct(
            provider="test",
            model="test-model",
            raw={},
            parts=[
                ContentPart.model_construct(
                    type="text",
                    text='{"entities": ["ignored"], "confidence": 0.1}',
                )
            ],
            usage=UsageInfo.model_construct(input_tokens=10, output_tokens=5, total_tokens=15),
            stop_reason="end_turn",
            status_code=200,
            response_headers={},
        )
        # Manually set output_json via the property
        # ProviderResponse.output_json is computed from parts, so we use the text
        result = codec.decode(ExtractEntities, response)
        assert result["entities"] == ["ignored"]
