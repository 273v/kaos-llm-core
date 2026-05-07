"""Tests for kaos_llm_core.codecs.chat_codec — ChatCodec encoding and decoding."""

from __future__ import annotations

import pytest
from kaos_llm_client import ProviderResponse, UsageInfo
from kaos_llm_client.types import ContentPart

from kaos_llm_core.codecs.chat_codec import ChatCodec
from kaos_llm_core.errors import CodecError
from kaos_llm_core.signatures import InputField, OutputField, Signature
from kaos_llm_core.types import Example


class Summarize(Signature):
    """Summarize the given text."""

    text: str = InputField(description="Text to summarize")
    summary: str = OutputField(description="A concise summary")
    key_points: str = OutputField(description="Bullet points of key ideas")


def _make_response(text: str) -> ProviderResponse:
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


class TestChatCodecEncode:
    def test_basic_encode(self) -> None:
        codec = ChatCodec()
        messages = codec.encode(Summarize, {"text": "Long document here."})

        assert len(messages) == 2
        system = messages[0]
        assert system["role"] == "system"
        assert "Summarize" in system["content"]
        assert "[summary]" in system["content"]
        assert "[key_points]" in system["content"]

        user = messages[1]
        assert user["role"] == "user"
        assert "Long document here" in user["content"]

    def test_encode_with_examples(self) -> None:
        codec = ChatCodec()
        examples = [
            Example(
                inputs={"text": "Example input."},
                outputs={"summary": "Brief.", "key_points": "- Point 1"},
            ),
        ]
        messages = codec.encode(Summarize, {"text": "actual input"}, examples=examples)
        # system + example_user + example_assistant + actual_user
        assert len(messages) == 4
        assert messages[2]["role"] == "assistant"
        assert "[summary]" in messages[2]["content"]
        assert "Brief." in messages[2]["content"]


class TestChatCodecDecode:
    def test_basic_decode(self) -> None:
        codec = ChatCodec()
        text = "[summary]\nThis is a brief summary.\n\n[key_points]\n- Point 1\n- Point 2"
        response = _make_response(text)
        result = codec.decode(Summarize, response)
        assert "brief summary" in result["summary"]
        assert "Point 1" in result["key_points"]

    def test_decode_strips_whitespace(self) -> None:
        codec = ChatCodec()
        text = "[summary]\n  Trimmed text  \n\n[key_points]\n  - A  "
        response = _make_response(text)
        result = codec.decode(Summarize, response)
        assert result["summary"] == "Trimmed text"

    def test_decode_missing_required_raises(self) -> None:
        codec = ChatCodec()
        text = "[summary]\nOnly summary here."
        response = _make_response(text)
        with pytest.raises(CodecError, match="key_points"):
            codec.decode(Summarize, response)

    def test_decode_empty_response_raises(self) -> None:
        codec = ChatCodec()
        response = _make_response("")
        with pytest.raises(CodecError, match="Empty response"):
            codec.decode(Summarize, response)
