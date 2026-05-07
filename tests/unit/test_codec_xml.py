"""Tests for XMLCodec — XML tag-based encoding and decoding."""

from __future__ import annotations

import pytest
from kaos_llm_client import ProviderResponse, UsageInfo
from kaos_llm_client.types import ContentPart

from kaos_llm_core.codecs.xml_codec import XMLCodec
from kaos_llm_core.errors import CodecError
from kaos_llm_core.signatures import InputField, OutputField, Signature
from kaos_llm_core.types import Example


class ExtractSig(Signature):
    """Extract entities from text."""

    text: str = InputField(description="Input text")
    entities: list[str] = OutputField(description="Extracted entity names")
    summary: str = OutputField(description="Brief summary")


def _make_response(text: str) -> ProviderResponse:
    return ProviderResponse.model_construct(
        provider="test",
        model="test",
        raw={},
        parts=[ContentPart.model_construct(type="text", text=text)],
        usage=UsageInfo.model_construct(input_tokens=10, output_tokens=5, total_tokens=15),
        stop_reason="end_turn",
        status_code=200,
        response_headers={},
    )


class TestXMLCodecEncode:
    def test_basic_encode(self) -> None:
        codec = XMLCodec()
        messages = codec.encode(ExtractSig, {"text": "The SEC filed suit."})

        assert len(messages) >= 2
        system = messages[0]["content"]
        assert "<entities>" in system
        assert "<summary>" in system
        assert "XML tags" in system

        user = messages[-1]["content"]
        assert "<text>" in user
        assert "SEC" in user

    def test_encode_with_examples(self) -> None:
        codec = XMLCodec()
        examples = [
            Example(
                inputs={"text": "Apple announced earnings."},
                outputs={"entities": ["Apple"], "summary": "Earnings report."},
            ),
        ]
        messages = codec.encode(ExtractSig, {"text": "test"}, examples=examples)
        assert len(messages) == 4  # system, example_user, example_assistant, user
        assert "<entities>" in messages[2]["content"]


class TestXMLCodecDecode:
    def test_basic_decode(self) -> None:
        codec = XMLCodec()
        text = '<entities>["SEC", "Acme"]</entities>\n<summary>SEC sued Acme.</summary>'
        response = _make_response(text)
        result = codec.decode(ExtractSig, response)
        assert result["entities"] == ["SEC", "Acme"]
        assert result["summary"] == "SEC sued Acme."

    def test_decode_with_whitespace(self) -> None:
        codec = XMLCodec()
        text = '<entities>\n  ["X"]\n</entities>\n<summary>\n  Brief.\n</summary>'
        response = _make_response(text)
        result = codec.decode(ExtractSig, response)
        assert result["entities"] == ["X"]
        assert result["summary"] == "Brief."

    def test_decode_missing_required_raises(self) -> None:
        codec = XMLCodec()
        text = '<entities>["X"]</entities>'  # missing summary
        response = _make_response(text)
        with pytest.raises(CodecError, match="summary"):
            codec.decode(ExtractSig, response)

    def test_decode_empty_response_raises(self) -> None:
        codec = XMLCodec()
        response = _make_response("")
        with pytest.raises(CodecError, match="Empty response"):
            codec.decode(ExtractSig, response)

    def test_decode_non_string_coercion(self) -> None:
        codec = XMLCodec()
        text = '<entities>["a", "b"]</entities>\n<summary>ok</summary>'
        response = _make_response(text)
        result = codec.decode(ExtractSig, response)
        assert isinstance(result["entities"], list)
        assert result["entities"] == ["a", "b"]

    def test_decode_multiline_content(self) -> None:
        codec = XMLCodec()
        text = (
            '<entities>["SEC"]</entities>\n'
            "<summary>\nThis is a multi-line\nsummary with details.\n</summary>"
        )
        response = _make_response(text)
        result = codec.decode(ExtractSig, response)
        assert "multi-line" in result["summary"]
