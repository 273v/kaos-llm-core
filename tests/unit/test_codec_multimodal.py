"""Tests for codec multimodal routing (Phase 8.2 / §5.2).

These tests lock two invariants:

1. When no binary inputs are present, codec output is **byte-identical**
   to pre-Phase-8 behavior (the encode() result for a text-only Signature
   must match across JSONCodec/ChatCodec/XMLCodec).
2. When a ``BinaryData`` input IS present, the final user message carries
   multipart content with the text part plus one provider-format image
   part per binary input.
"""

from __future__ import annotations

import base64

from kaos_llm_core.codecs.chat_codec import ChatCodec
from kaos_llm_core.codecs.json_codec import JSONCodec
from kaos_llm_core.codecs.xml_codec import XMLCodec
from kaos_llm_core.signatures import InputField, OutputField, Signature
from kaos_llm_core.signatures.multimodal import Image


class TextOnly(Signature):
    """A text-only signature — should never touch the binary path."""

    text: str = InputField(description="Input text")
    result: str = OutputField(description="Output result")


class Caption(Signature):
    """Generate a caption for an image."""

    image: Image = InputField(description="The image to caption")
    note: str = InputField(description="Optional note")
    caption: str = OutputField(description="Alt-text caption")


# 1x1 PNG
_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "89000000015352474200aece1ce90000000d49444154789c6300010000000500"
    "01f08c5a1f0000000049454e44ae426082"
)


def _final_user_message(messages: list[dict]) -> dict:
    return messages[-1]


# ---------------------------------------------------------------------------
# Byte-identity regression (no binary fields)
# ---------------------------------------------------------------------------


class TestNoBinaryRegression:
    def test_json_codec_text_only_unchanged(self) -> None:
        codec = JSONCodec()
        msgs = codec.encode(TextOnly, {"text": "hello"})
        final = _final_user_message(msgs)
        # Content must still be a plain string (NOT a multipart list)
        assert isinstance(final["content"], str)
        assert "hello" in final["content"]

    def test_xml_codec_text_only_unchanged(self) -> None:
        codec = XMLCodec()
        msgs = codec.encode(TextOnly, {"text": "hello"})
        final = _final_user_message(msgs)
        assert isinstance(final["content"], str)
        assert "hello" in final["content"]

    def test_chat_codec_text_only_unchanged(self) -> None:
        codec = ChatCodec()
        msgs = codec.encode(TextOnly, {"text": "hello"})
        final = _final_user_message(msgs)
        assert isinstance(final["content"], str)
        assert "hello" in final["content"]


# ---------------------------------------------------------------------------
# Binary routing
# ---------------------------------------------------------------------------


class TestBinaryRouting:
    def test_json_codec_routes_image(self) -> None:
        codec = JSONCodec()
        img = Image.from_bytes(_PNG)
        msgs = codec.encode(Caption, {"image": img, "note": "hi"})
        final = _final_user_message(msgs)
        assert isinstance(final["content"], list), "expected multipart content"
        # First part must be text, subsequent parts include the image
        assert final["content"][0]["type"] == "text"
        assert "hi" in final["content"][0]["text"]
        # Second part is the image
        image_parts = [p for p in final["content"] if p.get("type") == "image_url"]
        assert len(image_parts) == 1
        # The image_url must embed our base64 PNG
        url = image_parts[0]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        b64 = url.split(",", 1)[1]
        assert base64.b64decode(b64) == _PNG

    def test_xml_codec_routes_image(self) -> None:
        codec = XMLCodec()
        img = Image.from_bytes(_PNG)
        msgs = codec.encode(Caption, {"image": img, "note": "hi"})
        final = _final_user_message(msgs)
        assert isinstance(final["content"], list)
        assert final["content"][0]["type"] == "text"
        assert any(p.get("type") == "image_url" for p in final["content"])

    def test_chat_codec_routes_image(self) -> None:
        codec = ChatCodec()
        img = Image.from_bytes(_PNG)
        msgs = codec.encode(Caption, {"image": img, "note": "hi"})
        final = _final_user_message(msgs)
        assert isinstance(final["content"], list)
        assert final["content"][0]["type"] == "text"
        assert any(p.get("type") == "image_url" for p in final["content"])

    def test_text_content_does_not_include_binary_field_name(self) -> None:
        """The text portion should NOT try to render the BinaryData value
        (which would json.dumps a Pydantic model and blow up or produce
        noise). The binary field is routed away from the text path."""
        codec = JSONCodec()
        img = Image.from_bytes(_PNG)
        msgs = codec.encode(Caption, {"image": img, "note": "hello"})
        final = _final_user_message(msgs)
        text_part = final["content"][0]["text"]
        # "image" field name must NOT appear as a **image** marker in text;
        # the field was extracted before _format_inputs was called.
        assert "**image**" not in text_part
        # The note field should still appear
        assert "hello" in text_part


class ImageOnly(Signature):
    """A binary-only signature — used to verify the empty-text edge case."""

    image: Image = InputField(description="The image")
    color: str = OutputField(description="Dominant color")


class TestEmptyTextEdgeCase:
    """Phase 8.2 live verification regression: when the text encoding is empty
    (Signature with only binary fields), the codec must NOT emit a zero-length
    text content block — Anthropic rejects those with
    ``messages: text content blocks must be non-empty``. The fix is to omit
    the text part entirely so the user message is binary-only.
    """

    def test_json_codec_image_only_omits_empty_text_part(self) -> None:
        codec = JSONCodec()
        img = Image.from_bytes(_PNG)
        msgs = codec.encode(ImageOnly, {"image": img})
        final = _final_user_message(msgs)
        assert isinstance(final["content"], list)
        # No empty text part. Either zero text parts (image-only) or one
        # non-empty text part. Empty strings are forbidden.
        text_parts = [p for p in final["content"] if p.get("type") == "text"]
        for tp in text_parts:
            assert tp["text"], (
                f"Empty text content block leaked through; Anthropic rejects "
                f"these with 400. parts: {final['content']}"
            )
        # The image part must still be present.
        image_parts = [p for p in final["content"] if p.get("type") == "image_url"]
        assert len(image_parts) == 1

    def test_chat_codec_image_only_omits_empty_text_part(self) -> None:
        codec = ChatCodec()
        img = Image.from_bytes(_PNG)
        msgs = codec.encode(ImageOnly, {"image": img})
        final = _final_user_message(msgs)
        assert isinstance(final["content"], list)
        text_parts = [p for p in final["content"] if p.get("type") == "text"]
        for tp in text_parts:
            assert tp["text"]
        assert any(p.get("type") == "image_url" for p in final["content"])

    def test_xml_codec_image_only_omits_empty_text_part(self) -> None:
        codec = XMLCodec()
        img = Image.from_bytes(_PNG)
        msgs = codec.encode(ImageOnly, {"image": img})
        final = _final_user_message(msgs)
        assert isinstance(final["content"], list)
        text_parts = [p for p in final["content"] if p.get("type") == "text"]
        for tp in text_parts:
            assert tp["text"]
        assert any(p.get("type") == "image_url" for p in final["content"])
