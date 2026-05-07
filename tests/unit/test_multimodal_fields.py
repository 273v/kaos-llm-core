"""Unit tests for kaos_llm_core.signatures.multimodal — Image/Audio/Document."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from kaos_llm_client.types import BinaryData

from kaos_llm_core.signatures.multimodal import (
    Audio,
    Document,
    Image,
    is_binary_field,
)

# A 1x1 transparent PNG (smallest valid PNG, useful as a test fixture).
_ONE_PIXEL_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "89000000015352474200aece1ce90000000d49444154789c6300010000000500"
    "01f08c5a1f0000000049454e44ae426082"
)


class TestImage:
    def test_isinstance_binary_data(self) -> None:
        img = Image.from_bytes(_ONE_PIXEL_PNG, media_type="image/png")
        assert isinstance(img, BinaryData)
        assert isinstance(img, Image)

    def test_from_bytes(self) -> None:
        img = Image.from_bytes(_ONE_PIXEL_PNG)
        assert img.media_type == "image/png"
        assert img.is_image
        # Round-trip the base64 to ensure correctness
        assert base64.b64decode(img.data) == _ONE_PIXEL_PNG

    def test_from_path_roundtrip(self, tmp_path: Path) -> None:
        fixture = tmp_path / "tiny.png"
        fixture.write_bytes(_ONE_PIXEL_PNG)
        img = Image.from_path(fixture)
        assert isinstance(img, Image)
        assert img.media_type == "image/png"
        assert base64.b64decode(img.data) == _ONE_PIXEL_PNG

    def test_from_path_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            Image.from_path(tmp_path / "missing.png")

    def test_from_url_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError, match="from_bytes"):
            Image.from_url("https://example.com/x.png")

    def test_from_data_uri(self) -> None:
        b64 = base64.b64encode(_ONE_PIXEL_PNG).decode("ascii")
        uri = f"data:image/png;base64,{b64}"
        img = Image.from_data_uri(uri)
        assert isinstance(img, Image)
        assert img.media_type == "image/png"


class TestAudio:
    def test_from_bytes(self) -> None:
        audio = Audio.from_bytes(b"RIFF....", media_type="audio/wav")
        assert isinstance(audio, BinaryData)
        assert isinstance(audio, Audio)
        assert audio.is_audio

    def test_from_path_defaults_to_wav_on_unknown(self, tmp_path: Path) -> None:
        fixture = tmp_path / "snippet.wav"
        fixture.write_bytes(b"RIFF1234WAVEfmt ")
        audio = Audio.from_path(fixture)
        assert audio.media_type == "audio/wav" or audio.media_type.startswith("audio/")


class TestDocument:
    def test_from_bytes(self) -> None:
        doc = Document.from_bytes(b"%PDF-1.4...", media_type="application/pdf")
        assert isinstance(doc, BinaryData)
        assert isinstance(doc, Document)
        assert doc.is_document

    def test_from_path(self, tmp_path: Path) -> None:
        fixture = tmp_path / "note.txt"
        fixture.write_text("hello world")
        doc = Document.from_path(fixture)
        assert isinstance(doc, Document)
        # text/plain is a "document" per BinaryData.is_document set
        assert doc.media_type == "text/plain"


class TestIsBinaryField:
    def test_image_is_binary(self) -> None:
        assert is_binary_field(Image)

    def test_binary_data_is_binary(self) -> None:
        assert is_binary_field(BinaryData)

    def test_str_is_not_binary(self) -> None:
        assert not is_binary_field(str)

    def test_int_is_not_binary(self) -> None:
        assert not is_binary_field(int)

    def test_none_is_not_binary(self) -> None:
        assert not is_binary_field(None)

    def test_list_annotation_not_binary(self) -> None:
        assert not is_binary_field(list[str])
