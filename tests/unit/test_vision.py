"""Unit tests for the page-level vision programs.

Exercises the KaosImage → LLMImage bridge and the three programs
(describe, classify, ocr) with a monkey-patched Call so no real LLM
calls fire. Live tests against Claude Haiku live in
``tests/integration/test_vision_live.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast

import pytest
from kaos_content.images.model import KaosImage
from PIL import Image as PILImage

from kaos_llm_core.vision import (
    PageClassification,
    PageDescription,
    PageOCRResult,
    classify_page,
    describe_page,
    ocr_page,
)
from kaos_llm_core.vision.page import _kaos_image_to_llm_image

# ------------------------------------------------------------------
# Bridge: KaosImage → LLMImage
# ------------------------------------------------------------------


class _LLMImageLike(Protocol):
    data: str
    media_type: str


class TestBridge:
    def test_produces_llm_image_with_png_media(self) -> None:
        img = KaosImage(PILImage.new("RGB", (50, 50), "red"))
        llm_img = cast(_LLMImageLike, _kaos_image_to_llm_image(img))
        assert type(llm_img).__name__ == "Image"
        assert llm_img.media_type == "image/png"

    def test_data_is_base64_encoded(self) -> None:
        import base64

        img = KaosImage(PILImage.new("RGB", (10, 10), "blue"))
        llm_img = cast(_LLMImageLike, _kaos_image_to_llm_image(img))
        data = llm_img.data
        assert isinstance(data, str)
        raw = base64.b64decode(data)
        # PNG magic bytes
        assert raw[:4] == b"\x89PNG"


# ------------------------------------------------------------------
# Fake Call monkey-patch
# ------------------------------------------------------------------


@dataclass
class _FakeOutput:
    """Canned output object whose fields we set per-test."""

    fields: dict[str, Any]

    def __getattr__(self, name: str) -> Any:
        if name == "fields":
            return object.__getattribute__(self, "fields")
        try:
            return self.fields[name]
        except KeyError:
            raise AttributeError(name) from None


@dataclass
class _FakeInvocation:
    output: _FakeOutput


class _FakeCall:
    _canned: dict[str, Any] | None = None

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    async def invoke(self, **_inputs: Any) -> _FakeInvocation:
        assert _FakeCall._canned is not None
        return _FakeInvocation(output=_FakeOutput(fields=_FakeCall._canned))


def _patch_call(monkeypatch: pytest.MonkeyPatch, canned: dict[str, Any]) -> None:
    import kaos_llm_core.vision.page as page_mod

    monkeypatch.setattr(page_mod, "Call", _FakeCall)
    _FakeCall._canned = canned


# ------------------------------------------------------------------
# describe_page
# ------------------------------------------------------------------


class TestDescribePage:
    @pytest.mark.asyncio
    async def test_returns_description(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_call(monkeypatch, {"description": "A legal document with two paragraphs."})
        img = KaosImage(PILImage.new("RGB", (100, 100), "white"))
        result = await describe_page(img)
        assert isinstance(result, PageDescription)
        assert "legal document" in result.description
        assert result.model == "anthropic:claude-haiku-4-5"


# ------------------------------------------------------------------
# classify_page
# ------------------------------------------------------------------


class TestClassifyPage:
    @pytest.mark.asyncio
    async def test_returns_classification(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_call(
            monkeypatch,
            {
                "page_type": "table",
                "confidence": 0.92,
                "reasoning": "Contains a grid of numbers.",
            },
        )
        img = KaosImage(PILImage.new("RGB", (100, 100), "white"))
        result = await classify_page(img)
        assert isinstance(result, PageClassification)
        assert result.page_type == "table"
        assert result.confidence == pytest.approx(0.92)


# ------------------------------------------------------------------
# ocr_page
# ------------------------------------------------------------------


class TestOCRPage:
    @pytest.mark.asyncio
    async def test_returns_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_call(
            monkeypatch,
            {"text": "Section 10(b) prohibits fraud."},
        )
        img = KaosImage(PILImage.new("RGB", (100, 100), "white"))
        result = await ocr_page(img)
        assert isinstance(result, PageOCRResult)
        assert "Section 10(b)" in result.text
