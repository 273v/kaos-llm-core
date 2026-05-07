"""Live vision integration tests — real Claude Haiku on real images.

Generates a synthetic "legal page" image, sends it through the three
page-level vision programs, and asserts content-understanding on each
response. Gated on ``ANTHROPIC_API_KEY``.

Run::

    ANTHROPIC_API_KEY=... uv run pytest tests/integration/test_vision_live.py -m live -v
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from kaos_content.images.model import KaosImage
from PIL import Image, ImageDraw, ImageFont

from kaos_llm_core.vision import classify_page, describe_page, ocr_page


def _has_anthropic_key() -> bool:
    return bool(os.environ.get("KAOS_LLM_ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))


def _find_font(*, size: int) -> Any:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


@pytest.fixture
def legal_page_image() -> KaosImage:
    """A synthetic image of legal prose — looks like a rendered page."""
    img = Image.new("RGB", (850, 1100), "white")
    draw = ImageDraw.Draw(img)
    font = _find_font(size=24)
    lines = [
        "SECURITIES AND EXCHANGE COMMISSION",
        "",
        "RULE 10b-5 — EMPLOYMENT OF MANIPULATIVE",
        "AND DECEPTIVE DEVICES",
        "",
        "It shall be unlawful for any person, directly",
        "or indirectly, by the use of any means or",
        "instrumentality of interstate commerce,",
        "to employ any device, scheme, or artifice",
        "to defraud.",
    ]
    y = 80
    for line in lines:
        draw.text((60, y), line, fill="black", font=font)
        y += 36
    return KaosImage(img)


@pytest.mark.integration
@pytest.mark.live
@pytest.mark.asyncio
async def test_live_describe_page(legal_page_image: KaosImage) -> None:
    if not _has_anthropic_key():
        pytest.skip("ANTHROPIC_API_KEY not set")

    result = await describe_page(legal_page_image)
    desc_lower = result.description.lower()
    assert any(term in desc_lower for term in ("sec", "securities", "10b", "rule", "defraud")), (
        f"Description doesn't mention securities content: {result.description[:200]}"
    )


@pytest.mark.integration
@pytest.mark.live
@pytest.mark.asyncio
async def test_live_classify_page(legal_page_image: KaosImage) -> None:
    if not _has_anthropic_key():
        pytest.skip("ANTHROPIC_API_KEY not set")

    result = await classify_page(legal_page_image)
    assert result.page_type == "text", (
        f"Expected 'text' for a pure-prose page, got {result.page_type!r} "
        f"(reason: {result.reasoning})"
    )
    assert result.confidence > 0.5


@pytest.mark.integration
@pytest.mark.live
@pytest.mark.asyncio
async def test_live_ocr_page(legal_page_image: KaosImage) -> None:
    if not _has_anthropic_key():
        pytest.skip("ANTHROPIC_API_KEY not set")

    result = await ocr_page(legal_page_image)
    text_lower = result.text.lower()
    assert "unlawful" in text_lower, f"VLM OCR didn't recover 'unlawful': {result.text[:300]}"
    assert "defraud" in text_lower
