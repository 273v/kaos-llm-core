"""Page-level VLM programs — typed Signatures + async entry points.

Each public function is a thin wrapper around a
:class:`kaos_llm_core.programs.call.Call` with a vision-capable
:class:`~kaos_llm_core.signatures.signature.Signature`. The function
handles the ``KaosImage`` → :class:`~kaos_llm_core.signatures.multimodal.Image`
bridge so callers never touch base64 / ``BinaryData`` directly.

Default model: ``anthropic:claude-haiku-4-5`` — the cheapest vision-
capable model with reliable structured-output support. Callers can
override via ``model=`` to use Sonnet / GPT-5.4 / Gemini Flash for
accuracy-sensitive tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from kaos_content.images.model import KaosImage

from kaos_llm_core.programs.call import Call
from kaos_llm_core.signatures.fields import InputField, OutputField
from kaos_llm_core.signatures.multimodal import Image as LLMImage
from kaos_llm_core.signatures.signature import Signature

DEFAULT_VISION_MODEL = "anthropic:claude-haiku-4-5"


def _kaos_image_to_llm_image(image: KaosImage) -> LLMImage:
    """Bridge KaosImage → kaos-llm-core's Image binary type.

    Returns a :class:`~kaos_llm_core.signatures.multimodal.Image`
    instance built from the raw PNG bytes of the input.
    """
    raw_bytes = image.to_bytes(format="png")
    return LLMImage.from_bytes(raw_bytes, media_type="image/png")


# ------------------------------------------------------------------
# describe_page
# ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PageDescription:
    """Result of :func:`describe_page`."""

    description: str
    model: str


class _DescribePage(Signature):
    """Describe the contents of this document page in detail.
    Focus on document structure: headings, paragraphs, tables,
    images, signatures, stamps, redactions, annotations."""

    page_image: LLMImage = InputField(description="Rendered page image")
    description: str = OutputField(
        description=(
            "Detailed description of the page contents, structure, and any notable features"
        )
    )


async def describe_page(
    image: KaosImage,
    *,
    model: str = DEFAULT_VISION_MODEL,
    instruction: str | None = None,
) -> PageDescription:
    """Generate a free-form description of a rendered page.

    Args:
        image: Rendered page (e.g. from ``kaos_pdf.render_page``).
        model: Provider:model string. Default Haiku — cheap and fast.
        instruction: Optional extra instruction prepended to the
            system prompt. Use for domain guidance ("this is a legal
            filing — focus on clause structure").

    Returns:
        :class:`PageDescription` with the model's free-form text.
    """
    llm_image = _kaos_image_to_llm_image(image)
    call = Call(_DescribePage, model=model, instructions=instruction)
    invocation = await call.invoke(page_image=llm_image)
    return PageDescription(
        description=invocation.output.description,
        model=model,
    )


# ------------------------------------------------------------------
# classify_page
# ------------------------------------------------------------------


PAGE_TYPES = (
    "text",
    "table",
    "chart",
    "form",
    "signature_page",
    "exhibit",
    "photo",
    "diagram",
    "blank",
    "mixed",
)
PageType = Literal[
    "text",
    "table",
    "chart",
    "form",
    "signature_page",
    "exhibit",
    "photo",
    "diagram",
    "blank",
    "mixed",
]


@dataclass(frozen=True, slots=True)
class PageClassification:
    """Result of :func:`classify_page`."""

    page_type: str
    confidence: float
    reasoning: str
    model: str


class _ClassifyPage(Signature):
    """Classify this document page into exactly one category.
    Categories: text, table, chart, form, signature_page, exhibit,
    photo, diagram, blank, mixed."""

    page_image: LLMImage = InputField(description="Rendered page image")
    page_type: PageType = OutputField(description="The single best category for this page")
    confidence: float = OutputField(description="Confidence in the classification, 0.0 to 1.0")
    reasoning: str = OutputField(description="One sentence explaining the classification")


async def classify_page(
    image: KaosImage,
    *,
    model: str = DEFAULT_VISION_MODEL,
) -> PageClassification:
    """Classify a page image into a document-type category.

    Args:
        image: Rendered page.
        model: Provider:model string.

    Returns:
        :class:`PageClassification` with one of ``PAGE_TYPES``, a
        confidence score, and a brief reasoning string.
    """
    llm_image = _kaos_image_to_llm_image(image)
    call = Call(_ClassifyPage, model=model)
    invocation = await call.invoke(page_image=llm_image)
    output = invocation.output
    return PageClassification(
        page_type=output.page_type,
        confidence=output.confidence,
        reasoning=output.reasoning,
        model=model,
    )


# ------------------------------------------------------------------
# ocr_page (VLM-based OCR)
# ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PageOCRResult:
    """Result of :func:`ocr_page`."""

    text: str
    model: str


class _OCRPage(Signature):
    """Read and transcribe every word of text visible on this page.
    Preserve paragraph structure. Do not summarize or interpret —
    output the text exactly as it appears."""

    page_image: LLMImage = InputField(description="Scanned or photographed document page")
    text: str = OutputField(
        description="All visible text on the page, in reading order, with paragraph breaks"
    )


async def ocr_page(
    image: KaosImage,
    *,
    model: str = DEFAULT_VISION_MODEL,
) -> PageOCRResult:
    """Read all visible text from a page image using a VLM.

    The VLM-based complement to ``kaos_pdf``'s Tesseract engine:
    higher accuracy on degraded / handwritten / noisy scans, but
    ~10x more expensive per page (~$0.005 for Haiku vs ~$0 for
    Tesseract). Use when Tesseract's output is low-confidence or
    when the scan contains handwriting / mixed scripts.

    Args:
        image: Rendered page.
        model: Provider:model string.

    Returns:
        :class:`PageOCRResult` with the recognized text. The model
        typically returns text in reading order with paragraph breaks
        preserved.
    """
    llm_image = _kaos_image_to_llm_image(image)
    call = Call(_OCRPage, model=model)
    invocation = await call.invoke(page_image=llm_image)
    return PageOCRResult(text=invocation.output.text, model=model)


__all__ = [
    "PAGE_TYPES",
    "PageClassification",
    "PageDescription",
    "PageOCRResult",
    "PageType",
    "classify_page",
    "describe_page",
    "ocr_page",
]
