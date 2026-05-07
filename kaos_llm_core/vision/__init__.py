"""Vision programs — typed VLM functions for page-level image understanding.

These programs run a vision-capable model over a rendered page image
(``KaosImage`` from kaos-content) and return a typed result. They are
the architectural complement to text-only ``Signature`` programs:
same Call / Signature plumbing, image input.

Three ready-to-use page programs:

- :func:`describe_page` — free-form description of a rendered page.
- :func:`classify_page` — document-page classification across 10
  categories (table, chart, signature, exhibit, form, photo, diagram,
  text, blank, mixed).
- :func:`ocr_page` — VLM-based OCR ("read every word on this page");
  ~10x more expensive per page than Tesseract, materially better on
  degraded / handwritten / mixed-script scans.

All three accept a ``KaosImage`` directly — the bridge to
:class:`~kaos_llm_core.signatures.multimodal.Image` (the binary type
routed through kaos-llm-client's native attachment mechanism) is
internal so callers never touch base64.

Install with the ``[vision]`` extra to pull ``kaos-content[images]``
(Pillow + numpy)::

    uv add 'kaos-llm-core[vision]'
    pip install 'kaos-llm-core[vision]'

Usage::

    from kaos_llm_core.vision import describe_page

    # `image` is a KaosImage from any source — kaos-pdf.render_page,
    # a kaos-web screenshot, an office-doc rasterization, etc.
    result = await describe_page(image)
    print(result.description)
"""

from __future__ import annotations

from kaos_llm_core.vision.page import (
    PAGE_TYPES,
    PageClassification,
    PageDescription,
    PageOCRResult,
    PageType,
    classify_page,
    describe_page,
    ocr_page,
)

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
