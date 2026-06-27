"""KaosLLMCoreVisionOcrTool — VLM-based OCR of a page image.

Wraps :func:`kaos_llm_core.vision.ocr_page` so an MCP client can transcribe a
scanned / photographed page with a vision-capable model. The VLM complement to
``kaos-pdf``'s Tesseract engine: ~10x the cost per page, materially better on
degraded / handwritten / mixed-script scans.
"""

from __future__ import annotations

from typing import Any, ClassVar

from kaos_core import KaosContext, ToolResult
from kaos_core.types.enums import ToolCapability, ToolCategory
from kaos_core.types.parameters import ParameterSchema

from kaos_llm_core.integrations.mcp._common import _LLM_CORE_ANNOTATIONS, BaseLLMCoreTool
from kaos_llm_core.integrations.mcp._vision_common import (
    VISION_ERROR_HINT,
    VISION_INPUT_PARAMETERS,
    load_page_image,
)


class KaosLLMCoreVisionOcrTool(BaseLLMCoreTool):
    """VLM OCR of a single page image."""

    _NAME: ClassVar[str] = "kaos-llm-core-vision-ocr"
    _DISPLAY_NAME: ClassVar[str] = "LLM Core Vision OCR"
    _DESCRIPTION: ClassVar[str] = (
        "Transcribe every word of text on a page image using a vision-capable "
        "model (VLM OCR). The VLM complement to Tesseract: ~10x the cost per "
        "page but materially more accurate on degraded, handwritten, or "
        "mixed-script scans, and better at preserving reading order and "
        "paragraph structure. Accepts a filesystem 'path' or base64 "
        "'image_base64'. Returns the recognized text and the model used."
    )
    _CATEGORY: ClassVar[ToolCategory] = ToolCategory.INTEGRATION
    _CAPABILITY: ClassVar[ToolCapability] = ToolCapability.QUERY
    _ANNOTATIONS = _LLM_CORE_ANNOTATIONS
    _PARAMETERS: ClassVar[list[ParameterSchema]] = list(VISION_INPUT_PARAMETERS)
    _ERROR_HINT: ClassVar[str] = VISION_ERROR_HINT

    async def _run(self, inputs: dict[str, Any], context: KaosContext | None = None) -> ToolResult:
        image = load_page_image(inputs)
        if isinstance(image, ToolResult):
            return image

        from kaos_llm_core.vision import ocr_page

        model = inputs.get("model")
        kwargs = {"model": model} if isinstance(model, str) and model else {}
        result = await ocr_page(image, **kwargs)

        return ToolResult.create_success(
            output={"text": result.text, "model": result.model},
            summary=f"VLM OCR recovered {len(result.text)} chars ({result.model}).",
        )
