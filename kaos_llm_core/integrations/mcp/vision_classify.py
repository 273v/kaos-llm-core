"""KaosLLMCoreVisionClassifyTool — VLM document-page classification.

Wraps :func:`kaos_llm_core.vision.classify_page`: classify a rendered page into
one of ten document-page categories (table, chart, signature_page, exhibit,
form, photo, diagram, text, blank, mixed) with a confidence and reasoning.
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


class KaosLLMCoreVisionClassifyTool(BaseLLMCoreTool):
    """VLM document-page classification of a single page image."""

    _NAME: ClassVar[str] = "kaos-llm-core-vision-classify"
    _DISPLAY_NAME: ClassVar[str] = "LLM Core Vision Classify"
    _DESCRIPTION: ClassVar[str] = (
        "Classify a page image into one document-page category using a "
        "vision-capable model. Categories: table, chart, signature_page, "
        "exhibit, form, photo, diagram, text, blank, mixed. Use this to route "
        "pages (e.g. send 'table' pages to a table extractor, 'signature_page' "
        "to signature review). Accepts a filesystem 'path' or base64 "
        "'image_base64'. Returns page_type, confidence, reasoning, and the "
        "model used."
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

        from kaos_llm_core.vision import classify_page

        model = inputs.get("model")
        kwargs = {"model": model} if isinstance(model, str) and model else {}
        result = await classify_page(image, **kwargs)

        return ToolResult.create_success(
            output={
                "page_type": result.page_type,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "model": result.model,
            },
            summary=f"Page classified as '{result.page_type}' "
            f"(confidence {result.confidence:.2f}, {result.model}).",
        )
