"""KaosLLMCoreVisionDescribeTool — VLM structural description of a page image.

Wraps :func:`kaos_llm_core.vision.describe_page`: a free-form description of a
rendered page focused on document structure (headings, tables, signatures,
stamps, redactions, annotations).
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


class KaosLLMCoreVisionDescribeTool(BaseLLMCoreTool):
    """VLM structural description of a single page image."""

    _NAME: ClassVar[str] = "kaos-llm-core-vision-describe"
    _DISPLAY_NAME: ClassVar[str] = "LLM Core Vision Describe"
    _DESCRIPTION: ClassVar[str] = (
        "Describe the structure and contents of a page image using a "
        "vision-capable model: headings, paragraphs, tables, figures, "
        "signatures, stamps, redactions, and annotations. Use this to "
        "understand layout before extraction, or to triage what a scanned "
        "page contains. Accepts a filesystem 'path' or base64 'image_base64'. "
        "Returns a free-form description and the model used."
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

        from kaos_llm_core.vision import describe_page

        model = inputs.get("model")
        kwargs = {"model": model} if isinstance(model, str) and model else {}
        result = await describe_page(image, **kwargs)

        return ToolResult.create_success(
            output={"description": result.description, "model": result.model},
            summary=f"VLM page description ({result.model}).",
        )
