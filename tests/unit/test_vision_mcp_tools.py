"""Tests for the VLM vision MCP tools (OCR / describe / classify).

Validates, with the live vision programs monkeypatched (no provider calls):

- tool metadata (names, shared image-input parameter schema)
- happy-path execution from a base64 image through a patched vision program
- input validation (missing / ambiguous image source, malformed base64)
- registration in the vision group + the backward-compatible union
"""

from __future__ import annotations

from typing import Any

import pytest
from kaos_core import ToolResult

from kaos_llm_core.integrations.mcp.vision_classify import KaosLLMCoreVisionClassifyTool
from kaos_llm_core.integrations.mcp.vision_describe import KaosLLMCoreVisionDescribeTool
from kaos_llm_core.integrations.mcp.vision_ocr import KaosLLMCoreVisionOcrTool

# 1x1 transparent PNG — decoded by KaosImage (Pillow) in the happy-path tests.
_PNG_1x1_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lE"
    "QVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def _struct(result: ToolResult) -> dict[str, Any]:
    s = result.structuredContent
    assert s is not None, "tool produced no structured content"
    return s


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------


class TestVisionToolMetadata:
    def test_names(self) -> None:
        assert KaosLLMCoreVisionOcrTool().metadata.name == "kaos-llm-core-vision-ocr"
        assert KaosLLMCoreVisionDescribeTool().metadata.name == "kaos-llm-core-vision-describe"
        assert KaosLLMCoreVisionClassifyTool().metadata.name == "kaos-llm-core-vision-classify"

    def test_shared_image_input_schema(self) -> None:
        for tool_cls in (
            KaosLLMCoreVisionOcrTool,
            KaosLLMCoreVisionDescribeTool,
            KaosLLMCoreVisionClassifyTool,
        ):
            names = {p.name for p in tool_cls()._PARAMETERS}
            assert {"path", "image_base64", "model"} <= names


# ---------------------------------------------------------------------------
# happy paths (Pillow required to decode the image; patched vision programs)
# ---------------------------------------------------------------------------


class TestVisionToolExecute:
    @pytest.mark.asyncio
    async def test_ocr_passes_image_and_returns_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("PIL")
        import kaos_llm_core.vision as vision
        from kaos_llm_core.vision.page import PageOCRResult

        seen: dict[str, Any] = {}

        async def _stub(image: Any, *, model: str = "default") -> PageOCRResult:
            seen["image"] = image
            seen["model"] = model
            return PageOCRResult(text="recovered text", model=model)

        monkeypatch.setattr(vision, "ocr_page", _stub)
        result = await KaosLLMCoreVisionOcrTool()._run(
            {"image_base64": _PNG_1x1_B64, "model": "anthropic:claude-haiku-4-5"}
        )
        assert result.isError is False
        out = _struct(result)
        assert out["text"] == "recovered text"
        assert out["model"] == "anthropic:claude-haiku-4-5"
        assert seen["image"] is not None  # a KaosImage was built and passed through

    @pytest.mark.asyncio
    async def test_describe_returns_description(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("PIL")
        import kaos_llm_core.vision as vision
        from kaos_llm_core.vision.page import PageDescription

        async def _stub(image: Any, *, model: str = "default") -> PageDescription:
            _ = image
            return PageDescription(description="a scanned letter", model=model)

        monkeypatch.setattr(vision, "describe_page", _stub)
        result = await KaosLLMCoreVisionDescribeTool()._run({"image_base64": _PNG_1x1_B64})
        assert result.isError is False
        assert _struct(result)["description"] == "a scanned letter"

    @pytest.mark.asyncio
    async def test_classify_returns_page_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("PIL")
        import kaos_llm_core.vision as vision
        from kaos_llm_core.vision.page import PageClassification

        async def _stub(image: Any, *, model: str = "default") -> PageClassification:
            _ = image
            return PageClassification(
                page_type="form", confidence=0.95, reasoning="fields", model=model
            )

        monkeypatch.setattr(vision, "classify_page", _stub)
        result = await KaosLLMCoreVisionClassifyTool()._run({"image_base64": _PNG_1x1_B64})
        assert result.isError is False
        out = _struct(result)
        assert out["page_type"] == "form"
        assert out["confidence"] == 0.95


# ---------------------------------------------------------------------------
# input validation (no Pillow / no provider needed)
# ---------------------------------------------------------------------------


class TestVisionToolInputValidation:
    @pytest.mark.asyncio
    async def test_missing_image_source_errors(self) -> None:
        result = await KaosLLMCoreVisionOcrTool()._run({})
        assert result.isError is True

    @pytest.mark.asyncio
    async def test_both_sources_errors(self) -> None:
        result = await KaosLLMCoreVisionOcrTool()._run(
            {"path": "/tmp/x.png", "image_base64": _PNG_1x1_B64}
        )
        assert result.isError is True

    @pytest.mark.asyncio
    async def test_malformed_base64_errors(self) -> None:
        result = await KaosLLMCoreVisionOcrTool()._run({"image_base64": "not valid base64 !!!"})
        assert result.isError is True


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------


class TestVisionRegistration:
    def test_vision_group_registers_three(self) -> None:
        from kaos_core import KaosRuntime

        from kaos_llm_core.integrations.mcp.registration import register_llm_core_vision_tools

        runtime = KaosRuntime()
        n = register_llm_core_vision_tools(runtime)
        assert n == 3
        registered = set(runtime.tools.list_tools())
        assert {
            "kaos-llm-core-vision-ocr",
            "kaos-llm-core-vision-describe",
            "kaos-llm-core-vision-classify",
        } <= registered
