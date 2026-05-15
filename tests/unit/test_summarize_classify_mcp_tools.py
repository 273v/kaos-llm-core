"""Tests for the §7.3 ``summarize`` / ``classify`` MCP tools.

Validates:

- tool metadata (name, parameter schema)
- happy-path execution against a patched starter façade
- error path on missing / empty parameters
- registration in the program-tools group via
  :func:`register_llm_core_program_tools`
"""

from __future__ import annotations

from typing import Any

import pytest
from kaos_core import ToolResult

from kaos_llm_core import starter
from kaos_llm_core.integrations.mcp.classify import KaosLLMCoreClassifyTool
from kaos_llm_core.integrations.mcp.summarize import KaosLLMCoreSummarizeTool
from kaos_llm_core.labels import Label, LabelSet
from kaos_llm_core.results import Classification, Summary


def _struct(result: ToolResult) -> dict[str, Any]:
    s = result.structuredContent
    assert s is not None, "tool produced no structured content"
    return s


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------


class TestSummarizeToolMetadata:
    def test_name(self) -> None:
        assert KaosLLMCoreSummarizeTool().metadata.name == "kaos-llm-core-summarize"

    def test_parameters_include_text_and_strategy(self) -> None:
        names = {p.name for p in KaosLLMCoreSummarizeTool()._PARAMETERS}
        assert {"text", "long_strategy", "cited", "budget_tokens", "budget_usd"} <= names


class TestClassifyToolMetadata:
    def test_name(self) -> None:
        assert KaosLLMCoreClassifyTool().metadata.name == "kaos-llm-core-classify"

    def test_parameters_include_labels_and_supervision(self) -> None:
        names = {p.name for p in KaosLLMCoreClassifyTool()._PARAMETERS}
        assert {"text", "labels", "supervision", "long_strategy"} <= names


# ---------------------------------------------------------------------------
# happy paths
# ---------------------------------------------------------------------------


class TestSummarizeToolExecute:
    @pytest.mark.asyncio
    async def test_passes_through_to_starter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, Any] = {}

        async def _stub(doc: str, **kwargs: Any) -> Summary[str]:
            seen["doc"] = doc
            seen.update(kwargs)
            return Summary[str](
                text="mcp summary",
                method="abstractive",
                metadata={"starter.long_strategy": "single"},
            )

        monkeypatch.setattr(starter, "summarize_doc", _stub)
        tool = KaosLLMCoreSummarizeTool()
        result = await tool._run(
            {"text": "Hello world.", "long_strategy": "single"},
        )
        assert result.isError is False
        out = _struct(result)
        assert out["text"] == "mcp summary"
        assert seen["doc"] == "Hello world."
        assert seen["long_strategy"] == "single"

    @pytest.mark.asyncio
    async def test_budget_kwargs_assembled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, Any] = {}

        async def _stub(doc: str, **kwargs: Any) -> Summary[str]:
            seen.update(kwargs)
            return Summary[str](text="", method="abstractive")

        monkeypatch.setattr(starter, "summarize_doc", _stub)
        tool = KaosLLMCoreSummarizeTool()
        await tool._run({"text": "x", "budget_tokens": 100, "budget_usd": 0.05})
        assert seen["budget"] is not None
        assert seen["budget"].max_tokens == 100
        assert seen["budget"].max_cost_usd == 0.05

    @pytest.mark.asyncio
    async def test_empty_text_returns_error(self) -> None:
        tool = KaosLLMCoreSummarizeTool()
        result = await tool._run({"text": ""})
        assert result.isError is True

    @pytest.mark.asyncio
    async def test_missing_text_returns_error(self) -> None:
        tool = KaosLLMCoreSummarizeTool()
        result = await tool._run({})
        assert result.isError is True


class TestClassifyToolExecute:
    @pytest.mark.asyncio
    async def test_label_list_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen_labels: list[LabelSet] = []

        async def _stub(doc: str, labels: LabelSet, **kwargs: Any) -> Classification:
            seen_labels.append(labels)
            return Classification(
                labels=[Label(name="positive")],
                scores={"positive": 0.9, "negative": 0.1},
                metadata={"starter.long_strategy": "single"},
            )

        monkeypatch.setattr(starter, "classify_doc", _stub)
        tool = KaosLLMCoreClassifyTool()
        result = await tool._run(
            {"text": "good!", "labels": ["positive", "negative"]},
        )
        assert result.isError is False
        out = _struct(result)
        # JSON-mode dump renders labels as dicts with a 'name' field.
        assert out["labels"][0]["name"] == "positive"
        assert seen_labels[0].names == ("positive", "negative")

    @pytest.mark.asyncio
    async def test_labelset_object_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen_labels: list[LabelSet] = []

        async def _stub(doc: str, labels: LabelSet, **kwargs: Any) -> Classification:
            seen_labels.append(labels)
            return Classification(labels=[Label(name="A")])

        monkeypatch.setattr(starter, "classify_doc", _stub)
        labelset = LabelSet(
            labels=[Label(name="A", description="ay"), Label(name="B", description="bee")],
            exclusive=False,
        )
        tool = KaosLLMCoreClassifyTool()
        result = await tool._run(
            {"text": "go", "labels": [labelset.model_dump()]},
        )
        assert result.isError is False
        assert seen_labels[0].names == ("A", "B")
        # Multi-label / description carried through.
        assert seen_labels[0].exclusive is False
        assert seen_labels[0].by_name("A").description == "ay"

    @pytest.mark.asyncio
    async def test_empty_labels_returns_error(self) -> None:
        tool = KaosLLMCoreClassifyTool()
        result = await tool._run({"text": "x", "labels": []})
        assert result.isError is True

    @pytest.mark.asyncio
    async def test_mixed_label_types_returns_error(self) -> None:
        tool = KaosLLMCoreClassifyTool()
        result = await tool._run({"text": "x", "labels": ["A", {"name": "B"}]})
        assert result.isError is True


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_summarize_and_classify_registered_in_program_group(self) -> None:
        from kaos_core import KaosRuntime

        from kaos_llm_core.integrations.mcp.registration import (
            register_llm_core_program_tools,
        )

        runtime = KaosRuntime()
        count = register_llm_core_program_tools(runtime)
        assert count == 26  # 24 prior + summarize + classify
        names = set(runtime.tools.list_tools())
        assert "kaos-llm-core-summarize" in names
        assert "kaos-llm-core-classify" in names
