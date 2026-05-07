"""Unit tests for kaos_llm_core.starter — one-liner convenience API."""

from __future__ import annotations

import json
from typing import Any

import pytest
from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart
from pydantic import BaseModel

from kaos_llm_core import starter
from kaos_llm_core.errors import CallError


def _json_response(data: dict[str, Any]) -> ProviderResponse:
    return ProviderResponse.model_construct(
        provider="function",
        model="function-test",
        raw={},
        parts=[ContentPart.model_construct(type="text", text=json.dumps(data))],
        usage=UsageInfo.model_construct(input_tokens=10, output_tokens=5, total_tokens=15),
        stop_reason="end_turn",
        response_id="test-id",
        status_code=200,
        response_headers={},
        latency_ms=10.0,
    )


def _client_returning(payload: dict[str, Any]) -> FunctionClient:
    def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
        return _json_response(payload)

    return FunctionClient(function=fn)


# ---------------------------------------------------------------------------
# _resolve_default_model
# ---------------------------------------------------------------------------


class TestDefaultModelResolution:
    def test_explicit_wins(self) -> None:
        assert starter._resolve_default_model("anthropic:claude-haiku-4-5") == (
            "anthropic:claude-haiku-4-5"
        )

    def test_raises_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("KAOS_LLM_CORE_DEFAULT_MODEL", raising=False)
        with pytest.raises(CallError, match="No model specified"):
            starter._resolve_default_model(None)


# ---------------------------------------------------------------------------
# text()
# ---------------------------------------------------------------------------


class TestStarterText:
    async def test_text_happy_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _client_returning({"response": "blue"})

        # Patch Call to use our client
        from kaos_llm_core.programs import call as call_mod

        def fake_resolve(self: Any, model: str) -> Any:
            return client

        monkeypatch.setattr(call_mod.Call, "_resolve_client", fake_resolve)

        result = await starter.text("Name a primary color.", model="function-test")
        assert result == "blue"

    async def test_text_raises_without_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("KAOS_LLM_CORE_DEFAULT_MODEL", raising=False)
        with pytest.raises(CallError, match="No model specified"):
            await starter.text("hi")

    def test_text_sync(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _client_returning({"response": "sync ok"})

        from kaos_llm_core.programs import call as call_mod

        monkeypatch.setattr(call_mod.Call, "_resolve_client", lambda self, model: client)
        assert starter.text_sync("hi", model="function-test") == "sync ok"


# ---------------------------------------------------------------------------
# extract()
# ---------------------------------------------------------------------------


class TestStarterExtract:
    async def test_extract_dict_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _client_returning({"name": "John", "age": 32})

        from kaos_llm_core.programs import call as call_mod

        monkeypatch.setattr(call_mod.Call, "_resolve_client", lambda self, model: client)
        result = await starter.extract(
            "John is 32.",
            {"name": str, "age": int},
            model="function-test",
        )
        assert isinstance(result, dict)
        assert result == {"name": "John", "age": 32}

    async def test_extract_model_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class Person(BaseModel):
            name: str
            age: int

        client = _client_returning({"name": "Ada", "age": 37})

        from kaos_llm_core.programs import call as call_mod

        monkeypatch.setattr(call_mod.Call, "_resolve_client", lambda self, model: client)
        result = await starter.extract("Ada is 37.", Person, model="function-test")
        assert isinstance(result, Person)
        assert result.name == "Ada"
        assert result.age == 37

    def test_extract_sync(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _client_returning({"a": 1})

        from kaos_llm_core.programs import call as call_mod

        monkeypatch.setattr(call_mod.Call, "_resolve_client", lambda self, model: client)
        result = starter.extract_sync("x", {"a": int}, model="function-test")
        assert result == {"a": 1}


# ---------------------------------------------------------------------------
# classify()
# ---------------------------------------------------------------------------


class TestStarterClassify:
    async def test_classify_single(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _client_returning({"label": "positive"})

        from kaos_llm_core.programs import call as call_mod

        monkeypatch.setattr(call_mod.Call, "_resolve_client", lambda self, model: client)
        result = await starter.classify(
            "I love this!",
            labels=["positive", "negative", "neutral"],
            model="function-test",
        )
        assert result == "positive"

    async def test_classify_multi_label(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _client_returning({"labels": ["tech", "finance"]})

        from kaos_llm_core.programs import call as call_mod

        monkeypatch.setattr(call_mod.Call, "_resolve_client", lambda self, model: client)
        result = await starter.classify(
            "Report on AI stocks",
            labels=["tech", "finance", "sports"],
            multi_label=True,
            model="function-test",
        )
        assert result == ["tech", "finance"]

    async def test_classify_empty_labels_raises(self) -> None:
        with pytest.raises(CallError, match="non-empty"):
            await starter.classify("x", labels=[], model="function-test")

    async def test_classify_unknown_label_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _client_returning({"label": "unknown_value"})

        from kaos_llm_core.programs import call as call_mod

        monkeypatch.setattr(call_mod.Call, "_resolve_client", lambda self, model: client)
        with pytest.raises(CallError, match="not in"):
            await starter.classify(
                "x",
                labels=["a", "b", "c"],
                model="function-test",
            )

    async def test_classify_case_insensitive_recovery(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = _client_returning({"label": "POSITIVE"})

        from kaos_llm_core.programs import call as call_mod

        monkeypatch.setattr(call_mod.Call, "_resolve_client", lambda self, model: client)
        result = await starter.classify(
            "yay",
            labels=["positive", "negative"],
            model="function-test",
        )
        assert result == "positive"


# ---------------------------------------------------------------------------
# summarize()
# ---------------------------------------------------------------------------


class TestStarterSummarize:
    async def test_summarize_happy_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _client_returning({"summary": "A short summary."})

        from kaos_llm_core.programs import call as call_mod

        monkeypatch.setattr(call_mod.Call, "_resolve_client", lambda self, model: client)
        result = await starter.summarize(
            "A long piece of text that needs summarization.",
            model="function-test",
            max_words=10,
            style="bullet",
        )
        assert result == "A short summary."

    def test_summarize_sync(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _client_returning({"summary": "ok"})

        from kaos_llm_core.programs import call as call_mod

        monkeypatch.setattr(call_mod.Call, "_resolve_client", lambda self, model: client)
        assert starter.summarize_sync("hello", model="function-test") == "ok"
