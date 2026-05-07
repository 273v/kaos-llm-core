"""Tests for Router system — CascadeRouter and RuleRouter."""

from __future__ import annotations

import json
from typing import Any

import pytest
from kaos_llm_client import ProviderResponse, UsageInfo
from kaos_llm_client.types import ContentPart

from kaos_llm_core.errors import CallError
from kaos_llm_core.router.cascade import CascadeRouter
from kaos_llm_core.router.rules import Rule, RuleRouter
from kaos_llm_core.signatures import InputField, OutputField, Signature


class ClassifySig(Signature):
    """Classify risk."""

    text: str = InputField(description="Input")
    level: str = OutputField(description="Level")
    confidence: float = OutputField(description="Confidence 0-1")


class ExtractSig(Signature):
    """Extract entities."""

    text: str = InputField(description="Input")
    entities: list[str] = OutputField(description="Entities")


# --- RuleRouter Tests ---


class TestRuleRouter:
    async def test_matches_by_signature_name(self) -> None:
        router = RuleRouter(
            rules=[
                Rule(model="model-a", signature_name="ClassifySig"),
                Rule(model="model-b", signature_name="ExtractSig"),
            ]
        )
        assert await router.select_model(ClassifySig, {}) == "model-a"
        assert await router.select_model(ExtractSig, {}) == "model-b"

    async def test_matches_by_input_values(self) -> None:
        router = RuleRouter(
            rules=[
                Rule(model="model-fast", input_matches={"priority": "low"}),
                Rule(model="model-strong", input_matches={"priority": "high"}),
            ]
        )
        assert await router.select_model(ClassifySig, {"priority": "low"}) == "model-fast"
        assert await router.select_model(ClassifySig, {"priority": "high"}) == "model-strong"

    async def test_first_match_wins(self) -> None:
        router = RuleRouter(
            rules=[
                Rule(model="first", signature_name="ClassifySig"),
                Rule(model="second", signature_name="ClassifySig"),
            ]
        )
        assert await router.select_model(ClassifySig, {}) == "first"

    async def test_default_model(self) -> None:
        router = RuleRouter(rules=[], default_model="fallback-model")
        assert await router.select_model(ClassifySig, {}) == "fallback-model"

    async def test_no_match_no_default_raises(self) -> None:
        router = RuleRouter(rules=[])
        with pytest.raises(CallError, match="no rule matched"):
            await router.select_model(ClassifySig, {})


class TestRule:
    def test_matches_signature_name(self) -> None:
        rule = Rule(model="m", signature_name="ClassifySig")
        assert rule.matches(ClassifySig, {})
        assert not rule.matches(ExtractSig, {})

    def test_matches_input_values(self) -> None:
        rule = Rule(model="m", input_matches={"key": "val"})
        assert rule.matches(ClassifySig, {"key": "val", "other": 1})
        assert not rule.matches(ClassifySig, {"key": "wrong"})
        assert not rule.matches(ClassifySig, {})

    def test_matches_both(self) -> None:
        rule = Rule(model="m", signature_name="ClassifySig", input_matches={"k": "v"})
        assert rule.matches(ClassifySig, {"k": "v"})
        assert not rule.matches(ExtractSig, {"k": "v"})
        assert not rule.matches(ClassifySig, {})


# --- CascadeRouter Tests ---


def _make_response(data: dict[str, Any], model: str = "function-test") -> ProviderResponse:
    return ProviderResponse.model_construct(
        provider="function",
        model=model,
        raw={},
        parts=[ContentPart.model_construct(type="text", text=json.dumps(data))],
        usage=UsageInfo.model_construct(input_tokens=20, output_tokens=10, total_tokens=30),
        stop_reason="end_turn",
        status_code=200,
        response_headers={},
    )


class TestCascadeRouter:
    async def test_select_model_returns_first(self) -> None:
        router = CascadeRouter(models=["model-a", "model-b"])
        assert await router.select_model(ClassifySig, {}) == "model-a"

    def test_empty_models_raises(self) -> None:
        with pytest.raises(CallError, match="at least one model"):
            CascadeRouter(models=[])

    def test_last_traces_initially_empty(self) -> None:
        router = CascadeRouter(models=["m1"])
        assert router.last_traces == []

    def test_model_used_initially_none(self) -> None:
        router = CascadeRouter(models=["m1"])
        assert router.model_used is None
