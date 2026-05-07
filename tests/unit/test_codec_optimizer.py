"""Tests for CodecOptimizer."""

from __future__ import annotations

import json
from typing import Any

from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart

from kaos_llm_core.codecs.chat_codec import ChatCodec
from kaos_llm_core.codecs.json_codec import JSONCodec
from kaos_llm_core.codecs.xml_codec import XMLCodec
from kaos_llm_core.optimization.budget import Budget
from kaos_llm_core.optimization.codec_optimizer import (
    CodecOptimizer,
    _primary_call,
)
from kaos_llm_core.programs.call import Call
from kaos_llm_core.signatures import InputField, OutputField, Signature
from kaos_llm_core.types import Example


class _ClassifySig(Signature):
    """Classify."""

    text: str = InputField(description="Input")
    label: str = OutputField(description="Label")


def _make_call() -> Call:
    def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
        # Always return a well-formed JSON payload so any codec can parse it
        # via JSON; other codecs with different decoders may fail and score 0.
        return ProviderResponse.model_construct(
            provider="function",
            model="function-test",
            raw={},
            parts=[
                ContentPart.model_construct(
                    type="text",
                    text=json.dumps({"label": "positive"}),
                )
            ],
            usage=UsageInfo.model_construct(input_tokens=10, output_tokens=5, total_tokens=15),
            stop_reason="end_turn",
            status_code=200,
            response_headers={},
        )

    return Call(_ClassifySig, model="function-test", client=FunctionClient(function=fn))


def _exact_match(pred: Any, gold: dict[str, Any]) -> float:
    return 1.0 if getattr(pred, "label", None) == gold.get("label") else 0.0


class TestPrimaryCall:
    def test_returns_call_directly(self) -> None:
        call = _make_call()
        assert _primary_call(call) is call

    def test_raises_on_non_program(self) -> None:
        import pytest

        with pytest.raises(TypeError, match="primary Call"):
            _primary_call(object())


class TestCodecOptimizer:
    async def test_metadata_and_default_codecs(self) -> None:
        opt = CodecOptimizer(metric=_exact_match)
        assert JSONCodec in opt.codecs
        assert ChatCodec in opt.codecs
        assert XMLCodec in opt.codecs

    async def test_rejects_empty_codec_list(self) -> None:
        import pytest

        # Phase 16.5: validation moved to CodecOptimizerConfig.__post_init__.
        # The error message now references the config field name.
        with pytest.raises(ValueError, match="codecs must be None or a non-empty list"):
            CodecOptimizer(metric=_exact_match, codecs=[])

    async def test_swap_and_score(self) -> None:
        call = _make_call()
        val = [Example(inputs={"text": "good"}, outputs={"label": "positive"})]

        opt = CodecOptimizer(metric=_exact_match, codecs=[JSONCodec])
        result = await opt.optimize(call, val)

        assert result.best_codec is JSONCodec
        assert result.best_score == 1.0
        assert "JSONCodec" in result.scores_by_codec
        assert len(result.mutations) == 1
        assert result.mutations[0].mutation_type == "codec_swap"

    async def test_budget_exhaustion(self) -> None:
        call = _make_call()
        val = [Example(inputs={"text": "good"}, outputs={"label": "positive"})]

        opt = CodecOptimizer(
            metric=_exact_match,
            codecs=[JSONCodec, ChatCodec, XMLCodec],
            budget=Budget(max_trials=1),
        )
        result = await opt.optimize(call, val)
        # With max_trials=1, only one codec should be scored.
        assert len(result.scores_by_codec) == 1
