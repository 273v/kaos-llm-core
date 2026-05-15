"""Tests for :mod:`kaos_llm_core.programs.classify`.

LLM calls are stubbed via :class:`FunctionClient` so the gate stays
offline and deterministic. Each test fixture builds a labeled stub
that mirrors the dynamically constructed Signature.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart
from kaos_nlp_core.chunking import ParagraphChunker

from kaos_llm_core.composition import (
    MajorityAggregator,
    UnionAggregator,
)
from kaos_llm_core.labels import Label, LabelSet
from kaos_llm_core.programs.classify import (
    ChunkedClassify,
    EnsembleClassify,
    FewShotClassify,
    HierarchicalClassify,
    MultiLabelClassify,
    ZeroShotClassify,
)
from kaos_llm_core.results import Classification
from kaos_llm_core.types import Example


def _json_response(data: dict[str, Any]) -> ProviderResponse:
    return ProviderResponse.model_construct(
        provider="function",
        model="function-test",
        raw={},
        parts=[ContentPart.model_construct(type="text", text=json.dumps(data))],
        usage=UsageInfo.model_construct(input_tokens=50, output_tokens=25, total_tokens=75),
        stop_reason="end_turn",
        response_id="test-id",
        status_code=200,
        response_headers={},
        latency_ms=100.0,
    )


def _stub_zero_shot(label: str, *, confidence: float = 0.9, rationale: str = "x"):
    def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
        return _json_response({"label": label, "confidence": confidence, "rationale": rationale})

    return fn


def _stub_multi_label(picks: list[str], scores: dict[str, float] | None = None):
    def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
        return _json_response(
            {
                "picked_labels": picks,
                "confidence": scores or dict.fromkeys(picks, 1.0),
                "rationale": "x",
            }
        )

    return fn


@pytest.fixture
def exclusive_labels() -> LabelSet:
    return LabelSet.from_names(["positive", "negative", "neutral"], exclusive=True)


@pytest.fixture
def multi_labels() -> LabelSet:
    return LabelSet.from_names(["finance", "legal", "tech"], exclusive=False, allow_abstain=True)


@pytest.fixture
def hierarchical_labels() -> LabelSet:
    return LabelSet(
        labels=[
            Label(name="sentiment"),
            Label(name="positive", parent="sentiment"),
            Label(name="negative", parent="sentiment"),
            Label(name="topic"),
            Label(name="sports", parent="topic"),
        ],
        hierarchical=True,
    )


# ---------------------------------------------------------------------------
# ZeroShotClassify
# ---------------------------------------------------------------------------


class TestZeroShotClassify:
    @pytest.mark.asyncio
    async def test_basic_pick(self, exclusive_labels: LabelSet) -> None:
        client = FunctionClient(function=_stub_zero_shot("positive"))
        program = ZeroShotClassify(
            labels=exclusive_labels,
            model="function-test",
            client=client,
        )
        result = await program(text="Great product!")
        assert isinstance(result, Classification)
        assert result.names == ["positive"]
        assert result.abstained is False
        assert result.scores["positive"] == 0.9

    @pytest.mark.asyncio
    async def test_abstain(self, exclusive_labels: LabelSet) -> None:
        client = FunctionClient(function=_stub_zero_shot("__abstain__"))
        program = ZeroShotClassify(
            labels=exclusive_labels,
            model="function-test",
            client=client,
        )
        result = await program(text="ambiguous text")
        assert result.abstained is True
        assert result.labels == []

    def test_rejects_multi_label_set(self, multi_labels: LabelSet) -> None:
        with pytest.raises(ValueError, match="exclusive"):
            ZeroShotClassify(labels=multi_labels)

    @pytest.mark.asyncio
    async def test_parent_id_recorded(self, exclusive_labels: LabelSet) -> None:
        client = FunctionClient(function=_stub_zero_shot("neutral"))
        program = ZeroShotClassify(
            labels=exclusive_labels,
            model="function-test",
            client=client,
        )
        result = await program(text="text", parent_id="doc-1")
        assert "doc-1" in result.chunks_used


# ---------------------------------------------------------------------------
# FewShotClassify
# ---------------------------------------------------------------------------


class TestFewShotClassify:
    @pytest.mark.asyncio
    async def test_basic(self, exclusive_labels: LabelSet) -> None:
        examples = [
            Example(
                inputs={"text": "I love this!", "labels_description": "..."},
                outputs={"label": "positive", "confidence": 0.95, "rationale": ""},
            )
        ]
        client = FunctionClient(function=_stub_zero_shot("positive"))
        program = FewShotClassify(
            labels=exclusive_labels,
            examples=examples,
            model="function-test",
            client=client,
        )
        result = await program(text="I love this!")
        assert result.names == ["positive"]

    def test_requires_examples(self, exclusive_labels: LabelSet) -> None:
        with pytest.raises(ValueError, match="at least one Example"):
            FewShotClassify(labels=exclusive_labels, examples=[])


# ---------------------------------------------------------------------------
# MultiLabelClassify
# ---------------------------------------------------------------------------


class TestMultiLabelClassify:
    @pytest.mark.asyncio
    async def test_basic(self, multi_labels: LabelSet) -> None:
        client = FunctionClient(function=_stub_multi_label(["finance", "legal"]))
        program = MultiLabelClassify(
            labels=multi_labels,
            model="function-test",
            client=client,
        )
        result = await program(text="SEC filing.")
        assert set(result.names) == {"finance", "legal"}
        assert result.abstained is False

    @pytest.mark.asyncio
    async def test_unknown_labels_filtered(self, multi_labels: LabelSet) -> None:
        client = FunctionClient(function=_stub_multi_label(["finance", "made_up_label"]))
        program = MultiLabelClassify(
            labels=multi_labels,
            model="function-test",
            client=client,
        )
        result = await program(text="x")
        assert set(result.names) == {"finance"}
        assert result.metadata["validated_pick_count"] == 1

    @pytest.mark.asyncio
    async def test_canonical_order(self, multi_labels: LabelSet) -> None:
        client = FunctionClient(function=_stub_multi_label(["tech", "finance"]))
        program = MultiLabelClassify(
            labels=multi_labels,
            model="function-test",
            client=client,
        )
        result = await program(text="x")
        # LabelSet order is finance, legal, tech. Result should respect.
        assert result.names == ["finance", "tech"]

    @pytest.mark.asyncio
    async def test_abstain_on_empty(self, multi_labels: LabelSet) -> None:
        client = FunctionClient(function=_stub_multi_label([]))
        program = MultiLabelClassify(
            labels=multi_labels,
            model="function-test",
            client=client,
        )
        result = await program(text="x")
        assert result.abstained is True

    def test_rejects_exclusive_label_set(self, exclusive_labels: LabelSet) -> None:
        with pytest.raises(ValueError, match="multi-label"):
            MultiLabelClassify(labels=exclusive_labels)


# ---------------------------------------------------------------------------
# HierarchicalClassify
# ---------------------------------------------------------------------------


class TestHierarchicalClassify:
    @pytest.mark.asyncio
    async def test_two_level_walk(self, hierarchical_labels: LabelSet) -> None:
        # Level 1: pick "sentiment" (over "topic")
        # Level 2: pick "positive" (over "negative")
        calls: list[str] = []

        def fn(messages, profile):
            allowed = "positive" in json.dumps(messages)
            if not allowed:
                calls.append("level1")
                return _json_response({"label": "sentiment", "confidence": 0.9, "rationale": ""})
            calls.append("level2")
            return _json_response({"label": "positive", "confidence": 0.8, "rationale": ""})

        client = FunctionClient(function=fn)
        program = HierarchicalClassify(
            labels=hierarchical_labels,
            model="function-test",
            client=client,
        )
        result = await program(text="I love this product.")
        assert result.metadata["hierarchical.depth"] == 2
        assert result.metadata["hierarchical.path"] == ["sentiment", "positive"]
        assert result.names == ["positive"]

    def test_rejects_flat_label_set(self, exclusive_labels: LabelSet) -> None:
        with pytest.raises(ValueError, match="hierarchical"):
            HierarchicalClassify(labels=exclusive_labels)


# ---------------------------------------------------------------------------
# ChunkedClassify
# ---------------------------------------------------------------------------


_LONG_DOC = (
    "First paragraph about positive sentiment.\n\n"
    "Second paragraph about negative sentiment.\n\n"
    "Third paragraph definitely positive.\n"
)


class TestChunkedClassify:
    @pytest.mark.asyncio
    async def test_majority_picks_modal_label(
        self,
        exclusive_labels: LabelSet,
    ) -> None:
        # 2/3 chunks pick "positive"; one picks "negative". Majority wins.
        responses = iter(["positive", "negative", "positive"])

        def fn(messages, profile):
            return _json_response({"label": next(responses), "confidence": 0.9, "rationale": ""})

        client = FunctionClient(function=fn)
        per_chunk = ZeroShotClassify(
            labels=exclusive_labels,
            model="function-test",
            client=client,
        )
        program = ChunkedClassify(
            labels=exclusive_labels,
            per_chunk=per_chunk,
            chunker=ParagraphChunker(max_tokens=11),
            aggregator=MajorityAggregator(threshold=0.5),
        )
        result = await program(text=_LONG_DOC)
        assert result.names == ["positive"]
        assert result.metadata["program"] == "ChunkedClassify"
        assert result.metadata["chunks.count"] == 3
        assert result.metadata["aggregator"] == "MajorityAggregator"

    @pytest.mark.asyncio
    async def test_union_for_multi_label(self, multi_labels: LabelSet) -> None:
        responses = iter([["finance"], ["legal"], ["finance", "tech"]])

        def fn(messages, profile):
            picks = next(responses)
            return _json_response(
                {
                    "picked_labels": picks,
                    "confidence": dict.fromkeys(picks, 1.0),
                    "rationale": "",
                }
            )

        client = FunctionClient(function=fn)
        per_chunk = MultiLabelClassify(
            labels=multi_labels,
            model="function-test",
            client=client,
        )
        program = ChunkedClassify(
            labels=multi_labels,
            per_chunk=per_chunk,
            chunker=ParagraphChunker(max_tokens=11),
            aggregator=UnionAggregator(),
        )
        result = await program(text=_LONG_DOC)
        assert set(result.names) == {"finance", "legal", "tech"}

    @pytest.mark.asyncio
    async def test_empty_text_returns_abstain(self, exclusive_labels: LabelSet) -> None:
        per_chunk = ZeroShotClassify(
            labels=exclusive_labels,
            model="function-test",
            client=FunctionClient(function=_stub_zero_shot("positive")),
        )
        program = ChunkedClassify(
            labels=exclusive_labels,
            per_chunk=per_chunk,
            chunker=ParagraphChunker(max_tokens=11),
        )
        result = await program(text="")
        assert result.abstained is True


# ---------------------------------------------------------------------------
# EnsembleClassify
# ---------------------------------------------------------------------------


class TestEnsembleClassify:
    @pytest.mark.asyncio
    async def test_majority_vote(self, exclusive_labels: LabelSet) -> None:
        # Three members; two pick positive, one picks negative.
        member_a = ZeroShotClassify(
            labels=exclusive_labels,
            model="function-test",
            client=FunctionClient(function=_stub_zero_shot("positive")),
        )
        member_b = ZeroShotClassify(
            labels=exclusive_labels,
            model="function-test",
            client=FunctionClient(function=_stub_zero_shot("positive")),
        )
        member_c = ZeroShotClassify(
            labels=exclusive_labels,
            model="function-test",
            client=FunctionClient(function=_stub_zero_shot("negative")),
        )
        program = EnsembleClassify(
            labels=exclusive_labels,
            members=[member_a, member_b, member_c],
        )
        result = await program(text="x")
        assert result.names == ["positive"]
        assert result.metadata["ensemble.members"] == 3

    def test_requires_members(self, exclusive_labels: LabelSet) -> None:
        with pytest.raises(ValueError, match="at least one member"):
            EnsembleClassify(labels=exclusive_labels, members=[])


# ---------------------------------------------------------------------------
# Top-level exposure
# ---------------------------------------------------------------------------


def test_classification_programs_exposed_from_top_level() -> None:
    import kaos_llm_core

    for name in [
        "ZeroShotClassify",
        "FewShotClassify",
        "MultiLabelClassify",
        "HierarchicalClassify",
        "ChunkedClassify",
        "EnsembleClassify",
    ]:
        assert hasattr(kaos_llm_core, name)
        assert name in kaos_llm_core.__all__
