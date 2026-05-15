"""Tests for :class:`~kaos_llm_core.programs.summarize.HybridSummary`.

Offline: a stub Ranker drives the extractive stage; a
:class:`FunctionClient` stub drives the abstractive stage.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pytest
from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart
from kaos_nlp_core.types import Segment

from kaos_llm_core.programs.summarize import HybridSummary
from kaos_llm_core.programs.summarize.extractive import RankedSegment
from kaos_llm_core.results import Summary


def _resp(data: dict[str, Any]) -> ProviderResponse:
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
        latency_ms=10.0,
    )


@dataclass(frozen=True, slots=True)
class _StubScored:
    text: str
    start: int
    end: int
    score: float
    rank: int


class _StaticRanker:
    def __init__(self, indices: list[int]) -> None:
        self._indices = indices

    def rank(
        self,
        sentences: Sequence[Segment],
        *,
        query: str | None = None,
        top_k: int | None = None,
        diversify: float = 0.0,
    ) -> Sequence[RankedSegment]:
        cap = top_k if top_k is not None else len(self._indices)
        out: list[_StubScored] = []
        for rank, i in enumerate(self._indices[:cap]):
            seg = sentences[i]
            out.append(
                _StubScored(
                    text=seg.text, start=seg.start, end=seg.end, score=1.0 - 0.1 * rank, rank=rank
                )
            )
        return out


_DOC = (
    "Alice signed the lease on Monday. "
    "Bob countersigned on Tuesday. "
    "The term is two years. "
    "Rent is due monthly."
)


def _abstractive_fn(text: str):
    def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
        return _resp({"summary": text})

    return fn


class TestHybridSummary:
    @pytest.mark.asyncio
    async def test_uncited_routes_through_abstractive(self) -> None:
        ranker = _StaticRanker(indices=[0, 2])
        client = FunctionClient(function=_abstractive_fn("synthesised summary"))
        program = HybridSummary(
            ranker=ranker, top_k=2, cited=False, model="function-test", client=client
        )
        result = await program(text=_DOC)
        assert isinstance(result, Summary)
        assert result.method == "hybrid"
        assert result.text == "synthesised summary"
        # Two extractive picks → two source spans (carried through
        # the abstractive step).
        assert len(result.source_spans) == 2
        assert result.metadata["program"] == "HybridSummary"
        assert result.metadata["cited"] is False
        assert result.metadata["top_k"] == 2

    @pytest.mark.asyncio
    async def test_cited_routes_through_cited_summary(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Use the constructor-interception pattern to assert routing
        # without stubbing a GroundedAnswer response.
        from kaos_llm_core.programs.summarize.abstractive import CitedSummary

        recorded: list[type] = []
        original_init = CitedSummary.__init__

        def _patched(self: CitedSummary, **kwargs: Any) -> None:
            recorded.append(CitedSummary)
            raise _Intercepted

        monkeypatch.setattr(CitedSummary, "__init__", _patched)

        with pytest.raises(_Intercepted):
            HybridSummary(
                ranker=_StaticRanker(indices=[0]),
                top_k=1,
                cited=True,
                model="function-test",
                client=FunctionClient(function=_abstractive_fn("x")),
            )
        assert recorded == [CitedSummary]
        # Restore so other tests aren't affected.
        monkeypatch.setattr(CitedSummary, "__init__", original_init)

    @pytest.mark.asyncio
    async def test_empty_extraction_returns_empty_summary(self) -> None:
        # Ranker returns no picks → empty hybrid summary, no LLM call.
        ranker = _StaticRanker(indices=[])
        client = FunctionClient(function=_abstractive_fn("should not be reached"))
        program = HybridSummary(
            ranker=ranker, top_k=2, cited=False, model="function-test", client=client
        )
        result = await program(text=_DOC)
        # An empty source text takes the extract-only short-circuit
        # but our doc is non-empty; the ranker just returned [] picks.
        # In that case the joined picks are also empty → empty path.
        assert result.text == ""
        assert result.method == "hybrid"
        assert result.metadata["picks.count"] == 0


class _Intercepted(Exception):
    """Marker raised by patched __init__ to assert wiring only."""
