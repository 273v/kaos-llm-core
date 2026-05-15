"""Tests for :class:`~kaos_llm_core.programs.summarize.QueryFocusedSummary`.

Offline: a stub :class:`Embedder` returns hand-crafted unit vectors so
the cosine-based sentence ranking is deterministic; a
:class:`FunctionClient` stub drives the abstractive stage.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import numpy as np
import pytest
from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart

from kaos_llm_core.programs.summarize import QueryFocusedSummary
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


def _abstractive_fn(text: str):
    captured: list[list[dict[str, Any]]] = []

    def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
        captured.append(list(messages))
        return _resp({"summary": text})

    return fn, captured


class _StubEmbedder:
    """Returns unit vectors based on which keyword the text contains.

    Mapping: 'lease'→A axis, 'rent'→B axis, 'term'→C axis. Any other
    input falls back to the zero vector (cosine 0 against everything).
    """

    def embed(self, texts: Iterable[str], *, batch_size: int = 32) -> np.ndarray:
        rows: list[list[float]] = []
        for t in texts:
            t_lower = t.lower()
            if "lease" in t_lower:
                rows.append([1.0, 0.0, 0.0])
            elif "rent" in t_lower:
                rows.append([0.0, 1.0, 0.0])
            elif "term" in t_lower:
                rows.append([0.0, 0.0, 1.0])
            else:
                rows.append([0.1, 0.1, 0.1])  # near-zero, normalised
        return np.asarray(rows, dtype=np.float32)


_DOC = (
    "Alice signed the lease on Monday. "
    "Rent is due monthly. "
    "The term is two years. "
    "Bob countersigned on Tuesday."
)


class TestQueryFocusedSummary:
    @pytest.mark.asyncio
    async def test_query_biases_pick(self) -> None:
        # Query about rent → "Rent is due monthly." should land in
        # picks. Top-k=1 forces exactly one pick.
        fn, _ = _abstractive_fn("rent summary")
        program = QueryFocusedSummary(
            embedder=_StubEmbedder(),
            top_k=1,
            cited=False,
            model="function-test",
            client=FunctionClient(function=fn),
        )
        result = await program(text=_DOC, query="how much rent")
        assert isinstance(result, Summary)
        assert result.text == "rent summary"
        # Exactly one pick.
        assert len(result.source_spans) == 1
        # The picked span should be the rent sentence.
        span = result.source_spans[0]
        assert _DOC[span.start : span.end] == "Rent is due monthly."
        # Metadata exposes the picks (in score order).
        picks = result.metadata["picks"]
        assert len(picks) == 1
        assert picks[0]["score"] > 0.5

    @pytest.mark.asyncio
    async def test_different_query_picks_different_sentence(self) -> None:
        fn, _ = _abstractive_fn("lease summary")
        program = QueryFocusedSummary(
            embedder=_StubEmbedder(),
            top_k=1,
            cited=False,
            model="function-test",
            client=FunctionClient(function=fn),
        )
        result = await program(text=_DOC, query="who signed the lease")
        span = result.source_spans[0]
        assert _DOC[span.start : span.end] == "Alice signed the lease on Monday."

    @pytest.mark.asyncio
    async def test_top_k_returns_multiple_picks_in_source_order(self) -> None:
        fn, _ = _abstractive_fn("multi-pick summary")
        program = QueryFocusedSummary(
            embedder=_StubEmbedder(),
            top_k=3,
            cited=False,
            model="function-test",
            client=FunctionClient(function=fn),
        )
        # Generic query — all three keyword sentences should land in
        # the picks (rent, term, lease all match better than the
        # generic "Bob countersigned…").
        result = await program(text=_DOC, query="contract details")
        assert len(result.source_spans) == 3
        # Source-order ascending.
        starts = [span.start for span in result.source_spans]
        assert starts == sorted(starts)

    @pytest.mark.asyncio
    async def test_empty_query_raises(self) -> None:
        program = QueryFocusedSummary(
            embedder=_StubEmbedder(),
            top_k=1,
            cited=False,
            model="function-test",
            client=FunctionClient(function=_abstractive_fn("x")[0]),
        )
        with pytest.raises(ValueError, match="non-empty query"):
            await program(text=_DOC, query="")

    @pytest.mark.asyncio
    async def test_empty_text_returns_empty_summary(self) -> None:
        program = QueryFocusedSummary(
            embedder=_StubEmbedder(),
            top_k=1,
            cited=False,
            model="function-test",
            client=FunctionClient(function=_abstractive_fn("x")[0]),
        )
        result = await program(text="", query="anything")
        assert result.text == ""
        assert result.metadata["n_sentences"] == 0
