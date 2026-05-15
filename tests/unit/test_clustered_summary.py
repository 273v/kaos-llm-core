"""Tests for :class:`~kaos_llm_core.programs.summarize.ClusteredSummary`.

ClusteredSummary is a thin combination of ``_LongDocBase`` (chunk +
leaf summarize) and the ``Cluster`` reducer. The reducer is
exhaustively tested in ``test_cluster_reducer.py``; this module
confirms end-to-end wiring: chunks land as leaves, the Cluster
reducer runs, and the resulting Summary carries cluster metadata.
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
from kaos_nlp_core.chunking import SentenceChunker

from kaos_llm_core.programs.summarize import ClusteredSummary


def _resp(text: str) -> ProviderResponse:
    return ProviderResponse.model_construct(
        provider="function",
        model="function-test",
        raw={},
        parts=[ContentPart.model_construct(type="text", text=text)],
        usage=UsageInfo.model_construct(input_tokens=50, output_tokens=25, total_tokens=75),
        stop_reason="end_turn",
        response_id="test-id",
        status_code=200,
        response_headers={},
        latency_ms=10.0,
    )


class _CountingFn:
    def __init__(self) -> None:
        self.calls: int = 0

    def __call__(self, messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
        self.calls += 1
        # Alternate between two payload shapes:
        # - leaf summarizer expects ``summary`` field
        # - merge call expects ``summary`` field too
        return _resp(json.dumps({"summary": f"chunk-{self.calls}"}))


class _TopicEmbedder:
    """Embeds via simple keyword routing so cluster boundaries are deterministic."""

    def embed(self, texts: Iterable[str], *, batch_size: int = 32) -> np.ndarray:
        rows: list[list[float]] = []
        for t in texts:
            t_lower = t.lower()
            if "chunk-1" in t_lower or "chunk-2" in t_lower:
                rows.append([1.0, 0.0])
            else:
                rows.append([0.0, 1.0])
        return np.asarray(rows, dtype=np.float32)


# Four sentences → four chunks under SentenceChunker(max_tokens=1).
_DOC = "Sentence one. Sentence two. Sentence three. Sentence four."


class TestClusteredSummary:
    @pytest.mark.asyncio
    async def test_runs_end_to_end_with_cluster_metadata(self) -> None:
        fn = _CountingFn()
        program = ClusteredSummary(
            embedder=_TopicEmbedder(),
            k=2,
            chunker=SentenceChunker(max_tokens=1),
            model="function-test",
            client=FunctionClient(function=fn),
        )
        result = await program(text=_DOC)
        # 4 leaf summaries + 2 cluster merges + 1 final merge = 7 LLM
        # calls. (Or less if a cluster has one member — Cluster doesn't
        # call merge_fn on singletons, _LongDocBase does. Loose bound.)
        assert fn.calls >= 4  # at least one leaf per chunk
        assert result.method == "abstractive"
        assert result.metadata["reducer"] == "Cluster"
        assert result.metadata["program"] == "ClusteredSummary"
        # chunks.count is the number of source chunks.
        assert result.metadata["chunks.count"] == 4
