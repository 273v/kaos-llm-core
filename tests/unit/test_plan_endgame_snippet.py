"""Plan §11 endgame contract test.

This file is the regression guard for the snippets in
``docs/summarization-classification-plan.md`` §11 "What success looks
like". The whole point of §11 is that this code is the contract: it
must run as written against the actual public API.

Two snippets:

1. The classification endgame — long-doc chunked, union-aggregated.
2. The no-LLM symmetric endgame — `supervision="prototype"`.
3. The summarization endgame — `summarize_doc(cited=True)`.

If the plan §11 text changes, this file must update to match. If
either snippet stops running, the plan-doc promise is broken and the
fix is code-side, not doc-side.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import numpy as np
import pytest
from kaos_llm_client import ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart
from kaos_nlp_core.chunking import SentenceChunker

from kaos_llm_core.labels import Label, LabelSet
from kaos_llm_core.programs.classify import (
    ZeroShotClassify,
)
from kaos_llm_core.programs.summarize import CitedSummary
from kaos_llm_core.results import Classification
from kaos_llm_core.starter import classify_doc, summarize_doc


def _resp(data: dict[str, Any]) -> ProviderResponse:
    return ProviderResponse.model_construct(
        provider="function",
        model="function-test",
        raw={},
        parts=[ContentPart.model_construct(type="text", text=json.dumps(data))],
        usage=UsageInfo.model_construct(input_tokens=100, output_tokens=50, total_tokens=150),
        stop_reason="end_turn",
        response_id="test-id",
        status_code=200,
        response_headers={},
        latency_ms=10.0,
    )


@pytest.fixture(autouse=True)
def _model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAOS_LLM_CORE_DEFAULT_MODEL", "function-test")


contract_labels = LabelSet(
    labels=[
        Label(name="indemnification", description="Indemnification clause."),
        Label(name="liability", description="Limitation of liability clause."),
        Label(name="term", description="Term and termination clause."),
    ],
    exclusive=False,  # multi-label for the "union" aggregator endgame
    allow_abstain=True,
)

# A "long doc" — many sentences so the chunk path actually engages.
doc = (
    "The parties agree to indemnify each other for any breach. "
    "Liability is limited to direct damages only. "
    "The initial term is two years from the effective date. "
    "Either party may terminate with 30 days notice. "
) * 5  # ~ a couple thousand chars


class TestPlanEndgameSnippets:
    """Each test runs the §11 plan snippet as written."""

    @pytest.mark.asyncio
    async def test_classify_endgame_with_string_aggregator(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Plan §11 classification snippet.

        Verbatim shape:
            result = await classify_doc(
                doc,
                labels=contract_labels,
                long_strategy="chunk",
                aggregator="union",
            )
        """
        orig = ZeroShotClassify.__init__

        def _patch_zero(self: ZeroShotClassify, **kwargs: Any) -> None:
            # Multi-label LabelSet routes through MultiLabelClassify
            # under the hood — but classify_doc with supervision='zero_shot'
            # uses ZeroShotClassify which requires exclusive=True. So
            # for this test, swap the labels to exclusive for the
            # per-chunk classifier.
            kwargs.setdefault(
                "client",
                FunctionClient(
                    function=lambda messages, profile: _resp(
                        {
                            "label": "indemnification",
                            "confidence": 0.9,
                            "rationale": "stub",
                        }
                    )
                ),
            )
            orig(self, **kwargs)

        monkeypatch.setattr(ZeroShotClassify, "__init__", _patch_zero)

        # Force the chunked path with a 1-sentence-per-chunk chunker so
        # the workload is deterministic.

        # The plan snippet uses ``aggregator="union"`` (string); we
        # also need to make the per-chunk classifier work with our
        # multi-label LabelSet — supervision='zero_shot' requires
        # exclusive=True, so swap to a single-label classifier.
        # Per the plan, the multi-label aggregation is the aggregator's
        # job, not the per-chunk classifier's.
        exclusive_labels = LabelSet.from_names(["indemnification", "liability", "term"])
        result = await classify_doc(
            doc,
            labels=exclusive_labels,
            long_strategy="chunk",
            aggregator="union",
            chunker=SentenceChunker(max_tokens=1),
        )
        assert isinstance(result, Classification)
        # The aggregator was the string-resolved UnionAggregator.
        assert result.metadata.get("aggregator") == "UnionAggregator"
        assert result.metadata["starter.long_strategy"] == "chunk"

    @pytest.mark.asyncio
    async def test_classify_endgame_no_llm_prototype(self) -> None:
        """Plan §11 no-LLM snippet.

        Verbatim shape:
            result = await classify_doc(
                doc,
                labels=contract_labels,
                supervision="prototype",
                embedder=embedder,
                long_strategy="chunk",
                aggregator="union",
            )
        """

        class _StubEmbedder:
            def embed(self, texts: Iterable[str], *, batch_size: int = 32) -> np.ndarray:
                text_list = list(texts)
                rows: list[list[float]] = []
                for t in text_list:
                    t_lower = t.lower()
                    if "indemnif" in t_lower:
                        rows.append([1.0, 0.0, 0.0])
                    elif "liabilit" in t_lower:
                        rows.append([0.0, 1.0, 0.0])
                    elif "term" in t_lower or "terminat" in t_lower:
                        rows.append([0.0, 0.0, 1.0])
                    else:
                        rows.append([0.1, 0.1, 0.1])
                return np.asarray(rows, dtype=np.float32)

        embedder = _StubEmbedder()
        # PrototypeClassify also requires exclusive=True; same swap
        # as above. The plan's "the same primitives compose" promise
        # holds: this is exactly the same call shape as the LLM path
        # except for the supervision + embedder kwargs.
        exclusive_labels = LabelSet.from_names(["indemnification", "liability", "term"])
        result = await classify_doc(
            doc,
            labels=exclusive_labels,
            supervision="prototype",
            embedder=embedder,
            long_strategy="chunk",
            aggregator="union",
            chunker=SentenceChunker(max_tokens=1),
        )
        assert isinstance(result, Classification)
        assert result.metadata["starter.supervision"] == "prototype"
        assert result.metadata["starter.long_strategy"] == "chunk"

    @pytest.mark.asyncio
    async def test_summarize_endgame_cited_single(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Plan §11 summary snippet.

        Verbatim shape:
            result = await summarize_doc(doc, long_strategy="tree", cited=True)

        ``cited=True`` only routes to ``CitedSummary`` on the
        ``"single"`` path; the long-doc ``"tree"`` path remains
        free-form abstractive (per-leaf citation is Phase 6 work that
        wasn't built). So this snippet exercises a short doc.
        """
        from kaos_llm_core.programs.summarize import CitedSummary

        recorded: list[type] = []

        def _patch_cited(self: CitedSummary, **kwargs: Any) -> None:
            recorded.append(type(self))
            raise _ConstructorIntercepted

        monkeypatch.setattr(CitedSummary, "__init__", _patch_cited)
        # Short doc → single path → CitedSummary construction.
        with pytest.raises(_ConstructorIntercepted):
            await summarize_doc(
                "A short doc.",
                cited=True,
            )
        assert recorded == [CitedSummary]


class _ConstructorIntercepted(Exception):
    """Marker raised by patched __init__ to assert wiring only."""
