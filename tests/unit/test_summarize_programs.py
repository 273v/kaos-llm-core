"""Tests for :mod:`kaos_llm_core.programs.summarize`.

The LLM calls are stubbed via :class:`FunctionClient`, so these tests
stay offline and deterministic. Live integration tests against real
providers live behind ``@pytest.mark.live``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart
from kaos_nlp_core.chunking import ParagraphChunker
from pydantic import BaseModel

from kaos_llm_core.programs.summarize import (
    AbstractiveSummary,
    CitedSummary,
    HierarchicalSummary,
    MapReduceSummary,
    RefineSummary,
    StructuredSummary,
)
from kaos_llm_core.results import Summary
from kaos_llm_core.signatures.grounding import Answer


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


def _abstractive_fn(summary_text: str):
    def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
        return _json_response({"summary": summary_text})

    return fn


# ---------------------------------------------------------------------------
# AbstractiveSummary
# ---------------------------------------------------------------------------


class TestAbstractiveSummary:
    @pytest.mark.asyncio
    async def test_returns_summary_wrapper(self) -> None:
        client = FunctionClient(function=_abstractive_fn("This is the summary."))
        program = AbstractiveSummary(model="function-test", client=client)
        result = await program(text="Some long source text.")
        assert isinstance(result, Summary)
        assert result.text == "This is the summary."
        assert result.method == "abstractive"
        assert result.metadata["program"] == "AbstractiveSummary"

    @pytest.mark.asyncio
    async def test_parent_id_recorded(self) -> None:
        client = FunctionClient(function=_abstractive_fn("X"))
        program = AbstractiveSummary(model="function-test", client=client)
        result = await program(text="src", parent_id="doc-123")
        assert "doc-123" in result.chunks_used

    @pytest.mark.asyncio
    async def test_invocation_carries_trace_and_usage(self) -> None:
        client = FunctionClient(function=_abstractive_fn("Y"))
        program = AbstractiveSummary(model="function-test", client=client)
        invocation = await program.invoke(text="src")
        assert invocation.trace is not None
        assert invocation.usage.input_tokens >= 50
        assert invocation.usage.output_tokens >= 25


# ---------------------------------------------------------------------------
# StructuredSummary
# ---------------------------------------------------------------------------


class _ContractPayload(BaseModel):
    parties: list[str]
    term_years: int


class TestStructuredSummary:
    @pytest.mark.asyncio
    async def test_returns_typed_payload(self) -> None:
        def fn(messages, profile):
            return _json_response(
                {
                    "summary": "A two-year contract between Alpha and Beta.",
                    "payload": {"parties": ["Alpha", "Beta"], "term_years": 2},
                }
            )

        client = FunctionClient(function=fn)
        program = StructuredSummary(
            schema=_ContractPayload,
            model="function-test",
            client=client,
        )
        result = await program(text="Contract text...")
        assert result.text.startswith("A two-year contract")
        assert isinstance(result.payload, _ContractPayload)
        assert result.payload.parties == ["Alpha", "Beta"]
        assert result.payload.term_years == 2
        assert result.metadata["schema"] == "_ContractPayload"


# ---------------------------------------------------------------------------
# CitedSummary
# ---------------------------------------------------------------------------


class TestCitedSummary:
    @pytest.mark.asyncio
    async def test_returns_grounded_summary(self) -> None:
        # A minimal Answer payload with one claim and one supporting span.
        source = "Alpha entered into an agreement with Beta on 2026-01-15."
        grounded_payload = {
            "kind": "answer",
            "value": "Alpha and Beta signed an agreement on 2026-01-15.",
            "claims": [
                {
                    "claim_type": "factual",
                    "statement": "Alpha and Beta signed an agreement on 2026-01-15.",
                    "supporting_spans": [
                        {
                            "source_uri": "doc-1",
                            "char_span": [0, len(source)],
                            "quote": source,
                        }
                    ],
                    "confidence": 0.9,
                }
            ],
            "confidence": 0.9,
        }

        def fn(messages, profile):
            return _json_response({"summary": grounded_payload})

        client = FunctionClient(function=fn)
        program = CitedSummary(model="function-test", client=client)
        result = await program(text=source, parent_id="doc-1")
        assert result.text == "Alpha and Beta signed an agreement on 2026-01-15."
        assert isinstance(result.payload, Answer)
        assert len(result.payload.claims) == 1
        # source_spans pulled from the supporting span — and the span
        # actually verifies against the source.
        assert len(result.source_spans) == 1
        assert result.source_spans[0].start == 0
        assert result.source_spans[0].end == len(source)
        assert result.metadata["program"] == "CitedSummary"
        assert result.metadata["cited.verified_claim_count"] == 1
        assert result.metadata["cited.unverified_claim_count"] == 0
        assert result.metadata["cited.verified_ratio"] == 1.0
        assert result.metadata["cited.refused"] is False

    @pytest.mark.asyncio
    async def test_hallucinated_quote_is_excluded_from_source_spans(self) -> None:
        """A claim whose quote doesn't appear in the source must NOT be
        promoted to ``Summary.source_spans``. The full payload is preserved
        for diagnosis, but the curated span list reflects only verified
        evidence.
        """
        source = "Alpha entered into an agreement with Beta on 2026-01-15."
        fabricated_quote = "Alpha promised Beta exclusive rights forever."
        grounded_payload = {
            "kind": "answer",
            "value": "Alpha and Beta signed an agreement on 2026-01-15.",
            "claims": [
                {
                    "claim_type": "factual",
                    "statement": "Alpha and Beta signed an agreement on 2026-01-15.",
                    "supporting_spans": [
                        {
                            "source_uri": "doc-1",
                            "char_span": [0, len(source)],
                            "quote": source,  # real span: verifies
                        }
                    ],
                    "confidence": 0.9,
                },
                {
                    "claim_type": "factual",
                    "statement": "Alpha promised Beta exclusive rights forever.",
                    "supporting_spans": [
                        {
                            "source_uri": "doc-1",
                            "char_span": [0, len(fabricated_quote)],
                            "quote": fabricated_quote,  # hallucinated
                        }
                    ],
                    "confidence": 0.9,
                },
            ],
            "confidence": 0.9,
        }

        def fn(messages, profile):
            return _json_response({"summary": grounded_payload})

        client = FunctionClient(function=fn)
        program = CitedSummary(model="function-test", client=client)
        result = await program(text=source, parent_id="doc-1")

        # Full payload preserved (caller can inspect what the LLM said).
        assert isinstance(result.payload, Answer)
        assert len(result.payload.claims) == 2
        # source_spans contains the VERIFIED claim only (1 of 2).
        assert len(result.source_spans) == 1
        assert result.metadata["cited.claim_count"] == 2
        assert result.metadata["cited.verified_claim_count"] == 1
        assert result.metadata["cited.unverified_claim_count"] == 1
        assert result.metadata["cited.verified_ratio"] == 0.5
        # Without a refuse_below floor, the verified summary text stays.
        assert result.text == "Alpha and Beta signed an agreement on 2026-01-15."

    @pytest.mark.asyncio
    async def test_refuse_below_collapses_summary_when_too_few_verify(self) -> None:
        """With ``refuse_below=0.6`` and one of two claims verifying
        (ratio=0.5), the Program must refuse: empty text + empty
        source_spans + ``cited.refused=True``.
        """
        source = "Alpha entered into an agreement with Beta on 2026-01-15."
        fabricated = "Alpha promised Beta exclusive rights forever."
        grounded_payload = {
            "kind": "answer",
            "value": "Alpha and Beta signed an agreement on 2026-01-15.",
            "claims": [
                {
                    "claim_type": "factual",
                    "statement": "real",
                    "supporting_spans": [
                        {
                            "source_uri": "doc-1",
                            "char_span": [0, len(source)],
                            "quote": source,
                        }
                    ],
                    "confidence": 0.9,
                },
                {
                    "claim_type": "factual",
                    "statement": "fake",
                    "supporting_spans": [
                        {
                            "source_uri": "doc-1",
                            "char_span": [0, len(fabricated)],
                            "quote": fabricated,
                        }
                    ],
                    "confidence": 0.9,
                },
            ],
            "confidence": 0.9,
        }

        def fn(messages, profile):
            return _json_response({"summary": grounded_payload})

        client = FunctionClient(function=fn)
        program = CitedSummary(
            model="function-test",
            client=client,
            refuse_below=0.6,
        )
        result = await program(text=source, parent_id="doc-1")
        assert result.text == ""
        assert result.source_spans == []
        assert result.metadata["cited.refused"] is True
        assert result.metadata["cited.verified_ratio"] == 0.5
        # Payload still carries the full LLM output for diagnostic use.
        assert isinstance(result.payload, Answer)
        assert len(result.payload.claims) == 2

    def test_refuse_below_validation(self) -> None:
        with pytest.raises(ValueError, match=r"refuse_below must be in"):
            CitedSummary(refuse_below=1.5)
        with pytest.raises(ValueError, match=r"refuse_below must be in"):
            CitedSummary(refuse_below=-0.1)


# ---------------------------------------------------------------------------
# Long-doc Programs
# ---------------------------------------------------------------------------


_LONG_DOC = (
    "First paragraph about Alpha. Some details here.\n"
    "\n"
    "Second paragraph about Beta. More details.\n"
    "\n"
    "Third paragraph about Gamma. Final details.\n"
)


class _MultiSignatureStub:
    """Stub responder that handles both leaf summarize and merge calls.

    The leaf summarize signature has key ``summary``; the merge
    signature also outputs ``summary`` but accepts ``summaries`` as
    input. Dispatch is by scanning the rendered prompt text for the
    merge-call instructional cue.
    """

    def __init__(self) -> None:
        self.counts: dict[str, int] = {"leaf": 0, "merge": 0}

    def __call__(
        self,
        messages: list[dict[str, Any]],
        profile: ModelProfile,
    ) -> ProviderResponse:
        combined = " ".join(
            part.get("text", "") if isinstance(part, dict) else ""
            for m in messages
            for part in (
                m.get("content", [])
                if isinstance(m.get("content"), list)
                else [{"text": m.get("content", "")}]
            )
        )
        if "summaries" in combined and "Ordered list" in combined:
            self.counts["merge"] += 1
            return _json_response({"summary": f"merged-{self.counts['merge']}"})
        self.counts["leaf"] += 1
        return _json_response({"summary": f"leaf-{self.counts['leaf']}"})


class TestMapReduceSummary:
    @pytest.mark.asyncio
    async def test_chunks_and_merges(self) -> None:
        fn = _MultiSignatureStub()
        client = FunctionClient(function=fn)
        program = MapReduceSummary(
            chunker=ParagraphChunker(max_tokens=10),
            model="function-test",
            client=client,
        )
        result = await program(text=_LONG_DOC, parent_id="doc-1")
        assert isinstance(result, Summary)
        # The merge result should be one of the merged-* strings.
        assert result.text.startswith("merged-") or result.text.startswith("leaf-")
        # We expected ≥3 leaves (3 paragraphs) and at least one merge.
        assert fn.counts["leaf"] >= 3
        assert fn.counts["merge"] >= 1
        assert result.metadata["program"] == "MapReduceSummary"
        assert result.metadata["chunks.count"] >= 3

    @pytest.mark.asyncio
    async def test_empty_input(self) -> None:
        fn = _MultiSignatureStub()
        client = FunctionClient(function=fn)
        program = MapReduceSummary(model="function-test", client=client)
        result = await program(text="")
        assert result.text == ""
        assert result.metadata["chunks.count"] == 0


class TestRefineSummary:
    @pytest.mark.asyncio
    async def test_sequential_refinement(self) -> None:
        fn = _MultiSignatureStub()
        client = FunctionClient(function=fn)
        program = RefineSummary(
            chunker=ParagraphChunker(max_tokens=10),
            model="function-test",
            client=client,
        )
        result = await program(text=_LONG_DOC, parent_id="doc-1")
        assert result.metadata["program"] == "RefineSummary"
        # Refine over 3 leaves does (leaf1, leaf2) merge then (running, leaf3)
        # → 2 merge calls.
        assert fn.counts["leaf"] >= 3
        assert fn.counts["merge"] >= 1


class TestHierarchicalSummary:
    @pytest.mark.asyncio
    async def test_tree_merges(self) -> None:
        fn = _MultiSignatureStub()
        client = FunctionClient(function=fn)
        program = HierarchicalSummary(
            chunker=ParagraphChunker(max_tokens=10),
            branching=2,
            model="function-test",
            client=client,
        )
        result = await program(text=_LONG_DOC, parent_id="doc-1")
        assert result.metadata["program"] == "HierarchicalSummary"
        assert result.metadata["chunks.count"] >= 3
        # Tree with branching=2 over 3 leaves: 1 merge call at level 1
        # (the 3-leaf singleton-tail fold), no level 2.
        assert fn.counts["leaf"] >= 3
        assert fn.counts["merge"] >= 1


# ---------------------------------------------------------------------------
# Top-level exposure
# ---------------------------------------------------------------------------


def test_programs_exposed_from_top_level() -> None:
    import kaos_llm_core

    for name in [
        "AbstractiveSummary",
        "StructuredSummary",
        "CitedSummary",
        "MapReduceSummary",
        "RefineSummary",
        "HierarchicalSummary",
    ]:
        assert hasattr(kaos_llm_core, name)
        assert name in kaos_llm_core.__all__
