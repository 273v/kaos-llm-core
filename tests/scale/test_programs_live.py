"""Live-LLM scale tests for summarization and classification Programs.

These tests hit a real provider (OpenAI / Anthropic / Google) and
exercise the Programs against real corpora. The unit-tier
``FunctionClient`` tests prove the *plumbing*; these tests prove the
Programs work against actual LLMs on actual legal/patent text.

Cost discipline:

- Default sample sizes are deliberately tiny (3 USC sections for
  summarization, 12 USC sections for classification, 2 EDGAR
  contracts for long-doc) — full sweep typically < $0.10 on
  ``gpt-5.4-nano``.
- The model is the cheapest current-gen model whose provider key is
  in the environment. Override with ``KAOS_LLM_CORE_SCALE_MODEL``.
- All tests record total token usage + cost (via the framework's
  cost tracker) so we have provenance for what each run actually
  cost.

Gating:

- ``pytestmark = [pytest.mark.live, pytest.mark.slow]`` (from
  ``conftest.py``).
- Skipped if no provider API key is present.
- Skipped if the HF JSONL fixtures aren't available.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest
from kaos_nlp_core.chunking import ParagraphChunker
from pydantic import BaseModel

from kaos_llm_core.composition import MajorityAggregator
from kaos_llm_core.labels import Label, LabelSet
from kaos_llm_core.programs.classify import (
    ChunkedClassify,
    MultiLabelClassify,
    ZeroShotClassify,
)
from kaos_llm_core.programs.summarize import (
    AbstractiveSummary,
    HierarchicalSummary,
    MapReduceSummary,
    StructuredSummary,
)
from kaos_llm_core.results import Classification, Summary

from .conftest import record_text

# All tests in this module hit a real provider.
pytestmark = pytest.mark.live

_BENCH_DIR = Path(__file__).resolve().parent.parent.parent / "docs" / "benchmarks"


def _emit_report(name: str, payload: dict[str, Any]) -> None:
    if os.environ.get("KAOS_LLM_CORE_SCALE_NO_REPORT"):
        return
    try:
        _BENCH_DIR.mkdir(parents=True, exist_ok=True)
        (_BENCH_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# AbstractiveSummary — small USC sample
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_abstractive_summary_over_usc(
    live_model: str, usc_summarize_sample: list[dict[str, Any]]
) -> None:
    """Real LLM summarizes ~3 USC sections; provenance + non-empty text."""
    program = AbstractiveSummary(model=live_model)
    results: list[Summary] = []
    invocations: list[Any] = []
    t0 = time.perf_counter()
    for record in usc_summarize_sample:
        text = record_text(record)
        invocation = await program.invoke(text=text, parent_id=str(record["id"]))
        invocations.append(invocation)
        results.append(invocation.output)
    elapsed = time.perf_counter() - t0

    # Correctness invariants.
    for summary, record in zip(results, usc_summarize_sample, strict=True):
        assert isinstance(summary, Summary)
        assert summary.text.strip(), "summary must be non-empty"
        assert summary.method == "abstractive"
        assert str(record["id"]) in summary.chunks_used
        # Summary should be meaningfully shorter than the source.
        assert len(summary.text) < len(record_text(record)), "summary should compress the source"

    total_cost = sum(inv.usage.cost_usd or 0.0 for inv in invocations)
    total_input = sum(inv.usage.input_tokens or 0 for inv in invocations)
    total_output = sum(inv.usage.output_tokens or 0 for inv in invocations)
    _emit_report(
        "live-abstractive-summary-usc",
        {
            "program": "AbstractiveSummary",
            "model": live_model,
            "documents": len(results),
            "elapsed_seconds": round(elapsed, 2),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cost_usd": round(total_cost, 6),
            "sample_summaries": [s.text[:200] for s in results[:3]],
        },
    )


# ---------------------------------------------------------------------------
# StructuredSummary — schema-driven
# ---------------------------------------------------------------------------


class _USCSectionInfo(BaseModel):
    title_number: int
    primary_topic: str
    affected_entities: list[str]


@pytest.mark.asyncio
async def test_live_structured_summary(
    live_model: str, usc_summarize_sample: list[dict[str, Any]]
) -> None:
    """Schema-driven summarization returns a parseable Pydantic payload."""
    if not usc_summarize_sample:
        pytest.skip("USC sample empty")
    program = StructuredSummary(
        schema=_USCSectionInfo,
        model=live_model,
    )
    record = usc_summarize_sample[0]
    summary = await program(text=record_text(record), parent_id=str(record["id"]))
    assert isinstance(summary, Summary)
    assert isinstance(summary.payload, _USCSectionInfo)
    assert summary.payload.title_number > 0
    assert summary.payload.primary_topic.strip()


# ---------------------------------------------------------------------------
# MapReduceSummary — long-doc EDGAR
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_map_reduce_summary_edgar(
    live_model: str, edgar_long_sample: list[dict[str, Any]]
) -> None:
    """Long EDGAR contract chunks → per-chunk summaries → one merged."""
    if not edgar_long_sample:
        pytest.skip("EDGAR sample empty")
    program = MapReduceSummary(
        chunker=ParagraphChunker(max_tokens=512),
        model=live_model,
    )
    record = edgar_long_sample[0]
    t0 = time.perf_counter()
    invocation = await program.invoke(text=record_text(record), parent_id=str(record["id"]))
    elapsed = time.perf_counter() - t0
    summary = invocation.output
    assert isinstance(summary, Summary)
    assert summary.text.strip()
    assert summary.metadata["program"] == "MapReduceSummary"
    # At least some chunks were processed.
    assert summary.metadata["chunks.count"] >= 2
    _emit_report(
        "live-map-reduce-summary-edgar",
        {
            "program": "MapReduceSummary",
            "model": live_model,
            "elapsed_seconds": round(elapsed, 2),
            "chunks_count": summary.metadata["chunks.count"],
            "total_input_tokens": invocation.usage.input_tokens,
            "total_output_tokens": invocation.usage.output_tokens,
            "total_cost_usd": round(invocation.usage.cost_usd or 0.0, 6),
            "summary_head": summary.text[:100],
        },
    )


# ---------------------------------------------------------------------------
# HierarchicalSummary — k-ary merge tree
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_hierarchical_summary_edgar(
    live_model: str, edgar_long_sample: list[dict[str, Any]]
) -> None:
    """Tree reducer applies multiple merge levels on a long doc."""
    if not edgar_long_sample:
        pytest.skip("EDGAR sample empty")
    program = HierarchicalSummary(
        chunker=ParagraphChunker(max_tokens=400),
        model=live_model,
        branching=4,
    )
    record = edgar_long_sample[0]
    t0 = time.perf_counter()
    invocation = await program.invoke(text=record_text(record), parent_id=str(record["id"]))
    elapsed = time.perf_counter() - t0
    summary = invocation.output
    assert isinstance(summary, Summary)
    assert summary.text.strip()
    assert summary.metadata["program"] == "HierarchicalSummary"
    assert summary.metadata["chunks.count"] >= 2
    _emit_report(
        "live-hierarchical-summary-edgar",
        {
            "program": "HierarchicalSummary",
            "model": live_model,
            "elapsed_seconds": round(elapsed, 2),
            "chunks_count": summary.metadata["chunks.count"],
            "depth_applied": summary.metadata.get("reducer.depth_applied", 0),
            "total_cost_usd": round(invocation.usage.cost_usd or 0.0, 6),
            "summary_head": summary.text[:100],
        },
    )


# ---------------------------------------------------------------------------
# ZeroShotClassify — accuracy on a 3-class USC problem
# ---------------------------------------------------------------------------


_USC_LABELS = LabelSet(
    labels=[
        Label(
            name="armed_forces",
            description=(
                "United States Code Title 10 — Armed Forces. Statutes governing the "
                "Department of Defense, military branches, and military operations."
            ),
        ),
        Label(
            name="crimes",
            description=(
                "United States Code Title 18 — Crimes and Criminal Procedure. Federal "
                "criminal statutes and criminal procedure."
            ),
        ),
        Label(
            name="internal_revenue",
            description=(
                "United States Code Title 26 — Internal Revenue Code. Federal tax "
                "statutes covering income, payroll, estate, and excise taxes."
            ),
        ),
    ],
    exclusive=True,
    allow_abstain=False,
)


_TITLE_TO_LABEL = {
    10: "armed_forces",
    18: "crimes",
    26: "internal_revenue",
}


@pytest.mark.asyncio
async def test_live_zero_shot_classify_usc(
    live_model: str, usc_classify_sample: list[dict[str, Any]]
) -> None:
    """ZeroShotClassify hits ≥70% accuracy on a 3-title USC classification.

    The three titles (Armed Forces, Crimes, Internal Revenue) are
    deliberately well-separated domains. A correctly-wired ZeroShot
    classifier on any current-gen model should easily clear 70%.
    """
    if not usc_classify_sample:
        pytest.skip("USC classify sample empty")
    program = ZeroShotClassify(labels=_USC_LABELS, model=live_model)
    correct = 0
    confusion: dict[str, dict[str, int]] = {}
    invocations: list[Any] = []
    t0 = time.perf_counter()
    for record in usc_classify_sample:
        gold = _TITLE_TO_LABEL[record["_label_title"]]
        invocation = await program.invoke(text=record_text(record), parent_id=str(record["id"]))
        invocations.append(invocation)
        result: Classification = invocation.output
        predicted = result.top_label or "<abstain>"
        confusion.setdefault(gold, {}).setdefault(predicted, 0)
        confusion[gold][predicted] += 1
        if predicted == gold:
            correct += 1
    elapsed = time.perf_counter() - t0
    accuracy = correct / len(usc_classify_sample)
    total_cost = sum(inv.usage.cost_usd or 0.0 for inv in invocations)

    _emit_report(
        "live-zero-shot-classify-usc",
        {
            "program": "ZeroShotClassify",
            "model": live_model,
            "documents": len(usc_classify_sample),
            "accuracy": round(accuracy, 3),
            "confusion": confusion,
            "elapsed_seconds": round(elapsed, 2),
            "total_cost_usd": round(total_cost, 6),
        },
    )
    assert accuracy >= 0.70, (
        f"ZeroShotClassify accuracy {accuracy:.2f} below 0.70 floor "
        f"on 3-class USC problem (model={live_model}, confusion={confusion})"
    )


# ---------------------------------------------------------------------------
# MultiLabelClassify — EDGAR contract topic tags
# ---------------------------------------------------------------------------


_EDGAR_LABELS = LabelSet(
    labels=[
        Label(
            name="confidentiality",
            description="Obligations to keep information confidential or secret.",
        ),
        Label(
            name="indemnification",
            description="Indemnity, hold-harmless, or liability-shifting clauses.",
        ),
        Label(
            name="termination",
            description="Conditions, notice periods, or remedies for ending the agreement.",
        ),
        Label(
            name="governing_law",
            description="Choice of law, venue, or jurisdiction clauses.",
        ),
        Label(
            name="payment",
            description="Pricing, fees, royalties, payment schedules, or invoicing.",
        ),
    ],
    exclusive=False,
    allow_abstain=True,
)


@pytest.mark.asyncio
async def test_live_multi_label_classify_edgar(
    live_model: str, edgar_long_sample: list[dict[str, Any]]
) -> None:
    """MultiLabelClassify picks ≥2 topics on a typical EDGAR contract.

    A real commercial contract almost always contains governing-law,
    termination, and at least one other clause from the set above. A
    classifier returning fewer than 2 labels on a 13K-word contract
    is almost certainly broken.
    """
    if not edgar_long_sample:
        pytest.skip("EDGAR sample empty")
    program = MultiLabelClassify(labels=_EDGAR_LABELS, model=live_model)
    invocation = await program.invoke(
        text=record_text(edgar_long_sample[0])[:8000],  # keep context bounded
        parent_id=str(edgar_long_sample[0].get("id", "edgar")),
    )
    result: Classification = invocation.output
    _emit_report(
        "live-multi-label-classify-edgar",
        {
            "program": "MultiLabelClassify",
            "model": live_model,
            "picked_labels": result.names,
            "scores": result.scores,
            "rationale": (result.rationale or "")[:100],
            "total_cost_usd": round(invocation.usage.cost_usd or 0.0, 6),
        },
    )
    assert len(result.names) >= 2, (
        f"Expected ≥2 labels on a long EDGAR contract; got {result.names}"
    )


# ---------------------------------------------------------------------------
# ChunkedClassify — long-doc classification dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_chunked_classify_edgar(
    live_model: str, edgar_long_sample: list[dict[str, Any]]
) -> None:
    """ChunkedClassify dispatches per-chunk and aggregates via majority.

    Validates that the per-chunk classifier actually runs over
    multiple chunks (chunks.count ≥ 2 in the result metadata) and
    that the aggregated result is reasonable (any label or abstain).
    """
    if not edgar_long_sample:
        pytest.skip("EDGAR sample empty")
    # Compact label set; one chunk should map to one label cleanly.
    label_set = LabelSet(
        labels=[
            Label(name="obligation", description="Affirmative duty owed by a party."),
            Label(name="definition", description="Defined term clause."),
            Label(
                name="boilerplate", description="Generic clause: notices, waivers, severability."
            ),
        ],
        exclusive=True,
        allow_abstain=True,
    )
    per_chunk = ZeroShotClassify(labels=label_set, model=live_model)
    program = ChunkedClassify(
        labels=label_set,
        per_chunk=per_chunk,
        chunker=ParagraphChunker(max_tokens=400),
        aggregator=MajorityAggregator(threshold=0.4),
        max_concurrency=4,
    )
    record = edgar_long_sample[0]
    t0 = time.perf_counter()
    invocation = await program.invoke(
        text=record_text(record)[:6000],
        parent_id=str(record.get("id", "edgar")),
    )
    elapsed = time.perf_counter() - t0
    result: Classification = invocation.output
    _emit_report(
        "live-chunked-classify-edgar",
        {
            "program": "ChunkedClassify",
            "model": live_model,
            "chunks_count": result.metadata.get("chunks.count", 0),
            "label_histogram": result.metadata.get("aggregator.label_histogram", {}),
            "elapsed_seconds": round(elapsed, 2),
            "result_labels": result.names,
            "abstained": result.abstained,
            "total_cost_usd": round(invocation.usage.cost_usd or 0.0, 6),
        },
    )
    assert result.metadata.get("chunks.count", 0) >= 2, (
        "ChunkedClassify should have dispatched to ≥2 chunks on a long EDGAR doc"
    )
    assert result.metadata.get("program") == "ChunkedClassify"
