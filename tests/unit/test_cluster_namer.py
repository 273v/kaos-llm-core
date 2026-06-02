"""Unit tests for ClusterNamer -- LLM topic naming for clusters.

Uses ``FunctionClient`` from kaos-llm-client to simulate the LLM
deterministically, with no network.
"""

from __future__ import annotations

import json
from typing import Any

from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart

from kaos_llm_core.programs.cluster_namer import ClusterNamer, NameCluster
from kaos_llm_core.signatures.introspection import create_output_model


def _json_response(data: dict[str, Any]) -> ProviderResponse:
    return ProviderResponse.model_construct(
        provider="function",
        model="function-test",
        raw={},
        parts=[ContentPart.model_construct(type="text", text=json.dumps(data))],
        usage=UsageInfo.model_construct(input_tokens=10, output_tokens=5, total_tokens=15),
        stop_reason="end_turn",
        status_code=200,
        response_headers={},
    )


def test_signature_output_model_has_title() -> None:
    model = create_output_model(NameCluster)
    assert "title" in model.model_fields


async def test_name_returns_title() -> None:
    def llm_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
        return _json_response({"title": "Dispositive Motions"})

    namer = ClusterNamer(model="function-test")
    namer._call._client = FunctionClient(function=llm_fn)
    title = await namer.name(
        ["summary judgment", "motion", "dismiss"],
        excerpt="The court granted the motion for summary judgment.",
    )
    assert title == "Dispositive Motions"


async def test_name_strips_whitespace() -> None:
    def llm_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
        return _json_response({"title": "  Baking Recipes  "})

    namer = ClusterNamer(model="function-test")
    namer._call._client = FunctionClient(function=llm_fn)
    assert await namer.name(["flour", "batter"], excerpt="Mix the flour.") == "Baking Recipes"


async def test_name_works_without_excerpt() -> None:
    def llm_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
        return _json_response({"title": "Tax Filings"})

    namer = ClusterNamer(model="function-test")
    namer._call._client = FunctionClient(function=llm_fn)
    assert await namer.name(["tax", "filing", "quarterly"]) == "Tax Filings"


async def test_name_all_returns_one_title_per_cluster_in_order() -> None:
    # Echo a title derived from the first keyword so order is verifiable.
    def llm_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
        blob = " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for m in messages
            for part in (
                m.get("content", []) if isinstance(m.get("content"), list) else [m.get("content")]
            )
        )
        tag = "litigation" if "judgment" in blob else "baking" if "flour" in blob else "other"
        return _json_response({"title": tag})

    namer = ClusterNamer(model="function-test")
    namer._call._client = FunctionClient(function=llm_fn)
    titles = await namer.name_all(
        [
            (["summary judgment", "motion"], "The court granted the motion."),
            (["flour", "batter"], "Mix the flour and eggs."),
        ]
    )
    assert titles == ["litigation", "baking"]
