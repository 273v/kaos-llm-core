"""Unit tests for QueryExpander -- two-stage retrieval via LLM query expansion.

Tests use ``FunctionClient`` from kaos-llm-client to deterministically
simulate LLM responses without network access.

Tests cover:
1. ``ExpandQuery`` Signature validates correctly
2. ``QueryExpander`` with a mocked LLM returns expected queries
3. The original question is always included in the result
4. ``max_queries`` is respected
"""

from __future__ import annotations

import json
from typing import Any, cast

from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart

from kaos_llm_core.programs.query_expander import ExpandQuery, LLMQueryExpander
from kaos_llm_core.signatures.introspection import (
    create_output_model,
    get_input_fields,
    get_output_fields,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _json_response(data: dict[str, Any]) -> ProviderResponse:
    """Build a ProviderResponse from a dict (JSON-encoded text part)."""
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


SAMPLE_QUERIES = [
    "Rule 10b-5 prohibitions",
    "unlawful fraud deceptive devices securities",
    "manipulative deceptive practices purchase sale security",
    "10b-5 employment artifice defraud misleading",
]


def _make_expand_response(queries: list[str]) -> dict[str, Any]:
    """Build a valid ExpandQuery response dict."""
    return {"queries": queries}


def _validated_queries(data: dict[str, Any]) -> list[str]:
    model = create_output_model(ExpandQuery)
    validated = cast(Any, model.model_validate(data))
    queries = validated.queries
    assert isinstance(queries, list)
    assert all(isinstance(query, str) for query in queries)
    return cast(list[str], queries)


# ---------------------------------------------------------------------------
# 1. ExpandQuery Signature tests
# ---------------------------------------------------------------------------


class TestExpandQuerySignature:
    def test_output_model_creates_successfully(self) -> None:
        """The ExpandQuery Signature produces a valid output model."""
        model = create_output_model(ExpandQuery)
        assert model is not None
        assert "queries" in model.model_fields

    def test_has_question_input_field(self) -> None:
        """ExpandQuery has question as an input field."""
        fields = get_input_fields(ExpandQuery)
        assert "question" in fields

    def test_has_queries_output_field(self) -> None:
        """ExpandQuery has queries as an output field."""
        fields = get_output_fields(ExpandQuery)
        assert "queries" in fields

    def test_output_model_validates_query_list(self) -> None:
        """A valid list of queries validates through the output model."""
        data = _make_expand_response(SAMPLE_QUERIES)
        assert _validated_queries(data) == SAMPLE_QUERIES


# ---------------------------------------------------------------------------
# 2. QueryExpander with mocked LLM
# ---------------------------------------------------------------------------


class TestQueryExpander:
    async def test_expand_returns_queries(self) -> None:
        """QueryExpander returns the LLM-generated queries."""

        def llm_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response(_make_expand_response(SAMPLE_QUERIES))

        expander = LLMQueryExpander(model="function-test")
        expander._call._client = FunctionClient(function=llm_fn)

        result = await expander.expand("What does Rule 10b-5 prohibit?")

        # Original question should be prepended since it's not in SAMPLE_QUERIES
        assert result[0] == "What does Rule 10b-5 prohibit?"
        # All sample queries should be present (up to max_queries)
        for q in SAMPLE_QUERIES:
            assert q in result

    async def test_original_question_always_included(self) -> None:
        """The original question is always in the result, even if the LLM omits it."""
        original = "What is the filing fee?"

        def llm_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response(
                _make_expand_response(["filing fee amount", "certificate fee cost"])
            )

        expander = LLMQueryExpander(model="function-test")
        expander._call._client = FunctionClient(function=llm_fn)

        result = await expander.expand(original)
        assert original in result
        assert result[0] == original  # should be first

    async def test_original_question_not_duplicated(self) -> None:
        """If the LLM includes the original question, it is not duplicated."""
        original = "What is the filing fee?"

        def llm_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response(
                _make_expand_response([original, "filing fee amount", "certificate fee cost"])
            )

        expander = LLMQueryExpander(model="function-test")
        expander._call._client = FunctionClient(function=llm_fn)

        result = await expander.expand(original)
        assert result.count(original) == 1

    async def test_max_queries_respected(self) -> None:
        """max_queries limits the number of queries returned."""
        many_queries = [f"query {i}" for i in range(10)]

        def llm_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response(_make_expand_response(many_queries))

        expander = LLMQueryExpander(model="function-test", max_queries=3)
        expander._call._client = FunctionClient(function=llm_fn)

        result = await expander.expand("test question")

        # Original question is prepended, plus 3 from LLM = max 4
        # But the LLM queries are trimmed to max_queries (3) first,
        # then the original is inserted if not present.
        assert len(result) <= 4  # original + 3 expanded
        assert result[0] == "test question"

    async def test_max_queries_with_original_in_llm_output(self) -> None:
        """max_queries is applied before checking for original."""
        original = "query 0"  # same as first LLM query
        many_queries = [f"query {i}" for i in range(10)]

        def llm_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response(_make_expand_response(many_queries))

        expander = LLMQueryExpander(model="function-test", max_queries=3)
        expander._call._client = FunctionClient(function=llm_fn)

        result = await expander.expand(original)

        # LLM returns 10, trimmed to 3 (["query 0", "query 1", "query 2"]).
        # Original "query 0" is already in the list, so no insertion.
        assert len(result) == 3
        assert original in result
