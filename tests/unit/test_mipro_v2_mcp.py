"""Smoke test for KaosLLMCoreMiproV2Tool (Phase 17.1 F3).

Verifies the MCP tool wrapper around MiproV2Optimizer:
* metadata (name, schema) is well-formed
* required-input validation works
* full ``_run`` flow against a FunctionClient produces a structured
  output dict with all the expected keys

The optimizer correctness itself is covered by
``test_mipro_v2_flow_mocked.py``; this test is a thin smoke check of
the wire format and the input parsing.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest
from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart

from kaos_llm_core.integrations.mcp.mipro_v2 import KaosLLMCoreMiproV2Tool
from kaos_llm_core.observability.cost import PRICING, ModelPricing


def _stub_response(payload: dict[str, Any]) -> ProviderResponse:
    return ProviderResponse.model_construct(
        provider="function",
        model="function-test",
        raw={},
        parts=[ContentPart.model_construct(type="text", text=json.dumps(payload))],
        usage=UsageInfo.model_construct(input_tokens=10, output_tokens=5, total_tokens=15),
        stop_reason="end_turn",
        status_code=200,
        response_headers={},
    )


@pytest.fixture(autouse=True)
def _pricing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        PRICING,
        "function-test",
        ModelPricing(input_per_mtok=1.0, output_per_mtok=2.0),
    )


class _Recorder:
    """Routes proposer + target requests through canned responses."""

    def __init__(self) -> None:
        self._propose_counter = 0

    def __call__(self, messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
        system = next((m["content"] for m in messages if m.get("role") == "system"), "")
        if "DescribeDataset" in system:
            return _stub_response({"dataset_description": "Short text snippets."})
        if "DescribeProgram" in system:
            return _stub_response({"program_description": "Single-call answer."})
        if "DescribeModule" in system:
            return _stub_response({"module_description": "The answer module."})
        if "GenerateSingleModuleInstruction" in system:
            self._propose_counter += 1
            return _stub_response(
                {
                    "proposed_instruction": f"Variant {self._propose_counter}",
                    "rationale": f"v{self._propose_counter}",
                }
            )
        # Target Call: always emit the canonical answer.
        return _stub_response({"answer": "yes"})


def _patch_client(client: FunctionClient) -> Any:
    from contextlib import ExitStack

    stack = ExitStack()
    stack.enter_context(patch("kaos_llm_client.create_client", return_value=client))
    stack.enter_context(patch("kaos_llm_core.programs.call.create_client", return_value=client))
    return stack


def _examples(n: int, expected: str = "yes") -> list[dict[str, Any]]:
    return [{"input": f"q{i}", "expected_output": expected} for i in range(n)]


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_name_and_display(self) -> None:
        tool = KaosLLMCoreMiproV2Tool()
        meta = tool.metadata
        assert meta.name == "kaos-llm-core-mipro-v2"
        assert meta.display_name == "LLM Core MIPRO v2"
        assert meta.module_name == "kaos-llm-core"

    def test_schema_has_required_fields(self) -> None:
        tool = KaosLLMCoreMiproV2Tool()
        meta = tool.metadata
        names = {p.name for p in meta.input_schema}
        # Required parameters per design.
        assert {"model", "train_set", "val_set"} <= names
        # Optional knobs.
        assert {
            "initial_instruction",
            "proposer_model",
            "num_candidates",
            "num_trials",
            "minibatch_size",
            "minibatch_full_eval_steps",
            "max_bootstrapped_demos",
            "max_cost_usd",
            "metric_name",
            "seed",
        } <= names

    def test_annotations_marked_open_world(self) -> None:
        tool = KaosLLMCoreMiproV2Tool()
        meta = tool.metadata
        # MIPROv2 makes external LLM calls so it's open-world; not
        # read-only because it returns a new instruction (the call
        # itself is mutated by the optimizer).
        assert meta.annotations is not None
        assert meta.annotations.openWorldHint is True
        assert meta.annotations.readOnlyHint is False


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    async def test_train_set_too_small_rejected(self) -> None:
        tool = KaosLLMCoreMiproV2Tool()
        result = await tool.execute(
            {
                "model": "function-test",
                "train_set": _examples(2),  # < 4
                "val_set": _examples(3),
            }
        )
        assert result.isError
        assert "train_set" in (result.text or "")

    async def test_empty_val_set_rejected(self) -> None:
        tool = KaosLLMCoreMiproV2Tool()
        result = await tool.execute(
            {
                "model": "function-test",
                "train_set": _examples(8),
                "val_set": [],
            }
        )
        assert result.isError
        assert "val_set" in (result.text or "")

    async def test_missing_input_key_rejected(self) -> None:
        tool = KaosLLMCoreMiproV2Tool()
        result = await tool.execute(
            {
                "model": "function-test",
                "train_set": [{"foo": "bar"} for _ in range(8)],
                "val_set": _examples(3),
            }
        )
        assert result.isError
        assert "expected_output" in (result.text or "")

    async def test_unknown_metric_name_rejected(self) -> None:
        tool = KaosLLMCoreMiproV2Tool()
        result = await tool.execute(
            {
                "model": "function-test",
                "train_set": _examples(8),
                "val_set": _examples(3),
                "metric_name": "not_a_real_metric",
            }
        )
        assert result.isError
        assert "metric_name" in (result.text or "")


# ---------------------------------------------------------------------------
# End-to-end smoke
# ---------------------------------------------------------------------------


class TestEndToEnd:
    async def test_full_run_against_function_client(self) -> None:
        rec = _Recorder()
        client = FunctionClient(function=rec)
        tool = KaosLLMCoreMiproV2Tool()

        with _patch_client(client):
            result = await tool.execute(
                {
                    "model": "function-test",
                    "proposer_model": "function-test",
                    "train_set": _examples(20),
                    "val_set": _examples(10),
                    "num_candidates": 3,
                    "num_trials": 6,
                    "minibatch_size": 4,
                    "minibatch_full_eval_steps": 3,
                    "max_bootstrapped_demos": 2,
                    "max_cost_usd": 5.0,
                    "metric_name": "exact_match",
                    "seed": 11,
                }
            )

        assert not result.isError, result.text
        out = result.structuredContent
        assert isinstance(out, dict)
        # All documented response fields present.
        for key in (
            "best_instruction",
            "n_demos_added",
            "metric_before",
            "metric_after",
            "improvement",
            "accepted",
            "trials_run",
            "proposer_calls",
            "total_cost_usd",
            "total_tokens",
            "stop_reason",
        ):
            assert key in out, f"missing key {key!r} in output: {out}"
        # Type sanity.
        assert isinstance(out["best_instruction"], str)
        assert isinstance(out["n_demos_added"], int)
        assert 0.0 <= out["metric_before"] <= 1.0
        assert 0.0 <= out["metric_after"] <= 1.0
        assert isinstance(out["accepted"], bool)
        assert out["trials_run"] >= 1
        assert out["proposer_calls"] >= 1


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_registered_in_bulk_function(self) -> None:
        # The bulk-registration function imports the tool — confirm
        # that runtime registration succeeds and that our new tool
        # name appears in the registry.
        from kaos_core import KaosRuntime

        from kaos_llm_core.integrations.mcp.registration import register_llm_core_tools

        runtime = KaosRuntime.default()
        n = register_llm_core_tools(runtime)
        # 23 (Phase 17.1) + 6 alpha tools (WS-TR.PR-6f.7) + 1 program-of-thought (#91) = 30.
        assert n == 30
        names = set(runtime.tools.list_tools())
        assert "kaos-llm-core-mipro-v2" in names
