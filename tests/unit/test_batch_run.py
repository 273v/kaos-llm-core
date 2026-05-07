"""Core ``batch_run()`` primitive tests using FunctionClient.

Drives the Phase 15.2 batch primitive against a deterministic
``FunctionClient`` so the cost-attribution / concurrency / error-policy
contract is testable without hitting a real provider.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart

from kaos_llm_core.observability.cost import PRICING, ModelPricing
from kaos_llm_core.programs.batch import (
    BatchError,
    BatchItem,
    JsonlBatchWriter,
    ListInputSource,
    batch_run,
    list_input_source,
)
from kaos_llm_core.programs.call import Call
from kaos_llm_core.signatures import InputField, OutputField, Signature

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _Sig(Signature):
    """Echo the input."""

    text: str = InputField(description="Input text")
    answer: str = OutputField(description="Echoed answer")


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


def _make_call(answer: str = "ok") -> Call:
    """Build a Call wired to a deterministic FunctionClient."""

    def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
        return _json_response({"answer": answer})

    call = Call(_Sig, model="function-test")
    call._client = FunctionClient(function=fn)
    return call


def _make_failing_call(fail_indices: set[int]) -> Call:
    """Build a Call that raises on the i-th invocation iff i is in fail_indices."""
    call_count = {"n": 0}

    def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
        i = call_count["n"]
        call_count["n"] += 1
        if i in fail_indices:
            raise RuntimeError(f"simulated provider error on call {i}")
        return _json_response({"answer": f"ok_{i}"})

    call = Call(_Sig, model="function-test", max_retries=0)
    call._client = FunctionClient(function=fn)
    return call


@pytest.fixture(autouse=True)
def _pricing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject pricing for the function-test model so cost > 0 in assertions."""
    monkeypatch.setitem(
        PRICING,
        "function-test",
        ModelPricing(input_per_mtok=1.0, output_per_mtok=2.0),
    )


def _items(n: int, *, program_hash_value: str = "sha256:test") -> ListInputSource:
    return list_input_source(
        [{"text": f"item-{i}"} for i in range(n)],
        program_hash_value=program_hash_value,
    )


# ---------------------------------------------------------------------------
# Empty / single / many item runs
# ---------------------------------------------------------------------------


class TestBatchRunSizes:
    async def test_empty_source_completes(self, tmp_path: Path) -> None:
        call = _make_call()
        result = await batch_run(
            call,
            _items(0),
            output_dir=str(tmp_path / "empty"),
            resume=False,
        )
        assert result.n_total == 0
        assert result.n_succeeded == 0
        assert result.n_errored == 0
        assert result.status == "completed"
        # Manifest is still written
        manifest_path = Path(result.manifest_path)
        assert manifest_path.exists()

    async def test_single_item(self, tmp_path: Path) -> None:
        call = _make_call("hello")
        result = await batch_run(
            call,
            _items(1),
            output_dir=str(tmp_path / "single"),
            resume=False,
        )
        assert result.n_total == 1
        assert result.n_succeeded == 1
        assert result.n_errored == 0
        # Log file has header + 1 item
        log_lines = Path(result.log_path).read_text().strip().split("\n")
        assert len(log_lines) == 2
        header = json.loads(log_lines[0])
        item = json.loads(log_lines[1])
        assert header["kind"] == "header"
        assert item["kind"] == "item"
        assert item["status"] == "success"
        assert item["output"] == {"answer": "hello"}

    async def test_many_items(self, tmp_path: Path) -> None:
        call = _make_call()
        result = await batch_run(
            call,
            _items(50),
            output_dir=str(tmp_path / "many"),
            max_concurrency=8,
            resume=False,
        )
        assert result.n_total == 50
        assert result.n_succeeded == 50
        assert result.n_errored == 0


# ---------------------------------------------------------------------------
# Error policies
# ---------------------------------------------------------------------------


class TestErrorPolicies:
    async def test_continue_default_collects_errors(self, tmp_path: Path) -> None:
        call = _make_failing_call({2, 5})  # 2 of 10 fail
        result = await batch_run(
            call,
            _items(10),
            output_dir=str(tmp_path / "continue"),
            max_concurrency=1,  # serial so the failures are deterministic
            error_policy="continue",
            resume=False,
        )
        assert result.n_total == 10
        assert result.n_succeeded == 8
        assert result.n_errored == 2
        assert result.errors_by_type.get("CallError", 0) >= 1
        assert result.status == "completed"

    async def test_stop_aborts_on_first_failure(self, tmp_path: Path) -> None:
        call = _make_failing_call({3})
        result = await batch_run(
            call,
            _items(10),
            output_dir=str(tmp_path / "stop"),
            max_concurrency=1,
            error_policy="stop",
            resume=False,
        )
        # Stop policy aborted; partial results in the log
        assert result.status == "stopped"
        assert result.n_errored >= 1
        # Subsequent items not processed
        assert result.n_succeeded < 10

    async def test_stop_after_n(self, tmp_path: Path) -> None:
        call = _make_failing_call({1, 3, 5, 7})
        result = await batch_run(
            call,
            _items(10),
            output_dir=str(tmp_path / "stop_n"),
            max_concurrency=1,
            error_policy="stop_after_n",
            max_errors=2,
            resume=False,
        )
        assert result.status == "stopped"
        # The cap was 2 errors; we should see exactly 2 (or the boundary
        # in the case of concurrent fan-out, but max_concurrency=1 makes
        # this strict).
        assert result.n_errored == 2

    async def test_stop_after_n_requires_max_errors(self, tmp_path: Path) -> None:
        with pytest.raises(BatchError, match="max_errors"):
            await batch_run(
                _make_call(),
                _items(1),
                output_dir=str(tmp_path / "bad"),
                error_policy="stop_after_n",
                resume=False,
            )


# ---------------------------------------------------------------------------
# Cost attribution
# ---------------------------------------------------------------------------


class TestCostAttribution:
    async def test_cost_usd_aggregates_per_item(self, tmp_path: Path) -> None:
        call = _make_call()
        result = await batch_run(
            call,
            _items(10),
            output_dir=str(tmp_path / "cost"),
            resume=False,
        )
        # Each item is 10+5=15 tokens at $1/M input + $2/M output = small but > 0
        assert result.cost_usd > 0
        assert result.total_tokens == 150  # 10 items times 15 tokens each
        assert result.input_tokens == 100
        assert result.output_tokens == 50

    async def test_jsonl_log_per_item_cost_sums_to_total(self, tmp_path: Path) -> None:
        call = _make_call()
        result = await batch_run(
            call,
            _items(5),
            output_dir=str(tmp_path / "log_cost"),
            resume=False,
        )
        # Sum the per-item cost from the JSONL log; should match the trial total
        log_lines = Path(result.log_path).read_text().strip().split("\n")
        per_item_cost = 0.0
        per_item_tokens = 0
        for line in log_lines[1:]:  # skip header
            record = json.loads(line)
            if record["kind"] == "item":
                per_item_cost += record["usage"]["cost_usd"]
                per_item_tokens += record["usage"]["total_tokens"]
        # Cost is rounded to 6 decimal places per item; tolerate small drift
        assert abs(per_item_cost - result.cost_usd) < 1e-5
        assert per_item_tokens == result.total_tokens


# ---------------------------------------------------------------------------
# Resume contract
# ---------------------------------------------------------------------------


class TestResume:
    async def test_resume_skips_already_completed_items(self, tmp_path: Path) -> None:
        call = _make_call()
        out_dir = tmp_path / "resume"

        # First run: 5 items
        result1 = await batch_run(
            call,
            _items(5),
            output_dir=str(out_dir),
            resume=False,
        )
        assert result1.n_succeeded == 5

        # Second run: 10 items (5 of them are the same as run 1).
        # The first 5 should be skipped via the resume scan; the new 5
        # should run fresh.
        result2 = await batch_run(
            call,
            _items(10),
            output_dir=str(out_dir),
            resume=True,
        )
        # n_total counts items observed in *this* iteration of the source
        # (10), independently of how many were skipped vs re-run.
        assert result2.n_total == 10
        # n_skipped counts items the resume scan recognized as already done.
        assert result2.n_skipped == 5
        # n_succeeded is cumulative across the log: 5 from the prior run +
        # 5 from this run = 10. The contract is "items the log records as
        # successes," which makes resumed batches self-describing.
        assert result2.n_succeeded == 10

    async def test_resume_against_empty_dir_is_fresh_run(self, tmp_path: Path) -> None:
        call = _make_call()
        result = await batch_run(
            call,
            _items(3),
            output_dir=str(tmp_path / "fresh"),
            resume=True,  # log doesn't exist yet — treat as fresh
        )
        assert result.n_total == 3
        assert result.n_succeeded == 3
        assert result.n_skipped == 0


# ---------------------------------------------------------------------------
# JsonlBatchWriter
# ---------------------------------------------------------------------------


class TestJsonlBatchWriter:
    async def test_writes_header_then_items(self, tmp_path: Path) -> None:
        writer = JsonlBatchWriter(tmp_path / "test.jsonl")
        await writer.write_header({"run_id": "test-run", "program_hash": "sha256:abc"})
        await writer.write_item({"custom_id": "id1", "status": "success", "output": {"x": 1}})
        await writer.write_item(
            {"custom_id": "id2", "status": "error", "error": {"type": "X", "message": "y"}}
        )
        await writer.close()

        lines = (tmp_path / "test.jsonl").read_text().strip().split("\n")
        assert len(lines) == 3
        records = [json.loads(line) for line in lines]
        assert records[0]["kind"] == "header"
        assert records[0]["run_id"] == "test-run"
        assert records[1]["kind"] == "item"
        assert records[1]["custom_id"] == "id1"
        assert records[2]["custom_id"] == "id2"

    async def test_writer_appends_across_close_and_reopen(self, tmp_path: Path) -> None:
        path = tmp_path / "append.jsonl"
        # Open, write, close
        w1 = JsonlBatchWriter(path)
        await w1.write_header({"run_id": "test"})
        await w1.write_item({"custom_id": "a", "status": "success"})
        await w1.close()

        # Reopen, write more
        w2 = JsonlBatchWriter(path)
        await w2.write_item({"custom_id": "b", "status": "success"})
        await w2.close()

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 3
        custom_ids = [json.loads(line).get("custom_id") for line in lines]
        assert custom_ids == [None, "a", "b"]


# ---------------------------------------------------------------------------
# BatchItem deterministic_id
# ---------------------------------------------------------------------------


class TestDeterministicId:
    def test_same_inputs_same_id(self) -> None:
        h = "sha256:abc"
        id1 = BatchItem.deterministic_id(h, {"text": "hello", "n": 1})
        id2 = BatchItem.deterministic_id(h, {"text": "hello", "n": 1})
        assert id1 == id2
        assert len(id1) == 16

    def test_different_inputs_different_id(self) -> None:
        h = "sha256:abc"
        id1 = BatchItem.deterministic_id(h, {"text": "hello"})
        id2 = BatchItem.deterministic_id(h, {"text": "world"})
        assert id1 != id2

    def test_different_program_hash_different_id(self) -> None:
        id1 = BatchItem.deterministic_id("sha256:a", {"text": "hello"})
        id2 = BatchItem.deterministic_id("sha256:b", {"text": "hello"})
        assert id1 != id2

    def test_key_order_invariant(self) -> None:
        h = "sha256:abc"
        id1 = BatchItem.deterministic_id(h, {"a": 1, "b": 2})
        id2 = BatchItem.deterministic_id(h, {"b": 2, "a": 1})
        assert id1 == id2


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


class TestManifest:
    async def test_manifest_written_on_completion(self, tmp_path: Path) -> None:
        call = _make_call()
        result = await batch_run(
            call,
            _items(3),
            output_dir=str(tmp_path / "manifest"),
            resume=False,
        )
        manifest_path = Path(result.manifest_path)
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["kaos_batch_manifest"] == "1"
        assert manifest["n_total"] == 3
        assert manifest["n_succeeded"] == 3
        assert manifest["n_errored"] == 0
        assert manifest["status"] == "completed"
        assert manifest["program_hash"].startswith("sha256:")
        assert manifest["tokens"]["total"] > 0
        assert manifest["cost_usd"] > 0


# ---------------------------------------------------------------------------
# ListInputSource
# ---------------------------------------------------------------------------


class TestListInputSource:
    async def test_yields_items_in_order(self) -> None:
        source = ListInputSource(
            [{"text": "a"}, {"text": "b"}, {"text": "c"}],
            program_hash_value="sha256:test",
        )
        seen: list[BatchItem] = []
        async for item in source:
            seen.append(item)
        assert len(seen) == 3
        assert [s.inputs["text"] for s in seen] == ["a", "b", "c"]
        # custom_ids are deterministic
        assert all(len(s.custom_id) == 16 for s in seen)

    async def test_describe_includes_count(self) -> None:
        source = ListInputSource(
            [{"text": "a"}, {"text": "b"}],
            program_hash_value="sha256:test",
        )
        desc = source.describe()
        assert desc["type"] == "list"
        assert desc["n_items_hint"] == 2

    async def test_raw_dict_without_program_hash_raises(self) -> None:
        source = ListInputSource([{"text": "a"}])  # no program_hash_value
        with pytest.raises(BatchError, match="program_hash"):
            async for _ in source:
                pass
