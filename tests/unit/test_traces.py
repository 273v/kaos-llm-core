"""Tests for kaos_llm_core.observability — ExecutionTrace and JSONL export."""

from __future__ import annotations

from kaos_llm_core.observability import ExecutionTrace, export_jsonl, load_jsonl


class TestExecutionTrace:
    def test_construction(self) -> None:
        trace = ExecutionTrace(
            call_name="ExtractEntities",
            signature="ExtractEntities",
            model="anthropic:claude-sonnet-4-6",
            input_tokens=100,
            output_tokens=50,
        )
        assert trace.call_name == "ExtractEntities"
        assert trace.total_tokens == 0  # must be set explicitly
        assert trace.trace_id  # auto-generated
        assert trace.timestamp

    def test_to_dict(self) -> None:
        trace = ExecutionTrace(
            call_name="Test",
            signature="TestSig",
            inputs={"text": "hello"},
            outputs={"result": "world"},
            model="test-model",
            codec="JSONCodec",
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            cost_usd=0.001,
            latency_ms=250.0,
        )
        d = trace.model_dump()
        assert d["call_name"] == "Test"
        assert d["inputs"] == {"text": "hello"}
        assert d["model"] == "test-model"
        from datetime import datetime

        assert isinstance(d["timestamp"], datetime)

    def test_from_dict(self) -> None:
        d = {
            "trace_id": "abc123",
            "call_name": "Extract",
            "signature": "ExtractSig",
            "inputs": {"x": 1},
            "outputs": {"y": 2},
            "model": "test",
            "codec": "JSON",
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "cost_usd": 0.01,
            "latency_ms": 500.0,
            "retries": 1,
            "examples_used": 3,
            "children": [],
            "error": None,
            "timestamp": "2026-01-01T00:00:00+00:00",
        }
        trace = ExecutionTrace.model_validate(d)
        assert trace.trace_id == "abc123"
        assert trace.call_name == "Extract"
        assert trace.input_tokens == 100

    def test_round_trip(self) -> None:
        original = ExecutionTrace(
            call_name="RoundTrip",
            signature="RoundTripSig",
            inputs={"a": 1},
            outputs={"b": 2},
            model="test",
            codec="JSON",
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
        )
        restored = ExecutionTrace.model_validate(original.model_dump())
        assert restored.call_name == original.call_name
        assert restored.inputs == original.inputs
        assert restored.outputs == original.outputs

    def test_hierarchical_traces(self) -> None:
        child1 = ExecutionTrace(call_name="child1", cost_usd=0.01)
        child2 = ExecutionTrace(call_name="child2", cost_usd=0.02)
        parent = ExecutionTrace(call_name="parent", cost_usd=0.03, children=[child1, child2])

        # total_cost_usd for parent with children = sum of children (not double-counted)
        assert abs(parent.total_cost_usd - 0.03) < 1e-10
        # Leaf traces: total_cost_usd == cost_usd
        assert abs(child1.total_cost_usd - 0.01) < 1e-10
        d = parent.model_dump()
        assert len(d["children"]) == 2
        restored = ExecutionTrace.model_validate(d)
        assert len(restored.children) == 2


class TestJSONLExport:
    def test_export_and_load(self, tmp_path) -> None:  # type: ignore[no-any-explicit]
        traces = [
            ExecutionTrace(call_name="t1", inputs={"x": 1}),
            ExecutionTrace(call_name="t2", inputs={"x": 2}),
        ]
        path = tmp_path / "traces.jsonl"
        export_jsonl(traces, path)
        loaded = load_jsonl(path)
        assert len(loaded) == 2
        assert loaded[0].call_name == "t1"
        assert loaded[1].call_name == "t2"

    def test_append_mode(self, tmp_path) -> None:  # type: ignore[no-any-explicit]
        path = tmp_path / "traces.jsonl"
        export_jsonl([ExecutionTrace(call_name="a")], path)
        export_jsonl([ExecutionTrace(call_name="b")], path)
        loaded = load_jsonl(path)
        assert len(loaded) == 2

    def test_load_nonexistent_returns_empty(self, tmp_path) -> None:  # type: ignore[no-any-explicit]
        path = tmp_path / "nonexistent.jsonl"
        loaded = load_jsonl(path)
        assert loaded == []

    def test_hierarchical_jsonl_round_trip(self, tmp_path) -> None:  # type: ignore[no-any-explicit]
        child = ExecutionTrace(call_name="child", cost_usd=0.01)
        parent = ExecutionTrace(call_name="parent", children=[child])
        path = tmp_path / "traces.jsonl"
        export_jsonl([parent], path)
        loaded = load_jsonl(path)
        assert len(loaded) == 1
        assert len(loaded[0].children) == 1
        assert loaded[0].children[0].call_name == "child"
