"""Tests for kaos_llm_core.programs.base — Program composition, trace trees, and state."""

from __future__ import annotations

import json
from typing import Any

from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart

from kaos_llm_core.programs.base import Program
from kaos_llm_core.programs.call import Call
from kaos_llm_core.signatures import InputField, OutputField, Signature
from kaos_llm_core.types import Example


class ExtractSig(Signature):
    """Extract entities."""

    text: str = InputField(description="Input")
    entities: list[str] = OutputField(description="Entities")


class ClassifySig(Signature):
    """Classify risk."""

    text: str = InputField(description="Input")
    level: str = OutputField(description="Level")


def _make_client(response_data: dict[str, Any]) -> FunctionClient:
    def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
        return ProviderResponse.model_construct(
            provider="function",
            model="function-test",
            raw={},
            parts=[ContentPart.model_construct(type="text", text=json.dumps(response_data))],
            usage=UsageInfo.model_construct(input_tokens=10, output_tokens=5, total_tokens=15),
            stop_reason="end_turn",
            status_code=200,
            response_headers={},
        )

    return FunctionClient(function=fn)


class SimpleProgram(Program):
    def __init__(self) -> None:
        self.extract = Call(ExtractSig, model="function-test")
        self.classify = Call(ClassifySig, model="function-test")

    async def forward(self, **kwargs: Any) -> dict[str, Any]:
        return {"status": "ok"}


class TwoStepProgram(Program):
    """Program that actually calls its sub-Calls to produce trace children."""

    def __init__(self) -> None:
        extract_client = _make_client({"entities": ["SEC", "Acme"]})
        classify_client = _make_client({"level": "high"})
        self.extract = Call(ExtractSig, model="function-test", client=extract_client)
        self.classify = Call(ClassifySig, model="function-test", client=classify_client)

    async def forward(self, **kwargs: Any) -> dict[str, Any]:
        entities = await self.extract(text=kwargs["text"])
        risk = await self.classify(text=kwargs["text"])
        return {
            "entities": entities.entities,
            "level": risk.level,
        }


class TestProgramDiscovery:
    def test_named_calls(self) -> None:
        prog = SimpleProgram()
        calls = prog.named_calls()
        assert "extract" in calls
        assert "classify" in calls
        assert len(calls) == 2

    def test_named_calls_excludes_non_calls(self) -> None:
        prog = SimpleProgram()
        calls = prog.named_calls()
        for name in calls:
            assert isinstance(calls[name], (Call, Program))


class TestProgramState:
    def test_get_learnable_state(self) -> None:
        prog = SimpleProgram()
        prog.extract.instructions = "Custom extract instruction"
        prog.extract.examples = [
            Example(inputs={"text": "a"}, outputs={"entities": ["x"]}),
        ]
        state = prog.get_learnable_state()
        assert "extract" in state
        assert "classify" in state
        assert state["extract"]["instructions"] == "Custom extract instruction"
        assert len(state["extract"]["examples"]) == 1

    def test_set_learnable_state(self) -> None:
        prog = SimpleProgram()
        state = {
            "extract": {
                "instructions": "New extract instructions",
                "examples": [{"inputs": {"text": "b"}, "outputs": {"entities": ["y"]}}],
            },
            "classify": {
                "instructions": "New classify instructions",
                "examples": [],
            },
        }
        prog.set_learnable_state(state)
        assert prog.extract.instructions == "New extract instructions"
        assert len(prog.extract.examples) == 1
        assert prog.classify.instructions == "New classify instructions"


class TestProgramSaveLoad:
    def test_save_and_load(self, tmp_path) -> None:  # type: ignore[no-any-explicit]
        prog = SimpleProgram()
        prog.extract.instructions = "Custom instructions"
        prog.extract.examples = [
            Example(inputs={"text": "a"}, outputs={"entities": ["x"]}),
        ]
        path = tmp_path / "prog_state.json"
        prog.save(path)

        data = json.loads(path.read_text())
        assert data["program"] == "SimpleProgram"
        assert data["version"] == 2
        assert "extract" in data["state"]

        prog2 = SimpleProgram()
        prog2.load(path)
        assert prog2.extract.instructions == "Custom instructions"
        assert len(prog2.extract.examples) == 1

    def test_save_v2_includes_hyperparameters_codec_and_model(self, tmp_path) -> None:  # type: ignore[no-any-explicit]
        """v2 envelope must persist hyperparameters, codec, and model per call."""

        class HyperProgram(Program):
            def __init__(self) -> None:
                self.tuned = Call(
                    ExtractSig,
                    model="anthropic:claude-haiku-4-5",
                    temperature=0.3,
                    top_p=0.9,
                    max_tokens=512,
                )

            async def forward(self, **_kwargs: Any) -> dict[str, Any]:
                return {}

        prog = HyperProgram()
        path = tmp_path / "hyper.json"
        prog.save(path)

        data = json.loads(path.read_text())
        assert data["version"] == 2
        tuned_state = data["state"]["tuned"]
        assert tuned_state["hyperparameters"] == {
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 512,
        }
        assert tuned_state["model"] == "anthropic:claude-haiku-4-5"
        assert tuned_state["codec"].endswith("JSONCodec")

    def test_load_v2_round_trips_hyperparameters_and_model(self, tmp_path) -> None:  # type: ignore[no-any-explicit]
        """A v2 file should restore hyperparameters and model on a fresh Program."""

        class HyperProgram(Program):
            def __init__(self, *, temperature: float = 0.0) -> None:
                self.tuned = Call(
                    ExtractSig,
                    model="anthropic:claude-haiku-4-5",
                    temperature=temperature,
                )

            async def forward(self, **_kwargs: Any) -> dict[str, Any]:
                return {}

        prog = HyperProgram(temperature=0.7)
        prog.tuned._kwargs["max_tokens"] = 256
        path = tmp_path / "round_trip.json"
        prog.save(path)

        # Fresh program with different construction-time hyperparameters
        prog2 = HyperProgram(temperature=0.0)
        assert prog2.tuned._kwargs.get("temperature") == 0.0
        prog2.load(path)
        assert prog2.tuned._kwargs["temperature"] == 0.7
        assert prog2.tuned._kwargs["max_tokens"] == 256
        assert prog2.tuned._model == "anthropic:claude-haiku-4-5"

    def test_load_v1_file_is_non_fatal(self, tmp_path, caplog) -> None:  # type: ignore[no-any-explicit]
        """A v1 envelope must load without raising; hyperparameters fall back to defaults."""

        # Hand-craft a v1 envelope (no hyperparameters/codec/model fields)
        v1_envelope = {
            "program": "SimpleProgram",
            "version": 1,
            "state": {
                "extract": {
                    "instructions": "v1 instructions",
                    "examples": [
                        {"inputs": {"text": "a"}, "outputs": {"entities": ["x"]}},
                    ],
                },
                "classify": {
                    "instructions": "v1 classify instructions",
                    "examples": [],
                },
            },
        }
        path = tmp_path / "v1.json"
        path.write_text(json.dumps(v1_envelope), encoding="utf-8")

        prog = SimpleProgram()
        prog.tuned_temperature_before_load = prog.extract._kwargs.get("temperature")
        prog.load(path)

        # Instructions and examples come from the v1 file
        assert prog.extract.instructions == "v1 instructions"
        assert len(prog.extract.examples) == 1
        assert prog.classify.instructions == "v1 classify instructions"

        # Hyperparameters were not in the v1 file, so the in-code value sticks
        assert prog.extract._kwargs.get("temperature") == prog.tuned_temperature_before_load  # ty: ignore[unresolved-attribute]

    def test_load_future_version_raises(self, tmp_path) -> None:  # type: ignore[no-any-explicit]
        """A v999 file from a future kaos-llm-core release should be rejected."""
        envelope = {
            "program": "SimpleProgram",
            "version": 999,
            "state": {},
        }
        path = tmp_path / "future.json"
        path.write_text(json.dumps(envelope), encoding="utf-8")

        prog = SimpleProgram()
        try:
            prog.load(path)
        except ValueError as exc:
            assert "version 999" in str(exc)
            assert "Upgrade kaos-llm-core" in str(exc)
        else:
            raise AssertionError("Expected ValueError for future schema version")

    def test_load_missing_state_key_raises(self, tmp_path) -> None:  # type: ignore[no-any-explicit]
        """A corrupted envelope without 'state' should raise with recovery guidance."""
        envelope = {"program": "SimpleProgram", "version": 2}
        path = tmp_path / "corrupt.json"
        path.write_text(json.dumps(envelope), encoding="utf-8")

        prog = SimpleProgram()
        try:
            prog.load(path)
        except ValueError as exc:
            assert "missing the 'state' key" in str(exc)
            assert "Re-save" in str(exc)
        else:
            raise AssertionError("Expected ValueError for missing 'state'")

    def test_unknown_codec_in_state_keeps_existing(self, tmp_path) -> None:  # type: ignore[no-any-explicit]
        """Unknown codec class should fall back to current codec, not crash."""
        envelope = {
            "program": "SimpleProgram",
            "version": 2,
            "state": {
                "extract": {
                    "instructions": "x",
                    "examples": [],
                    "hyperparameters": {},
                    "codec": "definitely.not.a.real.module.NoSuchCodec",
                    "model": None,
                },
                "classify": {
                    "instructions": "y",
                    "examples": [],
                    "hyperparameters": {},
                    "codec": "kaos_llm_core.codecs.json_codec.JSONCodec",
                    "model": None,
                },
            },
        }
        path = tmp_path / "bad_codec.json"
        path.write_text(json.dumps(envelope), encoding="utf-8")

        prog = SimpleProgram()
        original_codec_type = type(prog.extract._codec)
        prog.load(path)
        # Bad codec name → fallback to original
        assert isinstance(prog.extract._codec, original_codec_type)
        # Good codec name → resolved correctly
        assert type(prog.classify._codec).__name__ == "JSONCodec"


class TestProgramExecution:
    async def test_call_delegates_to_forward(self) -> None:
        prog = SimpleProgram()
        result = await prog(text="test")
        assert result == {"status": "ok"}


class TestProgramTraceTree:
    """Verify that Program.__call__ builds a hierarchical trace with children."""

    async def test_trace_captures_children(self) -> None:
        """Program trace should contain child traces from each sub-Call."""
        prog = TwoStepProgram()
        invocation = await prog.invoke(text="The SEC filed suit against Acme Corp.")
        result = invocation.output

        assert result["entities"] == ["SEC", "Acme"]
        assert result["level"] == "high"

        # Program should have a trace
        trace = invocation.trace
        assert trace is not None
        assert trace.call_name == "TwoStepProgram"

        # Trace should have 2 children (extract + classify)
        assert len(trace.children) == 2
        child_names = {c.call_name for c in trace.children}
        assert "ExtractSig" in child_names
        assert "ClassifySig" in child_names

    async def test_trace_aggregates_tokens(self) -> None:
        """Program trace should sum token counts from children."""
        prog = TwoStepProgram()
        invocation = await prog.invoke(text="test")

        trace = invocation.trace
        assert trace is not None
        # Each child has 10 input + 5 output + 15 total (from _make_client)
        assert trace.input_tokens == 20  # 10 + 10
        assert trace.output_tokens == 10  # 5 + 5
        assert trace.total_tokens == 30  # 15 + 15

    async def test_trace_has_latency(self) -> None:
        """Program trace should measure wall-clock latency."""
        prog = TwoStepProgram()
        invocation = await prog.invoke(text="test")

        trace = invocation.trace
        assert trace is not None
        assert trace.latency_ms > 0

    async def test_trace_children_have_own_traces(self) -> None:
        """Each child trace should have its own complete data."""
        prog = TwoStepProgram()
        invocation = await prog.invoke(text="test")

        trace = invocation.trace
        assert trace is not None
        for child in trace.children:
            assert child.model == "function-test"
            assert child.codec == "JSONCodec"
            assert child.input_tokens > 0
            assert child.trace_id != trace.trace_id

    async def test_trace_serializes_with_children(self) -> None:
        """Trace tree should round-trip through to_dict/from_dict."""
        from kaos_llm_core.observability.traces import ExecutionTrace

        prog = TwoStepProgram()
        invocation = await prog.invoke(text="test")

        trace = invocation.trace
        assert trace is not None

        d = trace.model_dump()
        assert len(d["children"]) == 2

        restored = ExecutionTrace.model_validate(d)
        assert len(restored.children) == 2
        assert restored.call_name == "TwoStepProgram"
        assert restored.input_tokens == 20
