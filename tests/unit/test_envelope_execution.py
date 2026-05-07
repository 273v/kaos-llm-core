"""Execution tests for the Phase 15.1 ``Program.from_envelope`` constructor.

Drives a v3 envelope through ``from_envelope`` against the
``FunctionClient`` so we can assert deterministic outputs without
hitting a real provider. Verifies the constructor produces the
right ``Call`` / ``ChainOfThought`` instances per step kind, that
ProgramGraph (Phase 11) auto-registration picks each step up, that
the forward() execution walks steps in declaration order, and that
the output mapping resolves correctly.
"""

from __future__ import annotations

import json
from typing import Any

from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart

from kaos_llm_core.programs.base import Program
from kaos_llm_core.programs.call import Call
from kaos_llm_core.programs.chain_of_thought import ChainOfThought
from kaos_llm_core.programs.envelope import _EnvelopeProgram, from_envelope


def _step(program: Program, name: str) -> Any:
    """Type-erased accessor for ProgramGraph-registered child steps.

    The envelope-built program registers each step as a public attribute
    via Phase 11 ProgramGraph __setattr__. ty cannot statically trace
    these dynamic attributes, so the test helper returns Any.
    """
    return getattr(program, name)


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


def _function_client_returning(value_by_field: dict[str, Any]) -> FunctionClient:
    """Build a FunctionClient that returns the same JSON every call."""

    def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
        return _json_response(value_by_field)

    return FunctionClient(function=fn)


def _envelope(
    *,
    n_steps: int = 1,
    step_kinds: list[str] | None = None,
    output_value: str = "ok",
) -> dict[str, Any]:
    """Build a parametrized envelope: N steps, each producing {answer: output_value}."""
    kinds = step_kinds or ["call"] * n_steps
    steps = []
    for i in range(n_steps):
        steps.append(
            {
                "id": f"step_{i}",
                "kind": kinds[i],
                "client": "default",
                "instruction": f"Produce a value for step {i}",
                "inputs": {"text": "$.inputs.text"},
                "output_fields": {
                    "answer": {
                        "description": "The answer",
                        "type": {"type": "string"},
                    }
                },
            }
        )
    capabilities = sorted({*kinds, "jsonpointer_refs"})
    output_mapping = {f"answer_{i}": f"$.steps.step_{i}.output.answer" for i in range(n_steps)}
    return {
        "kaos_program": "1",
        "name": "exec-test",
        "inputs": {"text": {"type": "string"}},
        "clients": {"default": {"provider": "function", "model": "function-test"}},
        "steps": steps,
        "output": output_mapping,
        "capabilities": capabilities,
    }


# ---------------------------------------------------------------------------
# Step dispatch
# ---------------------------------------------------------------------------


class TestStepDispatch:
    def test_call_kind_builds_call(self) -> None:
        program = from_envelope(_envelope(n_steps=1, step_kinds=["call"]))
        step = _step(program, "step_0")
        assert isinstance(step, Call)
        # Not ChainOfThought (which is a Call subclass — narrower check)
        assert type(step).__name__ == "Call"

    def test_reason_kind_builds_chain_of_thought(self) -> None:
        program = from_envelope(_envelope(n_steps=1, step_kinds=["reason"]))
        step = _step(program, "step_0")
        assert isinstance(step, ChainOfThought)

    def test_two_step_program_has_both_attrs(self) -> None:
        program = from_envelope(_envelope(n_steps=2))
        assert hasattr(program, "step_0")
        assert hasattr(program, "step_1")

    def test_step_program_graph_auto_registers(self) -> None:
        program = from_envelope(_envelope(n_steps=2))
        names = program.named_calls()
        assert "step_0" in names
        assert "step_1" in names

    def test_step_program_graph_primary_returns_first_step(self) -> None:
        program = from_envelope(_envelope(n_steps=3))
        primary = program.graph.primary()
        # First declared step is step_0
        assert primary is _step(program, "step_0")


# ---------------------------------------------------------------------------
# Forward execution
# ---------------------------------------------------------------------------


class TestForwardExecution:
    async def test_single_step_executes_and_returns_mapped_output(self) -> None:
        program = from_envelope(_envelope(n_steps=1))
        # Inject FunctionClient onto the step
        _step(program, "step_0")._client = _function_client_returning({"answer": "hello"})
        result = await program(text="anything")
        # Result is the resolved output mapping: {answer_0: ...}
        assert result == {"answer_0": "hello"}

    async def test_multi_step_executes_in_declaration_order(self) -> None:
        program = from_envelope(_envelope(n_steps=3))
        for i in range(3):
            _step(program, f"step_{i}")._client = _function_client_returning(
                {"answer": f"value_{i}"}
            )
        result = await program(text="x")
        assert result == {
            "answer_0": "value_0",
            "answer_1": "value_1",
            "answer_2": "value_2",
        }

    async def test_step_chaining_via_pointer(self) -> None:
        """Step 2's input is step 1's output, threaded via $.steps.step_0.output.answer."""
        env = {
            "kaos_program": "1",
            "name": "chained",
            "inputs": {"text": {"type": "string"}},
            "clients": {"default": {"provider": "function", "model": "function-test"}},
            "steps": [
                {
                    "id": "first",
                    "kind": "call",
                    "client": "default",
                    "instruction": "Produce x",
                    "inputs": {"text": "$.inputs.text"},
                    "output_fields": {"x": {"description": "x", "type": {"type": "string"}}},
                },
                {
                    "id": "second",
                    "kind": "call",
                    "client": "default",
                    "instruction": "Consume x",
                    "inputs": {"x": "$.steps.first.output.x"},
                    "output_fields": {"y": {"description": "y", "type": {"type": "string"}}},
                },
            ],
            "output": {"y": "$.steps.second.output.y"},
            "capabilities": ["call", "jsonpointer_refs"],
        }
        program = from_envelope(env)

        # The first step receives 'text=...' and produces {x: 'first_out'}
        _step(program, "first")._client = _function_client_returning({"x": "first_out"})
        # The second step receives x='first_out' (from chaining) and produces {y: 'second_out'}
        _step(program, "second")._client = _function_client_returning({"y": "second_out"})

        result = await program(text="hello")
        assert result == {"y": "second_out"}

    async def test_program_returns_invocation_via_invoke(self) -> None:
        """from_envelope returns a real Program — invoke() returns Invocation."""
        program = from_envelope(_envelope(n_steps=1))
        _step(program, "step_0")._client = _function_client_returning({"answer": "v"})
        invocation = await program.invoke(text="anything")
        assert invocation.output == {"answer_0": "v"}
        assert invocation.trace is not None
        assert invocation.error is None


# ---------------------------------------------------------------------------
# Empty / degenerate envelopes
# ---------------------------------------------------------------------------


class TestEmptyEnvelopes:
    async def test_zero_step_envelope_passes_inputs_through(self) -> None:
        """Envelope with no steps is a degenerate input-renamer."""
        env = {
            "kaos_program": "1",
            "name": "passthrough",
            "inputs": {"src": {"type": "string"}, "name": {"type": "string"}},
            "clients": {},
            "steps": [],
            "output": {"renamed_src": "$.inputs.src", "renamed_name": "$.inputs.name"},
            "capabilities": ["jsonpointer_refs"],
        }
        program = from_envelope(env)
        result = await program(src="hello", name="world")
        assert result == {"renamed_src": "hello", "renamed_name": "world"}


# ---------------------------------------------------------------------------
# Envelope reference roundtrips on the program
# ---------------------------------------------------------------------------


class TestEnvelopeAccess:
    def test_envelope_program_exposes_envelope(self) -> None:
        env_dict = _envelope(n_steps=1)
        program = from_envelope(env_dict)
        assert isinstance(program, _EnvelopeProgram)
        assert program.envelope.name == "exec-test"
        assert len(program.envelope.steps) == 1
