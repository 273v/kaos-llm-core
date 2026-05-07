"""Round-trip tests for the Phase 15.1 envelope.

For each step kind that v3 supports, build (a) a hand-coded
``Program`` subclass and (b) the equivalent envelope, run both
through `FunctionClient`, and assert the outputs match. This pins
the contract that an envelope-built program is observably
equivalent to its hand-written counterpart.
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
from kaos_llm_core.programs.envelope import from_envelope
from kaos_llm_core.signatures import InputField, OutputField, Signature


def _step(program: Program, name: str) -> Any:
    """Type-erased accessor for ProgramGraph-registered child steps."""
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


def _client(value: dict[str, Any]) -> FunctionClient:
    def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
        return _json_response(value)

    return FunctionClient(function=fn)


# ---------------------------------------------------------------------------
# Single-step Call round-trip
# ---------------------------------------------------------------------------


class _ExtractSig(Signature):
    """Extract a one-sentence summary."""

    text: str = InputField(description="Text to summarize")
    summary: str = OutputField(description="One-sentence summary")


class _ExtractProgram(Program):
    def __init__(self) -> None:
        self.extract = Call(_ExtractSig, model="function-test")

    async def forward(self, **kwargs: Any) -> Any:
        result = await self.extract(text=kwargs["text"])
        return {"summary": result.summary}


def _extract_envelope() -> dict[str, Any]:
    return {
        "kaos_program": "1",
        "name": "extract",
        "inputs": {"text": {"type": "string"}},
        "clients": {"default": {"provider": "function", "model": "function-test"}},
        "steps": [
            {
                "id": "extract",
                "kind": "call",
                "client": "default",
                "instruction": "Extract a one-sentence summary.",
                "inputs": {"text": "$.inputs.text"},
                "output_fields": {
                    "summary": {
                        "description": "One-sentence summary",
                        "type": {"type": "string"},
                    }
                },
            }
        ],
        "output": {"summary": "$.steps.extract.output.summary"},
        "capabilities": ["call", "jsonpointer_refs"],
    }


class TestSingleCallRoundTrip:
    async def test_envelope_matches_handwritten(self) -> None:
        # Hand-coded program
        hand = _ExtractProgram()
        hand.extract._client = _client({"summary": "this is a summary"})
        hand_result = await hand(text="some input")

        # Envelope-built program
        env_program = from_envelope(_extract_envelope())
        _step(env_program, "extract")._client = _client({"summary": "this is a summary"})
        env_result = await env_program(text="some input")

        assert hand_result == env_result == {"summary": "this is a summary"}


# ---------------------------------------------------------------------------
# Two-step pipeline round-trip
# ---------------------------------------------------------------------------


class _StepASig(Signature):
    """Pull out a label."""

    text: str = InputField(description="Input text")
    label: str = OutputField(description="The label")


class _StepBSig(Signature):
    """Refine the label."""

    text: str = InputField(description="Original text")
    label: str = InputField(description="Initial label")
    final_label: str = OutputField(description="Refined label")


class _TwoStepProgram(Program):
    def __init__(self) -> None:
        self.step_a = Call(_StepASig, model="function-test")
        self.step_b = Call(_StepBSig, model="function-test")

    async def forward(self, **kwargs: Any) -> Any:
        text = kwargs["text"]
        a = await self.step_a(text=text)
        b = await self.step_b(text=text, label=a.label)
        return {"final": b.final_label}


def _two_step_envelope() -> dict[str, Any]:
    return {
        "kaos_program": "1",
        "name": "two-step",
        "inputs": {"text": {"type": "string"}},
        "clients": {"default": {"provider": "function", "model": "function-test"}},
        "steps": [
            {
                "id": "step_a",
                "kind": "call",
                "client": "default",
                "instruction": "Pull out a label.",
                "inputs": {"text": "$.inputs.text"},
                "output_fields": {"label": {"description": "label", "type": {"type": "string"}}},
            },
            {
                "id": "step_b",
                "kind": "call",
                "client": "default",
                "instruction": "Refine the label.",
                "inputs": {
                    "text": "$.inputs.text",
                    "label": "$.steps.step_a.output.label",
                },
                "output_fields": {
                    "final_label": {
                        "description": "refined",
                        "type": {"type": "string"},
                    }
                },
            },
        ],
        "output": {"final": "$.steps.step_b.output.final_label"},
        "capabilities": ["call", "jsonpointer_refs"],
    }


class TestTwoStepRoundTrip:
    async def test_envelope_matches_handwritten(self) -> None:
        hand = _TwoStepProgram()
        hand.step_a._client = _client({"label": "initial_label"})
        hand.step_b._client = _client({"final_label": "refined_label"})
        hand_result = await hand(text="some doc")

        env_program = from_envelope(_two_step_envelope())
        _step(env_program, "step_a")._client = _client({"label": "initial_label"})
        _step(env_program, "step_b")._client = _client({"final_label": "refined_label"})
        env_result = await env_program(text="some doc")

        assert hand_result == env_result == {"final": "refined_label"}


# ---------------------------------------------------------------------------
# ChainOfThought round-trip
# ---------------------------------------------------------------------------


class _ReasonSig(Signature):
    """Reason about a question."""

    question: str = InputField(description="The question")
    answer: str = OutputField(description="The answer")


class _ReasonProgram(Program):
    def __init__(self) -> None:
        self.reasoner = ChainOfThought(_ReasonSig, model="function-test")

    async def forward(self, **kwargs: Any) -> Any:
        result = await self.reasoner(question=kwargs["question"])
        return {"answer": result.answer}


def _reason_envelope() -> dict[str, Any]:
    return {
        "kaos_program": "1",
        "name": "reason",
        "inputs": {"question": {"type": "string"}},
        "clients": {"default": {"provider": "function", "model": "function-test"}},
        "steps": [
            {
                "id": "reasoner",
                "kind": "reason",
                "client": "default",
                "instruction": "Reason about the question.",
                "inputs": {"question": "$.inputs.question"},
                "output_fields": {"answer": {"description": "answer", "type": {"type": "string"}}},
            }
        ],
        "output": {"answer": "$.steps.reasoner.output.answer"},
        "capabilities": ["reason", "jsonpointer_refs"],
    }


class TestChainOfThoughtRoundTrip:
    async def test_envelope_matches_handwritten(self) -> None:
        # ChainOfThought returns reasoning + answer; FunctionClient returns
        # the dict literally so we set both fields.
        response = {"reasoning": "let me think", "answer": "42"}

        hand = _ReasonProgram()
        hand.reasoner._client = _client(response)
        hand_result = await hand(question="what is the meaning of life?")

        env_program = from_envelope(_reason_envelope())
        _step(env_program, "reasoner")._client = _client(response)
        env_result = await env_program(question="what is the meaning of life?")

        assert hand_result == env_result == {"answer": "42"}
