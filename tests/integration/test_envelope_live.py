"""Live integration tests for the Phase 15.1 Program v3 envelope.

Drives ``Program.from_envelope()`` against real Anthropic Haiku.
Distinct from ``test_phase15_mcp_wire.py`` which goes through the
MCP wire — this test exercises the Python API directly so failures
isolate the envelope/executor layer from the MCP transport layer.
"""

from __future__ import annotations

import os

import pytest

from kaos_llm_core.programs.envelope import from_envelope, program_hash

pytestmark = pytest.mark.integration


def _has_key(*env_vars: str) -> bool:
    return any(os.getenv(v) for v in env_vars)


requires_anthropic = pytest.mark.skipif(
    not _has_key("KAOS_LLM_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    reason="No Anthropic API key",
)


HAIKU = "claude-haiku-4-5"


@requires_anthropic
class TestEnvelopeLive:
    async def test_single_call_envelope_against_haiku(self) -> None:
        """Build a one-step envelope, execute against real Haiku, assert content."""
        envelope = {
            "kaos_program": "1",
            "name": "summarize",
            "inputs": {"text": {"type": "string"}},
            "clients": {"default": {"provider": "anthropic", "model": HAIKU}},
            "steps": [
                {
                    "id": "summarize",
                    "kind": "call",
                    "client": "default",
                    "instruction": "Write a one-sentence summary of the input text.",
                    "inputs": {"text": "$.inputs.text"},
                    "output_fields": {
                        "summary": {
                            "description": "One-sentence summary",
                            "type": {"type": "string"},
                        }
                    },
                }
            ],
            "output": {"summary": "$.steps.summarize.output.summary"},
            "capabilities": ["call", "jsonpointer_refs"],
        }

        program = from_envelope(envelope)
        invocation = await program.invoke(
            text=(
                "The Federal Reserve raised interest rates by 25 basis points "
                "today, the third hike this year, citing persistent inflation "
                "and a strong labor market."
            )
        )

        result = invocation.output
        assert "summary" in result
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 10
        assert invocation.usage.cost_usd > 0
        assert invocation.usage.total_tokens > 0
        print(
            f"[envelope_live single] summary={result['summary'][:120]!r} "
            f"cost=${invocation.usage.cost_usd:.6f} tokens={invocation.usage.total_tokens}"
        )

    async def test_two_step_envelope_chains_outputs(self) -> None:
        """Two-step envelope: extract entities, then classify; verify chaining works live."""
        envelope = {
            "kaos_program": "1",
            "name": "extract-and-classify",
            "inputs": {"text": {"type": "string"}},
            "clients": {"default": {"provider": "anthropic", "model": HAIKU}},
            "steps": [
                {
                    "id": "extract",
                    "kind": "call",
                    "client": "default",
                    "instruction": "Extract the main subject and the action from the sentence.",
                    "inputs": {"text": "$.inputs.text"},
                    "output_fields": {
                        "subject": {
                            "description": "Main subject",
                            "type": {"type": "string"},
                        },
                        "action": {
                            "description": "Main action",
                            "type": {"type": "string"},
                        },
                    },
                },
                {
                    "id": "classify",
                    "kind": "call",
                    "client": "default",
                    "instruction": (
                        "Given a subject and an action, classify the sentiment "
                        "as positive, negative, or neutral."
                    ),
                    "inputs": {
                        "subject": "$.steps.extract.output.subject",
                        "action": "$.steps.extract.output.action",
                    },
                    "output_fields": {
                        "sentiment": {
                            "description": "positive | negative | neutral",
                            "type": {"type": "string"},
                        }
                    },
                },
            ],
            "output": {
                "subject": "$.steps.extract.output.subject",
                "action": "$.steps.extract.output.action",
                "sentiment": "$.steps.classify.output.sentiment",
            },
            "capabilities": ["call", "jsonpointer_refs"],
        }

        program = from_envelope(envelope)
        invocation = await program.invoke(
            text="The court dismissed the wrongful termination lawsuit."
        )
        result = invocation.output

        assert "subject" in result
        assert "action" in result
        assert "sentiment" in result
        assert result["sentiment"].lower().strip() in (
            "positive",
            "negative",
            "neutral",
            "negative.",
            "neutral.",
            "positive.",
        )
        # Two steps fired against Haiku
        assert invocation.usage.cost_usd > 0
        # Both step children captured in trace
        assert invocation.trace is not None
        assert len(invocation.trace.children) == 2
        print(f"[envelope_live two_step] {result} cost=${invocation.usage.cost_usd:.6f}")

    async def test_chain_of_thought_step_kind_against_haiku(self) -> None:
        """The 'reason' step kind builds ChainOfThought and runs live."""
        envelope = {
            "kaos_program": "1",
            "name": "reason-test",
            "inputs": {"question": {"type": "string"}},
            "clients": {"default": {"provider": "anthropic", "model": HAIKU}},
            "steps": [
                {
                    "id": "reasoner",
                    "kind": "reason",
                    "client": "default",
                    "instruction": "Reason through the question step by step before answering.",
                    "inputs": {"question": "$.inputs.question"},
                    "output_fields": {
                        "answer": {
                            "description": "The final answer",
                            "type": {"type": "string"},
                        }
                    },
                }
            ],
            "output": {"answer": "$.steps.reasoner.output.answer"},
            "capabilities": ["reason", "jsonpointer_refs"],
        }

        program = from_envelope(envelope)
        invocation = await program.invoke(question="What is 17 multiplied by 23?")
        result = invocation.output

        assert "answer" in result
        # Real model should produce the correct answer 391, possibly wrapped in prose
        assert "391" in str(result["answer"]), f"Expected 391 in answer, got {result['answer']!r}"
        print(f"[envelope_live reason] answer={result['answer'][:120]!r}")

    async def test_program_hash_stable_across_invocations(self) -> None:
        """The program_hash function is deterministic — required for batch_run resume."""
        envelope = {
            "kaos_program": "1",
            "name": "stable-hash",
            "inputs": {"x": {"type": "string"}},
            "clients": {"default": {"provider": "anthropic", "model": HAIKU}},
            "steps": [
                {
                    "id": "s",
                    "kind": "call",
                    "client": "default",
                    "instruction": "Echo x",
                    "inputs": {"x": "$.inputs.x"},
                    "output_fields": {"y": {"description": "y", "type": {"type": "string"}}},
                }
            ],
            "output": {"y": "$.steps.s.output.y"},
            "capabilities": ["call", "jsonpointer_refs"],
        }
        # Hash twice; must match
        h1 = program_hash(envelope)
        h2 = program_hash(envelope)
        assert h1 == h2
        # Building the program does not affect the dict
        from_envelope(envelope)
        h3 = program_hash(envelope)
        assert h1 == h3
