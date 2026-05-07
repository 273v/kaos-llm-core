"""Live tests for the Phase 16.2 v3 step kinds against real Haiku.

  - ``multi_chain_comparison`` step: 3 reasoning chains, aggregator
    synthesizes a final answer. Verifies the envelope dispatch + the
    underlying ``MultiChainComparison`` Program both work end-to-end.
  - ``program_of_thought`` step: code-as-reasoning with the
    subprocess sandbox. Verifies the opt-in flag, the code generation
    phase, the sandbox execution, and the interpretation phase all
    chain through real Haiku.

Hard ``$0.05`` cost cap per test.
"""

from __future__ import annotations

import os
import sys

import pytest

from kaos_llm_core.programs.envelope import from_envelope
from kaos_llm_core.programs.multi_chain_comparison import (
    MultiChainComparison,
    MultiChainComparisonResult,
)
from kaos_llm_core.programs.program_of_thought import (
    ProgramOfThought,
    ProgramOfThoughtResult,
)
from kaos_llm_core.signatures import InputField, OutputField, Signature

pytestmark = pytest.mark.integration


def _has_key(*env_vars: str) -> bool:
    return any(os.getenv(v) for v in env_vars)


requires_anthropic = pytest.mark.skipif(
    not _has_key("KAOS_LLM_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    reason="No Anthropic API key",
)
posix_only = pytest.mark.skipif(
    sys.platform == "win32",
    reason="ProgramOfThought sandbox uses POSIX rlimits",
)


HAIKU = "claude-haiku-4-5"


class _MathSig(Signature):
    """Solve a math word problem and return the numeric answer as a string."""

    question: str = InputField(description="The math question")
    answer: str = OutputField(description="The numeric answer (digits only)")


@requires_anthropic
class TestMultiChainComparisonLive:
    async def test_python_api_against_haiku(self) -> None:
        """Direct Python API: 3 reasoning chains, aggregator synthesis."""
        m = MultiChainComparison(_MathSig, n=3, producer_model=HAIKU)
        result = await m(
            question=(
                "A bookstore sold 12 books on Monday and twice as many on "
                "Tuesday. How many books did it sell across both days?"
            )
        )
        assert isinstance(result, MultiChainComparisonResult)
        assert len(result.chains) == 3
        assert result.failed_count == 0
        # The synthesized answer should contain "36"
        assert "36" in str(result.answer), (
            f"Expected '36' in answer; got {result.answer!r}. "
            f"Chains: {[c.output for c in result.chains]}"
        )
        print(
            f"\n  [mcc_live] synthesized: {result.answer}, "
            f"chain answers: {[str(c.output.answer) for c in result.chains]}"
        )

    async def test_envelope_dispatch_against_haiku(self) -> None:
        """v3 envelope with kind='multi_chain_comparison' resolves and runs."""
        envelope = {
            "kaos_program": "1.1",
            "name": "mcc-math",
            "inputs": {"question": {"type": "string"}},
            "clients": {"default": {"provider": "anthropic", "model": HAIKU}},
            "steps": [
                {
                    "id": "solve",
                    "kind": "multi_chain_comparison",
                    "client": "default",
                    "instruction": (
                        "Solve the math word problem step by step. The "
                        "answer field must contain only the numeric answer."
                    ),
                    "n": 3,
                    "inputs": {"question": "$.inputs.question"},
                    "output_fields": {
                        "answer": {
                            "description": "The numeric answer",
                            "type": {"type": "string"},
                        }
                    },
                }
            ],
            "output": {"answer": "$.steps.solve.output.answer"},
            "capabilities": [
                "multi_chain_comparison",
                "jsonpointer_refs",
            ],
        }
        program = from_envelope(envelope)
        invocation = await program.invoke(
            question=(
                "If a train leaves at 3pm and arrives 4 hours later, "
                "what hour does it arrive? "
                "Answer with the hour as a number 1-24."
            )
        )
        assert "answer" in invocation.output
        # 3pm + 4h = 7pm = hour 19, but agents may pick "7" — accept either
        answer_str = str(invocation.output["answer"])
        assert any(token in answer_str for token in ("19", "7")), (
            f"Expected 7 or 19 in answer; got {answer_str!r}"
        )


@requires_anthropic
@posix_only
class TestProgramOfThoughtLive:
    async def test_python_api_against_haiku(self) -> None:
        """Direct Python API: code-as-reasoning for a math problem."""
        p = ProgramOfThought(
            _MathSig,
            producer_model=HAIKU,
            allow_code_execution=True,
            timeout_s=8.0,
        )
        result = await p(question="What is the sum of all integers from 1 to 100?")
        assert isinstance(result, ProgramOfThoughtResult)
        # Code should have been generated
        assert result.execution.code, "code writer should have produced code"
        # Sandbox should have executed it cleanly
        assert result.execution.return_code == 0, (
            f"Sandbox failed: rc={result.execution.return_code}, "
            f"stderr={result.execution.stderr[:300]}"
        )
        assert not result.execution.timed_out
        # The answer (5050) should appear in the interpreted output
        assert "5050" in str(result.answer), (
            f"Expected 5050; got answer={result.answer!r}, stdout={result.execution.stdout!r}"
        )
        print(
            f"\n  [pot_live] code lines: {len(result.execution.code.splitlines())}, "
            f"stdout: {result.execution.stdout.strip()[:80]}, "
            f"answer: {result.answer}"
        )

    async def test_envelope_dispatch_against_haiku(self) -> None:
        """v3 envelope with kind='program_of_thought' resolves and runs."""
        envelope = {
            "kaos_program": "1.1",
            "name": "pot-arith",
            "inputs": {"question": {"type": "string"}},
            "clients": {"default": {"provider": "anthropic", "model": HAIKU}},
            "steps": [
                {
                    "id": "compute",
                    "kind": "program_of_thought",
                    "client": "default",
                    "instruction": (
                        "Solve the arithmetic problem by writing a Python "
                        "program. The answer field must be a numeric string."
                    ),
                    "allow_code_execution": True,
                    "timeout_s": 8.0,
                    "inputs": {"question": "$.inputs.question"},
                    "output_fields": {
                        "answer": {
                            "description": "The numeric answer",
                            "type": {"type": "string"},
                        }
                    },
                }
            ],
            "output": {"answer": "$.steps.compute.output.answer"},
            "capabilities": [
                "program_of_thought",
                "jsonpointer_refs",
            ],
        }
        program = from_envelope(envelope)
        invocation = await program.invoke(
            question=("What is the product of the first 5 prime numbers (2, 3, 5, 7, 11)?")
        )
        # 2*3*5*7*11 = 2310
        answer_str = str(invocation.output.get("answer", ""))
        assert "2310" in answer_str, f"Expected 2310; got {answer_str!r}"

    async def test_envelope_rejects_missing_opt_in(self) -> None:
        """An envelope that uses program_of_thought without
        allow_code_execution=true must fail validation BEFORE any
        subprocess is spawned."""
        from kaos_llm_core.programs.envelope import ProgramEnvelopeError

        envelope = {
            "kaos_program": "1.1",
            "name": "no-opt-in",
            "inputs": {"question": {"type": "string"}},
            "clients": {"default": {"provider": "anthropic", "model": HAIKU}},
            "steps": [
                {
                    "id": "compute",
                    "kind": "program_of_thought",
                    "client": "default",
                    "instruction": "Anything",
                    # NO allow_code_execution → should be rejected
                    "inputs": {"question": "$.inputs.question"},
                    "output_fields": {
                        "answer": {
                            "description": "Answer",
                            "type": {"type": "string"},
                        }
                    },
                }
            ],
            "output": {"answer": "$.steps.compute.output.answer"},
            "capabilities": ["program_of_thought", "jsonpointer_refs"],
        }
        with pytest.raises(ProgramEnvelopeError, match="allow_code_execution"):
            from_envelope(envelope)
