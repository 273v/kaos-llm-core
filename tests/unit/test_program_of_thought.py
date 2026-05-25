"""Unit tests for the Phase 16.2 ProgramOfThought Program.

Exercises the subprocess sandbox directly (no LLM) and the full
two-phase pipeline through deterministic FunctionClient stubs.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import pytest
from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart

from kaos_llm_core.observability.cost import PRICING, ModelPricing
from kaos_llm_core.programs.program_of_thought import (
    ProgramOfThought,
    ProgramOfThoughtError,
    ProgramOfThoughtResult,
    _strip_code_fence,
)
from kaos_llm_core.signatures import InputField, OutputField, Signature

# Linux-only tests (rlimits via preexec_fn require POSIX)
pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="ProgramOfThought sandbox is POSIX-only"
)


class _MathSig(Signature):
    """Solve a math problem."""

    question: str = InputField(description="The math question")
    answer: str = OutputField(description="The numeric answer as a string")


def _json_response(payload: dict[str, Any]) -> ProviderResponse:
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


# ---------------------------------------------------------------------------
# Sandbox direct tests
# ---------------------------------------------------------------------------


class TestExamplesForwarding:
    """Pin the few-shot grounding forwarding contract for both inner Calls.

    Callers that enforce a grounded-Signature contract (e.g. kaos-agents'
    ``Call(SigClass, examples=load_examples("..."))`` pattern) need to
    keep the same calibration when routing through ``ProgramOfThought``.
    The two inner Calls (code writer + interpreter) operate on different
    synthesised Signatures, so the kwargs are independent.
    """

    def test_code_writer_examples_forwarded(self) -> None:
        from kaos_llm_core.types import Example

        examples = [
            Example(
                inputs={"question": "what is 2+2?"},
                outputs={"code": "print(2 + 2)"},
            ),
        ]
        p = ProgramOfThought(
            _MathSig,
            producer_model="function:function-test",
            code_writer_examples=examples,
        )
        assert list(p.code_writer.examples) == examples

    def test_interpreter_examples_forwarded(self) -> None:
        from kaos_llm_core.types import Example

        examples = [
            Example(
                inputs={"question": "what is 2+2?", "raw_output": "4\n"},
                outputs={"answer": "4"},
            ),
        ]
        p = ProgramOfThought(
            _MathSig,
            producer_model="function:function-test",
            interpreter_examples=examples,
        )
        assert list(p.interpreter.examples) == examples

    def test_independent_examples_kwargs(self) -> None:
        """The writer and interpreter kwargs do NOT cross-contaminate."""
        from kaos_llm_core.types import Example

        writer_examples = [
            Example(inputs={"question": "q1"}, outputs={"code": "print(1)"}),
        ]
        interpreter_examples = [
            Example(inputs={"question": "q1", "raw_output": "1\n"}, outputs={"answer": "1"}),
        ]
        p = ProgramOfThought(
            _MathSig,
            producer_model="function:function-test",
            code_writer_examples=writer_examples,
            interpreter_examples=interpreter_examples,
        )
        assert list(p.code_writer.examples) == writer_examples
        assert list(p.interpreter.examples) == interpreter_examples

    def test_examples_default_none_keeps_existing_behaviour(self) -> None:
        """Omitting both kwargs leaves both inner Calls ungrounded — back-compat."""
        p = ProgramOfThought(_MathSig, producer_model="function:function-test")
        assert not getattr(p.code_writer, "examples", None)
        assert not getattr(p.interpreter, "examples", None)


class TestSandbox:
    def test_default_refuses_execution(self) -> None:
        p = ProgramOfThought(_MathSig, producer_model="function:function-test")
        with pytest.raises(ProgramOfThoughtError, match="allow_code_execution"):
            p._execute_code("print(1+1)")

    def test_opt_in_runs_print(self) -> None:
        p = ProgramOfThought(
            _MathSig,
            producer_model="function:function-test",
            allow_code_execution=True,
        )
        result = p._execute_code("print(2 + 3)")
        assert result.return_code == 0
        assert result.timed_out is False
        assert result.stdout.strip() == "5"

    def test_wall_timeout_kills_runaway(self) -> None:
        p = ProgramOfThought(
            _MathSig,
            producer_model="function:function-test",
            allow_code_execution=True,
            timeout_s=1.0,
        )
        result = p._execute_code("import time; time.sleep(10); print('done')")
        assert result.timed_out is True
        assert result.return_code == -1

    def test_audit_callback_can_refuse(self) -> None:
        def auditor(code: str) -> None:
            if "rm" in code:
                raise RuntimeError("audit refused: contains 'rm'")

        p = ProgramOfThought(
            _MathSig,
            producer_model="function:function-test",
            allow_code_execution=True,
            audit_callback=auditor,
        )
        # Innocuous code passes
        result = p._execute_code("print(42)")
        assert result.return_code == 0

        # Forbidden code is refused before any subprocess spawns
        with pytest.raises(RuntimeError, match="audit refused"):
            p._execute_code("import os; os.system('rm -rf /')")

    def test_subprocess_runs_in_isolated_cwd(self) -> None:
        """Each call gets a fresh tempdir and the cwd does NOT leak the
        host project files."""
        p = ProgramOfThought(
            _MathSig,
            producer_model="function:function-test",
            allow_code_execution=True,
        )
        # The kaos-llm-core dir contains pyproject.toml. The sandboxed
        # cwd should be a fresh tempdir, so listing the cwd should
        # NOT include pyproject.toml.
        code = "import os; print(','.join(sorted(os.listdir('.'))))"
        result = p._execute_code(code)
        assert result.return_code == 0
        assert "pyproject.toml" not in result.stdout

    def test_subprocess_env_does_not_leak_secrets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Set a fake secret in the parent env and confirm the
        subprocess does NOT see it (only PATH is preserved)."""
        monkeypatch.setenv("KAOS_TEST_SECRET", "shhh")
        p = ProgramOfThought(
            _MathSig,
            producer_model="function:function-test",
            allow_code_execution=True,
        )
        code = "import os; print(os.environ.get('KAOS_TEST_SECRET', 'not-found'))"
        result = p._execute_code(code)
        assert result.return_code == 0
        assert result.stdout.strip() == "not-found"


# ---------------------------------------------------------------------------
# Helper: code-fence stripping
# ---------------------------------------------------------------------------


class TestStripCodeFence:
    def test_strips_python_fence(self) -> None:
        assert _strip_code_fence("```python\nprint(1)\n```") == "print(1)"

    def test_strips_unlabeled_fence(self) -> None:
        assert _strip_code_fence("```\nprint(2)\n```") == "print(2)"

    def test_no_fence_passthrough(self) -> None:
        assert _strip_code_fence("print(3)") == "print(3)"

    def test_strips_only_outer_fence(self) -> None:
        # Inner backticks in a multi-line program are preserved
        code = "```\nprint('inner ``` test')\n```"
        assert _strip_code_fence(code) == "print('inner ``` test')"


# ---------------------------------------------------------------------------
# Full pipeline through deterministic stubs
# ---------------------------------------------------------------------------


class TestFullPipeline:
    async def test_two_phase_pipeline_against_function_client(self) -> None:
        """Code writer returns 'print(2+2)', sandbox runs it, interpreter
        reads stdout '4' and produces the final answer."""
        phase = {"n": 0}

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            phase["n"] += 1
            text_blob = json.dumps(messages)
            if "code" in text_blob and "raw_output" not in text_blob:
                # Code writer phase
                return _json_response({"code": "print(2 + 2)"})
            # Interpreter phase
            return _json_response({"answer": "4"})

        client = FunctionClient(function=fn)
        p = ProgramOfThought(
            _MathSig,
            producer_model="function:function-test",
            allow_code_execution=True,
        )
        for sub in p.named_calls().values():
            sub._client = client

        result = await p(question="What is 2+2?")
        assert isinstance(result, ProgramOfThoughtResult)
        assert result.execution.stdout.strip() == "4"
        assert result.execution.return_code == 0
        assert result.answer == "4"

    async def test_pipeline_refuses_when_opt_in_missing(self) -> None:
        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"code": "print('hello')"})

        client = FunctionClient(function=fn)
        # Default: allow_code_execution=False
        p = ProgramOfThought(_MathSig, producer_model="function:function-test")
        for sub in p.named_calls().values():
            sub._client = client

        with pytest.raises(ProgramOfThoughtError, match="allow_code_execution"):
            await p(question="What is 1+1?")

    async def test_empty_code_raises(self) -> None:
        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"code": "   "})

        client = FunctionClient(function=fn)
        p = ProgramOfThought(
            _MathSig,
            producer_model="function:function-test",
            allow_code_execution=True,
        )
        for sub in p.named_calls().values():
            sub._client = client

        with pytest.raises(ProgramOfThoughtError, match="empty"):
            await p(question="What is 0?")
