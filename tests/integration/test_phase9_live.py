"""Live integration tests for kaos-llm-core Phase 9 (audit fixes).

These hit real LLM provider APIs to verify the Phase 9 audit fixes
end-to-end. Companion to ``tests/unit/test_audit_phase9.py`` which has
the deterministic ``FunctionClient``-backed regression suite.

Coverage:

- Audit #1 / #86: ReAct uses Call's public ``prepare_call`` API end-to-end
  against real Anthropic.
- Audit #2 / #87: ``CallHooks`` and ``ProgramHooks`` fire around a real
  ReAct loop and Refine cycle.
- Audit #5/#6 / #90: Two concurrent Refine.__call__ against the same
  Refine instance complete without racing on instructions.
- Audit #7 / #91: BestOfN with a real producer mutates only its clones.
- Audit Finding A / #108: estimate_eval_cost on a real Program-returned
  prediction reports non-zero cost (the fix attaches the parent trace).
- Audit Finding B / #109: BestOfN's parent trace has N sample children
  with non-zero token counts after a real run.
- Audit Finding C / #110: A Program that calls the same Call multiple
  times has every invocation captured in the trace.
- Audit Finding D / #111: CascadeRouter respects the parent's client= when
  resolving the routed Call.

Run::

    uv run pytest tests/integration/test_phase9_live.py -v -s
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

import pytest

from kaos_llm_core.optimization.budget import estimate_eval_cost
from kaos_llm_core.programs.base import Program
from kaos_llm_core.programs.best_of_n import BestOfN
from kaos_llm_core.programs.call import Call, CallPlan
from kaos_llm_core.programs.hooks import CallHooks
from kaos_llm_core.programs.judge import Judge
from kaos_llm_core.programs.program_hooks import ProgramHooks
from kaos_llm_core.programs.react import ReAct
from kaos_llm_core.programs.refine import Refine
from kaos_llm_core.programs.tool import Tool
from kaos_llm_core.signatures import InputField, OutputField, Signature

pytestmark = pytest.mark.integration


def _has_key(*env_vars: str) -> bool:
    return any(os.getenv(v) for v in env_vars)


requires_anthropic = pytest.mark.skipif(
    not _has_key("KAOS_LLM_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    reason="Anthropic API key not set",
)


# Cheapest current-generation model — every Phase 9 test uses this so the
# whole pass costs cents, not dollars.
HAIKU = "anthropic:claude-haiku-4-5"


# ---------------------------------------------------------------------------
# Signatures
# ---------------------------------------------------------------------------


class Question(Signature):
    """Answer the user's question. Use the calculator if you need to compute."""

    question: str = InputField(description="The question to answer")
    answer: str = OutputField(description="The final answer, concise")


class Haiku(Signature):
    """Write a haiku about the topic."""

    topic: str = InputField(description="What the haiku is about")
    haiku: str = OutputField(description="Three lines, 5-7-5 syllable structure")


class ShortAnswer(Signature):
    """Give a one-sentence answer."""

    question: str = InputField(description="The question")
    answer: str = OutputField(description="A single sentence")


# ---------------------------------------------------------------------------
# Tools (for ReAct)
# ---------------------------------------------------------------------------


def calculator(a: float, b: float, op: str) -> float:
    """Compute a + b, a - b, a * b, or a / b. ``op`` is one of '+','-','*','/'."""
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        return a / b
    raise ValueError(f"Unknown op {op!r}; use one of '+','-','*','/'.")


# ---------------------------------------------------------------------------
# Audit #1 / #86 — ReAct uses prepare_call against a real provider
# ---------------------------------------------------------------------------


@requires_anthropic
class TestReActPublicAPILive:
    async def test_prepare_call_against_real_provider(self) -> None:
        """Call.prepare_call returns a usable plan against a real provider."""
        call = Call(Question, model=HAIKU)
        plan = await call.prepare_call({"question": "What is 2+2?"})
        assert isinstance(plan, CallPlan)
        assert plan.model == HAIKU
        assert plan.client is not None
        assert plan.effective_signature is Question
        assert plan.validated_inputs == {"question": "What is 2+2?"}

    async def test_react_with_calculator_tool_against_real_haiku(self) -> None:
        """End-to-end ReAct against real Anthropic, using the public API."""
        react = ReAct(
            Question,
            tools=[Tool.from_callable(calculator)],
            model=HAIKU,
            max_iterations=4,
        )
        invocation = await react.invoke(question="What is 17 times 23? Use the calculator.")
        result = invocation.output
        assert result.is_complete()
        # Real model should compute the answer correctly
        assert "391" in result.answer  # type: ignore[attr-defined]
        # Trajectory shows at least one tool call
        tool_calls = sum(len(it.tool_calls) for it in result.trajectory)
        assert tool_calls >= 1
        # Trace tree has children with real tokens
        assert invocation.trace is not None
        assert invocation.trace.total_tokens > 0
        print(
            f"[react_calculator_live] answer='{result.answer}' "  # type: ignore[attr-defined]
            f"iterations={result.iterations_used} tokens={invocation.trace.total_tokens}"
        )


# ---------------------------------------------------------------------------
# Audit #2 / #87 — Hooks fire end-to-end
# ---------------------------------------------------------------------------


@requires_anthropic
class TestHooksFireLive:
    async def test_call_hooks_and_program_hooks_fire_around_react(self) -> None:
        """Both hook layers fire around a real ReAct loop."""
        events: list[str] = []

        def on_start(call: Any, inputs: Any, *, context: Any = None) -> None:
            events.append("call_start")

        def on_end(call: Any, inputs: Any, invocation: Any, *, context: Any = None) -> None:
            events.append("call_end")

        def on_iter(prog: Any, i: int, payload: Any, *, context: Any = None) -> None:
            events.append(f"iter_{i}")

        def on_prog_start(prog: Any, inputs: Any, *, context: Any = None) -> None:
            events.append("prog_start")

        def on_prog_end(prog: Any, inputs: Any, invocation: Any, *, context: Any = None) -> None:
            events.append("prog_end")

        hooks = CallHooks(on_call_start=on_start, on_call_end=on_end)
        prog_hooks = ProgramHooks(
            on_program_start=on_prog_start,
            on_program_end=on_prog_end,
            on_iteration=on_iter,
        )

        react = ReAct(
            Question,
            tools=[Tool.from_callable(calculator)],
            model=HAIKU,
            max_iterations=4,
            hooks=hooks,
            program_hooks=prog_hooks,
        )
        await react(question="What is 5 + 7?")
        assert "call_start" in events
        assert "call_end" in events
        assert "prog_start" in events
        assert "prog_end" in events
        # At least one iteration hook fired
        assert any(e.startswith("iter_") for e in events), events
        print(f"[hooks_live] events={events}")


# ---------------------------------------------------------------------------
# Audit #5 / #6 / #90 — Refine concurrency
# ---------------------------------------------------------------------------


@requires_anthropic
class TestRefineConcurrencyLive:
    async def test_two_concurrent_refines_against_real_provider(self) -> None:
        """Two parallel Refine.__call__ on the same instance complete cleanly.

        Pre-fix this would race on the producer's instructions slot. With
        the per-iteration clone fix, both calls have isolated clones and
        the original producer is never mutated.
        """
        producer = Call(Haiku, model=HAIKU)
        original_instructions = producer.instructions

        judge = Judge(
            Haiku,
            producer_model=HAIKU,
            judge_model=HAIKU,
            criteria="poetic quality, vivid imagery, 5-7-5 syllable structure",
        )

        refine = Refine(producer, judge, max_iterations=2, min_score=0.5)

        results = await asyncio.gather(
            refine(topic="autumn rain"),
            refine(topic="winter night"),
        )

        # Both completed without crashing
        assert all(r.iterations >= 1 for r in results)
        # The original producer's instructions were never mutated
        assert producer.instructions == original_instructions
        print(
            f"[refine_concurrency_live] iterations={[r.iterations for r in results]} "
            f"scores={[round(r.final_score, 2) for r in results]}"
        )


# ---------------------------------------------------------------------------
# Audit Finding A / #108 — estimate_eval_cost finds Program traces
# ---------------------------------------------------------------------------


class _AnalyzerProgram(Program):
    """Trivial Program that wraps a Call and returns its result."""

    def __init__(self, call: Call) -> None:
        self.extract = call

    async def forward(self, **kwargs: Any) -> Any:
        return await self.extract(**kwargs)


@requires_anthropic
class TestProgramTraceAttachmentLive:
    async def test_estimate_eval_cost_on_real_program_prediction(self) -> None:
        """Run a Program against real haiku, then verify estimate_eval_cost
        finds the per-example trace from each Invocation and reports
        nonzero cost. Phase 10: traces live on the Invocation, not on
        the prediction object."""
        producer = Call(ShortAnswer, model=HAIKU)
        prog = _AnalyzerProgram(producer)

        inv1 = await prog.invoke(question="What's the capital of France?")
        inv2 = await prog.invoke(question="What's 2+2?")

        @dataclass
        class FakeER:
            prediction: Any
            trace: Any

        @dataclass
        class FakeEvalResult:
            per_example: list[Any]

        eval_result = FakeEvalResult(
            per_example=[
                FakeER(inv1.output, inv1.trace),
                FakeER(inv2.output, inv2.trace),
            ]
        )
        cost, tokens = estimate_eval_cost(eval_result)
        # Tokens > 0 proves the per-example trace was captured from the
        # Invocation and the trace tree's total_tokens rolled up correctly.
        assert tokens > 0
        assert cost > 0.0
        print(f"[program_trace_attached_live] cost=${cost:.6f} tokens={tokens}")


# ---------------------------------------------------------------------------
# Audit Finding B / #109 — BestOfN trace tree has children
# ---------------------------------------------------------------------------


@requires_anthropic
class TestBestOfNTraceLive:
    async def test_best_of_n_parent_trace_has_real_children(self) -> None:
        """BestOfN against real haiku produces a parent trace with N children."""
        producer = Call(Haiku, model=HAIKU)

        def length_metric(output: Any, inputs: Any) -> float:
            return float(len(output.haiku))

        program = BestOfN(
            producer,
            n=3,
            selector=length_metric,
            seed_strategy="temperature",
            diversity_temperature=1.0,
        )
        invocation = await program.invoke(topic="cherry blossoms")
        result = invocation.output
        trace = invocation.trace
        assert trace is not None
        # Children = 3 sample clones (each pushed via the trace collector)
        assert len(trace.children) >= 3, (
            f"Expected at least 3 sample children, got {len(trace.children)}"
        )
        assert trace.total_tokens > 0
        print(
            f"[bestofn_trace_live] children={len(trace.children)} "
            f"total_tokens={trace.total_tokens} winner_score={result.scores[result.selected_index]}"
        )


# ---------------------------------------------------------------------------
# Audit Finding C / #110 — Repeated invocations of same child are captured
# ---------------------------------------------------------------------------


class _LoopProgram(Program):
    """Calls the same Call three times in forward()."""

    def __init__(self, call: Call) -> None:
        self.producer = call

    async def forward(self, **kwargs: Any) -> Any:
        outputs = []
        for _ in range(3):
            outputs.append(await self.producer(**kwargs))
        return outputs


@requires_anthropic
class TestRepeatedInvocationsLive:
    async def test_program_loop_captures_every_invocation(self) -> None:
        """A real Program that calls a real Call 3 times has 3 child traces."""
        producer = Call(ShortAnswer, model=HAIKU)
        prog = _LoopProgram(producer)
        invocation = await prog.invoke(question="Pick a random color.")
        trace = invocation.trace
        assert trace is not None
        assert len(trace.children) == 3, (
            f"Expected 3 child traces (one per loop iteration), got {len(trace.children)}"
        )
        # Each child has real tokens
        assert all(c.total_tokens > 0 for c in trace.children)
        print(
            f"[repeated_invocations_live] children={len(trace.children)} "
            f"total_tokens={trace.total_tokens}"
        )


# ---------------------------------------------------------------------------
# Audit Finding D / #111 — CascadeRouter propagates client (and hooks)
# ---------------------------------------------------------------------------


@requires_anthropic
class TestCascadeRouterPropagationLive:
    async def test_cascade_router_uses_caller_client_and_hooks(self) -> None:
        """A CascadeRouter run against real haiku must fire CallHooks if the
        parent had any. Pre-fix the cascade clones dropped hooks=."""
        from kaos_llm_core.router.cascade import CascadeRouter

        events: list[str] = []

        def on_start(call: Any, inputs: Any, *, context: Any = None) -> None:
            events.append("start")

        def on_end(call: Any, inputs: Any, invocation: Any, *, context: Any = None) -> None:
            events.append("end")

        hooks = CallHooks(on_call_start=on_start, on_call_end=on_end)
        # Single-element cascade so the first attempt is the accepted one.
        router = CascadeRouter(models=[HAIKU])
        call = Call(ShortAnswer, router=router, hooks=hooks)
        await call(question="What is 1+1?")
        # The hooks fired on the routed clone, not just the parent
        assert "start" in events
        assert "end" in events
        print(f"[cascade_router_live] hook_events={events}")
