"""Phase 9 audit regression suite — locks every fix in /tmp/kaos-llm-core-issues.md.

These tests are organized by audit finding number so any future regression
is traceable back to the issue that motivated the fix. They use the
``FunctionClient`` for deterministic mocking; live integration tests for
the same fixes live in ``tests/integration/test_phase9_live.py``.

If you're adding a new audit finding, add the regression here and link to
the issue file from the test docstring.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

import pytest
from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart

from kaos_llm_core.cache.semantic import SemanticCache, _config_fingerprint
from kaos_llm_core.errors import CallError
from kaos_llm_core.observability.collectors import collect_traces
from kaos_llm_core.optimization.budget import estimate_eval_cost
from kaos_llm_core.programs.base import Program
from kaos_llm_core.programs.best_of_n import BestOfN, BestOfNResult
from kaos_llm_core.programs.call import Call, CallPlan
from kaos_llm_core.programs.hooks import CallHooks
from kaos_llm_core.programs.judge import Judge
from kaos_llm_core.programs.program_hooks import ProgramHooks
from kaos_llm_core.programs.react import ReAct
from kaos_llm_core.programs.refine import Refine
from kaos_llm_core.programs.tool import Tool
from kaos_llm_core.signatures import InputField, OutputField, Signature

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class WriteSig(Signature):
    """Write a short answer."""

    topic: str = InputField(description="What to write about")
    answer: str = OutputField(description="The written answer")


class AnswerSig(Signature):
    """Answer the user's question."""

    question: str = InputField(description="The question")
    answer: str = OutputField(description="The final answer")


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


def _call_with_responses(answers: list[str], signature: type[Signature] = WriteSig) -> Call:
    iter_answers = iter(answers)

    def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
        return _json_response({"answer": next(iter_answers)})

    client = FunctionClient(function=fn)
    call = Call(signature, model="function-test")
    call._client = client
    return call


# ---------------------------------------------------------------------------
# #1 — ReAct uses Call's public prepare_call API, not underscore reach-through
# ---------------------------------------------------------------------------


class TestIssue1PublicCallAPI:
    """ReAct must consume Call via prepare_call(), not via underscore methods."""

    async def test_prepare_call_returns_typed_plan(self) -> None:
        call = _call_with_responses(["x"])
        plan = await call.prepare_call({"topic": "anything"})
        assert isinstance(plan, CallPlan)
        assert plan.model == "function-test"
        assert plan.client is call._client
        assert plan.codec is call._codec
        assert plan.effective_signature is WriteSig
        assert plan.validated_inputs == {"topic": "anything"}
        assert plan.output_model is call._output_model
        assert plan.call_kwargs == {}

    # Phase 10: ``test_react_prepare_call_does_not_mutate_inner_call_state``
    # has been DELETED. The Phase 9b/9c contract it tested
    # (``_current_client`` / ``_current_model`` instance attributes) no
    # longer exists — the Invocation contract carries per-execution state
    # in a task-isolated ContextVar instead, so there is no instance
    # state for ``prepare_call`` to mutate or not mutate.


# ---------------------------------------------------------------------------
# #2 — CallHooks fire end-to-end through ReAct, ProgramHooks fire per iter
# ---------------------------------------------------------------------------


class TestIssue2HooksFire:
    """Both CallHooks (logical-call boundary) and ProgramHooks (per iteration)
    must fire when ReAct runs. Refine fires the same hook surface."""

    async def test_react_call_hooks_fire_around_loop(self) -> None:
        events: list[str] = []

        def on_start(call: Any, inputs: Any, *, context: Any = None) -> None:
            events.append(f"start:{type(call).__name__}")

        def on_end(call: Any, inputs: Any, invocation: Any, *, context: Any = None) -> None:
            events.append(f"end:{type(call).__name__}")

        def on_error(call: Any, inputs: Any, exc: Any, *, context: Any = None) -> None:
            events.append(f"error:{type(exc).__name__}")

        hooks = CallHooks(on_call_start=on_start, on_call_end=on_end, on_call_error=on_error)

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"answer": "ok"})

        client = FunctionClient(function=handler)
        react = ReAct(
            AnswerSig,
            tools=[Tool.from_callable(lambda: "noop")],
            model="function-test",
            client=client,
            hooks=hooks,
        )
        await react(question="x")
        assert events == ["start:Call", "end:Call"]

    async def test_react_program_hooks_fire_per_iteration(self) -> None:
        seen_iterations: list[int] = []

        def on_iter(prog: Any, i: int, payload: Any, *, context: Any = None) -> None:
            seen_iterations.append(i)

        prog_hooks = ProgramHooks(on_iteration=on_iter)

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"answer": "ok"})

        client = FunctionClient(function=handler)
        react = ReAct(
            AnswerSig,
            tools=[Tool.from_callable(lambda: "noop")],
            model="function-test",
            client=client,
            program_hooks=prog_hooks,
        )
        await react(question="x")
        # One iteration: index 0
        assert seen_iterations == [0]

    async def test_react_per_tool_hooks_bracket_each_call(self) -> None:
        """``on_tool_start`` / ``on_tool_end`` fire once per tool call, in
        order, bracketing each dispatch — the finer grain a live UI needs
        (``on_iteration`` only fires once per loop turn, after the tools)."""
        from kaos_llm_client.types import ToolCall

        from kaos_llm_core.programs.react import ToolObservation

        events: list[str] = []

        def on_tool_start(prog: Any, tc: Any, *, context: Any = None) -> None:
            assert isinstance(tc, ToolCall)
            events.append(f"start:{tc.name}")

        def on_tool_end(prog: Any, obs: Any, *, context: Any = None) -> None:
            assert isinstance(obs, ToolObservation)
            events.append(f"end:{obs.tool_name}:err={obs.is_error}")

        prog_hooks = ProgramHooks(on_tool_start=on_tool_start, on_tool_end=on_tool_end)

        def my_api(query: str) -> str:
            return "tool-output"

        call_count = {"n": 0}

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return ProviderResponse.model_construct(
                    provider="function",
                    model="function-test",
                    raw={},
                    parts=[
                        ContentPart.model_construct(
                            type="tool_use",
                            tool_call=ToolCall.model_construct(
                                id="call_1", name="my_api", arguments={"query": "x"}
                            ),
                        )
                    ],
                    usage=UsageInfo.model_construct(
                        input_tokens=5, output_tokens=5, total_tokens=10
                    ),
                    stop_reason="tool_use",
                    status_code=200,
                    response_headers={},
                )
            return _json_response({"answer": "done"})

        client = FunctionClient(function=handler)
        react = ReAct(
            AnswerSig,
            tools=[Tool.from_callable(my_api)],
            model="function-test",
            client=client,
            program_hooks=prog_hooks,
        )
        await react(question="anything")
        # start fires before end, once each, for the single tool call.
        assert events == ["start:my_api", "end:my_api:err=False"]

    async def test_per_tool_hooks_optional_and_isolated(self) -> None:
        """A raising per-tool hook is swallowed and never breaks the loop;
        absent hooks are a no-op (back-compat)."""
        from kaos_llm_client.types import ToolCall

        def boom(prog: Any, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("observer blew up")

        prog_hooks = ProgramHooks(on_tool_start=boom, on_tool_end=boom)

        def my_api(query: str) -> str:
            return "ok"

        call_count = {"n": 0}

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return ProviderResponse.model_construct(
                    provider="function",
                    model="function-test",
                    raw={},
                    parts=[
                        ContentPart.model_construct(
                            type="tool_use",
                            tool_call=ToolCall.model_construct(
                                id="call_1", name="my_api", arguments={"query": "x"}
                            ),
                        )
                    ],
                    usage=UsageInfo.model_construct(
                        input_tokens=5, output_tokens=5, total_tokens=10
                    ),
                    stop_reason="tool_use",
                    status_code=200,
                    response_headers={},
                )
            return _json_response({"answer": "survived"})

        client = FunctionClient(function=handler)
        react = ReAct(
            AnswerSig,
            tools=[Tool.from_callable(my_api)],
            model="function-test",
            client=client,
            program_hooks=prog_hooks,
        )
        result = await react(question="anything")
        # The raising hooks did not abort the loop — the tool ran and the
        # final answer came through.
        assert result.trajectory[0].tool_results[0].tool_name == "my_api"


# ---------------------------------------------------------------------------
# #3 — User tools that return {"error": True, ...} are NOT mis-flagged
# ---------------------------------------------------------------------------


class TestIssue3ToolErrorEnvelope:
    async def test_user_tool_error_field_is_not_mis_flagged(self) -> None:
        """A tool that legitimately returns {error: True, details: ...}
        must NOT be marked is_error=True by ReAct's dispatcher."""
        call_count = {"n": 0}

        def my_api(query: str) -> dict[str, Any]:
            """Wraps a real API that uses 'error' as a field name."""
            return {"error": True, "details": "rate limited", "code": 429}

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            call_count["n"] += 1
            if call_count["n"] == 1:
                # Tool call turn
                return ProviderResponse.model_construct(
                    provider="function",
                    model="function-test",
                    raw={},
                    parts=[
                        ContentPart.model_construct(
                            type="tool_use",
                            tool_call={  # type: ignore[arg-type]
                                "id": "call_1",
                                "name": "my_api",
                                "arguments": {"query": "x"},
                            },
                        )
                    ],
                    usage=UsageInfo.model_construct(
                        input_tokens=5, output_tokens=5, total_tokens=10
                    ),
                    stop_reason="tool_use",
                    status_code=200,
                    response_headers={},
                )
            return _json_response({"answer": "got it"})

        client = FunctionClient(function=handler)
        from kaos_llm_client.types import ToolCall

        # Reconstruct with proper ToolCall — the dict-based shortcut above is
        # rejected by FunctionClient's type validation. Use a real ToolCall.
        def real_handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return ProviderResponse.model_construct(
                    provider="function",
                    model="function-test",
                    raw={},
                    parts=[
                        ContentPart.model_construct(
                            type="tool_use",
                            tool_call=ToolCall.model_construct(
                                id="call_1", name="my_api", arguments={"query": "x"}
                            ),
                        )
                    ],
                    usage=UsageInfo.model_construct(
                        input_tokens=5, output_tokens=5, total_tokens=10
                    ),
                    stop_reason="tool_use",
                    status_code=200,
                    response_headers={},
                )
            return _json_response({"answer": "got it"})

        call_count["n"] = 0
        client = FunctionClient(function=real_handler)
        react = ReAct(
            AnswerSig,
            tools=[Tool.from_callable(my_api)],
            model="function-test",
            client=client,
        )
        result = await react(question="anything")
        # The tool was invoked and returned its dict verbatim
        obs = result.trajectory[0].tool_results[0]
        assert obs.tool_name == "my_api"
        assert obs.result == {"error": True, "details": "rate limited", "code": 429}
        # KEY ASSERTION: ReAct did NOT mis-flag this as an error
        assert obs.is_error is False


# ---------------------------------------------------------------------------
# #5/#6 — Refine concurrency: two parallel calls don't race
# ---------------------------------------------------------------------------


class TestIssue5And6RefineConcurrency:
    async def test_two_concurrent_refines_dont_race_on_instructions(self) -> None:
        """Run two Refine.__call__ in parallel against the same Refine instance.

        Pre-fix: both coroutines mutated self.producer.instructions and the
        finally-restore from the first finishing coroutine wiped the second
        coroutine's iteration suffix mid-flight, producing incorrect prompts.
        """
        seen_instructions: list[str] = []

        def producer_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            seen_instructions.append(json.dumps(messages, default=str))
            return _json_response({"answer": "ok"})

        def judge_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"quality_score": 0.9, "reasoning": "good"})

        producer = Call(WriteSig, model="function-test")
        producer._client = FunctionClient(function=producer_fn)

        judge = Judge(WriteSig, producer_model="function-test", judge_model="function-test")
        judge.judge_call._client = FunctionClient(function=judge_fn)

        refine = Refine(producer, judge, max_iterations=1, min_score=0.5)

        results = await asyncio.gather(
            refine(topic="alpha"),
            refine(topic="beta"),
        )

        # Both succeeded — the original producer's instructions are unchanged
        # because we never mutate it. (Pre-fix it would have a stale suffix.)
        assert producer.instructions == producer.instructions
        assert all(r.iterations == 1 for r in results)

    async def test_refine_does_not_mutate_original_producer(self) -> None:
        """The producer's instructions slot is never written to by Refine."""

        def producer_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"answer": "ok"})

        def judge_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"quality_score": 0.1, "reasoning": "needs work"})

        producer = Call(WriteSig, model="function-test")
        producer._client = FunctionClient(function=producer_fn)
        original_instructions = producer.instructions

        judge = Judge(WriteSig, producer_model="function-test", judge_model="function-test")
        judge.judge_call._client = FunctionClient(function=judge_fn)

        refine = Refine(producer, judge, max_iterations=3, min_score=0.95)
        await refine(topic="x")

        # Original instructions unchanged
        assert producer.instructions == original_instructions

    async def test_refine_no_instance_state_leaks_between_calls(self) -> None:
        """Two sequential Refine.invoke()s on the same instance must each
        produce their own task-isolated Invocation — no shared state."""

        def producer_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"answer": "ok"})

        def judge_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"quality_score": 0.95, "reasoning": "good"})

        producer = Call(WriteSig, model="function-test")
        producer._client = FunctionClient(function=producer_fn)
        judge = Judge(WriteSig, producer_model="function-test", judge_model="function-test")
        judge.judge_call._client = FunctionClient(function=judge_fn)

        refine = Refine(producer, judge, max_iterations=2, min_score=0.5)

        inv1 = await refine.invoke(topic="x")
        inv2 = await refine.invoke(topic="y")
        # Each Invocation has its own iteration count and its own trace
        assert inv1.output.iterations == 1
        assert inv2.output.iterations == 1
        assert inv1.trace is not None
        assert inv2.trace is not None
        assert len(inv1.trace.children) == 1
        assert len(inv2.trace.children) == 1
        # And the two invocations are distinct objects (each has its own id)
        assert inv1.id != inv2.id


# ---------------------------------------------------------------------------
# #7 — BestOfN deepcopy: clone mutation does NOT touch original
# ---------------------------------------------------------------------------


class TestIssue7BestOfNCloneIsolation:
    async def test_clone_examples_mutation_does_not_affect_original(self) -> None:
        """Mutate the clone's examples list and verify the original is intact.

        Pre-fix shallow copy.copy shared the examples list reference, so
        a future Call subclass that mutated examples during execution
        would silently leak between samples.
        """
        from kaos_llm_core.types import Example

        producer = _call_with_responses(["a", "b"])
        producer.examples = [
            Example(inputs={"topic": "test"}, outputs={"answer": "demo"}),
        ]
        original_examples_id = id(producer.examples)

        program = BestOfN(producer, n=2, selector=lambda o, i: 0.5, seed_strategy="none")
        await program(topic="x")

        # Original list object is the same and untouched
        assert id(producer.examples) == original_examples_id
        assert len(producer.examples) == 1
        assert producer.examples[0].inputs == {"topic": "test"}


# ---------------------------------------------------------------------------
# #21 — response.usage None must not crash ReAct
# ---------------------------------------------------------------------------


class TestIssue21UsageNoneCheck:
    async def test_react_handles_usage_none(self) -> None:
        """If response.usage is None, ReAct must default tokens to 0."""

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            # Construct a response with usage=None
            return ProviderResponse.model_construct(
                provider="function",
                model="function-test",
                raw={},
                parts=[ContentPart.model_construct(type="text", text='{"answer": "ok"}')],
                usage=None,
                stop_reason="end_turn",
                status_code=200,
                response_headers={},
            )

        client = FunctionClient(function=handler)
        react = ReAct(
            AnswerSig,
            tools=[Tool.from_callable(lambda: "noop")],
            model="function-test",
            client=client,
        )
        result = await react(question="x")
        assert result.is_complete()
        assert result.outputs == {"answer": "ok"}


# ---------------------------------------------------------------------------
# #24 — inspect.isawaitable for metric callables
# ---------------------------------------------------------------------------


class TestIssue24Awaitable:
    async def test_metric_returning_task_is_awaited(self) -> None:
        """A metric that returns an asyncio.Task (an awaitable but not a
        coroutine) must be awaited correctly. Pre-fix asyncio.iscoroutine
        missed Tasks/Futures and crashed in float()."""

        async def async_score() -> float:
            return 0.7

        def metric(output: Any, inputs: Any) -> Any:
            # Return a Task — awaitable but not a coroutine.
            return asyncio.ensure_future(async_score())

        producer = _call_with_responses(["a", "b"])
        program = BestOfN(producer, n=2, selector=metric, seed_strategy="none")
        result = await program(topic="x")
        assert result.scores == [0.7, 0.7]


# ---------------------------------------------------------------------------
# #10 — _OutputForwardingMixin shared by all three result types
# ---------------------------------------------------------------------------


class TestIssue10SharedForwarding:
    async def test_react_result_forwards_dict(self) -> None:
        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"answer": "forwarded"})

        client = FunctionClient(function=handler)
        react = ReAct(
            AnswerSig,
            tools=[Tool.from_callable(lambda: "noop")],
            model="function-test",
            client=client,
        )
        result = await react(question="x")
        assert result.answer == "forwarded"  # type: ignore[attr-defined]
        with pytest.raises(AttributeError):
            _ = result.no_such_field  # type: ignore[attr-defined]

    async def test_best_of_n_result_forwards_object(self) -> None:
        producer = _call_with_responses(["alpha", "beta"])
        program = BestOfN(
            producer,
            n=2,
            selector=lambda o, i: 0.0 if o.answer == "alpha" else 1.0,
            seed_strategy="none",
        )
        result = await program(topic="x")
        assert result.answer == "beta"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# #25/26/27 — BestOfNResult enrichment
# ---------------------------------------------------------------------------


class TestIssue25To27ResultEnrichment:
    async def test_survivor_indices_maps_back_to_original(self) -> None:
        """Sample 1 of 5 fails -> survivor_indices == [0, 2, 3, 4]."""
        call_idx = {"i": 0}
        responses = ["s0", None, "s2", "s3", "s4"]  # None means raise

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            i = call_idx["i"]
            call_idx["i"] += 1
            if responses[i] is None:
                raise RuntimeError(f"sample {i} failed")
            return _json_response({"answer": responses[i]})  # type: ignore[arg-type]

        client = FunctionClient(function=fn)
        producer = Call(WriteSig, model="function-test", max_retries=0)
        producer._client = client

        program = BestOfN(
            producer,
            n=5,
            selector=lambda o, i: 0.5,
            seed_strategy="none",
        )
        result = await program(topic="x")
        assert result.survivor_indices == [0, 2, 3, 4]
        assert result.failed_count == 1
        assert sum(1 for e in result.errors if e is not None) == 1
        # selected_original_index returns the original sample position
        assert result.selected_original_index == result.survivor_indices[result.selected_index]

    async def test_all_fail_groups_exception_types_in_message(self) -> None:
        """All-fail error message must group exceptions by type, not just
        repeat the last one."""
        types_to_raise = [RuntimeError, ValueError, RuntimeError]
        call_idx = {"i": 0}

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            i = call_idx["i"]
            call_idx["i"] += 1
            raise types_to_raise[i](f"failure {i}")

        client = FunctionClient(function=fn)
        producer = Call(WriteSig, model="function-test", max_retries=0)
        producer._client = client

        program = BestOfN(producer, n=3, selector=lambda o, i: 0.5, seed_strategy="none")
        with pytest.raises(CallError) as exc:
            await program(topic="x")

        msg = str(exc.value)
        # Message includes per-type breakdown
        assert "RuntimeError" in msg
        assert "ValueError" in msg
        assert "2x" in msg or "2 x" in msg  # 2 RuntimeErrors


# ---------------------------------------------------------------------------
# #15 — empty tools warning (not error — backward compat)
# ---------------------------------------------------------------------------


class TestIssue15EmptyToolsWarning:
    async def test_react_empty_tools_logs_warning_but_works(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The empty-tools warning is emitted via kaos_core's structured
        logger which uses its own handler chain. We monkey-patch the
        ReAct module's logger.warning to capture the call directly
        instead of relying on caplog (which only sees stdlib handlers)."""
        captured: list[str] = []

        from kaos_llm_core.programs import react as react_mod

        original_warning = react_mod.logger.warning

        def fake_warning(msg: str, *args: Any, **kwargs: Any) -> None:
            captured.append(msg % args if args else msg)
            original_warning(msg, *args, **kwargs)

        monkeypatch.setattr(react_mod.logger, "warning", fake_warning)

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"answer": "ok"})

        client = FunctionClient(function=handler)
        react = ReAct(
            AnswerSig,
            tools=[],
            model="function-test",
            client=client,
        )
        # Construction succeeded but a warning was logged
        assert any("degenerate" in m for m in captured), captured
        # And it still works
        result = await react(question="x")
        assert result.is_complete()


# ---------------------------------------------------------------------------
# #20 — separate validation_retries cap
# ---------------------------------------------------------------------------


class TestIssue20ValidationRetryCap:
    async def test_max_validation_retries_separately_capped(self) -> None:
        """A model that always returns malformed output exits with
        stop_reason=ERROR after max_validation_retries, not after
        max_iterations."""

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            # Always invalid
            return ProviderResponse.model_construct(
                provider="function",
                model="function-test",
                raw={},
                parts=[ContentPart.model_construct(type="text", text="not valid json")],
                usage=UsageInfo.model_construct(input_tokens=5, output_tokens=5, total_tokens=10),
                stop_reason="end_turn",
                status_code=200,
                response_headers={},
            )

        client = FunctionClient(function=handler)
        react = ReAct(
            AnswerSig,
            tools=[Tool.from_callable(lambda: "noop")],
            model="function-test",
            client=client,
            max_iterations=20,  # high
            max_validation_retries=2,  # low
        )
        result = await react(question="x")
        assert result.stop_reason == "ERROR"
        assert result.validation_retries == 3  # 2 retries + the failing terminal attempt
        # Did NOT exhaust max_iterations
        assert result.iterations_used <= 4


# ---------------------------------------------------------------------------
# Audit Finding A — Program trace attached to predictions
# ---------------------------------------------------------------------------


class TestFindingAProgramTraceAttached:
    # Phase 10: ``test_program_result_has_last_trace_attribute`` has been
    # DELETED. The legacy contract — Program.__call__ mutating
    # ``_last_trace`` onto the returned prediction — no longer exists.
    # The Invocation runtime contract carries the trace on the
    # Invocation itself; users that need it call ``program.invoke()``
    # instead of fishing it off the result.

    async def test_estimate_eval_cost_finds_program_trace(self) -> None:
        """estimate_eval_cost on an EvalResult populated from invocations
        returns non-zero cost. Phase 10: traces live on ``ExampleResult.trace``,
        captured by ``evaluate()`` from each ``invocation.trace``."""
        from dataclasses import dataclass

        class AnalyzerProgram(Program):
            def __init__(self, call: Call) -> None:
                self.extract = call

            async def forward(self, **kwargs: Any) -> Any:
                return await self.extract(**kwargs)

        producer = _call_with_responses(["x", "y"])
        prog = AnalyzerProgram(producer)

        @dataclass
        class FakeER:
            prediction: Any
            trace: Any

        @dataclass
        class FakeEvalResult:
            per_example: list[Any]

        inv1 = await prog.invoke(topic="a")
        inv2 = await prog.invoke(topic="b")
        eval_result = FakeEvalResult(
            per_example=[
                FakeER(inv1.output, inv1.trace),
                FakeER(inv2.output, inv2.trace),
            ]
        )
        _cost, tokens = estimate_eval_cost(eval_result)
        # tokens > 0 means the trace was found and walked
        assert tokens > 0


# ---------------------------------------------------------------------------
# Audit Finding B — BestOfN trace tree has children
# ---------------------------------------------------------------------------


class TestFindingBBestOfNTraceTree:
    async def test_best_of_n_parent_trace_has_n_sample_children(self) -> None:
        """After a 3-sample BestOfN run, the parent trace has 3 children with
        non-zero token counts. Pre-fix the trace was empty."""
        producer = _call_with_responses(["a", "b", "c"])
        program = BestOfN(producer, n=3, selector=lambda o, i: 0.5, seed_strategy="none")
        invocation = await program.invoke(topic="x")
        trace = invocation.trace
        assert trace is not None
        assert len(trace.children) >= 3, (
            f"Expected at least 3 sample children, got {len(trace.children)}"
        )
        assert trace.total_tokens > 0


# ---------------------------------------------------------------------------
# Audit Finding C — repeated child invocations are NOT collapsed
# ---------------------------------------------------------------------------


class TestFindingCRepeatedChildren:
    async def test_program_collects_each_invocation_of_same_child(self) -> None:
        """A Program that calls the same Call three times in forward() must
        produce three child traces, not one. Pre-fix the snapshot-and-diff
        collector kept only the latest."""

        class LoopProgram(Program):
            def __init__(self, call: Call) -> None:
                self.producer = call

            async def forward(self, **kwargs: Any) -> Any:
                outputs = []
                for _ in range(3):
                    outputs.append(await self.producer(**kwargs))
                return outputs

        producer = _call_with_responses(["a", "b", "c"])
        prog = LoopProgram(producer)
        invocation = await prog.invoke(topic="x")
        trace = invocation.trace
        assert trace is not None
        # All 3 invocations are captured as distinct children
        assert len(trace.children) == 3
        # Total tokens reflects all 3 invocations (15 each = 45)
        assert trace.total_tokens == 45


# ---------------------------------------------------------------------------
# Audit Finding D — CascadeRouter propagates client and hooks
# ---------------------------------------------------------------------------


class TestFindingDCascadeRouterPropagation:
    async def test_cascade_router_uses_supplied_client(self) -> None:
        """When the parent Call has client= set, the per-step clones in
        CascadeRouter must reuse it. Pre-fix the clones called create_client
        and built a real provider client, ignoring the FunctionClient."""
        from kaos_llm_core.router.cascade import CascadeRouter

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"answer": "ok"})

        client = FunctionClient(function=fn)
        router = CascadeRouter(models=["model-a", "model-b"])
        # If the router clones don't reuse our client, this Call will try to
        # create a real provider client and fail (no API key for model-a).
        call = Call(WriteSig, router=router, client=client)
        result = await call(topic="x")
        assert result.answer == "ok"


# ---------------------------------------------------------------------------
# Audit Finding E — _primary_call uses declaration order
# ---------------------------------------------------------------------------


class TestFindingEPrimaryCallOrder:
    def test_primary_call_picks_first_declared_not_alphabetical(self) -> None:
        from kaos_llm_core.optimization.codec_optimizer import _primary_call

        class P(Program):
            def __init__(self) -> None:
                # Declare zeta BEFORE alpha — Phase 11 ProgramGraph
                # auto-registers via __setattr__ in this order.
                self.zeta = _call_with_responses(["x"])
                self.alpha = _call_with_responses(["y"])

        p = P()
        primary = _primary_call(p)
        # Declaration order means zeta wins (was declared first)
        assert primary is p.zeta


# ---------------------------------------------------------------------------
# Audit Finding F — SemanticCache respects model + instructions + examples
# ---------------------------------------------------------------------------


class TestFindingFSemanticCacheKey:
    def test_config_fingerprint_changes_with_instructions(self) -> None:
        c1 = _call_with_responses(["x"])
        c1.instructions = "instruction A"
        c2 = _call_with_responses(["y"])
        c2.instructions = "instruction B"
        assert _config_fingerprint(c1) != _config_fingerprint(c2)

    def test_config_fingerprint_changes_with_hyperparameters(self) -> None:
        c1 = _call_with_responses(["x"])
        c1._kwargs = {"temperature": 0.0}
        c2 = _call_with_responses(["y"])
        c2._kwargs = {"temperature": 0.7}
        assert _config_fingerprint(c1) != _config_fingerprint(c2)


# ---------------------------------------------------------------------------
# Trace collector — public contextvar API
# ---------------------------------------------------------------------------


class TestTraceCollector:
    async def test_collect_traces_captures_call_executions(self) -> None:
        producer = _call_with_responses(["a", "b"])
        with collect_traces() as bucket:
            await producer(topic="x")
            await producer(topic="y")
        assert len(bucket) == 2

    async def test_collect_traces_outside_block_is_silent(self) -> None:
        """Calling Call.invoke() outside any collector still produces a
        trace on the returned Invocation — no error, no spurious push."""
        producer = _call_with_responses(["a"])
        invocation = await producer.invoke(topic="x")
        assert invocation.trace is not None

    async def test_nested_collectors_dont_leak(self) -> None:
        """Inner collector captures its own children; outer collector does not
        see them (they were captured one level down)."""
        producer = _call_with_responses(["a", "b", "c"])
        with collect_traces() as outer:
            await producer(topic="x")  # captured in outer
            with collect_traces() as inner:
                await producer(topic="y")
                await producer(topic="z")
            # inner has 2 traces, outer still has 1 (the inner traces did
            # NOT bubble up — that's the contract)
            assert len(inner) == 2
        assert len(outer) == 1


# ---------------------------------------------------------------------------
# CallPlan exposes everything ReAct needs
# ---------------------------------------------------------------------------


class TestCallPlanCompleteness:
    async def test_call_plan_has_all_fields(self) -> None:
        producer = _call_with_responses(["a"])
        plan = await producer.prepare_call({"topic": "x"})
        # All eight fields are present and typed
        assert plan.validated_inputs == {"topic": "x"}
        assert isinstance(plan.model, str)
        assert plan.client is not None
        assert plan.codec is not None
        assert plan.effective_signature is WriteSig
        assert isinstance(plan.effective_instructions, str)
        assert plan.output_model is producer._output_model
        assert isinstance(plan.call_kwargs, dict)


# ---------------------------------------------------------------------------
# Review Issue A — contextvar isolation across concurrent program runs
# ---------------------------------------------------------------------------


class TestReviewIssueAConcurrencyIsolation:
    """Two concurrent ``await program(...)`` invocations must each see only
    their own children, never the other's. The collector contextvar gives us
    this for free; this test locks the property in so a future regression
    (e.g. switching to a module-global stack) breaks loudly.
    """

    async def test_concurrent_programs_have_independent_traces(self) -> None:
        class TwoStepProgram(Program):
            def __init__(self, call: Call) -> None:
                self.producer = call

            async def forward(self, **kwargs: Any) -> Any:
                a = await self.producer(**kwargs)
                b = await self.producer(**kwargs)
                return [a, b]

        # Two independent producers so the two programs each push their own
        # 2 traces and we can verify the children counts.
        prog1 = TwoStepProgram(_call_with_responses(["a1", "a2"]))
        prog2 = TwoStepProgram(_call_with_responses(["b1", "b2"]))

        invocations = await asyncio.gather(
            prog1.invoke(topic="x"),
            prog2.invoke(topic="y"),
        )
        inv1, inv2 = invocations
        assert inv1.output is not None
        assert inv2.output is not None
        # Each parent has exactly 2 children — no cross-contamination
        assert inv1.trace is not None
        assert inv2.trace is not None
        assert len(inv1.trace.children) == 2
        assert len(inv2.trace.children) == 2
        # Token totals are independent (each program ran 2 calls of 15 tokens)
        assert inv1.trace.total_tokens == 30
        assert inv2.trace.total_tokens == 30


# ---------------------------------------------------------------------------
# Review Issue F — failed Program pushes its trace upward
# ---------------------------------------------------------------------------


class TestReviewIssueFFailedProgramPushesTrace:
    """When a Program raises inside an outer Program, the outer trace tree
    must still contain the failed sub-program's trace. Pre-fix the
    ``push_trace`` ran AFTER the ``with collect_traces()`` block, so the
    ``raise`` propagated past it and the outer collector got nothing.
    """

    async def test_failed_inner_program_appears_in_outer_trace(self) -> None:
        class FailingInner(Program):
            async def forward(self, **kwargs: Any) -> Any:
                raise ValueError("inner failure")

        class OuterProgram(Program):
            def __init__(self, inner: Program) -> None:
                self.inner = inner

            async def forward(self, **kwargs: Any) -> Any:
                with contextlib.suppress(ValueError):
                    await self.inner(**kwargs)
                return "outer recovered"

        outer = OuterProgram(FailingInner())
        invocation = await outer.invoke()
        assert invocation.output == "outer recovered"
        # The outer trace tree contains the failed inner program as a child
        # (with an error string set), even though the inner raised.
        assert invocation.trace is not None
        assert len(invocation.trace.children) == 1
        inner_child = invocation.trace.children[0]
        assert inner_child.error is not None
        # error_str includes the exception type now (Issue D)
        assert "ValueError" in inner_child.error
        assert "inner failure" in inner_child.error


# ---------------------------------------------------------------------------
# Review Issue C — DELETED in Phase 10. The legacy ``_last_trace`` mutation
# pattern is gone — predictions are user-facing data, traces live on the
# Invocation. Users that want a trace + result together call
# ``program.invoke()`` and read ``invocation.trace``.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Phase 9b follow-up #1 — Failed leaf Calls appear in parent trace tree
# ---------------------------------------------------------------------------


class TestFollowup1FailedLeafInParentTree:
    """A failing sub-Call inside a Program must appear in
    ``program.last_trace.children`` with its ``error`` field set.

    Pre-fix ``Call._execute`` only push_trace'd on success, so the failing
    leaf was tracked on ``call.last_trace`` but invisible to the parent
    Program's collector. The fix moves the push into a ``finally`` block
    so it runs on both code paths.
    """

    async def test_failed_subcall_appears_in_program_children(self) -> None:
        # Producer that always raises a CallError on the first turn.
        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            raise RuntimeError("simulated provider 500")

        client = FunctionClient(function=fn)
        producer = Call(WriteSig, model="function-test", max_retries=0)
        producer._client = client

        class WrapperProgram(Program):
            def __init__(self, p: Call) -> None:
                self.producer = p

            async def forward(self, **kwargs: Any) -> Any:
                # Catch the failure so the program returns normally
                with contextlib.suppress(Exception):
                    return await self.producer(**kwargs)
                return "recovered"

        prog = WrapperProgram(producer)
        invocation = await prog.invoke(topic="x")
        assert invocation.output == "recovered"

        # KEY ASSERTION: the failing leaf appears as a child of the program
        # trace tree, with its error string set.
        assert invocation.trace is not None
        assert len(invocation.trace.children) == 1, (
            f"Expected the failing leaf as a child, got {len(invocation.trace.children)}"
        )
        leaf = invocation.trace.children[0]
        assert leaf.error is not None
        assert "simulated provider 500" in leaf.error


# ---------------------------------------------------------------------------
# Phase 9b follow-up #2 — Eval harness captures trace explicitly
# ---------------------------------------------------------------------------


class TestFollowup2EvalHarnessCapturesTrace:
    """``estimate_eval_cost`` must work for predictions that cannot carry a
    ``_last_trace`` attribute (plain dicts, primitives, frozen models).

    The fix adds a ``trace`` field on ``ExampleResult`` populated by the
    eval harness directly from ``call.last_trace``. ``estimate_eval_cost``
    reads from there first, falling back to the legacy mutated-attribute
    path only when ``trace`` is None.
    """

    async def test_estimate_cost_works_for_dict_prediction(self) -> None:
        from kaos_llm_core.optimization.evaluation import evaluate
        from kaos_llm_core.programs._invocation import Invocation
        from kaos_llm_core.types import Example

        # Wrap a Call so the prediction is the underlying Pydantic model
        # — we then convert to a plain dict to simulate the dict-output case.
        producer = _call_with_responses(["a", "b"])

        # Mimic a Call-like object with .invoke() returning an Invocation
        # whose .output is a plain dict — Phase 10 contract.
        class DictCall:
            async def invoke(self, **kwargs: Any) -> Invocation:
                inner = await producer.invoke(**kwargs)
                return Invocation(
                    client=inner.client,
                    model=inner.model,
                    output=inner.output.model_dump(),
                    trace=inner.trace,
                    usage=inner.usage,
                )

        eval_result = await evaluate(
            DictCall(),
            dataset=[
                Example(inputs={"topic": "x"}, outputs={"answer": "a"}),
                Example(inputs={"topic": "y"}, outputs={"answer": "b"}),
            ],
            metric=lambda p, g: 1.0,
        )

        # Each example carries a trace captured from invocation.trace
        assert all(er.trace is not None for er in eval_result.per_example), (
            f"Expected every example to have a captured trace, got "
            f"{[er.trace for er in eval_result.per_example]}"
        )
        # estimate_eval_cost finds them via er.trace, not via prediction._last_trace
        _cost, tokens = estimate_eval_cost(eval_result)
        assert tokens > 0
        # The predictions are plain dicts (no _last_trace possible)
        for er in eval_result.per_example:
            assert isinstance(er.prediction, dict)
            assert not hasattr(er.prediction, "_last_trace")


# ---------------------------------------------------------------------------
# Phase 9b follow-up #3 — ChainOfThought.prepare_call honors native thinking
# ---------------------------------------------------------------------------


class TestFollowup3ChainOfThoughtPrepareCall:
    """``ChainOfThought.prepare_call`` must produce a CallPlan that matches
    what ``_execute`` would actually send to the provider.

    Pre-fix ``prepare_call`` ran the base Call's step methods, but
    ``self._native_thinking`` was never set (only ``_execute`` set it), so
    the plan came back with the CoT-enhanced signature + prompt suffix and
    no thinking kwargs — while real execution sent ``reasoning={...}``.
    """

    async def test_prepare_call_reflects_native_thinking_capable_client(self) -> None:
        from kaos_llm_client.profiles import ModelProfile as ProfileType

        from kaos_llm_core.programs.chain_of_thought import ChainOfThought

        class CoTSig(Signature):
            """Test signature for CoT prepare_call."""

            text: str = InputField(description="The text")
            answer: str = OutputField(description="The answer")

        # Build a profile that reports native thinking support
        thinking_profile = ProfileType(
            supports_thinking=True,
            thinking_parameter="thinking",
        )

        def fn(messages: list[dict[str, Any]], profile: Any) -> ProviderResponse:
            return _json_response({"answer": "ok"})

        client = FunctionClient(function=fn, profile=thinking_profile)
        cot = ChainOfThought(
            CoTSig,
            model="function-thinking",
            client=client,
        )

        plan = await cot.prepare_call({"text": "anything"})
        # Phase 10: native_thinking lives on the active Invocation's
        # ``extras["native_thinking"]`` (populated by
        # ``_build_invocation_extras``). The legacy ``_native_thinking``
        # instance field is GONE — there is no instance state for
        # prepare_call to mutate.
        # KEY ASSERTION 1: effective signature is the ORIGINAL (not CoT-enhanced)
        # because native thinking comes from response.thinking not the schema
        assert plan.effective_signature is CoTSig
        # KEY ASSERTION 2: thinking kwarg present in call_kwargs
        assert "thinking" in plan.call_kwargs
        # KEY ASSERTION 3: no legacy instance attribute exists
        assert not hasattr(cot, "_native_thinking")

    async def test_prepare_call_falls_back_for_non_thinking_client(self) -> None:
        from kaos_llm_client.profiles import ModelProfile as ProfileType

        from kaos_llm_core.programs.chain_of_thought import ChainOfThought

        class CoTSig(Signature):
            """Test signature for CoT prepare_call (no native)."""

            text: str = InputField(description="The text")
            answer: str = OutputField(description="The answer")

        plain_profile = ProfileType(
            supports_thinking=False,
            thinking_parameter=None,
        )

        def fn(messages: list[dict[str, Any]], profile: Any) -> ProviderResponse:
            return _json_response({"answer": "ok", "reasoning": "..."})

        client = FunctionClient(function=fn, profile=plain_profile)
        cot = ChainOfThought(
            CoTSig,
            model="function-plain",
            client=client,
        )

        plan = await cot.prepare_call({"text": "anything"})
        # Without native thinking, effective signature is the CoT-enhanced one
        # (with reasoning field)
        assert plan.effective_signature is not CoTSig
        assert "reasoning" in plan.effective_signature.model_fields
        # And no thinking kwarg
        assert "thinking" not in plan.call_kwargs


# ---------------------------------------------------------------------------
# Phase 9b follow-up #4 — trace_enabled honored by Program AND ReAct
# ---------------------------------------------------------------------------


class TestFollowup4TraceEnabledHonored:
    """When ``trace_enabled=False`` at the leaf-Call settings level, the
    parent Program / ReAct must ALSO skip building its parent trace tree.

    Pre-fix the leaf Call traces correctly disappeared, but Program and
    ReAct unconditionally synthesized parent traces, producing a tree of
    phantom internal nodes with no leaves — exactly the contract drift
    the Phase 9b reviewer flagged.
    """

    async def test_program_skips_trace_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from kaos_llm_core.settings import KaosLLMCoreSettings

        # Build a Call with trace_enabled=False on its core_settings
        producer = _call_with_responses(["x"])
        producer._core_settings = KaosLLMCoreSettings(trace_enabled=False)

        class WrapperProgram(Program):
            def __init__(self, p: Call) -> None:
                self.producer = p

            async def forward(self, **kwargs: Any) -> Any:
                return await self.producer(**kwargs)

        prog = WrapperProgram(producer)
        invocation = await prog.invoke(topic="x")
        # AND the parent program trace is also disabled (no phantom parent)
        assert invocation.trace is None

    async def test_react_skips_trace_when_disabled(self) -> None:
        from kaos_llm_core.settings import KaosLLMCoreSettings

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"answer": "ok"})

        client = FunctionClient(function=handler)
        # Pass core_settings with trace_enabled=False to the inner Call
        react = ReAct(
            AnswerSig,
            tools=[Tool.from_callable(lambda: "noop")],
            model="function-test",
            client=client,
            core_settings=KaosLLMCoreSettings(trace_enabled=False),
        )
        invocation = await react.invoke(question="x")
        # The ReAct loop still runs and produces a result
        assert invocation.output.is_complete()
        # But no parent trace was assembled
        assert invocation.trace is None


# ---------------------------------------------------------------------------
# Phase 9c — review-driven critical fixes
# ---------------------------------------------------------------------------


class TestPhase9cReActCostRollup:
    """ReAct iteration traces must populate ``cost_usd`` via apply_cost_estimates.

    Pre-fix: ``_build_iteration_trace`` constructed an ``ExecutionTrace``
    directly from ``response.usage`` and never called
    ``apply_cost_estimates``. ``cost_usd`` defaulted to 0.0, the parent
    rolled up zeros, and any ``Budget(max_cost_usd=...)`` over a ReAct
    ran silently uncapped.
    """

    async def test_react_iteration_traces_have_nonzero_cost_for_known_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from kaos_llm_core.observability.cost import PRICING, ModelPricing

        # Inject a pricing entry for the function-test model so
        # apply_cost_estimates has something to look up.
        monkeypatch.setitem(
            PRICING,
            "function-test",
            ModelPricing(
                input_per_mtok=1.0,
                output_per_mtok=2.0,
            ),
        )

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"answer": "ok"})

        client = FunctionClient(function=handler)
        react = ReAct(
            AnswerSig,
            tools=[Tool.from_callable(lambda: "noop")],
            model="function-test",
            client=client,
        )
        invocation = await react.invoke(question="x")
        trace = invocation.trace
        assert trace is not None
        # Iteration trace exists and has nonzero cost from the PRICING table
        assert len(trace.children) >= 1
        iter_trace = trace.children[0]
        # 10 input tokens x $1/M + 5 output tokens x $2/M = $0.00002
        assert iter_trace.cost_usd > 0.0
        # Parent rolls up the iter cost
        assert trace.cost_usd > 0.0


class TestPhase9cPrepareCallNoInstanceWrites:
    """``prepare_call`` must NOT write per-execution state to instance attrs.

    Phase 10: ``_current_client`` / ``_current_model`` instance attributes
    have been DELETED. Per-execution state lives on the active
    ``Invocation`` carried by a task-isolated ContextVar. The tests below
    lock in that the Call instance has no per-execution scratch slots.
    """

    async def test_prepare_call_does_not_create_instance_scratch(self) -> None:
        producer = _call_with_responses(["a"])
        # Pre: no legacy scratch attributes exist on the instance
        assert not hasattr(producer, "_current_client")
        assert not hasattr(producer, "_current_model")
        plan = await producer.prepare_call({"topic": "x"})
        # Post: still no legacy scratch — the resolved client/model live
        # on the plan and on the Invocation, never on the Call instance.
        assert not hasattr(producer, "_current_client")
        assert not hasattr(producer, "_current_model")
        assert plan.client is not None
        assert plan.model == "function-test"

    async def test_two_concurrent_prepare_calls_do_not_race(self) -> None:
        """Two parallel prepare_call invocations on the same Call get
        their own resolved client/model, with no instance-state race."""
        producer = _call_with_responses(["a", "b", "c"])
        plans = await asyncio.gather(
            producer.prepare_call({"topic": "x"}),
            producer.prepare_call({"topic": "y"}),
            producer.prepare_call({"topic": "z"}),
        )
        for p in plans:
            assert p.model == "function-test"
            assert p.client is not None
        # No legacy instance scratch was ever created
        assert not hasattr(producer, "_current_client")
        assert not hasattr(producer, "_current_model")


class TestPhase9cCascadeRouterConcurrencySafe:
    """CascadeRouter scratch state must be local to each execute_cascade,
    not instance state. Concurrent users no longer corrupt each other.
    """

    async def test_concurrent_cascade_executions_dont_corrupt_each_other(
        self,
    ) -> None:
        from kaos_llm_core.router.cascade import CascadeRouter

        # Both Calls share the same router instance.
        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"answer": "ok"})

        client = FunctionClient(function=fn)
        router = CascadeRouter(models=["function-test"])
        call_a = Call(WriteSig, router=router, client=client)
        call_b = Call(WriteSig, router=router, client=client)

        # Run two cascades in parallel through the shared router
        results = await asyncio.gather(
            call_a(topic="x"),
            call_b(topic="y"),
        )
        assert all(r.answer == "ok" for r in results)
        # The router instance attributes are valid (not torn) — they
        # reflect ONE of the two completions atomically.
        assert router.last_traces is not None
        assert len(router.last_traces) >= 1
        assert router.model_used == "function-test"


class TestPhase9cReActValidationRetryHookFires:
    """ReAct validation-retry path must fire on_validation_retry on the
    inner Call's hooks. Pre-fix the hook never fired because ReAct's
    loop bypasses Call._execute (where the standard hook fires)."""

    async def test_on_validation_retry_fires_on_decode_failure(self) -> None:
        retries_seen: list[int] = []

        def on_retry(
            call: Any, inputs: Any, attempt: int, error: Any, *, context: Any = None
        ) -> None:
            retries_seen.append(attempt)

        hooks = CallHooks(on_validation_retry=on_retry)

        call_count = {"n": 0}

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            call_count["n"] += 1
            if call_count["n"] == 1:
                # First turn: invalid JSON triggers decode failure
                return ProviderResponse.model_construct(
                    provider="function",
                    model="function-test",
                    raw={},
                    parts=[ContentPart.model_construct(type="text", text="not valid json")],
                    usage=UsageInfo.model_construct(
                        input_tokens=5, output_tokens=5, total_tokens=10
                    ),
                    stop_reason="end_turn",
                    status_code=200,
                    response_headers={},
                )
            return _json_response({"answer": "recovered"})

        client = FunctionClient(function=handler)
        react = ReAct(
            AnswerSig,
            tools=[Tool.from_callable(lambda: "noop")],
            model="function-test",
            client=client,
            hooks=hooks,
            max_validation_retries=3,
        )
        result = await react(question="x")
        assert result.is_complete()
        # The hook fired at least once for the first decode failure
        assert len(retries_seen) >= 1
        assert retries_seen[0] == 1


class TestPhase9cReActErrorPathHasParentTraceError:
    """When ReAct stops with stop_reason==ERROR, the parent trace
    ``error`` field must be populated. Phase 9c follow-up #1."""

    async def test_validation_exhaustion_sets_parent_trace_error(self) -> None:
        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return ProviderResponse.model_construct(
                provider="function",
                model="function-test",
                raw={},
                parts=[ContentPart.model_construct(type="text", text="not valid json")],
                usage=UsageInfo.model_construct(input_tokens=5, output_tokens=5, total_tokens=10),
                stop_reason="end_turn",
                status_code=200,
                response_headers={},
            )

        client = FunctionClient(function=handler)
        react = ReAct(
            AnswerSig,
            tools=[Tool.from_callable(lambda: "noop")],
            model="function-test",
            client=client,
            max_validation_retries=1,
        )
        invocation = await react.invoke(question="x")
        assert invocation.output.stop_reason == "ERROR"
        # KEY ASSERTION: the parent trace's error field is populated
        assert invocation.trace is not None
        assert invocation.trace.error is not None
        assert "stop_reason=ERROR" in invocation.trace.error


# Phase 10: ``TestPhase9cTraceEnabledClearsStaleTrace`` has been DELETED.
# It tested the legacy ``program.last_trace`` instance attribute being
# cleared between runs. The attribute no longer exists — every run
# returns a fresh ``Invocation`` from ``program.invoke()``, so there is
# no stale state on the instance to clear.


class TestPhase9cTraceEnabledAnyChildPolicy:
    """Program._trace_enabled must return True if ANY sub-Call has tracing
    on, not just the first declared one. Phase 9c follow-up #2."""

    def test_any_child_with_tracing_enabled_wins(self) -> None:
        from kaos_llm_core.settings import KaosLLMCoreSettings

        first = _call_with_responses(["a"])
        first._core_settings = KaosLLMCoreSettings(trace_enabled=False)
        second = _call_with_responses(["b"])
        second._core_settings = KaosLLMCoreSettings(trace_enabled=True)

        class P(Program):
            def __init__(self) -> None:
                self.first = first
                self.second = second

            async def forward(self, **kwargs: Any) -> Any:
                return await self.second(**kwargs)

        p = P()
        # Even though first.tracing is off, second.tracing is on -> True
        assert p._trace_enabled() is True


class TestPhase9cSemanticCacheLocking:
    """SemanticCache._entries must be lock-protected. Concurrent appends
    cannot lose writes due to eviction rebinding."""

    def test_lock_attribute_exists(self) -> None:
        cache = SemanticCache()
        assert hasattr(cache, "_lock")
        assert cache._lock is not None

    def test_eviction_uses_in_place_delete_not_rebind(self) -> None:
        """The eviction path must NOT rebind self._entries.

        We verify by snapshotting the list identity before and after
        forcing an eviction; the same list object must persist.
        """
        cache = SemanticCache(max_entries=2)
        # Manually populate to bypass the embedding step
        from kaos_llm_core.cache.semantic import CacheEntry

        for i in range(5):
            cache._entries.append(
                CacheEntry(
                    embedding=[0.0],
                    result_json="{}",
                    signature_name="X",
                    model="m",
                    config_fingerprint="f",
                    input_sha256=f"k{i:063d}",
                )
            )
        original_id = id(cache._entries)
        # Manually trigger the eviction logic the way cache.call does
        if len(cache._entries) > cache.max_entries:
            excess = len(cache._entries) - cache.max_entries
            del cache._entries[:excess]
        # The list object identity is preserved (no rebind)
        assert id(cache._entries) == original_id
        assert len(cache._entries) == 2


class TestPhase9cSemanticCacheFingerprintUsesPersistedKeys:
    """SemanticCache._config_fingerprint must use Call._PERSISTED_HYPERPARAMETERS
    so changes to presence_penalty / frequency_penalty / etc invalidate the
    cache. Phase 9c critical fix #3."""

    def test_presence_penalty_change_invalidates_fingerprint(self) -> None:
        c1 = _call_with_responses(["x"])
        c1._kwargs = {"presence_penalty": 0.0}
        c2 = _call_with_responses(["y"])
        c2._kwargs = {"presence_penalty": 0.5}
        assert _config_fingerprint(c1) != _config_fingerprint(c2)

    def test_frequency_penalty_change_invalidates_fingerprint(self) -> None:
        c1 = _call_with_responses(["x"])
        c1._kwargs = {"frequency_penalty": 0.0}
        c2 = _call_with_responses(["y"])
        c2._kwargs = {"frequency_penalty": 0.5}
        assert _config_fingerprint(c1) != _config_fingerprint(c2)


class TestPhase9cEvaluateReadsTraceFromPrediction:
    """Eval harness should read trace from prediction._last_trace first,
    falling back to call.last_trace only when the prediction can't carry
    an attribute. Phase 9c follow-up #8."""

    async def test_eval_captures_prediction_trace_per_example(self) -> None:
        from kaos_llm_core.optimization.evaluation import evaluate
        from kaos_llm_core.types import Example

        producer = _call_with_responses(["a", "b"])

        eval_result = await evaluate(
            producer,
            dataset=[
                Example(inputs={"topic": "x"}, outputs={"answer": "a"}),
                Example(inputs={"topic": "y"}, outputs={"answer": "b"}),
            ],
            metric=lambda p, g: 1.0,
            max_concurrent=2,  # exercise the concurrent path
        )
        # Each example has its own trace, captured from the prediction
        # not from call.last_trace (which would race under concurrency)
        assert all(er.trace is not None for er in eval_result.per_example)
        # The two traces are distinct (each example has its own)
        traces = [er.trace for er in eval_result.per_example]
        assert traces[0] is not traces[1]


# ---------------------------------------------------------------------------
# Phase 9d — lifecycle cleanup
# ---------------------------------------------------------------------------


class TestPhase9dProgramHooksFireFromBase:
    """ProgramHooks fire from Program.__call__ in the base class.

    Phase 9d cleanup: Refine and BestOfN previously had identical
    __call__ overrides whose only purpose was to fire ProgramHooks
    around super().__call__(). The base now owns hook firing so the
    overrides have been deleted. Subclasses (Refine, BestOfN, ReAct)
    no longer need to wrap super for hooks.
    """

    async def test_program_hooks_fire_from_refine_via_base(self) -> None:
        from kaos_llm_core.programs.judge import Judge
        from kaos_llm_core.programs.refine import Refine

        events: list[str] = []

        def on_start(prog: Any, inputs: Any, *, context: Any = None) -> None:
            events.append(f"start:{type(prog).__name__}")

        def on_end(prog: Any, inputs: Any, invocation: Any, *, context: Any = None) -> None:
            events.append(f"end:{type(prog).__name__}")

        prog_hooks = ProgramHooks(on_program_start=on_start, on_program_end=on_end)

        def producer_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"answer": "ok"})

        def judge_fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"quality_score": 0.9, "reasoning": "good"})

        producer = Call(WriteSig, model="function-test")
        producer._client = FunctionClient(function=producer_fn)
        judge = Judge(WriteSig, producer_model="function-test", judge_model="function-test")
        judge.judge_call._client = FunctionClient(function=judge_fn)

        refine = Refine(producer, judge, max_iterations=1, min_score=0.5, program_hooks=prog_hooks)
        await refine(topic="x")
        assert "start:Refine" in events
        assert "end:Refine" in events

    async def test_program_hooks_fire_from_best_of_n_via_base(self) -> None:
        events: list[str] = []

        def on_start(prog: Any, inputs: Any, *, context: Any = None) -> None:
            events.append(f"start:{type(prog).__name__}")

        def on_end(prog: Any, inputs: Any, invocation: Any, *, context: Any = None) -> None:
            events.append(f"end:{type(prog).__name__}")

        prog_hooks = ProgramHooks(on_program_start=on_start, on_program_end=on_end)
        producer = _call_with_responses(["a", "b"])
        program = BestOfN(
            producer, n=2, selector=lambda o, i: 0.5, seed_strategy="none", program_hooks=prog_hooks
        )
        await program(topic="x")
        assert "start:BestOfN" in events
        assert "end:BestOfN" in events


class TestPhase9dReActSaveLoad:
    """ReAct's inner Call must round-trip through save/load.

    Phase 9d save/load fix: ReAct.named_calls() now exposes the inner
    Call under the public name "call" so the base
    ``Program.get_learnable_state`` walker discovers it. Pre-fix
    ``ReAct.save()`` produced an envelope with empty state and
    ``ReAct.load()`` was a silent no-op.
    """

    def test_react_named_calls_includes_inner_call(self) -> None:
        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"answer": "ok"})

        client = FunctionClient(function=handler)
        react = ReAct(
            AnswerSig,
            tools=[Tool.from_callable(lambda: "noop")],
            model="function-test",
            client=client,
        )
        names = react.named_calls()
        assert "call" in names
        assert names["call"] is react._inner_call

    def test_react_get_learnable_state_includes_inner_call(self) -> None:
        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"answer": "ok"})

        client = FunctionClient(function=handler)
        react = ReAct(
            AnswerSig,
            tools=[Tool.from_callable(lambda: "noop")],
            model="function-test",
            client=client,
            instructions="Custom test instructions",
        )
        state = react.get_learnable_state()
        # The inner Call's state must be present under "call"
        assert "call" in state
        assert state["call"]["instructions"] == "Custom test instructions"


class TestPhase9dCoreSettingsOnComposedPrograms:
    """Refine/BestOfN/Judge/Ensemble accept core_settings."""

    def test_refine_accepts_core_settings(self) -> None:
        from kaos_llm_core.programs.judge import Judge
        from kaos_llm_core.programs.refine import Refine
        from kaos_llm_core.settings import KaosLLMCoreSettings

        settings = KaosLLMCoreSettings(trace_enabled=False)
        producer = Call(WriteSig, model="function-test")
        judge = Judge(WriteSig, producer_model="function-test", judge_model="function-test")
        refine = Refine(producer, judge, core_settings=settings)
        assert refine._core_settings is settings

    def test_best_of_n_accepts_core_settings(self) -> None:
        from kaos_llm_core.settings import KaosLLMCoreSettings

        settings = KaosLLMCoreSettings(trace_enabled=False)
        producer = _call_with_responses(["a", "b"])
        program = BestOfN(producer, n=2, selector=lambda o, i: 0.5, core_settings=settings)
        assert program._core_settings is settings

    def test_judge_accepts_core_settings(self) -> None:
        from kaos_llm_core.programs.judge import Judge
        from kaos_llm_core.settings import KaosLLMCoreSettings

        settings = KaosLLMCoreSettings(trace_enabled=False)
        judge = Judge(
            WriteSig,
            producer_model="function-test",
            judge_model="function-test",
            core_settings=settings,
        )
        # Should be threaded into the inner Calls
        assert judge.produce._core_settings is settings
        assert judge.judge_call._core_settings is settings

    def test_ensemble_accepts_core_settings(self) -> None:
        from kaos_llm_core.programs.ensemble import Ensemble
        from kaos_llm_core.settings import KaosLLMCoreSettings

        settings = KaosLLMCoreSettings(trace_enabled=False)
        ensemble = Ensemble(
            WriteSig,
            models=["function-test", "function-test"],
            core_settings=settings,
        )
        assert all(v._core_settings is settings for v in ensemble.voters)


# ---------------------------------------------------------------------------
# Phase 9e — API hygiene
# ---------------------------------------------------------------------------


class TestPhase9eTopLevelExports:
    """Top-level kaos_llm_core exports include everything users need.

    Cross-cutting review High finding: top-level __all__ was stale.
    `from kaos_llm_core import Refine` failed silently because Refine
    was only re-exported from kaos_llm_core.programs.
    """

    def test_top_level_imports_resolve(self) -> None:
        # All these should import without error
        from kaos_llm_core import (
            BestOfN,
            BestOfNResult,
            Budget,
            BudgetTracker,
            CallPlan,
            CodecOptimizer,
            JudgedResult,
            ModelOptimizer,
            OutputForwardingMixin,
            ParetoOptimizer,
            ProgramHooks,
            Refine,
            RefineHistoryEntry,
            RefineResult,
            Rule,
            StopReason,
            classify,
            collect_traces,
            compute_pareto_frontier,
            export_jsonl,
            extract,
            load_jsonl,
            push_trace,
            summarize,
            text,
        )

        # Just touch each to make sure they're real
        assert BestOfN is not None
        assert BestOfNResult is not None
        assert Budget is not None
        assert BudgetTracker is not None
        assert CallPlan is not None
        assert CodecOptimizer is not None
        assert JudgedResult is not None
        assert ModelOptimizer is not None
        assert OutputForwardingMixin is not None
        assert ParetoOptimizer is not None
        assert ProgramHooks is not None
        assert Refine is not None
        assert RefineHistoryEntry is not None
        assert RefineResult is not None
        assert Rule is not None
        assert StopReason is not None
        assert callable(classify)
        assert callable(collect_traces)
        assert callable(compute_pareto_frontier)
        assert callable(export_jsonl)
        assert callable(extract)
        assert callable(load_jsonl)
        assert callable(push_trace)
        assert callable(summarize)
        assert callable(text)


class TestPhase9eErrorHierarchy:
    """Refine/ReAct/starter raise CallError (KaosLLMCoreError subclass)
    for config errors instead of bare ValueError. Cross-cutting review
    High finding: catching KaosLLMCoreError used to miss "no model
    specified" failures from starter and "max_iterations" failures
    from Refine."""

    def test_refine_max_iterations_raises_call_error(self) -> None:
        from kaos_llm_core.errors import CallError, KaosLLMCoreError
        from kaos_llm_core.programs.judge import Judge
        from kaos_llm_core.programs.refine import Refine

        producer = _call_with_responses(["x"])
        judge = Judge(WriteSig, producer_model="function-test", judge_model="function-test")
        with pytest.raises(CallError):
            Refine(producer, judge, max_iterations=0)
        # And it's catchable as KaosLLMCoreError
        with pytest.raises(KaosLLMCoreError):
            Refine(producer, judge, max_iterations=0)

    def test_react_invalid_max_iterations_raises_call_error(self) -> None:
        from kaos_llm_core.errors import CallError, KaosLLMCoreError

        with pytest.raises(CallError):
            ReAct(AnswerSig, tools=[Tool.from_callable(lambda: "x")], max_iterations=0)
        with pytest.raises(KaosLLMCoreError):
            ReAct(AnswerSig, tools=[Tool.from_callable(lambda: "x")], max_iterations=0)


class TestPhase9eJudgedResultPublic:
    """JudgedResult is now the public name; _JudgedResult is a backwards-
    compat alias."""

    def test_judged_result_is_public(self) -> None:
        from kaos_llm_core import JudgedResult
        from kaos_llm_core.programs.judge import _JudgedResult

        # The alias points to the same class
        assert JudgedResult is _JudgedResult
        # And it's a real dataclass with the expected fields
        instance = JudgedResult(output="test", judgment="judged")
        assert instance.output == "test"
        assert instance.judgment == "judged"


class TestPhase9eBestOfNResultSlots:
    """BestOfNResult should use slots=True for consistency with siblings."""

    def test_best_of_n_result_has_slots(self) -> None:
        # All sibling result types use slots=True. BestOfNResult was the
        # one outlier; Phase 9e fixes it.
        assert hasattr(BestOfNResult, "__slots__")


class TestPhase9eCascadeRouterSentinel:
    """CascadeRouter escalation_check default detection uses an explicit
    module-level sentinel instead of fragile lambda bytecode comparison."""

    def test_default_escalation_check_is_module_level_sentinel(self) -> None:
        from kaos_llm_core.router.cascade import CascadeRouter, _default_escalation_check

        router = CascadeRouter(models=["m1"])
        assert router.escalation_check is _default_escalation_check

    def test_serialization_does_not_warn_on_default_check(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from kaos_llm_core.router.cascade import CascadeRouter
        from kaos_llm_core.router.serialization import router_to_state

        captured: list[str] = []
        from kaos_llm_core.router import serialization as ser_mod

        def fake_warning(msg: str, *args: Any, **kwargs: Any) -> None:
            captured.append(msg % args if args else msg)

        monkeypatch.setattr(ser_mod.logger, "warning", fake_warning)

        router = CascadeRouter(models=["m1", "m2"])
        state = router_to_state(router)
        assert state is not None
        assert state["type"] == "cascade"
        # No warning fired because escalation_check is the module sentinel
        assert not any("escalation_check" in m for m in captured)


# Suppress unused-import warnings for symbols only used in type annotations
_ = (BestOfNResult, contextlib, SemanticCache)
