"""Unit tests for the Phase 12A LoopRunner + clone_call utilities.

LoopRunner is exercised against a deterministic mock step function so
the iteration / stop-reason / error semantics are isolated from the
real ReAct/Refine programs (which migrate onto the runner in Phases
12B and 12C). clone_call is exercised against a real Call instance
to verify the deepcopy + shallow-fallback path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest
from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart

from kaos_llm_core.programs.call import Call
from kaos_llm_core.programs.cloning import clone_call
from kaos_llm_core.programs.loop_runner import (
    LoopConfig,
    LoopRunner,
    StepOutcome,
    StopReason,
)
from kaos_llm_core.signatures import InputField, OutputField, Signature
from kaos_llm_core.types import Example

# ---------------------------------------------------------------------------
# LoopConfig
# ---------------------------------------------------------------------------


class TestLoopConfig:
    def test_max_iterations_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="max_iterations must be >= 1"):
            LoopConfig(max_iterations=0)

    def test_max_iterations_one_is_valid(self) -> None:
        cfg = LoopConfig(max_iterations=1)
        assert cfg.max_iterations == 1
        assert cfg.propagate_errors is True


# ---------------------------------------------------------------------------
# LoopRunner — completion paths
# ---------------------------------------------------------------------------


@dataclass
class _CountState:
    """Tiny state used by the test step functions."""

    seen: list[int] = field(default_factory=list)


class TestLoopRunnerCompletion:
    async def test_runs_until_max_iterations(self) -> None:
        """No early stop — the loop runs exactly max_iterations times."""

        async def step(i: int, state: _CountState) -> StepOutcome[int]:
            state.seen.append(i)
            return StepOutcome(record=i * 10)

        runner: LoopRunner[_CountState, int, list[int]] = LoopRunner(
            config=LoopConfig(max_iterations=4),
            make_state=_CountState,
            step=step,
            build_result=lambda state, records: state.seen,
        )
        out = await runner.run()
        assert out.iterations_used == 4
        assert out.records == [0, 10, 20, 30]
        assert out.result == [0, 1, 2, 3]
        assert out.stop_reason == StopReason.COMPLETED.value
        assert out.error is None

    async def test_step_returns_early_stop(self) -> None:
        """A step that returns ``stop_reason`` halts the loop after recording."""

        async def step(i: int, state: _CountState) -> StepOutcome[int]:
            state.seen.append(i)
            if i == 2:
                return StepOutcome(record=999, stop_reason="found_answer")
            return StepOutcome(record=i)

        runner: LoopRunner[_CountState, int, int] = LoopRunner(
            config=LoopConfig(max_iterations=10),
            make_state=_CountState,
            step=step,
            build_result=lambda state, records: len(records),
        )
        out = await runner.run()
        # Three iterations: 0, 1, 2 (the last one short-circuits)
        assert out.iterations_used == 3
        assert out.records == [0, 1, 999]
        assert out.result == 3
        assert out.stop_reason == "found_answer"

    async def test_make_state_called_once_per_run(self) -> None:
        """``make_state`` is invoked exactly once at the top of ``run``."""
        calls = {"n": 0}

        def make() -> _CountState:
            calls["n"] += 1
            return _CountState()

        async def step(i: int, state: _CountState) -> StepOutcome[int]:
            return StepOutcome(record=i, stop_reason="done")

        runner: LoopRunner[_CountState, int, None] = LoopRunner(
            config=LoopConfig(max_iterations=5),
            make_state=make,
            step=step,
            build_result=lambda state, records: None,
        )
        await runner.run()
        assert calls["n"] == 1


# ---------------------------------------------------------------------------
# LoopRunner — error paths
# ---------------------------------------------------------------------------


class TestLoopRunnerErrors:
    async def test_step_exception_propagates_by_default(self) -> None:
        """``propagate_errors=True`` re-raises uncaught step exceptions."""

        async def step(i: int, state: _CountState) -> StepOutcome[int]:
            raise RuntimeError("boom")

        runner: LoopRunner[_CountState, int, None] = LoopRunner(
            config=LoopConfig(max_iterations=3),
            make_state=_CountState,
            step=step,
            build_result=lambda state, records: None,
        )
        with pytest.raises(RuntimeError, match="boom"):
            await runner.run()

    async def test_step_exception_with_propagate_false_returns_partial(self) -> None:
        """``propagate_errors=False`` lets the loop return a partial
        result with ``stop_reason=ERROR`` and the exception attached."""

        async def step(i: int, state: _CountState) -> StepOutcome[int]:
            state.seen.append(i)
            if i == 1:
                raise RuntimeError("simulated step failure")
            return StepOutcome(record=i)

        runner: LoopRunner[_CountState, int, list[int]] = LoopRunner(
            config=LoopConfig(max_iterations=10, propagate_errors=False),
            make_state=_CountState,
            step=step,
            build_result=lambda state, records: state.seen,
        )
        out = await runner.run()
        assert out.iterations_used == 2
        assert out.records == [0]  # iter 0 succeeded; iter 1 raised before append
        assert out.stop_reason == StopReason.ERROR.value
        assert out.error is not None
        assert isinstance(out.error, RuntimeError)
        assert "simulated step failure" in str(out.error)

    async def test_on_step_error_can_convert_to_clean_stop(self) -> None:
        """``on_step_error`` returning a string halts cleanly with that
        stop reason instead of letting the exception escape."""

        async def step(i: int, state: _CountState) -> StepOutcome[int]:
            if i == 1:
                raise RuntimeError("recoverable")
            state.seen.append(i)
            return StepOutcome(record=i)

        def on_error(state: _CountState, exc: BaseException) -> str | None:
            assert isinstance(exc, RuntimeError)
            return "recovered_via_handler"

        runner: LoopRunner[_CountState, int, list[int]] = LoopRunner(
            config=LoopConfig(max_iterations=5),
            make_state=_CountState,
            step=step,
            build_result=lambda state, records: state.seen,
            on_step_error=on_error,
        )
        out = await runner.run()
        assert out.stop_reason == "recovered_via_handler"
        # Iter 0 succeeded, iter 1 raised — handler intercepted it.
        assert out.records == [0]
        assert out.error is not None


# ---------------------------------------------------------------------------
# clone_call
# ---------------------------------------------------------------------------


class _S(Signature):
    """Trivial signature for clone_call tests."""

    topic: str = InputField(description="What to write about")
    answer: str = OutputField(description="The answer")


def _make_call() -> Call:
    """Build a Call wired to a deterministic FunctionClient."""

    def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
        return ProviderResponse.model_construct(
            provider="function",
            model="function-test",
            raw={},
            parts=[ContentPart.model_construct(type="text", text=json.dumps({"answer": "x"}))],
            usage=UsageInfo.model_construct(input_tokens=5, output_tokens=5, total_tokens=10),
            stop_reason="end_turn",
            status_code=200,
            response_headers={},
        )

    call = Call(_S, model="function-test")
    call._client = FunctionClient(function=fn)
    call._kwargs = {"temperature": 0.4}
    call.examples = [Example(inputs={"topic": "x"}, outputs={"answer": "x"})]
    return call


class TestCloneCall:
    def test_clone_returns_distinct_instance(self) -> None:
        original = _make_call()
        clone = clone_call(original)
        assert clone is not original

    def test_clone_examples_list_is_isolated(self) -> None:
        """Mutating the clone's examples list does NOT touch the original."""
        original = _make_call()
        clone = clone_call(original)
        clone.examples.append(Example(inputs={"topic": "y"}, outputs={"answer": "y"}))
        assert len(original.examples) == 1
        assert len(clone.examples) == 2

    def test_clone_kwargs_are_isolated(self) -> None:
        original = _make_call()
        clone = clone_call(original)
        clone._kwargs["temperature"] = 0.9
        assert original._kwargs["temperature"] == 0.4

    def test_shallow_fallback_still_isolates_examples_list(self) -> None:
        """When deepcopy fails, the shallow-copy fallback rebuilds the
        examples list so concurrent samples don't race on it."""
        original = _make_call()
        # Pin a non-pickleable resource on the call to force the deepcopy
        # path to fail. A lambda binding to a local closure is enough.
        original._unpickleable = lambda: original  # ty: ignore[unresolved-attribute]
        clone = clone_call(original)
        # Even with the fallback, the examples list is a fresh object.
        assert clone.examples is not original.examples
        # Mutating clone's examples does not affect original.
        clone.examples.append(Example(inputs={"topic": "z"}, outputs={"answer": "z"}))
        assert len(original.examples) == 1

    def test_deep_false_skips_deepcopy(self) -> None:
        """``deep=False`` goes straight to shallow copy."""
        original = _make_call()
        clone = clone_call(original, deep=False)
        assert clone is not original
        assert clone.examples is not original.examples
