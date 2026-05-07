"""Unit tests for the Phase 13A TrialRunner + OptimizerBase framework.

The runner is exercised against real :class:`Call` invocations driven
by a deterministic :class:`FunctionClient` so the cost-attribution
contract is tested end-to-end (Call → publish_invocation → Trial
accumulator). The base is exercised in isolation to verify the
``_make_run_state`` / ``_evaluate_in_trial`` / ``_consume_trial``
helpers behave as documented.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart

from kaos_llm_core.observability.cost import PRICING, ModelPricing
from kaos_llm_core.optimization.base import MetaOptimizerBase, OptimizerBase
from kaos_llm_core.optimization.budget import Budget, StopReason
from kaos_llm_core.optimization.mutations import RunContext
from kaos_llm_core.optimization.trial_runner import (
    Trial,
    TrialRunner,
    current_trial,
    publish_invocation,
)
from kaos_llm_core.programs._invocation import Invocation, TokenUsage
from kaos_llm_core.programs.call import Call
from kaos_llm_core.signatures import InputField, OutputField, Signature
from kaos_llm_core.types import Example


class _Sig(Signature):
    """Trivial signature for trial-runner tests."""

    topic: str = InputField(description="What to write about")
    answer: str = OutputField(description="The answer")


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


def _make_call() -> Call:
    """Build a Call wired to a deterministic FunctionClient."""

    def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
        return _json_response({"answer": "ok"})

    call = Call(_Sig, model="function-test")
    call._client = FunctionClient(function=fn)
    return call


@pytest.fixture(autouse=True)
def _pricing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure ``function-test`` has a known per-model price.

    Without a pricing entry the trial would accumulate cost_usd=0 and
    the budget-cap test would not exercise the cost path.
    """
    monkeypatch.setitem(
        PRICING,
        "function-test",
        ModelPricing(input_per_mtok=1.0, output_per_mtok=2.0),
    )


# ---------------------------------------------------------------------------
# publish_invocation contract
# ---------------------------------------------------------------------------


class TestPublishInvocation:
    def test_publish_outside_trial_is_noop(self) -> None:
        """``publish_invocation`` outside any trial scope is a silent no-op."""
        inv = Invocation(
            usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15, cost_usd=0.001),
        )
        # No active trial — should not raise.
        publish_invocation(inv)
        # And current_trial() returns None.
        assert current_trial() is None

    def test_publish_charges_active_trial(self) -> None:
        """Inside a trial scope, ``publish_invocation`` accumulates."""
        runner = TrialRunner()
        with runner.trial("manual_publish_test") as trial:
            inv = Invocation(
                usage=TokenUsage(
                    input_tokens=10, output_tokens=5, total_tokens=15, cost_usd=0.0042
                ),
            )
            publish_invocation(inv)
            inv2 = Invocation(
                usage=TokenUsage(
                    input_tokens=20, output_tokens=10, total_tokens=30, cost_usd=0.0084
                ),
            )
            publish_invocation(inv2)

        assert trial.n_invocations == 2
        assert trial.input_tokens == 30
        assert trial.output_tokens == 15
        assert trial.total_tokens == 45
        assert trial.cost_usd == pytest.approx(0.0126)


# ---------------------------------------------------------------------------
# TrialRunner end-to-end with real Call invocations
# ---------------------------------------------------------------------------


class TestTrialRunnerWithCall:
    async def test_trial_charges_call_invocations(self) -> None:
        """A real ``Call.invoke()`` inside a trial scope charges its usage
        to the trial accumulator via ``Call._run_pipeline``'s
        ``publish_invocation`` hook."""
        call = _make_call()
        runner = TrialRunner()
        with runner.trial("call_invoke_test") as trial:
            inv = await call.invoke(topic="x")
            assert inv.output is not None

        # 10 input + 5 output tokens, priced at $1/M and $2/M respectively
        # = 0.00001 + 0.00001 = $0.00002
        assert trial.n_invocations == 1
        assert trial.total_tokens == 15
        assert trial.cost_usd > 0.0

    async def test_concurrent_trials_are_isolated(self) -> None:
        """Two concurrent ``trial(...)`` scopes never cross-charge.

        The contextvar is task-isolated so a Call running in one task
        sees only its own task's trial.
        """
        call_a = _make_call()
        call_b = _make_call()

        async def run_trial(call: Call, name: str) -> Trial:
            runner = TrialRunner()
            with runner.trial(name) as trial:
                # Drive the Call enough times that any cross-contamination
                # would be observable.
                for _ in range(3):
                    await call.invoke(topic="x")
            return trial

        trial_a, trial_b = await asyncio.gather(
            run_trial(call_a, "trial_a"),
            run_trial(call_b, "trial_b"),
        )
        # Each trial saw exactly its own three invocations.
        assert trial_a.n_invocations == 3
        assert trial_b.n_invocations == 3
        assert trial_a.total_tokens == 45
        assert trial_b.total_tokens == 45

    async def test_trial_duration_populated_on_exit(self) -> None:
        """``trial.duration_ms`` is set when the context manager exits."""
        call = _make_call()
        runner = TrialRunner()
        with runner.trial("duration_test") as trial:
            await call.invoke(topic="x")
        assert trial.duration_ms > 0.0

    async def test_trial_outside_runner_no_charge(self) -> None:
        """A Call invocation outside any ``trial(...)`` scope does not
        accumulate anywhere — and a subsequent trial sees only its own
        invocation."""
        call = _make_call()

        # First invocation runs outside any trial.
        await call.invoke(topic="outside")

        runner = TrialRunner()
        with runner.trial("inside") as trial:
            await call.invoke(topic="inside")
        assert trial.n_invocations == 1


# ---------------------------------------------------------------------------
# OptimizerBase helpers
# ---------------------------------------------------------------------------


def _exact_match(prediction: Any, gold: dict[str, Any]) -> float:
    return 1.0 if getattr(prediction, "answer", None) == gold.get("answer") else 0.0


class _DemoOptimizer(OptimizerBase):
    """Trivial OptimizerBase subclass exercising the helper methods."""


class TestOptimizerBaseHelpers:
    def test_make_run_state_returns_fresh_components(self) -> None:
        opt = _DemoOptimizer(metric=_exact_match, budget=Budget(max_trials=10))
        tracker, ctx, runner = opt._make_run_state()
        assert tracker is not None
        assert tracker.trials == 0
        assert isinstance(ctx, RunContext)
        assert isinstance(runner, TrialRunner)

    def test_make_run_state_no_budget_returns_none_tracker(self) -> None:
        opt = _DemoOptimizer(metric=_exact_match)
        tracker, _ctx, _runner = opt._make_run_state()
        assert tracker is None

    def test_make_run_state_uses_supplied_run_context(self) -> None:
        """Composite optimizers thread their own RunContext through."""
        opt = _DemoOptimizer(metric=_exact_match)
        outer_ctx = RunContext()
        _tracker, ctx, _runner = opt._make_run_state(outer_ctx)
        assert ctx is outer_ctx

    async def test_evaluate_in_trial_charges_the_trial(self) -> None:
        """``_evaluate_in_trial`` runs ``evaluate()`` inside a trial scope
        and returns both the EvalResult and the populated trial."""
        opt = _DemoOptimizer(metric=_exact_match)
        _tracker, _ctx, runner = opt._make_run_state()
        call = _make_call()
        dataset = [
            Example(inputs={"topic": "a"}, outputs={"answer": "ok"}),
            Example(inputs={"topic": "b"}, outputs={"answer": "ok"}),
        ]
        eval_result, trial = await opt._evaluate_in_trial(
            call, dataset, runner=runner, trial_name="baseline"
        )
        assert eval_result.score == 1.0
        assert trial.name == "baseline"
        assert trial.n_invocations == 2
        assert trial.total_tokens == 30

    def test_consume_trial_advances_tracker(self) -> None:
        opt = _DemoOptimizer(metric=_exact_match, budget=Budget(max_trials=2))
        tracker, _ctx, _runner = opt._make_run_state()
        assert tracker is not None
        trial = Trial(name="t1", cost_usd=0.001, total_tokens=15, n_invocations=1)
        stop = opt._consume_trial(tracker, trial)
        assert stop is None
        assert tracker.trials == 1
        # Second consumption hits the cap
        stop = opt._consume_trial(tracker, trial)
        assert stop == StopReason.BUDGET_TRIALS

    def test_consume_trial_no_tracker_is_noop(self) -> None:
        opt = _DemoOptimizer(metric=_exact_match)
        tracker, _ctx, _runner = opt._make_run_state()
        trial = Trial(name="t1", cost_usd=0.001, total_tokens=15)
        assert opt._consume_trial(tracker, trial) is None

    def test_consume_trial_cost_cap_triggers(self) -> None:
        opt = _DemoOptimizer(metric=_exact_match, budget=Budget(max_cost_usd=0.005))
        tracker, _ctx, _runner = opt._make_run_state()
        trial = Trial(name="t1", cost_usd=0.006, total_tokens=10)
        stop = opt._consume_trial(tracker, trial)
        assert stop == StopReason.BUDGET_COST


class TestMetaOptimizerBaseSharedTracker:
    """``MetaOptimizerBase._attach_shared_tracker`` injects ONE BudgetTracker
    into a child so a composite Budget cap is enforced cumulatively across
    children, instead of each child building its own fresh tracker."""

    def test_attach_shared_tracker_propagates_into_make_run_state(self) -> None:
        """A child whose ``_inherited_tracker`` is set returns that exact
        tracker from ``_make_run_state``, not a fresh one from its own
        ``self.budget``."""
        meta = MetaOptimizerBase(metric=_exact_match, budget=Budget(max_trials=10))
        child = _DemoOptimizer(metric=_exact_match, budget=Budget(max_trials=99))
        # Build the shared tracker the meta would normally create.
        shared_tracker, _ctx, _runner = meta._make_run_state()
        assert shared_tracker is not None
        assert shared_tracker.trials == 0
        # Pre-consume a trial against the shared tracker so we can prove
        # the child's _make_run_state returns the *same* mutated object.
        shared_tracker.consume(trials=3, cost_usd=0.0, tokens=0)
        meta._attach_shared_tracker(child, shared_tracker)
        child_tracker, _ctx2, _runner2 = child._make_run_state()
        assert child_tracker is shared_tracker
        assert child_tracker.trials == 3

    def test_attach_shared_tracker_none_falls_through(self) -> None:
        """When the meta has no budget, the child still uses its own
        budget instead of inheriting None silently."""
        meta = MetaOptimizerBase(metric=_exact_match)
        child = _DemoOptimizer(metric=_exact_match, budget=Budget(max_trials=4))
        meta._attach_shared_tracker(child, None)
        child_tracker, _ctx, _runner = child._make_run_state()
        # ``_inherited_tracker`` was set to None — the contract says fall
        # through to the child's own ``self.budget``. The child gets its
        # own fresh tracker.
        assert child_tracker is not None
        assert child_tracker.budget.max_trials == 4
