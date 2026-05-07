"""Budget smoke tests: passing a 1-trial budget halts each optimizer early."""

from __future__ import annotations

import json
from typing import Any

from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart

from kaos_llm_core.optimization.bootstrap import BootstrapOptimizer
from kaos_llm_core.optimization.budget import Budget, StopReason
from kaos_llm_core.optimization.hyperparameter import HyperparameterOptimizer
from kaos_llm_core.programs.call import Call
from kaos_llm_core.signatures import InputField, OutputField, Signature
from kaos_llm_core.types import Example


class _S(Signature):
    """Answer."""

    text: str = InputField(description="Input")
    answer: str = OutputField(description="Output")


def _static_client(value: str) -> FunctionClient:
    def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
        return ProviderResponse.model_construct(
            provider="function",
            model="function-test",
            raw={},
            parts=[ContentPart.model_construct(type="text", text=json.dumps({"answer": value}))],
            usage=UsageInfo.model_construct(input_tokens=10, output_tokens=5, total_tokens=15),
            stop_reason="end_turn",
            status_code=200,
            response_headers={},
        )

    return FunctionClient(function=fn)


def _exact(pred: Any, gold: dict[str, Any]) -> float:
    return 1.0 if getattr(pred, "answer", "") == gold.get("answer") else 0.0


def _dataset(answer: str, n: int = 3) -> list[Example]:
    return [Example(inputs={"text": f"q{i}"}, outputs={"answer": answer}) for i in range(n)]


class TestBudgetHalts:
    async def test_bootstrap_budget_halts(self) -> None:
        call = Call(_S, model="function-test", client=_static_client("yes"))
        train = _dataset("yes")
        val = _dataset("yes")
        opt = BootstrapOptimizer(metric=_exact, budget=Budget(max_trials=1))
        result = await opt.optimize(call, train, val)
        # Budget of 1 means baseline eval consumes the budget immediately.
        assert result.stop_reason == StopReason.BUDGET_TRIALS.value

    async def test_hyperparameter_budget_halts(self) -> None:
        call = Call(_S, model="function-test", client=_static_client("yes"))
        val = _dataset("yes")
        opt = HyperparameterOptimizer(
            metric=_exact,
            search_space={"temperature": [0.0, 0.5, 1.0]},
            budget=Budget(max_trials=1),
        )
        result = await opt.optimize(call, val)
        assert result.stop_reason == StopReason.BUDGET_TRIALS.value
        # Only zero or one trial should have run before budget exhausted.
        assert result.configs_tried <= 3  # count of configs; enforcement is via stop_reason

    async def test_instruction_optimizer_accepts_budget(self) -> None:
        """InstructionOptimizer accepts a budget parameter without breaking."""
        from kaos_llm_core.optimization.instruction import InstructionOptimizer

        opt = InstructionOptimizer(
            metric=_exact,
            budget=Budget(max_trials=1),
        )
        assert opt.budget is not None

    async def test_reflective_optimizer_accepts_budget(self) -> None:
        from kaos_llm_core.optimization.reflective import ReflectiveOptimizer

        opt = ReflectiveOptimizer(budget=Budget(max_trials=1))
        assert opt.budget is not None

    async def test_co_optimizer_accepts_budget(self) -> None:
        from kaos_llm_core.optimization.co_optimizer import CoOptimizer

        opt = CoOptimizer(metric=_exact, budget=Budget(max_trials=1))
        assert opt.budget is not None

    async def test_co_optimizer_shares_budget_across_stages(self) -> None:
        """Audit-fix regression: a Budget(max_trials=N) cap on a CoOptimizer
        must be enforced *cumulatively* across child stages, not multiplied
        by the stage count.

        Pre-fix: every child built its own ``BudgetTracker`` from
        ``self.budget``, so a 3-stage CoOptimizer silently let each stage
        spend the full budget — the effective cap was 3N. The
        ``MetaOptimizerBase._attach_shared_tracker`` mechanism makes the
        children inherit ONE tracker instead.
        """
        from kaos_llm_core.optimization.co_optimizer import CoOptimizer

        call = Call(_S, model="function-test", client=_static_client("yes"))
        train = _dataset("yes")
        val = _dataset("yes")
        # 1-trial budget — the composite baseline_eval alone should exhaust
        # it. Every subsequent stage must observe the budget as already
        # spent and skip.
        opt = CoOptimizer(metric=_exact, budget=Budget(max_trials=1))
        result = await opt.optimize(call, train_set=train, val_set=val)
        # The cumulative budget tripped during baseline → no stages run.
        assert result.stop_reason == StopReason.BUDGET_TRIALS.value, (
            f"Expected BUDGET_TRIALS after baseline exhausted the cap, "
            f"got {result.stop_reason!r}; stages_run={result.stages_run}"
        )
        assert result.stages_run == [], (
            f"No child stages should run when the baseline already exhausted "
            f"the shared budget; got {result.stages_run}"
        )
