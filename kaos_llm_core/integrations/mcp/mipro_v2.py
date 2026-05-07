"""KaosLLMCoreMiproV2Tool — MCP wrapper for MiproV2Optimizer (Phase 17.1).

Wraps the full DSPy-style joint (instruction, demos) optimizer for
MCP clients. Parallel to ``KaosLLMCoreOptimizeTool`` (which wraps
``InstructionOptimizer`` — instruction-only) but searches the joint
space via TPE + GroundedProposer + minibatch + full-eval promotion.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from kaos_core import KaosContext, ToolResult
from kaos_core.types.annotations import ToolAnnotations
from kaos_core.types.enums import ToolCapability, ToolCategory
from kaos_core.types.parameters import ParameterSchema

from kaos_llm_core.integrations.mcp._common import (
    BaseLLMCoreTool,
    _resolve_metric_by_name,
    _settings_for,
)


class KaosLLMCoreMiproV2Tool(BaseLLMCoreTool):
    """Run the full DSPy MIPROv2 joint (instruction, demos) optimizer."""

    _NAME: ClassVar[str] = "kaos-llm-core-mipro-v2"
    _DISPLAY_NAME: ClassVar[str] = "LLM Core MIPRO v2"
    _DESCRIPTION: ClassVar[str] = (
        "Run the full DSPy MIPROv2 port: joint search over (instruction, demos) "
        "with categorical TPE, a GroundedProposer that builds dataset / program / "
        "module context for instruction proposals, minibatch evaluation during "
        "search, and periodic full-eval promotion. Provide labeled train and val "
        "examples; the tool returns the best (instruction, demos) bundle plus "
        "the metric_before / metric_after lift. More expensive than "
        "kaos-llm-core-optimize (which only tunes instructions) — set max_cost_usd "
        "to cap the spend. Single-predictor only in Phase 17.1; for multi-step "
        "programs use kaos-llm-core-optimize on each predictor sequentially. "
        "Design: docs/internal/design/mipro-v2-equivalent.md."
    )
    _CATEGORY: ClassVar[ToolCategory] = ToolCategory.INTEGRATION
    _CAPABILITY: ClassVar[ToolCapability] = ToolCapability.TRANSFORM
    _ANNOTATIONS: ClassVar[ToolAnnotations] = ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    )
    _PARAMETERS: ClassVar[list[ParameterSchema]] = [
        ParameterSchema(
            name="model",
            type="string",
            description=(
                "Target model id (e.g., 'anthropic:claude-haiku-4-5'). The "
                "instruction and demos will be optimized for this specific model."
            ),
            required=True,
        ),
        ParameterSchema(
            name="train_set",
            type="array",
            description=(
                "Array of {input, expected_output} objects used for demo "
                "bootstrapping. 15-30 examples is a good starting point."
            ),
            required=True,
        ),
        ParameterSchema(
            name="val_set",
            type="array",
            description=(
                "Array of {input, expected_output} objects used for evaluation "
                "(both minibatch and full-eval promotions). MUST be disjoint from "
                "train_set. 10-30 examples is a good starting point."
            ),
            required=True,
        ),
        ParameterSchema(
            name="initial_instruction",
            type="string",
            description=(
                "Optional starting instruction. If omitted, the optimizer uses "
                "the signature docstring as the baseline."
            ),
            required=False,
        ),
        ParameterSchema(
            name="proposer_model",
            type="string",
            description=(
                "Meta-LM for the GroundedInstructionProposer. Defaults to "
                "'anthropic:claude-sonnet-4-6'. Should be a strong model."
            ),
            required=False,
        ),
        ParameterSchema(
            name="num_candidates",
            type="integer",
            description=(
                "Number of instruction and demo-set candidates per predictor "
                "(default 4). Higher values explore more configurations but cost "
                "more proposer LM calls."
            ),
            required=False,
            constraints={"minimum": 2, "maximum": 18},
        ),
        ParameterSchema(
            name="num_trials",
            type="integer",
            description=(
                "Number of TPE search trials (default 12). Each trial runs one "
                "minibatch evaluation."
            ),
            required=False,
            constraints={"minimum": 1, "maximum": 100},
        ),
        ParameterSchema(
            name="minibatch_size",
            type="integer",
            description=(
                "Number of val examples sampled per search trial (default 8). "
                "Smaller minibatches are cheaper but noisier."
            ),
            required=False,
            constraints={"minimum": 1, "maximum": 100},
        ),
        ParameterSchema(
            name="minibatch_full_eval_steps",
            type="integer",
            description=(
                "Promote the best-mean candidate to a full-val evaluation every "
                "(N+1) trials (default 4 -> promotion every 5 trials)."
            ),
            required=False,
            constraints={"minimum": 1, "maximum": 50},
        ),
        ParameterSchema(
            name="max_bootstrapped_demos",
            type="integer",
            description=(
                "Maximum number of bootstrapped demos per candidate set "
                "(default 3). 0 enables zero-shot mode (no demo search)."
            ),
            required=False,
            constraints={"minimum": 0, "maximum": 16},
        ),
        ParameterSchema(
            name="max_cost_usd",
            type="number",
            description=(
                "Hard cost cap in USD (default 1.00). The optimizer halts when "
                "the shared budget tracker reports exhaustion. Use a smaller cap "
                "for cheaper experiments and a larger cap for more thorough runs."
            ),
            required=False,
            constraints={"minimum": 0.01, "maximum": 100.0},
        ),
        ParameterSchema(
            name="metric_name",
            type="string",
            description=(
                "Metric to optimize against. One of 'normalized_match' (default), "
                "'exact_match', 'case_insensitive_match', 'length_ratio'."
            ),
            required=False,
            constraints={
                "enum": [
                    "normalized_match",
                    "exact_match",
                    "case_insensitive_match",
                    "length_ratio",
                ]
            },
        ),
        ParameterSchema(
            name="seed",
            type="integer",
            description="Random seed for demo shuffling, tip choice, TPE sampling.",
            required=False,
        ),
    ]
    _ERROR_HINT: ClassVar[str] = (
        "Check that (1) model id and proposer_model are valid (e.g. "
        "'anthropic:claude-haiku-4-5'), (2) the API key is set in the environment, "
        "(3) train_set and val_set each contain at least 4 examples with both "
        "'input' and 'expected_output' keys. For instruction-only optimization "
        "without demo search, use kaos-llm-core-optimize instead."
    )

    async def _run(self, inputs: dict[str, Any], context: KaosContext | None = None) -> ToolResult:
        from pydantic import create_model

        from kaos_llm_core import InputField, OutputField, Signature
        from kaos_llm_core.optimization.budget import Budget
        from kaos_llm_core.optimization.mipro_v2 import (
            MiproV2Optimizer,
            MiproV2Result,
        )
        from kaos_llm_core.optimization.mutations import MutationLog
        from kaos_llm_core.programs.call import Call
        from kaos_llm_core.types import Example

        # ----- Parse inputs -----
        model = inputs["model"]
        train_raw = inputs["train_set"]
        val_raw = inputs["val_set"]
        initial_instruction = inputs.get("initial_instruction") or "Answer based on the input."
        proposer_model = inputs.get("proposer_model") or "anthropic:claude-sonnet-4-6"
        num_candidates = int(inputs.get("num_candidates", 4))
        num_trials = int(inputs.get("num_trials", 12))
        minibatch_size = int(inputs.get("minibatch_size", 8))
        minibatch_full_eval_steps = int(inputs.get("minibatch_full_eval_steps", 4))
        max_bootstrapped_demos = int(inputs.get("max_bootstrapped_demos", 3))
        max_cost_usd = float(inputs.get("max_cost_usd", 1.00))
        metric_name = inputs.get("metric_name", "normalized_match")
        seed = int(inputs.get("seed", 9))

        # ----- Parse train_set / val_set (string-or-list permissive) -----
        if isinstance(train_raw, str):
            train_raw = json.loads(train_raw)
        if isinstance(val_raw, str):
            val_raw = json.loads(val_raw)
        if not isinstance(train_raw, list) or not isinstance(val_raw, list):
            return ToolResult.create_error(
                "train_set and val_set must each be a list of "
                "{input, expected_output} objects (or a JSON-encoded string of "
                "such a list). Use kaos-llm-core-evaluate first to establish "
                "a baseline before optimizing."
            )
        if len(train_raw) < 4:
            return ToolResult.create_error(
                f"train_set must contain at least 4 examples for meaningful demo "
                f"bootstrap; got {len(train_raw)}. Add more labeled examples or "
                f"use kaos-llm-core-optimize for instruction-only tuning on "
                f"smaller datasets."
            )
        if len(val_raw) < 1:
            return ToolResult.create_error(
                "val_set must be non-empty. Provide held-out examples for evaluation."
            )

        # ----- Build dynamic Signature -----
        sig = create_model(
            "DynamicMiproV2Signature",
            __base__=Signature,
            __doc__=initial_instruction,
            text=(str, InputField(description="Input text")),
            answer=(str, OutputField(description="The answer")),
        )

        def _to_examples(rows: list[dict[str, Any]]) -> list[Example] | ToolResult:
            out: list[Example] = []
            for ex in rows:
                if "input" not in ex or "expected_output" not in ex:
                    return ToolResult.create_error(
                        f"Each example must have 'input' and 'expected_output' "
                        f"keys. Got keys: {list(ex.keys())}."
                    )
                out.append(
                    Example(
                        inputs={"text": str(ex["input"])},
                        outputs={"answer": str(ex["expected_output"])},
                    )
                )
            return out

        train = _to_examples(train_raw)
        if isinstance(train, ToolResult):
            return train
        val = _to_examples(val_raw)
        if isinstance(val, ToolResult):
            return val

        # ----- Resolve metric -----
        metric_fn = _resolve_metric_by_name(metric_name)
        if metric_fn is None:
            return ToolResult.create_error(
                f"Unknown metric_name {metric_name!r}. Valid: normalized_match, "
                f"exact_match, case_insensitive_match, length_ratio."
            )

        # ----- Build the call + optimizer -----
        settings = _settings_for(context)
        call = Call(sig, model=model, instructions=initial_instruction, core_settings=settings)
        log = MutationLog()
        opt = MiproV2Optimizer(
            metric=metric_fn,
            proposer_model=proposer_model,
            auto=None,
            num_candidates=num_candidates,
            num_trials=num_trials,
            minibatch=True,
            minibatch_size=minibatch_size,
            minibatch_full_eval_steps=minibatch_full_eval_steps,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=0,
            seed=seed,
            mutation_log=log,
            budget=Budget(max_cost_usd=max_cost_usd),
        )

        # ----- Run -----
        result: MiproV2Result = await opt.optimize(
            call, train_set=train, val_set=val, max_concurrent=4
        )

        # ----- Build the response payload -----
        output: dict[str, Any] = {
            "best_instruction": result.best_config.instruction,
            "n_demos_added": len(result.best_config.demos),
            "metric_before": round(result.metric_before, 4),
            "metric_after": round(result.metric_after, 4),
            "improvement": round(result.metric_after - result.metric_before, 4),
            "accepted": result.accepted,
            "trials_run": result.trials_run,
            "proposer_calls": result.proposer_calls,
            "total_cost_usd": round(log.total_cost(), 6),
            "total_tokens": log.total_tokens(),
            "stop_reason": result.stop_reason,
        }

        summary_parts = [
            f"Score: {result.metric_before:.1%} -> {result.metric_after:.1%}",
            f"accepted={result.accepted}",
            f"trials={result.trials_run}",
            f"cost=${log.total_cost():.4f}",
        ]

        return ToolResult.create_success(
            output=output,
            summary=", ".join(summary_parts),
        )
