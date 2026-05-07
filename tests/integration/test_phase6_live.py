"""Live integration tests for kaos-llm-core Phase 6 (optimizers + Budget).

These hit real LLM provider APIs with **production models** — `claude-opus-4-6`,
`claude-sonnet-4-6`, `claude-haiku-4-5`, `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano`,
`gemini-2.5-pro`, `gemini-2.5-flash`. Each test is gated on the corresponding API
key. This file matches the discipline of `test_react_live.py` /
`test_refine_live.py` and is the E2E verification step for Phase 6.

Real-domain task: contract clause classification. The signature returns one of
seven enterprise contract clause types (indemnification, limitation_of_liability,
confidentiality, termination, payment_terms, warranty, governing_law). The
training and validation sets are 7-clause and 6-clause respectively, drawn from
boilerplate language commonly seen in enterprise SaaS / NDA / MSA contracts.

Run::

    uv run pytest tests/integration/test_phase6_live.py -v -m integration -s

The ``-s`` flag is recommended so you can watch live cost / token output.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from kaos_llm_core.codecs.chat_codec import ChatCodec
from kaos_llm_core.codecs.json_codec import JSONCodec
from kaos_llm_core.codecs.xml_codec import XMLCodec
from kaos_llm_core.optimization.budget import Budget, StopReason
from kaos_llm_core.optimization.codec_optimizer import CodecOptimizer
from kaos_llm_core.optimization.hyperparameter import HyperparameterOptimizer
from kaos_llm_core.optimization.model_optimizer import ModelOptimizer
from kaos_llm_core.optimization.mutations import MutationLog
from kaos_llm_core.optimization.pareto import ParetoOptimizer
from kaos_llm_core.optimization.recipes import (
    CostAwareModelSelector,
    FriendlyPromptTuner,
)
from kaos_llm_core.programs.call import Call
from kaos_llm_core.signatures import InputField, OutputField, Signature
from kaos_llm_core.types import Example

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------


def _has_key(*env_vars: str) -> bool:
    return any(os.getenv(v) for v in env_vars)


requires_anthropic = pytest.mark.skipif(
    not _has_key("KAOS_LLM_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    reason="No Anthropic API key",
)
requires_openai = pytest.mark.skipif(
    not _has_key("KAOS_LLM_OPENAI_API_KEY", "OPENAI_API_KEY"),
    reason="No OpenAI API key",
)
requires_google = pytest.mark.skipif(
    not _has_key("KAOS_LLM_GOOGLE_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY"),
    reason="No Google API key",
)
requires_anthropic_and_openai = pytest.mark.skipif(
    not (
        _has_key("KAOS_LLM_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY")
        and _has_key("KAOS_LLM_OPENAI_API_KEY", "OPENAI_API_KEY")
    ),
    reason="Need both Anthropic + OpenAI keys",
)


# ---------------------------------------------------------------------------
# Real legal-domain task: contract clause classification
# ---------------------------------------------------------------------------


VALID_LABELS = (
    "indemnification",
    "limitation_of_liability",
    "confidentiality",
    "termination",
    "payment_terms",
    "warranty",
    "governing_law",
)


class ClauseLabel(Signature):
    """You are a contracts attorney triaging clauses from a commercial agreement.
    Read the clause and return the single most accurate label from this set:
    indemnification, limitation_of_liability, confidentiality, termination,
    payment_terms, warranty, governing_law. Return only the label string,
    in lowercase, with underscores. No prose, no quotation marks.
    """

    clause: str = InputField(description="The contract clause text to classify")
    label: str = OutputField(
        description=(
            "Exactly one label from the set: indemnification, "
            "limitation_of_liability, confidentiality, termination, "
            "payment_terms, warranty, governing_law"
        )
    )


# Realistic clause language. Sourced from common enterprise SaaS / NDA / MSA
# boilerplate (paraphrased).
TRAIN_EXAMPLES: list[Example] = [
    Example(
        inputs={
            "clause": (
                "Each party shall defend, indemnify, and hold harmless the other "
                "party from any third-party claims arising out of or related to the "
                "indemnifying party's breach of this Agreement."
            )
        },
        outputs={"label": "indemnification"},
    ),
    Example(
        inputs={
            "clause": (
                "Notwithstanding anything to the contrary in this Agreement, neither "
                "party's aggregate liability shall exceed the total fees paid by "
                "Customer in the twelve months preceding the claim."
            )
        },
        outputs={"label": "limitation_of_liability"},
    ),
    Example(
        inputs={
            "clause": (
                "Recipient shall not disclose Confidential Information to any third "
                "party except its employees and contractors who have a need to know "
                "and are bound by written obligations no less protective than those "
                "set forth herein."
            )
        },
        outputs={"label": "confidentiality"},
    ),
    Example(
        inputs={
            "clause": (
                "Either party may terminate this Agreement for convenience upon "
                "ninety (90) days prior written notice to the other party."
            )
        },
        outputs={"label": "termination"},
    ),
    Example(
        inputs={
            "clause": (
                "Customer shall pay all undisputed invoices within thirty (30) days "
                "of the invoice date. Late payments shall accrue interest at the "
                "lesser of 1.5% per month or the maximum rate permitted by law."
            )
        },
        outputs={"label": "payment_terms"},
    ),
    Example(
        inputs={
            "clause": (
                "Vendor warrants that the Services will be performed in a "
                "professional and workmanlike manner consistent with industry "
                "standards generally applicable to similar services."
            )
        },
        outputs={"label": "warranty"},
    ),
    Example(
        inputs={
            "clause": (
                "This Agreement shall be governed by and construed in accordance "
                "with the laws of the State of Delaware, without regard to its "
                "conflict-of-laws principles."
            )
        },
        outputs={"label": "governing_law"},
    ),
]


VAL_EXAMPLES: list[Example] = [
    Example(
        inputs={
            "clause": (
                "Supplier agrees to indemnify Customer against any claim that the "
                "Services infringe a U.S. patent, copyright, or trademark of any "
                "third party."
            )
        },
        outputs={"label": "indemnification"},
    ),
    Example(
        inputs={
            "clause": (
                "In no event shall either party be liable for any indirect, "
                "incidental, special, consequential, or punitive damages, including "
                "lost profits or revenue."
            )
        },
        outputs={"label": "limitation_of_liability"},
    ),
    Example(
        inputs={
            "clause": (
                "All technical, business, and financial information disclosed "
                "hereunder shall be deemed Confidential Information whether or not "
                "marked as such at the time of disclosure."
            )
        },
        outputs={"label": "confidentiality"},
    ),
    Example(
        inputs={
            "clause": (
                "This Agreement shall commence on the Effective Date and continue "
                "for an initial term of two (2) years, after which it will "
                "automatically renew for successive one-year terms unless either "
                "party provides written notice of non-renewal at least sixty (60) "
                "days prior to the end of the then-current term."
            )
        },
        outputs={"label": "termination"},
    ),
    Example(
        inputs={
            "clause": (
                "All fees are exclusive of taxes. Customer shall be responsible for "
                "all sales, use, and similar taxes assessed on the Services, "
                "excluding taxes based on Vendor's net income."
            )
        },
        outputs={"label": "payment_terms"},
    ),
    Example(
        inputs={
            "clause": (
                "Any dispute arising out of or relating to this Agreement shall be "
                "resolved exclusively in the state and federal courts located in "
                "the Borough of Manhattan, New York, and the parties hereby submit "
                "to the personal jurisdiction of such courts."
            )
        },
        outputs={"label": "governing_law"},
    ),
]


def clause_match(prediction: Any, gold: dict[str, Any]) -> float:
    """Strict label match. Lowercased, whitespace-stripped, underscore-normalized."""
    pred_label = str(getattr(prediction, "label", "")).strip().lower().replace(" ", "_")
    # Models sometimes wrap in quotes or add prose; pull the first matching label.
    for valid in VALID_LABELS:
        if valid in pred_label:
            pred_label = valid
            break
    gold_label = str(gold.get("label", "")).strip().lower()
    return 1.0 if pred_label == gold_label else 0.0


def _make_call(model: str, codec: Any | None = None) -> Call:
    """Build a fresh Call instance bound to a specific model."""
    kwargs: dict[str, Any] = {}
    if codec is not None:
        kwargs["codec"] = codec
    return Call(ClauseLabel, model=model, **kwargs)


def _print_cost(label: str, *, cost: float, tokens: int) -> None:
    print(f"\n  [{label}] cost=${cost:.6f} tokens={tokens}")


# ---------------------------------------------------------------------------
# 1. Budget exhaustion (live)
# ---------------------------------------------------------------------------


class TestBudgetLive:
    """Verify Budget actually halts a real optimizer with a real model."""

    @requires_anthropic
    async def test_max_trials_stops_codec_optimizer(self) -> None:
        """A budget of 1 trial should let exactly one codec score and halt."""
        budget = Budget(max_trials=1)
        opt = CodecOptimizer(
            metric=clause_match,
            codecs=[JSONCodec, ChatCodec, XMLCodec],
            budget=budget,
        )
        call = _make_call("anthropic:claude-haiku-4-5")
        result = await opt.optimize(call, VAL_EXAMPLES[:3])

        print(
            f"\n  [budget_max_trials] stop_reason={result.stop_reason} "
            f"scored={list(result.scores_by_codec.keys())}"
        )
        # Budget halts the LOOP at the *next* iteration after consuming the cap,
        # so exactly one codec should have scored before the cap kicked in.
        assert len(result.scores_by_codec) == 1, (
            f"Expected exactly 1 codec to be evaluated under max_trials=1; "
            f"got {len(result.scores_by_codec)}: {list(result.scores_by_codec)}"
        )
        assert result.stop_reason == StopReason.BUDGET_TRIALS.value, (
            f"Expected stop_reason=budget_trials, got {result.stop_reason}"
        )
        # Cost and tokens were captured (GAP-2 verification).
        only_mutation = result.mutations[0]
        assert only_mutation.cost_usd > 0.0, (
            f"Expected cost_usd > 0 from real Anthropic call; got {only_mutation.cost_usd}"
        )
        assert only_mutation.tokens_used > 0, (
            f"Expected tokens_used > 0 from real Anthropic call; got {only_mutation.tokens_used}"
        )
        _print_cost(
            "budget_max_trials", cost=only_mutation.cost_usd, tokens=only_mutation.tokens_used
        )

    @requires_anthropic
    async def test_max_cost_usd_caps_codec_optimizer(self) -> None:
        """A near-zero cost cap should halt before all codecs are tried."""
        # 1e-9 USD = effectively zero. The cap will trip on the second iteration
        # because exhausted() is checked AT THE TOP of each loop iteration after
        # the previous trial has consumed cost.
        budget = Budget(max_cost_usd=1e-9)
        opt = CodecOptimizer(
            metric=clause_match,
            codecs=[JSONCodec, ChatCodec, XMLCodec],
            budget=budget,
        )
        call = _make_call("anthropic:claude-haiku-4-5")
        result = await opt.optimize(call, VAL_EXAMPLES[:2])

        print(
            f"\n  [budget_max_cost] stop_reason={result.stop_reason} "
            f"scored={list(result.scores_by_codec.keys())}"
        )
        # First codec gets to run because the tracker starts empty; the cap is
        # checked at the top of the second iteration. Expect 1 codec scored.
        assert len(result.scores_by_codec) == 1, (
            f"Expected 1 codec under near-zero cost cap; got {len(result.scores_by_codec)}"
        )
        assert result.stop_reason == StopReason.BUDGET_COST.value, (
            f"Expected stop_reason=budget_cost, got {result.stop_reason}"
        )


# ---------------------------------------------------------------------------
# 2. CodecOptimizer (live)
# ---------------------------------------------------------------------------


class TestCodecOptimizerLive:
    """Real codec selection across JSON / Chat / XML on the legal task."""

    @requires_anthropic
    async def test_codec_selection_with_sonnet_4_6(self) -> None:
        log = MutationLog()
        opt = CodecOptimizer(
            metric=clause_match,
            codecs=[JSONCodec, ChatCodec, XMLCodec],
            mutation_log=log,
        )
        call = _make_call("anthropic:claude-sonnet-4-6")
        result = await opt.optimize(call, VAL_EXAMPLES)

        print(
            f"\n  [codec_sonnet_4_6] best={result.best_codec.__name__} "
            f"score={result.best_score:.3f} "
            f"scores={result.scores_by_codec} "
            f"stop={result.stop_reason}"
        )

        assert result.best_codec is not None
        assert result.best_score >= 0.5, (
            f"sonnet-4-6 should score >= 50% on a 6-clause classification task; "
            f"got {result.best_score:.3f}. scores={result.scores_by_codec}"
        )
        assert len(result.scores_by_codec) == 3, (
            f"All 3 codecs should be scored; got {list(result.scores_by_codec)}"
        )
        assert result.stop_reason == StopReason.COMPLETED.value
        # GAP-2 verification: every mutation has real cost and token data.
        total_cost = sum(m.cost_usd for m in result.mutations)
        total_tokens = sum(m.tokens_used for m in result.mutations)
        assert total_cost > 0.0, "All 3 codec trials should have non-zero cost"
        assert total_tokens > 0, "All 3 codec trials should have non-zero tokens"
        _print_cost("codec_sonnet_4_6_total", cost=total_cost, tokens=total_tokens)
        # MutationLog persistence (in-memory) populated.
        assert len(log.mutations) == 3
        assert log.total_cost() == pytest.approx(total_cost, rel=1e-9)


# ---------------------------------------------------------------------------
# 3. ModelOptimizer (live)
# ---------------------------------------------------------------------------


class TestModelOptimizerLive:
    """Real model selection — opus / sonnet / haiku, cheapest above threshold."""

    @requires_anthropic
    async def test_anthropic_tier_selection(self) -> None:
        models = [
            "anthropic:claude-opus-4-6",
            "anthropic:claude-sonnet-4-6",
            "anthropic:claude-haiku-4-5",
        ]
        opt = ModelOptimizer(
            metric=clause_match,
            models=models,
            min_score=0.8,
        )
        call = _make_call("anthropic:claude-haiku-4-5")
        result = await opt.optimize(call, VAL_EXAMPLES)

        print(
            f"\n  [model_anthropic_tier] best={result.best_model} "
            f"score={result.best_score:.3f} "
            f"scores={result.scores_by_model} "
            f"costs={result.cost_by_model} "
            f"stop={result.stop_reason}"
        )

        # All 3 models should be scored.
        assert set(result.scores_by_model.keys()) == set(models)
        # Costs were extracted from ExecutionTrace (GAP-2 verification).
        for m in models:
            assert result.cost_by_model[m] > 0.0, (
                f"Expected non-zero cost for {m}; got {result.cost_by_model[m]}. "
                f"This is the GAP-2 fix — every model trial must extract cost from "
                f"the per-trial ExecutionTrace via estimate_eval_cost()."
            )
            assert result.tokens_by_model[m] > 0, f"Expected non-zero tokens for {m}"

        # Cheapest qualifying model should be picked. On a 6-example task all
        # three Anthropic flagships should easily clear 0.8, so the selection
        # should be the cheapest of them (haiku-4-5).
        qualifying = {m for m, s in result.scores_by_model.items() if s >= 0.8}
        if qualifying:
            cheapest = min(qualifying, key=lambda m: result.cost_by_model[m])
            assert result.best_model == cheapest, (
                f"Expected cheapest qualifying model {cheapest}; got {result.best_model}"
            )
            assert result.stop_reason == StopReason.COMPLETED.value
        else:
            # Fallback path — all three failed to clear 0.8. We still want to
            # surface the highest scorer with stop_reason=THRESHOLD_NOT_MET.
            assert result.stop_reason == StopReason.THRESHOLD_NOT_MET.value
            print(
                "  [model_anthropic_tier] No model cleared min_score=0.8; "
                f"highest scorer = {result.best_model} @ {result.best_score:.3f}"
            )

        total_cost = sum(result.cost_by_model.values())
        total_tokens = sum(result.tokens_by_model.values())
        _print_cost("model_anthropic_tier_total", cost=total_cost, tokens=total_tokens)


# ---------------------------------------------------------------------------
# 4. ParetoOptimizer (live, multi-provider)
# ---------------------------------------------------------------------------


class TestParetoLive:
    """Real Pareto frontier across multiple providers and capability tiers."""

    @requires_anthropic_and_openai
    async def test_cross_provider_frontier(self) -> None:
        models = [
            "anthropic:claude-opus-4-6",
            "anthropic:claude-sonnet-4-6",
            "anthropic:claude-haiku-4-5",
            "openai:gpt-5.4",
            "openai:gpt-5.4-mini",
            "openai:gpt-5.4-nano",
        ]
        if _has_key("KAOS_LLM_GOOGLE_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY"):
            models.extend(["google:gemini-2.5-pro", "google:gemini-2.5-flash"])

        inner = ModelOptimizer(
            metric=clause_match,
            models=models,
            min_score=0.0,  # Pareto wants every trial scored.
        )
        pareto = ParetoOptimizer(metric=clause_match, inner=inner)
        call = _make_call("anthropic:claude-haiku-4-5")
        result = await pareto.optimize(call, VAL_EXAMPLES)

        print(
            f"\n  [pareto_cross_provider] {len(result.all_trials)} trials, "
            f"{len(result.frontier)} on frontier"
        )
        for cfg, metric, cost in result.frontier:
            model = cfg.get("model", "?")
            print(f"    frontier: {model:40s} metric={metric:.3f} cost=${cost:.6f}")
        for cfg, metric, cost in result.all_trials:
            model = cfg.get("model", "?")
            print(f"    trial:    {model:40s} metric={metric:.3f} cost=${cost:.6f}")

        assert len(result.all_trials) == len(models), (
            f"Expected {len(models)} trials in Pareto; got {len(result.all_trials)}"
        )
        assert len(result.frontier) >= 1, "Pareto frontier must contain at least one point"
        assert len(result.frontier) <= len(result.all_trials)
        # Frontier is sorted by cost ascending.
        costs = [c for _, _, c in result.frontier]
        assert costs == sorted(costs), f"Frontier not sorted by cost ascending: {costs}"
        # Every frontier point is non-dominated by every trial.
        for fi, (_, m_i, c_i) in enumerate(result.frontier):
            for _, m_j, c_j in result.all_trials:
                if m_j >= m_i and c_j <= c_i and (m_j > m_i or c_j < c_i):
                    raise AssertionError(
                        f"Frontier point {fi} (metric={m_i:.3f} cost=${c_i:.6f}) is "
                        f"dominated by trial (metric={m_j:.3f} cost=${c_j:.6f})"
                    )
        total_cost = sum(c for _, _, c in result.all_trials)
        _print_cost("pareto_cross_provider_total", cost=total_cost, tokens=0)


# ---------------------------------------------------------------------------
# 5. Recipes — FriendlyPromptTuner (live)
# ---------------------------------------------------------------------------


class TestRecipesLive:
    """End-to-end recipe runs against real models."""

    @requires_anthropic
    async def test_friendly_prompt_tuner_with_sonnet_proposer(self) -> None:
        # Sonnet 4.6 as the instruction proposer; the Call itself runs against
        # haiku-4-5 to keep cost low while exercising both layers.
        opt = FriendlyPromptTuner(
            metric=clause_match,
            proposer_model="anthropic:claude-sonnet-4-6",
        )
        call = _make_call("anthropic:claude-haiku-4-5")
        result = await opt.optimize(
            call,
            train_set=TRAIN_EXAMPLES,
            val_set=VAL_EXAMPLES,
        )

        print(
            f"\n  [recipe_friendly_prompt] stages_run={result.stages_run} "
            f"baseline={result.metric_before:.3f} final={result.metric_after:.3f} "
            f"stop={result.stop_reason}"
        )
        # The recipe must actually have run something.
        assert len(result.stages_run) >= 1
        assert result.metric_after >= result.metric_before - 0.01, (
            f"Optimization should not significantly regress; "
            f"baseline={result.metric_before:.3f} final={result.metric_after:.3f}"
        )
        # The shared mutation log captured trials across stages.
        mutations = result.mutation_log.mutations if result.mutation_log else []
        assert len(mutations) > 0, "Recipe should have recorded at least one mutation"
        cost = sum(m.cost_usd for m in mutations)
        tokens = sum(m.tokens_used for m in mutations)
        # Note: instruction-stage mutations populate cost from estimate_eval_cost,
        # but cost may legitimately be 0 if traces don't carry pricing for the
        # specific provider/model combo. Don't hard-assert > 0 here — the
        # codec/model live tests already prove cost extraction works.
        _print_cost("recipe_friendly_prompt_total", cost=cost, tokens=tokens)

    @requires_anthropic
    async def test_cost_aware_model_selector(self) -> None:
        opt = CostAwareModelSelector(
            metric=clause_match,
            models=[
                "anthropic:claude-haiku-4-5",
                "anthropic:claude-sonnet-4-6",
                "anthropic:claude-opus-4-6",
            ],
            min_score=0.6,  # Achievable on this task by every Anthropic tier.
        )
        call = _make_call("anthropic:claude-haiku-4-5")
        result = await opt.optimize(call, VAL_EXAMPLES)

        print(
            f"\n  [recipe_cost_aware_selector] best={result.best_model} "
            f"score={result.best_score:.3f} stop={result.stop_reason}"
        )
        assert result.best_model in {
            "anthropic:claude-haiku-4-5",
            "anthropic:claude-sonnet-4-6",
            "anthropic:claude-opus-4-6",
        }
        # CostAwareModelSelector defaults to min_score=0.85 in the recipe wrapper
        # but we override to 0.6 here, so we expect the cheapest Anthropic model
        # qualifying — which on this task should be haiku-4-5.
        qualifying = {m for m, s in result.scores_by_model.items() if s >= 0.6}
        if qualifying:
            cheapest = min(qualifying, key=lambda m: result.cost_by_model[m])
            assert result.best_model == cheapest


# ---------------------------------------------------------------------------
# 6. LatinHypercubeSearch (live)
# ---------------------------------------------------------------------------


class TestLatinHypercubeSearchLive:
    """Real temperature/top_p sweep verifying LHS samples land in bins."""

    @requires_anthropic
    async def test_latin_hypercube_temperature_sweep(self) -> None:
        # NOTE: Anthropic models reject `temperature` AND `top_p` set together
        # ("temperature and top_p cannot both be specified for this model").
        # We sweep temperature only here. A two-parameter sweep would need
        # OpenAI / Google or a different provider. This is exactly the kind
        # of constraint the optimizer's diagnostic-on-100%-error path needs
        # to surface — see the carry-forward note in the roadmap §9.
        search_space = {
            "temperature": [0.0, 0.3, 0.7, 1.0],
        }
        log = MutationLog()
        opt = HyperparameterOptimizer(
            metric=clause_match,
            search_space=search_space,
            strategy="latin_hypercube",
            max_trials=4,
            seed=42,
            mutation_log=log,
        )
        call = _make_call("anthropic:claude-haiku-4-5")
        # HyperparameterOptimizer.optimize() takes (call, val_set) — no train_set.
        result = await opt.optimize(call, val_set=VAL_EXAMPLES)

        print(
            f"\n  [lhs_temperature_sweep] best={result.best_params} "
            f"baseline={result.eval_before.score:.3f} after={result.eval_after.score:.3f} "
            f"configs_tried={result.configs_tried} stop={result.stop_reason} "
            f"accepted={result.accepted}"
        )
        print(f"  [lhs_temperature_sweep] mutation log: {len(log.mutations)} trials")
        for m in log.mutations:
            print(
                f"    trial: temp={m.after.get('temperature')!s:>5} "
                f"score={m.metric_after:.3f} cost=${m.cost_usd:.6f}"
            )

        # LHS with n_samples=4 over a 4-value parameter should produce 4 trials.
        assert result.configs_tried == 4, (
            f"LHS should generate exactly max_trials=4 configurations; got {result.configs_tried}"
        )
        # The mutation log captured every trial — verify LHS actually emitted them.
        assert len(log.mutations) == 4, f"Expected 4 LHS trials in log; got {len(log.mutations)}"
        # Every trial config is a member of the search space.
        for m in log.mutations:
            assert m.after["temperature"] in search_space["temperature"], (
                f"LHS produced out-of-grid temperature: {m.after}"
            )
        # LHS stratification: each candidate value should appear exactly once
        # when n_samples == len(values).
        temps_seen = sorted(m.after["temperature"] for m in log.mutations)
        assert temps_seen == sorted(search_space["temperature"]), (
            f"LHS failed to stratify temperature; expected each value once, got {temps_seen}"
        )
        # GAP-2 verification: every LHS trial should have real cost extracted.
        total_cost = sum(m.cost_usd for m in log.mutations)
        total_tokens = sum(m.tokens_used for m in log.mutations)
        assert total_cost > 0.0, "Expected non-zero total cost from LHS trials (GAP-2 check)"
        assert total_tokens > 0, "Expected non-zero total tokens from LHS trials"
        _print_cost("lhs_temperature_sweep_total", cost=total_cost, tokens=total_tokens)
        # Don't regress.
        assert result.eval_after.score >= result.eval_before.score - 0.01
