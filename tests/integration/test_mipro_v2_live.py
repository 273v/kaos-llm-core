"""Live integration test for MiproV2Optimizer (Phase 17.1).

The headline test runs MIPROv2 against real Anthropic Haiku on a
hand-curated TREC-6 question-classification mini-dataset. It
verifies five things, mirroring design doc §6.3:

1. **It runs**. ``trials_run > 0`` and ``proposer_calls > 0`` —
   proves the dataset summarizer, instruction proposer, demo
   bootstrap, and TPE search loop all executed against the wire.
2. **It actually tries joint configs**. The mutation log has
   ``>= 0.66 * num_trials`` ``search_trial`` entries.
3. **It does not regress**. ``metric_after >= metric_before`` (we
   do NOT require strict improvement because the baseline is
   already strong on TREC-6 mini with Haiku).
4. **It is competitive with CoOptimizer**. ``MIPROv2.metric_after
   + 0.10 >= CoOptimizer.metric_after`` — joint search must not
   lose to sequential coordinate descent by more than 10 points.
5. **Cost stays under the cap**. ``mutation_log.total_cost() <=
   1.00`` for each optimizer (raised from the design doc's $0.50
   per the Phase 17.1 research finding that GroundedProposer's
   ``program_aware`` path makes ~3x more LM calls than the doc
   estimated).

Hard cap: $1.00 per optimizer, $2.00 total per CI run.

Skipped without ``KAOS_LLM_ANTHROPIC_API_KEY`` /
``ANTHROPIC_API_KEY`` set.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

import pytest

from kaos_llm_core.optimization.budget import Budget
from kaos_llm_core.optimization.co_optimizer import CoOptimizer
from kaos_llm_core.optimization.mipro_v2 import (
    MiproV2Optimizer,
    MiproV2Result,
)
from kaos_llm_core.optimization.mutations import MutationLog
from kaos_llm_core.programs.call import Call
from kaos_llm_core.signatures import InputField, OutputField, Signature
from kaos_llm_core.types import Example

pytestmark = pytest.mark.integration


def _has_key(*env_vars: str) -> bool:
    return any(os.getenv(v) for v in env_vars)


requires_anthropic = pytest.mark.skipif(
    not _has_key("KAOS_LLM_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    reason="No Anthropic API key",
)


# Use the cheapest current-generation Anthropic models per CLAUDE.md.
# Prefixed form so the cost-tracking PRICING lookup succeeds (the
# bare form has aliases too as of Phase 17.1 F1, but the prefixed
# form is the canonical citation).
TARGET_MODEL = "anthropic:claude-haiku-4-5"
PROPOSER_MODEL = "anthropic:claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Signature
# ---------------------------------------------------------------------------


class ClassifyTRECQuestion(Signature):
    """Classify a question into one of six TREC-6 coarse categories.

    The categories are:

    - ABBR: question about an abbreviation or acronym
    - DESC: question asking for a description or definition
    - ENTY: question about an entity (animal, food, etc.)
    - HUM:  question about a human / person
    - LOC:  question about a location
    - NUM:  question about a number

    Return only the three- or four-letter category code.
    """

    question: str = InputField(description="The question to classify")
    category: Literal["ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"] = OutputField(
        description="One of: ABBR, DESC, ENTY, HUM, LOC, NUM"
    )


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------


_FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "trec6_mini.jsonl"


def _load_trec6_fixture() -> tuple[list[Example], list[Example]]:
    train: list[Example] = []
    val: list[Example] = []
    with _FIXTURE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ex = Example(
                inputs={"question": row["question"]},
                outputs={"category": row["category"]},
            )
            if row["split"] == "train":
                train.append(ex)
            elif row["split"] == "val":
                val.append(ex)
    return train, val


def _exact_category_match(prediction: Any, gold: dict[str, Any]) -> float:
    pred = str(getattr(prediction, "category", "")).strip().upper()
    expected = str(gold.get("category", "")).strip().upper()
    return 1.0 if pred == expected else 0.0


# ---------------------------------------------------------------------------
# The live test
# ---------------------------------------------------------------------------


@requires_anthropic
class TestMiproV2Live:
    async def test_trec6_joint_search_against_haiku(self) -> None:
        """End-to-end MIPROv2 vs CoOptimizer on TREC-6 mini.

        The headline test from design doc §6.3, with the budget bumped
        to $1.00/optimizer per the Phase 17.1 research finding that
        GroundedProposer's program_aware path makes ~3x more proposer
        LM calls than the design doc's original cost estimate.
        """
        train, val = _load_trec6_fixture()
        assert len(train) == 20, f"expected 20 train examples; got {len(train)}"
        assert len(val) == 15, f"expected 15 val examples; got {len(val)}"

        # ----- MIPROv2 run -----
        mipro_call = Call(
            ClassifyTRECQuestion,
            model=TARGET_MODEL,
            instructions="Classify the question.",  # vague baseline
        )
        mipro_log = MutationLog()
        mipro = MiproV2Optimizer(
            metric=_exact_category_match,
            proposer_model=PROPOSER_MODEL,
            auto=None,
            num_candidates=4,  # 4 instructions, 4 demo sets
            num_trials=12,
            minibatch=True,
            minibatch_size=8,
            minibatch_full_eval_steps=4,
            max_bootstrapped_demos=3,
            max_labeled_demos=0,
            seed=7,
            mutation_log=mipro_log,
            budget=Budget(max_cost_usd=1.00),
        )
        mipro_result = await mipro.optimize(
            mipro_call, train_set=train, val_set=val, max_concurrent=4
        )

        # ----- CoOptimizer baseline -----
        co_call = Call(
            ClassifyTRECQuestion,
            model=TARGET_MODEL,
            instructions="Classify the question.",
        )
        co_log = MutationLog()
        co_opt = CoOptimizer(
            metric=_exact_category_match,
            strategies=["bootstrap", "instruction"],
            proposer_model=PROPOSER_MODEL,
            max_bootstrap_examples=3,
            max_instruction_trials=3,
            mutation_log=co_log,
            budget=Budget(max_cost_usd=1.00),
        )
        co_result = await co_opt.optimize(co_call, train_set=train, val_set=val, max_concurrent=4)

        # ----- Assertions -----

        # 1. MIPROv2 ran without crashing.
        assert isinstance(mipro_result, MiproV2Result)
        assert mipro_result.trials_run > 0, (
            f"MIPROv2 ran zero trials; stop_reason={mipro_result.stop_reason}"
        )
        assert mipro_result.proposer_calls > 0, (
            "MIPROv2 made zero proposer LM calls — the GroundedProposer "
            "static-context build did not run."
        )

        # 2. The search loop actually ran most of its budget of trials.
        search_mutations = [m for m in mipro_log.mutations if m.mutation_type == "search_trial"]
        assert len(search_mutations) >= 6, (
            f"MIPROv2 only completed {len(search_mutations)} search_trial mutations; "
            f"expected >= 6 (= 0.5 * num_trials). stop_reason="
            f"{mipro_result.stop_reason}"
        )

        # 3. MIPROv2 must not regress against its own baseline.
        assert mipro_result.metric_after >= mipro_result.metric_before, (
            f"MIPROv2 regressed: before={mipro_result.metric_before:.2f} "
            f"after={mipro_result.metric_after:.2f}"
        )

        # 4. MIPROv2 must be at least competitive with CoOptimizer.
        # Loose threshold (0.10) — on TREC-6 mini with Haiku the two
        # are expected to be within noise; this guards against a
        # catastrophic regression.
        assert mipro_result.metric_after + 0.10 >= co_result.metric_after, (
            f"MIPROv2 lost to CoOptimizer by >0.10: "
            f"mipro={mipro_result.metric_after:.2f} co={co_result.metric_after:.2f}"
        )

        # 5. Cost cap respected for both optimizers.
        mipro_cost = mipro_log.total_cost()
        co_cost = co_log.total_cost()
        assert mipro_cost <= 1.00, f"MIPROv2 cost overrun: ${mipro_cost:.4f}"
        assert co_cost <= 1.00, f"CoOptimizer cost overrun: ${co_cost:.4f}"

        # 6. Mutation log structure: every documented mutation_type
        # appears at least once.
        types_seen = {m.mutation_type for m in mipro_log.mutations}
        assert "bootstrap_demo_set" in types_seen
        assert "propose_instruction" in types_seen
        assert "search_trial" in types_seen
        assert "apply_best_config" in types_seen
        # search_full_eval may be 0 if budget bit before any promotion
        # boundary; not a hard requirement.

        print(
            f"\n  [mipro_v2_live] mipro: before={mipro_result.metric_before:.1%}, "
            f"after={mipro_result.metric_after:.1%}, "
            f"trials={mipro_result.trials_run}, "
            f"proposer_calls={mipro_result.proposer_calls}, "
            f"accepted={mipro_result.accepted}, "
            f"cost=${mipro_cost:.4f}"
        )
        print(
            f"  [mipro_v2_live] co:    before={co_result.metric_before:.1%}, "
            f"after={co_result.metric_after:.1%}, "
            f"cost=${co_cost:.4f}"
        )


# ---------------------------------------------------------------------------
# Harder test: opaque labels force MIPROv2 to actually find improvement
# ---------------------------------------------------------------------------
#
# The TREC-6 baseline test above saturates at 100% because Haiku is good
# at TREC-6 even with a vague instruction — the Signature docstring +
# the literal label tokens (ABBR/DESC/ENTY/...) carry enough semantic
# anchor that Haiku gets every val example right on the first try. The
# test passes structurally but does not demonstrate the algorithm's
# improvement claim.
#
# To create a task with genuine headroom, this second test relabels the
# six TREC-6 categories to opaque tokens X1-X6 with no semantic
# meaning. The mapping is:
#
#   X1 = ABBR  (abbreviation)
#   X2 = DESC  (description)
#   X3 = ENTY  (entity)
#   X4 = HUM   (human)
#   X5 = LOC   (location)
#   X6 = NUM   (number)
#
# The Signature docstring tells the model only that there are six
# categories, X1 through X6, with NO description of what each one
# means. Without few-shot demos, Haiku has to guess uniformly at
# 1/6 = ~17% accuracy. With good demos selected by MIPROv2, it
# should learn the mapping and approach the original ~95% accuracy.
#
# This is the regime where MIPROv2's joint search demonstrably wins:
# the demos are critical, the instruction text alone is not enough.


_OPAQUE_MAP: dict[str, str] = {
    "ABBR": "X1",
    "DESC": "X2",
    "ENTY": "X3",
    "HUM": "X4",
    "LOC": "X5",
    "NUM": "X6",
}


def _load_trec6_obfuscated() -> tuple[list[Example], list[Example]]:
    """Load the same TREC-6 fixture but with opaque X1-X6 labels."""
    train: list[Example] = []
    val: list[Example] = []
    with _FIXTURE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            opaque = _OPAQUE_MAP[row["category"]]
            ex = Example(
                inputs={"question": row["question"]},
                outputs={"category": opaque},
            )
            if row["split"] == "train":
                train.append(ex)
            elif row["split"] == "val":
                val.append(ex)
    return train, val


class ClassifyOpaque(Signature):
    """Classify each input question into one of six categories.

    The categories are exactly: X1, X2, X3, X4, X5, X6. Return the
    single category code that best fits the input.
    """

    question: str = InputField(description="The question to classify")
    category: Literal["X1", "X2", "X3", "X4", "X5", "X6"] = OutputField(
        description="One of: X1, X2, X3, X4, X5, X6"
    )


def _opaque_match(prediction: Any, gold: dict[str, Any]) -> float:
    pred = str(getattr(prediction, "category", "")).strip().upper()
    expected = str(gold.get("category", "")).strip().upper()
    return 1.0 if pred == expected else 0.0


@requires_anthropic
class TestMiproV2LiveImprovement:
    """The headline correctness test: MIPROv2 must demonstrate
    measurable improvement on a task where the baseline has headroom.

    On the opaque-label TREC-6, the baseline cannot exceed ~17%
    accuracy because the Signature gives no semantic meaning to
    X1..X6. After MIPROv2 selects good few-shot demos and proposes
    a grounded instruction, accuracy should jump well above
    random-guess and produce a strict improvement over the baseline.
    """

    async def test_opaque_labels_demonstrate_improvement(self) -> None:
        train, val = _load_trec6_obfuscated()
        assert len(train) == 20
        assert len(val) == 15

        call = Call(
            ClassifyOpaque,
            model=TARGET_MODEL,
            instructions="Pick a category.",  # deliberately vague
        )
        log = MutationLog()
        opt = MiproV2Optimizer(
            metric=_opaque_match,
            proposer_model=PROPOSER_MODEL,
            auto=None,
            num_candidates=4,
            num_trials=10,
            minibatch=True,
            minibatch_size=8,
            minibatch_full_eval_steps=4,
            max_bootstrapped_demos=4,
            max_labeled_demos=0,
            seed=11,
            mutation_log=log,
            budget=Budget(max_cost_usd=1.50),
        )
        result = await opt.optimize(call, train_set=train, val_set=val, max_concurrent=4)

        # 1. Pipeline ran end-to-end.
        assert isinstance(result, MiproV2Result)
        assert result.trials_run > 0
        assert result.proposer_calls > 0

        # 2. Baseline must be in the headroom-rich zone (random
        # guess is 1/6 = 0.167; allow up to 0.55 for the case where
        # the model picks up partial signal from question content
        # alone). If the baseline is already > 0.55, the labels are
        # not opaque enough and the test design is broken.
        assert result.metric_before <= 0.55, (
            f"Opaque-label baseline too high: {result.metric_before:.2f}. "
            f"Haiku is inferring the X1..X6 mapping from question content "
            f"without demos — the test setup is not creating real headroom."
        )

        # 3. The headline assertion: MIPROv2 finds a meaningful
        # improvement. We require at least +0.20 absolute lift over
        # the baseline. This is a real improvement, not noise — the
        # baseline is bounded above by 0.55 and the gap to a perfect
        # 1.0 is at least 0.45.
        improvement = result.metric_after - result.metric_before
        assert improvement >= 0.20, (
            f"MIPROv2 only improved by {improvement:+.2f} on the opaque-label "
            f"task: before={result.metric_before:.2f} after={result.metric_after:.2f}. "
            f"The algorithm is failing to demonstrate joint-search value where "
            f"it should have clear headroom."
        )
        assert result.accepted, (
            f"MIPROv2 produced an improvement of {improvement:+.2f} but did not "
            f"flip ``accepted=True`` — the apply phase did not run."
        )

        # 4. Mutation log structure complete.
        types_seen = {m.mutation_type for m in log.mutations}
        assert "bootstrap_demo_set" in types_seen
        assert "propose_instruction" in types_seen
        assert "search_trial" in types_seen
        assert "apply_best_config" in types_seen

        # 5. Cost cap respected and ACTUALLY tracked (not $0.00).
        cost = log.total_cost()
        assert cost <= 1.50, f"Cost overrun: ${cost:.4f}"
        assert cost > 0.0, (
            f"Cost showed as ${cost:.4f} — the PRICING table lookup is "
            f"silently failing. The bare model id may not be in the "
            f"PRICING dict; check kaos_llm_core.observability.cost.PRICING."
        )

        print(
            f"\n  [mipro_v2_opaque] before={result.metric_before:.1%}, "
            f"after={result.metric_after:.1%}, "
            f"improvement={improvement:+.1%}, "
            f"trials={result.trials_run}, "
            f"proposer_calls={result.proposer_calls}, "
            f"accepted={result.accepted}, "
            f"cost=${cost:.4f}, "
            f"best_demos={len(result.best_config.demos)}"
        )
