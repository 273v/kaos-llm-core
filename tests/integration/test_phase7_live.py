"""Live integration tests for kaos-llm-core Phase 7 (metrics + LLMJudge + analysis).

These hit real LLM provider APIs with **production models** for the LLMJudge
path and exercise the deterministic metric library on realistic legal data.
The analysis layer is tested against a mutation log produced by a real
optimizer run rather than a hand-built fixture, so the trial cards reflect
what an enterprise legal-tech user would actually see.

Run::

    uv run pytest tests/integration/test_phase7_live.py -v -m integration -s
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

from kaos_llm_core.metrics import (
    LLMJudge,
    accuracy,
    case_insensitive_match,
    contains,
    exact_match,
    json_field_match,
    normalized_match,
    numeric_close,
    precision_recall_f1,
    regex_match,
)
from kaos_llm_core.optimization.analysis import (
    load_mutations,
    make_trial_cards,
    strategy_contributions,
    summarize_run,
)
from kaos_llm_core.optimization.codec_optimizer import CodecOptimizer
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
requires_openai = pytest.mark.skipif(
    not _has_key("KAOS_LLM_OPENAI_API_KEY", "OPENAI_API_KEY"),
    reason="No OpenAI API key",
)


# ---------------------------------------------------------------------------
# Realistic legal data fixtures (small to keep tests fast)
# ---------------------------------------------------------------------------


class ClauseLabel(Signature):
    """You are a contracts attorney. Read the clause and return the single
    most accurate label from this set: indemnification, limitation_of_liability,
    confidentiality, termination, payment_terms, warranty, governing_law.
    Return only the label string in lowercase with underscores.
    """

    clause: str = InputField(description="The contract clause text to classify")
    label: str = OutputField(description="The clause label")


SHORT_TRAIN: list[Example] = [
    Example(
        inputs={"clause": "Each party shall indemnify the other from third-party claims."},
        outputs={"label": "indemnification"},
    ),
    Example(
        inputs={
            "clause": ("Aggregate liability shall not exceed fees paid in the prior twelve months.")
        },
        outputs={"label": "limitation_of_liability"},
    ),
    Example(
        inputs={
            "clause": ("Recipient shall not disclose Confidential Information to any third party.")
        },
        outputs={"label": "confidentiality"},
    ),
    Example(
        inputs={"clause": "Either party may terminate this Agreement upon thirty days notice."},
        outputs={"label": "termination"},
    ),
]


SHORT_VAL: list[Example] = [
    Example(
        inputs={
            "clause": ("Vendor agrees to defend and hold Customer harmless from any IP claims.")
        },
        outputs={"label": "indemnification"},
    ),
    Example(
        inputs={
            "clause": (
                "All technical and financial information disclosed "
                "shall be treated as confidential."
            )
        },
        outputs={"label": "confidentiality"},
    ),
    Example(
        inputs={
            "clause": "Customer shall pay all undisputed invoices within thirty days of receipt.",
        },
        outputs={"label": "payment_terms"},
    ),
]


# ---------------------------------------------------------------------------
# 1. Deterministic metrics (no API calls — fast smoke against real types)
# ---------------------------------------------------------------------------


class TestDeterministicMetricsLive:
    """The deterministic metrics need no API key but exercise the same code
    paths the live LLMJudge and optimizer-CLI tests will hit. They guard
    against accidental regressions in the public surface."""

    def test_exact_match_strict(self) -> None:
        assert exact_match("foo", "foo") == 1.0
        assert exact_match("foo", "Foo") == 0.0
        assert exact_match("foo ", "foo") == 0.0  # whitespace matters

    def test_case_insensitive_match(self) -> None:
        assert case_insensitive_match("Termination", "termination") == 1.0
        assert case_insensitive_match("TERMINATION", "termination") == 1.0
        assert case_insensitive_match("indemnification", "termination") == 0.0

    def test_normalized_match_strips_punctuation(self) -> None:
        assert normalized_match("termination.", "termination") == 1.0
        assert normalized_match('"Termination"', "termination") == 1.0
        assert normalized_match("  Termination  ", "termination") == 1.0

    def test_contains_factory(self) -> None:
        metric = contains("termination", case_insensitive=True)
        assert metric("the answer is termination", "ignored") == 1.0
        assert metric("indemnification clause", "ignored") == 0.0

    def test_regex_match_factory(self) -> None:
        metric = regex_match(r"^[a-z_]+$")
        assert metric("indemnification", "ignored") == 1.0
        assert metric("Indemnification!", "ignored") == 0.0

    def test_accuracy_alias(self) -> None:
        assert accuracy("foo", "foo") == 1.0
        assert accuracy("foo", "bar") == 0.0

    def test_json_field_match_factory(self) -> None:
        metric = json_field_match("label")
        assert metric({"label": "termination"}, {"label": "termination"}) == 1.0
        assert metric({"label": "indemnification"}, {"label": "termination"}) == 0.0

    def test_numeric_close_factory(self) -> None:
        metric = numeric_close(tolerance=0.05)
        assert metric(0.85, 0.84) == 1.0
        assert metric(0.85, 0.50) == 0.0

    def test_precision_recall_f1_returns_dict(self) -> None:
        # precision_recall_f1 returns a dict with all three values (factory design).
        # To plug into a single-float optimizer loop, wrap with
        # ``lambda p, g: prf(p, g)["f1"]``.
        metric = precision_recall_f1()
        score = metric(["a", "b", "c"], ["a", "b", "d"])
        assert isinstance(score, dict)
        # 2 tp, 1 fp, 1 fn -> precision=2/3, recall=2/3, f1=2/3
        assert 0.6 < score["f1"] < 0.7
        assert 0.6 < score["precision"] < 0.7
        assert 0.6 < score["recall"] < 0.7


# ---------------------------------------------------------------------------
# 2. LLMJudge against real models
# ---------------------------------------------------------------------------


class TestLLMJudgeLive:
    """LLMJudge metric — distinct from the Judge program. Returns a float."""

    @requires_anthropic
    async def test_helpfulness_rubric_anthropic_haiku(self) -> None:
        judge = LLMJudge(model="anthropic:claude-haiku-4-5", rubric="helpfulness")
        # A clearly correct answer should score high.
        good_score = await judge.acall(
            prediction="The capital of France is Paris.",
            gold="What is the capital of France?",
        )
        assert 0.0 <= good_score <= 1.0
        assert good_score >= 0.5, (
            f"Expected helpfulness >= 0.5 for a clearly correct answer, got {good_score}"
        )
        print(f"\n  [llm_judge_haiku helpfulness] score={good_score:.3f}")

    @requires_anthropic
    async def test_factuality_rubric_anthropic_sonnet(self) -> None:
        judge = LLMJudge(model="anthropic:claude-sonnet-4-6", rubric="factuality")
        score = await judge.acall(
            prediction=(
                "Delaware corporate law is governed by Title 8 of the Delaware Code, "
                "primarily the Delaware General Corporation Law."
            ),
            gold=("What body of law governs Delaware corporations?"),
        )
        assert 0.0 <= score <= 1.0
        print(f"\n  [llm_judge_sonnet factuality] score={score:.3f}")

    @requires_anthropic
    async def test_conciseness_rubric_clamped(self) -> None:
        """A score must always be clamped to [0, 1]."""
        judge = LLMJudge(model="anthropic:claude-haiku-4-5", rubric="conciseness")
        score = await judge.acall(
            prediction="Yes.",
            gold="Did the contract include an indemnification clause?",
        )
        assert 0.0 <= score <= 1.0
        print(f"\n  [llm_judge_haiku conciseness] score={score:.3f}")

    @requires_anthropic
    async def test_custom_rubric_passthrough(self) -> None:
        """A non-builtin rubric string is forwarded to the judge prompt verbatim."""
        judge = LLMJudge(
            model="anthropic:claude-haiku-4-5",
            rubric=(
                "Does the prediction correctly identify the contract clause type? "
                "Score 1.0 only if the label is exactly correct."
            ),
        )
        score = await judge.acall(
            prediction="indemnification",
            gold="indemnification",
        )
        assert score >= 0.5
        print(f"\n  [llm_judge custom_rubric] score={score:.3f}")


# ---------------------------------------------------------------------------
# 3. Analysis layer against a REAL optimizer-produced mutation log
# ---------------------------------------------------------------------------


class TestAnalysisLayerLive:
    """The analysis layer should consume real mutation log JSONL produced by
    a real optimizer run, not a hand-built fixture. We do the cheapest possible
    real run (CodecOptimizer with one example) and then point analysis at it.
    """

    @requires_anthropic
    async def test_load_real_optimizer_log_and_render(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "real_run.jsonl"
            log = MutationLog(path=log_path)

            # Run a real CodecOptimizer with one example. This produces 3
            # mutations (one per default codec) with non-zero cost from the
            # GAP-2 fix.
            def _label_match(prediction: Any, gold: dict[str, Any]) -> float:
                return (
                    1.0
                    if str(getattr(prediction, "label", "")).strip().lower()
                    == gold.get("label", "")
                    else 0.0
                )

            opt = CodecOptimizer(metric=_label_match, mutation_log=log)
            call = Call(ClauseLabel, model="anthropic:claude-haiku-4-5")
            result = await opt.optimize(call, SHORT_VAL[:1])
            print(
                f"\n  [analysis_live] codec optimizer ran: best={result.best_codec.__name__} "
                f"score={result.best_score:.3f}"
            )
            assert log_path.exists(), "MutationLog should have written the JSONL"

            # Now feed the freshly-produced log through the analysis layer.
            mutations = load_mutations(log_path)
            assert len(mutations) >= 1
            # Every mutation should carry GAP-1/4/6/7 fields from RunContext.
            for m in mutations:
                assert m.mutation_id, "GAP-1: every mutation must have a mutation_id"
                assert m.run_id is not None, "GAP-6: every mutation must have a run_id"
                assert m.trial_id >= 0, "GAP-7: every mutation must have a trial_id"
            # All mutations from the same run share the same run_id.
            run_ids = {m.run_id for m in mutations}
            assert len(run_ids) == 1, f"Expected one run_id, got {run_ids}"
            # Linked-list parent chain (every mutation after the first points
            # at the previous one).
            from itertools import pairwise

            for prev, curr in pairwise(mutations):
                assert curr.parent_mutation_id == prev.mutation_id, (
                    f"GAP-8: parent_mutation_id chain broken at trial {curr.trial_id}"
                )
            # GAP-2 verification at the analysis layer: cost numbers are real.
            cost_total = sum(m.cost_usd for m in mutations)
            tokens_total = sum(m.tokens_used for m in mutations)
            assert cost_total > 0.0, "Expected non-zero total cost from real optimizer run"
            assert tokens_total > 0, "Expected non-zero total tokens"
            print(
                f"  [analysis_live] {len(mutations)} mutations, "
                f"cost=${cost_total:.6f}, tokens={tokens_total}"
            )

            # Strategy contributions and trial cards render without crashing.
            cards = make_trial_cards(mutations)
            contribs = strategy_contributions(mutations)
            summary = summarize_run(mutations)

            assert len(cards) == len(mutations)
            # The first codec trial's before == after (best_codec starts at
            # the original codec), so its diff may be empty. Subsequent
            # trials have a real diff between codec names.
            assert any(c.diff for c in cards)
            assert summary["total_trials"] == len(mutations)
            assert summary["total_cost_usd"] == pytest.approx(cost_total)
            assert any(c.strategy == "codec_selection" for c in contribs)
