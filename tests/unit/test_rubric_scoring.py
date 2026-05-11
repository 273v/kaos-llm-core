"""Unit tests for kaos_llm_core.signatures.rubric + .programs.scoring.

Deterministic-only — covers the hard channel of ``apply_rubric`` plus
the ``HybridRubric`` invariants and JSON round-trip. The soft-channel
(Judge-style) path is covered by the live integration tests since it
exercises a real LLM.
"""

from __future__ import annotations

import asyncio

import pytest
from kaos_content.model.tabular import Column, ColumnType, Table, TabularDocument

from kaos_llm_core.programs.scoring import (
    _evaluate_criterion,
    _hard_score,
    apply_rubric,
)
from kaos_llm_core.signatures import (
    Criterion,
    HybridRubric,
)

# ---------------------------------------------------------------------------
# HybridRubric — construction + invariants
# ---------------------------------------------------------------------------


class TestHybridRubricInvariants:
    def test_weights_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match=r"must equal 1\.0"):
            HybridRubric(
                objective="x",
                criteria=(Criterion(column="c", operator="truthy", weight=1.0),),
                hard_weight=0.5,
                soft_weight=0.4,
            )

    def test_no_criteria_no_guidance_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one of"):
            HybridRubric(
                objective="x",
                qualitative_guidance="",
                hard_weight=0.6,
                soft_weight=0.4,
            )

    def test_no_criteria_with_positive_hard_weight_rejected(self) -> None:
        with pytest.raises(ValueError, match="hard_weight="):
            HybridRubric(
                objective="x",
                qualitative_guidance="some guidance",
                hard_weight=0.6,
                soft_weight=0.4,
            )

    def test_no_guidance_with_positive_soft_weight_rejected(self) -> None:
        with pytest.raises(ValueError, match="soft_weight="):
            HybridRubric(
                objective="x",
                criteria=(Criterion(column="c", operator="truthy", weight=1.0),),
                qualitative_guidance="",
                hard_weight=0.6,
                soft_weight=0.4,
            )

    def test_pure_deterministic_mode(self) -> None:
        # All hard, no soft — valid.
        r = HybridRubric(
            objective="x",
            criteria=(Criterion(column="c", operator="truthy", weight=1.0),),
            qualitative_guidance="",
            hard_weight=1.0,
            soft_weight=0.0,
        )
        assert r.hard_weight == 1.0
        assert r.soft_weight == 0.0

    def test_pure_qualitative_mode(self) -> None:
        # All soft, no hard — valid.
        r = HybridRubric(
            objective="x",
            qualitative_guidance="some guidance",
            hard_weight=0.0,
            soft_weight=1.0,
        )
        assert r.hard_weight == 0.0
        assert r.soft_weight == 1.0


# ---------------------------------------------------------------------------
# JSON round-trip — agents emit rubrics via Call output; must be lossless
# ---------------------------------------------------------------------------


class TestJsonRoundTrip:
    def test_full_rubric_roundtrip(self) -> None:
        r = HybridRubric(
            objective="Pick the best NDA.",
            criteria=(
                Criterion(column="law", operator="eq", value="Michigan", weight=1.0),
                Criterion(column="term", operator="lte", value=2, weight=0.5),
                Criterion(
                    column="solicit",
                    operator="falsy",
                    value=None,
                    weight=0.5,
                    rationale="No non-solicit",
                ),
            ),
            qualitative_guidance="Penalize Delaware.",
            hard_weight=0.6,
            soft_weight=0.4,
        )
        encoded = r.model_dump_json()
        decoded = HybridRubric.model_validate_json(encoded)
        assert decoded == r

    def test_minimal_rubric_roundtrip(self) -> None:
        r = HybridRubric(
            objective="x",
            criteria=(Criterion(column="c", operator="truthy", weight=1.0),),
            qualitative_guidance="",
            hard_weight=1.0,
            soft_weight=0.0,
        )
        assert HybridRubric.model_validate_json(r.model_dump_json()) == r

    def test_unknown_operator_rejected_at_decode(self) -> None:
        bad_json = (
            '{"objective":"x","criteria":[{"column":"c","operator":"unknown",'
            '"value":null,"weight":1.0,"rationale":""}],'
            '"qualitative_guidance":"","hard_weight":1.0,"soft_weight":0.0}'
        )
        with pytest.raises(ValueError, match="operator"):
            HybridRubric.model_validate_json(bad_json)


# ---------------------------------------------------------------------------
# _evaluate_criterion — the dispatch table
# ---------------------------------------------------------------------------


class TestEvaluateCriterion:
    @pytest.mark.parametrize(
        ("op", "cell", "value", "expected"),
        [
            ("eq", "Michigan", "Michigan", True),
            ("eq", "Delaware", "Michigan", False),
            ("neq", "Delaware", "Michigan", True),
            ("lt", 2, 5, True),
            ("lte", 5, 5, True),
            ("gt", 10, 5, True),
            ("gte", 5, 5, True),
            ("in", "Michigan", ["Michigan", "Ohio"], True),
            ("in", "Delaware", ["Michigan", "Ohio"], False),
            ("not_in", "Delaware", ["Michigan", "Ohio"], True),
            ("contains", "Michigan, USA", "Michigan", True),
            ("truthy", "anything", None, True),
            ("truthy", "", None, False),
            ("falsy", "", None, True),
            ("falsy", "non-empty", None, False),
        ],
    )
    def test_dispatch(self, op: str, cell: object, value: object, expected: bool) -> None:
        c = Criterion(
            column="ignored",
            operator=op,  # ty: ignore[invalid-argument-type]
            value=value,
            weight=1.0,
        )
        assert _evaluate_criterion(c, cell) is expected

    def test_none_cell_only_matches_falsy_and_not_in(self) -> None:
        none_matchers = [
            ("falsy", None, True),
            ("not_in", ["a", "b"], True),
            ("eq", "x", False),
            ("lt", 5, False),
            ("contains", "x", False),
        ]
        for op, value, expected in none_matchers:
            c = Criterion(
                column="ignored",
                operator=op,  # ty: ignore[invalid-argument-type]
                value=value,
                weight=1.0,
            )
            assert _evaluate_criterion(c, None) is expected, op

    def test_type_mismatch_returns_false_silently(self) -> None:
        # Comparing a string to a number with `lt` would raise TypeError;
        # the dispatcher catches and returns False.
        c = Criterion(column="ignored", operator="lt", value=5, weight=1.0)
        assert _evaluate_criterion(c, "michigan") is False


# ---------------------------------------------------------------------------
# _hard_score — normalisation + edge cases
# ---------------------------------------------------------------------------


class TestHardScore:
    def test_all_match_returns_one(self) -> None:
        r = HybridRubric(
            objective="x",
            criteria=(
                Criterion(column="a", operator="eq", value=1, weight=1.0),
                Criterion(column="b", operator="eq", value=2, weight=1.0),
            ),
            qualitative_guidance="",
            hard_weight=1.0,
            soft_weight=0.0,
        )
        score, _ = _hard_score(r, {"a": 1, "b": 2})
        assert score == pytest.approx(1.0)

    def test_no_match_returns_neutral(self) -> None:
        # No criteria matched and no penalty (negative-weight) criteria
        # matched either → raw=0, normalised=0, remapped 0.5. The neutral
        # midpoint is the right answer when the rubric is silent — only
        # negative-weight criteria can push the score below 0.5.
        r = HybridRubric(
            objective="x",
            criteria=(
                Criterion(column="a", operator="eq", value=1, weight=1.0),
                Criterion(column="b", operator="eq", value=2, weight=1.0),
            ),
            qualitative_guidance="",
            hard_weight=1.0,
            soft_weight=0.0,
        )
        score, _ = _hard_score(r, {"a": 99, "b": 99})
        assert score == pytest.approx(0.5)

    def test_half_match_returns_half_plus_remap(self) -> None:
        # 1 of 2 criteria match → raw=1, max_mag=2 → normalised=0.5 →
        # remapped 0.75 (because raw/max=0.5 in [-1,1] is mapped to 0.75 in [0,1]).
        r = HybridRubric(
            objective="x",
            criteria=(
                Criterion(column="a", operator="eq", value=1, weight=1.0),
                Criterion(column="b", operator="eq", value=2, weight=1.0),
            ),
            qualitative_guidance="",
            hard_weight=1.0,
            soft_weight=0.0,
        )
        score, _ = _hard_score(r, {"a": 1, "b": 99})
        assert score == pytest.approx(0.75)

    def test_negative_weight_penalises(self) -> None:
        r = HybridRubric(
            objective="x",
            criteria=(Criterion(column="bad", operator="truthy", weight=-1.0),),
            qualitative_guidance="",
            hard_weight=1.0,
            soft_weight=0.0,
        )
        # When the penalty matches: raw=-1, max_mag=1, normalised=-1, remap=0.
        s_match, _ = _hard_score(r, {"bad": True})
        assert s_match == pytest.approx(0.0)
        # When the penalty doesn't match: raw=0, normalised=0, remap=0.5.
        s_skip, _ = _hard_score(r, {"bad": False})
        assert s_skip == pytest.approx(0.5)

    def test_missing_column_no_match_is_neutral(self) -> None:
        # Single positive-weight criterion, column missing → no match,
        # raw=0, remapped 0.5 (neutral). The reasoning string surfaces
        # the cell=None observation so audit can spot the misalignment.
        r = HybridRubric(
            objective="x",
            criteria=(Criterion(column="missing", operator="eq", value=1, weight=1.0),),
            qualitative_guidance="",
            hard_weight=1.0,
            soft_weight=0.0,
        )
        score, _ = _hard_score(r, {"other": 1})
        assert score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# apply_rubric — deterministic-only happy path
# ---------------------------------------------------------------------------


def _build_synthetic_table() -> TabularDocument:
    return TabularDocument(
        tables=(
            Table(
                name="contracts",
                columns=(
                    Column(name="doc_id", column_type=ColumnType.TEXT),
                    Column(name="law", column_type=ColumnType.TEXT),
                    Column(name="years", column_type=ColumnType.INTEGER, nullable=True),
                ),
                rows=(
                    ("a", "Michigan", 2),
                    ("b", "Delaware", 5),
                ),
                row_count=2,
            ),
        )
    )


class TestApplyRubric:
    def test_pure_deterministic_no_llm_calls(self) -> None:
        # When rubric is pure-hard (no qualitative_guidance), apply_rubric
        # must produce results without any LLM call — verified here by
        # the absence of an API key dependency (no env var read).
        r = HybridRubric(
            objective="prefer Michigan + short term",
            criteria=(
                Criterion(column="law", operator="eq", value="Michigan", weight=1.0),
                Criterion(column="years", operator="lte", value=2, weight=1.0),
            ),
            qualitative_guidance="",
            hard_weight=1.0,
            soft_weight=0.0,
        )
        doc = _build_synthetic_table()
        scored = asyncio.run(apply_rubric(r, doc))
        # 2 rows; original 3 cols + _score + _reasoning = 5 cols.
        t = scored.tables[0]
        assert len(t.columns) == 5
        assert t.columns[-2].name == "_score"
        assert t.columns[-1].name == "_reasoning"
        # Row "a" (Michigan, 2 yrs) matches both positive criteria → 1.0.
        # Row "b" (Delaware, 5 yrs) matches neither — raw=0, neutral 0.5.
        a_row = t.rows[0]
        b_row = t.rows[1]
        assert a_row[-2] == pytest.approx(1.0)
        assert b_row[-2] == pytest.approx(0.5)

    def test_empty_table_raises(self) -> None:
        empty = TabularDocument(tables=())
        with pytest.raises(ValueError, match="no tables"):
            asyncio.run(
                apply_rubric(
                    HybridRubric(
                        objective="x",
                        criteria=(Criterion(column="c", operator="truthy", weight=1.0),),
                        qualitative_guidance="",
                        hard_weight=1.0,
                        soft_weight=0.0,
                    ),
                    empty,
                )
            )
