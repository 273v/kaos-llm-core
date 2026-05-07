"""Unit tests for the kaos_llm_core.metrics package — text and structured."""

from __future__ import annotations

import pytest

from kaos_llm_core.metrics import (
    accuracy,
    case_insensitive_match,
    contains,
    exact_match,
    json_field_match,
    normalized_match,
    numeric_close,
    numeric_ratio,
    precision_recall_f1,
    regex_match,
)


class TestExactMatch:
    def test_happy_path(self) -> None:
        assert exact_match("hello", "hello") == 1.0
        assert exact_match("hello", "world") == 0.0

    def test_none_inputs(self) -> None:
        assert exact_match(None, "x") == 0.0
        assert exact_match("x", None) == 0.0
        assert exact_match(None, None) == 0.0

    def test_dict_unwrap(self) -> None:
        # gold may be dict from Example.outputs
        assert exact_match("hello", {"answer": "hello"}) == 1.0
        assert exact_match({"answer": "x"}, {"answer": "x"}) == 1.0


class TestCaseInsensitiveMatch:
    def test_happy_path(self) -> None:
        assert case_insensitive_match("Hello", "hello") == 1.0
        assert case_insensitive_match("HELLO WORLD", "hello world") == 1.0

    def test_unicode(self) -> None:
        assert case_insensitive_match("ÉCOLE", "école") == 1.0


class TestNormalizedMatch:
    def test_whitespace_collapse(self) -> None:
        assert normalized_match("  hello   world  ", "hello world") == 1.0
        assert normalized_match("Hello\tworld", "hello world") == 1.0

    def test_mismatch(self) -> None:
        assert normalized_match("hello world", "hello there") == 0.0


class TestRegexMatch:
    def test_happy_path(self) -> None:
        m = regex_match(r"\d{3}-\d{4}")
        assert m("555-1234", None) == 1.0
        assert m("not a phone", None) == 0.0

    def test_invalid_pattern(self) -> None:
        with pytest.raises(ValueError):
            regex_match(r"(unclosed")

    def test_none_input(self) -> None:
        m = regex_match(r".*")
        assert m(None, None) == 0.0


class TestContains:
    def test_happy_path(self) -> None:
        m = contains("foo")
        assert m("hello foo bar", None) == 1.0
        assert m("nope", None) == 0.0

    def test_case_insensitive(self) -> None:
        m = contains("FOO", case_insensitive=True)
        assert m("hello foo bar", None) == 1.0

    def test_invalid_substring(self) -> None:
        with pytest.raises(TypeError):
            contains(123)  # ty: ignore[invalid-argument-type]


class TestAccuracy:
    def test_alias(self) -> None:
        assert accuracy("a", "a") == 1.0
        assert accuracy("a", "b") == 0.0


class TestJsonFieldMatch:
    def test_happy_path(self) -> None:
        m = json_field_match("name")
        assert m({"name": "Alice"}, {"name": "Alice"}) == 1.0
        assert m({"name": "Alice"}, {"name": "Bob"}) == 0.0

    def test_missing_field(self) -> None:
        m = json_field_match("name")
        assert m({"other": "x"}, {"name": "Alice"}) == 0.0
        assert m({"name": "Alice"}, {"other": "x"}) == 0.0

    def test_attr_object(self) -> None:
        class Obj:
            name = "Alice"

        m = json_field_match("name")
        assert m(Obj(), {"name": "Alice"}) == 1.0

    def test_invalid_field(self) -> None:
        with pytest.raises(TypeError):
            json_field_match("")


class TestNumericClose:
    def test_within_tolerance(self) -> None:
        m = numeric_close(0.01)
        assert m(1.0, 1.005) == 1.0
        assert m(1.0, 2.0) == 0.0

    def test_default_tolerance(self) -> None:
        m = numeric_close()
        assert m(1.0, 1.0) == 1.0
        assert m(1.0, 1.01) == 0.0  # 0.01 > 1e-6 default

    def test_non_numeric(self) -> None:
        m = numeric_close()
        assert m("not a number", 1.0) == 0.0

    def test_negative_tolerance(self) -> None:
        with pytest.raises(ValueError):
            numeric_close(-1.0)


class TestNumericRatio:
    def test_perfect(self) -> None:
        assert numeric_ratio(5.0, 5.0) == 1.0

    def test_off_by_some(self) -> None:
        # |3-5|/max(5,1)=0.4 → 0.6
        assert numeric_ratio(3.0, 5.0) == pytest.approx(0.6)

    def test_clamped_to_zero(self) -> None:
        # |100-1|/max(1,1)=99 → clamps to 1.0 → score = 0.0
        assert numeric_ratio(100.0, 1.0) == 0.0

    def test_none(self) -> None:
        assert numeric_ratio(None, 1.0) == 0.0


class TestPrecisionRecallF1:
    def test_set_mode_perfect(self) -> None:
        m = precision_recall_f1()
        result = m(["a", "b", "c"], ["a", "b", "c"])
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_set_mode_partial(self) -> None:
        m = precision_recall_f1()
        # pred={a,b}, gold={a,c} → tp=1, fp=1, fn=1
        result = m(["a", "b"], ["a", "c"])
        assert result["precision"] == 0.5
        assert result["recall"] == 0.5
        assert result["f1"] == 0.5

    def test_multiset_counts_repetitions(self) -> None:
        m_set = precision_recall_f1(mode="set")
        m_multi = precision_recall_f1(mode="multiset")
        # set: pred={a}, gold={a} → 1.0; multiset: pred=2*a, gold=1*a → tp=1, fp=1
        assert m_set(["a", "a"], ["a"])["f1"] == 1.0
        multi_result = m_multi(["a", "a"], ["a"])
        assert multi_result["precision"] == 0.5
        assert multi_result["recall"] == 1.0

    def test_empty_inputs(self) -> None:
        m = precision_recall_f1()
        result = m([], [])
        assert result == {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def test_invalid_mode(self) -> None:
        with pytest.raises(ValueError):
            precision_recall_f1(mode="bogus")  # ty: ignore[invalid-argument-type]

    def test_none_inputs(self) -> None:
        m = precision_recall_f1()
        assert m(None, ["a"])["f1"] == 0.0
