"""Tests for :mod:`kaos_llm_core.labels` foundation types."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from kaos_llm_core.labels import ABSTAIN_LABEL, Label, LabelSet

# ---------------------------------------------------------------------------
# Label
# ---------------------------------------------------------------------------


class TestLabel:
    def test_minimal_label(self) -> None:
        label = Label(name="positive")
        assert label.name == "positive"
        assert label.description is None
        assert label.examples == []
        assert label.parent is None

    def test_full_label(self) -> None:
        label = Label(
            name="positive",
            description="A positive sentiment.",
            examples=["Great product!", "Loved it."],
            parent="sentiment",
        )
        assert label.description == "A positive sentiment."
        assert label.examples == ["Great product!", "Loved it."]
        assert label.parent == "sentiment"

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Label(name="")

    def test_prompt_text_falls_back_to_name(self) -> None:
        assert Label(name="x").prompt_text == "x"

    def test_prompt_text_uses_description_when_set(self) -> None:
        label = Label(name="x", description="A long description")
        assert label.prompt_text == "A long description"

    def test_label_is_json_serializable(self) -> None:
        label = Label(name="x", description="d", examples=["e"], parent="p")
        data = label.model_dump_json()
        round_trip = Label.model_validate_json(data)
        assert round_trip == label

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Label(name="x", weight=0.5)  # ty: ignore[unknown-argument]


# ---------------------------------------------------------------------------
# LabelSet construction
# ---------------------------------------------------------------------------


class TestLabelSetConstruction:
    def test_basic_set(self) -> None:
        ls = LabelSet(labels=[Label(name="a"), Label(name="b")])
        assert len(ls) == 2
        assert ls.exclusive is True
        assert ls.allow_abstain is True
        assert ls.hierarchical is False

    def test_empty_labels_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LabelSet(labels=[])

    def test_duplicate_names_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate label name"):
            LabelSet(labels=[Label(name="a"), Label(name="a")])

    def test_reserved_abstain_name_rejected(self) -> None:
        with pytest.raises(ValidationError, match="reserved"):
            LabelSet(labels=[Label(name=ABSTAIN_LABEL)])

    def test_hierarchical_unknown_parent_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unknown parent"):
            LabelSet(
                labels=[Label(name="leaf", parent="missing")],
                hierarchical=True,
            )

    def test_hierarchical_cycle_rejected(self) -> None:
        with pytest.raises(ValidationError, match="cycle"):
            LabelSet(
                labels=[
                    Label(name="a", parent="b"),
                    Label(name="b", parent="a"),
                ],
                hierarchical=True,
            )

    def test_hierarchical_self_cycle_rejected(self) -> None:
        with pytest.raises(ValidationError, match="cycle"):
            LabelSet(
                labels=[Label(name="a", parent="a")],
                hierarchical=True,
            )

    def test_flat_set_ignores_parent_field(self) -> None:
        ls = LabelSet(labels=[Label(name="leaf", parent="missing")], hierarchical=False)
        assert ls.labels[0].parent == "missing"


# ---------------------------------------------------------------------------
# LabelSet container behavior
# ---------------------------------------------------------------------------


class TestLabelSetContainer:
    def setup_method(self) -> None:
        self.ls = LabelSet(
            labels=[
                Label(name="a"),
                Label(name="b"),
                Label(name="c"),
            ]
        )

    def test_iter_yields_labels(self) -> None:
        names = [label.name for label in self.ls]
        assert names == ["a", "b", "c"]

    def test_len(self) -> None:
        assert len(self.ls) == 3

    def test_contains_by_name(self) -> None:
        assert "a" in self.ls
        assert "missing" not in self.ls

    def test_contains_rejects_non_string(self) -> None:
        assert 1 not in self.ls
        assert None not in self.ls

    def test_names_property(self) -> None:
        assert self.ls.names == ("a", "b", "c")

    def test_by_name(self) -> None:
        assert self.ls.by_name("a").name == "a"

    def test_by_name_missing_raises(self) -> None:
        with pytest.raises(KeyError):
            self.ls.by_name("missing")


class TestLabelSetHierarchy:
    def setup_method(self) -> None:
        self.ls = LabelSet(
            labels=[
                Label(name="sentiment"),
                Label(name="positive", parent="sentiment"),
                Label(name="negative", parent="sentiment"),
                Label(name="topic"),
                Label(name="sports", parent="topic"),
            ],
            hierarchical=True,
        )

    def test_roots(self) -> None:
        roots = self.ls.roots()
        assert {label.name for label in roots} == {"sentiment", "topic"}

    def test_children_of_root(self) -> None:
        children = self.ls.children("sentiment")
        assert {label.name for label in children} == {"positive", "negative"}

    def test_children_of_leaf_is_empty(self) -> None:
        assert self.ls.children("positive") == ()

    def test_children_of_none_returns_roots(self) -> None:
        assert self.ls.roots() == self.ls.children(None)


class TestLabelSetPickValidation:
    def setup_method(self) -> None:
        self.ls = LabelSet(labels=[Label(name="a"), Label(name="b"), Label(name="c")])

    def test_validate_picks_filters_unknown(self) -> None:
        assert self.ls.validate_picks(["a", "x", "b"]) == ["a", "b"]

    def test_validate_picks_preserves_order(self) -> None:
        assert self.ls.validate_picks(["c", "a"]) == ["c", "a"]

    def test_validate_picks_empty(self) -> None:
        assert self.ls.validate_picks([]) == []

    def test_assert_picks_raises_on_unknown(self) -> None:
        with pytest.raises(KeyError, match="unknown labels"):
            self.ls.assert_picks(["a", "x"])

    def test_assert_picks_accepts_all_known(self) -> None:
        assert self.ls.assert_picks(["a", "b"]) == ["a", "b"]


# ---------------------------------------------------------------------------
# LabelSet constructors / serialization
# ---------------------------------------------------------------------------


class TestLabelSetFromNames:
    def test_basic(self) -> None:
        ls = LabelSet.from_names(["a", "b", "c"])
        assert ls.names == ("a", "b", "c")
        assert ls.exclusive is True
        assert ls.allow_abstain is True
        assert ls.hierarchical is False

    def test_with_flags(self) -> None:
        ls = LabelSet.from_names(["a", "b"], exclusive=False, allow_abstain=False)
        assert ls.exclusive is False
        assert ls.allow_abstain is False


class TestLabelSetSerialization:
    def test_json_round_trip(self) -> None:
        original = LabelSet(
            labels=[
                Label(name="a", description="alpha", examples=["e1"]),
                Label(name="b"),
            ],
            exclusive=False,
        )
        data = original.model_dump_json()
        round_trip = LabelSet.model_validate_json(data)
        assert round_trip == original

    def test_hierarchical_round_trip(self) -> None:
        original = LabelSet(
            labels=[
                Label(name="parent"),
                Label(name="child", parent="parent"),
            ],
            hierarchical=True,
        )
        round_trip = LabelSet.model_validate_json(original.model_dump_json())
        assert round_trip == original

    def test_model_dump_is_jsonable(self) -> None:
        ls = LabelSet.from_names(["a", "b"])
        # Should not raise.
        json.dumps(ls.model_dump())


# ---------------------------------------------------------------------------
# Public API exposure
# ---------------------------------------------------------------------------


def test_labels_exposed_from_top_level_module() -> None:
    import kaos_llm_core

    assert kaos_llm_core.Label is Label
    assert kaos_llm_core.LabelSet is LabelSet
    assert kaos_llm_core.ABSTAIN_LABEL == ABSTAIN_LABEL
    assert "Label" in kaos_llm_core.__all__
    assert "LabelSet" in kaos_llm_core.__all__
    assert "ABSTAIN_LABEL" in kaos_llm_core.__all__
