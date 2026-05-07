"""Tests for kaos_llm_core.types — Example dataclass."""

from __future__ import annotations

from kaos_llm_core.types import Example


class TestExample:
    """Tests for the Example dataclass."""

    def test_construction(self) -> None:
        ex = Example(
            inputs={"text": "hello world"},
            outputs={"entities": ["hello", "world"]},
        )
        assert ex.inputs == {"text": "hello world"}
        assert ex.outputs == {"entities": ["hello", "world"]}
        assert ex.metadata == {}

    def test_construction_with_metadata(self) -> None:
        ex = Example(
            inputs={"text": "test"},
            outputs={"result": "ok"},
            metadata={"source": "manual", "quality": 0.9},
        )
        assert ex.metadata["source"] == "manual"
        assert ex.metadata["quality"] == 0.9

    def test_to_dict(self) -> None:
        ex = Example(
            inputs={"a": 1},
            outputs={"b": 2},
            metadata={"tag": "test"},
        )
        d = ex.model_dump()
        assert d == {"inputs": {"a": 1}, "outputs": {"b": 2}, "metadata": {"tag": "test"}}

    def test_to_dict_empty_metadata(self) -> None:
        ex = Example(inputs={"a": 1}, outputs={"b": 2})
        d = ex.model_dump()
        assert d["metadata"] == {}  # present but empty

    def test_from_dict(self) -> None:
        d = {"inputs": {"text": "hi"}, "outputs": {"result": "ok"}, "metadata": {"k": "v"}}
        ex = Example.model_validate(d)
        assert ex.inputs == {"text": "hi"}
        assert ex.outputs == {"result": "ok"}
        assert ex.metadata == {"k": "v"}

    def test_from_dict_no_metadata(self) -> None:
        d = {"inputs": {"x": 1}, "outputs": {"y": 2}}
        ex = Example.model_validate(d)
        assert ex.metadata == {}

    def test_round_trip(self) -> None:
        original = Example(
            inputs={"text": "legal doc", "jurisdiction": "US"},
            outputs={"entities": [{"name": "SEC", "type": "ORG"}]},
            metadata={"confidence": 0.95},
        )
        restored = Example.model_validate(original.model_dump())
        assert restored.inputs == original.inputs
        assert restored.outputs == original.outputs
        assert restored.metadata == original.metadata

    def test_equality(self) -> None:
        a = Example(inputs={"x": 1}, outputs={"y": 2})
        b = Example(inputs={"x": 1}, outputs={"y": 2})
        assert a == b

    def test_inequality(self) -> None:
        a = Example(inputs={"x": 1}, outputs={"y": 2})
        b = Example(inputs={"x": 1}, outputs={"y": 3})
        assert a != b
