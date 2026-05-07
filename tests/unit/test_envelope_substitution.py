"""Substitution tests for the Phase 15.1 Program v3 envelope.

Exercises the JSON-pointer resolution helpers (``$.inputs.X``,
``$.steps.<id>.output.<field>``) directly without going through
``from_envelope``. This is the unit-level test for the substitution
contract documented in ``program-envelope-v3.md`` §"Substitution
syntax".
"""

from __future__ import annotations

import pytest

from kaos_llm_core.programs.envelope import (
    ProgramEnvelopeError,
    _resolve_output_mapping,
    _resolve_pointer,
    _resolve_step_inputs,
)

# ---------------------------------------------------------------------------
# Pointer resolution
# ---------------------------------------------------------------------------


class TestPointerResolution:
    def test_resolve_input_field(self) -> None:
        ctx = {"inputs": {"text": "hello world"}, "steps": {}}
        assert _resolve_pointer("$.inputs.text", ctx) == "hello world"

    def test_resolve_step_output_field(self) -> None:
        ctx = {
            "inputs": {},
            "steps": {"extract": {"output": {"clauses": ["a", "b"]}}},
        }
        assert _resolve_pointer("$.steps.extract.output.clauses", ctx) == ["a", "b"]

    def test_resolve_whole_step_output_dict(self) -> None:
        ctx = {
            "inputs": {},
            "steps": {"extract": {"output": {"x": 1, "y": 2}}},
        }
        assert _resolve_pointer("$.steps.extract.output", ctx) == {"x": 1, "y": 2}

    def test_resolve_nested_step_output(self) -> None:
        ctx = {
            "inputs": {},
            "steps": {"extract": {"output": {"meta": {"author": "Alice", "year": 2026}}}},
        }
        assert _resolve_pointer("$.steps.extract.output.meta.author", ctx) == "Alice"

    def test_resolve_pointer_passes_through_dict_values(self) -> None:
        """Whole dict / list values pass through by reference, not stringified."""
        ctx = {"inputs": {"items": [{"id": 1}, {"id": 2}]}, "steps": {}}
        result = _resolve_pointer("$.inputs.items", ctx)
        assert result == [{"id": 1}, {"id": 2}]
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Pointer error handling
# ---------------------------------------------------------------------------


class TestPointerErrors:
    def test_pointer_without_dollar_raises(self) -> None:
        with pytest.raises(ProgramEnvelopeError, match=r"\$\."):
            _resolve_pointer("inputs.text", {"inputs": {"text": "x"}, "steps": {}})

    def test_unknown_input_field_raises(self) -> None:
        ctx = {"inputs": {"text": "x"}, "steps": {}}
        with pytest.raises(ProgramEnvelopeError, match="not found"):
            _resolve_pointer("$.inputs.bogus", ctx)

    def test_unknown_step_id_raises(self) -> None:
        ctx = {"inputs": {}, "steps": {"first": {"output": {"x": 1}}}}
        with pytest.raises(ProgramEnvelopeError, match="not found"):
            _resolve_pointer("$.steps.second.output.x", ctx)

    def test_unknown_step_output_field_raises(self) -> None:
        ctx = {"inputs": {}, "steps": {"first": {"output": {"x": 1}}}}
        with pytest.raises(ProgramEnvelopeError, match="not found"):
            _resolve_pointer("$.steps.first.output.bogus", ctx)


# ---------------------------------------------------------------------------
# Step inputs resolution
# ---------------------------------------------------------------------------


class TestResolveStepInputs:
    def test_resolves_all_pointers(self) -> None:
        step_inputs = {
            "text": "$.inputs.text",
            "n": "$.inputs.count",
        }
        ctx = {"inputs": {"text": "hello", "count": 5}, "steps": {}}
        resolved = _resolve_step_inputs(step_inputs, ctx)
        assert resolved == {"text": "hello", "n": 5}

    def test_resolves_mixed_input_and_step_pointers(self) -> None:
        step_inputs = {
            "doc": "$.inputs.text",
            "clauses": "$.steps.extract.output.clauses",
        }
        ctx = {
            "inputs": {"text": "doc text"},
            "steps": {"extract": {"output": {"clauses": ["a", "b", "c"]}}},
        }
        resolved = _resolve_step_inputs(step_inputs, ctx)
        assert resolved == {"doc": "doc text", "clauses": ["a", "b", "c"]}

    def test_empty_step_inputs(self) -> None:
        assert _resolve_step_inputs({}, {"inputs": {}, "steps": {}}) == {}


# ---------------------------------------------------------------------------
# Output mapping resolution
# ---------------------------------------------------------------------------


class TestResolveOutputMapping:
    def test_resolves_top_level_output(self) -> None:
        output_mapping = {
            "summary": "$.steps.summarize.output.text",
            "input_echo": "$.inputs.text",
        }
        ctx = {
            "inputs": {"text": "original input"},
            "steps": {"summarize": {"output": {"text": "summary text"}}},
        }
        resolved = _resolve_output_mapping(output_mapping, ctx)
        assert resolved == {"summary": "summary text", "input_echo": "original input"}

    def test_empty_output_mapping(self) -> None:
        assert _resolve_output_mapping({}, {"inputs": {}, "steps": {}}) == {}

    def test_output_mapping_preserves_complex_types(self) -> None:
        output_mapping = {
            "items": "$.steps.extract.output.items",
            "metadata": "$.steps.extract.output.meta",
        }
        ctx = {
            "inputs": {},
            "steps": {
                "extract": {
                    "output": {
                        "items": [{"id": 1}, {"id": 2}],
                        "meta": {"author": "Alice"},
                    }
                }
            },
        }
        resolved = _resolve_output_mapping(output_mapping, ctx)
        assert resolved == {
            "items": [{"id": 1}, {"id": 2}],
            "metadata": {"author": "Alice"},
        }
