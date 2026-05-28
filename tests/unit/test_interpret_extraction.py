"""Pin the contract for ``InterpretExtractionSignature``.

The synthesizer Signature is the "interpret" half of the dynamic
deliverable-schema architecture. The "extract" half produces typed
grounded rows via ``design_schema`` + per-doc fan-out; this Signature
reformulates them into the user's desired deliverable (memo, table,
comparison) and signals when it needs more extraction.

These unit tests pin the field contract — names, types, defaults,
required vs optional — without exercising the live LLM ``Call``
path (which is covered by the kaos-agents
``kaos-agent-interpret-extraction`` integration test).
"""

from __future__ import annotations

from kaos_llm_core.programs import InterpretExtractionSignature
from kaos_llm_core.signatures.introspection import (
    get_input_fields,
    get_output_fields,
)


class TestSignatureShape:
    def test_re_exported_from_programs(self) -> None:
        """``InterpretExtractionSignature`` must be importable from
        ``kaos_llm_core.programs``."""
        from kaos_llm_core.programs import InterpretExtractionSignature as Sig

        assert Sig is InterpretExtractionSignature

    def test_re_exported_from_interpret_extraction_module(self) -> None:
        from kaos_llm_core.programs.interpret_extraction import (
            InterpretExtractionSignature as Sig,
        )

        assert Sig is InterpretExtractionSignature


class TestInputs:
    def test_input_field_names(self) -> None:
        names = set(get_input_fields(InterpretExtractionSignature).keys())
        assert names == {
            "user_question",
            "extracted_rows",
            "deliverable_hint",
            "iteration",
        }

    def test_user_question_is_required(self) -> None:
        fields = get_input_fields(InterpretExtractionSignature)
        assert fields["user_question"].is_required()

    def test_extracted_rows_is_required(self) -> None:
        fields = get_input_fields(InterpretExtractionSignature)
        assert fields["extracted_rows"].is_required()

    def test_deliverable_hint_defaults_to_empty(self) -> None:
        fields = get_input_fields(InterpretExtractionSignature)
        assert not fields["deliverable_hint"].is_required()
        assert fields["deliverable_hint"].default == ""

    def test_iteration_defaults_to_one(self) -> None:
        fields = get_input_fields(InterpretExtractionSignature)
        assert not fields["iteration"].is_required()
        assert fields["iteration"].default == 1


class TestOutputs:
    def test_output_field_names(self) -> None:
        names = set(get_output_fields(InterpretExtractionSignature).keys())
        assert names == {
            "memo",
            "score",
            "needs_more_extraction",
            "requested_columns",
        }

    def test_memo_is_required(self) -> None:
        fields = get_output_fields(InterpretExtractionSignature)
        assert fields["memo"].is_required()

    def test_score_defaults_to_seven(self) -> None:
        """A default keeps the Signature decodable when the LLM omits
        score for whatever reason. The midpoint default keeps downstream
        threshold logic conservative without biasing high or low."""
        fields = get_output_fields(InterpretExtractionSignature)
        assert not fields["score"].is_required()
        assert fields["score"].default == 7

    def test_needs_more_extraction_defaults_to_false(self) -> None:
        """Default ``false`` is the safe loop-stopping choice: callers
        that wire the loop must explicitly opt-in to continuing, never
        be surprised by an unbounded loop."""
        fields = get_output_fields(InterpretExtractionSignature)
        assert not fields["needs_more_extraction"].is_required()
        assert fields["needs_more_extraction"].default is False

    def test_requested_columns_defaults_to_empty_tuple(self) -> None:
        fields = get_output_fields(InterpretExtractionSignature)
        assert not fields["requested_columns"].is_required()
        assert fields["requested_columns"].default == ()


class TestPydanticOutputModel:
    """The output model must round-trip via the introspection helper —
    that's the contract the JSON codec uses to validate LLM responses."""

    def test_output_model_round_trips_with_minimum_payload(self) -> None:
        from kaos_llm_core.signatures.introspection import create_output_model

        OutModel = create_output_model(InterpretExtractionSignature)
        # Pydantic ``create_model`` returns a runtime BaseModel subclass;
        # ty can't statically narrow the dynamically-attached attributes,
        # so assert via ``model_dump()`` rather than attribute access.
        dumped = OutModel(memo="hello world").model_dump()
        assert dumped["memo"] == "hello world"
        assert dumped["score"] == 7
        assert dumped["needs_more_extraction"] is False
        assert dumped["requested_columns"] == ()

    def test_output_model_round_trips_with_full_payload(self) -> None:
        from kaos_llm_core.signatures.introspection import create_output_model

        OutModel = create_output_model(InterpretExtractionSignature)
        dumped = OutModel(
            memo="full memo body",
            score=9,
            needs_more_extraction=True,
            requested_columns=("governing_law: state whose law governs",),
        ).model_dump()
        assert dumped["score"] == 9
        assert dumped["needs_more_extraction"] is True
        assert dumped["requested_columns"] == ("governing_law: state whose law governs",)

    def test_score_bounds_one_to_ten(self) -> None:
        import pytest
        from pydantic import ValidationError

        from kaos_llm_core.signatures.introspection import create_output_model

        OutModel = create_output_model(InterpretExtractionSignature)
        with pytest.raises(ValidationError):
            OutModel(memo="x", score=0)
        with pytest.raises(ValidationError):
            OutModel(memo="x", score=11)


class TestInstruction:
    """The docstring becomes the system instruction shipped to the LLM.
    Pin the structural properties so a careless docstring edit doesn't
    silently degrade the LLM's grounding contract."""

    def test_docstring_mentions_grounding_constraint(self) -> None:
        from kaos_llm_core.signatures.introspection import get_instruction

        instr = get_instruction(InterpretExtractionSignature)
        # The bounded-by-extraction contract is the central rule.
        assert "EXTRACTED_ROWS" in instr
        assert "MUST" in instr or "must" in instr
        assert "trace" in instr.lower() or "cite" in instr.lower()

    def test_docstring_describes_iteration_signal(self) -> None:
        from kaos_llm_core.signatures.introspection import get_instruction

        instr = get_instruction(InterpretExtractionSignature)
        assert "needs_more_extraction" in instr
        assert "requested_columns" in instr
