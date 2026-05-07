"""Tests for kaos_llm_core.signatures — Signature, InputField, OutputField, introspection."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from kaos_llm_core.errors import SignatureError
from kaos_llm_core.signatures import (
    InputField,
    OutputField,
    Signature,
    create_output_model,
    get_input_fields,
    get_instruction,
    get_output_fields,
    signature_to_json_schema,
)

# --- Test Signatures ---


class SimpleExtract(Signature):
    """Extract entities from text."""

    text: str = InputField(description="Input text")
    entities: list[str] = OutputField(description="Extracted entity names")


class MultiField(Signature):
    """Classify and score a document."""

    text: str = InputField(description="Document text")
    category: str = InputField(description="Category hint", default="general")
    label: str = OutputField(description="Classification label")
    confidence: float = OutputField(description="Confidence score 0-1")


class NestedOutput(Signature):
    """Extract structured entities."""

    text: str = InputField(description="Source text")
    entities: list[dict[str, str]] = OutputField(description="List of entity dicts")


class NoOutputs(Signature):
    """A signature with no output fields (invalid)."""

    text: str = InputField(description="Input text")


class NoDocstring(Signature):
    text: str = InputField(description="Input text")
    result: str = OutputField(description="Result")


# --- Tests ---


class TestInputField:
    def test_basic(self) -> None:
        fields = get_input_fields(SimpleExtract)
        assert "text" in fields
        assert fields["text"].description == "Input text"

    def test_with_default(self) -> None:
        fields = get_input_fields(MultiField)
        assert "category" in fields
        assert not fields["category"].is_required()

    def test_required(self) -> None:
        fields = get_input_fields(SimpleExtract)
        assert fields["text"].is_required()


class TestOutputField:
    def test_basic(self) -> None:
        fields = get_output_fields(SimpleExtract)
        assert "entities" in fields
        assert fields["entities"].description == "Extracted entity names"

    def test_multiple_outputs(self) -> None:
        fields = get_output_fields(MultiField)
        assert "label" in fields
        assert "confidence" in fields
        assert len(fields) == 2

    def test_no_outputs_raises(self) -> None:
        with pytest.raises(SignatureError, match="no OutputFields"):
            get_output_fields(NoOutputs)


class TestFieldClassification:
    def test_inputs_exclude_outputs(self) -> None:
        inputs = get_input_fields(MultiField)
        assert "label" not in inputs
        assert "confidence" not in inputs

    def test_outputs_exclude_inputs(self) -> None:
        outputs = get_output_fields(MultiField)
        assert "text" not in outputs
        assert "category" not in outputs


class TestGetInstruction:
    def test_docstring(self) -> None:
        instruction = get_instruction(SimpleExtract)
        assert instruction == "Extract entities from text."

    def test_multiline_docstring(self) -> None:
        instruction = get_instruction(MultiField)
        assert instruction == "Classify and score a document."

    def test_no_docstring(self) -> None:
        instruction = get_instruction(NoDocstring)
        assert instruction == ""


class TestSignatureToJsonSchema:
    def test_simple_schema(self) -> None:
        schema = signature_to_json_schema(SimpleExtract)
        assert schema["type"] == "object"
        assert "entities" in schema["properties"]

    def test_multi_field_schema(self) -> None:
        schema = signature_to_json_schema(MultiField)
        assert "label" in schema["properties"]
        assert "confidence" in schema["properties"]
        # Input fields should NOT appear in output schema
        assert "text" not in schema["properties"]
        assert "category" not in schema["properties"]


class TestCreateOutputModel:
    def test_simple(self) -> None:
        model = create_output_model(SimpleExtract)
        assert model.__name__ == "SimpleExtractOutput"
        instance = model.model_validate({"entities": ["SEC", "Acme"]})
        assert instance.entities == ["SEC", "Acme"]  # ty: ignore[unresolved-attribute]

    def test_multi_field(self) -> None:
        model = create_output_model(MultiField)
        instance = model.model_validate({"label": "legal", "confidence": 0.95})
        assert instance.label == "legal"  # ty: ignore[unresolved-attribute]
        assert instance.confidence == 0.95  # ty: ignore[unresolved-attribute]

    def test_validation_error(self) -> None:
        model = create_output_model(MultiField)
        with pytest.raises(Exception):  # noqa: B017
            model.model_validate({"label": "legal"})  # missing confidence


class TestSignatureConstruction:
    def test_signature_is_pydantic_model(self) -> None:
        assert issubclass(SimpleExtract, BaseModel)

    def test_forbids_extra_fields(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            SimpleExtract(text="hello", entities=["a"], extra_field="bad")  # ty: ignore[unknown-argument]
