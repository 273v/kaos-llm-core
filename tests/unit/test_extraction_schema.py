"""Unit tests for ExtractionSchema — compilation + provenance wrapping.

Covers WS-TR.PR-2 acceptance:
- Three constructors (from_dict / from_pydantic / from_signature) produce
  equivalent schemas.
- ``to_signature(provenance=...)`` produces a Signature with the right field
  wrappers (bare / Cited[T] / GroundedAnswer[T]).
- Nested types (list[str], Optional[str]) compile correctly.
- Instruction text is embedded in the Signature docstring.
- Schema compiles + emits valid JSON Schema.
"""

from __future__ import annotations

import datetime
from typing import get_args, get_origin

import pytest
from pydantic import BaseModel, ValidationError

from kaos_llm_core.signatures.extraction import (
    ColumnSpec,
    ExtractionSchema,
)
from kaos_llm_core.signatures.fields import InputField, OutputField
from kaos_llm_core.signatures.grounding import Cited
from kaos_llm_core.signatures.signature import Signature


class TestColumnSpecValidation:
    def test_valid_snake_case_id(self) -> None:
        col = ColumnSpec(id="effective_date", column_type="date")
        assert col.id == "effective_date"

    def test_id_rejects_non_identifier(self) -> None:
        with pytest.raises(ValidationError):
            ColumnSpec(id="Effective-Date", column_type="date")
        with pytest.raises(ValidationError):
            ColumnSpec(id="1st_party", column_type="string")

    def test_id_rejects_empty(self) -> None:
        with pytest.raises(ValidationError):
            ColumnSpec(id="", column_type="string")


class TestExtractionSchemaFromDict:
    def test_minimal(self) -> None:
        schema = ExtractionSchema.from_dict(
            {
                "id": "contract-v1",
                "columns": [
                    {"id": "date", "column_type": "date"},
                ],
            }
        )
        assert schema.id == "contract-v1"
        assert schema.version == 1
        assert len(schema.columns) == 1

    def test_full(self) -> None:
        schema = ExtractionSchema.from_dict(
            {
                "id": "spa",
                "version": 3,
                "recipe_id": "spa-deal-points",
                "columns": [
                    {
                        "id": "purchase_price",
                        "column_type": "money",
                        "description": "Total purchase consideration.",
                        "required": True,
                    },
                    {
                        "id": "parties",
                        "column_type": "list",
                        "constraints": {"inner": "string"},
                    },
                ],
            }
        )
        assert schema.recipe_id == "spa-deal-points"
        assert schema.version == 3
        assert schema.columns[0].column_type == "money"
        assert schema.columns[1].constraints == {"inner": "string"}

    def test_empty_columns_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ExtractionSchema.from_dict({"id": "x", "columns": []})


class TestExtractionSchemaFromPydantic:
    def test_basic_types(self) -> None:
        class Contract(BaseModel):
            effective_date: datetime.date
            parties: list[str]
            governing_law: str
            total_value: float

        schema = ExtractionSchema.from_pydantic(Contract)
        assert schema.id == "Contract"
        by_id = {c.id: c for c in schema.columns}
        assert by_id["effective_date"].column_type == "date"
        assert by_id["parties"].column_type == "list"
        assert by_id["parties"].constraints["inner"] == "string"
        assert by_id["governing_law"].column_type == "string"
        assert by_id["total_value"].column_type == "number"

    def test_optional_field_inferred(self) -> None:
        class Contract(BaseModel):
            name: str
            termination: str | None = None

        schema = ExtractionSchema.from_pydantic(Contract)
        term_col = next(c for c in schema.columns if c.id == "termination")
        assert term_col.required is False
        assert term_col.column_type == "string"


class TestExtractionSchemaFromSignature:
    def test_discards_input_fields(self) -> None:
        class MySig(Signature):
            """Extract fields."""

            document: str = InputField(description="The document")
            name: str = OutputField(description="Name")
            amount: float = OutputField(description="Amount")

        schema = ExtractionSchema.from_signature(MySig)
        ids = [c.id for c in schema.columns]
        assert "document" not in ids
        assert "name" in ids
        assert "amount" in ids

    def test_signature_with_no_outputs_rejected(self) -> None:
        class MySig(Signature):
            """Only inputs."""

            document: str = InputField()

        with pytest.raises(ValueError, match="no OutputField"):
            ExtractionSchema.from_signature(MySig)


class TestToSignatureBareProvenance:
    def test_bare_types_preserved(self) -> None:
        schema = ExtractionSchema.from_dict(
            {
                "id": "x",
                "columns": [
                    {"id": "name", "column_type": "string"},
                    {"id": "age", "column_type": "integer"},
                    {"id": "active", "column_type": "boolean"},
                ],
            }
        )
        Sig = schema.to_signature(provenance="none")
        fields = Sig.model_fields
        assert fields["name"].annotation is str
        assert fields["age"].annotation is int
        assert fields["active"].annotation is bool

    def test_source_text_input_present(self) -> None:
        schema = ExtractionSchema.from_dict(
            {"id": "x", "columns": [{"id": "n", "column_type": "string"}]}
        )
        Sig = schema.to_signature(provenance="none")
        assert "source_text" in Sig.model_fields
        assert Sig.model_fields["source_text"].annotation is str


def _generic_args(cls: type) -> tuple:
    """Extract the type args a Pydantic PEP-695 generic was parametrized with.

    ``Cited[date]`` is a concrete subclass of ``Cited`` with
    ``__pydantic_generic_metadata__['args']`` holding ``(datetime.date,)``.
    ``get_args(Cited[date])`` returns ``()`` because Cited[date] is already
    a class, not a generic alias.
    """
    meta = getattr(cls, "__pydantic_generic_metadata__", {})
    return meta.get("args", ())


class TestToSignatureCitedProvenance:
    def test_each_output_wrapped_in_cited(self) -> None:
        schema = ExtractionSchema.from_dict(
            {
                "id": "x",
                "columns": [
                    {"id": "date", "column_type": "date"},
                    {"id": "parties", "column_type": "list", "constraints": {"inner": "string"}},
                ],
            }
        )
        Sig = schema.to_signature(provenance="cited")

        date_field = Sig.model_fields["date"].annotation
        assert issubclass(date_field, Cited)
        assert _generic_args(date_field)[0] is datetime.date

        parties_field = Sig.model_fields["parties"].annotation
        assert issubclass(parties_field, Cited)
        # Cited[list[str]] — inner is parametrized list.
        inner = _generic_args(parties_field)[0]
        assert get_origin(inner) is list
        assert get_args(inner)[0] is str

    def test_source_text_not_wrapped(self) -> None:
        schema = ExtractionSchema.from_dict(
            {"id": "x", "columns": [{"id": "n", "column_type": "string"}]}
        )
        Sig = schema.to_signature(provenance="cited")
        assert Sig.model_fields["source_text"].annotation is str


class TestToSignatureGroundedProvenance:
    def test_grounded_wraps_in_grounded_answer(self) -> None:
        schema = ExtractionSchema.from_dict(
            {"id": "x", "columns": [{"id": "answer", "column_type": "string"}]}
        )
        Sig = schema.to_signature(provenance="grounded")
        # ``GroundedAnswer`` is a PEP-695 ``type`` alias; Pydantic preserves
        # it on the field. ``get_origin`` returns the alias, ``get_args``
        # returns the ``T`` parameter (``str`` in this case).
        from kaos_llm_core.signatures.grounding import GroundedAnswer

        annot = Sig.model_fields["answer"].annotation
        assert get_origin(annot) is GroundedAnswer
        assert get_args(annot) == (str,)


class TestInstructionEmbedding:
    def test_instruction_in_docstring(self) -> None:
        schema = ExtractionSchema.from_dict(
            {
                "id": "my-schema",
                "columns": [
                    {"id": "name", "column_type": "string", "description": "The party's name."},
                ],
            }
        )
        Sig = schema.to_signature(provenance="cited")
        doc = Sig.__doc__ or ""
        assert "my-schema" in doc
        assert "name" in doc
        assert "The party's name." in doc

    def test_enum_values_mentioned(self) -> None:
        schema = ExtractionSchema.from_dict(
            {
                "id": "s",
                "columns": [
                    {
                        "id": "risk",
                        "column_type": "enum",
                        "constraints": {"values": ["low", "medium", "high"]},
                    }
                ],
            }
        )
        Sig = schema.to_signature(provenance="cited")
        doc = Sig.__doc__ or ""
        assert "low" in doc
        assert "medium" in doc
        assert "high" in doc


class TestJsonSchemaEmission:
    def test_schema_has_cited_defs(self) -> None:
        schema = ExtractionSchema.from_dict(
            {"id": "x", "columns": [{"id": "date", "column_type": "date"}]}
        )
        Sig = schema.to_signature(provenance="cited")
        js = Sig.model_json_schema()
        # Cited[date] lands as a $defs entry.
        assert "$defs" in js
        assert any("Cited" in k for k in js["$defs"])
        # Span is referenced from within Cited.
        assert "Span" in js["$defs"]

    def test_bare_schema_has_no_cited(self) -> None:
        schema = ExtractionSchema.from_dict(
            {"id": "x", "columns": [{"id": "n", "column_type": "string"}]}
        )
        Sig = schema.to_signature(provenance="none")
        js = Sig.model_json_schema()
        # No Cited in the schema — field is bare string.
        defs = js.get("$defs", {})
        assert not any("Cited" in k for k in defs)

    def test_multi_provenance_signatures_have_distinct_field_shapes(self) -> None:
        """Same schema, different provenance → different Signature field types."""
        base = ExtractionSchema.from_dict(
            {"id": "x", "columns": [{"id": "n", "column_type": "string"}]}
        )
        bare = base.to_signature(provenance="none")
        cited = base.to_signature(provenance="cited")
        grounded = base.to_signature(provenance="grounded")

        assert bare.model_fields["n"].annotation is str
        assert issubclass(cited.model_fields["n"].annotation, Cited)
        # Grounded is a discriminated union; distinct from both.
        assert cited.model_fields["n"].annotation is not grounded.model_fields["n"].annotation
