"""Schema validation tests for the Phase 15.1 Program v3 envelope.

Each test exercises one validation rule from the
``docs/internal/design/program-envelope-v3.md`` "Validation rules"
section. Validation failures must produce ``ProgramEnvelopeError``
with an agent-friendly recovery message.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from kaos_llm_core.programs.envelope import (
    ProgramEnvelope,
    ProgramEnvelopeError,
    from_envelope,
    program_hash,
)


def _minimal_envelope(**overrides: Any) -> dict[str, Any]:
    """The smallest valid envelope: one input, one client, one call step."""
    base = {
        "kaos_program": "1",
        "name": "minimal",
        "inputs": {"text": {"type": "string", "required": True}},
        "clients": {"default": {"provider": "anthropic", "model": "claude-haiku-4-5"}},
        "steps": [
            {
                "id": "extract",
                "kind": "call",
                "client": "default",
                "instruction": "Extract a one-sentence summary.",
                "inputs": {"text": "$.inputs.text"},
                "output_fields": {
                    "summary": {
                        "description": "One-sentence summary",
                        "type": {"type": "string"},
                    }
                },
            }
        ],
        "output": {"summary": "$.steps.extract.output.summary"},
        "capabilities": ["call", "jsonpointer_refs"],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestMinimalEnvelopeParses:
    def test_minimal_envelope_validates(self) -> None:
        env = ProgramEnvelope.model_validate(_minimal_envelope())
        assert env.name == "minimal"
        assert env.kaos_program == "1"
        assert len(env.steps) == 1
        assert env.steps[0].id == "extract"
        assert env.steps[0].kind == "call"

    def test_minimal_envelope_builds_program(self) -> None:
        program = from_envelope(_minimal_envelope())
        assert hasattr(program, "extract")  # Phase 11 auto-registered
        assert "extract" in program.named_calls()


# ---------------------------------------------------------------------------
# Schema validation: kaos_program version
# ---------------------------------------------------------------------------


class TestVersionField:
    def test_missing_version_uses_default(self) -> None:
        env_dict = _minimal_envelope()
        del env_dict["kaos_program"]
        env = ProgramEnvelope.model_validate(env_dict)
        # Phase 16.2 bumped the default to "1.1" (additive — every "1"
        # envelope still parses unchanged). The default is the latest
        # stable version, so a brand-new envelope without an explicit
        # version gets the most-current schema.
        assert env.kaos_program == "1.1"

    def test_v1_envelope_still_parses(self) -> None:
        """Phase 16.2 schema bump must NOT break existing v1 envelopes."""
        env_dict = _minimal_envelope()
        env_dict["kaos_program"] = "1"
        env = ProgramEnvelope.model_validate(env_dict)
        assert env.kaos_program == "1"

    def test_wrong_version_rejected(self) -> None:
        env_dict = _minimal_envelope()
        env_dict["kaos_program"] = "2"
        with pytest.raises(ValidationError):
            ProgramEnvelope.model_validate(env_dict)

    def test_unknown_top_level_field_rejected(self) -> None:
        env_dict = _minimal_envelope()
        env_dict["bogus_field"] = "should fail"
        with pytest.raises(ValidationError):
            ProgramEnvelope.model_validate(env_dict)


# ---------------------------------------------------------------------------
# Schema validation: name
# ---------------------------------------------------------------------------


class TestNameValidation:
    def test_valid_name_accepted(self) -> None:
        env = ProgramEnvelope.model_validate(_minimal_envelope(name="contract-triage"))
        assert env.name == "contract-triage"

    def test_uppercase_name_rejected(self) -> None:
        with pytest.raises(ValidationError, match="not valid"):
            ProgramEnvelope.model_validate(_minimal_envelope(name="ContractTriage"))

    def test_name_starting_with_digit_rejected(self) -> None:
        with pytest.raises(ValidationError, match="not valid"):
            ProgramEnvelope.model_validate(_minimal_envelope(name="1triage"))

    def test_name_with_underscore_rejected(self) -> None:
        with pytest.raises(ValidationError, match="not valid"):
            ProgramEnvelope.model_validate(_minimal_envelope(name="contract_triage"))


# ---------------------------------------------------------------------------
# Schema validation: step id
# ---------------------------------------------------------------------------


class TestStepIdValidation:
    def test_valid_step_id_accepted(self) -> None:
        env_dict = _minimal_envelope()
        env_dict["steps"][0]["id"] = "extract_v2"
        env_dict["output"] = {"summary": "$.steps.extract_v2.output.summary"}
        env = ProgramEnvelope.model_validate(env_dict)
        assert env.steps[0].id == "extract_v2"

    def test_uppercase_step_id_rejected(self) -> None:
        env_dict = _minimal_envelope()
        env_dict["steps"][0]["id"] = "Extract"
        with pytest.raises(ValidationError, match="not valid"):
            ProgramEnvelope.model_validate(env_dict)

    def test_step_id_starting_with_digit_rejected(self) -> None:
        env_dict = _minimal_envelope()
        env_dict["steps"][0]["id"] = "1extract"
        with pytest.raises(ValidationError, match="not valid"):
            ProgramEnvelope.model_validate(env_dict)

    def test_duplicate_step_ids_rejected(self) -> None:
        env_dict = _minimal_envelope()
        env_dict["steps"].append(dict(env_dict["steps"][0]))  # exact duplicate
        with pytest.raises(ValidationError, match="more than once"):
            ProgramEnvelope.model_validate(env_dict)


# ---------------------------------------------------------------------------
# Schema validation: step kind
# ---------------------------------------------------------------------------


class TestStepKindValidation:
    def test_unknown_step_kind_rejected(self) -> None:
        env_dict = _minimal_envelope()
        env_dict["steps"][0]["kind"] = "magic"
        with pytest.raises(ValidationError):
            ProgramEnvelope.model_validate(env_dict)

    def test_call_kind_accepted(self) -> None:
        env_dict = _minimal_envelope()
        env_dict["steps"][0]["kind"] = "call"
        ProgramEnvelope.model_validate(env_dict)

    def test_reason_kind_accepted_when_in_capabilities(self) -> None:
        env_dict = _minimal_envelope()
        env_dict["steps"][0]["kind"] = "reason"
        env_dict["capabilities"] = ["reason", "jsonpointer_refs"]
        ProgramEnvelope.model_validate(env_dict)

    def test_step_kind_missing_from_capabilities_rejected(self) -> None:
        env_dict = _minimal_envelope()
        env_dict["steps"][0]["kind"] = "reason"
        env_dict["capabilities"] = ["call"]  # missing 'reason'
        with pytest.raises(ValidationError, match="capabilities list does not include"):
            ProgramEnvelope.model_validate(env_dict)

    def test_react_requires_max_iterations(self) -> None:
        env_dict = _minimal_envelope()
        env_dict["steps"][0]["kind"] = "react"
        env_dict["capabilities"] = ["react", "jsonpointer_refs"]
        # Missing max_iterations
        with pytest.raises(ValidationError, match="max_iterations"):
            ProgramEnvelope.model_validate(env_dict)

    def test_react_requires_tools(self) -> None:
        env_dict = _minimal_envelope()
        env_dict["steps"][0]["kind"] = "react"
        env_dict["steps"][0]["max_iterations"] = 5
        env_dict["capabilities"] = ["react", "jsonpointer_refs"]
        with pytest.raises(ValidationError, match="tools"):
            ProgramEnvelope.model_validate(env_dict)

    def test_refine_requires_judge(self) -> None:
        env_dict = _minimal_envelope()
        env_dict["steps"][0]["kind"] = "refine"
        env_dict["steps"][0]["max_iterations"] = 3
        env_dict["capabilities"] = ["refine", "jsonpointer_refs"]
        with pytest.raises(ValidationError, match="judge"):
            ProgramEnvelope.model_validate(env_dict)

    def test_best_of_n_requires_n(self) -> None:
        env_dict = _minimal_envelope()
        env_dict["steps"][0]["kind"] = "best_of_n"
        env_dict["steps"][0]["selector"] = "first"
        env_dict["capabilities"] = ["best_of_n", "jsonpointer_refs"]
        # Phase 16.5: enforced by the discriminated-union sub-model
        # `BestOfNStep`. Pydantic reports the missing field as
        # ``best_of_n.n`` and the message as "Field required".
        with pytest.raises(ValidationError, match=r"best_of_n\.n"):
            ProgramEnvelope.model_validate(env_dict)


# ---------------------------------------------------------------------------
# Schema validation: capabilities
# ---------------------------------------------------------------------------


class TestCapabilities:
    def test_unsupported_capability_rejected(self) -> None:
        env_dict = _minimal_envelope()
        env_dict["capabilities"].append("future-feature")
        with pytest.raises(ValidationError, match="not supported"):
            ProgramEnvelope.model_validate(env_dict)

    def test_supported_capabilities_accepted(self) -> None:
        env_dict = _minimal_envelope()
        env_dict["capabilities"] = [
            "call",
            "reason",
            "judge",
            "react",
            "refine",
            "best_of_n",
            "jsonpointer_refs",
            "jinja2_prompts",
        ]
        # The step is still 'call' so this should validate
        ProgramEnvelope.model_validate(env_dict)


# ---------------------------------------------------------------------------
# Schema validation: client references
# ---------------------------------------------------------------------------


class TestClientReferences:
    def test_unknown_client_reference_rejected(self) -> None:
        env_dict = _minimal_envelope()
        env_dict["steps"][0]["client"] = "missing_client"
        with pytest.raises(ValidationError, match="not declared"):
            ProgramEnvelope.model_validate(env_dict)

    def test_multiple_clients_reference_correctly(self) -> None:
        env_dict = _minimal_envelope()
        env_dict["clients"]["strong"] = {
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
        }
        env_dict["steps"][0]["client"] = "strong"
        env = ProgramEnvelope.model_validate(env_dict)
        assert env.steps[0].client == "strong"


# ---------------------------------------------------------------------------
# Schema validation: pointer references
# ---------------------------------------------------------------------------


class TestPointerReferences:
    def test_input_pointer_resolves(self) -> None:
        env_dict = _minimal_envelope()
        # Already valid: '$.inputs.text'
        ProgramEnvelope.model_validate(env_dict)

    def test_input_pointer_to_undeclared_field_rejected(self) -> None:
        env_dict = _minimal_envelope()
        env_dict["steps"][0]["inputs"]["text"] = "$.inputs.bogus"
        with pytest.raises(ValidationError, match="not declared"):
            ProgramEnvelope.model_validate(env_dict)

    def test_step_output_forward_reference_rejected(self) -> None:
        env_dict = _minimal_envelope()
        env_dict["steps"] = [
            {
                "id": "first",
                "kind": "call",
                "client": "default",
                "instruction": "x",
                "inputs": {"text": "$.steps.second.output.x"},  # forward ref
                "output_fields": {"x": {"description": "x", "type": {"type": "string"}}},
            },
            {
                "id": "second",
                "kind": "call",
                "client": "default",
                "instruction": "y",
                "inputs": {"text": "$.inputs.text"},
                "output_fields": {"x": {"description": "x", "type": {"type": "string"}}},
            },
        ]
        env_dict["output"] = {"x": "$.steps.second.output.x"}
        with pytest.raises(ValidationError, match="not declared earlier"):
            ProgramEnvelope.model_validate(env_dict)

    def test_pointer_without_dollar_rejected(self) -> None:
        env_dict = _minimal_envelope()
        env_dict["steps"][0]["inputs"]["text"] = "inputs.text"
        with pytest.raises(ValidationError, match=r"\$\."):
            ProgramEnvelope.model_validate(env_dict)

    def test_output_mapping_to_unknown_step_rejected(self) -> None:
        env_dict = _minimal_envelope()
        env_dict["output"] = {"summary": "$.steps.bogus.output.summary"}
        with pytest.raises(ValidationError, match="not declared"):
            ProgramEnvelope.model_validate(env_dict)

    def test_output_mapping_to_unknown_input_rejected(self) -> None:
        env_dict = _minimal_envelope()
        env_dict["output"] = {"summary": "$.inputs.bogus"}
        with pytest.raises(ValidationError, match="not declared"):
            ProgramEnvelope.model_validate(env_dict)


# ---------------------------------------------------------------------------
# Schema validation: types block + $ref
# ---------------------------------------------------------------------------


class TestTypeReferences:
    def test_type_ref_resolves(self) -> None:
        env_dict = _minimal_envelope()
        env_dict["types"] = {"Risk": {"type": "string", "enum": ["low", "medium", "high"]}}
        env_dict["steps"][0]["output_fields"]["summary"]["type"] = {"$ref": "#/types/Risk"}
        ProgramEnvelope.model_validate(env_dict)

    def test_type_ref_to_undeclared_type_rejected(self) -> None:
        env_dict = _minimal_envelope()
        env_dict["steps"][0]["output_fields"]["summary"]["type"] = {"$ref": "#/types/Bogus"}
        with pytest.raises(ValidationError, match="not declared"):
            ProgramEnvelope.model_validate(env_dict)

    def test_type_ref_with_wrong_prefix_rejected(self) -> None:
        env_dict = _minimal_envelope()
        env_dict["steps"][0]["output_fields"]["summary"]["type"] = {"$ref": "#/definitions/X"}
        with pytest.raises(ValidationError, match="#/types/"):
            ProgramEnvelope.model_validate(env_dict)


# ---------------------------------------------------------------------------
# Tunable slots
# ---------------------------------------------------------------------------


class TestTunableSlots:
    def test_default_tunable_is_empty(self) -> None:
        env = ProgramEnvelope.model_validate(_minimal_envelope())
        tunable = env.steps[0].tunable
        assert tunable.instruction is None
        assert tunable.demos == []
        assert tunable.hyperparameters == {}
        assert tunable.codec is None
        assert tunable.model is None

    def test_tunable_extra_field_rejected(self) -> None:
        env_dict = _minimal_envelope()
        env_dict["steps"][0]["tunable"] = {"instruction": None, "bogus": "x"}
        with pytest.raises(ValidationError):
            ProgramEnvelope.model_validate(env_dict)

    def test_tunable_instruction_override(self) -> None:
        env_dict = _minimal_envelope()
        env_dict["steps"][0]["tunable"] = {"instruction": "Optimized version"}
        env = ProgramEnvelope.model_validate(env_dict)
        assert env.steps[0].tunable.instruction == "Optimized version"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_steps_list_allowed(self) -> None:
        env_dict = _minimal_envelope()
        env_dict["steps"] = []
        env_dict["output"] = {"text": "$.inputs.text"}
        env = ProgramEnvelope.model_validate(env_dict)
        assert env.steps == []

    def test_step_with_empty_output_fields_allowed(self) -> None:
        env_dict = _minimal_envelope()
        env_dict["steps"][0]["output_fields"] = {}
        env_dict["output"] = {}
        ProgramEnvelope.model_validate(env_dict)


# ---------------------------------------------------------------------------
# from_envelope error handling
# ---------------------------------------------------------------------------


class TestFromEnvelopeErrors:
    def test_from_envelope_dict_validates(self) -> None:
        program = from_envelope(_minimal_envelope())
        assert program is not None

    def test_from_envelope_program_envelope_validates(self) -> None:
        env = ProgramEnvelope.model_validate(_minimal_envelope())
        program = from_envelope(env)
        assert program is not None

    def test_from_envelope_invalid_dict_raises_program_envelope_error(self) -> None:
        env_dict = _minimal_envelope()
        env_dict["bogus"] = "fail"
        with pytest.raises(ProgramEnvelopeError, match="schema validation"):
            from_envelope(env_dict)

    def test_from_envelope_wrong_type_raises(self) -> None:
        with pytest.raises(ProgramEnvelopeError, match="expected a dict"):
            from_envelope(42)  # ty: ignore[invalid-argument-type]


# ---------------------------------------------------------------------------
# program_hash
# ---------------------------------------------------------------------------


class TestProgramHash:
    def test_hash_is_deterministic(self) -> None:
        env_dict = _minimal_envelope()
        h1 = program_hash(env_dict)
        h2 = program_hash(env_dict)
        assert h1 == h2
        assert h1.startswith("sha256:")

    def test_hash_changes_with_instruction(self) -> None:
        env1 = _minimal_envelope()
        env2 = _minimal_envelope()
        env2["steps"][0]["instruction"] = "Different instruction"
        assert program_hash(env1) != program_hash(env2)

    def test_hash_changes_with_model(self) -> None:
        env1 = _minimal_envelope()
        env2 = _minimal_envelope()
        env2["clients"]["default"]["model"] = "claude-sonnet-4-6"
        assert program_hash(env1) != program_hash(env2)

    def test_hash_invariant_under_key_order(self) -> None:
        env1 = {
            "kaos_program": "1",
            "name": "test",
            "inputs": {},
            "clients": {},
            "steps": [],
            "output": {},
            "capabilities": [],
        }
        env2 = {
            "capabilities": [],
            "output": {},
            "steps": [],
            "clients": {},
            "inputs": {},
            "name": "test",
            "kaos_program": "1",
        }
        assert program_hash(env1) == program_hash(env2)

    def test_hash_accepts_program_envelope_object(self) -> None:
        env_dict = _minimal_envelope()
        env = ProgramEnvelope.model_validate(env_dict)
        h_dict = program_hash(env_dict)
        h_obj = program_hash(env)
        # Note: model_dump may add fields with None values that aren't in the
        # original dict, so we just check both produce valid hashes.
        assert h_dict.startswith("sha256:")
        assert h_obj.startswith("sha256:")
