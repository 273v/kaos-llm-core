"""Pydantic models for the Program v3 envelope.

Splits the schema models out of the monolithic envelope.py for
navigability. Cross-step validation lives in
:mod:`.validation`; this module imports it lazily inside
``ProgramEnvelope._validate_envelope`` to avoid an import cycle.

Phase 16.5: ``EnvelopeStep`` is a discriminated union of per-kind
sub-models (CallStep, ReasonStep, JudgeStep, ReActStep, RefineStep,
BestOfNStep, MultiChainComparisonStep, ProgramOfThoughtStep). Each
sub-model declares only the fields its kind needs. Pydantic enforces
required fields at parse time per discriminator value, so the
pre-Phase-16.5 ``_validate_step_kind_fields`` cross-step helper is
no longer needed. The field constraints (``n >= 2`` for
multi_chain_comparison, ``allow_code_execution=True`` opt-in for
program_of_thought, etc.) live on the per-kind classes themselves.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kaos_llm_core.programs.envelope._constants import (
    _NAME_RE,
    _STEP_ID_RE,
    SUPPORTED_CAPABILITIES,
)


class InputSpec(BaseModel):
    """Top-level program input field declaration.

    Each entry in the envelope's ``inputs`` block is one of these,
    keyed by field name. ``type`` is a JSON Schema fragment string
    (``string``, ``integer``, ``number``, ``boolean``, ``object``,
    ``array``) — complex types use a ``$ref`` into the hoisted
    ``types`` block.
    """

    model_config = ConfigDict(extra="forbid")

    type: str = "string"
    description: str | None = None
    required: bool = True
    default: Any = None


class ClientSpec(BaseModel):
    """Symbolic LLM client identifier.

    The envelope's top-level ``clients`` block holds these by name;
    steps reference clients by name (``"client": "default"``). The
    actual provider connection resolves from
    :class:`KaosLLMSettings` at execution time — credentials are
    never in the envelope.
    """

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    kwargs: dict[str, Any] = Field(default_factory=dict)

    @property
    def model_string(self) -> str:
        """Return the kaos-llm-client ``provider:model`` identifier."""
        return f"{self.provider}:{self.model}"


class Tunable(BaseModel):
    """Optimizer-writable slots on every step.

    Author fields stay separate from optimizer-written fields.
    A v1 envelope leaves these null/empty; future Phase 16
    optimizers populate them in place without touching author
    fields. Direct lift from DSPy's JSON save format — the only
    mature answer to "what happens when an optimizer changes the
    program."
    """

    model_config = ConfigDict(extra="forbid")

    instruction: str | None = None  # null = use the step's author instruction
    demos: list[dict[str, Any]] = Field(default_factory=list)  # few-shot examples
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    codec: Literal["json", "chat", "xml"] | None = None
    model: str | None = None  # overrides the step's client.model when set


class JudgeSpec(BaseModel):
    """Sub-spec for the critic inside a ``refine`` step."""

    model_config = ConfigDict(extra="forbid")

    client: str
    instruction: str
    min_score: float = 0.7
    criteria: str | None = None


class ToolSpec(BaseModel):
    """Declarative tool spec for a ``react`` step.

    Phase 15.1 supports two tool kinds:

    - ``builtin``: an allowlisted pure-Python tool that ships with
      kaos-llm-core (e.g. ``math.eval``, ``regex.match``).
    - ``mcp``: a proxy to another KAOS MCP tool (resolved by name
      against the runtime's tool registry at execution time).

    User-defined Python tools remain Python-API-only and are
    explicitly out of scope for v3 envelopes.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    kind: Literal["builtin", "mcp"]
    builtin: str | None = None  # required iff kind == "builtin"
    mcp_tool: str | None = None  # required iff kind == "mcp"

    @model_validator(mode="after")
    def _check_kind_specific_fields(self) -> ToolSpec:
        if self.kind == "builtin" and not self.builtin:
            msg = (
                f"ToolSpec({self.name!r}): kind='builtin' requires the 'builtin' field "
                "to name an allowlisted built-in tool. "
                "Available built-ins: math.eval, regex.match, string.length."
            )
            raise ValueError(msg)
        if self.kind == "mcp" and not self.mcp_tool:
            msg = (
                f"ToolSpec({self.name!r}): kind='mcp' requires the 'mcp_tool' field "
                "to name a registered KAOS MCP tool."
            )
            raise ValueError(msg)
        return self


class StepOutputField(BaseModel):
    """One output field declared on a step.

    The ``type`` field is either a JSON Schema fragment dict or a
    ``{"$ref": "#/types/Foo"}`` reference into the envelope's
    hoisted ``types`` block.
    """

    model_config = ConfigDict(extra="forbid")

    description: str
    type: dict[str, Any] = Field(default_factory=lambda: {"type": "string"})


class _StepBase(BaseModel):
    """Shared fields and validators for every envelope step.

    Phase 16.5: factored out so the per-kind sub-models can inherit
    the common surface. Sub-models add only their kind-specific
    required fields. The discriminated-union ``EnvelopeStep`` type
    alias below ties the sub-models together.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    client: str
    instruction: str
    inputs: dict[str, str] = Field(default_factory=dict)
    output_fields: dict[str, StepOutputField] = Field(default_factory=dict)
    tunable: Tunable = Field(default_factory=Tunable)

    @field_validator("id")
    @classmethod
    def _check_id(cls, v: str) -> str:
        if not _STEP_ID_RE.match(v):
            msg = (
                f"Step id {v!r} is not valid. Step ids must match "
                f"{_STEP_ID_RE.pattern!r} — start with a lowercase "
                "letter, then lowercase letters / digits / underscores."
            )
            raise ValueError(msg)
        return v

    @field_validator("inputs")
    @classmethod
    def _check_inputs_are_pointers(cls, v: dict[str, str]) -> dict[str, str]:
        for field_name, ref in v.items():
            if not isinstance(ref, str):
                msg = (
                    f"Step inputs[{field_name!r}] must be a JSON-pointer string "
                    f"like '$.inputs.text' or '$.steps.extract.output.field'; "
                    f"got {type(ref).__name__}."
                )
                raise ValueError(msg)
            if not ref.startswith("$."):
                msg = (
                    f"Step inputs[{field_name!r}] must start with '$.', got "
                    f"{ref!r}. Use '$.inputs.X' for top-level inputs and "
                    "'$.steps.<id>.output.X' for upstream step outputs."
                )
                raise ValueError(msg)
        return v


class CallStep(_StepBase):
    """A simple structured-output Call step. No kind-specific fields."""

    kind: Literal["call"] = "call"


class ReasonStep(_StepBase):
    """A ChainOfThought step. No kind-specific fields."""

    kind: Literal["reason"] = "reason"


class JudgeStepModel(_StepBase):
    """An LLM-as-judge step. No kind-specific fields beyond the base.

    Class is named ``JudgeStepModel`` to avoid colliding with the
    pre-existing :class:`JudgeSpec` (the sub-spec used inside
    :class:`RefineStep`'s critic).
    """

    kind: Literal["judge"] = "judge"


class ReActStep(_StepBase):
    """A ReAct loop step. NOT EXECUTABLE — reserved for forward compatibility.

    The envelope schema validates this step kind but the executor will reject
    it at runtime. Use the ``kaos-llm-core-react`` MCP tool directly or the
    ``kaos_llm_core.programs.react.ReAct`` Python API instead.

    Required fields: ``max_iterations`` (1+) and ``tools`` (declarative).
    """

    kind: Literal["react"] = "react"
    max_iterations: int = Field(ge=1)
    tools: list[ToolSpec]


class RefineStep(_StepBase):
    """A Refine loop step. NOT EXECUTABLE — reserved for forward compatibility.

    The envelope schema validates this step kind but the executor will reject
    it at runtime. Use the ``kaos-llm-core-refine`` MCP tool directly or
    compose ``kind='call'`` + ``kind='judge'`` steps as an alternative.

    Required fields: ``max_iterations`` and a ``judge`` :class:`JudgeSpec`
    sub-spec for the critic.
    """

    kind: Literal["refine"] = "refine"
    max_iterations: int = Field(ge=1)
    judge: JudgeSpec


class BestOfNStep(_StepBase):
    """A BestOfN sampling step. NOT EXECUTABLE — reserved for forward compatibility.

    The envelope schema validates this step kind but the executor will reject
    it at runtime. Use the ``kaos-llm-core-best-of-n`` MCP tool directly or
    run multiple ``kind='call'`` steps and select externally.

    Required fields: ``n`` (1+) and ``selector``
    (``"first"`` or ``"longest"``).
    """

    kind: Literal["best_of_n"] = "best_of_n"
    n: int = Field(ge=1)
    selector: str


class MultiChainComparisonStep(_StepBase):
    """Phase 16.2 — sample N chains and synthesize via aggregator LM.

    Required fields: ``n`` (>= 2). Optional ``aggregator_client``
    overrides the synthesis-stage model.
    """

    kind: Literal["multi_chain_comparison"] = "multi_chain_comparison"
    n: int = Field(ge=2)
    aggregator_client: str | None = None


class ProgramOfThoughtStep(_StepBase):
    """Phase 16.2 — code-as-reasoning with subprocess sandbox.

    Required: ``allow_code_execution`` MUST be true. The
    ``_must_be_true`` validator below rejects ``False`` at parse time
    so the safety opt-in cannot be missed.
    """

    kind: Literal["program_of_thought"] = "program_of_thought"
    allow_code_execution: bool
    timeout_s: float | None = None

    @field_validator("allow_code_execution")
    @classmethod
    def _must_be_true(cls, v: bool) -> bool:
        if not v:
            msg = (
                "kind='program_of_thought' requires 'allow_code_execution': "
                "true. This is the deliberate opt-in for executing "
                "model-generated Python in a subprocess sandbox. Read "
                "kaos_llm_core/programs/program_of_thought.py for the "
                "security model before enabling."
            )
            raise ValueError(msg)
        return v


# Discriminated union of every step kind. Pydantic enforces the
# per-kind required fields automatically when an envelope is parsed
# via ProgramEnvelope.model_validate(). Adding a new step kind in
# Phase 17+ is one new sub-class plus one entry in this union plus
# one builder in ``builders.py`` plus one entry in
# ``StepKind`` (in ``_constants.py``) and ``SUPPORTED_CAPABILITIES``.
EnvelopeStep = Annotated[
    CallStep
    | ReasonStep
    | JudgeStepModel
    | ReActStep
    | RefineStep
    | BestOfNStep
    | MultiChainComparisonStep
    | ProgramOfThoughtStep,
    Field(discriminator="kind"),
]


class ProgramEnvelope(BaseModel):
    """A complete Program v3 envelope.

    The JSON-declarable specification of a multi-step LLM program.
    See ``docs/internal/design/program-envelope-v3.md`` for the
    full schema spec, design rationale, and edge cases.
    """

    model_config = ConfigDict(extra="forbid")

    kaos_program: Literal["1", "1.1"] = Field(default="1.1", alias="kaos_program")
    name: str
    description: str | None = None
    inputs: dict[str, InputSpec] = Field(default_factory=dict)
    clients: dict[str, ClientSpec] = Field(default_factory=dict)
    types: dict[str, dict[str, Any]] = Field(default_factory=dict)
    steps: list[EnvelopeStep] = Field(default_factory=list)
    output: dict[str, str] = Field(default_factory=dict)
    capabilities: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _check_name(cls, v: str) -> str:
        if not _NAME_RE.match(v):
            msg = (
                f"Program name {v!r} is not valid. Names must match "
                f"{_NAME_RE.pattern!r} — start with a lowercase letter, "
                "then lowercase letters / digits / hyphens."
            )
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def _validate_envelope(self) -> ProgramEnvelope:
        """Cross-field validation: capabilities, references, step ids, etc.

        The cross-step validation helpers live in
        :mod:`.validation` and are imported lazily here to avoid a
        circular import (validation depends on the model classes).
        """
        # Phase 16.5: per-kind required fields are now enforced by the
        # discriminated-union sub-models at parse time, so this method
        # only needs the cross-step validators (pointer resolution,
        # type refs). _validate_step_kind_fields is removed.
        from kaos_llm_core.programs.envelope.validation import (
            _validate_pointer,
            _validate_type_refs,
        )

        # 1. Capabilities are all supported
        unsupported = set(self.capabilities) - SUPPORTED_CAPABILITIES
        if unsupported:
            msg = (
                f"Envelope declares capabilities not supported by this executor: "
                f"{sorted(unsupported)}. Supported capabilities: "
                f"{sorted(SUPPORTED_CAPABILITIES)}. Either remove the unsupported "
                "capabilities or upgrade kaos-llm-core."
            )
            raise ValueError(msg)

        # 2. Every step kind in steps[] is in capabilities[]
        declared_caps = set(self.capabilities)
        for step in self.steps:
            if step.kind not in declared_caps:
                msg = (
                    f"Step {step.id!r} has kind={step.kind!r} but the envelope's "
                    f"capabilities list does not include {step.kind!r}. Add "
                    f"{step.kind!r} to the top-level 'capabilities' array."
                )
                raise ValueError(msg)

        # 3. Step ids are unique
        step_ids: list[str] = []
        for step in self.steps:
            if step.id in step_ids:
                msg = (
                    f"Step id {step.id!r} appears more than once. Every step must have a unique id."
                )
                raise ValueError(msg)
            step_ids.append(step.id)

        # 4. Every step's client reference resolves
        for step in self.steps:
            if step.client not in self.clients:
                msg = (
                    f"Step {step.id!r} references client {step.client!r}, "
                    f"which is not declared in the envelope's 'clients' block. "
                    f"Available clients: {sorted(self.clients)}."
                )
                raise ValueError(msg)

        # 5. Every input pointer in every step refers to a valid source
        seen_step_ids: set[str] = set()
        for step in self.steps:
            for field_name, pointer in step.inputs.items():
                _validate_pointer(
                    pointer,
                    self.inputs,
                    seen_step_ids,
                    where=f"step {step.id!r} inputs[{field_name!r}]",
                )
            seen_step_ids.add(step.id)

        # 6. Every output pointer resolves
        for output_name, pointer in self.output.items():
            _validate_pointer(
                pointer,
                self.inputs,
                set(step_ids),
                where=f"top-level output[{output_name!r}]",
            )

        # 7. Step-kind-specific required fields — Phase 16.5: enforced
        # by the discriminated-union sub-models, no longer needed here.

        # 8. Every $ref in step output_fields resolves into the types block
        for step in self.steps:
            for field_name, output_field in step.output_fields.items():
                _validate_type_refs(
                    output_field.type,
                    self.types,
                    where=f"step {step.id!r} output_fields[{field_name!r}].type",
                )

        return self
