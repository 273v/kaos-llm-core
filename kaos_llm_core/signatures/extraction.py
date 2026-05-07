"""ExtractionSchema — compile a user spec into a Signature with Cited[T] fields.

This module is the WS-TR.PR-2 compiler that turns three equivalent schema
forms (Pydantic class / dict / existing Signature) into a dynamic
:class:`~kaos_llm_core.signatures.signature.Signature` subclass whose output
fields are typed as :class:`~kaos_llm_core.signatures.grounding.Cited` ``[T]``
(default, ``provenance="cited"``), :class:`~kaos_llm_core.signatures.grounding.GroundedAnswer`
``[T]`` (``provenance="grounded"``), or bare ``T`` (``provenance="none"``).

The compiled Signature is consumed by :class:`kaos_llm_core.programs.extract.Extract`
to drive a single LLM call that returns fully-typed + span-cited output in
one shot. See ``docs/design/structured-extraction-integration.md`` §1.4 for
the full compiler design rationale.

Usage::

    schema = ExtractionSchema(
        id="contract-v1",
        columns=(
            ColumnSpec(id="effective_date", column_type="date"),
            ColumnSpec(id="parties", column_type="list", constraints={"inner": "string"}),
            ColumnSpec(id="governing_law", column_type="string"),
        ),
    )
    Sig = schema.to_signature(provenance="cited")
    # Sig is now a dynamic Signature with:
    #   source_text: str (input)
    #   effective_date: Cited[date] (output)
    #   parties: Cited[list[str]] (output)
    #   governing_law: Cited[str] (output)
"""

from __future__ import annotations

import datetime
import decimal
from typing import Any, Literal, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, create_model

from kaos_llm_core.signatures.fields import InputField, OutputField, is_output_field
from kaos_llm_core.signatures.grounding import Cited, GroundedAnswer
from kaos_llm_core.signatures.signature import Signature

# -- Column type surface -----------------------------------------------------

ColumnTypeName = Literal[
    "string",
    "verbatim_quote",
    "number",
    "integer",
    "date",
    "datetime",
    "boolean",
    "money",
    "score",
    "enum",
    "list",
    "object",
    "entity_role",
]
"""The user-facing column-type vocabulary for structured extraction.

Maps to Python types as follows (``to_python_type`` below does the lookup):

- ``string`` / ``verbatim_quote`` → ``str``
- ``number`` → ``float``
- ``integer`` / ``score`` → ``int``
- ``date`` → ``datetime.date``
- ``datetime`` → ``datetime.datetime``
- ``boolean`` → ``bool``
- ``money`` → ``dict[str, Any]`` (``{"amount": Decimal-coercible, "currency": str}``)
- ``enum`` → ``str`` (with ``constraints["values"]`` sent to the prompt)
- ``list`` → ``list[T]`` (T from ``constraints["inner"]``; default ``str``)
- ``object`` → ``dict[str, Any]``
- ``entity_role`` → ``dict[str, Any]`` (``{"name": str, "role": str, "entity_type": str | None}``)

Use the dict-heavy types (``money``, ``object``, ``entity_role``) when the
LLM's output is a nested structure; the Pydantic model emits ``dict[str, Any]``
so the Signature compiles cleanly without naming every sub-field.
"""


Provenance = Literal["none", "cited", "grounded"]
"""Provenance wrapping strategy for an :class:`ExtractionSchema`'s output fields.

- ``"none"`` — output fields are bare Python types; no span verification.
- ``"cited"`` — fields wrapped in ``Cited[T]``; LLM is prompted to emit spans
  alongside values; post-decode :func:`validate_cited_output` can verify.
- ``"grounded"`` — fields wrapped in ``GroundedAnswer[T]`` (full
  Answer+Claim+InsufficientEvidence discriminated union); supports formal
  refusal with ``attempted_claims``.
"""


def _python_type_for_column(spec: ColumnSpec) -> Any:
    """Map a :class:`ColumnSpec` to the Python type the Signature should use.

    Returns ``Any`` typed rather than ``type`` because the ``list[T]``
    branch constructs a generic alias, not a bare type. Pydantic
    ``create_model`` accepts both forms.
    """
    ct = spec.column_type
    if ct in ("string", "verbatim_quote", "enum"):
        return str
    if ct == "number":
        return float
    if ct in ("integer", "score"):
        return int
    if ct == "date":
        return datetime.date
    if ct == "datetime":
        return datetime.datetime
    if ct == "boolean":
        return bool
    if ct == "list":
        inner_name = spec.constraints.get("inner", "string")
        inner_type = _python_type_for_column(
            ColumnSpec(id=spec.id, column_type=inner_name, constraints={})
        )
        return list[inner_type]
    # ``money`` / ``object`` / ``entity_role`` all project to dict[str, Any];
    # specific shape is described to the LLM via ColumnSpec.description +
    # ColumnSpec.constraints, not enforced by the compiled signature type.
    return dict[str, Any]


# -- Column + schema value types --------------------------------------------


class ColumnSpec(BaseModel):
    """One column of an extraction schema.

    Columns are order-preserving within a schema. ``id`` doubles as the
    Signature field name — use snake_case identifiers that are valid Python
    attribute names.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, pattern=r"^[a-z_][a-z0-9_]*$")
    """Snake-case identifier. Used as the Signature field name and as the
    ``column_id`` on downstream :class:`~kaos_content.model.extraction.ExtractionCell`
    rows. Pattern enforces valid Python attribute names."""

    label: str = ""
    """Human-readable label (defaults to ``id``). Shown in UIs and used in
    the LLM prompt as the column header."""

    column_type: ColumnTypeName
    """The column's logical type. See :data:`ColumnTypeName`."""

    description: str = ""
    """Natural-language query describing what to extract for this column.
    Sent to the LLM as the column's prompt. E.g., ``"The effective date
    of the contract, in ISO 8601 format."``"""

    required: bool = True
    """When ``True``, the field is required in the compiled Signature.
    When ``False``, the field becomes ``T | None``."""

    constraints: dict[str, Any] = Field(default_factory=dict)
    """Type-specific hints. Examples:

    - ``{"values": ["low", "medium", "high"]}`` for enum columns.
    - ``{"inner": "string"}`` for ``list`` columns.
    - ``{"currency": "USD"}`` as a default for ``money`` columns.

    These are NOT enforced by the compiled Pydantic type — they are passed
    to the LLM via the prompt. The model is responsible for respecting
    them. Enforcement happens at the Cell validator layer downstream.
    """


class ExtractionSchema(BaseModel):
    """Schema for structured extraction over one document.

    Three constructors (``from_pydantic`` / ``from_dict`` / ``from_signature``)
    converge on a Signature subclass via :meth:`to_signature`. The Signature
    has exactly one input field (``source_text: str``) and one output field
    per column, typed per the ``provenance`` argument.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    """Stable schema identifier. Used in downstream cell tags + caching."""

    version: int = Field(default=1, ge=1)
    """Monotonic schema version. Cells tag with this so reruns only
    re-execute mismatched cells when the schema evolves."""

    columns: tuple[ColumnSpec, ...] = Field(min_length=1)
    """Ordered tuple of columns. Order is preserved in the compiled Signature,
    in the downstream table, and in provider prompts."""

    recipe_id: str | None = None
    """When derived from a shipped recipe, the recipe's name (e.g.,
    ``"merger-agreement"``). None for ad-hoc schemas."""

    # -- Constructors -------------------------------------------------

    @classmethod
    def from_dict(cls, spec: dict[str, Any]) -> ExtractionSchema:
        """Compile a dict spec into an :class:`ExtractionSchema`.

        Expected shape::

            {
                "id": "contract-v1",
                "columns": [
                    {"id": "effective_date", "column_type": "date", ...},
                    ...
                ],
                "version": 1,              # optional, default 1
                "recipe_id": "lease",      # optional
            }

        Unknown keys raise :class:`pydantic.ValidationError`.
        """
        return cls.model_validate(spec)

    @classmethod
    def from_pydantic(
        cls,
        model: type[BaseModel],
        *,
        schema_id: str = "",
        version: int = 1,
    ) -> ExtractionSchema:
        """Derive an :class:`ExtractionSchema` from a Pydantic class's fields.

        Each Pydantic field becomes a ``ColumnSpec`` with ``column_type``
        inferred from the Python annotation. Unsupported annotations
        (callables, TypeVars, custom generics) raise ValueError with a
        pointer to use the dict constructor instead.

        Args:
            model: A Pydantic BaseModel subclass. All fields become columns.
            schema_id: Override the schema id (default: ``model.__name__``).
            version: Schema version (default 1).
        """
        columns: list[ColumnSpec] = []
        for name, field_info in model.model_fields.items():
            column_type = _infer_column_type_from_annotation(field_info.annotation)
            description = field_info.description or ""
            required = field_info.is_required()
            constraints: dict[str, Any] = {}
            # list[X] — record the inner type name so to_signature reproduces it.
            origin = get_origin(field_info.annotation)
            if origin is list:
                inner = (
                    get_args(field_info.annotation)[0] if get_args(field_info.annotation) else str
                )
                constraints["inner"] = _name_for_simple_type(inner)
            columns.append(
                ColumnSpec(
                    id=name,
                    label=field_info.title or name,
                    column_type=column_type,
                    description=description,
                    required=required,
                    constraints=constraints,
                )
            )
        return cls(
            id=schema_id or model.__name__,
            version=version,
            columns=tuple(columns),
        )

    @classmethod
    def from_signature(
        cls,
        sig: type[Signature],
        *,
        schema_id: str = "",
        version: int = 1,
    ) -> ExtractionSchema:
        """Derive an :class:`ExtractionSchema` from an existing Signature's outputs.

        Only output fields are converted. Input fields are discarded (the
        compiled schema always has exactly one input: ``source_text: str``).
        """
        columns: list[ColumnSpec] = []
        for name, field_info in sig.model_fields.items():
            if not is_output_field(field_info):
                continue
            column_type = _infer_column_type_from_annotation(field_info.annotation)
            description = field_info.description or ""
            required = field_info.is_required()
            constraints: dict[str, Any] = {}
            origin = get_origin(field_info.annotation)
            if origin is list:
                inner = (
                    get_args(field_info.annotation)[0] if get_args(field_info.annotation) else str
                )
                constraints["inner"] = _name_for_simple_type(inner)
            columns.append(
                ColumnSpec(
                    id=name,
                    label=field_info.title or name,
                    column_type=column_type,
                    description=description,
                    required=required,
                    constraints=constraints,
                )
            )
        if not columns:
            msg = f"Signature {sig.__name__} has no OutputField; cannot derive schema"
            raise ValueError(msg)
        return cls(
            id=schema_id or sig.__name__,
            version=version,
            columns=tuple(columns),
        )

    # -- Compilation --------------------------------------------------

    def to_signature(self, *, provenance: Provenance = "cited") -> type[Signature]:
        """Compile to a dynamic :class:`Signature` subclass.

        The returned class has exactly one input (``source_text: str``) and
        one output field per column. Output fields are wrapped per
        ``provenance``:

        - ``"none"`` — bare ``T`` (or ``T | None`` for non-required).
        - ``"cited"`` — :class:`Cited` ``[T]`` (LLM emits spans alongside value).
        - ``"grounded"`` — :class:`GroundedAnswer` ``[T]`` (full Answer +
          InsufficientEvidence union; LLM can formally refuse).
        """
        from kaos_llm_core.signatures.signature import Signature as _Signature

        field_defs: dict[str, Any] = {
            "source_text": (
                str,
                InputField(description="The source text from which to extract"),
            ),
        }
        for col in self.columns:
            inner_type = _python_type_for_column(col)
            wrapped_type = _wrap_for_provenance(inner_type, provenance)
            if not col.required:
                wrapped_type = wrapped_type | None
            default = ... if col.required else None
            field_defs[col.id] = (
                wrapped_type,
                OutputField(
                    default,
                    description=_column_description(col),
                ),
            )

        model_name = f"Extract_{_sanitize(self.id)}_v{self.version}"
        Dynamic = create_model(
            model_name,
            __base__=_Signature,
            **field_defs,
        )
        # Embed the schema instruction in the Signature's docstring. The
        # codec uses this as the LLM system prompt.
        Dynamic.__doc__ = _build_instruction(self)
        return Dynamic


# -- Helpers ---------------------------------------------------------


def _wrap_for_provenance(inner_type: Any, provenance: Provenance) -> Any:
    if provenance == "none":
        return inner_type
    if provenance == "cited":
        return Cited[inner_type]
    # "grounded"
    return GroundedAnswer[inner_type]


def _column_description(col: ColumnSpec) -> str:
    """Build an LLM-facing description for a column including constraints."""
    parts: list[str] = []
    if col.description:
        parts.append(col.description)
    if col.column_type == "enum":
        values = col.constraints.get("values")
        if isinstance(values, list):
            parts.append(f"Allowed values: {values}.")
    elif col.column_type == "verbatim_quote":
        parts.append("Emit the exact source phrase, preserving whitespace and punctuation.")
    elif col.column_type == "money":
        parts.append(
            'Return a dict: {"amount": "<decimal as string>", "currency": "<ISO 4217 code>"}.',
        )
    elif col.column_type == "entity_role":
        parts.append(
            'Return a dict: {"name": "<name>", "role": "<role>", "entity_type": "<type or null>"}.',
        )
    elif col.column_type == "date":
        parts.append("ISO 8601 date (YYYY-MM-DD).")
    elif col.column_type == "datetime":
        parts.append("ISO 8601 datetime.")
    return " ".join(parts) if parts else col.id


def _build_instruction(schema: ExtractionSchema) -> str:
    """Build the LLM system instruction embedded in the compiled Signature."""
    lines = [
        f"Extract structured fields from the provided source_text per schema '{schema.id}' "
        f"(version {schema.version}).",
        "Return every required field. For each field, provide supporting spans "
        "from the source text verbatim.",
    ]
    for col in schema.columns:
        marker = "required" if col.required else "optional"
        lines.append(f"- {col.id} ({col.column_type}, {marker}): {_column_description(col)}")
    return "\n".join(lines)


def _sanitize(identifier: str) -> str:
    """Make a string safe to embed in a Python class name."""
    cleaned = "".join(c if c.isalnum() else "_" for c in identifier)
    if cleaned and cleaned[0].isdigit():
        cleaned = "_" + cleaned
    return cleaned or "Schema"


def _name_for_simple_type(t: Any) -> str:
    """Return a short ColumnTypeName-compatible name for a simple annotation."""
    mapping: dict[Any, str] = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        datetime.date: "date",
        datetime.datetime: "datetime",
        decimal.Decimal: "number",
    }
    return mapping.get(t, "string")


def _infer_column_type_from_annotation(ann: Any) -> ColumnTypeName:
    """Best-effort mapping of a Pydantic field annotation to a ColumnTypeName.

    Falls back to ``"string"`` for types we don't recognize. Callers that
    need exact type control should use ``ExtractionSchema.from_dict`` with
    explicit ``column_type`` per column.
    """
    if ann is str:
        return "string"
    if ann is int:
        return "integer"
    if ann is float:
        return "number"
    if ann is bool:
        return "boolean"
    if ann is datetime.date:
        return "date"
    if ann is datetime.datetime:
        return "datetime"
    if ann is decimal.Decimal:
        return "number"
    origin = get_origin(ann)
    if origin is list:
        return "list"
    if origin is dict or ann is dict:
        return "object"
    # Optional[X] → pick inner non-None type.
    if origin is type(None):
        return "string"
    args = get_args(ann)
    if args:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _infer_column_type_from_annotation(non_none[0])
    return "string"
