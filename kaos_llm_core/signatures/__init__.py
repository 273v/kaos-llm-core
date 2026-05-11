"""Signature system — typed I/O contracts for LLM functions."""

from kaos_llm_core.signatures.extraction import (
    ColumnSpec,
    ColumnTypeName,
    ExtractionSchema,
    Provenance,
)
from kaos_llm_core.signatures.fields import InputField, OutputField
from kaos_llm_core.signatures.grounding import (
    DEFAULT_MATCH_STRATEGIES,
    Answer,
    Cited,
    Claim,
    ClaimError,
    ClaimType,
    GroundedAnswer,
    InsufficientEvidence,
    MatchStrategy,
    NeedsAggregation,
    RefusalPolicy,
    Span,
    SpanError,
    StrategyResult,
)
from kaos_llm_core.signatures.introspection import (
    create_output_model,
    get_input_fields,
    get_instruction,
    get_output_fields,
    signature_to_json_schema,
)
from kaos_llm_core.signatures.rubric import (
    Criterion,
    CriterionOperator,
    HybridRubric,
)
from kaos_llm_core.signatures.signature import Signature
from kaos_llm_core.signatures.span_tagging import (
    tag_paragraphs,
    tag_passages,
    validate_cited_output,
)

__all__ = [
    "DEFAULT_MATCH_STRATEGIES",
    "Answer",
    "Cited",
    "Claim",
    "ClaimError",
    "ClaimType",
    "ColumnSpec",
    "ColumnTypeName",
    "Criterion",
    "CriterionOperator",
    "ExtractionSchema",
    "GroundedAnswer",
    "HybridRubric",
    "InputField",
    "InsufficientEvidence",
    "MatchStrategy",
    "NeedsAggregation",
    "OutputField",
    "Provenance",
    "RefusalPolicy",
    "Signature",
    "Span",
    "SpanError",
    "StrategyResult",
    "create_output_model",
    "get_input_fields",
    "get_instruction",
    "get_output_fields",
    "signature_to_json_schema",
    "tag_paragraphs",
    "tag_passages",
    "validate_cited_output",
]
