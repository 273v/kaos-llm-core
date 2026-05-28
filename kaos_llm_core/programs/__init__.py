"""Programs — composable LLM functions."""

from kaos_llm_core.programs._invocation import Invocation, TokenUsage, current_invocation
from kaos_llm_core.programs.base import Program
from kaos_llm_core.programs.batch import (
    BatchError,
    BatchInputSource,
    BatchItem,
    BatchProgress,
    BatchResult,
    BatchResumeMismatchError,
    BatchStopAfterNError,
    BatchUnsupportedBackendError,
    BatchWriter,
    JsonlBatchWriter,
    JsonlInputSource,
    ListInputSource,
    batch_run,
    jsonl_input_source,
    list_input_source,
)
from kaos_llm_core.programs.batch_bridges import (
    BatchBridgeUnavailableError,
    PdfPagesInputSource,
    TabularRowsInputSource,
    pdf_pages_input_source,
    tabular_rows_input_source,
)
from kaos_llm_core.programs.best_of_n import BestOfN, BestOfNResult
from kaos_llm_core.programs.call import Call, CallPlan
from kaos_llm_core.programs.chain_of_thought import ChainOfThought
from kaos_llm_core.programs.deal_review import DealReviewProgram, DealReviewResult
from kaos_llm_core.programs.decorator import llm_call
from kaos_llm_core.programs.designers import (
    RubricDesignerSignature,
    SchemaDesignerSignature,
    design_rubric,
    design_schema,
    sample_corpus_text,
)
from kaos_llm_core.programs.ensemble import Ensemble
from kaos_llm_core.programs.envelope import (
    ClientSpec,
    EnvelopeStep,
    InputSpec,
    ProgramEnvelope,
    ProgramEnvelopeError,
    Tunable,
    from_envelope,
    program_hash,
)
from kaos_llm_core.programs.extract import (
    CorpusExtractionResult,
    CorpusInputSource,
    Extract,
    ExtractCorpusError,
    ExtractionResult,
    extract_corpus,
)
from kaos_llm_core.programs.hooks import CallHooks
from kaos_llm_core.programs.interpret_extraction import InterpretExtractionSignature
from kaos_llm_core.programs.judge import Judge, JudgedResult
from kaos_llm_core.programs.multi_chain_comparison import (
    MultiChainComparison,
    MultiChainComparisonResult,
)
from kaos_llm_core.programs.program_hooks import ProgramHooks
from kaos_llm_core.programs.program_of_thought import (
    ProgramOfThought,
    ProgramOfThoughtError,
    ProgramOfThoughtResult,
)
from kaos_llm_core.programs.query_expander import ExpandQuery, LLMQueryExpander, QueryExpander
from kaos_llm_core.programs.rag import RAG, CorpusQA, RAGResult, build_passage_context
from kaos_llm_core.programs.react import Iteration, ReAct, ReActResult, ToolObservation
from kaos_llm_core.programs.refine import Refine, RefineHistoryEntry, RefineResult
from kaos_llm_core.programs.reranker import JudgeReranker, RelevanceSignature
from kaos_llm_core.programs.result_mixin import HasOutputs, OutputForwardingMixin
from kaos_llm_core.programs.scoring import RowJudgmentSignature, apply_rubric
from kaos_llm_core.programs.tool import Tool

__all__ = [
    "RAG",
    "BatchBridgeUnavailableError",
    "BatchError",
    "BatchInputSource",
    "BatchItem",
    "BatchProgress",
    "BatchResult",
    "BatchResumeMismatchError",
    "BatchStopAfterNError",
    "BatchUnsupportedBackendError",
    "BatchWriter",
    "BestOfN",
    "BestOfNResult",
    "Call",
    "CallHooks",
    "CallPlan",
    "ChainOfThought",
    "ClientSpec",
    "CorpusExtractionResult",
    "CorpusInputSource",
    "CorpusQA",
    "DealReviewProgram",
    "DealReviewResult",
    "Ensemble",
    "EnvelopeStep",
    "ExpandQuery",
    "Extract",
    "ExtractCorpusError",
    "ExtractionResult",
    "HasOutputs",
    "InputSpec",
    "InterpretExtractionSignature",
    "Invocation",
    "Iteration",
    "JsonlBatchWriter",
    "JsonlInputSource",
    "Judge",
    "JudgeReranker",
    "JudgedResult",
    "LLMQueryExpander",
    "ListInputSource",
    "MultiChainComparison",
    "MultiChainComparisonResult",
    "OutputForwardingMixin",
    "PdfPagesInputSource",
    "Program",
    "ProgramEnvelope",
    "ProgramEnvelopeError",
    "ProgramHooks",
    "ProgramOfThought",
    "ProgramOfThoughtError",
    "ProgramOfThoughtResult",
    "QueryExpander",
    "RAGResult",
    "ReAct",
    "ReActResult",
    "Refine",
    "RefineHistoryEntry",
    "RefineResult",
    "RelevanceSignature",
    "RowJudgmentSignature",
    "RubricDesignerSignature",
    "SchemaDesignerSignature",
    "TabularRowsInputSource",
    "TokenUsage",
    "Tool",
    "ToolObservation",
    "Tunable",
    "apply_rubric",
    "batch_run",
    "build_passage_context",
    "current_invocation",
    "design_rubric",
    "design_schema",
    "extract_corpus",
    "from_envelope",
    "jsonl_input_source",
    "list_input_source",
    "llm_call",
    "pdf_pages_input_source",
    "program_hash",
    "sample_corpus_text",
    "tabular_rows_input_source",
]
