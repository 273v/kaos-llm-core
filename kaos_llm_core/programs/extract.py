"""Extract — schema-driven structured extraction with grounded provenance.

The WS-TR primitive. Composes:

- :class:`~kaos_llm_core.signatures.extraction.ExtractionSchema` (compiled to
  a dynamic Signature with ``Cited[T]`` output fields)
- :class:`~kaos_llm_core.programs.call.Call` (one LLM call using provider-native
  structured output)
- Optionally :class:`~kaos_llm_core.programs.grounded.Grounded` wrapper (post-
  decode span verification with retry)

Returns an :class:`ExtractionResult` containing a tuple of
:class:`~kaos_content.model.extraction.ExtractionCell` — the per-cell audit
shape consumed by the downstream TabularDocument sink.

See ``docs/design/structured-extraction-integration.md`` §3 for rationale.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Literal

from kaos_content.model.extraction import (
    ExtractionCell,
    ExtractionCitation,
    ExtractionError,
    ExtractionErrorCode,
)
from kaos_core.logging import get_logger
from pydantic import BaseModel, ConfigDict, Field

from kaos_llm_core.errors import CallError, ValidationRetryExhaustedError
from kaos_llm_core.extract.merge import AlphaLLMMerger
from kaos_llm_core.programs.base import Program
from kaos_llm_core.programs.call import Call
from kaos_llm_core.programs.chunk_retry import (
    ChunkExtractResult,
    ChunkRetry,
    ChunkRetryConfig,
)
from kaos_llm_core.programs.grounded import Grounded
from kaos_llm_core.signatures.extraction import (
    ColumnSpec,
    ExtractionSchema,
    Provenance,
)
from kaos_llm_core.signatures.grounding import Answer, Cited, InsufficientEvidence, Span
from kaos_llm_core.signatures.signature import Signature

logger = get_logger(__name__)


class ExtractionResult(BaseModel):
    """Result of a single-document :class:`Extract` invocation.

    ``cells`` is the primary consumption surface — one :class:`ExtractionCell`
    per schema column, in the schema's declared order. ``value`` was previously
    the raw decoded Pydantic model, but to keep this type round-trippable
    through ``batch_run``'s JSONL log (``records.py`` calls
    ``output.model_dump()``), we drop the generic ``value`` slot in favour of
    serializable projections: the full Cell list is authoritative, and callers
    who need the raw typed model can re-decode it from ``raw_output_json``.

    Pydantic (not a frozen dataclass) so :meth:`model_dump` and
    :meth:`model_validate` round-trip cleanly through ``batch_run``.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    cells: tuple[ExtractionCell, ...]
    schema_id: str
    schema_version: int
    doc_id: str
    is_verified: bool = True
    refused_columns: tuple[str, ...] = ()
    cost_usd: float = Field(default=0.0, ge=0.0)
    tokens_total: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    model: str = ""
    verification_error_count: int = Field(default=0, ge=0)
    attempts: int = Field(default=1, ge=1)
    # Raw decoded output, JSON-dumped for round-trip safety. Null for error
    # results. Callers that need the typed model can re-decode it against
    # ``schema.to_signature(provenance=...).model_fields["result"]`` type.
    raw_output_json: str | None = None


class Extract(Program):
    """Schema-driven structured extraction program.

    Takes an :class:`ExtractionSchema` (or a Pydantic class / dict) and, on
    ``.extract(text=...)``, produces an :class:`ExtractionResult` with one
    :class:`~kaos_content.model.extraction.ExtractionCell` per column.

    ``provenance="cited"`` (default) wraps each output in ``Cited[T]`` — the
    LLM is prompted to return spans alongside values. When a corpus is
    attached (via the ``corpus`` kwarg on ``extract``), those spans are
    verified; verification failures do NOT retry (use ``"grounded"`` for
    that). ``provenance="grounded"`` wraps each output in ``GroundedAnswer[T]``
    and composes with :class:`Grounded` for retry-with-feedback.

    Args:
        schema: An :class:`ExtractionSchema`, a Pydantic BaseModel subclass,
            a dict spec (for :meth:`ExtractionSchema.from_dict`), or an
            existing :class:`Signature` subclass.
        model: Provider/model string (e.g., ``"anthropic:claude-haiku-4-5"``).
            Defaults to the value in ``KaosLLMCoreSettings``.
        provenance: ``"none"`` / ``"cited"`` / ``"grounded"``. See
            :class:`~kaos_llm_core.signatures.extraction.Provenance`.
        max_retries: On ``"grounded"``, max span-verification retries.
        refusal: Scope of refusal — ``"field"`` allows per-cell
            ``not_in_document`` status; ``"row"`` fails the whole result
            on any refusal; ``"never"`` forbids refusals (model must answer).

    Usage::

        schema = ExtractionSchema.from_dict({
            "id": "document-v1",
            "columns": [
                {"id": "effective_date", "column_type": "date"},
                {"id": "parties", "column_type": "list",
                 "constraints": {"inner": "string"}},
            ],
        })
        extractor = Extract(schema, model="anthropic:claude-haiku-4-5")
        result = await extractor.extract(
            text="This document is dated January 1, 2025, between Acme and Beta.",
            doc_id="doc-001",
        )
        # result.cells[0].ai_value == "2025-01-01"
        # result.cells[1].ai_value == ["Acme", "Beta"]
    """

    def __init__(
        self,
        schema: ExtractionSchema | type[BaseModel] | dict[str, Any] | type[Signature],
        *,
        model: str | None = None,
        provenance: Provenance = "cited",
        max_retries: int = 2,
        refusal: Literal["field", "row", "never"] = "field",
        instructions: str | None = None,
        alpha_merger: AlphaLLMMerger | Literal["default"] | None = "default",
        chunk_retry: ChunkRetry | ChunkRetryConfig | Literal["default"] | None = None,
    ) -> None:
        super().__init__()
        self.schema = self._normalize_schema(schema)
        self.provenance = provenance
        self.refusal = refusal
        self._max_retries = max_retries

        # Compile the Signature, build the inner Call, optionally wrap in
        # Grounded. Call is auto-registered as a child via Program.__setattr__.
        SignatureCls = self.schema.to_signature(provenance=provenance)
        self._signature_cls = SignatureCls
        self.call = Call(
            SignatureCls,
            model=model,
            instructions=instructions,
        )
        self._grounded: Grounded | None = None  # populated per-extract when corpus provided

        # WS-TR.PR-6f.3 — rule-based alpha extractors augment LLM cells.
        # - "default" instantiates a merger with the stock dispatch.
        # - Pass ``None`` to disable the alpha pass entirely (pure-LLM mode
        #   for ablation benchmarks / debugging).
        # - Pass a concrete merger to customize dispatch (e.g., extra
        #   extractors).
        #
        # Runs under all provenance modes (including ``"none"``) — the
        # merger is enrichment, not verification.
        self._alpha_merger: AlphaLLMMerger | None
        if alpha_merger == "default":
            self._alpha_merger = AlphaLLMMerger()
        else:
            self._alpha_merger = alpha_merger

        # WS-TR.PR-6b — recursive chunk-retry. Disabled by default
        # because it doubles latency on long documents and only adds
        # value when the original LLM call refused on
        # content-availability grounds. Pass ``"default"`` for the
        # stock ChunkRetry, a ChunkRetryConfig to tune, or a fully
        # built ChunkRetry instance to inject custom logic.
        self._chunk_retry: ChunkRetry | None
        if chunk_retry == "default":
            self._chunk_retry = ChunkRetry()
        elif isinstance(chunk_retry, ChunkRetryConfig):
            self._chunk_retry = ChunkRetry(config=chunk_retry)
        else:
            self._chunk_retry = chunk_retry

    @staticmethod
    def _normalize_schema(
        schema: ExtractionSchema | type[BaseModel] | dict[str, Any] | type[Signature],
    ) -> ExtractionSchema:
        if isinstance(schema, ExtractionSchema):
            return schema
        if isinstance(schema, dict):
            return ExtractionSchema.from_dict(schema)
        if isinstance(schema, type) and issubclass(schema, Signature):
            return ExtractionSchema.from_signature(schema)
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return ExtractionSchema.from_pydantic(schema)
        msg = (
            f"Extract schema must be ExtractionSchema, dict, Pydantic BaseModel, "
            f"or Signature subclass; got {type(schema).__name__}"
        )
        raise TypeError(msg)

    # -- forward() / extract() ----------------------------------------

    async def forward(self, **kwargs: Any) -> ExtractionResult:
        """Invoked by ``Program.__call__`` / ``Program.invoke``.

        Accepts ``source_text`` (required), ``doc_id`` (default ``"doc-1"``),
        ``corpus`` (optional). Most callers use :meth:`extract` instead for
        the clearer kwarg name.
        """
        source_text = kwargs.get("source_text") or kwargs.get("text")
        if not isinstance(source_text, str):
            msg = "Extract.forward requires 'source_text' (or 'text') as a string input"
            raise TypeError(msg)
        return await self.extract(
            text=source_text,
            doc_id=kwargs.get("doc_id", "doc-1"),
            corpus=kwargs.get("corpus"),
        )

    async def extract(
        self,
        *,
        text: str,
        doc_id: str = "doc-1",
        corpus: dict[str, str] | None = None,
    ) -> ExtractionResult:
        """Run the extraction pipeline on one source.

        Args:
            text: The source text to extract from.
            doc_id: Stable identifier for the source document. Stamped onto
                every returned :class:`ExtractionCell`.
            corpus: Optional ``{source_uri: source_text}`` mapping for span
                verification. When ``provenance == "grounded"``, this is
                required; verification failures drive retry-with-feedback.
                When ``provenance == "cited"``, verification happens once
                (no retry). When ``provenance == "none"``, ignored.

        Returns:
            An :class:`ExtractionResult` with one :class:`ExtractionCell`
            per schema column.
        """
        start = time.perf_counter()

        # Build the corpus used for verification. If none provided and we
        # need one, fall back to a single-doc corpus keyed by doc_id so
        # Cited.verify has something to check against.
        effective_corpus = corpus if corpus is not None else {doc_id: text}

        # Track per-doc cost + tokens — populated below from the inner
        # invocation (or set to 0 for the grounded path, which doesn't
        # currently surface cost on GroundedResult).
        invocation_cost_usd: float = 0.0
        invocation_tokens_total: int = 0

        if self.provenance == "grounded":
            grounded = Grounded(
                self.call,
                corpus=effective_corpus,
                max_retries=self._max_retries,
            )
            try:
                g_result = await grounded(source_text=text)
            except (CallError, ValidationRetryExhaustedError) as exc:
                return self._error_result(
                    doc_id=doc_id,
                    error_msg=str(exc),
                    error_code=_classify_error(exc),
                    latency_ms=(time.perf_counter() - start) * 1000.0,
                )
            latency_ms = (time.perf_counter() - start) * 1000.0
            value = g_result.output
            verification_errors = g_result.verification_errors
            is_verified = g_result.is_verified
            attempts = g_result.attempts
            # TODO: GroundedResult doesn't currently expose cost/tokens.
            # Once it does, populate invocation_cost_usd / invocation_tokens_total
            # from g_result.usage. Until then this branch reports 0 — same as
            # historical behaviour, no regression.
        else:
            try:
                invocation = await self.call.invoke(source_text=text)
            except (CallError, ValidationRetryExhaustedError) as exc:
                return self._error_result(
                    doc_id=doc_id,
                    error_msg=str(exc),
                    error_code=_classify_error(exc),
                    latency_ms=(time.perf_counter() - start) * 1000.0,
                )
            latency_ms = (time.perf_counter() - start) * 1000.0
            value = invocation.output
            verification_errors = ()
            is_verified = True
            attempts = 1
            # Capture cost + tokens from the inner Call.invoke() so the
            # ExtractionResult.cost_usd / tokens_total fields are actually
            # populated. Pre-fix these were silently zero — any caller
            # doing budget enforcement on ExtractionResult under-counted
            # extraction spend.
            usage = invocation.usage
            invocation_cost_usd = float(getattr(usage, "cost_usd", 0.0) or 0.0)
            invocation_tokens_total = int(getattr(usage, "total_tokens", 0) or 0)

            # For cited provenance, verify spans once (no retry) for the
            # caller's audit trail. Grounded covered this via retry already.
            if self.provenance == "cited":
                from kaos_llm_core.signatures.span_tagging import validate_cited_output

                errs = validate_cited_output(value, effective_corpus)
                verification_errors = tuple(errs)
                is_verified = not errs

        # Project the typed output back to per-column Cells.
        model_name = getattr(self.call, "_model", None) or ""
        cells = self._project_cells(value, doc_id=doc_id, model=model_name)

        # WS-TR.PR-6f.3 — fold rule-based alpha extractions into the LLM
        # cells. The merger fires under ALL provenance modes including
        # ``"none"``, because it is enrichment (not verification):
        # alpha-promoted cells add citations even in "no-provenance"
        # mode, but the cost is trivial and the precision lift on
        # refused cells is the whole point. Disable via
        # ``alpha_merger=None`` at construction time.
        if self._alpha_merger is not None:
            cells = self._alpha_merger.merge(
                cells,
                columns=tuple(self.schema.columns),
                source_text=text,
                source_uri=doc_id,
            )

        # WS-TR.PR-6b — recursive chunk-retry on cells the LLM and the
        # alpha merger BOTH failed to resolve. Composes after the merger
        # so we don't waste retry budget on cells alpha could have
        # promoted with zero LLM cost. Only fires when chunk-retry was
        # explicitly enabled (default off because it doubles latency
        # for the long-doc case). The closure invokes only the inner
        # ``self.call`` (no merger, no chunk-retry) on each chunk to
        # avoid infinite recursion / double-counting.
        if self._chunk_retry is not None and self._chunk_retry.config.enabled:

            async def _extract_chunk(chunk_text: str, chunk_doc_id: str) -> ChunkExtractResult:
                inner = await self.call.invoke(source_text=chunk_text)
                inner_cells = self._project_cells(
                    inner.output, doc_id=chunk_doc_id, model=model_name
                )
                inner_cost = float(getattr(inner.usage, "cost_usd", 0.0) or 0.0)
                return ChunkExtractResult(cells=inner_cells, cost_usd=inner_cost)

            cells, _retry_cost = await self._chunk_retry.retry_refused_cells(
                cells,
                columns=tuple(self.schema.columns),
                source_text=text,
                doc_id=doc_id,
                extract_chunk=_extract_chunk,
            )

        refused_columns = tuple(c.column_id for c in cells if c.status != "extracted")

        # Respect refusal policy.
        if self.refusal == "never" and refused_columns:
            return self._error_result(
                doc_id=doc_id,
                error_msg=(
                    f"refusal='never' but model refused {len(refused_columns)} columns: "
                    f"{list(refused_columns)}"
                ),
                error_code="refusal",
                latency_ms=latency_ms,
            )
        if self.refusal == "row" and refused_columns:
            # Fail every cell — aggregate refusal at row level.
            return ExtractionResult(
                cells=tuple(
                    _cell_as_error(c, "refusal", "row-level refusal policy") for c in cells
                ),
                schema_id=self.schema.id,
                schema_version=self.schema.version,
                doc_id=doc_id,
                is_verified=False,
                refused_columns=refused_columns,
                latency_ms=latency_ms,
                attempts=attempts,
                model=model_name,
                cost_usd=invocation_cost_usd,
                tokens_total=invocation_tokens_total,
            )

        # Round-trip-safe raw output dump. ``value`` is a dynamic Pydantic
        # model (or an Answer/InsufficientEvidence); .model_dump_json() gives
        # us a stable string the caller can re-hydrate if they want the
        # typed form back. None for non-BaseModel outputs.
        raw_output_json: str | None = None
        if isinstance(value, BaseModel):
            raw_output_json = value.model_dump_json()

        return ExtractionResult(
            cells=cells,
            schema_id=self.schema.id,
            schema_version=self.schema.version,
            doc_id=doc_id,
            is_verified=is_verified,
            refused_columns=refused_columns,
            latency_ms=latency_ms,
            cost_usd=invocation_cost_usd,
            tokens_total=invocation_tokens_total,
            attempts=attempts,
            model=model_name,
            verification_error_count=len(verification_errors),
            raw_output_json=raw_output_json,
        )

    # -- Cell projection ----------------------------------------------

    def _project_cells(
        self,
        value: Any,
        *,
        doc_id: str,
        model: str,
    ) -> tuple[ExtractionCell, ...]:
        """Project a decoded Signature output into per-column ExtractionCells."""
        cells: list[ExtractionCell] = []
        for col in self.schema.columns:
            field_value = _safe_getattr(value, col.id)
            cells.append(
                self._cell_for_column(
                    col,
                    field_value=field_value,
                    doc_id=doc_id,
                    model=model,
                )
            )
        return tuple(cells)

    def _cell_for_column(
        self,
        col: ColumnSpec,
        *,
        field_value: Any,
        doc_id: str,
        model: str,
    ) -> ExtractionCell:
        # Unwrap per provenance.
        if isinstance(field_value, Cited):
            return _cell_from_cited(
                col=col,
                cited=field_value,
                doc_id=doc_id,
                schema_version=self.schema.version,
                model=model,
            )
        if isinstance(field_value, Answer):
            return _cell_from_answer(
                col=col,
                answer=field_value,
                doc_id=doc_id,
                schema_version=self.schema.version,
                model=model,
            )
        if isinstance(field_value, InsufficientEvidence):
            return ExtractionCell(
                doc_id=doc_id,
                column_id=col.id,
                schema_version=self.schema.version,
                status="not_in_document",
                rationale=field_value.reason,
                model=model,
            )
        if field_value is None and not col.required:
            return ExtractionCell(
                doc_id=doc_id,
                column_id=col.id,
                schema_version=self.schema.version,
                status="not_in_document",
                model=model,
            )
        # Bare-provenance path — treat the raw value as extracted with
        # full confidence. No citations.
        return ExtractionCell(
            doc_id=doc_id,
            column_id=col.id,
            schema_version=self.schema.version,
            status="extracted",
            ai_value=field_value,
            confidence=1.0,
            model=model,
        )

    def _error_result(
        self,
        *,
        doc_id: str,
        error_msg: str,
        error_code: ExtractionErrorCode,
        latency_ms: float,
    ) -> ExtractionResult:
        err = ExtractionError(
            code=error_code,
            message=error_msg,
            retry_recommended=error_code in ("model_api_error", "timeout"),
        )
        cells = tuple(
            ExtractionCell(
                doc_id=doc_id,
                column_id=col.id,
                schema_version=self.schema.version,
                status="error",
                error=err,
                model=getattr(self.call, "_model", None) or "",
            )
            for col in self.schema.columns
        )
        return ExtractionResult(
            cells=cells,
            schema_id=self.schema.id,
            schema_version=self.schema.version,
            doc_id=doc_id,
            is_verified=False,
            refused_columns=tuple(c.id for c in self.schema.columns),
            latency_ms=latency_ms,
            model=getattr(self.call, "_model", None) or "",
        )


# -- Projection helpers ----------------------------------------------------


def _safe_getattr(obj: Any, name: str) -> Any:
    """Read an attribute or dict key from obj; return None on miss."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _cell_from_cited(
    *,
    col: ColumnSpec,
    cited: Cited,
    doc_id: str,
    schema_version: int,
    model: str,
) -> ExtractionCell:
    """Project a Cited[T] value into an ExtractionCell."""
    citations = tuple(_citation_from_span(s) for s in cited.spans)
    return ExtractionCell(
        doc_id=doc_id,
        column_id=col.id,
        schema_version=schema_version,
        status="extracted",
        ai_value=cited.value,
        citations=citations,
        confidence=cited.confidence,
        model=model,
    )


def _cell_from_answer(
    *,
    col: ColumnSpec,
    answer: Answer,
    doc_id: str,
    schema_version: int,
    model: str,
) -> ExtractionCell:
    """Project an Answer[T] value into an ExtractionCell.

    Answer carries claims (with their own spans); flatten them into a single
    citation list for the Cell, preserving the first span per claim.
    """
    citations: list[ExtractionCitation] = []
    rationale_parts: list[str] = []
    for claim in answer.claims:
        rationale_parts.append(claim.statement)
        for s in claim.supporting_spans:
            citations.append(_citation_from_span(s))
    return ExtractionCell(
        doc_id=doc_id,
        column_id=col.id,
        schema_version=schema_version,
        status="extracted",
        ai_value=answer.value,
        rationale="; ".join(rationale_parts) if rationale_parts else None,
        citations=tuple(citations),
        confidence=answer.confidence,
        model=model,
    )


def _citation_from_span(span: Span) -> ExtractionCitation:
    """Convert a grounding :class:`Span` into an :class:`ExtractionCitation`.

    We synthesize the kaos-content fields the Span doesn't carry natively:

    - ``block_ref`` = ``#/span/{source_uri}`` — placeholder pointer that
      callers can later rewrite when the source is a ContentDocument.
    - ``page`` = 1 (Span has no page concept; corpus fan-out can enrich).
    - ``parent_block_ref`` = None.
    """
    snippet = span.quote
    # Always recompute a full 64-char SHA-256 hex. ``Span.quote_hash`` is a
    # truncated 16-char identifier (used only for Span dedup); the Cell-level
    # citation requires the full digest for tamper-evident verification.
    snippet_hash = hashlib.sha256(snippet.encode("utf-8")).hexdigest()
    return ExtractionCitation(
        block_ref=f"#/span/{span.source_uri}",
        page=1,
        char_span=span.char_span,
        snippet=snippet,
        snippet_sha256=snippet_hash,
        confidence=1.0,
    )


def _cell_as_error(cell: ExtractionCell, code: ExtractionErrorCode, message: str) -> ExtractionCell:
    err = ExtractionError(code=code, message=message)
    return ExtractionCell(
        doc_id=cell.doc_id,
        column_id=cell.column_id,
        schema_version=cell.schema_version,
        status="error",
        error=err,
        model=cell.model,
    )


def _classify_error(exc: Exception) -> ExtractionErrorCode:
    """Map a raised exception to an ExtractionErrorCode.

    Call.chat_async wraps provider failures in ``CallError`` via
    ``raise CallError(...) from e``, so the original
    :class:`~kaos_llm_client.errors.KaosLLMProviderError` / ``AuthError`` /
    ``TransportError`` lives on ``exc.__cause__``. Unwrap one level before
    classifying.
    """
    from kaos_llm_client.errors import (
        KaosLLMAuthError,
        KaosLLMProviderError,
        KaosLLMTransportError,
    )

    # Unwrap CallError → provider exception for classification.
    candidates: list[BaseException] = [exc]
    cause = exc.__cause__
    if isinstance(cause, BaseException):
        candidates.append(cause)

    for e in candidates:
        if isinstance(e, KaosLLMAuthError):
            return "model_api_error"
        if isinstance(e, KaosLLMProviderError | KaosLLMTransportError):
            return "model_api_error"
        if isinstance(e, ValidationRetryExhaustedError):
            return "schema_validation_failed"
        if isinstance(e, TimeoutError):
            return "timeout"
    return "other"


# =====================================================================
# Corpus fan-out (WS-TR.PR-3) — extract_corpus + CorpusInputSource
# =====================================================================


class CorpusInputSource:
    """``BatchInputSource`` that iterates a corpus for structured extraction.

    Accepts any of:

    - ``dict[doc_id, text]`` — explicit doc-id → source-text mapping.
    - ``kaos_content.corpus.Corpus`` — any implementation of the Corpus
      Protocol. Doc text is concatenated from ``iter_passages()`` per doc,
      matching the convention used by the WS-3.7 multiformat benchmark.

    ``custom_id`` is the doc_id verbatim — resume and per-doc skip-set logic
    in ``batch_run`` works without modification.

    The iterator is deterministic across runs given a stable corpus input.
    """

    def __init__(
        self,
        corpus: dict[str, str] | Any,
        *,
        describe_label: str = "corpus",
    ) -> None:
        self._corpus = corpus
        self._label = describe_label

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        from kaos_llm_core.programs.batch.models import BatchItem

        if isinstance(self._corpus, dict):
            for doc_id in self._corpus:
                text = self._corpus[doc_id]
                yield BatchItem(
                    custom_id=doc_id,
                    inputs={"source_text": text, "doc_id": doc_id},
                )
            return

        # Corpus protocol fallback. Concatenate passages per doc_id,
        # preserving source order. Lazy import to avoid a hard dep.
        try:
            from kaos_content.corpus import Corpus as _Corpus  # noqa: F401
        except ImportError as exc:  # pragma: no cover — kaos-content is a hard dep
            msg = (
                "CorpusInputSource received a non-dict corpus but kaos-content is "
                "not importable. Pass a dict[doc_id, text] instead."
            )
            raise RuntimeError(msg) from exc

        # Group passages by doc_uri while preserving first-seen order.
        by_doc: dict[str, list[str]] = {}
        order: list[str] = []
        for passage in self._corpus.iter_passages():
            uri = passage.doc_uri
            if uri not in by_doc:
                by_doc[uri] = []
                order.append(uri)
            by_doc[uri].append(passage.text)
        for uri in order:
            yield BatchItem(
                custom_id=uri,
                inputs={"source_text": "\n\n".join(by_doc[uri]), "doc_id": uri},
            )

    def describe(self) -> dict[str, Any]:
        size = (
            len(self._corpus)
            if isinstance(self._corpus, dict)
            else getattr(self._corpus, "size", lambda: None)()
        )
        return {
            "type": "kaos_tr.corpus_input_source",
            "label": self._label,
            "n_docs": size,
        }


async def extract_corpus(
    schema: ExtractionSchema | type[BaseModel] | dict[str, Any] | type[Signature],
    corpus: dict[str, str] | Any,
    *,
    output_dir: str,
    model: str | None = None,
    provenance: Provenance = "cited",
    max_retries: int = 2,
    refusal: Literal["field", "row", "never"] = "field",
    max_concurrency: int = 8,
    error_policy: Literal["continue", "stop", "stop_after_n"] = "continue",
    max_errors: int | None = None,
    resume: bool = True,
    item_timeout_s: float | None = None,
) -> CorpusExtractionResult:
    """Fan an :class:`Extract` program out over a corpus with ``batch_run``.

    Composes the WS-TR :class:`Extract` primitive with the shipped
    :func:`batch_run` (resumable JSONL log, content-addressed ``program_hash``,
    cost attribution via :class:`TrialRunner`) into a single call that:

    1. Builds an :class:`Extract` program against ``schema`` + ``model``.
    2. Iterates ``corpus`` via :class:`CorpusInputSource` to produce one
       ``BatchItem`` per doc.
    3. Runs the batch with ``resume=True`` — a killed-then-restarted run
       produces byte-identical output.
    4. Reads the JSONL log back into ``ExtractionResult`` instances and
       assembles a :class:`~kaos_content.model.tabular.TabularDocument` via
       :meth:`TabularDocument.from_cells`.

    Args:
        schema: Any form accepted by :class:`Extract`
            (``ExtractionSchema`` / ``dict`` / Pydantic class / Signature).
        corpus: ``dict[doc_id, text]`` or a ``Corpus`` Protocol instance.
        output_dir: Where ``batch_run`` writes ``items.jsonl`` + manifest.
        model: Provider/model (e.g. ``"anthropic:claude-haiku-4-5"``).
        provenance: ``"none" | "cited" | "grounded"`` — same semantics as
            :class:`Extract`.
        max_retries: On ``"grounded"``, max span-verification retries.
        refusal: Refusal policy for individual cells.
        max_concurrency: Max parallel LLM calls (default 8).
        error_policy: ``batch_run`` error policy. ``stop_after_n`` requires
            ``max_errors``.
        max_errors: Count cap for ``stop_after_n``.
        resume: When ``True`` (default), resume an interrupted run from
            the JSONL log. False forces a fresh run (will error if the
            log already exists and ``program_hash`` mismatches).
        item_timeout_s: Per-doc timeout in seconds. ``None`` = no timeout.

    Returns:
        A :class:`CorpusExtractionResult` bundling the per-doc
        :class:`ExtractionResult` instances and the assembled
        :class:`TabularDocument`.
    """
    from kaos_content.model.tabular import TabularDocument

    from kaos_llm_core.programs.batch import batch_run

    extractor = Extract(
        schema,
        model=model,
        provenance=provenance,
        max_retries=max_retries,
        refusal=refusal,
    )
    source = CorpusInputSource(corpus)

    batch_kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "max_concurrency": max_concurrency,
        "error_policy": error_policy,
        "resume": resume,
    }
    if max_errors is not None:
        batch_kwargs["max_errors"] = max_errors
    if item_timeout_s is not None:
        batch_kwargs["item_timeout_s"] = item_timeout_s

    batch_result = await batch_run(extractor, source, **batch_kwargs)

    # Hydrate per-doc ExtractionResult rows from the JSONL log.
    from pathlib import Path

    log_path = Path(output_dir) / "items.jsonl"
    results: list[ExtractionResult] = []
    if log_path.exists():
        import json

        with log_path.open(encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("kind") != "item":
                    continue
                if record.get("status") != "success":
                    # Skip error rows — they show up in batch_result summary.
                    continue
                output = record.get("output")
                if not isinstance(output, dict):
                    continue
                try:
                    results.append(ExtractionResult.model_validate(output))
                except Exception as exc:  # pragma: no cover — defensive
                    logger.warning(
                        "extract_corpus: failed to hydrate row %s: %s",
                        record.get("custom_id"),
                        exc,
                    )

    # Map ColumnSpec → TabularDocument ColumnType. Reuses the Tier-4
    # extraction types (VERBATIM_QUOTE, MONEY, SCORE, ENTITY_ROLE) plus
    # Tier-1/2/3 equivalents for string/number/date/list/object columns.
    column_specs = tuple(
        (col.id, _column_type_for_tabular(col.column_type)) for col in extractor.schema.columns
    )
    # Flatten every cell from every row.
    all_cells = tuple(c for r in results for c in r.cells)
    table = (
        TabularDocument.from_cells(
            all_cells,
            column_specs=column_specs,
            table_name=extractor.schema.id,
        )
        if all_cells
        else TabularDocument()
    )

    return CorpusExtractionResult(
        table=table,
        rows=tuple(results),
        schema_id=extractor.schema.id,
        schema_version=extractor.schema.version,
        n_docs_processed=len(results),
        n_docs_errored=batch_result.n_errored,
        n_docs_skipped=batch_result.n_skipped,
        cost_usd=batch_result.cost_usd,
        tokens_total=batch_result.total_tokens,
        output_dir=str(output_dir),
    )


def _column_type_for_tabular(column_type: str) -> Any:
    """Map an extraction ``ColumnTypeName`` → kaos-content ``ColumnType``."""
    from kaos_content.model.tabular import ColumnType

    mapping: dict[str, ColumnType] = {
        "string": ColumnType.TEXT,
        "verbatim_quote": ColumnType.VERBATIM_QUOTE,
        "number": ColumnType.FLOAT,
        "integer": ColumnType.INTEGER,
        "date": ColumnType.DATE,
        "datetime": ColumnType.DATETIME,
        "boolean": ColumnType.BOOLEAN,
        "money": ColumnType.MONEY,
        "score": ColumnType.SCORE,
        "enum": ColumnType.TEXT,
        "list": ColumnType.LIST,
        "object": ColumnType.STRUCT,
        "entity_role": ColumnType.ENTITY_ROLE,
    }
    return mapping.get(column_type, ColumnType.TEXT)


class CorpusExtractionResult(BaseModel):
    """Aggregate result of :func:`extract_corpus`.

    Pydantic so callers can persist a whole corpus run (including the
    nested ``ExtractionResult`` rows) via ``model_dump_json()`` for
    post-hoc analysis.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    table: Any
    """The assembled :class:`~kaos_content.model.tabular.TabularDocument`.
    Typed as ``Any`` to avoid a hard-Pydantic round-trip dependency on
    kaos-content's frozen dataclass-based ``Table``."""

    rows: tuple[ExtractionResult, ...]
    schema_id: str
    schema_version: int
    n_docs_processed: int
    n_docs_errored: int
    n_docs_skipped: int
    cost_usd: float
    tokens_total: int
    output_dir: str
