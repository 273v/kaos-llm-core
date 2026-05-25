"""RAG — Retrieve -> Reason -> Verify pipeline.

Composes kaos-content retrieval with kaos-llm-core grounding to produce
verified ``GroundedAnswer[T]`` outputs.  The pipeline:

1. **Retrieve** top_k passages from a corpus (via ``search_corpus`` from
   kaos-content, or any ``Retriever`` protocol implementation)
2. **Build context** — assemble passages into ``=== SOURCE: uri ===``
   headed blocks (the same layout the existing ``GroundedQA`` tests use)
3. **Generate** — ``Call(CorpusQA, ...)`` produces a ``GroundedAnswer[str]``
4. **Verify** — ``Answer.verify(source_map)`` checks every citation
5. **Retry** — on verification failure, re-call with error feedback
6. Return the ``RAGResult``

Dependency boundary: the RAG class does NOT import kaos-content at the
module level.  It accepts ``Retriever`` protocol instances and
``dict[str, str]`` corpora.  The kaos-content integration
(``SearchableDocument``) lives in ``kaos_content.search.search_corpus``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from kaos_core.logging import get_logger

from kaos_llm_core.programs.base import Program
from kaos_llm_core.programs.call import Call
from kaos_llm_core.programs.result_mixin import OutputForwardingMixin
from kaos_llm_core.signatures import (
    Answer,
    ClaimError,
    GroundedAnswer,
    InputField,
    InsufficientEvidence,
    MatchStrategy,
    OutputField,
    Signature,
)
from kaos_llm_core.signatures.grounding import DEFAULT_MATCH_STRATEGIES

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Retrieval result — imported lazily to avoid hard dep on kaos-nlp-core
# at module level for environments that only need the Signature/Result types.
# ---------------------------------------------------------------------------

try:
    from kaos_nlp_core.retrieval.protocol import RetrievalResult
except ImportError:  # pragma: no cover
    RetrievalResult = Any  # type: ignore[assignment,misc]  # ty: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------


def _passage_uri(result: Any, index: int = 0) -> str:
    """Build a passage-level URI from a retrieval result.

    Each passage needs a URI that uniquely identifies it within
    its parent document so the LLM can cite it and verification
    checks against the short passage text.

    Precedence:
    1. ``block_ref`` from the AST — but only if it's specific
       enough to distinguish this passage from others.  A
       ``block_ref`` of ``#/body/0`` means the whole document is
       one block, which is not useful for passage-level IDs.
    2. ``char_start`` offset in the original document.
    3. Hash of the passage text (deterministic fallback).
    """
    doc_id = result.doc_id
    meta = result.metadata if hasattr(result, "metadata") and result.metadata else {}

    explicit_uri = meta.get("passage_uri")
    if isinstance(explicit_uri, str) and explicit_uri:
        return explicit_uri

    # If doc_id already contains a fragment (#), it IS a passage URI.
    # Don't append another fragment.
    if "#" in doc_id:
        return doc_id

    # block_ref is useful only if it's more specific than #/body/0
    block_ref = meta.get("block_ref")
    if block_ref and block_ref != "#/body/0":
        return f"{doc_id}{block_ref}"

    # Character offset — unique per passage within a document
    char_start = getattr(result, "char_start", None)
    if char_start is not None:
        return f"{doc_id}#c{char_start}"

    # Fallback: hash of the passage text
    import hashlib

    text_hash = hashlib.sha256(result.text.encode()).hexdigest()[:8]
    return f"{doc_id}#h{text_hash}"


def build_passage_context(
    results: list[RetrievalResult],
    corpus: dict[str, str],
    *,
    max_chars: int = 0,
) -> tuple[str, dict[str, str]]:
    """Build a prompt context string from retrieval results.

    Each retrieved passage gets its own ``=== SOURCE: uri ===``
    header with a **passage-level URI** (e.g.
    ``doc:10k#/body/47``).  The ``source_map`` maps each passage
    URI to its **passage text** — NOT the full document text.

    This is the correct architecture for citation verification:
    the LLM cites text from the passages it saw, and verification
    checks against those same short passages.  Provenance back to
    the original document is carried by the passage URI (which
    includes the block_ref or char offset).

    Args:
        results: Retrieval results carrying ``doc_id`` and ``text``.
        corpus: The full document texts keyed by document URI.
            Used only as fallback context; passage text comes from
            the retrieval results themselves.

    Returns:
        A 2-tuple of ``(context_string, source_map)`` where
        source_map maps **passage URIs** to **passage text**.
    """
    if not results:
        return "", {}

    # Deduplicate passages (same text from same doc) while preserving
    # rank order.  Use passage URI as the dedup key.
    seen: set[str] = set()
    unique_results: list[tuple[str, Any]] = []
    for r in results:
        uri = _passage_uri(r)
        if uri not in seen:
            seen.add(uri)
            unique_results.append((uri, r))

    # Sort by document then by position within document, so passages
    # from the same document appear together in reading order.
    unique_results.sort(
        key=lambda pair: (
            pair[1].doc_id,
            pair[1].char_start
            if hasattr(pair[1], "char_start") and pair[1].char_start is not None
            else 0,
        )
    )

    # Build the context string — one SOURCE header per passage.
    # If max_chars is set, include passages in rank order until the
    # budget is exhausted.  Passages that don't fit are dropped
    # (they were lower-ranked anyway).
    parts: list[str] = []
    source_map: dict[str, str] = {}
    total_chars = 0

    for uri, r in unique_results:
        header = f"=== SOURCE: {uri} ==="
        entry_chars = len(header) + 2 + len(r.text)  # +2 for \n\n join
        if max_chars > 0 and total_chars + entry_chars > max_chars and parts:
            # Budget exhausted — stop adding lower-ranked passages.
            # This is a safety net, not the primary mechanism.  The
            # primary mechanism is proper chunking: callers should
            # chunk large documents via SectionChunker BEFORE
            # retrieval so each passage is 2-4KB, not 285KB.
            logger.warning(
                "Context budget reached (%d chars); dropping %d remaining "
                "passages.  If this fires frequently, chunk your documents "
                "before retrieval (e.g., SectionChunker + EmbeddingRetriever).",
                max_chars,
                len(unique_results) - len(source_map),
            )
            break
        parts.append(header)
        parts.append(r.text)
        source_map[uri] = r.text
        total_chars += entry_chars

    context_string = "\n\n".join(parts)
    return context_string, source_map


# ---------------------------------------------------------------------------
# CorpusQA Signature
# ---------------------------------------------------------------------------


class CorpusQA(Signature):
    """Answer the question from the provided corpus passages, or refuse.

    The corpus is divided into passages, each prefixed by a
    ``=== SOURCE: <uri> ===`` header.  Each passage is a short
    excerpt from a larger document.

    If the corpus contains a clear answer, return ``kind: "answer"``
    with:
    - ``value``: the answer in a concise form
    - ``confidence``: your calibrated confidence in [0, 1]
    - ``claims``: one or more Claims, each with a ``statement``,
      a ``claim_type``, and at least one Span in
      ``supporting_spans``.

    Each Span must have:
    - ``source_uri``: copy the EXACT URI from the ``SOURCE`` header
      of the passage you are quoting (e.g. ``doc:10k#/body/47``)
    - ``char_span``: set to ``[0, 0]`` (offsets are tracked
      server-side; you do not need to compute them)
    - ``quote``: an EXACT verbatim substring copied from that
      passage, preserving whitespace and punctuation

    If the corpus does NOT contain enough information, return
    ``kind: "insufficient_evidence"`` with a specific ``reason``.

    Rules:
    - Only quote text that appears verbatim in a passage.
    - Each quote must come from a single passage.
    - If evidence spans two passages, emit two Spans.
    - Never speculate or use outside knowledge.
    """

    question: str = InputField(description="The natural-language question to answer.")
    corpus: str = InputField(
        description=(
            "Source passages with headers. Each passage starts with "
            "'=== SOURCE: <uri> ===' followed by the passage text."
        ),
    )
    result: GroundedAnswer[str] = OutputField(
        description="A grounded answer with cited Spans, or InsufficientEvidence."
    )


# ---------------------------------------------------------------------------
# RAGResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RAGResult(OutputForwardingMixin):
    """Result of a RAG pipeline execution.

    Attribute access falls through to the ``grounded_answer`` via
    :class:`OutputForwardingMixin` (e.g. ``result.value`` reads
    ``result.grounded_answer.value`` when the answer variant is
    ``Answer``).

    Attributes:
        grounded_answer: The ``Answer[str]`` or ``InsufficientEvidence``.
        outputs: Alias for ``grounded_answer`` used by OutputForwardingMixin.
        retrieved_passages: The retrieval results used to build context.
        verification_errors: Span/Claim errors from ``Answer.verify()``.
        context_text: The assembled prompt context string.
        source_map: The ``uri -> full_text`` map for verification.
        is_verified: ``True`` iff verification returned no errors.
        confidence: The answer confidence (0.0 for InsufficientEvidence).
    """

    grounded_answer: Answer[str] | InsufficientEvidence
    outputs: Any
    retrieved_passages: list[Any] = field(default_factory=list)
    verification_errors: list[ClaimError] = field(default_factory=list)
    context_text: str = ""
    source_map: dict[str, str] = field(default_factory=dict)
    is_verified: bool = False
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# RAG Program
# ---------------------------------------------------------------------------


class RAG(Program):
    """Retrieve -> Reason -> Verify pipeline.

    Composes kaos-content retrieval with kaos-llm-core grounding
    to produce verified ``GroundedAnswer[T]`` outputs.

    Usage::

        from kaos_llm_core.programs.rag import RAG

        rag = RAG(model="anthropic:claude-sonnet-4-6")
        result = await rag.query(
            question="What is the project deadline?",
            documents={"doc:spec": "The project deadline is March 2025..."},
        )
        print(result.grounded_answer)
        print(result.is_verified)

    The pipeline does NOT depend on kaos-content at import time.
    When called with a ``dict[str, str]`` corpus, it uses
    ``kaos_content.search.search_corpus`` lazily.  When called with
    a ``Retriever`` protocol instance, it uses that directly.
    """

    def __init__(
        self,
        model: str,
        *,
        retriever: Any | None = None,
        query_expander: Any | None = None,
        reranker: Any | None = None,
        match_strategies: tuple[MatchStrategy, ...]
        | Iterable[MatchStrategy] = DEFAULT_MATCH_STRATEGIES,
        threshold: float = 0.9,
        top_k: int = 10,
        max_context_chars: int = 100_000,
        max_retries: int = 2,
        signature: type[Signature] | None = None,
        core_settings: Any = None,
        **call_kwargs: Any,
    ) -> None:
        """
        Args:
            model: LLM model string (e.g. ``"anthropic:claude-haiku-4-5"``).
            retriever: Optional ``Retriever`` protocol instance (from
                ``kaos_nlp_core.retrieval``).  When provided, used for
                the retrieval step instead of the default
                ``kaos_content.search.search_corpus`` BM25 path.
                Supports ``BM25Retriever``, ``EmbeddingRetriever``,
                ``HybridRetriever``, or any custom implementation.
            query_expander: Optional ``QueryExpander`` protocol instance.
                When provided, the user's question is expanded into
                multiple search queries before retrieval.  Results
                from all queries are merged.
            reranker: Optional ``Reranker`` protocol instance (from
                ``kaos_nlp_core.retrieval``).  When provided, retrieved
                passages are reranked before context assembly.
                Pipeline: retrieve → [expand] → [rerank] → reason → verify.
            match_strategies: Verification match strategies.
            threshold: Verification similarity threshold.
            top_k: Number of passages to retrieve.
            max_retries: Retry count on verification failure.
            signature: Override the default ``CorpusQA`` Signature.
            core_settings: Optional :class:`KaosLLMCoreSettings`
                forwarded to the inner ``Call``. Required for
                ``KAOS_LLM_CORE_TRACE_ENABLED`` and per-request
                ``_meta.kaos_config`` overrides (from the MCP
                wrapper) to reach the QA call. Without it,
                per-request config silently drops on the floor
                — mirrors the Phase 9d wiring in :class:`Judge`.
            **call_kwargs: Additional kwargs forwarded to the inner
                :class:`Call` constructor. This is the entry point for
                grounding the QA call with few-shot examples — pass
                ``examples=load_examples("...")`` here and they flow
                through to the inner ``Call(self._signature, ...)`` so
                callers that enforce a grounded-Signature contract
                (e.g. kaos-agents'
                ``Call(SigClass, examples=load_examples("..."))``
                pattern) preserve calibration when routing through
                :class:`RAG`.
        """
        self._model = model
        self._retriever = retriever
        self._query_expander = query_expander
        self._reranker = reranker
        self._match_strategies = tuple(match_strategies)
        self._threshold = threshold
        self._top_k = top_k
        self._max_context_chars = max_context_chars
        self._max_retries = max_retries
        self._signature = signature or CorpusQA
        self._core_settings = core_settings
        self._call_kwargs = call_kwargs

        # The inner Call is assigned as a public attribute so Program's
        # graph / named_calls / get_learnable_state see it.
        call_init_kwargs: dict[str, Any] = {
            "model": self._model,
            "max_retries": max_retries,
            **call_kwargs,
        }
        if core_settings is not None:
            call_init_kwargs["core_settings"] = core_settings
        self.call = Call(self._signature, **call_init_kwargs)

    async def query(
        self,
        question: str,
        documents: list[Any] | dict[str, str] | Any,
    ) -> RAGResult:
        """The full pipeline: retrieve -> build context -> generate -> verify.

        Args:
            question: The natural-language question.
            documents: Either a list of ``SearchableDocument`` instances,
                a ``dict[str, str]`` mapping ``uri -> text``, or a
                ``kaos_ml_core.Corpus`` instance (AST-grounded).

        Returns:
            A :class:`RAGResult` with the grounded answer and metadata.
        """
        return await self(question=question, documents=documents)

    async def forward(  # ty: ignore[invalid-method-override]
        self,
        *,
        question: str,
        documents: list[Any] | dict[str, str] | Any,
    ) -> RAGResult:
        """Execute the RAG pipeline.

        This is the ``Program.forward()`` implementation; prefer calling
        ``rag.query(question, documents)`` or ``await rag(question=..., documents=...)``.
        """
        # ── 1. Build the source corpus dict ──────────────────────────
        if isinstance(documents, dict):
            corpus = documents
        elif _is_corpus(documents):
            # kaos_ml_core.Corpus — build source_map from units
            corpus = {_corpus_unit_passage_uri(unit): unit.text for unit in documents}
        else:
            # SearchableDocument list — preserve AST-grounded passage units.
            corpus = _searchable_document_corpus(documents)

        # ── 2. Retrieve ─────────────────────────────────────────────
        # When documents is a Corpus and no retriever was provided,
        # lazily build a BM25Retriever from the Corpus.
        if self._retriever is None and _is_corpus(documents):
            from kaos_nlp_core.retrieval.bm25 import BM25Retriever

            self._retriever = BM25Retriever.from_corpus(documents)

        # Query expansion: expand the question into multiple queries
        # before retrieval.  When no expander is set, just use the
        # original question.
        if self._query_expander is not None:
            queries = await self._query_expander.expand(question)
            logger.debug("Expanded query into %d variants: %s", len(queries), queries)
        else:
            queries = [question]

        if self._retriever is not None:
            # Use the plugged-in Retriever (BM25, Embedding, Hybrid, etc.)
            # Retrieve for each expanded query and merge results.
            all_raw_hits: list[Any] = []
            for q in queries:
                raw_hits = await self._retriever.retrieve(q, top_k=self._top_k)
                all_raw_hits.extend(raw_hits)
            # Convert SearchHit → RetrievalResult if needed
            try:
                from kaos_nlp_core.retrieval.protocol import (
                    RetrievalResult as _RR,
                )
                from kaos_nlp_core.retrieval.protocol import (
                    search_hit_to_retrieval_result,
                )
                from kaos_nlp_core.search import SearchHit

                converted = []
                for h in all_raw_hits:
                    if isinstance(h, SearchHit):
                        rr = search_hit_to_retrieval_result(h)
                        # If doc_id is just an int, check metadata for a
                        # more meaningful string id (external_id or doc_id).
                        if rr.doc_id.isdigit() and rr.metadata:
                            better_id = rr.metadata.get("doc_id") or rr.metadata.get("external_id")
                            if better_id:
                                rr = _RR(
                                    text=rr.text,
                                    score=rr.score,
                                    doc_id=better_id,
                                    metadata=rr.metadata,
                                    char_start=rr.char_start,
                                    char_end=rr.char_end,
                                    page=rr.page,
                                )
                        converted.append(rr)
                    else:
                        converted.append(h)
                retrieved = converted
            except ImportError:
                retrieved = all_raw_hits
        else:
            from kaos_content.search import search_corpus as _search_corpus

            all_retrieved: list[Any] = []
            for q in queries:
                hits = await _search_corpus(
                    documents,
                    q,
                    top_k=self._top_k,
                )
                all_retrieved.extend(hits)
            retrieved = all_retrieved

        # Merge multi-query results: deduplicate by passage URI,
        # keeping the highest score across all queries.
        if len(queries) > 1:
            retrieved = _merge_retrieval_results(retrieved, top_k=self._top_k)

        # ── 2b. Rerank (optional) ──────────────────────────────────
        if self._reranker is not None and retrieved:
            try:
                ranked = await self._reranker.rerank(question, retrieved, top_k=self._top_k)
                # RankedResult.result is a RetrievalResult; extract them.
                retrieved = [r.result for r in ranked]
            except Exception:
                logger.warning("Reranker failed; using original retrieval order", exc_info=True)

        # ── 3. Build context ────────────────────────────────────────
        context_text, source_map = build_passage_context(
            retrieved, corpus, max_chars=self._max_context_chars
        )

        # Warn if context is very large — may exceed model context window.
        # Typical context windows: Haiku 200K chars, Sonnet 680K chars.
        # A 100K char context plus system prompt leaves little room.
        ctx_len = len(context_text)
        if ctx_len > 50_000:
            logger.warning(
                "RAG context is %d chars (%d passages). This may approach "
                "model context limits. Consider reducing max_context_chars "
                "or chunking documents more aggressively.",
                ctx_len,
                len(source_map),
            )

        # ── 4. Generate ─────────────────────────────────────────────
        call_result = await self.call(
            question=question,
            corpus=context_text,
        )
        outcome = call_result.result

        # ── 5. Verify ───────────────────────────────────────────────
        # Verification checks each quote against the passage it cites.
        # If a quote is mis-attributed (cites the wrong passage URI but
        # the quote IS in another passage), we fall back to checking
        # against ALL passages — the claim is grounded, just mis-labeled.
        verification_errors: list[ClaimError] = []
        if isinstance(outcome, Answer):
            verification_errors = outcome.verify(
                source_map,
                strategies=self._match_strategies,
                threshold=self._threshold,
            )

            # Retry failed spans against the FULL context string (not just
            # the cited passage).  Catches two failure modes:
            # 1. Mis-attribution: quote is in a different passage than cited
            # 2. Cross-passage quotes: quote spans a passage boundary
            if verification_errors:
                remaining: list[ClaimError] = []
                for err in verification_errors:
                    found = err.span.verify(context_text, strategy=MatchStrategy.SUBSTRING)
                    if not found:
                        found = err.span.verify(
                            context_text, strategy=MatchStrategy.NORMALIZED_TOKEN
                        )
                    if not found:
                        remaining.append(err)
                verification_errors = remaining

            # ── 5b. Retry on verification failure ───────────────────
            if verification_errors and self._max_retries > 0:
                # Build feedback about which spans failed
                failed_quotes = [
                    f"- Claim {e.claim_index}, span {e.span_index}: "
                    f"quote={e.span.quote!r} reason={e.reason}"
                    for e in verification_errors
                ]
                feedback = (
                    "Your previous answer had citation verification errors:\n"
                    + "\n".join(failed_quotes)
                    + "\n\nPlease re-answer, ensuring every Span.quote is an "
                    "EXACT verbatim substring from the source text."
                )

                # Re-call with feedback injected via a cloned Call.
                # clone_call preserves the client + settings from
                # self.call so test doubles (FunctionClient) work.
                from kaos_llm_core.programs.cloning import clone_call

                retry_call = clone_call(self.call)
                retry_call.instructions = self.call.instructions + "\n\n" + feedback
                retry_result = await retry_call(
                    question=question,
                    corpus=context_text,
                )
                retry_outcome = retry_result.result

                if isinstance(retry_outcome, Answer):
                    retry_errors = retry_outcome.verify(
                        source_map,
                        strategies=self._match_strategies,
                        threshold=self._threshold,
                    )
                    # Use the retry if it's better
                    if len(retry_errors) < len(verification_errors):
                        outcome = retry_outcome
                        verification_errors = retry_errors

        # ── 6. Build result ─────────────────────────────────────────
        confidence = 0.0
        if isinstance(outcome, Answer):
            confidence = outcome.confidence

        is_verified = isinstance(outcome, Answer) and len(verification_errors) == 0

        return RAGResult(
            grounded_answer=outcome,
            outputs=outcome,
            retrieved_passages=retrieved,
            verification_errors=verification_errors,
            context_text=context_text,
            source_map=source_map,
            is_verified=is_verified,
            confidence=confidence,
        )


def _merge_retrieval_results(results: list[Any], *, top_k: int) -> list[Any]:
    """Merge multi-query retrieval results by passage URI.

    For each passage, keep the result with the highest score across all
    queries.  Return the top_k results sorted by score descending.
    """
    best: dict[str, Any] = {}
    for r in results:
        uri = _passage_uri(r)
        if uri not in best or r.score > best[uri].score:
            best[uri] = r
    merged = sorted(best.values(), key=lambda r: r.score, reverse=True)
    return merged[:top_k]


def _is_corpus(obj: Any) -> bool:
    """Check if *obj* satisfies the :class:`kaos_content.corpus.Corpus` Protocol.

    Preferred path: ``isinstance(obj, kaos_content.corpus.Corpus)`` — the
    formal ``@runtime_checkable`` Protocol from WS-3 that both
    ``kaos_ml_core.Corpus`` and ``kaos_content.corpus.ContentDocumentCorpus``
    satisfy via ``iter_passages`` / ``get_passage`` / ``size``.

    Fallback duck-type: kept for defensive reasons — kaos-llm-core must
    not hard-depend on kaos-content at import time, and older
    ``kaos_ml_core.Corpus`` revisions (pre-WS-3) predate the Protocol
    alias methods. The legacy check demands a class named ``"Corpus"``
    plus the ``units / unit / __iter__`` attribute triple.
    """
    import contextlib

    _CorpusProtocol: Any = None
    with contextlib.suppress(Exception):
        from kaos_content.corpus import Corpus as _CorpusProtocol
    if _CorpusProtocol is not None and isinstance(obj, _CorpusProtocol):
        return True
    # Legacy duck-type fallback — see docstring.
    cls_name = type(obj).__name__
    if cls_name != "Corpus":
        return False
    return hasattr(obj, "units") and hasattr(obj, "unit") and hasattr(obj, "__iter__")


def _corpus_unit_passage_uri(unit: Any) -> str:
    """Build a passage-level URI from a ``CorpusUnit``.

    Delegates to the canonical ``corpus_unit_passage_uri`` in
    ``kaos_nlp_core.retrieval.protocol``.
    """
    from kaos_nlp_core.retrieval.protocol import corpus_unit_passage_uri

    return corpus_unit_passage_uri(unit)


def _searchable_document_corpus(documents: Iterable[Any]) -> dict[str, str]:
    """Build a passage map from SearchableDocument units without flattening ASTs."""
    corpus: dict[str, str] = {}
    for sdoc in documents:
        source = sdoc.document.metadata.source
        doc_uri = source.uri if source is not None else f"doc:{id(sdoc)}"
        for unit in sdoc.units:
            corpus[
                _searchable_passage_uri(
                    doc_uri=doc_uri,
                    text=unit.text,
                    block_ref=unit.block_ref,
                    char_start=getattr(unit, "char_start", None),
                    level=sdoc.level,
                )
            ] = unit.text
    return corpus


def _searchable_passage_uri(
    *,
    doc_uri: str,
    text: str,
    block_ref: str | None,
    char_start: int | None = None,
    level: str | None = None,
) -> str:
    """Build a stable passage URI for SearchableDocument-backed retrieval."""
    if level == "sentence" and char_start is not None:
        return f"{doc_uri}#c{char_start}"
    if block_ref and block_ref != "#/body/0" and "#" not in doc_uri:
        return f"{doc_uri}{block_ref}"
    if char_start is not None:
        return f"{doc_uri}#c{char_start}"

    import hashlib

    text_hash = hashlib.sha256(text.encode()).hexdigest()[:8]
    return f"{doc_uri}#h{text_hash}"


__all__ = [
    "RAG",
    "CorpusQA",
    "RAGResult",
    "build_passage_context",
]
