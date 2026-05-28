"""Grounding primitives — Span, Claim, Answer, GroundedAnswer.

This module defines the three-tier trust hierarchy that supersedes
the v0 ``GroundedAnswer[T]`` skeleton.  See
``docs/design/kaos-agents-grounding.md`` for the full design
rationale.  The short version:

- ``Span`` is the atomic unit of evidence: a verifiable substring in
  a source document.  Cannot exist without coordinates; the quote
  and its hash are carried inside the type so verification does not
  require re-fetching the source.
- ``Claim`` is an atomic proposition backed by one or more Spans.
  Carries a ``claim_type`` tag (factual / temporal / normative /
  counterfactual / modal / quantitative) that hints at how it
  should be evaluated, plus optional bitemporal validity
  (``valid_from`` / ``valid_to``).
- ``Answer[T]`` is a typed response containing one or more Claims
  plus a value of whatever shape ``T`` the Signature needs.  The
  claims justify the value and form the audit trail.
- ``InsufficientEvidence`` is a formal refusal carrying
  ``attempted_claims`` and a ``missing`` list so a replan loop can
  see *what* the agent tried and *what* it needed.
- ``GroundedAnswer[T]`` is the discriminated union of ``Answer[T]``
  and ``InsufficientEvidence``, discriminator field ``kind``.

Every tier exposes a ``verify`` method for byte-for-byte checking
against a corpus.  The lesson from kelvin's graceful-degradation
failure mode is that verification must be a method on the type,
not a utility in another module — otherwise it gets forgotten.

Example (simple factual Q&A)::

    from kaos_llm_core.signatures import (
        Signature, InputField, OutputField, GroundedAnswer,
    )

    class AnswerFromCorpus(Signature):
        '''Answer the question from the provided corpus or refuse.'''
        question: str = InputField()
        corpus: str = InputField()
        result: GroundedAnswer[str] = OutputField()

Example (multi-claim structured extraction)::

    class ExtractObligations(Signature):
        '''Extract all obligations from the contract.'''
        contract: str = InputField()
        result: GroundedAnswer[list[str]] = OutputField()

    # Answer returned will have claims=[...] — one per obligation —
    # each with its own supporting_spans citing the clause that
    # creates the obligation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from kaos_nlp_core.algorithms import longest_common_substring_length, token_jaccard
from kaos_nlp_core.matching import substring_find_all_case_insensitive
from kaos_nlp_core.tokenizer import Tokenizer
from pydantic import BaseModel, ConfigDict, Field, model_validator

# PEP 695 type parameter syntax is used throughout:
#   class Answer[T](BaseModel): ...
#   type GroundedAnswer[T] = Annotated[...]
# kaos-llm-core already requires Python >=3.13 so this is safe.


# --- Hash utility -----------------------------------------------------


def _hash_quote(quote: str) -> str:
    """Return truncated SHA-256 of a quote string (hex, 16 chars).

    SHA-256 is used for *identity*, not locality-preserving hashing.
    It answers "is this the same span?" not "is this an approximate
    match?" — the fuzzy-match job belongs to the verification layer,
    where kaos-nlp-core's algorithms module handles edit distance,
    Jaccard, Jaro-Winkler, LCS, and friends.
    """
    return hashlib.sha256(quote.encode("utf-8")).hexdigest()[:16]


# --- Match strategies -------------------------------------------------


class MatchStrategy(StrEnum):
    """How a :meth:`Span.verify` call should compare quote to source.

    Ordered roughly by strictness, from paranoid to lenient.  Each
    strategy is implemented via kaos-nlp-core primitives (Rust-backed
    where applicable); there is no hand-rolled normalization logic.

    Values:
        STRICT: byte-for-byte at the exact ``char_span`` position.
            The most paranoid check; fails on any edit.  Use when the
            source is canonical and stable (statutes, RFCs).
        SUBSTRING: byte-for-byte, but anywhere in the source.  Uses
            Python's ``in`` operator.  Tolerates position drift from
            LLMs that miscount characters but quote faithfully.
        CASE_INSENSITIVE: case-folded substring via
            ``kaos_nlp_core.matching.substring_find_all_case_insensitive``.
            Tolerates position + case drift.
        NORMALIZED_TOKEN: tokenize both sides via
            ``kaos_nlp_core.tokenizer.Tokenizer(lowercase=True)`` and
            check if the quote tokens appear as a contiguous
            subsequence in the source tokens.  Tolerates position,
            case, whitespace, and punctuation drift.
        FUZZY_CHAR: character-level similarity via
            ``kaos_nlp_core.algorithms.jaro_winkler``, compared
            against ``threshold``.  Good for short quotes with
            character edits (typos, OCR errors).
        FUZZY_TOKEN: token-level Jaccard via
            ``kaos_nlp_core.algorithms.token_jaccard``, compared
            against ``threshold``.  Good for quotes with extra /
            missing words but equivalent content.
    """

    STRICT = "strict"
    SUBSTRING = "substring"
    CASE_INSENSITIVE = "case_insensitive"
    NORMALIZED_TOKEN = "normalized_token"
    FUZZY_CHAR = "fuzzy_char"
    FUZZY_TOKEN = "fuzzy_token"


# Default escalation: try exact-positional first, fall back to
# exact-substring.  Matches the behavior of the pre-strategy version
# of Claim.verify so existing callers see no behavior change.
DEFAULT_MATCH_STRATEGIES: tuple[MatchStrategy, ...] = (
    MatchStrategy.STRICT,
    MatchStrategy.SUBSTRING,
)


# Shared lowercase tokenizer used by NORMALIZED_TOKEN strategy.  Module
# scoped so we construct the Rust-backed Tokenizer once.
_NORMALIZED_TOKENIZER = Tokenizer(lowercase=True)


# Unicode normalization table for NORMALIZED_TOKEN.  Maps curly
# quotes, em-dashes, and other common Unicode variants to their ASCII
# equivalents.  This lets the tokenizer produce the same tokens
# regardless of whether the source or the LLM used curly vs straight
# quotes.  Applied BEFORE tokenization.
_UNICODE_NORM_TABLE = str.maketrans(
    {
        "\u2018": "'",  # left single curly quote → straight
        "\u2019": "'",  # right single curly quote → straight
        "\u201c": '"',  # left double curly quote → straight
        "\u201d": '"',  # right double curly quote → straight
        "\u2013": "-",  # en-dash → hyphen
        "\u2014": "-",  # em-dash → hyphen
        "\u2212": "-",  # minus sign → hyphen
        "\u00a0": " ",  # non-breaking space → space
        "\ufb01": "fi",  # fi ligature
        "\ufb02": "fl",  # fl ligature
    }
)


def _normalize_unicode(text: str) -> str:
    """Normalize common Unicode variants to ASCII equivalents.

    Used by the NORMALIZED_TOKEN match strategy so curly quotes,
    em-dashes, ligatures, and non-breaking spaces don't cause
    token mismatches between LLM output and source text.
    """
    return text.translate(_UNICODE_NORM_TABLE)


# --- Tier 1: Span -----------------------------------------------------


class Span(BaseModel):
    """Atomic unit of evidence: a verifiable substring in a source.

    A Span is literally "these bytes at this position in this source."
    Coordinates and the quoted text are required — a Span without a
    real substring at a real position is not evidence.

    ``quote_hash`` is **always recomputed from quote** by a model
    validator.  This is deliberate: LLMs cannot reliably compute
    SHA-256, so leaving ``quote_hash`` as an LLM-provided field
    would let them fabricate plausible-looking hex strings that
    don't match the quote.  The hash is derived from the quote on
    every construction regardless of what the LLM (or any caller)
    passes in.  The hash exists only as a stable short identifier
    for logging and deduplication — the authoritative integrity
    check is the byte-for-byte comparison in :meth:`verify`.

    Attributes:
        source_uri: Stable identifier for the source artifact.
            Opaque; may be a URL, ``vfs://`` URI, doc id, etc.
        page: Optional 1-indexed page number within the artifact.
        char_span: Required ``(start, end)`` character offsets.
            Half-open: ``source_text[start:end]`` should equal
            ``quote``.
        quote: The literal substring from the source.  Non-empty.
        quote_hash: Truncated SHA-256 hex digest of ``quote``.
            Always derived from ``quote``; any value passed in is
            silently overwritten.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_uri: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=1)
    char_span: tuple[int, int]
    quote: str = Field(min_length=1)
    quote_hash: str = Field(default="")

    @model_validator(mode="before")
    @classmethod
    def _force_recompute_quote_hash(cls, data: Any) -> Any:
        """Always derive ``quote_hash`` from ``quote``, ignoring input.

        Runs before field validation so whatever the caller passes
        for ``quote_hash`` is silently replaced with the true hash
        of ``quote``.  LLMs cannot fabricate a hash that escapes
        this check.
        """
        if isinstance(data, dict):
            quote = data.get("quote")
            if isinstance(quote, str) and quote:
                data = dict(data)  # don't mutate caller's dict
                data["quote_hash"] = _hash_quote(quote)
        return data

    def verify(
        self,
        source_text: str,
        *,
        strategy: MatchStrategy = MatchStrategy.STRICT,
        threshold: float = 0.9,
    ) -> bool:
        """Check whether the span's quote matches the source under a strategy.

        ``strategy`` is a single :class:`MatchStrategy` value — this
        method does *one* check per call.  For escalation across
        multiple strategies (try strict, then substring, then
        fuzzy), use :meth:`Claim.verify` / :meth:`Answer.verify`,
        which accept a ``strategies`` iterable.

        Args:
            source_text: The source text to check against.
            strategy: The matching strategy to use.  See
                :class:`MatchStrategy`.
            threshold: Minimum similarity score for the ``FUZZY_*``
                strategies (ignored otherwise).  Range ``[0.0, 1.0]``;
                default 0.9.

        Returns:
            ``True`` iff the span matches ``source_text`` under the
            chosen strategy.
        """
        if strategy is MatchStrategy.STRICT:
            start, end = self.char_span
            if not (0 <= start <= end <= len(source_text)):
                return False
            return source_text[start:end] == self.quote

        if strategy is MatchStrategy.SUBSTRING:
            return self.quote in source_text

        if strategy is MatchStrategy.CASE_INSENSITIVE:
            # Rust-backed case-insensitive substring search from
            # kaos-nlp-core.  Returns a list of MatchSpan; any hit
            # means the quote is present.
            hits = substring_find_all_case_insensitive(source_text, self.quote)
            return len(hits) > 0

        if strategy is MatchStrategy.NORMALIZED_TOKEN:
            # Tokenize both sides with lowercase + Unicode
            # normalization + whitespace handling via the Rust
            # tokenizer.  Tolerates case, whitespace, punctuation
            # drift, AND Unicode quote/dash variants (curly vs
            # straight apostrophes, em-dashes vs hyphens, etc.).
            src_norm = _normalize_unicode(source_text)
            q_norm = _normalize_unicode(self.quote)
            src_tokens = _NORMALIZED_TOKENIZER.tokenize_words(src_norm)
            q_tokens = _NORMALIZED_TOKENIZER.tokenize_words(q_norm)
            n = len(q_tokens)
            if n == 0:
                return False
            if len(src_tokens) < n:
                return False
            return any(src_tokens[i : i + n] == q_tokens for i in range(len(src_tokens) - n + 1))

        if strategy is MatchStrategy.FUZZY_CHAR:
            # Character-level local fuzzy match: how much of the
            # quote appears as a contiguous substring of the
            # source?  Uses kaos-nlp-core's Rust-backed
            # ``longest_common_substring_length``, which is a DP
            # algorithm that finds the longest stretch of matching
            # characters between two strings.
            #
            # Scoring: LCS length / len(quote).  A threshold of
            # 0.9 means "90% of the quote's characters appear as a
            # contiguous substring of the source."  Handles typos
            # and small OCR errors gracefully because a single
            # character edit only breaks one LCS segment.
            q_len = len(self.quote)
            if q_len == 0:
                return False
            lcs_len = longest_common_substring_length(self.quote, source_text)
            return (lcs_len / q_len) >= threshold

        if strategy is MatchStrategy.FUZZY_TOKEN:
            # Token-level Jaccard via kaos-nlp-core.  Tolerates
            # token reordering and moderate addition/removal.  This
            # is the most permissive matcher — use with caution.
            return token_jaccard(self.quote, source_text).similarity >= threshold

        msg = f"Unknown MatchStrategy: {strategy}"
        raise ValueError(msg)

    @classmethod
    def from_source(
        cls,
        source_uri: str,
        source_text: str,
        char_span: tuple[int, int],
        *,
        page: int | None = None,
    ) -> Span:
        """Build a Span by extracting the quote from source.

        The canonical constructor for programmatic Span creation.
        Callers pass the source text and a character range; this
        method pulls the substring and assembles a Span whose
        ``quote`` is guaranteed to verify against ``source_text``.
        """
        start, end = char_span
        if not (0 <= start <= end <= len(source_text)):
            raise ValueError(
                f"char_span {char_span} out of range for source of length {len(source_text)}"
            )
        quote = source_text[start:end]
        if not quote:
            raise ValueError(f"char_span {char_span} yields empty quote")
        return cls(
            source_uri=source_uri,
            page=page,
            char_span=char_span,
            quote=quote,
        )


# --- Tier 2: Claim ----------------------------------------------------


ClaimType = Literal[
    "factual",
    "temporal",
    "normative",
    "counterfactual",
    "modal",
    "quantitative",
]
"""Taxonomy of claim types, ported from kelvin-nlp's LogicType.

Different claim types need different verifiers:
    - ``factual``       — span check: does the source assert this?
    - ``temporal``      — date check: is ``valid_from``/``valid_to`` consistent with the source?
    - ``normative``     — rule check: does the source define this as required/forbidden?
    - ``counterfactual``— hypothetical: used for "what if" reasoning
    - ``modal``         — permission/obligation: "may" / "must" / "shall not"
    - ``quantitative``  — numeric check: does the source support this number?

Most claims are ``factual``; the default reflects that.
"""


class Claim(BaseModel):
    """Atomic proposition backed by source spans.

    The minimal semantic unit of a grounded answer.  A Claim is one
    testable assertion with at least one citation.

    Decomposition principle: if a Claim can be split into two
    statements that verify independently, it should be split.
    "Company breached the contract on March 1 because clause 4.2
    was violated" is three claims:
        - factual: "company did X on March 1"
        - normative: "X violates clause 4.2"
        - temporal: "as of March 1"

    Derived claims (quantitative computations, normative inferences,
    temporal interval reasoning) are supported without a separate
    schema: the Claim's ``supporting_spans`` cite the *premises*
    of the derivation, and the ``statement`` describes the
    computation.  For example, a quantitative claim "6 amendments
    have been proposed but not ratified" cites the spans that quote
    "33 proposed" and "27 ratified"; an auditor can re-run the
    subtraction from those cited premises.  This preserves the
    guarantee that every claim is traceable to source text without
    requiring a cross-reference graph among claims.

    Temporal validity uses **half-open intervals**:
    ``[valid_from, valid_to)`` — the claim is true at
    ``valid_from`` and all instants strictly before ``valid_to``.
    At the instant ``valid_to`` the claim is no longer in effect;
    a successor claim (if any) should carry the new
    ``valid_from == old.valid_to``.  ``valid_to = None`` means the
    claim is currently in effect (open-ended interval).

    Attributes:
        statement: Natural-language proposition, non-empty.
            Describes whatever the claim asserts, including any
            derivation over cited premises.
        claim_type: One of :data:`ClaimType`.  Defaults to
            ``"factual"``.
        supporting_spans: At least one :class:`Span` that cites the
            source material justifying ``statement``.  For derived
            claims, these cite the premises of the derivation; the
            derived value itself need not appear in the source.
        relevance: Optional rationale explaining *why* the spans
            support the statement.  Per-claim, not per-span —
            aggregates across all supporting spans.  For derived
            claims, this is the place to document the computation
            (e.g. "33 - 27 = 6").
        confidence: Calibrated confidence in ``[0.0, 1.0]``.
        valid_from: Optional start of temporal validity (inclusive).
            The earliest instant at which the claim is true.
        valid_to: Optional end of temporal validity (**exclusive** —
            half-open interval).  The earliest instant at which the
            claim is NO LONGER true.  ``None`` means the claim is
            currently in effect, with no known end date.
    """

    model_config = ConfigDict(extra="forbid")

    statement: str = Field(min_length=1)
    claim_type: ClaimType = "factual"
    supporting_spans: list[Span] = Field(min_length=1)
    relevance: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    valid_from: datetime | None = None
    valid_to: datetime | None = None

    def verify(
        self,
        corpus: Callable[[str], str] | dict[str, str],
        *,
        strategies: Iterable[MatchStrategy] = DEFAULT_MATCH_STRATEGIES,
        threshold: float = 0.9,
        diagnostics: bool = False,
    ) -> list[SpanError]:
        """Verify every supporting span against the corpus.

        Each span is checked against the corpus under the given
        ``strategies`` in order.  The first strategy that matches
        wins — no error is reported for that span.  If every
        strategy fails, the span is flagged with the diagnostic
        reason that best fits the *final* (most lenient) attempt.

        Args:
            corpus: Either a dict mapping ``source_uri`` to source
                text, or a callable performing the same lookup.
                A ``KeyError`` / ``LookupError`` from the callable
                is caught and reported as ``source_missing``.
            strategies: Ordered iterable of :class:`MatchStrategy`
                values to try.  Default is
                ``(STRICT, SUBSTRING)`` — the pre-strategy
                behavior.  Pass longer escalation chains to
                tolerate case, whitespace, or fuzzy drift.
            threshold: Similarity threshold for the ``FUZZY_*``
                strategies.  Default 0.9.
            diagnostics: If ``True``, populate ``strategy_results``
                on each :class:`SpanError` with per-strategy
                details explaining why each strategy failed.

        Returns:
            A list of :class:`SpanError` entries — empty iff every
            span matched at least one strategy in ``strategies``.
        """
        strategies = tuple(strategies)
        if not strategies:
            msg = "strategies must be a non-empty iterable"
            raise ValueError(msg)

        errors: list[SpanError] = []
        # Use a local typed getter to help ty resolve the Callable|dict union.
        _get: Callable[[str], str] = corpus.__getitem__ if isinstance(corpus, dict) else corpus  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
        for i, span in enumerate(self.supporting_spans):
            try:
                source_text = _get(span.source_uri)
            except (KeyError, LookupError):
                errors.append(SpanError(span_index=i, span=span, reason="source_missing"))
                continue

            matched = False
            strategy_results: list[StrategyResult] = []
            for strategy in strategies:
                hit = span.verify(source_text, strategy=strategy, threshold=threshold)
                if diagnostics:
                    detail = _strategy_failure_detail(span, source_text, strategy, threshold, hit)
                    strategy_results.append(
                        StrategyResult(strategy=strategy.value, matched=hit, detail=detail)
                    )
                if hit:
                    matched = True
                    break
            if matched:
                continue

            # Diagnose: pick the most specific failure reason based
            # on what the strictest strategy would have reported.
            last_strategy = strategies[-1]
            if last_strategy in (MatchStrategy.FUZZY_CHAR, MatchStrategy.FUZZY_TOKEN):
                reason: Literal[
                    "out_of_range",
                    "quote_mismatch",
                    "source_missing",
                    "similarity_below_threshold",
                ] = "similarity_below_threshold"
            else:
                start, end = span.char_span
                if not (0 <= start <= end <= len(source_text)):
                    reason = "out_of_range"
                else:
                    reason = "quote_mismatch"
            errors.append(
                SpanError(
                    span_index=i,
                    span=span,
                    reason=reason,
                    strategy_results=strategy_results,
                )
            )
        return errors


# --- Tier 3: Answer ---------------------------------------------------


class Answer[T](BaseModel):
    """A grounded response carrying a typed value justified by claims.

    For simple questions the ``claims`` list has one entry.  For
    complex questions (chronologies, list extractions, multi-part
    analyses) the claims are the audit trail: each element is an
    atomic assertion that can be verified independently.

    Attributes:
        kind: Discriminant, always ``"answer"``.
        value: The typed answer payload.  Shape is driven by the
            Signature — ``str`` for free-form, ``list[Obligation]``
            for extractions, a structured model for classifications,
            etc.
        claims: At least one :class:`Claim` justifying ``value``.
        confidence: Aggregate calibrated confidence in ``[0.0, 1.0]``.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["answer"] = "answer"
    value: T
    claims: list[Claim] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)

    # --- Convenience accessors ---

    @property
    def spans(self) -> list[Span]:
        """Flat list of every Span across every Claim.

        Useful for consumers that don't care about claim structure
        and just want "all the citations this answer carries."
        """
        return [s for c in self.claims for s in c.supporting_spans]

    @property
    def sources(self) -> set[str]:
        """Set of unique source URIs cited anywhere in the answer."""
        return {s.source_uri for s in self.spans}

    def verify(
        self,
        corpus: Callable[[str], str] | dict[str, str],
        *,
        strategies: Iterable[MatchStrategy] = DEFAULT_MATCH_STRATEGIES,
        threshold: float = 0.9,
    ) -> list[ClaimError]:
        """Verify every span in every claim against the corpus.

        ``strategies`` and ``threshold`` are forwarded to each
        claim's verify() call.  Returns a list of
        :class:`ClaimError` entries — empty iff every span of every
        claim matches at least one strategy.  Does not
        short-circuit; all failures are reported together so
        callers can make holistic decisions (accept partial, reject
        entirely, replan for specific missing sources, etc.).
        """
        errors: list[ClaimError] = []
        for ci, claim in enumerate(self.claims):
            for span_err in claim.verify(corpus, strategies=strategies, threshold=threshold):
                errors.append(
                    ClaimError(
                        claim_index=ci,
                        span_index=span_err.span_index,
                        span=span_err.span,
                        reason=span_err.reason,
                    )
                )
        return errors


class InsufficientEvidence(BaseModel):
    """Formal refusal: the question cannot be answered from context.

    Richer than v0: carries ``attempted_claims`` so a replan loop
    can see what the agent tried, and a ``missing`` list as
    actionable hints for what evidence would resolve the refusal.

    Attributes:
        kind: Discriminant, always ``"insufficient_evidence"``.
        reason: Human-readable explanation.  Should be specific,
            not "I don't know."
        attempted_claims: Optional claims the agent tried to assert
            but could not adequately support.  Each carries its
            own (failed) span set.
        missing: Optional list of short strings naming kinds of
            evidence that would be needed (e.g., "Delaware LLC
            franchise tax schedule", "GDPR Art. 9(2) text").
        what_would_resolve: Optional single-sentence actionable
            hint for a replan loop.

    See also :class:`NeedsAggregation` — used when the underlying
    facts ARE present but the question is comparative/aggregate
    and the answer must be derived rather than retrieved.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["insufficient_evidence"] = "insufficient_evidence"
    reason: str = Field(min_length=1)
    attempted_claims: list[Claim] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    what_would_resolve: str | None = None


class NeedsAggregation(BaseModel):
    """Aggregation-route signal: the corpus has the facts; the answer
    needs to be derived from them, not retrieved.

    Distinct from :class:`InsufficientEvidence`. Insufficient means
    the corpus genuinely lacks the underlying facts. Aggregation means
    the facts ARE present (one fact per retrieved item) but no single
    item *is* the answer — the answer is something the agent must
    compute (max, min, count, average, longest, shortest, etc.) over
    the retrieved values.

    Pattern matching consumers can route on ``kind``:

      - ``"insufficient_evidence"`` → refuse and ask user for more sources
      - ``"needs_aggregation"`` → switch to compute mode (call a
        stats / manifest / sandbox-eval tool over what's already retrieved)

    Empirically, every model from haiku-4-5 through gpt-5.4-mini /
    sonnet-4-6 / gpt-5.5 will collapse aggregation queries to
    ``InsufficientEvidence`` if not told otherwise. Adding this typed
    result + the corresponding ReAct prompt discipline lets the agent
    loop distinguish the two cases instead of refusing both.

    Attributes:
        kind: Discriminant, always ``"needs_aggregation"``.
        reason: Human-readable explanation of why aggregation is
            needed (e.g., "comparing term lengths across 5 contracts —
            no single passage is the longest").
        operation: Optional short name of the aggregate operation
            (e.g., ``"max"``, ``"min"``, ``"count"``, ``"avg"``,
            ``"sort_desc"``). Consumers can use this to pick the
            right downstream tool.
        relevant_values: Optional list of the retrieved values the
            agent identified as inputs to the aggregation (each
            paired with the source span/uri when available, free-form
            otherwise).
        suggested_tool: Optional MCP tool name the consumer should
            call next (e.g., ``"kaos-content-stats"``,
            ``"kaos-retrieval-corpus-manifest"``,
            ``"kaos-tabular-top-k"``, ``"kaos-llm-core-compute"``).
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["needs_aggregation"] = "needs_aggregation"
    reason: str = Field(min_length=1)
    operation: str | None = None
    relevant_values: list[str] = Field(default_factory=list)
    suggested_tool: str | None = None


type GroundedAnswer[T] = Annotated[
    Answer[T] | InsufficientEvidence,
    Field(discriminator="kind"),
]
"""Discriminated-union output type for trust-critical Signatures.

Declare an output field as ``GroundedAnswer[T]`` and Pydantic will
validate the LLM response against the union based on ``kind``.
Returned values are either a fully-typed ``Answer[T]`` (with typed
``value``, at least one :class:`Claim`, and per-span citations) or
an ``InsufficientEvidence`` refusal (with ``attempted_claims`` and
``missing`` hints for replan).

Use cases:
    - ``GroundedAnswer[str]`` — cited free-form text answer
    - ``GroundedAnswer[bool]`` — cited yes/no classification
    - ``GroundedAnswer[list[Obligation]]`` — cited extraction, one
      Claim per extracted item
    - ``GroundedAnswer[RiskScore]`` — cited structured scoring

See :class:`RefusalPolicy` for configuring when downstream programs
should collapse a low-quality answer into ``InsufficientEvidence``.
"""


# --- Cited[T] — lightweight extraction wrapper -------------------------


class Cited[T](BaseModel):
    """A single extracted value backed by source spans.

    ``Cited[T]`` is the lightweight counterpart to ``Answer[T]`` for
    extraction use cases.  Where ``Answer[T]`` carries a full
    ``claims`` hierarchy (decomposed propositions with typed evaluation),
    ``Cited[T]`` carries ``spans`` directly — no intermediate Claim
    layer.

    Use ``Cited[T]`` for individual extracted fields (a date, a clause,
    a dollar amount) where each value maps to one or a few source
    passages.  Use ``Answer[T]`` for complex reasoning outputs where
    claims decompose and verify independently.

    A ``Cited[T]`` can be promoted to a ``Claim`` via :meth:`to_claim`
    when the extracted value needs to enter the full grounding hierarchy
    (e.g., stored in an agent's FINDINGS memory section).

    Usage in Signatures::

        class ContractExtraction(Signature):
            '''Extract key terms from the contract.'''
            document: str = InputField()
            effective_date: Cited[str] = OutputField(description="Effective date")
            governing_law: Cited[str] = OutputField(description="Governing law jurisdiction")
            termination_fee: Cited[str] = OutputField(description="Early termination fee")

    Attributes:
        value: The extracted value.
        spans: Source passages that justify ``value``.  At least one.
        confidence: Calibrated confidence in ``[0.0, 1.0]``.
    """

    model_config = ConfigDict(extra="forbid")

    value: T
    spans: list[Span] = Field(min_length=1)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    # --- Convenience accessors ---

    @property
    def sources(self) -> set[str]:
        """Set of unique source URIs cited."""
        return {s.source_uri for s in self.spans}

    @property
    def quote(self) -> str:
        """The first span's quote text (convenience for single-span cases)."""
        return self.spans[0].quote

    # --- Verification ---

    def verify(
        self,
        corpus: Callable[[str], str] | dict[str, str],
        *,
        strategies: Iterable[MatchStrategy] = DEFAULT_MATCH_STRATEGIES,
        threshold: float = 0.9,
    ) -> list[SpanError]:
        """Verify every span against the corpus.

        Delegates to :meth:`Span.verify` for each span, trying
        strategies in order (first match wins).

        Args:
            corpus: Either a ``dict[str, str]`` mapping ``source_uri``
                to source text, or a callable performing the lookup.
            strategies: Match strategies to try in order.
            threshold: Similarity threshold for fuzzy strategies.

        Returns:
            A list of :class:`SpanError` — empty if all spans verified.
        """
        strategies = tuple(strategies)
        errors: list[SpanError] = []
        _get: Callable[[str], str] = corpus.__getitem__ if isinstance(corpus, dict) else corpus  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
        for i, span in enumerate(self.spans):
            try:
                source_text = _get(span.source_uri)
            except (KeyError, LookupError):
                errors.append(SpanError(span_index=i, span=span, reason="source_missing"))
                continue

            for strategy in strategies:
                if span.verify(source_text, strategy=strategy, threshold=threshold):
                    break
            else:
                # All strategies failed
                last = strategies[-1] if strategies else MatchStrategy.STRICT
                if last in (MatchStrategy.FUZZY_CHAR, MatchStrategy.FUZZY_TOKEN):
                    reason: Literal[
                        "out_of_range",
                        "quote_mismatch",
                        "source_missing",
                        "similarity_below_threshold",
                    ] = "similarity_below_threshold"
                else:
                    start, end = span.char_span
                    if not (0 <= start <= end <= len(source_text)):
                        reason = "out_of_range"
                    else:
                        reason = "quote_mismatch"
                errors.append(SpanError(span_index=i, span=span, reason=reason))

        return errors

    # --- Promotion to Claim ---

    def to_claim(
        self,
        statement: str | None = None,
        *,
        claim_type: ClaimType = "factual",
    ) -> Claim:
        """Promote this cited value to a full Claim.

        Useful when an extracted value needs to enter the agent's
        FINDINGS memory section or participate in Answer-level
        verification.

        Args:
            statement: The proposition text.  Defaults to
                ``str(self.value)`` if not provided.
            claim_type: Claim type for typed evaluation dispatch.

        Returns:
            A :class:`Claim` backed by this value's spans.
        """
        return Claim(
            statement=statement or str(self.value),
            claim_type=claim_type,
            supporting_spans=list(self.spans),
            confidence=self.confidence,
        )


# --- Source-uri stamping (dispatcher-owned citation identity) ----------


def stamp_source_uri[T](cited: Cited[T], *, source_uri: str) -> Cited[T]:
    """Return a new ``Cited[T]`` with every span's ``source_uri`` overridden.

    Use this in dispatchers that know the source identity better than
    the LLM does.  The canonical case: an extraction Program calls
    one document at a time; the dispatcher already knows which
    document went in, so the LLM's emitted ``source_uri`` is at best
    redundant and at worst fabricated.  Pydantic validation enforces
    ``source_uri`` is non-empty per :class:`Span`, but cannot enforce
    that the value MATCHES the document the LLM was shown — the
    extraction-time literature documents 3-13% URL-hallucination
    rates on uncontrolled ``source_uri`` outputs.

    This helper closes the gap by stamping the dispatcher-supplied
    identity onto every span post-extraction.  The ``value`` and
    ``confidence`` fields are preserved verbatim; only the
    ``source_uri`` field of each :class:`Span` changes.  ``quote``,
    ``char_span``, ``page``, and ``quote_hash`` are preserved (their
    integrity is enforced by :class:`Span`'s own validators — see
    ``_force_recompute_quote_hash`` and :meth:`Span.verify`).

    Args:
        cited: The value to stamp.  Treated as immutable — a new
            :class:`Cited` is returned; the input is not mutated.
        source_uri: The authoritative identifier the dispatcher
            assigns.  Must be non-empty (Pydantic enforces).

    Returns:
        A new :class:`Cited` whose ``spans`` are new :class:`Span`
        instances with the overridden ``source_uri``.

    Raises:
        pydantic.ValidationError: If ``source_uri`` is empty (per
            :class:`Span`'s ``min_length=1`` constraint).
    """
    stamped_spans = [
        Span(
            source_uri=source_uri,
            page=span.page,
            char_span=span.char_span,
            quote=span.quote,
            # quote_hash is force-recomputed by Span's validator;
            # passing it through is harmless but the validator wins.
        )
        for span in cited.spans
    ]
    return Cited[T](
        value=cited.value,
        spans=stamped_spans,
        confidence=cited.confidence,
    )


# --- Diagnostic helpers -----------------------------------------------


def _strategy_failure_detail(
    span: Span,
    source_text: str,
    strategy: MatchStrategy,
    threshold: float,
    matched: bool,
) -> str:
    """Build a human-readable diagnostic string for a strategy attempt."""
    if matched:
        return f"{strategy.value}: matched"

    q = span.quote
    q_preview = q[:80] + "..." if len(q) > 80 else q

    if strategy is MatchStrategy.STRICT:
        start, end = span.char_span
        if not (0 <= start <= end <= len(source_text)):
            return (
                f"STRICT: char_span ({start}, {end}) out of range (source len={len(source_text)})"
            )
        actual = source_text[start:end]
        actual_preview = actual[:80] + "..." if len(actual) > 80 else actual
        if actual != q:
            # Show where the first difference is
            for j, (a, b) in enumerate(zip(actual, q, strict=False)):
                if a != b:
                    return (
                        f"STRICT: first diff at char {j}: "
                        f"source={a!r} vs quote={b!r}. "
                        f"Source excerpt: {actual_preview!r}"
                    )
            # Length mismatch
            return (
                f"STRICT: length mismatch: source[{start}:{end}] has {len(actual)} chars, "
                f"quote has {len(q)} chars"
            )
        return "STRICT: no match (unknown reason)"

    if strategy is MatchStrategy.SUBSTRING:
        return f"SUBSTRING: quote not found in source. Quote: {q_preview!r}"

    if strategy is MatchStrategy.CASE_INSENSITIVE:
        return f"CASE_INSENSITIVE: no case-insensitive match. Quote: {q_preview!r}"

    if strategy is MatchStrategy.NORMALIZED_TOKEN:
        src_norm = _normalize_unicode(source_text)
        q_norm = _normalize_unicode(q)
        src_tokens = _NORMALIZED_TOKENIZER.tokenize_words(src_norm)
        q_tokens = _NORMALIZED_TOKENIZER.tokenize_words(q_norm)
        return (
            f"NORMALIZED_TOKEN: no token subsequence match. "
            f"Quote tokens ({len(q_tokens)}): "
            f"{q_tokens[:10]}{'...' if len(q_tokens) > 10 else ''}. "
            f"Source has {len(src_tokens)} tokens"
        )

    if strategy is MatchStrategy.FUZZY_CHAR:
        q_len = len(q)
        if q_len > 0:
            lcs_len = longest_common_substring_length(q, source_text)
            score = lcs_len / q_len
            return f"FUZZY_CHAR: LCS score {score:.3f} < threshold {threshold}"
        return "FUZZY_CHAR: empty quote"

    if strategy is MatchStrategy.FUZZY_TOKEN:
        result = token_jaccard(q, source_text)
        return f"FUZZY_TOKEN: Jaccard {result.similarity:.3f} < threshold {threshold}"

    return f"{strategy.value}: no match"


# --- Error types ------------------------------------------------------


class StrategyResult(BaseModel):
    """Result of a single match-strategy attempt during verification."""

    model_config = ConfigDict(extra="forbid")

    strategy: str
    """Name of the strategy tried (e.g. 'STRICT', 'SUBSTRING')."""

    matched: bool
    """Whether the strategy matched."""

    detail: str = ""
    """Human-readable explanation of why it failed or matched."""


class SpanError(BaseModel):
    """A single span verification failure.

    ``reason`` diagnoses *why* the span did not match the source:
        - ``out_of_range``: ``char_span`` does not fit inside the
          source text.
        - ``quote_mismatch``: position is in range but the bytes
          differ; exact-match strategies (STRICT / SUBSTRING /
          CASE_INSENSITIVE / NORMALIZED_TOKEN) failed.
        - ``source_missing``: the corpus has no entry for the
          span's ``source_uri``.
        - ``similarity_below_threshold``: a FUZZY_* strategy was
          attempted and the similarity score fell below the
          configured threshold.
    """

    model_config = ConfigDict(extra="forbid")

    span_index: int
    span: Span
    reason: Literal[
        "out_of_range",
        "quote_mismatch",
        "source_missing",
        "similarity_below_threshold",
    ]
    strategy_results: list[StrategyResult] = Field(default_factory=list)
    """Per-strategy diagnostic results. Empty unless diagnostic mode is enabled."""


class ClaimError(BaseModel):
    """A span verification failure enriched with its claim index."""

    model_config = ConfigDict(extra="forbid")

    claim_index: int
    span_index: int
    span: Span
    reason: Literal[
        "out_of_range",
        "quote_mismatch",
        "source_missing",
        "similarity_below_threshold",
    ]


# --- Refusal policy configuration -------------------------------------


class RefusalPolicy(BaseModel):
    """Configuration for how a program should enforce refusal.

    Orthogonal to the type system: the type ``GroundedAnswer[T]``
    guarantees the *shape* of the output; ``RefusalPolicy`` governs
    *when* downstream code should collapse a low-quality Answer
    into ``InsufficientEvidence``.

    Attributes:
        min_confidence: Minimum ``Answer.confidence`` below which a
            program should re-classify the answer as
            ``InsufficientEvidence``.  Default 0.7 follows
            ``docs/competitive/capabilities/18-refuses-when-uncertain.md``.
        require_verification: If ``True``, programs should run
            ``answer.verify(corpus)`` after decode and collapse to
            ``InsufficientEvidence`` on any span error.
        min_spans_per_claim: Minimum number of supporting spans
            required on each Claim.  The schema already enforces 1;
            set higher (2-3) for workflows that want cross-citation
            redundancy.
        calibration: Confidence calibration strategy.
            ``self_report`` trusts the LLM's confidence value;
            ``judged`` has a judge re-score the answer; ``ensemble``
            runs N models and derives confidence from agreement.
            Interpretation is left to consuming programs.
    """

    model_config = ConfigDict(extra="forbid")

    min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    require_verification: bool = False
    min_spans_per_claim: int = Field(default=1, ge=1)
    calibration: Literal["self_report", "judged", "ensemble"] = "self_report"
    allow_aggregation_route: bool = Field(
        default=False,
        description=(
            "When True, consumers should distinguish ``NeedsAggregation`` "
            "from ``InsufficientEvidence`` (the corpus has the facts but "
            "the answer must be computed). When False (default, "
            "backward-compatible), aggregation gaps collapse to "
            "InsufficientEvidence the same as evidence gaps. Recommended "
            "on for legal / financial / comparative review workflows."
        ),
    )
