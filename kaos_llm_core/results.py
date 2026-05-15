"""Result containers for summarization and classification programs.

A :class:`Summary` wraps the natural-language summary text plus an
optional typed payload (e.g., a Pydantic schema instance), provenance
back to the source chunks, and the method tag.

A :class:`Classification` wraps the picked labels, an optional score
map, an abstention flag, optional rationale, and provenance.

Both types are Pydantic models so they serialize cleanly through MCP
tools, CLI ``--json``, batch run output, and observability traces.
Both are generic on the structured payload type so callers using
schema-driven summarization/classification keep their static types.
"""

from __future__ import annotations

from typing import Any, Literal

from kaos_core.types.content import KaosModel
from pydantic import Field

from kaos_llm_core.labels import Label

SummaryMethod = Literal["abstractive", "extractive", "hybrid"]
"""Tag describing how a :class:`Summary` was produced."""


class SourceSpan(KaosModel):
    """Half-open ``(start, end)`` reference to a region in a source.

    Lightweight provenance pointer. Distinct from
    :class:`kaos_llm_core.signatures.grounding.Span` — that type carries
    a quoted substring + hash for byte-for-byte verification and is
    used inside :class:`~kaos_llm_core.signatures.Cited` payloads.
    ``SourceSpan`` is a positional reference only, used when the
    caller already has the source text or the chunk id and only needs
    "where in the source".

    Attributes:
        start: Half-open start offset in the source's character space.
        end: Half-open end offset (exclusive).
        source_id: Optional identifier of the source document or
            parent chunk. ``None`` is allowed when the source is
            unambiguous from context (a single-doc summary, for
            example).
    """

    start: int = Field(ge=0)
    end: int = Field(ge=0)
    source_id: str | None = None

    def model_post_init(self, _context: Any, /) -> None:
        if self.end < self.start:
            raise ValueError(f"end ({self.end}) must be >= start ({self.start})")

    @property
    def length(self) -> int:
        return self.end - self.start


class Summary[T](KaosModel):
    """A summary of source text.

    Attributes:
        text: The natural-language summary.
        payload: Optional structured payload. ``T`` is the schema type
            for schema-driven summarization (e.g., a Pydantic model
            extracting parties, dates, obligations from a contract).
            Plain abstractive summaries leave ``payload`` as ``None``.
        method: How this summary was produced. See
            :data:`SummaryMethod`.
        depth: Recursive depth. ``0`` for a summary produced directly
            from source chunks; ``n > 0`` for a summary produced by a
            reducer at level ``n``.
        chunks_used: Identifiers of the :class:`~kaos_nlp_core.chunking.Chunk`
            instances this summary was drawn from. Empty for
            single-shot summaries over source text.
        source_spans: Positional references into the source. Optional;
            populated when the chunker passes spans through to the
            result.
        metadata: Free-form annotations. Programs should namespace
            keys to avoid collisions ("router.model_id",
            "reducer.branching", etc.).
    """

    text: str
    payload: T | None = None
    method: SummaryMethod = "abstractive"
    depth: int = 0
    chunks_used: list[str] = Field(default_factory=list)
    source_spans: list[SourceSpan] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Classification[L: (Label, str)](KaosModel):
    """A classification result.

    Attributes:
        labels: The picked labels. Length 0..1 for exclusive
            classification, 0..n for multi-label. Element type ``L``
            is either :class:`~kaos_llm_core.labels.Label` (when the
            caller wants the full record back) or :class:`str` (when
            the caller only needs the name).
        scores: Optional confidence map. Keys are label *names*
            regardless of whether ``labels`` contains :class:`Label`
            instances or strings. Values are in ``[0.0, 1.0]`` for
            probabilistic classifiers; arbitrary positive floats for
            score-based classifiers. Empty when the underlying
            classifier does not produce scores.
        abstained: ``True`` when the classifier chose to return no
            label. Only meaningful when the consumed
            :class:`~kaos_llm_core.labels.LabelSet` has
            ``allow_abstain=True``.
        rationale: Optional natural-language justification. Populated
            by LLM-backed classifiers when prompted for one.
        chunks_used: Identifiers of the chunks this result was drawn
            from. Empty for single-shot classification over source
            text.
        source_spans: Optional positional references for the spans of
            source text that supported the decision.
        metadata: Free-form annotations.
    """

    labels: list[L] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)
    abstained: bool = False
    rationale: str | None = None
    chunks_used: list[str] = Field(default_factory=list)
    source_spans: list[SourceSpan] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def names(self) -> list[str]:
        """Return label names as strings, regardless of ``L``.

        Convenience accessor that hides whether the caller chose to
        carry :class:`Label` records or bare names in ``labels``.
        """
        out: list[str] = []
        for label in self.labels:
            if isinstance(label, Label):
                out.append(label.name)
            elif isinstance(label, str):
                out.append(label)
            else:  # pragma: no cover - exhaustive over ``L`` bound
                raise TypeError(f"unsupported label payload: {type(label)!r}")
        return out

    @property
    def top_label(self) -> str | None:
        """Return the highest-scoring label name, or ``None`` if empty.

        For exclusive classifiers with no scores, returns the first
        picked label. For abstained results, returns ``None``.
        """
        if self.abstained or not self.labels:
            return None
        if self.scores:
            return max(self.scores, key=lambda k: self.scores[k])
        return self.names[0]


__all__ = [
    "Classification",
    "SourceSpan",
    "Summary",
    "SummaryMethod",
]
