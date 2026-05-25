"""QueryExpander -- two-stage retrieval via LLM query expansion.

Ports the core insight from kelvin-agent's search actions: search with the
*document's* vocabulary, not just the *user's* vocabulary.  kelvin's
``IdentifyTokensAction`` inferred search terms from context before every
retrieval.  This module provides the same capability as a lightweight
``Signature`` + ``Call`` pair that plugs into ``RAG``.

Example::

    expander = QueryExpander(model="anthropic:claude-haiku-4-5")
    queries = await expander.expand("What are the environmental impacts of deforestation?")
    # ["What are the environmental impacts of deforestation?",
    #  "environmental effects deforestation",
    #  "habitat loss biodiversity decline forest clearing",
    #  "carbon emissions soil erosion watershed degradation",
    #  "tropical forest destruction ecological consequences"]
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from kaos_core.logging import get_logger

from kaos_llm_core.programs.call import Call
from kaos_llm_core.signatures.fields import InputField, OutputField
from kaos_llm_core.signatures.signature import Signature
from kaos_llm_core.types import Example

logger = get_logger(__name__)


class ExpandQuery(Signature):
    """Generate alternative search queries for a question.

    Given a user's question, produce 3-5 alternative phrasings that
    use different vocabulary to search for the same information.
    Include the original terms AND synonyms, related terms, and
    domain-specific terminology that source documents might use.

    Think about: technical vs. plain language, acronyms vs. expanded
    forms, formal vs. informal terms, and broader/narrower concepts.
    """

    question: str = InputField(description="The user's question")
    queries: list[str] = OutputField(
        description="3-5 alternative search queries using different vocabulary"
    )


@runtime_checkable
class QueryExpander(Protocol):
    """Protocol for query expansion strategies.

    Any object that can expand a user question into multiple search
    queries.  The RAG program accepts this protocol so callers can
    plug in LLM-based expansion, synonym tables, user-feedback
    reformulation, or any other strategy.
    """

    async def expand(self, question: str) -> list[str]:
        """Return the original question plus expanded variants."""
        ...


class LLMQueryExpander:
    """Expands a user query into multiple search queries via an LLM.

    The core insight from kelvin-agent's two-stage retrieval: search
    with the document's vocabulary, not just the user's vocabulary.
    """

    def __init__(
        self,
        model: str,
        *,
        max_queries: int = 5,
        examples: list[Example] | None = None,
    ) -> None:
        """Construct the expander.

        Args:
            model: Provider:model string for the underlying ``Call``.
            max_queries: Hard cap on returned variants (original
                question included). Default 5.
            examples: Optional few-shot grounding examples forwarded to
                the inner ``Call(ExpandQuery, ...)``. Required by
                callers that enforce a grounded-Signature contract
                (e.g. kaos-agents'
                ``Call(SigClass, examples=load_examples("..."))``
                pattern) so query-expansion samples carry the same
                calibration the non-wrapper path uses. Default
                ``None`` preserves the prior behaviour.
        """
        self._call = Call(ExpandQuery, model=model, examples=examples)
        self._max_queries = max_queries

    async def expand(self, question: str) -> list[str]:
        """Return the original question plus expanded variants.

        The original question is always first.  Duplicates and
        near-duplicates (case-insensitive) are removed before
        truncation so every slot carries a distinct query.
        """
        result = await self._call(question=question)

        # Dedup: keep first occurrence, case-insensitive
        seen: set[str] = set()
        unique: list[str] = []
        for q in result.queries:
            key = q.strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(q)

        # Ensure original question is always first
        orig_key = question.strip().lower()
        if orig_key in seen:
            unique = [q for q in unique if q.strip().lower() != orig_key]
        unique.insert(0, question)

        return unique[: self._max_queries]


__all__ = [
    "ExpandQuery",
    "LLMQueryExpander",
    "QueryExpander",
]
