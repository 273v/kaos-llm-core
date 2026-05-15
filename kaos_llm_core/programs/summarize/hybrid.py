"""Hybrid extractive→abstractive summarization (plan §6.1).

:class:`HybridSummary` is the "rank first, then summarise the
survivors" Program. The extractive top-k pre-filter caps the LLM
cost at ``O(top_k)`` sentences regardless of source length, and the
picks are verbatim so the downstream abstractive call has a tight,
on-source context. Pair with :class:`CitedSummary` (default) for
spans that tie back to the picked sentences.
"""

from __future__ import annotations

from typing import Any

from kaos_llm_client import BaseProviderClient
from kaos_llm_client.settings import KaosLLMSettings

from kaos_llm_core.codecs.base import Codec
from kaos_llm_core.programs.base import Program
from kaos_llm_core.programs.summarize.abstractive import AbstractiveSummary, CitedSummary
from kaos_llm_core.programs.summarize.extractive import ExtractiveSummary, Ranker
from kaos_llm_core.results import SourceSpan, Summary
from kaos_llm_core.settings import KaosLLMCoreSettings
from kaos_llm_core.signatures.grounding import GroundedAnswer
from kaos_llm_core.types import Example


class HybridSummary(Program):
    """Extractive top-k → abstractive summary of the picks.

    Args:
        ranker: A :class:`Ranker` (canonical:
            ``kaos_nlp_transformers.extraction.ExtractiveRanker``).
            Used by the inner :class:`ExtractiveSummary` to pick the
            top-``top_k`` salient sentences.
        top_k: Number of sentences the extractive stage emits to the
            abstractive stage. Default ``5``.
        diversify: MMR diversity parameter forwarded to
            ``ExtractiveSummary``. ``0`` is pure salience.
        cited: When ``True`` (default), route the abstractive step
            through :class:`CitedSummary` so every claim ties back to
            the picked sentences via verified spans. When ``False``,
            use :class:`AbstractiveSummary` and return a plain
            ``Summary[str]``.
        model / codec / client / settings / core_settings / examples /
            instructions / max_retries / **kwargs: forwarded to the
            abstractive sub-Program.

    The Program returns:

    - ``Summary[str]`` when ``cited=False``.
    - ``Summary[GroundedAnswer[str]]`` when ``cited=True`` — the
      abstractive payload is the full grounded answer; the curated
      ``source_spans`` reflects only verified claims (same contract
      as :class:`CitedSummary`).

    The ``method`` field is ``"hybrid"`` and ``metadata["program"]``
    is ``"HybridSummary"``. ``source_spans`` is the union of the
    extractive picks' offsets and the cited summary's verified spans
    (the extractive picks are always retained — they are the input
    to the abstractive stage and therefore part of the result's
    provenance even when the LLM didn't quote them directly).
    """

    def __init__(
        self,
        *,
        ranker: Ranker,
        top_k: int = 5,
        diversify: float = 0.0,
        cited: bool = True,
        model: str | None = None,
        codec: Codec | None = None,
        client: BaseProviderClient | None = None,
        settings: KaosLLMSettings | None = None,
        core_settings: KaosLLMCoreSettings | None = None,
        examples: list[Example] | None = None,
        instructions: str | None = None,
        max_retries: int | None = None,
        **kwargs: Any,
    ) -> None:
        self._cited = cited
        self._top_k = top_k
        # Extractive stage. Auto-registers as a Program child via
        # ``Program.__setattr__`` so the trace tree shows the
        # extract→summarize wiring.
        self.extract = ExtractiveSummary(
            ranker=ranker,
            top_k=top_k,
            diversify=diversify,
        )
        # Abstractive stage. Use CitedSummary by default — the whole
        # point of the hybrid path is to keep the abstractive output
        # tied to the source through the extractive picks.
        abstractive_kwargs: dict[str, Any] = {
            "model": model,
            "codec": codec,
            "client": client,
            "settings": settings,
            "core_settings": core_settings,
            "examples": examples,
            "instructions": instructions,
            "max_retries": max_retries,
            **kwargs,
        }
        # ``CitedSummary`` and ``AbstractiveSummary`` are sibling
        # Programs (not a sub-class relationship), so the attribute
        # is typed at the ``Program`` boundary and the call sites
        # discriminate via ``self._cited``.
        self.summarize: Program
        if cited:
            self.summarize = CitedSummary(**abstractive_kwargs)
        else:
            self.summarize = AbstractiveSummary(**abstractive_kwargs)

    async def forward(  # ty: ignore[invalid-method-override]
        self,
        *,
        text: str,
        parent_id: str | None = None,
    ) -> Summary[str] | Summary[GroundedAnswer[str]]:
        # 1. Extractive pre-filter.
        picks = await self.extract(text=text, parent_id=parent_id)
        if not picks.text:
            return Summary[str](
                text="",
                method="hybrid",
                chunks_used=[parent_id] if parent_id else [],
                metadata={
                    "program": "HybridSummary",
                    "top_k": self._top_k,
                    "cited": self._cited,
                    "picks.count": 0,
                },
            )

        # 2. Abstractive summary over the joined picks. CitedSummary
        # returns Summary[GroundedAnswer[str]]; AbstractiveSummary
        # returns Summary[str]. We preserve the payload type by not
        # touching it here.
        abstractive = await self.summarize(text=picks.text, parent_id=parent_id)

        # 3. Pool source spans. The extractive picks always survive
        # (they are the abstractive's input); any verified cited
        # spans from the abstractive step are appended.
        extractive_spans = list(picks.source_spans)
        abstractive_spans = list(abstractive.source_spans)
        # Dedup by (start, end, source_id) preserving order.
        seen: set[tuple[int, int, str | None]] = set()
        pooled: list[SourceSpan] = []
        for span in extractive_spans + abstractive_spans:
            key = (span.start, span.end, span.source_id)
            if key in seen:
                continue
            seen.add(key)
            pooled.append(span)

        meta: dict[str, Any] = {
            **dict(abstractive.metadata),
            "program": "HybridSummary",
            "top_k": self._top_k,
            "cited": self._cited,
            "picks.count": len(extractive_spans),
            "extract.metadata": dict(picks.metadata),
        }
        return abstractive.model_copy(
            update={
                "method": "hybrid",
                "source_spans": pooled,
                "chunks_used": (
                    [parent_id] if parent_id and parent_id not in abstractive.chunks_used else []
                )
                + list(abstractive.chunks_used),
                "metadata": meta,
            }
        )


__all__ = ["HybridSummary"]
