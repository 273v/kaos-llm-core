"""Query-focused summarization (plan §6.1).

:class:`QueryFocusedSummary` takes a ``(text, query)`` pair and
returns a summary biased toward the query. The pipeline is:

1. Segment the source via
   :func:`kaos_nlp_core.segmentation.segment_sentences`.
2. Embed every sentence + the query via the supplied
   :class:`Embedder`.
3. Score each sentence by cosine similarity against the query via
   :func:`kaos_nlp_core.similarity.cosine_one_to_many_normalized` —
   the same Rust-backed SIMD fast path used by
   :class:`~kaos_llm_core.programs.classify.PrototypeClassify`.
4. Pick the top-``top_k`` highest-scoring sentences.
5. Send the joined passages through :class:`CitedSummary` so every
   claim ties back to the source via verified spans.

Acceptance criteria (plan §6.1 / §8.6 D):

> "≥80% of returned spans must verify against the source via
> ``grounded.verify()``."

That guarantee is delivered by :class:`CitedSummary`'s own runtime
verification — this Program does not re-verify; it just routes.

The ``Embedder`` protocol is the same one
:class:`~kaos_llm_core.programs.classify.PrototypeClassify` consumes,
so the canonical implementation in production is
``kaos_nlp_transformers.EmbeddingModel``. Stubs substitute freely in
offline tests.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from kaos_llm_client import BaseProviderClient
from kaos_llm_client.settings import KaosLLMSettings
from kaos_nlp_core.segmentation import segment_sentences
from kaos_nlp_core.similarity import (
    cosine_one_to_many_normalized as _cosine_one_to_many_normalized,
)
from kaos_nlp_core.similarity import (
    l2_normalize_in_place as _l2_normalize_in_place,
)

from kaos_llm_core.codecs.base import Codec
from kaos_llm_core.optimization.budget import Budget
from kaos_llm_core.programs.base import Program
from kaos_llm_core.programs.classify.prototype import Embedder
from kaos_llm_core.programs.summarize.abstractive import (
    AbstractiveSummary,
    CitedSummary,
)
from kaos_llm_core.results import SourceSpan, Summary
from kaos_llm_core.settings import KaosLLMCoreSettings
from kaos_llm_core.signatures.grounding import GroundedAnswer
from kaos_llm_core.types import Example


def _to_contiguous_f32(arr: np.ndarray) -> np.ndarray:
    out = arr if arr.dtype == np.float32 else arr.astype(np.float32, copy=False)
    if not out.flags["C_CONTIGUOUS"]:
        out = np.ascontiguousarray(out)
    return out


class QueryFocusedSummary(Program):
    """Embedding-retrieval + abstractive summarization (plan §6.1).

    Args:
        embedder: A :class:`Embedder` (canonical:
            ``kaos_nlp_transformers.EmbeddingModel``). Must produce
            ``(N, dim)`` ``float32`` arrays; rows should be unit-norm
            (the Program defensively L2-normalises by default).
        top_k: Number of sentences forwarded to the abstractive
            stage. Default ``5``.
        cited: When ``True`` (default), the abstractive step uses
            :class:`CitedSummary` so every claim is verified against
            the picked passages. When ``False``, use plain
            :class:`AbstractiveSummary`.
        normalize: When ``True`` (default), L2-normalise the
            sentence + query embeddings in place before the cosine
            call. Set ``False`` to trust the embedder's unit-norm
            contract.
        joiner: String inserted between picks when materialising the
            passage handed to the abstractive stage. Defaults to
            ``" "`` (single space).
        model / codec / client / settings / core_settings / examples
            / instructions / max_retries / **kwargs: forwarded to
            the abstractive sub-Program.
    """

    def __init__(
        self,
        *,
        embedder: Embedder,
        top_k: int = 5,
        cited: bool = True,
        normalize: bool = True,
        joiner: str = " ",
        budget: Budget | None = None,
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
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k}")
        self._embedder = embedder
        self._top_k = top_k
        self._cited = cited
        self._normalize = normalize
        self._joiner = joiner
        self._budget = budget
        # Cache: ``ChunkCache`` (plan §5.3) is chunk-id-keyed for the
        # chunked-reducer Programs; ``QueryFocusedSummary`` is a
        # single-call path with no chunker, so chunk-id caching does
        # not naturally apply. Callers needing per-(doc, query) reuse
        # should wrap the call themselves; we keep the constructor
        # surface honest by not advertising a cache= param that has
        # no implementation.
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
        # is typed at the ``Program`` boundary.
        self.summarize: Program
        if cited:
            self.summarize = CitedSummary(**abstractive_kwargs)
        else:
            self.summarize = AbstractiveSummary(**abstractive_kwargs)

    async def forward(  # ty: ignore[invalid-method-override]
        self,
        *,
        text: str,
        query: str,
        parent_id: str | None = None,
    ) -> Summary[str] | Summary[GroundedAnswer[str]]:
        if not query:
            raise ValueError("QueryFocusedSummary requires a non-empty query")
        if not text:
            return Summary[str](
                text="",
                method="abstractive",
                chunks_used=[parent_id] if parent_id else [],
                metadata={
                    "program": "QueryFocusedSummary",
                    "query": query,
                    "top_k": self._top_k,
                    "n_sentences": 0,
                    "cited": self._cited,
                },
            )

        sentences = list(segment_sentences(text))
        if not sentences:
            return Summary[str](
                text="",
                method="abstractive",
                chunks_used=[parent_id] if parent_id else [],
                metadata={
                    "program": "QueryFocusedSummary",
                    "query": query,
                    "top_k": self._top_k,
                    "n_sentences": 0,
                    "cited": self._cited,
                },
            )

        # Embed the sentences and the query in a single batch when the
        # embedder supports it; otherwise two calls. Either way the
        # result is two ``(?, dim)`` arrays.
        sentence_texts = [s.text for s in sentences]
        sentence_vecs = _to_contiguous_f32(self._embedder.embed(sentence_texts))
        query_vec = _to_contiguous_f32(self._embedder.embed([query])[0])
        if self._normalize:
            for row in sentence_vecs:
                _l2_normalize_in_place(row)
            _l2_normalize_in_place(query_vec)

        scores = np.asarray(
            _cosine_one_to_many_normalized(query_vec, sentence_vecs),
            dtype=np.float32,
        )
        # Pick the top-k by score; cap at the number of available
        # sentences. The Rust top_k_cosine path is overkill here —
        # numpy ``argpartition`` is fine on the small k we work with.
        cap = min(self._top_k, scores.shape[0])
        if cap == scores.shape[0]:
            order_by_score = np.argsort(-scores)
        else:
            partition = np.argpartition(-scores, cap - 1)[:cap]
            order_by_score = partition[np.argsort(-scores[partition])]

        # Materialise picks in source order so the joined passage
        # reads as a coherent narrative.
        picks_idx = sorted(int(i) for i in order_by_score)
        passage = self._joiner.join(sentences[i].text for i in picks_idx)
        picks_meta = [
            {
                "rank": int(rank),
                "score": float(scores[i]),
                "start": int(sentences[i].start),
                "end": int(sentences[i].end),
            }
            for rank, i in enumerate(order_by_score)
        ]
        pick_spans = [
            SourceSpan(start=sentences[i].start, end=sentences[i].end, source_id=parent_id)
            for i in picks_idx
        ]

        # Budget gate (plan §6.1 / §8.5 P1-8). The Program makes one
        # LLM call, so the cheapest correct semantics is pre-check
        # exhausted() and abort with a partial result; otherwise run
        # the call and consume from the tracker afterward.
        tracker = self._budget.make_tracker() if self._budget is not None else None
        if tracker is not None and tracker.exhausted() is not None:
            return Summary[str](
                text="",
                method="abstractive",
                source_spans=pick_spans,
                chunks_used=[parent_id] if parent_id else [],
                metadata={
                    "program": "QueryFocusedSummary",
                    "query": query,
                    "top_k": self._top_k,
                    "n_sentences": len(sentences),
                    "cited": self._cited,
                    "picks": picks_meta,
                    "partial": True,
                    "budget.exhausted": str(tracker.exhausted()),
                },
            )

        # Hand the joined passage to the abstractive sub-Program.
        # ``parent_id`` is forwarded so the abstractive's
        # ``chunks_used`` carries the original source id. Use
        # ``invoke()`` (not ``__call__``) so we get access to the
        # token-usage breakdown for the budget tracker.
        abstractive_invocation = await self.summarize.invoke(text=passage, parent_id=parent_id)
        abstractive = abstractive_invocation.output
        if tracker is not None:
            tracker.consume(
                cost_usd=float(abstractive_invocation.usage.cost_usd or 0.0),
                tokens=int(abstractive_invocation.usage.total_tokens or 0),
            )

        # Pool the picks' spans with whatever verified spans the
        # abstractive (typically CitedSummary) emitted; dedup by
        # (start, end, source_id).
        pooled: list[SourceSpan] = []
        seen: set[tuple[int, int, str | None]] = set()
        for span in (*pick_spans, *abstractive.source_spans):
            key = (span.start, span.end, span.source_id)
            if key in seen:
                continue
            seen.add(key)
            pooled.append(span)

        meta = {
            **dict(abstractive.metadata),
            "program": "QueryFocusedSummary",
            "query": query,
            "top_k": self._top_k,
            "n_sentences": len(sentences),
            "cited": self._cited,
            "picks": picks_meta,
        }
        if tracker is not None:
            meta["budget.cost_usd"] = round(tracker.cost_usd, 6)
            meta["budget.tokens"] = tracker.tokens
        return abstractive.model_copy(
            update={
                "method": "abstractive",
                "source_spans": pooled,
                "chunks_used": (
                    [parent_id] if parent_id and parent_id not in abstractive.chunks_used else []
                )
                + list(abstractive.chunks_used),
                "metadata": meta,
            }
        )


__all__ = ["QueryFocusedSummary"]
