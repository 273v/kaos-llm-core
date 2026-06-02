"""ClusterNamer -- turn cluster keywords + an exemplar into a human title.

Pairs with the deterministic cluster labellers (``kaos_nlp_core.ctfidf`` /
``kaos_content.cluster.label_clusters``): those produce a cluster's most
distinctive keywords plus a representative member -- the substantive,
zero-cost label. This program is the optional presentation tier that turns
them into a readable topic title, e.g.::

    namer = ClusterNamer(model="anthropic:claude-haiku-4-5")
    await namer.name(["summary judgment", "motion", "dismiss"],
                     excerpt="The court granted the motion ...")
    # "Dispositive Motions"

Model-vs-program split: the keywords are computed in Rust (kaos-nlp-core);
only the human naming -- which genuinely needs a language model -- lives
here as a ``Signature`` + ``Call``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from kaos_llm_core.programs.call import Call
from kaos_llm_core.signatures.fields import InputField, OutputField
from kaos_llm_core.signatures.signature import Signature
from kaos_llm_core.types import Example


class NameCluster(Signature):
    """Write a short, human-readable topic label for a cluster of documents.

    You are given the cluster's most distinctive keywords (ranked, most
    distinctive first) and a short excerpt from one representative document.
    Return a concise title -- at most a few words, in title case, with no
    trailing punctuation -- that names the shared topic. Prefer the specific
    shared subject over a broad category, and rely only on the keywords and
    excerpt provided; do not invent details that are not supported by them.
    """

    keywords: list[str] = InputField(
        description="The cluster's distinctive keywords, most distinctive first"
    )
    excerpt: str = InputField(
        description="A short excerpt from one representative document in the cluster"
    )
    title: str = OutputField(
        description="A concise topic title: a few words, title case, no trailing punctuation"
    )


class ClusterNamer:
    """LLM topic-naming for clusters, over the :class:`NameCluster` Signature.

    Wraps a single ``Call(NameCluster, ...)``. One LLM round-trip per
    cluster; :meth:`name_all` fans out concurrently.
    """

    def __init__(
        self,
        model: str,
        *,
        max_excerpt_chars: int = 600,
        examples: list[Example] | None = None,
        core_settings: Any = None,
    ) -> None:
        """Construct the namer.

        Args:
            model: provider:model string for the underlying ``Call``.
            max_excerpt_chars: truncate the exemplar excerpt to this many
                characters before sending (keeps the prompt bounded).
            examples: optional few-shot grounding examples forwarded to the
                inner ``Call`` (same grounded-Signature contract as the
                other Call-based programs).
            core_settings: optional ``KaosLLMCoreSettings`` forwarded to the
                inner ``Call`` (trace + per-request config propagation).
        """
        self._call = Call(
            NameCluster,
            model=model,
            examples=examples,
            core_settings=core_settings,
        )
        self._max_excerpt_chars = max_excerpt_chars

    async def name(self, keywords: Sequence[str], excerpt: str = "") -> str:
        """Return a concise topic title for one cluster.

        Args:
            keywords: the cluster's keywords, most distinctive first.
            excerpt: text of a representative document (truncated to
                ``max_excerpt_chars``); may be empty.

        Returns:
            The model's title, stripped of surrounding whitespace.
        """
        result = await self._call(
            keywords=list(keywords),
            excerpt=(excerpt or "")[: self._max_excerpt_chars],
        )
        return str(result.title).strip()

    async def name_all(self, clusters: Sequence[tuple[Sequence[str], str]]) -> list[str]:
        """Name many clusters concurrently.

        Args:
            clusters: a sequence of ``(keywords, excerpt)`` pairs.

        Returns:
            One title per input cluster, in order.
        """
        return list(await asyncio.gather(*(self.name(kw, ex) for kw, ex in clusters)))


__all__ = ["ClusterNamer", "NameCluster"]
