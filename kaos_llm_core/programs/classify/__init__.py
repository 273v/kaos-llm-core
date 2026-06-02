"""Classification Programs.

Phases 4 and 5 of the cross-module summarization / classification plan
land seven Programs here:

- :class:`ZeroShotClassify` — single LLM call against a
  :class:`~kaos_llm_core.labels.LabelSet`; constrained decode.
- :class:`FewShotClassify` — :class:`ZeroShotClassify` with a few-shot
  example pool prepended to the prompt.
- :class:`PrototypeClassify` — no-LLM classifier driven by embedding
  cosine against label prototypes (canonical embedder:
  ``kaos_nlp_transformers.EmbeddingModel``); the §6.2 Phase-5
  deliverable that landed alongside ``ExtractiveRanker``.
- :class:`MultiLabelClassify` — LLM emits a *subset* of labels plus
  per-label confidence; consumes a multi-label
  :class:`~kaos_llm_core.labels.LabelSet`.
- :class:`HierarchicalClassify` — coarse-to-fine taxonomy walk over a
  ``hierarchical=True``
  :class:`~kaos_llm_core.labels.LabelSet`.
- :class:`ChunkedClassify` — long-document wrapper that chunks, runs
  any per-chunk classifier in parallel, and combines per-chunk
  :class:`~kaos_llm_core.results.Classification` results via an
  :class:`~kaos_llm_core.composition.Aggregator`.
- :class:`EnsembleClassify` — runs multiple member classifiers over
  the same input and combines them via an
  :class:`~kaos_llm_core.composition.Aggregator`.

All Programs return a
:class:`~kaos_llm_core.results.Classification[Label]`, regardless of
whether the underlying decision was made by the LLM, by a chunked
vote, or by an ensemble.

.. warning::
   Classification outputs are **statistical approximations** from
   upstream LLM providers. They may pick wrong labels, abstain
   when they shouldn't, or produce unreliable confidence scores.
   For decisions with legal, financial, medical, or compliance
   consequences, treat classifier output as a triage signal that
   must be confirmed by a domain expert. See :doc:`/README` for
   the package-level data-handling note (text passed to a Program
   is transmitted to the configured LLM provider).
"""

from __future__ import annotations

from kaos_llm_core.programs.classify.chunked import ChunkedClassify
from kaos_llm_core.programs.classify.ensemble import EnsembleClassify
from kaos_llm_core.programs.classify.hierarchical import HierarchicalClassify
from kaos_llm_core.programs.classify.multi_label import MultiLabelClassify
from kaos_llm_core.programs.classify.nli import (
    DEFAULT_HYPOTHESIS_TEMPLATE,
    ClaimVerdict,
    NLIScore,
    NLIScorer,
    ZeroShotNLIClassifier,
    verify_claim,
    verify_claims,
)
from kaos_llm_core.programs.classify.prototype import (
    Embedder,
    PrototypeClassify,
)
from kaos_llm_core.programs.classify.retrieval import RetrievalClassify
from kaos_llm_core.programs.classify.zero_shot import (
    FewShotClassify,
    ZeroShotClassify,
)

__all__ = [
    "DEFAULT_HYPOTHESIS_TEMPLATE",
    "ChunkedClassify",
    "ClaimVerdict",
    "Embedder",
    "EnsembleClassify",
    "FewShotClassify",
    "HierarchicalClassify",
    "MultiLabelClassify",
    "NLIScore",
    "NLIScorer",
    "PrototypeClassify",
    "RetrievalClassify",
    "ZeroShotClassify",
    "ZeroShotNLIClassifier",
    "verify_claim",
    "verify_claims",
]
