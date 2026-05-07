"""Built-in metric library for kaos-llm-core.

Metrics are callables ``(prediction, gold) -> float`` (or ``-> dict[str, float]``
for multi-value metrics like precision/recall/F1) that plug into the optimizer
``metric=`` argument and into ``evaluate()``.

Metrics are split into three modules:

- :mod:`kaos_llm_core.metrics.text` — string equality, regex, contains,
  semantic similarity (factory).
- :mod:`kaos_llm_core.metrics.structured` — accuracy, JSON field match,
  numeric comparisons, precision/recall/F1.
- :mod:`kaos_llm_core.metrics.llm_judge` — :class:`LLMJudge` metric, an
  LLM-as-a-metric scorer (distinct from the :class:`Judge` *program*).

All metrics handle ``None`` gracefully (returning ``0.0``) and accept
non-string inputs by coercing via ``str()`` where appropriate. Errors raise
``TypeError`` with a recovery hint.
"""

from __future__ import annotations

from kaos_llm_core.metrics.llm_judge import LLMJudge
from kaos_llm_core.metrics.retrieval import (
    DEFAULT_RAGAS_WEIGHTS,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
    ragas_composite,
)
from kaos_llm_core.metrics.structured import (
    accuracy,
    json_field_match,
    numeric_close,
    numeric_ratio,
    precision_recall_f1,
)
from kaos_llm_core.metrics.text import (
    case_insensitive_match,
    contains,
    exact_match,
    normalized_match,
    regex_match,
    semantic_similarity,
)

__all__ = [
    "DEFAULT_RAGAS_WEIGHTS",
    "LLMJudge",
    "accuracy",
    "answer_relevancy",
    "case_insensitive_match",
    "contains",
    "context_precision",
    "context_recall",
    "exact_match",
    "faithfulness",
    "json_field_match",
    "normalized_match",
    "numeric_close",
    "numeric_ratio",
    "precision_recall_f1",
    "ragas_composite",
    "regex_match",
    "semantic_similarity",
]
