"""Zero-shot NLI classification (plan §4.2.3 + Phase 8).

:class:`ZeroShotNLIClassifier` formulates each label as a natural-
language hypothesis (``"This text is about {label}."`` by default) and
asks an NLI scorer to grade ``(premise=text, hypothesis=…)`` for
entailment. The label with the highest entailment probability wins.

The NLI model itself is **not** bundled in :mod:`kaos_llm_core` — the
canonical implementation belongs in :mod:`kaos_nlp_transformers` once
a licence-audited checkpoint ships through that package's registry
(plan §4.2.3, §8 Phase 8). This Program defines an
:class:`NLIScorer` Protocol that the future
``kaos_nlp_transformers.NliModel`` will satisfy without an explicit
``Protocol`` import, mirroring the
:class:`~kaos_llm_core.programs.classify.prototype.Embedder` /
:class:`~kaos_llm_core.programs.summarize.extractive.Ranker` pattern.
Until that ships, callers wire any object that produces three-class
NLI probabilities — including offline stubs and existing
HuggingFace NLI cross-encoder pipelines they already use.

Decision rule:

- For each label, build the hypothesis via
  ``hypothesis_template.format(label.prompt_text)``.
- Compute ``P(entailment), P(neutral), P(contradiction)`` via
  :meth:`NLIScorer.score`. Pick the label maximising
  ``P(entailment)``.
- Optional ``min_score`` floor abstains when the top entailment
  probability is below the threshold and the LabelSet permits
  abstention.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from kaos_llm_core.errors import CallError
from kaos_llm_core.labels import ABSTAIN_LABEL, Label, LabelSet
from kaos_llm_core.programs.base import Program
from kaos_llm_core.results import Classification

DEFAULT_HYPOTHESIS_TEMPLATE = "This text is about {}."
"""Sane default hypothesis template (plan §4.2.3).

Callers with domain-specific phrasings (``"This contract clause covers
{}."``) override via the ``hypothesis_template`` constructor arg.
"""


@runtime_checkable
class NLIScore(Protocol):
    """Structural type for a single NLI score record.

    Mirrors the shape any transformer NLI head produces: three
    probabilities summing to (approximately) ``1.0``. Implementations
    that emit logits should pass them through a softmax before
    handing the result back to :class:`ZeroShotNLIClassifier`.
    """

    entailment: float
    neutral: float
    contradiction: float


@runtime_checkable
class NLIScorer(Protocol):
    """Structural type for an NLI scorer.

    Implementations score a single premise against a list of
    hypotheses in one call so the model can amortise its forward
    pass; :class:`ZeroShotNLIClassifier` issues exactly one
    :meth:`score` call per :meth:`forward`.
    """

    def score(
        self,
        premise: str,
        hypotheses: Sequence[str],
    ) -> Sequence[NLIScore]:  # pragma: no cover - protocol
        ...


class ZeroShotNLIClassifier(Program):
    """Zero-shot classifier via natural-language entailment.

    Args:
        labels: Label space. Exclusive single-label only at 0.1.0a10
            — multi-label entailment with thresholding is a future
            extension once a calibrated NLI model lands in
            ``kaos-nlp-transformers``.
        scorer: Object conforming to :class:`NLIScorer`.
        hypothesis_template: Format string with one ``{}`` placeholder
            for the label's ``prompt_text``. Default
            :data:`DEFAULT_HYPOTHESIS_TEMPLATE`.
        min_score: Optional abstention floor in ``[0, 1]``. When the
            top entailment probability is strictly less than
            ``min_score`` and ``labels.allow_abstain`` is ``True``,
            the Program abstains. ``None`` (default) disables the
            floor.

    The Program holds no :class:`~kaos_llm_core.programs.call.Call`
    children — it runs entirely on the supplied NLI scorer.
    :meth:`Program.invoke` builds a childless trace with zero token
    usage so the surface is type-stable with the LLM-backed
    classifiers.

    Returns:
        :class:`Classification` whose ``scores`` map carries each
        label's entailment probability. ``metadata["program"]`` is
        ``"ZeroShotNLIClassifier"`` and ``metadata["top_score"]``
        holds the picked label's entailment probability. The full
        three-class distribution per label is preserved in
        ``metadata["nli"]`` as a list of
        ``{"label": …, "entailment": …, "neutral": …,
        "contradiction": …}`` records (in LabelSet declaration
        order).
    """

    def __init__(
        self,
        *,
        labels: LabelSet,
        scorer: NLIScorer,
        hypothesis_template: str = DEFAULT_HYPOTHESIS_TEMPLATE,
        min_score: float | None = None,
    ) -> None:
        if not labels.exclusive:
            raise CallError(
                "ZeroShotNLIClassifier requires an exclusive LabelSet. "
                "Multi-label entailment is a future extension."
            )
        if "{}" not in hypothesis_template and "{0}" not in hypothesis_template:
            raise CallError(
                "hypothesis_template must contain a `{}` (or `{0}`) "
                f"placeholder for the label name, got {hypothesis_template!r}."
            )
        if min_score is not None and not (0.0 <= min_score <= 1.0):
            raise ValueError(f"min_score must be in [0, 1] or None, got {min_score}")
        self._label_set = labels
        self._scorer = scorer
        self._hypothesis_template = hypothesis_template
        self._min_score = float(min_score) if min_score is not None else None

    def _abstain(self, *, parent_id: str | None, reason: str) -> Classification[Label]:
        return Classification[Label](
            labels=[],
            scores={ABSTAIN_LABEL: 0.0},
            abstained=True,
            chunks_used=[parent_id] if parent_id else [],
            metadata={
                "program": "ZeroShotNLIClassifier",
                "scorer": type(self._scorer).__name__,
                "abstain_reason": reason,
            },
        )

    async def forward(  # ty: ignore[invalid-method-override]
        self,
        *,
        text: str,
        parent_id: str | None = None,
    ) -> Classification[Label]:
        if not text:
            if self._label_set.allow_abstain:
                return self._abstain(parent_id=parent_id, reason="empty_input")
            first = self._label_set.labels[0]
            return Classification[Label](
                labels=[first],
                scores=dict.fromkeys(self._label_set.names, 0.0),
                abstained=False,
                chunks_used=[parent_id] if parent_id else [],
                metadata={
                    "program": "ZeroShotNLIClassifier",
                    "scorer": type(self._scorer).__name__,
                    "fallback_reason": "empty_input_no_abstain",
                },
            )

        hypotheses = [
            self._hypothesis_template.format(label.prompt_text) for label in self._label_set
        ]
        raw_scores = list(self._scorer.score(text, hypotheses))
        if len(raw_scores) != len(self._label_set.labels):
            raise CallError(
                f"NLIScorer returned {len(raw_scores)} scores for "
                f"{len(self._label_set.labels)} hypotheses — implementations "
                "must produce one score per hypothesis in input order."
            )

        # Entailment-only scores keyed by label name; preserve LabelSet
        # declaration order in the returned ``scores`` map.
        entailment_scores: dict[str, float] = {}
        nli_records: list[dict[str, Any]] = []
        for label, score in zip(self._label_set.labels, raw_scores, strict=True):
            entailment_scores[label.name] = float(score.entailment)
            nli_records.append(
                {
                    "label": label.name,
                    "entailment": float(score.entailment),
                    "neutral": float(score.neutral),
                    "contradiction": float(score.contradiction),
                }
            )

        # Argmax with deterministic tie-break by LabelSet order.
        ordered = sorted(
            entailment_scores.items(),
            key=lambda kv: (-kv[1], self._label_set.names.index(kv[0])),
        )
        top_name, top_score = ordered[0]
        base_meta: dict[str, Any] = {
            "program": "ZeroShotNLIClassifier",
            "scorer": type(self._scorer).__name__,
            "hypothesis_template": self._hypothesis_template,
            "min_score": self._min_score,
            "top_score": top_score,
            "nli": nli_records,
        }

        if (
            self._min_score is not None
            and top_score < self._min_score
            and self._label_set.allow_abstain
        ):
            return Classification[Label](
                labels=[],
                scores=entailment_scores,
                abstained=True,
                chunks_used=[parent_id] if parent_id else [],
                metadata={**base_meta, "abstain_reason": "below_min_score"},
            )

        picked = self._label_set.by_name(top_name)
        return Classification[Label](
            labels=[picked],
            scores=entailment_scores,
            abstained=False,
            chunks_used=[parent_id] if parent_id else [],
            metadata=base_meta,
        )


__all__ = [
    "DEFAULT_HYPOTHESIS_TEMPLATE",
    "NLIScore",
    "NLIScorer",
    "ZeroShotNLIClassifier",
]
