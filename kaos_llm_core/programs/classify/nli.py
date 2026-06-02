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
from dataclasses import dataclass
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


# --------------------------------------------------------------------------
# Claim verification (fact-checking framing of NLI)
# --------------------------------------------------------------------------
#
# Where :class:`ZeroShotNLIClassifier` asks "which LABEL fits this text?"
# (text = premise, label-templates = hypotheses), claim verification asks
# "does this EVIDENCE support this CLAIM?" (evidence = premise, claim =
# hypothesis). Both are NLI-score → decision reductions over the same
# :class:`NLIScorer` Protocol, so they live together and stay
# backend-agnostic: any scorer that satisfies the Protocol — the
# canonical ``kaos_nlp_transformers.NliModel``, an offline stub, an
# existing HF cross-encoder pipeline — works unchanged. The NLI *model*
# stays in ``kaos-nlp-transformers``; this decision layer stays here.

_CLAIM_LABELS: tuple[str, str, str] = ("entailment", "neutral", "contradiction")
"""Verdict labels in argmax tie-break order — an exact tie resolves to the
earliest (so a flat ``(1/3, 1/3, 1/3)`` distribution → ``"entailment"``),
matching :class:`ZeroShotNLIClassifier`'s deterministic ordering."""


@dataclass(frozen=True, slots=True)
class ClaimVerdict:
    """Whether evidence entails, contradicts, or is neutral on a claim.

    The fact-checking reframing of an NLI score: the premise is the
    *evidence* (source / authority text) and the hypothesis is the
    *claim*. ``label`` is the argmax of the three-class distribution and
    ``confidence`` is that label's probability; the full distribution is
    kept in ``entailment`` / ``neutral`` / ``contradiction``.
    """

    claim: str
    label: str
    """One of ``"entailment"`` / ``"neutral"`` / ``"contradiction"``."""
    confidence: float
    """Probability of ``label`` — the max of the three-class distribution."""
    entailment: float
    neutral: float
    contradiction: float

    @property
    def supported(self) -> bool:
        """``True`` iff the evidence entails (supports) the claim."""
        return self.label == "entailment"

    @property
    def refuted(self) -> bool:
        """``True`` iff the evidence contradicts (refutes) the claim."""
        return self.label == "contradiction"


def _to_verdict(claim: str, score: NLIScore) -> ClaimVerdict:
    """Reduce a single :class:`NLIScore` to a labelled :class:`ClaimVerdict`."""
    probs = (score.entailment, score.neutral, score.contradiction)
    best = max(range(3), key=lambda i: probs[i])  # ties → earliest label
    return ClaimVerdict(
        claim=claim,
        label=_CLAIM_LABELS[best],
        confidence=float(probs[best]),
        entailment=float(score.entailment),
        neutral=float(score.neutral),
        contradiction=float(score.contradiction),
    )


def verify_claims(
    scorer: NLIScorer,
    claims: Sequence[str],
    evidence: str,
) -> list[ClaimVerdict]:
    """Check each claim against a body of evidence in one scorer call.

    Reframes NLI for fact-checking: ``evidence`` is the premise and each
    claim is a hypothesis, so each verdict's
    :attr:`~ClaimVerdict.label` says whether the evidence **entails**
    (supports), **contradicts** (refutes), or is **neutral** on that
    claim. Because all claims share the one premise they batch into a
    single :meth:`NLIScorer.score` call (one model forward pass), not one
    call per claim.

    Args:
        scorer: any object satisfying the :class:`NLIScorer` Protocol
            (e.g. ``kaos_nlp_transformers.NliModel``).
        claims: the claim texts to check, each against ``evidence``.
        evidence: the premise — the source / authority text.

    Returns:
        One :class:`ClaimVerdict` per claim, in input order. Empty input
        returns ``[]`` without calling the scorer.

    Raises:
        CallError: if the scorer returns a number of scores that does not
            match the number of claims.
    """
    if not claims:
        return []
    scores = list(scorer.score(evidence, claims))
    if len(scores) != len(claims):
        raise CallError(
            f"NLIScorer returned {len(scores)} scores for {len(claims)} claims — "
            "implementations must produce one score per hypothesis in input order."
        )
    return [_to_verdict(claim, score) for claim, score in zip(claims, scores, strict=True)]


def verify_claim(scorer: NLIScorer, claim: str, evidence: str) -> ClaimVerdict:
    """Check a single claim against a body of evidence.

    Convenience wrapper over :func:`verify_claims` for the one-claim
    case; see it for the premise/hypothesis (evidence/claim) framing.

    Args:
        scorer: any object satisfying the :class:`NLIScorer` Protocol.
        claim: the claim text.
        evidence: the premise the claim is checked against.

    Returns:
        A single :class:`ClaimVerdict`.
    """
    return verify_claims(scorer, [claim], evidence)[0]


__all__ = [
    "DEFAULT_HYPOTHESIS_TEMPLATE",
    "ClaimVerdict",
    "NLIScore",
    "NLIScorer",
    "ZeroShotNLIClassifier",
    "verify_claim",
    "verify_claims",
]
