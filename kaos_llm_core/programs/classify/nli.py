"""Zero-shot NLI classification (plan §4.2.3 + Phase 8).

:class:`ZeroShotNLIClassifier` formulates each label as a natural-
language hypothesis (``"This text is about {label}."`` by default) and
asks an NLI scorer to grade ``(premise=text, hypothesis=…)`` for
entailment. The label with the highest entailment probability wins.

Claim verification (:func:`verify_claim` / :func:`verify_claims`) reuses
the same scorer for fact-checking: evidence is the premise, the claim is
the hypothesis, and the argmax over the three-class distribution decides
support / refutation / neutrality.

Hybrid fallback (opt-in)
------------------------
NLI cross-encoders sometimes return a high-confidence ``neutral`` verdict
on a claim that the evidence actually supports but phrases very
differently — a lexical or structural mismatch the entailment head does
not bridge. To recover those cases, :func:`verify_claim` /
:func:`verify_claims` accept an optional second-opinion ``judge`` (any
object satisfying :class:`ClaimJudge`) and an optional ``embedder`` (the
:class:`~kaos_llm_core.programs.classify.prototype.Embedder` Protocol),
both injected by the caller so the core stays backend-agnostic and free
of any provider dependency. The escalation is narrow and cost-gated:

- Confident entailment or contradiction verdicts are returned unchanged —
  NLI is reliable there and never escalates.
- Only a high-confidence ``neutral`` verdict (neutral probability at or
  above ``neutral_floor``) is eligible to escalate.
- When an ``embedder`` is supplied, the embedding cosine between claim and
  evidence acts as a cheap pre-filter: escalate to the judge only when the
  cosine is at or above ``gate`` (enough semantic overlap to be worth a
  closer look); below the gate the NLI ``neutral`` verdict is kept and no
  judge call is made. Cosine only gates the judge — it never flips a
  verdict on its own, because a claim and its negation both sit close to
  the evidence in embedding space.
- The judge's supported / refuted / neutral outcome then becomes the
  verdict. With no judge supplied, behaviour is exactly the pure-NLI path.

Each verdict records which method decided it (:attr:`ClaimVerdict.method`
is ``"nli"`` or ``"llm_judge"``), the gate cosine when one was computed,
and any judge rationale, so callers can audit and cost-account the
escalation. All claims still share a single batched scorer forward pass;
only the neutral-band subset escalates.

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

import numpy as np
from kaos_nlp_core.similarity import cosine as _cosine

from kaos_llm_core.errors import CallError
from kaos_llm_core.labels import ABSTAIN_LABEL, Label, LabelSet
from kaos_llm_core.programs.base import Program
from kaos_llm_core.programs.classify.prototype import Embedder
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

    Members are read-only (covariant): callers only ever *read* these
    probabilities, so any object that exposes them — including frozen
    dataclass stubs — satisfies the protocol.
    """

    @property
    def entailment(self) -> float: ...
    @property
    def neutral(self) -> float: ...
    @property
    def contradiction(self) -> float: ...


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

    The provenance fields (``method`` / ``gate_cosine`` / ``rationale``)
    are appended after the original fields with defaults that reproduce
    the pure-NLI verdict, so adding them is non-breaking: existing
    positional and keyword construction sites still work, and two
    pure-NLI verdicts that were equal before remain equal (they all carry
    the same defaults). Extending the frozen dataclass keeps the single
    ``ClaimVerdict`` return type rather than introducing a richer subtype
    callers would have to branch on.
    """

    claim: str
    label: str
    """One of ``"entailment"`` / ``"neutral"`` / ``"contradiction"``."""
    confidence: float
    """Probability of ``label`` — the max of the three-class distribution."""
    entailment: float
    neutral: float
    contradiction: float
    method: str = "nli"
    """Which path decided this verdict: ``"nli"`` or ``"llm_judge"``."""
    gate_cosine: float | None = None
    """Claim/evidence embedding cosine, when the gate was evaluated; else ``None``."""
    rationale: str | None = None
    """Optional judge rationale when the verdict came from a ``ClaimJudge``."""

    @property
    def supported(self) -> bool:
        """``True`` iff the evidence entails (supports) the claim."""
        return self.label == "entailment"

    @property
    def refuted(self) -> bool:
        """``True`` iff the evidence contradicts (refutes) the claim."""
        return self.label == "contradiction"


@runtime_checkable
class JudgeVerdict(Protocol):
    """Structural type for a :class:`ClaimJudge` second-opinion result.

    The judge returns a three-way ``label`` for whether the evidence
    supports the claim, a ``confidence`` in ``[0, 1]``, and an optional
    short ``rationale``. ``label`` is mapped onto the NLI verdict space:
    ``"supported"`` → ``entailment``, ``"refuted"`` → ``contradiction``,
    ``"neutral"`` → ``neutral`` (synonyms ``"entailment"`` /
    ``"contradiction"`` are also accepted).

    Members are read-only (covariant): the verdict is consumed, not
    mutated, so frozen concrete/stub types satisfy the protocol.
    """

    @property
    def label(self) -> str: ...
    @property
    def confidence(self) -> float: ...
    @property
    def rationale(self) -> str | None: ...


@runtime_checkable
class ClaimJudge(Protocol):
    """Structural type for an LLM second opinion on a claim.

    Implementations grade a single ``(claim, evidence)`` pair and return a
    :class:`JudgeVerdict`. The judge is injected by the caller so the core
    stays backend-agnostic and free of any provider dependency; a natural
    in-repo implementation wraps a :class:`~kaos_llm_core.programs.call.Call`
    over a small :class:`~kaos_llm_core.signatures.signature.Signature`,
    but :func:`verify_claim` / :func:`verify_claims` depend only on this
    Protocol, never on a concrete backend.
    """

    def judge(
        self,
        claim: str,
        evidence: str,
    ) -> JudgeVerdict:  # pragma: no cover - protocol
        ...


# Judge label vocabulary → NLI verdict label. Accept both the
# fact-checking phrasing (supported / refuted) and the raw NLI phrasing.
_JUDGE_LABEL_MAP: dict[str, str] = {
    "supported": "entailment",
    "entailment": "entailment",
    "refuted": "contradiction",
    "contradiction": "contradiction",
    "neutral": "neutral",
}


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


def _embedding_cosine(embedder: Embedder, claim: str, evidence: str) -> float:
    """Cosine similarity between the claim and evidence embeddings.

    Embeds both in a single batched :meth:`Embedder.embed` call and uses
    the Rust-backed :func:`kaos_nlp_core.similarity.cosine` primitive (the
    same module :mod:`prototype` already wires in). ``cosine`` (not the
    pre-normalised fast path) is used so a non-unit-norm row never yields
    a silently wrong value.
    """
    matrix = np.asarray(embedder.embed([claim, evidence]), dtype=np.float32)
    claim_vec = np.ascontiguousarray(matrix[0])
    evidence_vec = np.ascontiguousarray(matrix[1])
    return float(_cosine(claim_vec, evidence_vec))


def _judge_to_verdict(
    claim: str,
    result: JudgeVerdict,
    *,
    gate_cosine: float | None,
) -> ClaimVerdict:
    """Map a :class:`JudgeVerdict` onto a ``method="llm_judge"`` verdict.

    The judge reports only the winning label and its confidence, so the
    three-class distribution is reconstructed by placing ``confidence`` on
    the winning class and splitting the remainder evenly across the other
    two. This keeps :attr:`ClaimVerdict.confidence` consistent with the
    distribution without claiming false precision the judge did not give.
    """
    raw = str(result.label).strip().lower()
    label = _JUDGE_LABEL_MAP.get(raw)
    if label is None:
        raise CallError(
            f"ClaimJudge returned unrecognised label {result.label!r}; expected one of "
            "supported/refuted/neutral (or entailment/contradiction)."
        )
    confidence = float(result.confidence)
    rest = max(0.0, (1.0 - confidence) / 2.0)
    dist = {"entailment": rest, "neutral": rest, "contradiction": rest}
    dist[label] = confidence
    return ClaimVerdict(
        claim=claim,
        label=label,
        confidence=confidence,
        entailment=dist["entailment"],
        neutral=dist["neutral"],
        contradiction=dist["contradiction"],
        method="llm_judge",
        gate_cosine=gate_cosine,
        rationale=result.rationale,
    )


def verify_claims(
    scorer: NLIScorer,
    claims: Sequence[str],
    evidence: str,
    *,
    judge: ClaimJudge | None = None,
    embedder: Embedder | None = None,
    confident: float = 0.5,
    neutral_floor: float = 0.5,
    gate: float = 0.5,
) -> list[ClaimVerdict]:
    """Check each claim against a body of evidence in one scorer call.

    Reframes NLI for fact-checking: ``evidence`` is the premise and each
    claim is a hypothesis, so each verdict's
    :attr:`~ClaimVerdict.label` says whether the evidence **entails**
    (supports), **contradicts** (refutes), or is **neutral** on that
    claim. Because all claims share the one premise they batch into a
    single :meth:`NLIScorer.score` call (one model forward pass), not one
    call per claim.

    With no ``judge`` and no ``embedder`` this is the pure-NLI reduction.
    When a ``judge`` is supplied, a high-confidence ``neutral`` verdict is
    escalated for a second opinion (see the module docstring); an optional
    ``embedder`` first gates that escalation on claim/evidence embedding
    cosine so the expensive judge runs only when the two are semantically
    close. The one batched scorer call still covers all claims; only the
    neutral-band subset escalates.

    Args:
        scorer: any object satisfying the :class:`NLIScorer` Protocol
            (e.g. ``kaos_nlp_transformers.NliModel``).
        claims: the claim texts to check, each against ``evidence``.
        evidence: the premise — the source / authority text.
        judge: optional :class:`ClaimJudge` second opinion. When ``None``,
            no escalation happens and the pure-NLI verdict is returned.
        embedder: optional :class:`Embedder` used as the cost gate; when
            ``None`` the judge is consulted directly on every neutral-band
            claim.
        confident: NLI confidence at or above which an entailment or
            contradiction verdict is trusted and never escalated, in
            ``[0, 1]``. Default ``0.5``.
        neutral_floor: neutral probability at or above which a ``neutral``
            verdict is eligible to escalate, in ``[0, 1]``. Default
            ``0.5``.
        gate: embedding-cosine threshold in ``[-1, 1]``; the judge is
            consulted only when claim/evidence cosine is at or above this
            value. Ignored when no ``embedder`` is supplied. Default
            ``0.5``.

    Returns:
        One :class:`ClaimVerdict` per claim, in input order. Empty input
        returns ``[]`` without calling the scorer.

    Raises:
        CallError: if the scorer returns a number of scores that does not
            match the number of claims, or the judge returns an
            unrecognised label.
        ValueError: if ``confident`` / ``neutral_floor`` are outside
            ``[0, 1]`` or ``gate`` is outside ``[-1, 1]``.
    """
    if not (0.0 <= confident <= 1.0):
        raise ValueError(f"confident must be in [0, 1], got {confident}")
    if not (0.0 <= neutral_floor <= 1.0):
        raise ValueError(f"neutral_floor must be in [0, 1], got {neutral_floor}")
    if not (-1.0 <= gate <= 1.0):
        raise ValueError(f"gate must be in [-1, 1], got {gate}")
    if not claims:
        return []
    scores = list(scorer.score(evidence, claims))
    if len(scores) != len(claims):
        raise CallError(
            f"NLIScorer returned {len(scores)} scores for {len(claims)} claims — "
            "implementations must produce one score per hypothesis in input order."
        )

    verdicts = [_to_verdict(claim, score) for claim, score in zip(claims, scores, strict=True)]
    if judge is None:
        return verdicts

    for i, verdict in enumerate(verdicts):
        # Eligibility for a second opinion:
        #   - a high-confidence ``neutral`` (neutral prob >= neutral_floor),
        #     the lexical/structural-mismatch band the judge recovers; or
        #   - an entailment / contradiction the NLI head is NOT confident
        #     about (confidence < ``confident``).
        # A confident entailment / contradiction is trusted and kept, and a
        # low-confidence neutral is left as NLI neutral — neither spends a
        # judge call.
        if verdict.label == "neutral":
            eligible = verdict.neutral >= neutral_floor
        else:
            eligible = verdict.confidence < confident
        if not eligible:
            continue
        gate_cosine: float | None = None
        if embedder is not None:
            gate_cosine = _embedding_cosine(embedder, verdict.claim, evidence)
            if gate_cosine < gate:
                # Genuinely unrelated: keep the NLI neutral verdict, but
                # record the cosine that gated the (skipped) judge call.
                verdicts[i] = ClaimVerdict(
                    claim=verdict.claim,
                    label=verdict.label,
                    confidence=verdict.confidence,
                    entailment=verdict.entailment,
                    neutral=verdict.neutral,
                    contradiction=verdict.contradiction,
                    method="nli",
                    gate_cosine=gate_cosine,
                )
                continue
        verdicts[i] = _judge_to_verdict(
            verdict.claim, judge.judge(verdict.claim, evidence), gate_cosine=gate_cosine
        )
    return verdicts


def verify_claim(
    scorer: NLIScorer,
    claim: str,
    evidence: str,
    *,
    judge: ClaimJudge | None = None,
    embedder: Embedder | None = None,
    confident: float = 0.5,
    neutral_floor: float = 0.5,
    gate: float = 0.5,
) -> ClaimVerdict:
    """Check a single claim against a body of evidence.

    Convenience wrapper over :func:`verify_claims` for the one-claim
    case; see it for the premise/hypothesis (evidence/claim) framing and
    the optional hybrid ``judge`` / ``embedder`` fallback.

    Args:
        scorer: any object satisfying the :class:`NLIScorer` Protocol.
        claim: the claim text.
        evidence: the premise the claim is checked against.
        judge: optional :class:`ClaimJudge` second opinion.
        embedder: optional :class:`Embedder` cost gate.
        confident: NLI confidence floor for trusting entail/contradict.
        neutral_floor: neutral probability floor for escalation eligibility.
        gate: embedding-cosine gate threshold.

    Returns:
        A single :class:`ClaimVerdict`.
    """
    return verify_claims(
        scorer,
        [claim],
        evidence,
        judge=judge,
        embedder=embedder,
        confident=confident,
        neutral_floor=neutral_floor,
        gate=gate,
    )[0]


__all__ = [
    "DEFAULT_HYPOTHESIS_TEMPLATE",
    "ClaimJudge",
    "ClaimVerdict",
    "JudgeVerdict",
    "NLIScore",
    "NLIScorer",
    "ZeroShotNLIClassifier",
    "verify_claim",
    "verify_claims",
]
