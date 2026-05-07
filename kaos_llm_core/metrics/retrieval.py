"""Retrieval-augmented generation metrics.

RAGAS-style metrics for evaluating RAG pipelines.  Metrics follow the
same ``(prediction, gold) -> float`` protocol as the rest of
``kaos_llm_core.metrics`` and plug into ``evaluate()`` and
``optimizer.metric=``.

The unique advantage: when predictions use ``GroundedAnswer[T]``, the
:func:`faithfulness` metric uses structural ``Span`` verification
(deterministic, zero LLM cost, auditable) instead of LLM-as-judge NLI.

Five public metric factories:

- :func:`faithfulness` -- structural span verification or LLMJudge fallback
- :func:`context_precision` -- precision of retrieved refs vs. gold-relevant set
- :func:`context_recall` -- recall of retrieved refs vs. gold-relevant set
- :func:`answer_relevancy` -- embedding similarity between answer and question
- :func:`ragas_composite` -- weighted average of the four above

All factories return callables with signature
``(prediction: Any, gold: dict[str, Any]) -> float`` that satisfy the
:data:`~kaos_llm_core.optimization.evaluation.MetricFn` protocol.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from kaos_core.logging import get_logger

from kaos_llm_core.signatures.grounding import (
    DEFAULT_MATCH_STRATEGIES,
    Answer,
    ClaimError,
    InsufficientEvidence,
    MatchStrategy,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_refs(obj: Any, field: str) -> set[str] | None:
    """Extract a set of string references from *obj* by field name.

    Supports dicts (``obj[field]``), attribute objects (``obj.field``),
    and iterables.  Returns ``None`` when the field is absent or not
    iterable.
    """
    raw: Any = None
    if isinstance(obj, dict):
        raw = obj.get(field)
    elif hasattr(obj, field):
        try:
            raw = getattr(obj, field)
        except Exception as exc:
            logger.debug(
                "_extract_refs: getattr(%r, %r) failed: %s", type(obj).__name__, field, exc
            )
            return None
    if raw is None:
        return None
    if isinstance(raw, set):
        return {str(r) for r in raw}
    if isinstance(raw, str):
        # A bare string is a single ref, not an iterable of characters
        return {raw}
    try:
        return {str(r) for r in raw}
    except TypeError:
        return None


def _answer_text(prediction: Any) -> str | None:
    """Extract a plain-text answer string from a prediction.

    Handles ``Answer`` objects (via ``.value``), dicts (common field
    names), and bare strings.  Returns ``None`` when nothing sensible
    can be extracted.
    """
    if prediction is None:
        return None
    if isinstance(prediction, Answer):
        return str(prediction.value)
    if isinstance(prediction, InsufficientEvidence):
        return prediction.reason
    if isinstance(prediction, str):
        return prediction
    if isinstance(prediction, dict):
        for key in ("answer", "value", "text", "output"):
            if key in prediction:
                v = prediction[key]
                return None if v is None else str(v)
        return str(prediction)
    if hasattr(prediction, "value"):
        try:
            return str(prediction.value)
        except Exception as exc:
            logger.debug("_answer_text: access to .value failed: %s", exc)
    return str(prediction)


# ---------------------------------------------------------------------------
# 1. Faithfulness
# ---------------------------------------------------------------------------


def _faithfulness_structural(
    answer: Answer[Any],
    corpus: dict[str, str] | Callable[[str], str],
    strategies: tuple[MatchStrategy, ...],
    threshold: float,
) -> float:
    """Compute faithfulness via deterministic span verification.

    This is the zero-LLM-cost path.  Each claim is verified by
    checking whether every supporting span matches the corpus under
    the given strategy chain.  A claim with zero span errors is
    considered fully verified.

    Returns the fraction of claims that are fully verified.
    """
    errors: list[ClaimError] = answer.verify(corpus, strategies=strategies, threshold=threshold)
    n_claims = len(answer.claims)
    if n_claims == 0:
        return 0.0
    failed_claim_indices = {e.claim_index for e in errors}
    n_verified = n_claims - len(failed_claim_indices)
    return n_verified / n_claims


def faithfulness(
    corpus: dict[str, str] | Callable[[str], str] | None = None,
    *,
    strategies: Iterable[MatchStrategy] = DEFAULT_MATCH_STRATEGIES,
    threshold: float = 0.9,
    llm_judge_model: str | None = None,
) -> Callable[[Any, dict[str, Any]], float]:
    """Factory: faithfulness metric from structural span verification.

    When the prediction is an :class:`~kaos_llm_core.signatures.grounding.Answer`
    (from a ``GroundedAnswer[T]`` output), verifies each claim's spans
    against the *corpus* deterministically -- **zero LLM cost**.

    When the prediction is a plain string or unstructured type, falls
    back to :class:`~kaos_llm_core.metrics.llm_judge.LLMJudge` with
    the ``"factuality"`` rubric.  This path requires an LLM call and
    is only used when structural verification is unavailable.

    When the prediction is ``InsufficientEvidence``, returns ``1.0``
    (a refusal is faithful by definition -- it does not assert
    unsupported facts).

    Args:
        corpus: Dict mapping ``source_uri`` to source text, or a
            callable performing the same lookup.  Required for the
            structural path; ignored for the LLM fallback.
        strategies: Ordered iterable of :class:`MatchStrategy` values
            to try during span verification.  Default is
            ``(STRICT, SUBSTRING)``.
        threshold: Similarity threshold for ``FUZZY_*`` strategies.
        llm_judge_model: Model identifier for the LLM fallback path
            (e.g. ``"anthropic:claude-haiku-4-5"``).  If ``None``,
            the fallback returns ``0.0`` with a warning instead of
            making an LLM call.

    Returns:
        A metric callable ``(prediction, gold) -> float`` in ``[0, 1]``.
    """
    strategies_tuple = tuple(strategies)

    def _metric(prediction: Any, gold: dict[str, Any]) -> float:
        # Structural path: Answer with claims and spans
        if isinstance(prediction, Answer):
            if corpus is None:
                logger.warning(
                    "faithfulness metric received an Answer but no corpus "
                    "was provided. Cannot verify spans. Returning 0.0."
                )
                return 0.0
            return _faithfulness_structural(prediction, corpus, strategies_tuple, threshold)

        # Refusal path: InsufficientEvidence is faithful by definition
        # (it does not assert unsupported facts). Use refusal_rate()
        # metric separately to track how often the model refuses.
        if isinstance(prediction, InsufficientEvidence):
            logger.debug("faithfulness: InsufficientEvidence → 1.0 (refusal is faithful)")
            return 1.0

        # Fallback: LLM-as-judge factuality
        if llm_judge_model is not None:
            try:
                from kaos_llm_core.metrics.llm_judge import LLMJudge

                judge = LLMJudge(model=llm_judge_model, rubric="factuality")
                return judge(prediction, gold)
            except Exception as exc:
                logger.warning("faithfulness LLM fallback failed: %s. Returning 0.0.", exc)
                return 0.0

        logger.warning(
            "faithfulness metric received a non-Answer prediction and no "
            "llm_judge_model was configured. Returning 0.0. "
            "Either use GroundedAnswer[T] outputs or set llm_judge_model."
        )
        return 0.0

    _metric.__name__ = "faithfulness"
    return _metric


# ---------------------------------------------------------------------------
# 2. Context Precision
# ---------------------------------------------------------------------------


def context_precision(
    *,
    ref_field: str = "relevant_refs",
    retrieved_field: str = "retrieved_refs",
) -> Callable[[Any, dict[str, Any]], float]:
    """Factory: context precision over block_ref lists.

    Measures: of the chunks we retrieved, how many are actually relevant?

    ``precision = |retrieved AND relevant| / |retrieved|``

    The retrieved refs are extracted from the *prediction* (via attribute
    or dict key ``retrieved_field``).  The gold-relevant refs are
    extracted from *gold* (via key ``ref_field``).

    Args:
        ref_field: Key in *gold* dict for the gold-relevant ref set.
        retrieved_field: Key/attribute on *prediction* for retrieved refs.

    Returns:
        A metric callable ``(prediction, gold) -> float`` in ``[0, 1]``.
        Returns ``0.0`` when retrieved set is empty or refs are missing.
    """

    def _metric(prediction: Any, gold: dict[str, Any]) -> float:
        retrieved = _extract_refs(prediction, retrieved_field)
        relevant = _extract_refs(gold, ref_field)

        # Also try gold for retrieved_refs (some eval setups put
        # everything in gold/Example.outputs)
        if retrieved is None:
            retrieved = _extract_refs(gold, retrieved_field)

        if retrieved is None or relevant is None:
            return 0.0
        if not retrieved:
            return 0.0

        tp = len(retrieved & relevant)
        return tp / len(retrieved)

    _metric.__name__ = f"context_precision[{ref_field}]"
    return _metric


# ---------------------------------------------------------------------------
# 3. Context Recall
# ---------------------------------------------------------------------------


def context_recall(
    *,
    ref_field: str = "relevant_refs",
    retrieved_field: str = "retrieved_refs",
) -> Callable[[Any, dict[str, Any]], float]:
    """Factory: context recall over block_ref lists.

    Measures: of all the chunks that should have been retrieved, how
    many actually were?

    ``recall = |retrieved AND relevant| / |relevant|``

    Args:
        ref_field: Key in *gold* dict for the gold-relevant ref set.
        retrieved_field: Key/attribute on *prediction* for retrieved refs.

    Returns:
        A metric callable ``(prediction, gold) -> float`` in ``[0, 1]``.
        Returns ``0.0`` when relevant set is empty or refs are missing.
    """

    def _metric(prediction: Any, gold: dict[str, Any]) -> float:
        retrieved = _extract_refs(prediction, retrieved_field)
        relevant = _extract_refs(gold, ref_field)

        # Also try gold for retrieved_refs
        if retrieved is None:
            retrieved = _extract_refs(gold, retrieved_field)

        if retrieved is None or relevant is None:
            return 0.0
        if not relevant:
            return 0.0

        tp = len(retrieved & relevant)
        return tp / len(relevant)

    _metric.__name__ = f"context_recall[{ref_field}]"
    return _metric


# ---------------------------------------------------------------------------
# 4. Answer Relevancy
# ---------------------------------------------------------------------------


def answer_relevancy(
    *,
    model: str = "openai:text-embedding-3-small",
    question_field: str = "question",
) -> Callable[[Any, dict[str, Any]], float]:
    """Factory: answer relevancy via embedding similarity.

    Measures: does the answer actually address the question?

    Computes cosine similarity between the answer text and the
    original question (pulled from ``gold[question_field]``).  Wraps
    :func:`~kaos_llm_core.metrics.text.semantic_similarity` from the
    existing text metrics.

    Args:
        model: Embedding model identifier for cosine similarity.
        question_field: Key in *gold* dict that holds the question text.

    Returns:
        A metric callable ``(prediction, gold) -> float`` in ``[0, 1]``.
        Returns ``0.0`` when the question is missing or embedding fails.
    """
    from kaos_llm_core.metrics.text import semantic_similarity

    sim_fn = semantic_similarity(model)

    def _metric(prediction: Any, gold: dict[str, Any]) -> float:
        question = gold.get(question_field) if isinstance(gold, dict) else None
        if question is None:
            return 0.0
        answer_str = _answer_text(prediction)
        if answer_str is None:
            return 0.0
        # semantic_similarity expects (prediction, gold) -- both coerced to str
        return sim_fn(answer_str, str(question))

    _metric.__name__ = f"answer_relevancy[{question_field}]"
    return _metric


# ---------------------------------------------------------------------------
# 5. RAGAS Composite
# ---------------------------------------------------------------------------

DEFAULT_RAGAS_WEIGHTS: dict[str, float] = {
    "faithfulness": 0.25,
    "context_precision": 0.25,
    "context_recall": 0.25,
    "answer_relevancy": 0.25,
}


def ragas_composite(
    corpus: dict[str, str] | Callable[[str], str] | None = None,
    *,
    embedding_model: str = "openai:text-embedding-3-small",
    weights: dict[str, float] | None = None,
    strategies: Iterable[MatchStrategy] = DEFAULT_MATCH_STRATEGIES,
    threshold: float = 0.9,
    llm_judge_model: str | None = None,
    ref_field: str = "relevant_refs",
    retrieved_field: str = "retrieved_refs",
    question_field: str = "question",
) -> Callable[[Any, dict[str, Any]], float]:
    """Factory: all four RAGAS metrics as a single composite score.

    Computes a weighted average of faithfulness, context precision,
    context recall, and answer relevancy.  The default weights give
    equal importance to all four (0.25 each).

    Args:
        corpus: Corpus for span verification (faithfulness metric).
        embedding_model: Model for answer relevancy embeddings.
        weights: Dict of metric name to weight.  Must contain exactly
            the keys ``faithfulness``, ``context_precision``,
            ``context_recall``, ``answer_relevancy``.  Weights are
            normalized to sum to 1.0.
        strategies: Match strategies for faithfulness span verification.
        threshold: Fuzzy match threshold for faithfulness.
        llm_judge_model: Fallback model for unstructured faithfulness.
        ref_field: Gold field for relevant refs.
        retrieved_field: Field for retrieved refs.
        question_field: Gold field for question text.

    Returns:
        A metric callable ``(prediction, gold) -> float`` in ``[0, 1]``.
    """
    w = dict(weights) if weights is not None else dict(DEFAULT_RAGAS_WEIGHTS)
    expected_keys = {"faithfulness", "context_precision", "context_recall", "answer_relevancy"}
    if set(w.keys()) != expected_keys:
        msg = (
            f"ragas_composite weights must have keys {sorted(expected_keys)}, "
            f"got {sorted(w.keys())}"
        )
        raise ValueError(msg)

    # Normalize weights to sum to 1.0
    total = sum(w.values())
    if total <= 0:
        msg = "ragas_composite weights must sum to a positive value"
        raise ValueError(msg)
    w = {k: v / total for k, v in w.items()}

    # Build sub-metrics
    faith_fn = faithfulness(
        corpus,
        strategies=strategies,
        threshold=threshold,
        llm_judge_model=llm_judge_model,
    )
    cp_fn = context_precision(ref_field=ref_field, retrieved_field=retrieved_field)
    cr_fn = context_recall(ref_field=ref_field, retrieved_field=retrieved_field)
    ar_fn = answer_relevancy(model=embedding_model, question_field=question_field)

    def _metric(prediction: Any, gold: dict[str, Any]) -> float:
        scores = {
            "faithfulness": faith_fn(prediction, gold),
            "context_precision": cp_fn(prediction, gold),
            "context_recall": cr_fn(prediction, gold),
            "answer_relevancy": ar_fn(prediction, gold),
        }
        return sum(w[k] * scores[k] for k in w)

    _metric.__name__ = "ragas_composite"
    return _metric


# ---------------------------------------------------------------------------
# 6. Expansion lift
# ---------------------------------------------------------------------------


def expansion_lift(
    *,
    ref_field: str = "relevant_refs",
    expanded_field: str = "expanded_refs",
    raw_field: str = "raw_refs",
) -> Callable[[Any, dict[str, Any]], float]:
    """Factory: measures whether query expansion improves retrieval.

    Compares context precision with expanded queries vs the raw (single)
    query.  A positive lift means expansion helped; negative means it hurt.

    Both ``expanded_refs`` and ``raw_refs`` are extracted from the *gold*
    dict.  They should contain the refs retrieved by the expanded and
    raw query sets respectively.  ``relevant_refs`` is the gold set.

    Returns::

        precision(expanded, relevant) - precision(raw, relevant)

    Range: ``[-1.0, 1.0]``.  Zero means no change.

    Example gold dict::

        {
            "relevant_refs": ["#/body/3", "#/body/7"],
            "expanded_refs": ["#/body/3", "#/body/7", "#/body/12"],
            "raw_refs": ["#/body/12", "#/body/15"],
        }
    """

    def _precision(retrieved: set[str], relevant: set[str]) -> float:
        if not retrieved:
            return 0.0
        return len(retrieved & relevant) / len(retrieved)

    def _metric(prediction: Any, gold: dict[str, Any]) -> float:
        relevant_refs = _extract_refs(gold, ref_field)
        expanded_refs = _extract_refs(gold, expanded_field)
        raw_refs = _extract_refs(gold, raw_field)
        relevant = set(relevant_refs) if relevant_refs is not None else set()
        expanded = set(expanded_refs) if expanded_refs is not None else set()
        raw = set(raw_refs) if raw_refs is not None else set()
        if not relevant:
            return 0.0
        return _precision(expanded, relevant) - _precision(raw, relevant)

    _metric.__name__ = "expansion_lift"
    return _metric


# ---------------------------------------------------------------------------
# 7. Refusal rate
# ---------------------------------------------------------------------------


def refusal_rate() -> Callable[[Any, dict[str, Any]], float]:
    """Factory: metric that tracks how often the model refuses to answer.

    Returns 1.0 when the prediction is ``InsufficientEvidence`` (a
    refusal), 0.0 otherwise.  Use alongside ``faithfulness()`` to
    distinguish "verified answer" from "refused to answer" — both
    score 1.0 on faithfulness, but only refusals score 1.0 here.

    Example::

        faith = faithfulness(corpus)
        refusal = refusal_rate()
        # High faithfulness + high refusal = model is conservative
        # High faithfulness + low refusal = model is accurate
    """

    def _metric(prediction: Any, gold: dict[str, Any]) -> float:
        if isinstance(prediction, InsufficientEvidence):
            return 1.0
        return 0.0

    _metric.__name__ = "refusal_rate"
    return _metric


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "DEFAULT_RAGAS_WEIGHTS",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "expansion_lift",
    "faithfulness",
    "ragas_composite",
    "refusal_rate",
]
