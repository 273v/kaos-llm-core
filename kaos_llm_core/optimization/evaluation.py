"""Evaluation framework — run a Program against labeled data and compute metrics.

The foundation for all optimizers. An evaluation runs a Call or Program on
every Example in a dataset, applies a metric function to each (prediction, gold)
pair, and aggregates the results.

Example::

    def exact_match(prediction, gold) -> float:
        return 1.0 if prediction.level == gold["level"] else 0.0

    result = await evaluate(
        call=my_call,
        dataset=labeled_examples,
        metric=exact_match,
    )
    print(f"Accuracy: {result.score:.1%} ({result.n_correct}/{result.n_total})")
"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from kaos_core.logging import get_logger

from kaos_llm_core.types import Example

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ExampleResult:
    """Result of evaluating a single example.

    ``trace`` is the canonical per-example execution trace, captured by the
    eval harness directly from ``call.last_trace`` after the invocation
    completes (success or failure). Phase 9 audit follow-up: previously
    budget tools relied on a ``_last_trace`` attribute being mutated onto
    the prediction object, which silently broke for predictions that
    cannot carry an attribute (plain dicts, primitives, frozen models).
    Storing the trace on the eval result is the right ownership boundary
    — predictions are user-facing data, traces are observability metadata.
    """

    example: Example
    prediction: Any
    score: float
    error: str | None = None
    # Exception class name (e.g. ``CallError``, ``ValidationRetryExhaustedError``)
    # for the error path. Captured separately from ``error`` so the optimizer
    # error-rate aggregator can group by class without parsing the message.
    error_class: str | None = None
    # Per-example execution trace. ``None`` if the call/program has no
    # ``last_trace`` (extremely rare — only when execution failed before
    # any trace was built).
    trace: Any = None


@dataclass(frozen=True, slots=True)
class EvalResult:
    """Aggregated evaluation result across a dataset."""

    score: float
    n_total: int
    n_correct: int
    n_errors: int
    per_example: list[ExampleResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        """Fraction of examples with score >= 1.0 (exact match)."""
        if self.n_total == 0:
            return 0.0
        return self.n_correct / self.n_total

    @property
    def error_rate(self) -> float:
        """Fraction of examples that raised exceptions."""
        if self.n_total == 0:
            return 0.0
        return self.n_errors / self.n_total

    def failures(self) -> list[ExampleResult]:
        """Return examples that scored below 1.0 (for error analysis)."""
        return [r for r in self.per_example if r.score < 1.0 and r.error is None]

    def errors(self) -> list[ExampleResult]:
        """Return examples that raised exceptions."""
        return [r for r in self.per_example if r.error is not None]


MetricFn = Callable[[Any, dict[str, Any]], float]
"""A metric function: (prediction, gold_outputs) -> score in [0, 1]."""


def summarize_eval_errors(eval_result: EvalResult) -> str | None:
    """Return a one-line summary of the most common error class in this eval.

    Returns ``None`` when no example errored. Returns a string of the form
    ``"<ClassName>: <example message> (n=<count>)"`` describing the most
    frequent exception class plus a representative message from one such
    failure. Used by every optimizer's per-trial mutation record (GAP-5)
    and by F1 — surfacing the dominant exception when ``error_rate`` is
    high enough that the user would otherwise see a silent "no
    improvement" run.

    The previous code had no such summary; an optimizer running with
    100% errored trials would silently report ``score=0`` and the user
    had no diagnostic.
    """
    errored = [er for er in eval_result.per_example if er.error is not None]
    if not errored:
        return None
    classes = Counter(er.error_class or "Exception" for er in errored)
    top_class, count = classes.most_common(1)[0]
    sample = next(
        (er for er in errored if (er.error_class or "Exception") == top_class),
        errored[0],
    )
    msg = (sample.error or "")[:120]
    return f"{top_class}: {msg} (n={count})"


async def evaluate(
    call: Any,
    dataset: list[Example],
    metric: MetricFn,
    *,
    max_concurrent: int = 5,
    return_on_error: float = 0.0,
) -> EvalResult:
    """Evaluate a Call or Program against a labeled dataset.

    Args:
        call: A Call or Program instance (anything with async __call__).
        dataset: List of Examples with inputs and expected outputs.
        metric: Function (prediction, gold_outputs) -> score in [0, 1].
        max_concurrent: Max concurrent evaluations.
        return_on_error: Score to assign when a call raises an exception.

    Returns:
        EvalResult with aggregate score and per-example details.
    """
    if not dataset:
        return EvalResult(score=0.0, n_total=0, n_correct=0, n_errors=0)

    semaphore = asyncio.Semaphore(max_concurrent)

    async def eval_one(example: Example) -> ExampleResult:
        async with semaphore:
            # Phase 10: ``call.invoke()`` returns the canonical Invocation
            # bundling output, trace, usage, and error. The success path
            # reads ``invocation.trace`` directly. On exception, the
            # partial Invocation is tagged on the exception as
            # ``exc.invocation`` so we can capture failure traces from
            # the except block without a separate ContextVar — and
            # without a race because each ``invoke()`` returns its own
            # task-local Invocation.
            try:
                invocation = await call.invoke(**example.inputs)
            except Exception as e:
                logger.warning(
                    "Evaluation error on example %s: %s",
                    str(example.inputs)[:80],
                    str(e),
                )
                attached = getattr(e, "invocation", None)
                trace = getattr(attached, "trace", None)
                return ExampleResult(
                    example=example,
                    prediction=None,
                    score=return_on_error,
                    error=str(e),
                    error_class=type(e).__name__,
                    trace=trace,
                )
            score = metric(invocation.output, example.outputs)
            return ExampleResult(
                example=example,
                prediction=invocation.output,
                score=score,
                trace=invocation.trace,
            )

    results = await asyncio.gather(*[eval_one(ex) for ex in dataset])

    scores = [r.score for r in results]
    n_correct = sum(1 for s in scores if s >= 1.0)
    n_errors = sum(1 for r in results if r.error is not None)
    avg_score = sum(scores) / len(scores) if scores else 0.0

    return EvalResult(
        score=avg_score,
        n_total=len(dataset),
        n_correct=n_correct,
        n_errors=n_errors,
        per_example=list(results),
    )
