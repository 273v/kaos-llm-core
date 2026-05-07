"""BestOfN -- sampling-based reliability via parallel best-of-N selection.

Sample the same producer ``Call`` ``n`` times in parallel against one model and
pick the best candidate. Distinct from :class:`~kaos_llm_core.programs.ensemble.Ensemble`
which votes across **different** models -- ``BestOfN`` runs the same producer
``n`` times against **one** model and selects by metric or judge.

Two selection modes:

- **metric**: a callable ``(output, inputs) -> float``; the highest score wins.
- **judge**: a :class:`~kaos_llm_core.programs.judge.Judge` instance scores each
  candidate; the highest ``judgment.quality_score`` wins.

There is no implicit default selector. ``selector`` is required at construction
time -- silently picking by ``str(output)`` would be surprising and almost never
what the caller wants. Forcing the choice keeps the API predictable.

Phase 9 audit hardening
-----------------------

Sampling fans out via :func:`asyncio.gather`. Producer hyperparameters are
**never mutated**: each sample runs against a transient deep clone whose
mutable internals (``_kwargs``, examples list, ``_router`` state, scratch
fields) are isolated from the original. Audit finding #7 noted that the
previous shallow ``copy.copy`` shared the examples list and the router's
mutable state with the original — this version uses ``copy.deepcopy`` to
fix that, with a fallback for non-pickle-safe internals (e.g. transport
clients) by replacing them on the clone.

Trace tree (audit finding B)
----------------------------

The pre-Phase-9 implementation relied on the base ``Program.__call__``
collector, which only sees children listed in ``named_calls()``. The
transient clones were not in ``named_calls()`` so the parent ReAct trace
came back with ``children=[]`` and ``total_tokens=0``. This version uses
the push-based trace collector: each sample's clone goes through
``Call._execute`` which appends its trace to the active collector. The
final parent trace lists every sample as a child plus selector traces if
the selector is a Judge.

Result enrichment (audit findings #25/26/27/28)
-----------------------------------------------

``BestOfNResult`` now exposes:

- ``survivor_indices``: maps each survivor back to its original sample
  index, so a caller who sees ``selected_index=2`` knows which of the
  original ``n`` samples won (rather than guessing).
- ``failed_count`` / ``errors``: programmatic access to per-sample
  failures, instead of forcing the caller to scrape warning logs.
- ``selection_method``: typed as ``Literal["judge", "metric"]`` so a typo
  in user code is caught by ``ty``.
- All-fail error message: groups exception types so the caller sees
  ``"3x RateLimitError, 1x ValidationError, 1x ConnectionError"`` instead
  of just the last failure.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from kaos_core.logging import get_logger

from kaos_llm_core.errors import CallError
from kaos_llm_core.programs.base import Program
from kaos_llm_core.programs.call import Call, _call_context_var
from kaos_llm_core.programs.judge import Judge
from kaos_llm_core.programs.program_hooks import ProgramHooks, fire_program_hook
from kaos_llm_core.programs.result_mixin import OutputForwardingMixin

logger = get_logger(__name__)


def _root_cause_type(exc: BaseException) -> type[BaseException]:
    """Walk ``__cause__`` to find the ORIGINAL exception type.

    Used by :class:`BestOfN` to group all-fail breakdowns by root cause
    rather than by the wrapping ``CallError``. Bounded at 16 hops to
    avoid pathological exception chains.
    """
    seen: set[int] = set()
    current = exc
    for _ in range(16):
        if id(current) in seen:
            break
        seen.add(id(current))
        cause = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
        if cause is None or cause is current:
            break
        current = cause
    return type(current)


type SelectorFn = Callable[[Any, dict[str, Any]], float]
"""User-supplied scoring function: ``(output, inputs) -> score``."""

type Selector = Judge | SelectorFn
"""Either a :class:`Judge` or a metric callable."""


@dataclass(frozen=True, slots=True)
class BestOfNResult(OutputForwardingMixin):
    """Container for a BestOfN execution.

    Attributes:
        outputs: The winning candidate (the producer's raw output object).
        selected_index: Index of the winning candidate within ``candidates``.
            Note: this references the position in the **survivor** list, not
            the original sample order. Use ``survivor_indices[selected_index]``
            to get the original sample position.
        candidates: All successful candidates (failed samples are excluded).
        scores: Scores parallel to ``candidates``.
        survivor_indices: Maps each candidate position back to its original
            sample index. ``survivor_indices[i]`` is the position of
            ``candidates[i]`` in the original ``range(n)``. If sample 1 of
            5 raised, this list is ``[0, 2, 3, 4]``.
        failed_count: Number of samples that raised before scoring.
        errors: List of exceptions raised by failing samples (parallel to
            the original sample order, with ``None`` placeholders for
            successful samples).
        selection_method: Either ``"judge"`` or ``"metric"``. Typed as a
            Literal so static type checking catches typos in caller code.

    Attribute access on the result transparently forwards to fields on
    ``outputs`` via :class:`OutputForwardingMixin` so you can write
    ``result.field_name`` instead of ``result.outputs.field_name``.
    """

    outputs: Any
    selected_index: int
    candidates: list[Any] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    survivor_indices: list[int] = field(default_factory=list)
    failed_count: int = 0
    errors: list[BaseException | None] = field(default_factory=list)
    selection_method: Literal["judge", "metric"] = "metric"

    @property
    def selected_original_index(self) -> int:
        """Position of the winning candidate in the original ``range(n)``.

        Useful when callers want to correlate the winner with logs or
        metrics keyed by the original sample index.
        """
        if not self.survivor_indices:
            return self.selected_index
        return self.survivor_indices[self.selected_index]

    def __repr__(self) -> str:
        return (
            f"BestOfNResult(selected_index={self.selected_index}, "
            f"n_candidates={len(self.candidates)}, "
            f"failed_count={self.failed_count}, scores={self.scores}, "
            f"method={self.selection_method!r})"
        )


class BestOfN(Program):
    """Sample one model ``n`` times in parallel and pick the best candidate.

    Args:
        producer: The :class:`Call` (or subclass) to sample from. Its
            hyperparameters are never mutated.
        n: Number of samples. Must be ``>= 2``.
        selector: Required. Either a :class:`Judge` or a callable
            ``(output, inputs) -> float``. The candidate with the **highest**
            score wins.
        seed_strategy: Diversity strategy.

            - ``"temperature"`` (default): bump ``temperature`` to
              ``diversity_temperature`` on the transient producer clone if
              the producer's current value is lower (or absent).
            - ``"explicit_seed"``: pass ``seed=i+1`` for ``i in range(n)``.
              Some providers (OpenAI) treat ``seed=0`` as "unset", so this
              starts at 1 to ensure each sample gets a real seed.
            - ``"none"``: do nothing; rely entirely on provider non-determinism.
        diversity_temperature: Temperature value used by the
            ``"temperature"`` strategy. Defaults to ``1.0`` which is the
            Anthropic maximum and noticeably high for some providers; lower
            this if you find samples too divergent.
        program_hooks: Optional :class:`ProgramHooks` for observing the
            sample fan-out.

    Raises:
        CallError: If ``n < 2``, or ``n > MAX_N``, or if ``selector`` is
            ``None``, or if **all** samples fail (the per-type breakdown
            is included in the error message).
    """

    # Hard cap on ``n`` — see ReAct.MAX_ITERATIONS for rationale. Each
    # additional sample is a full LLM call so a low cap is appropriate.
    MAX_N: int = 100

    def __init__(
        self,
        producer: Call,
        n: int,
        *,
        selector: Selector | None = None,
        seed_strategy: Literal["temperature", "explicit_seed", "none"] = "temperature",
        diversity_temperature: float = 1.0,
        program_hooks: ProgramHooks | None = None,
        core_settings: Any = None,
    ) -> None:
        if n < 2:
            raise CallError(
                f"BestOfN requires n >= 2, got n={n}. "
                f"For a single sample just call the producer directly. "
                f"For 2+ samples pass e.g. n=3 or n=5.",
            )
        if n > self.MAX_N:
            raise CallError(
                f"BestOfN: n={n} exceeds the hard cap of {self.MAX_N}. "
                f"Each additional sample is a full LLM call; the cap protects "
                "against runaway API spend. Pass a smaller n, or override the "
                "class attribute MAX_N in a subclass with full awareness of "
                "the cost implications."
            )
        if selector is None:
            raise CallError(
                "BestOfN requires an explicit selector. "
                "Pass either selector=Judge(...) for LLM-as-judge scoring, "
                "or selector=lambda output, inputs: float(...) for a custom metric. "
                "There is no default selector because lexicographic comparison of "
                "raw outputs is almost never what the caller wants.",
            )
        if seed_strategy not in ("temperature", "explicit_seed", "none"):
            raise CallError(
                f"BestOfN: unknown seed_strategy={seed_strategy!r}. "
                f"Use one of: 'temperature', 'explicit_seed', 'none'.",
            )
        if diversity_temperature < 0.0:
            raise CallError(
                f"BestOfN: diversity_temperature must be >= 0.0, "
                f"got {diversity_temperature!r}. "
                "Typical values: 0.7-1.0 for varied samples, 0.0-0.3 for low diversity.",
            )

        self.producer = producer
        self.n = n
        self.selector = selector
        self.seed_strategy = seed_strategy
        self.diversity_temperature = diversity_temperature
        self.program_hooks = program_hooks
        # Phase 9d: per-BestOfN core_settings (see Refine.__init__ for the
        # rationale).
        self._core_settings = core_settings
        self._selection_method: Literal["judge", "metric"] = (
            "judge" if isinstance(selector, Judge) else "metric"
        )

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _kwargs_for_sample(self, i: int) -> dict[str, Any]:
        """Build the per-sample ``_kwargs`` override for the transient clone.

        Never mutates ``self.producer._kwargs``.
        """
        base = dict(self.producer._kwargs)  # snapshot, not a reference

        if self.seed_strategy == "temperature":
            current = base.get("temperature")
            if current is None or (
                isinstance(current, (int, float)) and current < self.diversity_temperature
            ):
                base["temperature"] = self.diversity_temperature
        elif self.seed_strategy == "explicit_seed":
            # Start at 1, not 0, because OpenAI treats seed=0 as "unset" so
            # sample 0 with the legacy ``seed=i`` strategy got no seed at all.
            # Audit finding #23.
            base["seed"] = i + 1
        # "none" -> leave base untouched

        return base

    def _clone_producer(self, sample_kwargs: dict[str, Any]) -> Call:
        """Build a per-sample clone of the producer.

        Uses ``copy.deepcopy`` so mutable internals (``_kwargs``, the
        ``examples`` list, the codec, the router state) are isolated from
        the original. The previous shallow ``copy.copy`` shared the
        examples list and any router/scratch state, which races on
        concurrent samples (audit finding #7).

        Falls back to a shallow ``copy.copy`` if deepcopy fails — some
        transport clients hold non-pickleable resources (sockets, locks)
        and cannot survive a deepcopy. In that case we still replace the
        per-execution scratch slots so concurrent samples don't race on
        them, but we accept that the codec/examples may be shared. Users
        who hit this can pass a deepcopy-safe client.
        """
        try:
            clone = copy.deepcopy(self.producer)
        except Exception as exc:
            logger.debug(
                "BestOfN: deepcopy of producer failed (%s); falling back to "
                "shallow copy. Per-execution scratch slots are still isolated, "
                "but mutable shared state (examples list, router) may race.",
                exc,
            )
            clone = copy.copy(self.producer)
            # Shallow path: still need fresh per-execution scratch and a
            # fresh examples list so the clone does not race on those.
            clone.examples = list(self.producer.examples)

        clone._kwargs = sample_kwargs
        # Phase 10: per-execution scratch lives in the contextvar
        # Invocation, not on the Call instance — clones have nothing
        # to reset.
        return clone

    async def _run_sample(self, i: int, inputs: dict[str, Any]) -> Any:
        """Run sample ``i`` against a fresh producer clone."""
        sample_kwargs = self._kwargs_for_sample(i)
        clone = self._clone_producer(sample_kwargs)
        return await clone(**inputs)

    async def _score_candidate(
        self,
        candidate: Any,
        inputs: dict[str, Any],
    ) -> float:
        """Score a single candidate via the configured selector.

        For a :class:`Judge` selector, this uses :meth:`Judge.score_response`
        which scores the **supplied candidate** against the judge's criteria
        without re-running the judge's internal producer.

        For a metric callable, supports both sync and async callables (and
        any awaitable — Tasks, Futures — via :func:`inspect.isawaitable`,
        which is broader than ``asyncio.iscoroutine``).
        """
        if isinstance(self.selector, Judge):
            judged = await self.selector.score_response(candidate, **inputs)
            score = getattr(judged.judgment, "quality_score", None)
            if score is None:
                raise CallError(
                    "BestOfN judge selector did not return a quality_score. "
                    "Ensure the Judge is configured with the standard JudgmentSignature "
                    "or supply a custom metric callable instead.",
                )
            return float(score)

        # Metric callable -- may be sync or async (or return any awaitable).
        raw: Any = self.selector(candidate, inputs)
        if inspect.isawaitable(raw):
            raw = await raw
        return float(cast("float", raw))

    # ------------------------------------------------------------------
    # forward()
    # ------------------------------------------------------------------

    async def forward(self, **inputs: Any) -> BestOfNResult:
        """Run ``n`` samples in parallel, score, and pick the best."""
        ctx = _call_context_var.get()

        # Parallel fan-out -- collect exceptions instead of failing the whole batch.
        sample_results = await asyncio.gather(
            *[self._run_sample(i, inputs) for i in range(self.n)],
            return_exceptions=True,
        )

        candidates: list[Any] = []
        survivor_indices: list[int] = []
        per_sample_errors: list[BaseException | None] = []
        failures: list[BaseException] = []
        for i, r in enumerate(sample_results):
            if isinstance(r, BaseException):
                logger.warning("BestOfN sample %d/%d failed: %s", i + 1, self.n, r)
                per_sample_errors.append(r)
                failures.append(r)
            else:
                candidates.append(r)
                survivor_indices.append(i)
                per_sample_errors.append(None)

        if not candidates:
            # All samples failed. Raise CallError with a per-type breakdown
            # so the caller sees the failure pattern, not just the last
            # exception. Audit finding #27.
            #
            # We walk the __cause__ chain to find the ORIGINAL exception
            # type. Call._execute wraps provider failures in CallError, so
            # naive ``type(e).__name__`` would just say "3x CallError" which
            # hides the real story. Walking the chain reveals the underlying
            # RateLimitError / ValidationError / ConnectionError that the
            # user is actually trying to debug.
            type_counts = Counter(_root_cause_type(e).__name__ for e in failures)
            breakdown = ", ".join(f"{count}x {tname}" for tname, count in type_counts.most_common())
            last_error = failures[-1]
            raise CallError(
                f"BestOfN: all {self.n} samples failed ({breakdown}). "
                f"Last error: {last_error}. "
                f"Check the producer's model and API key, then retry; "
                f"if the producer raises validation errors, lower n or "
                f"switch to a stronger model.",
            ) from last_error

        # Score the survivors. Both Judge and metric paths now run through
        # asyncio.gather (audit finding #13 — async metric callables used to
        # be serialized in a sequential loop, killing parallelism for users
        # who passed an async scoring function).
        score_tasks = [self._score_candidate(c, inputs) for c in candidates]
        score_results = await asyncio.gather(*score_tasks, return_exceptions=True)
        scores: list[float] = []
        for i, s in enumerate(score_results):
            if isinstance(s, BaseException):
                logger.warning(
                    "BestOfN scoring failed for survivor %d: %s; using -inf",
                    i,
                    s,
                )
                scores.append(float("-inf"))
            else:
                scores.append(float(s))

        # Argmax over scores. ``max`` with ``key`` returns the first match
        # on ties, which is fine and deterministic.
        selected_index = max(range(len(scores)), key=lambda i: scores[i])

        result = BestOfNResult(
            outputs=candidates[selected_index],
            selected_index=selected_index,
            candidates=candidates,
            scores=scores,
            survivor_indices=survivor_indices,
            failed_count=len(failures),
            errors=per_sample_errors,
            selection_method=self._selection_method,
        )

        fire_program_hook(
            self.program_hooks.on_iteration if self.program_hooks else None,
            self,
            0,
            {
                "result": result,
                "scores": list(scores),
                "selected_index": selected_index,
                "failed_count": len(failures),
            },
            context=ctx,
        )

        return result

    # ------------------------------------------------------------------
    # Optimizer / trace integration
    # ------------------------------------------------------------------

    # Phase 11: ``named_calls`` override deleted. ``self.producer`` and
    # ``self.selector`` (when it's a Judge) are public attributes that
    # ProgramGraph auto-registers via ``Program.__setattr__``. Function
    # selectors are not Calls/Programs, so they remain ignored.
