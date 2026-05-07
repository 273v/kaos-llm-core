"""Refine — inline iterative refinement program.

Produce → critique → refine → repeat. ``Refine`` runs at *execution* time
and is distinct from ``ReflectiveOptimizer``, which runs offline at
*optimization* time. ``Refine`` consumes a :class:`Judge` instance as its
critic and re-invokes the producer with judge feedback injected into the
prompt until either:

1. The judge's score reaches ``min_score`` (early stop), or
2. ``max_iterations`` is exhausted.

On overrun, ``Refine`` returns the **best** output seen across all
iterations, never the most recent one. Many naive implementations return
the last attempt and end up worse than not refining at all; ``Refine``
treats overrun as "best of N" so adding refinement never hurts.

Concurrency-safety (Phase 9 audit findings #5/#6)
-------------------------------------------------

The pre-Phase-9 implementation mutated ``self.producer.instructions`` in
place between iterations and restored it in a ``finally`` block. That
made every concurrent call against the same Refine instance race on the
shared producer's instruction text — and on a shared
``self._iteration_traces`` list. Two coroutines calling the same Refine
in parallel produced nondeterministic prompt corruption.

The fix is to **clone the producer per iteration** with
:func:`copy.copy`. The clone shares the codec, client, settings, hooks,
examples, kwargs, and output model with the original (read-only sharing
is fine), but has its own ``instructions`` slot we can freely mutate
without touching ``self.producer``. The original producer is never
modified, so concurrent ``Refine.__call__`` invocations are safe even
when they share the same Refine instance and the same producer.

Per-iteration trace assembly uses **nested trace collectors** instead of
instance state. Each iteration opens a :func:`collect_traces` context
which captures the producer + judge sub-traces, builds an iteration
``ExecutionTrace`` with those as children, then pushes the iteration
parent into the surrounding collector (the one opened by
``Program.__call__``). No ``self._iteration_traces`` field, no race.

Score stability caveat (Phase 9 audit finding #14)
--------------------------------------------------

Refine selects the "best" output by ``max(history, key=score)``, where
``score`` is from a stochastic LLM judge. LLM judgments routinely swing
20-40 points across re-runs of the same input — that is too noisy to be
a stable ranking signal. Treat the selection as "approximately the best"
and not "the demonstrably best." If you need a deterministic re-rank
across all candidates, run a final scoring pass with ``temperature=0``
on the full set yourself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from kaos_core.logging import get_logger

from kaos_llm_core.errors import CallError
from kaos_llm_core.observability.collectors import collect_traces, push_trace
from kaos_llm_core.observability.traces import ExecutionTrace
from kaos_llm_core.programs.base import Program
from kaos_llm_core.programs.call import Call, _call_context_var
from kaos_llm_core.programs.cloning import clone_call
from kaos_llm_core.programs.judge import Judge, JudgmentSignature
from kaos_llm_core.programs.loop_runner import LoopConfig, LoopRunner, StepOutcome
from kaos_llm_core.programs.program_hooks import ProgramHooks, fire_program_hook
from kaos_llm_core.programs.result_mixin import OutputForwardingMixin

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RefineHistoryEntry:
    """One iteration of a Refine loop: the produced output, score, feedback."""

    iteration: int
    output: Any
    score: float
    feedback: str


@dataclass(frozen=True, slots=True)
class RefineResult(OutputForwardingMixin):
    """Result of a :class:`Refine` execution.

    Attribute access falls through to the underlying ``outputs`` via
    :class:`OutputForwardingMixin`. ``result.haiku`` reads
    ``result.outputs.haiku`` for object outputs and
    ``result.outputs["haiku"]`` for dict outputs — Refine no longer
    cares which shape the producer returns.
    """

    outputs: Any
    iterations: int
    final_score: float
    stop_reason: str
    history: list[RefineHistoryEntry] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"RefineResult(iterations={self.iterations}, "
            f"final_score={self.final_score:.3f}, "
            f"stop_reason={self.stop_reason!r})"
        )


# ---------------------------------------------------------------------------
# Stop reasons
# ---------------------------------------------------------------------------

STOP_QUALITY_MET = "QUALITY_MET"
STOP_MAX_ITERATIONS = "MAX_ITERATIONS"


# ---------------------------------------------------------------------------
# Refine
# ---------------------------------------------------------------------------


class Refine(Program):
    """Iteratively refine a producer's output using a Judge as critic.

    Algorithm::

        1. Run producer with inputs                            → output_1
        2. Run judge on (inputs, output_1)                     → score_1, feedback_1
        3. If score_1 >= min_score: return output_1            (QUALITY_MET)
        4. Build a refined producer with feedback in its instructions
        5. Repeat steps 1-4 up to max_iterations times.
        6. On overrun: return BEST output across iterations    (MAX_ITERATIONS)

    The producer's :class:`Signature` stays fixed across iterations.
    Feedback is injected by **cloning** the producer per iteration and
    setting ``instructions`` on the clone — the original producer is
    never modified, so concurrent calls against the same Refine are
    safe.

    Example::

        producer = Call(WriteHaiku, model="anthropic:claude-haiku-4-5")
        judge = Judge(
            WriteHaiku,
            producer_model="anthropic:claude-haiku-4-5",
            judge_model="anthropic:claude-sonnet-4-6",
            criteria="poetic quality, imagery, syllable structure",
        )
        refine = Refine(producer, judge, max_iterations=3, min_score=0.85)
        result = await refine(topic="autumn rain")
        print(result.haiku)            # forwarded via __getattr__
        print(result.iterations)
        print(result.final_score)
        print(result.stop_reason)
    """

    # Hard cap on ``max_iterations`` — see ReAct.MAX_ITERATIONS for rationale.
    MAX_ITERATIONS: int = 100

    def __init__(
        self,
        producer: Call,
        judge: Judge,
        *,
        max_iterations: int = 3,
        min_score: float = 0.8,
        feedback_field: str | None = None,
        program_hooks: ProgramHooks | None = None,
        core_settings: Any = None,
    ) -> None:
        if max_iterations < 1:
            raise CallError(
                f"Refine: max_iterations must be at least 1. "
                f"Got max_iterations={max_iterations!r}. Use max_iterations=1 to disable "
                f"refinement and only run the producer once."
            )
        if max_iterations > self.MAX_ITERATIONS:
            raise CallError(
                f"Refine: max_iterations={max_iterations!r} exceeds the hard "
                f"cap of {self.MAX_ITERATIONS}. The cap protects against "
                "runaway API spend; pass a smaller value or override the "
                "class attribute MAX_ITERATIONS in a subclass."
            )
        if not 0.0 <= min_score <= 1.0:
            raise CallError(
                f"Refine: min_score must be in [0.0, 1.0]. "
                f"Got min_score={min_score!r}. Quality scores are normalized to that range."
            )

        self.producer = producer
        self.judge = judge
        self._max_iterations = max_iterations
        self._min_score = min_score
        # Resolve feedback_field. The previous default was the literal string
        # "feedback", which silently failed because the standard JudgmentSignature
        # exposes ``reasoning`` and ``quality_score`` — there is no field named
        # ``feedback``, so getattr returned "" and Refine injected an empty
        # critique on every iteration. Auto-detect the right field from the
        # judge's actual signature instead, falling back to ``reasoning`` for
        # the standard JudgmentSignature.
        self._feedback_field = feedback_field or _resolve_feedback_field(judge)
        self.program_hooks = program_hooks
        # Phase 9d: per-Refine core_settings so the program-level
        # _trace_enabled() helper sees this Refine's settings even when
        # the producer/judge have their own. Optional — when None, the
        # parent trace_enabled resolution falls through to the sub-Calls
        # then env, matching the previous behavior.
        self._core_settings = core_settings

    # ------------------------------------------------------------------
    # Scoring helper
    # ------------------------------------------------------------------

    async def _score(self, producer_inputs: dict[str, Any], producer_output: Any) -> Any:
        """Score a candidate output via the judge.

        Delegates to :meth:`Judge.score_response`, which is the canonical
        external-scoring entry point shared between :class:`Refine` and
        :class:`BestOfN`. Returns the judgment object so callers can read
        ``quality_score`` and the feedback field directly.
        """
        judged = await self.judge.score_response(producer_output, **producer_inputs)
        return judged.judgment

    # ------------------------------------------------------------------
    # forward()
    # ------------------------------------------------------------------

    async def forward(self, **kwargs: Any) -> RefineResult:
        """Run the refine loop via :class:`LoopRunner` and return a :class:`RefineResult`.

        ``kwargs`` are passed straight through to both the producer and
        the judge on every iteration. The producer's :class:`Signature`
        determines which kwargs are required.

        Phase 12C: the loop body lives on :meth:`_refine_step` and the
        result assembly lives on :meth:`_refine_build_result`. The
        runner owns iteration counting and the early-stop ``QUALITY_MET``
        signal; trace plumbing — nested ``collect_traces`` per
        iteration plus a ``push_trace`` of the per-iteration parent —
        stays in the step function so an outer Program collector still
        sees one ``Refine.iter_N`` child per iteration with the
        producer + judge traces nested underneath.
        """
        state = _RefineLoopState(
            kwargs=kwargs,
            original_instructions=self.producer.instructions,
        )
        runner: LoopRunner[_RefineLoopState, RefineHistoryEntry, RefineResult] = LoopRunner(
            config=LoopConfig(
                max_iterations=self._max_iterations,
                propagate_errors=True,
            ),
            make_state=lambda s=state: s,
            step=self._refine_step,
            build_result=self._refine_build_result,
        )
        loop_result = await runner.run()
        return loop_result.result

    async def _refine_step(
        self, iteration: int, state: _RefineLoopState
    ) -> StepOutcome[RefineHistoryEntry]:
        """Run one Refine iteration: produce → judge → record → maybe stop.

        ``iteration`` is 0-indexed for the runner; the per-iteration
        trace name and the history entry use the 1-indexed value to
        match the legacy contract.
        """
        ctx = _call_context_var.get()
        iteration_one_indexed = iteration + 1

        # Per-iteration producer clone. Phase 12A's ``clone_call``
        # extracted the deepcopy/shallow-fallback pattern so this and
        # BestOfN share one canonical isolation primitive. The clone
        # gets its own ``instructions`` slot we can mutate without
        # touching ``self.producer``.
        iteration_producer = clone_call(self.producer, deep=False)
        if iteration_one_indexed > 1 and state.history:
            last_entry = state.history[-1]
            iteration_producer.instructions = _build_refined_instructions(
                state.original_instructions,
                previous_output=last_entry.output,
                critique=last_entry.feedback,
            )
        else:
            iteration_producer.instructions = state.original_instructions

        # Capture producer + judge traces into a nested collector. The
        # outer Program.__call__ collector only receives the
        # ``Refine.iter_N`` parent we push at the end.
        with collect_traces() as iter_children:
            output = await iteration_producer(**state.kwargs)
            judgment = await self._score(state.kwargs, output)

        score = float(getattr(judgment, "quality_score", 0.0))
        feedback = str(getattr(judgment, self._feedback_field, "") or "")

        entry = RefineHistoryEntry(
            iteration=iteration_one_indexed,
            output=output,
            score=score,
            feedback=feedback,
        )
        state.history.append(entry)

        iter_trace = ExecutionTrace(
            call_name=f"Refine.iter_{iteration_one_indexed}",
            signature=type(self).__name__,
            inputs={"iteration": iteration_one_indexed, "score": score},
            model="(refine-iteration)",
            codec="(program)",
            children=list(iter_children),
            cost_usd=sum(c.cost_usd for c in iter_children),
            input_tokens=sum(c.input_tokens for c in iter_children),
            output_tokens=sum(c.output_tokens for c in iter_children),
            total_tokens=sum(c.total_tokens for c in iter_children),
        )
        push_trace(iter_trace)

        fire_program_hook(
            self.program_hooks.on_iteration if self.program_hooks else None,
            self,
            iteration,  # 0-indexed for hook payload (legacy contract)
            {
                "output": output,
                "score": score,
                "feedback": feedback,
                "trace": iter_trace,
            },
            context=ctx,
        )

        logger.debug(
            "Refine iteration %d/%d: score=%.3f stop_threshold=%.3f",
            iteration_one_indexed,
            self._max_iterations,
            score,
            self._min_score,
        )

        if score >= self._min_score:
            state.exit_reason = STOP_QUALITY_MET
            state.exit_output = output
            state.exit_score = score
            return StepOutcome(record=entry, stop_reason=STOP_QUALITY_MET)
        return StepOutcome(record=entry)

    def _refine_build_result(
        self, state: _RefineLoopState, records: list[RefineHistoryEntry]
    ) -> RefineResult:
        """Assemble the final :class:`RefineResult` from the loop state.

        On a clean ``QUALITY_MET`` exit the step function recorded the
        winning output on the state. On overrun (the runner reached
        ``max_iterations`` without an early stop), pick the BEST output
        we saw rather than the most recent one — the same "best of N"
        semantics the pre-Phase-12C loop had.
        """
        del records  # state.history is the canonical record list
        if state.exit_reason == STOP_QUALITY_MET:
            return RefineResult(
                outputs=state.exit_output,
                iterations=len(state.history),
                final_score=state.exit_score,
                stop_reason=STOP_QUALITY_MET,
                history=state.history,
            )
        # Overrun: pick the BEST output we saw, not the last one. Note:
        # judge scores are stochastic so this is "approximately the
        # best", not a deterministic ranking — see module docstring
        # for the caveat.
        best = max(state.history, key=lambda h: h.score)
        return RefineResult(
            outputs=best.output,
            iterations=len(state.history),
            final_score=best.score,
            stop_reason=STOP_MAX_ITERATIONS,
            history=state.history,
        )

    # Phase 11: ``named_calls`` override deleted. ``self.producer`` and
    # ``self.judge`` are public attributes; ProgramGraph auto-registers
    # them through ``Program.__setattr__``.


# ---------------------------------------------------------------------------
# LoopRunner state (Phase 12C)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _RefineLoopState:
    """Mutable per-run state threaded through :meth:`Refine._refine_step`.

    Phase 12C: Refine's loop body migrates onto :class:`LoopRunner`. This
    dataclass holds the kwargs, the snapshot of the producer's original
    instructions, the accumulating history list, and the early-stop
    bookkeeping that the result builder reads to disambiguate
    ``QUALITY_MET`` from ``MAX_ITERATIONS``.
    """

    kwargs: dict[str, Any]
    original_instructions: str
    history: list[RefineHistoryEntry] = field(default_factory=list)
    exit_reason: str | None = None
    exit_output: Any = None
    exit_score: float = 0.0


# ---------------------------------------------------------------------------
# Feedback field resolution
# ---------------------------------------------------------------------------


# Output field name preferences in order of priority. The first one that exists
# on the judge's signature is used. ``reasoning`` is the standard name on
# :class:`JudgmentSignature`; the others are common alternatives users tend to
# pick when defining custom judges.
_FEEDBACK_FIELD_CANDIDATES = ("feedback", "critique", "rationale", "reasoning")


def _resolve_feedback_field(judge: Judge) -> str:
    """Pick a feedback field name from the judge's actual signature.

    Walks ``_FEEDBACK_FIELD_CANDIDATES`` and returns the first one that exists
    as an output field on the judge's :class:`JudgmentSignature` (or any custom
    signature the judge was constructed with). Defaults to ``reasoning`` —
    which matches :class:`JudgmentSignature` — if nothing else is found, since
    that is the field a stock judge actually populates.
    """
    sig = getattr(judge.judge_call, "signature", JudgmentSignature)
    fields = getattr(sig, "model_fields", {}) or {}
    for candidate in _FEEDBACK_FIELD_CANDIDATES:
        if candidate in fields:
            return candidate
    return "reasoning"


# ---------------------------------------------------------------------------
# Feedback prompt construction
# ---------------------------------------------------------------------------


def _build_refined_instructions(
    base_instructions: str,
    *,
    previous_output: Any,
    critique: str,
) -> str:
    """Build a per-iteration instruction suffix that includes prior context.

    The previous output is rendered as JSON (audit finding #16 — the previous
    implementation used ``repr()`` which produced Python single-quoted dicts
    that confuse some models that expect JSON). Falls back gracefully when the
    output cannot be JSON-serialized.

    The critique is appended verbatim and the model is asked to address each
    point and produce an improved response in the same format.
    """
    rendered_text = _render_for_prompt(previous_output)

    suffix = (
        "\n\n"
        "## Refinement context\n"
        "You previously produced the following response:\n"
        f"{rendered_text}\n\n"
        "A critic reviewed it and provided this feedback:\n"
        f"{critique}\n\n"
        "Please produce an improved response that addresses each point in the "
        "feedback. Keep the same output format. Do not apologize or restate the "
        "feedback in your answer."
    )
    return base_instructions + suffix


def _render_for_prompt(value: Any) -> str:
    """Render a producer output as JSON for inclusion in a refinement prompt.

    Tries in order:

    1. ``model_dump()`` → ``json.dumps`` — works for Pydantic models, which
       is the common case for ``Call`` outputs.
    2. ``vars(value)`` → ``json.dumps`` — covers plain Python objects with
       an instance dict.
    3. ``json.dumps(value, default=str)`` — final fallback for primitives,
       lists, and dicts. ``default=str`` rescues anything weird without
       raising.
    """
    try:
        dumped = value.model_dump()  # type: ignore[union-attr]
        return json.dumps(dumped, indent=2, default=str)
    except (AttributeError, TypeError):
        pass
    try:
        instance_dict = {k: v for k, v in vars(value).items() if not k.startswith("_")}
        if instance_dict:
            return json.dumps(instance_dict, indent=2, default=str)
    except TypeError:
        pass
    try:
        return json.dumps(value, indent=2, default=str)
    except (TypeError, ValueError):
        return str(value)
