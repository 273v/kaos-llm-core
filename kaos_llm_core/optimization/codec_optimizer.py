"""CodecOptimizer — try each codec, pick the best by metric score.

Runs the target :class:`Call` with each candidate codec (``JSONCodec``,
``ChatCodec``, ``XMLCodec`` by default), evaluates on the validation set,
and selects the highest-scoring codec. Ties broken by first-tried order.

Example::

    optimizer = CodecOptimizer(metric=exact_match)
    result = await optimizer.optimize(call, val_set)
    print(result.best_codec.__name__, result.scores_by_codec)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kaos_core.logging import get_logger

from kaos_llm_core.codecs.base import Codec
from kaos_llm_core.codecs.chat_codec import ChatCodec
from kaos_llm_core.codecs.json_codec import JSONCodec
from kaos_llm_core.codecs.xml_codec import XMLCodec
from kaos_llm_core.optimization.base import OptimizerBase, resolve_config
from kaos_llm_core.optimization.budget import Budget, StopReason
from kaos_llm_core.optimization.evaluation import MetricFn, summarize_eval_errors
from kaos_llm_core.optimization.mutations import Mutation, MutationLog, RunContext
from kaos_llm_core.programs.call import Call
from kaos_llm_core.types import Example

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CodecOptimizationResult:
    """Result of a codec-selection run."""

    best_codec: type[Codec]
    best_score: float
    scores_by_codec: dict[str, float]
    mutations: list[Mutation] = field(default_factory=list)
    stop_reason: str = StopReason.COMPLETED.value


DEFAULT_CODECS: tuple[type[Codec], ...] = (JSONCodec, ChatCodec, XMLCodec)


@dataclass(slots=True)
class CodecOptimizerConfig:
    """Typed configuration for :class:`CodecOptimizer`. Phase 16.5."""

    codecs: list[type[Codec]] | None = None  # None → use DEFAULT_CODECS

    def __post_init__(self) -> None:
        if self.codecs is not None and not self.codecs:
            msg = (
                "CodecOptimizerConfig.codecs must be None or a non-empty list. "
                "Pass codecs=[JSONCodec, ChatCodec, XMLCodec] or leave None for the default."
            )
            raise ValueError(msg)


def _primary_call(program: Any) -> Call:
    """Resolve the primary :class:`Call` for an optimizer target.

    Phase 11: returns ``program`` directly if it's a :class:`Call`,
    otherwise delegates to ``program.graph.primary()`` for any
    :class:`Program`. Other inputs raise :class:`TypeError`.
    """
    from kaos_llm_core.programs.base import Program

    if isinstance(program, Call):
        return program
    if isinstance(program, Program):
        return program.graph.primary()
    raise TypeError(
        f"CodecOptimizer: could not locate a primary Call on {type(program).__name__}. "
        "Pass a Call instance directly, or a Program subclass that exposes at "
        "least one public Call attribute. Alternative: optimize each sub-Call "
        "individually."
    )


class CodecOptimizer(OptimizerBase):
    """Select the best codec for a Call by exhaustive scoring.

    Phase 13D: migrated onto :class:`OptimizerBase`. Per-codec trials
    run inside named :class:`Trial` scopes; cost / tokens / duration
    come straight off the trial accumulator.

    Args:
        metric: Scoring function (prediction, gold_outputs) -> float in [0, 1].
        codecs: Candidate codec classes. Default: ``[JSONCodec, ChatCodec, XMLCodec]``.
        budget: Optional :class:`Budget` cap. None disables budgeting.
        mutation_log: Optional shared mutation log.
    """

    def __init__(
        self,
        metric: MetricFn,
        *,
        config: CodecOptimizerConfig | None = None,
        budget: Budget | None = None,
        mutation_log: MutationLog | None = None,
        **overrides: Any,
    ) -> None:
        super().__init__(metric=metric, mutation_log=mutation_log, budget=budget)
        cfg = resolve_config(CodecOptimizerConfig, config, overrides)
        self.config = cfg
        if cfg.codecs is None:
            self.codecs: list[type[Codec]] = list(DEFAULT_CODECS)
        else:
            self.codecs = list(cfg.codecs)

    async def optimize(
        self,
        program: Any,
        val_set: list[Example],
        *,
        max_concurrent: int = 5,
        run_context: RunContext | None = None,
    ) -> CodecOptimizationResult:
        call = _primary_call(program)
        tracker, ctx, runner = self._make_run_state(run_context)

        original_codec = call._codec
        scores_by_codec: dict[str, float] = {}
        mutations: list[Mutation] = []
        best_codec: type[Codec] = type(original_codec)
        best_score: float = float("-inf")
        stop_reason: str = StopReason.COMPLETED.value

        logger.info(
            "CodecOptimizer: evaluating %d codec(s) on %d examples",
            len(self.codecs),
            len(val_set),
        )

        try:
            for codec_cls in self.codecs:
                if tracker is not None:
                    exhausted = tracker.exhausted()
                    if exhausted is not None:
                        stop_reason = exhausted.value
                        logger.info("CodecOptimizer: budget exhausted (%s)", stop_reason)
                        break

                before_codec_name = type(call._codec).__name__
                try:
                    candidate = codec_cls()
                except Exception as exc:
                    logger.warning(
                        "CodecOptimizer: failed to instantiate %s: %s", codec_cls.__name__, exc
                    )
                    continue

                call._codec = candidate
                trial_eval, trial = await self._evaluate_in_trial(
                    call,
                    val_set,
                    runner=runner,
                    trial_name=f"codec_{codec_cls.__name__}",
                    max_concurrent=max_concurrent,
                )
                score = trial_eval.score
                scores_by_codec[codec_cls.__name__] = score

                trial_error = summarize_eval_errors(trial_eval)
                if trial_eval.error_rate >= 0.5 and trial_error:
                    logger.warning(
                        "CodecOptimizer: %s has %.0f%% error rate; most common: %s",
                        codec_cls.__name__,
                        trial_eval.error_rate * 100,
                        trial_error,
                    )
                mutation = ctx.make_mutation(
                    strategy="codec_selection",
                    mutation_type="codec_swap",
                    call_name=call.signature.__name__,
                    before={"codec": before_codec_name},
                    after={"codec": codec_cls.__name__},
                    rationale=f"Tried {codec_cls.__name__}; score={score:.3f}",
                    metric_before=best_score if best_score != float("-inf") else 0.0,
                    metric_after=score,
                    tokens_used=trial.total_tokens,
                    cost_usd=trial.cost_usd,
                    accepted=score > best_score,
                    duration_ms=trial.duration_ms,
                    error=trial_error,
                )
                self.mutation_log.record(mutation)
                mutations.append(mutation)

                if score > best_score:
                    best_score = score
                    best_codec = codec_cls
                    logger.info("CodecOptimizer: %s improved to %.3f", codec_cls.__name__, score)

                self._consume_trial(tracker, trial)
        finally:
            # Restore the best codec (or the original if nothing scored).
            if best_score == float("-inf"):
                call._codec = original_codec
            else:
                call._codec = best_codec()

        if best_score == float("-inf"):
            best_score = 0.0

        return CodecOptimizationResult(
            best_codec=best_codec,
            best_score=best_score,
            scores_by_codec=scores_by_codec,
            mutations=mutations,
            stop_reason=stop_reason,
        )
