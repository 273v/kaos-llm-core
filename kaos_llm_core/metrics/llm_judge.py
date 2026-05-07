"""LLMJudge — LLM-as-a-metric.

A callable that scores ``(prediction, gold)`` pairs using an LLM. Distinct
from :class:`kaos_llm_core.programs.judge.Judge`, which is a *program*
producing a (output, judgment) pair. ``LLMJudge`` returns a single float and
plugs into ``optimizer.metric=...`` exactly like the deterministic metrics in
:mod:`kaos_llm_core.metrics.text`.

Built-in rubrics: ``"helpfulness"``, ``"factuality"``, ``"conciseness"``.
Custom rubrics may be passed as free-form strings.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

from kaos_core.logging import get_logger
from kaos_llm_client import BaseProviderClient

from kaos_llm_core.programs.call import Call
from kaos_llm_core.signatures.fields import InputField, OutputField
from kaos_llm_core.signatures.signature import Signature

logger = get_logger(__name__)


_BUILTIN_RUBRICS: dict[str, str] = {
    "helpfulness": (
        "Does the prediction directly answer what the gold is asking for? "
        "Score 0.0 (not helpful) to 1.0 (perfectly helpful)."
    ),
    "factuality": (
        "Is the prediction factually consistent with the gold? "
        "Score 0.0 (contradicts the gold) to 1.0 (fully consistent)."
    ),
    "conciseness": (
        "Is the prediction appropriately concise without losing essential "
        "information from the gold? "
        "Score 0.0 (verbose or missing essential info) to 1.0 (concise and complete)."
    ),
}


class LLMJudgeSignature(Signature):
    """Score a prediction against a gold reference using a rubric."""

    rubric: str = InputField(description="The evaluation rubric describing what to score.")
    prediction: str = InputField(description="The model output to evaluate.")
    gold: str = InputField(description="The reference / expected output.")
    score: float = OutputField(
        description="Quality score from 0.0 (worst) to 1.0 (best). Must be a float."
    )


class LLMJudge:
    """LLM-as-a-metric. A callable that scores ``(prediction, gold)`` pairs.

    This is *not* the :class:`Judge` program. ``LLMJudge`` is a thin metric
    suitable for passing to optimizer ``metric=`` arguments.

    Example::

        from kaos_llm_core.metrics import LLMJudge
        from kaos_llm_core.optimization import BootstrapOptimizer

        metric = LLMJudge(
            model="anthropic:claude-haiku-4-5",
            rubric="helpfulness",
        )
        optimizer = BootstrapOptimizer(metric=metric, ...)
    """

    def __init__(
        self,
        model: str,
        *,
        rubric: str = "helpfulness",
        system: str | None = None,
        client: BaseProviderClient | None = None,
    ) -> None:
        if not model:
            raise ValueError(
                "LLMJudge requires a model. Pass model='provider:name', e.g., "
                "'anthropic:claude-haiku-4-5'. Alternative: use a deterministic "
                "metric from kaos_llm_core.metrics.text."
            )
        self.model = model
        self.rubric_name = rubric
        self.rubric_text = _BUILTIN_RUBRICS.get(rubric, rubric)
        instructions = system or (
            "You are an impartial evaluator. Read the rubric, the prediction, "
            "and the gold reference, then output a single floating-point score "
            "in [0.0, 1.0] following the rubric."
        )
        self._call = Call(
            LLMJudgeSignature,
            model=model,
            instructions=instructions,
            client=client,
        )

    @property
    def __name__(self) -> str:
        return f"LLMJudge[{self.rubric_name}@{self.model}]"

    async def acall(self, prediction: Any, gold: Any) -> float:
        """Async scoring entry point."""
        try:
            result = await self._call(
                rubric=self.rubric_text,
                prediction=_stringify(prediction),
                gold=_stringify(gold),
            )
            score = float(getattr(result, "score", 0.0))
        except Exception as exc:
            logger.warning(
                "LLMJudge scoring failed: %s. Returning 0.0. "
                "Check API key and model id, or fall back to a deterministic metric.",
                exc,
            )
            return 0.0
        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))

    def __call__(self, prediction: Any, gold: Any) -> float:
        """Sync entry point. Wraps :meth:`acall` via ``asyncio.run`` or thread fallback."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                result = pool.submit(asyncio.run, self.acall(prediction, gold)).result()
                return cast(float, result)
        return asyncio.run(self.acall(prediction, gold))


def _stringify(value: Any) -> str:
    """Coerce metric input to a string for the judge prompt."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("answer", "output", "value", "text"):
            if key in value:
                return _stringify(value[key])
        try:
            import json

            return json.dumps(value, default=str)
        except Exception as exc:
            logger.debug("_stringify: json.dumps failed, falling back to str(): %s", exc)
            return str(value)
    if hasattr(value, "answer"):
        try:
            return _stringify(value.answer)  # type: ignore[attr-defined]
        except Exception as exc:
            logger.debug("_stringify: access to .answer failed: %s", exc)
    return str(value)
