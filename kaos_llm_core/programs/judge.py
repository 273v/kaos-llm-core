"""Judge — LLM-as-evaluator for another LLM's output.

Uses a stronger model to evaluate the quality of a weaker model's output.
The producer generates output with the original Signature, then the judge
evaluates it against specified criteria and produces a quality score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kaos_llm_core.programs.base import Program
from kaos_llm_core.programs.call import Call
from kaos_llm_core.signatures.fields import InputField, OutputField
from kaos_llm_core.signatures.introspection import get_instruction
from kaos_llm_core.signatures.signature import Signature
from kaos_llm_core.types import Example


class JudgmentSignature(Signature):
    """Evaluate the quality of a response."""

    original_input: str = InputField(description="The original input text")
    task_description: str = InputField(description="What the task was supposed to accomplish")
    response: str = InputField(description="The response to evaluate")
    criteria: str = InputField(description="Evaluation criteria")
    quality_score: float = OutputField(
        description="Quality score from 0.0 (terrible) to 1.0 (perfect)"
    )
    reasoning: str = OutputField(description="Explanation of the quality assessment")


class Judge(Program):
    """Program that uses a stronger model to evaluate a weaker model's output.

    The workflow:
    1. Producer Call generates output using the original Signature
    2. Judge Call evaluates the output quality against criteria
    3. If quality is below threshold, optionally re-generates with a stronger model

    Example::

        judge = Judge(
            ExtractEntities,
            producer_model="anthropic:claude-haiku-4-5",
            judge_model="anthropic:claude-sonnet-4-6",
            criteria="accuracy and completeness of entity extraction",
        )
        result = await judge(text="The team submitted their report...")
        print(result.output)     # The produced output
        print(result.judgment)   # Quality assessment
    """

    def __init__(
        self,
        signature: type[Signature],
        *,
        producer_model: str | None = None,
        judge_model: str | None = None,
        criteria: str = "accuracy, completeness, and correctness",
        quality_threshold: float = 0.7,
        retry_model: str | None = None,
        producer_examples: list[Example] | None = None,
        core_settings: Any = None,
        **kwargs: Any,
    ) -> None:
        # Phase 9d: thread core_settings through to the inner Calls so
        # KAOS_LLM_CORE_TRACE_ENABLED and per-request _meta.kaos_config
        # overrides reach the producer + judge_call. Without this, MCP
        # tools that constructed a Judge silently dropped per-request
        # configuration.
        self.produce = Call(
            signature,
            model=producer_model,
            examples=producer_examples,
            core_settings=core_settings,
            **kwargs,
        )
        self.judge_call = Call(
            JudgmentSignature,
            model=judge_model,
            core_settings=core_settings,
        )
        # Lazy: only construct the retry producer when one is requested. Stored
        # as a named attribute so :meth:`Program.named_calls` discovers it and
        # collects its trace into ``last_trace.children`` — without this, the
        # retry-model spend was invisible to cost rollups.
        self.retry_produce: Call | None = None
        if retry_model is not None:
            self.retry_produce = Call(signature, model=retry_model, core_settings=core_settings)
        self._signature = signature
        self._criteria = criteria
        self._quality_threshold = quality_threshold
        self._retry_model = retry_model
        self._core_settings = core_settings

    # ------------------------------------------------------------------
    # Public scoring entry point
    # ------------------------------------------------------------------

    async def score_response(self, response: Any, **inputs: Any) -> JudgedResult:
        """Score an externally-produced ``response`` against this judge's criteria.

        This is the entry point used by :class:`BestOfN`, :class:`Refine`, and
        any other caller that already has a candidate output and wants to score
        it without re-running this judge's internal producer. ``inputs`` is the
        original kwargs the producer would have received and is used only to
        synthesize ``original_input`` for the judge prompt — the producer is
        **not** invoked here.

        Returns a :class:`JudgedResult` whose ``output`` is the supplied
        ``response`` (passed through unchanged) and ``judgment`` is the typed
        judgment Pydantic object with ``quality_score`` and ``reasoning``.
        """
        response_text = _render_response_text(response)
        task_desc = get_instruction(self._signature)
        original_input = next((str(v) for v in inputs.values() if isinstance(v, str)), str(inputs))
        judgment = await self.judge_call(
            original_input=original_input,
            task_description=task_desc,
            response=response_text,
            criteria=self._criteria,
        )
        return JudgedResult(output=response, judgment=judgment)

    async def forward(self, **kwargs: Any) -> Any:
        """Produce output, judge it, optionally retry with stronger model."""
        # Step 1: Produce
        result = await self.produce(**kwargs)

        # Step 2: Judge — reuse the public scoring path so the implementation
        # is shared with score_response/Refine/BestOfN.
        judged = await self.score_response(result, **kwargs)
        judgment = judged.judgment

        # Step 3: Optionally retry with stronger model and re-judge
        if judgment.quality_score < self._quality_threshold and self.retry_produce is not None:
            result = await self.retry_produce(**kwargs)
            judged = await self.score_response(result, **kwargs)
            judgment = judged.judgment

        return JudgedResult(output=result, judgment=judgment)


def _render_response_text(response: Any) -> str:
    """Render an arbitrary response object as a JSON-ish string for the judge.

    Tries Pydantic ``model_dump`` first, falls back to ``__dict__`` (filtering
    private attributes), then to ``str()``. Used by :meth:`Judge.score_response`
    and (transitively) by :class:`Refine` and :class:`BestOfN`.
    """
    import json

    try:
        dumped = response.model_dump()  # type: ignore[union-attr]
        return json.dumps(dumped, default=str)
    except (AttributeError, TypeError):
        pass
    try:
        dumped = {k: v for k, v in response.__dict__.items() if not k.startswith("_")}
        return json.dumps(dumped, default=str)
    except (AttributeError, TypeError):
        return str(response)


@dataclass(frozen=True, slots=True)
class JudgedResult:
    """Container for a judged result: the output plus the judgment.

    Returned by :meth:`Judge.score_response` and :meth:`Judge.forward`.
    Phase 9e: this used to be ``_JudgedResult`` (private), but it is the
    de facto public return type that ``Refine`` and ``BestOfN`` consume.
    Promoted to a public name and re-exported from
    ``kaos_llm_core.programs`` and the top-level ``kaos_llm_core``.
    """

    output: Any
    judgment: Any

    def __repr__(self) -> str:
        score = getattr(self.judgment, "quality_score", "?")
        return f"JudgedResult(score={score}, output={self.output!r})"


# Backwards-compat alias — drop in a future major release.
_JudgedResult = JudgedResult
