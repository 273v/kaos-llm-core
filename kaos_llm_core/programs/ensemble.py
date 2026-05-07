"""Ensemble — multi-model voting for robust outputs.

Run the same Signature across multiple models in parallel, then aggregate
results via majority vote or best confidence.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

from kaos_llm_core.programs.base import Program
from kaos_llm_core.programs.call import Call
from kaos_llm_core.signatures.signature import Signature


class Ensemble(Program):
    """Run the same Signature across multiple models and aggregate results.

    Supports two aggregation strategies:
    - ``majority_vote``: Extract a key field and pick the most common value.
    - ``all``: Return all results for manual inspection.

    Example::

        ensemble = Ensemble(
            ClassifyRisk,
            models=[
                "anthropic:claude-sonnet-4-6",
                "openai:gpt-5.4-nano",
                "google:gemini-2.5-flash",
            ],
            vote_field="level",
        )
        result = await ensemble(text="Complex scenario...")
        print(result.selected)   # The majority-voted result
        print(result.all_results) # All individual results
        print(result.votes)      # Vote counts
    """

    def __init__(
        self,
        signature: type[Signature],
        *,
        models: list[str],
        aggregation: Literal["majority_vote", "all"] = "majority_vote",
        vote_field: str | None = None,
        core_settings: Any = None,
        **kwargs: Any,
    ) -> None:
        # Lazy import to avoid pulling errors at module load.
        from kaos_llm_core.errors import CallError

        if not models:
            raise CallError(
                "Ensemble requires at least one model. Pass models=['model-a', 'model-b', ...]."
            )
        # Phase 9d: thread core_settings through to every voter Call so
        # KAOS_LLM_CORE_TRACE_ENABLED and per-request _meta.kaos_config
        # overrides reach the underlying Calls.
        self.voters = [
            Call(signature, model=m, core_settings=core_settings, **kwargs) for m in models
        ]
        self._models = models
        self._aggregation = aggregation
        self._vote_field = vote_field
        self._signature = signature
        self._core_settings = core_settings

    async def forward(self, **kwargs: Any) -> Any:
        """Run all models in parallel and aggregate."""
        results = await asyncio.gather(
            *[voter(**kwargs) for voter in self.voters],
            return_exceptions=True,
        )

        # Filter out exceptions
        valid_results: list[Any] = []
        errors: list[Exception] = []
        for r in results:
            if isinstance(r, Exception):
                errors.append(r)
            else:
                valid_results.append(r)

        if not valid_results:
            from kaos_llm_core.errors import CallError

            raise CallError(
                f"Ensemble: all {len(self._models)} models failed. "
                f"Errors: {[str(e) for e in errors]}."
            )

        if self._aggregation == "majority_vote" and self._vote_field:
            return _majority_vote(valid_results, self._vote_field, self._models)

        return _EnsembleResult(
            selected=valid_results[0],
            all_results=valid_results,
            votes={},
        )

    def named_calls(self) -> dict[str, Any]:
        """Expose voter Calls (held in a list) for optimizer discovery.

        Phase 11: voters live in ``self.voters`` (a list, which the
        ProgramGraph auto-registration cannot expand). This override
        is the documented structural exception — list-held children
        always need an explicit override because the registry only
        sees individual attribute assignments.
        """
        base = super().named_calls()
        for i, voter in enumerate(self.voters):
            base[f"voter_{i}"] = voter
        return base


def _majority_vote(results: list[Any], vote_field: str, models: list[str]) -> _EnsembleResult:
    """Pick the result whose vote_field value appears most often."""
    values = []
    for r in results:
        val = getattr(r, vote_field, None)
        values.append(val)

    counts = Counter(values)
    winner = counts.most_common(1)[0][0]

    # Pick the first result with the winning value
    selected = next(r for r, v in zip(results, values, strict=False) if v == winner)

    return _EnsembleResult(
        selected=selected,
        all_results=results,
        votes=dict(counts),
    )


@dataclass(frozen=True, slots=True)
class _EnsembleResult:
    """Container for ensemble results."""

    selected: Any
    all_results: list[Any]
    votes: dict[Any, int]

    def __repr__(self) -> str:
        return f"EnsembleResult(votes={self.votes}, n_results={len(self.all_results)})"
