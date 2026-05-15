"""Ensemble classifier — combine multiple member classifiers via an Aggregator."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from kaos_llm_core.composition import Aggregator, MajorityAggregator
from kaos_llm_core.labels import LabelSet
from kaos_llm_core.programs.base import Program
from kaos_llm_core.programs.call import Call
from kaos_llm_core.results import Classification


class EnsembleClassify(Program):
    """Run multiple classifiers over the same input and combine.

    Members are Program instances whose ``forward(text=…)`` returns a
    :class:`Classification`. Common compositions:

    - One :class:`ZeroShotClassify` per model in a cascade router.
    - A :class:`PrototypeClassify` plus an LLM classifier for
      consensus voting.
    - The same :class:`ZeroShotClassify` repeated with different
      ``temperature`` settings.

    Args:
        labels: Shared label space.
        members: Sequence of member classifiers.
        aggregator: Strategy. Defaults to :class:`MajorityAggregator`
            for exclusive label sets.
        max_concurrency: Cap on concurrent member invocations.
            Default ``len(members)``.
    """

    # Children stored in a private list and exposed via named_calls
    # override so the trace tree still sees every member.
    def __init__(
        self,
        *,
        labels: LabelSet,
        members: Sequence[Program],
        aggregator: Aggregator | None = None,
        max_concurrency: int | None = None,
    ) -> None:
        if not members:
            raise ValueError("EnsembleClassify requires at least one member")
        self._label_set = labels
        self._members: list[Program] = list(members)
        self._aggregator = aggregator or MajorityAggregator()
        self._max_concurrency = max_concurrency or len(members)

    # Override named_calls so the Program graph + trace tree see each
    # member even though they are stored in a private list.
    def named_calls(self) -> dict[str, Call | Program]:
        return {f"member_{i}": member for i, member in enumerate(self._members)}

    async def forward(  # ty: ignore[invalid-method-override]
        self,
        *,
        text: str,
        parent_id: str | None = None,
    ) -> Classification:
        sem = asyncio.Semaphore(self._max_concurrency)

        async def _one(member: Program) -> Classification:
            async with sem:
                return await member(text=text, parent_id=parent_id)

        results: list[Classification] = await asyncio.gather(
            *(_one(member) for member in self._members)
        )
        combined = self._aggregator.combine(results, self._label_set)
        return combined.model_copy(
            update={
                "metadata": {
                    **dict(combined.metadata),
                    "program": "EnsembleClassify",
                    "ensemble.members": len(results),
                    "aggregator": type(self._aggregator).__name__,
                },
            }
        )


__all__ = ["EnsembleClassify"]
