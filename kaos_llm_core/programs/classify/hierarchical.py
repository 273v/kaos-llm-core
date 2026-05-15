"""Hierarchical (coarse-to-fine) taxonomy classification.

Walks a hierarchical :class:`~kaos_llm_core.labels.LabelSet` top-down:
at each level, restrict the classifier to the current set of
candidate children and pick the best. Recurse into the picked label's
children until a leaf is reached (or the classifier abstains).
"""

from __future__ import annotations

from typing import Any

from kaos_llm_client import BaseProviderClient
from kaos_llm_client.settings import KaosLLMSettings

from kaos_llm_core.codecs.base import Codec
from kaos_llm_core.labels import Label, LabelSet
from kaos_llm_core.programs.base import Program
from kaos_llm_core.programs.classify.zero_shot import ZeroShotClassify
from kaos_llm_core.results import Classification
from kaos_llm_core.settings import KaosLLMCoreSettings
from kaos_llm_core.types import Example


class HierarchicalClassify(Program):
    """Coarse-to-fine classification over a hierarchical LabelSet.

    Requires ``label_set.hierarchical=True``. At each level the
    Program constructs a flat :class:`LabelSet` over the candidate
    children, runs :class:`ZeroShotClassify`, and uses the pick to
    decide where to recurse.

    Output:
        The returned :class:`Classification` carries the **leaf-most**
        label the walk reached as ``labels[0]``. ``metadata`` records
        the full path (``hierarchical.path``).
    """

    def __init__(
        self,
        *,
        labels: LabelSet,
        model: str | None = None,
        codec: Codec | None = None,
        client: BaseProviderClient | None = None,
        settings: KaosLLMSettings | None = None,
        core_settings: KaosLLMCoreSettings | None = None,
        examples: list[Example] | None = None,
        instructions: str | None = None,
        max_retries: int | None = None,
        max_depth: int = 5,
        **kwargs: Any,
    ) -> None:
        if not labels.hierarchical:
            raise ValueError(
                "HierarchicalClassify requires a hierarchical LabelSet "
                "(hierarchical=True with at least one parent reference)."
            )
        self._label_set = labels
        self._max_depth = max_depth
        # Shared knobs threaded through every level's classifier.
        self._classifier_kwargs: dict[str, Any] = dict(
            model=model,
            codec=codec,
            client=client,
            settings=settings,
            core_settings=core_settings,
            examples=examples,
            instructions=instructions,
            max_retries=max_retries,
            **kwargs,
        )

    def _children_label_set(self, parent_name: str | None) -> LabelSet | None:
        """Build a flat :class:`LabelSet` over the children of ``parent_name``."""
        children = self._label_set.children(parent_name)
        if not children:
            return None
        return LabelSet(
            labels=list(children),
            exclusive=True,
            allow_abstain=self._label_set.allow_abstain,
            hierarchical=False,
        )

    async def _walk(
        self,
        text: str,
        parent_id: str | None,
    ) -> tuple[list[Label], list[Classification]]:
        """Walk the taxonomy. Return (path_labels, level_results)."""
        path: list[Label] = []
        level_results: list[Classification] = []
        current_parent: str | None = None
        for _level in range(self._max_depth):
            children_set = self._children_label_set(current_parent)
            if children_set is None:
                break
            level_classifier = ZeroShotClassify(
                labels=children_set,
                **self._classifier_kwargs,
            )
            level_result: Classification = await level_classifier(
                text=text,
                parent_id=parent_id,
            )
            level_results.append(level_result)
            if level_result.abstained or not level_result.labels:
                break
            picked_name = level_result.names[0]
            path.append(self._label_set.by_name(picked_name))
            current_parent = picked_name
        return path, level_results

    async def forward(  # ty: ignore[invalid-method-override]
        self,
        *,
        text: str,
        parent_id: str | None = None,
    ) -> Classification:
        path, level_results = await self._walk(text, parent_id)
        if not path:
            return Classification(
                labels=[],
                abstained=True,
                chunks_used=[parent_id] if parent_id else [],
                metadata={
                    "program": "HierarchicalClassify",
                    "hierarchical.depth": 0,
                    "hierarchical.path": [],
                },
            )
        leaf = path[-1]
        leaf_result = level_results[-1]
        return Classification(
            labels=[leaf],
            scores=dict(leaf_result.scores),
            abstained=False,
            rationale=leaf_result.rationale,
            chunks_used=[parent_id] if parent_id else [],
            metadata={
                "program": "HierarchicalClassify",
                "hierarchical.depth": len(path),
                "hierarchical.path": [label.name for label in path],
            },
        )


__all__ = ["HierarchicalClassify"]
