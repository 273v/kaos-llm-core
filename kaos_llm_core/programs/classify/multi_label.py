"""Multi-label LLM classification."""

from __future__ import annotations

from typing import Any

from kaos_llm_client import BaseProviderClient
from kaos_llm_client.settings import KaosLLMSettings
from pydantic import create_model

from kaos_llm_core.codecs.base import Codec
from kaos_llm_core.labels import LabelSet
from kaos_llm_core.programs.base import Program
from kaos_llm_core.programs.call import Call
from kaos_llm_core.programs.classify.zero_shot import _format_labels_block
from kaos_llm_core.results import Classification
from kaos_llm_core.settings import KaosLLMCoreSettings
from kaos_llm_core.signatures import InputField, OutputField, Signature
from kaos_llm_core.types import Example


def _build_multi_label_signature(label_set: LabelSet) -> type[Signature]:
    """Construct a Signature whose output is a subset of label names."""
    return create_model(
        f"MultiLabelClassify_{abs(hash(label_set.names)) % 10_000:04d}_Signature",
        __base__=Signature,
        __doc__=(
            "Pick every label that applies to the input text. "
            "May be zero, one, or several labels. Provide a confidence "
            "for each picked label."
        ),
        text=(str, InputField(description="Source text to classify.")),
        labels_description=(
            str,
            InputField(
                description=(
                    "Description of the allowed label space. Pick all labels "
                    "that apply, none if none fit."
                ),
            ),
        ),
        picked_labels=(
            list[str],
            OutputField(
                default_factory=list,
                description=(
                    "List of label names that apply. Each must be one of the allowed labels."
                ),
            ),
        ),
        confidence=(
            dict[str, float],
            OutputField(
                default_factory=dict,
                description=(
                    "Map of picked label name to confidence in [0.0, 1.0]. "
                    "Labels not in ``picked_labels`` should be omitted."
                ),
            ),
        ),
        rationale=(
            str,
            OutputField(
                default="",
                description="Brief explanation of why these labels apply.",
            ),
        ),
    )


class MultiLabelClassify(Program):
    """LLM multi-label classification.

    Requires ``label_set.exclusive=False``. The LLM is prompted to emit
    every applicable label name plus per-label confidence.

    The returned :class:`Classification` preserves the
    :class:`LabelSet`'s declared order — even if the LLM lists labels
    out of order, the result mirrors the canonical declaration order
    so consumers see deterministic output across runs.
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
        **kwargs: Any,
    ) -> None:
        if labels.exclusive:
            raise ValueError(
                "MultiLabelClassify requires a multi-label LabelSet "
                "(exclusive=False); use ZeroShotClassify for exclusive."
            )
        self._label_set = labels
        self._labels_block = _format_labels_block(labels)
        signature = _build_multi_label_signature(labels)
        self.classifier = Call(
            signature,
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

    async def forward(  # ty: ignore[invalid-method-override]
        self,
        *,
        text: str,
        parent_id: str | None = None,
    ) -> Classification:
        result = await self.classifier(
            text=text,
            labels_description=self._labels_block,
        )
        picked = self._label_set.validate_picks(result.picked_labels)
        # Stable canonical order: keep only the picks that resolve to
        # known labels, in declared order.
        ordered = [name for name in self._label_set.names if name in set(picked)]
        scores = {name: float(result.confidence.get(name, 1.0)) for name in ordered}
        return Classification(
            labels=[self._label_set.by_name(name) for name in ordered],
            scores=scores,
            abstained=not ordered,
            rationale=result.rationale or None,
            chunks_used=[parent_id] if parent_id else [],
            metadata={
                "program": "MultiLabelClassify",
                "llm_pick_count": len(result.picked_labels),
                "validated_pick_count": len(ordered),
            },
        )


__all__ = ["MultiLabelClassify"]
