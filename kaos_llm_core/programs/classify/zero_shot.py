"""Zero-shot and few-shot LLM classification.

The Signature is constructed once at Program-init time around the
supplied :class:`~kaos_llm_core.labels.LabelSet`. The ``label`` output
field is declared with the label-name ``Literal`` so providers that
support constrained decoding (OpenAI's ``response_format``, Anthropic's
``tool_use``, etc.) push the constraint into the wire format. For
providers without constrained decoding, the Program validates the
LLM's pick against the label set at forward-time and surfaces a
:class:`~kaos_llm_core.errors.ValidationRetryExhaustedError` when the
LLM is stubbornly wrong.
"""

from __future__ import annotations

from typing import Any, Literal

from kaos_llm_client import BaseProviderClient
from kaos_llm_client.settings import KaosLLMSettings
from pydantic import create_model

from kaos_llm_core.codecs.base import Codec
from kaos_llm_core.labels import ABSTAIN_LABEL, LabelSet
from kaos_llm_core.programs.base import Program
from kaos_llm_core.programs.call import Call
from kaos_llm_core.results import Classification
from kaos_llm_core.settings import KaosLLMCoreSettings
from kaos_llm_core.signatures import InputField, OutputField, Signature
from kaos_llm_core.types import Example


def _format_labels_block(label_set: LabelSet) -> str:
    """Render a LabelSet as an instructional block for the prompt.

    Output format::

        Allowed labels:
        - <name>: <description>
        - <name>: <description>
        ...
        Abstention: <yes|no>
    """
    lines = ["Allowed labels:"]
    for label in label_set:
        if label.description:
            lines.append(f"- {label.name}: {label.description}")
        else:
            lines.append(f"- {label.name}")
    abstain = "yes" if label_set.allow_abstain else "no"
    lines.append(f"Abstention permitted: {abstain}")
    return "\n".join(lines)


def _build_zero_shot_signature(label_set: LabelSet) -> type[Signature]:
    """Construct a Signature with a Literal-typed ``label`` output field."""
    names: tuple[str, ...] = label_set.names
    choices: tuple[str, ...] = (*names, ABSTAIN_LABEL) if label_set.allow_abstain else names
    label_literal = Literal[choices]  # ty: ignore[invalid-type-form]

    return create_model(
        f"ZeroShotClassify_{abs(hash(names)) % 10_000:04d}_Signature",
        __base__=Signature,
        __doc__=(
            "Classify the input text into one of the allowed labels. "
            "Pick exactly one label. Explain your reasoning briefly."
        ),
        text=(str, InputField(description="Source text to classify.")),
        labels_description=(
            str,
            InputField(
                description=(
                    "Description of the allowed label space. Read this carefully "
                    "before picking a label."
                ),
            ),
        ),
        label=(
            label_literal,
            OutputField(
                description=("Chosen label name. Must be one of the allowed labels above."),
            ),
        ),
        confidence=(
            float,
            OutputField(
                default=1.0,
                description="Confidence in [0.0, 1.0]; default 1.0.",
            ),
        ),
        rationale=(
            str,
            OutputField(
                default="",
                description="Brief explanation of why this label was chosen.",
            ),
        ),
    )


class ZeroShotClassify(Program):
    """Zero-shot LLM classification against a :class:`LabelSet`.

    Args:
        labels: Label space. Must be ``exclusive=True``;
            :class:`MultiLabelClassify` handles the multi-label case.
        model / codec / client / settings / core_settings /
        examples / instructions / max_retries: forwarded to the
        underlying :class:`Call`.
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
        if not labels.exclusive:
            raise ValueError(
                "ZeroShotClassify requires an exclusive LabelSet; "
                "use MultiLabelClassify for multi-label spaces."
            )
        self._label_set = labels
        self._labels_block = _format_labels_block(labels)
        signature = _build_zero_shot_signature(labels)
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
        picked: str = result.label
        if picked == ABSTAIN_LABEL or picked not in self._label_set:
            return Classification(
                labels=[],
                scores={ABSTAIN_LABEL: float(result.confidence)},
                abstained=True,
                rationale=result.rationale or None,
                chunks_used=[parent_id] if parent_id else [],
                metadata={
                    "program": "ZeroShotClassify",
                    "abstain_reason": "llm_abstained"
                    if picked == ABSTAIN_LABEL
                    else "unknown_label",
                },
            )
        return Classification(
            labels=[self._label_set.by_name(picked)],
            scores={picked: float(result.confidence)},
            abstained=False,
            rationale=result.rationale or None,
            chunks_used=[parent_id] if parent_id else [],
            metadata={"program": "ZeroShotClassify"},
        )


class FewShotClassify(ZeroShotClassify):
    """:class:`ZeroShotClassify` with a few-shot example pool.

    Identical surface to :class:`ZeroShotClassify`; the only difference
    is that ``examples`` is treated as required. Examples are
    rendered into the prompt via the existing
    :class:`~kaos_llm_core.programs.call.Call` few-shot pipeline.
    """

    def __init__(
        self,
        *,
        labels: LabelSet,
        examples: list[Example],
        **kwargs: Any,
    ) -> None:
        if not examples:
            raise ValueError(
                "FewShotClassify requires at least one Example; "
                "use ZeroShotClassify for zero examples."
            )
        super().__init__(labels=labels, examples=examples, **kwargs)


__all__ = [
    "FewShotClassify",
    "ZeroShotClassify",
]
