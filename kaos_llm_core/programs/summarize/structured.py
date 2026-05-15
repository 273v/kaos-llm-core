"""Schema-driven structured summarization."""

from __future__ import annotations

from typing import Any

from kaos_llm_client import BaseProviderClient
from kaos_llm_client.settings import KaosLLMSettings
from pydantic import BaseModel, create_model

from kaos_llm_core.codecs.base import Codec
from kaos_llm_core.programs.base import Program
from kaos_llm_core.programs.call import Call
from kaos_llm_core.results import Summary
from kaos_llm_core.settings import KaosLLMCoreSettings
from kaos_llm_core.signatures import InputField, OutputField, Signature
from kaos_llm_core.types import Example


def _build_signature(schema: type[BaseModel]) -> type[Signature]:
    """Construct a :class:`Signature` whose output payload is ``schema``.

    The constructed signature has two output fields:

    - ``summary``: plain-text rendering of the summary.
    - ``payload``: an instance of ``schema``.

    Programs that don't want a textual summary alongside the schema
    can ignore ``summary`` on the consumer side; it costs ~one
    sentence per call and dramatically improves trace readability.
    """
    return create_model(
        f"StructuredSummary_{schema.__name__}_Signature",
        __base__=Signature,
        __doc__=(
            f"Summarize the input text and extract a structured "
            f"{schema.__name__} payload from it. "
            f"The plain-text summary should match the structured payload."
        ),
        text=(str, InputField(description="Source text to summarize.")),
        summary=(
            str,
            OutputField(
                description=("Plain-text summary that mirrors the structured payload."),
            ),
        ),
        payload=(
            schema,
            OutputField(
                description=(f"Structured {schema.__name__} extracted from the source."),
            ),
        ),
    )


class StructuredSummary(Program):
    """Schema-driven abstractive summarization.

    Constructs a :class:`Signature` at runtime whose ``payload`` output
    field carries an arbitrary caller-supplied Pydantic schema. The
    LLM produces both a plain-text summary and a typed structured
    payload in a single call.

    Args:
        schema: Pydantic model class describing the structured output.
        Remaining arguments forwarded to the underlying :class:`Call`.

    Example:
        >>> class ContractSummary(BaseModel):
        ...     parties: list[str]
        ...     effective_date: str
        ...     term_years: int
        >>> prog = StructuredSummary(schema=ContractSummary, model="...")
        >>> summary = await prog(text=contract_text)
        >>> summary.payload.parties
        ['Alpha', 'Beta']
    """

    def __init__(
        self,
        *,
        schema: type[BaseModel],
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
        self._schema = schema
        signature = _build_signature(schema)
        self.summarizer = Call(
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
    ) -> Summary[BaseModel]:
        result = await self.summarizer(text=text)
        return Summary[BaseModel](
            text=result.summary,
            payload=result.payload,
            method="abstractive",
            chunks_used=[parent_id] if parent_id else [],
            metadata={
                "program": "StructuredSummary",
                "schema": self._schema.__name__,
            },
        )


__all__ = ["StructuredSummary"]
