"""Signature — typed I/O contract for LLM functions.

A Signature is a Pydantic model that declares the inputs and outputs of an
LLM function. The class docstring becomes the instruction. Fields are marked
as ``InputField`` or ``OutputField``.

Example::

    class ExtractEntities(Signature):
        '''Extract named entities from legal text.'''
        text: str = InputField(description="Legal document text")
        entities: list[Entity] = OutputField(description="Extracted entities")
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Signature(BaseModel):
    """Base class for typed LLM function contracts.

    Subclass and declare ``InputField``/``OutputField`` attributes.
    The class docstring becomes the system instruction for the LLM call.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )
