"""Core types for kaos-llm-core.

- ``Example`` — a labeled input-output pair for few-shot learning.
"""

from __future__ import annotations

from typing import Any

from kaos_core.types.content import KaosModel
from pydantic import Field


class Example(KaosModel):
    """A labeled input-output pair used for few-shot demonstrations.

    Examples teach the LLM by showing concrete instances of the desired behavior.
    They are the primary "learnable parameters" that optimizers tune.
    """

    inputs: dict[str, Any]
    outputs: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)
