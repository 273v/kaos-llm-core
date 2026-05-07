"""ExecutionTrace — structured trace model for Call and Program execution."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from kaos_core.types.content import KaosModel
from pydantic import Field


class ExecutionTrace(KaosModel):
    """Structured trace of a single Call or Program execution.

    Traces are hierarchical: a Program's trace contains its children's traces.
    Every trace includes model, token usage, cost, and timing information.

    Serializable via ``model_dump()`` / ``model_validate()`` (KaosModel).
    """

    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    call_name: str = ""
    signature: str = ""
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    model: str = ""
    codec: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    retries: int = 0
    examples_used: int = 0
    children: list[ExecutionTrace] = Field(default_factory=list)
    error: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def total_cost_usd(self) -> float:
        """Total cost of this trace tree.

        For leaf traces (no children): returns own cost_usd.
        For program traces (has children): returns sum of children's total_cost_usd.
        This avoids double-counting since Program nodes store aggregated child
        cost in cost_usd already.
        """
        if self.children:
            return sum(c.total_cost_usd for c in self.children)
        return self.cost_usd

    @property
    def total_latency_ms(self) -> float:
        """Total latency of this trace (not summing children — they may be parallel)."""
        return self.latency_ms
