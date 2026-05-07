"""Cost estimation for LLM calls based on model pricing.

Provides per-model pricing tables and functions to estimate cost from
token usage. Prices are approximate and updated periodically.
"""

from __future__ import annotations

from dataclasses import dataclass

from kaos_llm_core.observability.traces import ExecutionTrace


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Per-million-token pricing for a model."""

    input_per_mtok: float
    output_per_mtok: float

    def estimate(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD."""
        return (
            input_tokens * self.input_per_mtok / 1_000_000
            + output_tokens * self.output_per_mtok / 1_000_000
        )


# Approximate pricing as of April 2026.
# Source: provider pricing pages. Updated periodically.
#
# kaos-llm-client's ``infer_provider`` accepts both ``provider:model``
# and bare model names (e.g. ``claude-haiku-4-5``). The pricing
# lookup must support both forms — if a caller passes the bare name,
# the cost tracker would otherwise return $0.00 silently. Every
# entry below has a ``provider:model`` key plus a bare alias.
PRICING: dict[str, ModelPricing] = {
    # Anthropic
    "anthropic:claude-opus-4-6": ModelPricing(15.0, 75.0),
    "anthropic:claude-sonnet-4-6": ModelPricing(3.0, 15.0),
    "anthropic:claude-haiku-4-5": ModelPricing(0.80, 4.0),
    "claude-opus-4-6": ModelPricing(15.0, 75.0),
    "claude-sonnet-4-6": ModelPricing(3.0, 15.0),
    "claude-haiku-4-5": ModelPricing(0.80, 4.0),
    # OpenAI
    "openai:gpt-5.4": ModelPricing(2.50, 10.0),
    "openai:gpt-5.4-mini": ModelPricing(0.40, 1.60),
    "openai:gpt-5.4-nano": ModelPricing(0.10, 0.40),
    "openai:gpt-5": ModelPricing(2.0, 8.0),
    "openai:o4-mini": ModelPricing(1.10, 4.40),
    "openai:o3": ModelPricing(10.0, 40.0),
    "openai:o3-mini": ModelPricing(1.10, 4.40),
    "gpt-5.4": ModelPricing(2.50, 10.0),
    "gpt-5.4-mini": ModelPricing(0.40, 1.60),
    "gpt-5.4-nano": ModelPricing(0.10, 0.40),
    "gpt-5": ModelPricing(2.0, 8.0),
    "o4-mini": ModelPricing(1.10, 4.40),
    "o3": ModelPricing(10.0, 40.0),
    "o3-mini": ModelPricing(1.10, 4.40),
    # Google
    "google:gemini-3.1-pro-preview": ModelPricing(1.25, 10.0),
    "google:gemini-3-flash-preview": ModelPricing(0.15, 0.60),
    "google:gemini-2.5-pro": ModelPricing(1.25, 10.0),
    "google:gemini-2.5-flash": ModelPricing(0.15, 0.60),
    "gemini-3.1-pro-preview": ModelPricing(1.25, 10.0),
    "gemini-3-flash-preview": ModelPricing(0.15, 0.60),
    "gemini-2.5-pro": ModelPricing(1.25, 10.0),
    "gemini-2.5-flash": ModelPricing(0.15, 0.60),
}


def estimate_cost(trace: ExecutionTrace) -> float:
    """Estimate the USD cost of an ExecutionTrace using known pricing.

    If the model is not in the pricing table, returns 0.0 (unknown cost).
    For traces with children, recursively estimates and sums child costs.

    Args:
        trace: An ExecutionTrace from a Call or Program execution.

    Returns:
        Estimated cost in USD.
    """
    # Leaf trace: estimate from model pricing
    own_cost = 0.0
    pricing = PRICING.get(trace.model)
    if pricing:
        own_cost = pricing.estimate(trace.input_tokens, trace.output_tokens)

    # Recursive: sum child costs
    child_cost = sum(estimate_cost(child) for child in trace.children)

    return own_cost + child_cost


def apply_cost_estimates(trace: ExecutionTrace) -> None:
    """Mutate an ExecutionTrace tree, filling in cost_usd from pricing.

    Walks the trace tree bottom-up, setting ``cost_usd`` on each node.

    Args:
        trace: Root ExecutionTrace to annotate.
    """
    for child in trace.children:
        apply_cost_estimates(child)

    pricing = PRICING.get(trace.model)
    if pricing:
        trace.cost_usd = pricing.estimate(trace.input_tokens, trace.output_tokens)

    # If this is a program-level trace (has children), aggregate
    if trace.children and trace.model == "(program)":
        trace.cost_usd = sum(c.cost_usd for c in trace.children)


def format_cost_report(trace: ExecutionTrace) -> str:
    """Format a human-readable cost report from a trace tree.

    Args:
        trace: Root ExecutionTrace.

    Returns:
        Multi-line string with cost breakdown.
    """
    lines: list[str] = []
    total = estimate_cost(trace)

    lines.append(f"Program: {trace.call_name}")
    lines.append(f"  Total cost: ${total:.4f}")
    lines.append(f"  Total tokens: {trace.total_tokens:,}")
    lines.append(f"  Latency: {trace.latency_ms:.0f}ms")

    if trace.children:
        lines.append("  Calls:")
        for child in trace.children:
            child_cost = estimate_cost(child)
            lines.append(
                f"    {child.call_name}: ${child_cost:.4f} "
                f"({child.input_tokens:,}+{child.output_tokens:,} tokens, "
                f"{child.latency_ms:.0f}ms, model={child.model})"
            )

    return "\n".join(lines)
