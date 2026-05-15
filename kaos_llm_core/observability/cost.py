"""Cost estimation for LLM calls based on model pricing.

Provides per-model pricing and functions to estimate cost from token usage.

The canonical hand-maintained pricing table lives in
:mod:`kaos_llm_client.cost` (``MODEL_PRICING``). This module derives the
``PRICING`` mapping below from that single source at import time —
both bare model names (``"gpt-5"``) and provider-qualified aliases
(``"openai:gpt-5"``) end up in the dict so callers can look up costs
either way. There is exactly one place where rates are defined, and
everyone else imports.

Unknown-model handling
----------------------
When :func:`apply_cost_estimates` (or :func:`estimate_cost`) is invoked
with a trace whose ``model`` is **not** in :data:`PRICING`, the
function logs a one-shot warning naming the unknown model and falls
through to ``cost_usd=0.0``. This was previously silent — which let
the entire cost-rollup pipeline report ``$0.00`` end-to-end whenever
the installed ``kaos-llm-client`` lockfile lagged the model rate
card. The warning is rate-limited per-model so a long-running batch
doesn't flood logs.
"""

from __future__ import annotations

from dataclasses import dataclass

from kaos_core.logging import get_logger
from kaos_llm_client.cost import MODEL_PRICING
from kaos_llm_client.profiles import infer_provider

from kaos_llm_core.observability.traces import ExecutionTrace

logger = get_logger(__name__)

# Rate-limit the unknown-model warning to one log line per model name
# per process. Reset only on a fresh import.
_warned_unknown_models: set[str] = set()


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Per-million-token pricing for a model.

    This is a thin view over the dict-shaped rate card in
    :data:`kaos_llm_client.cost.MODEL_PRICING`. The client's table also
    carries ``cache_read`` / ``cache_creation`` columns for providers
    that publish them, but the trace-level cost roll-ups here only need
    the (input, output) tuple.
    """

    input_per_mtok: float
    output_per_mtok: float

    def estimate(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD."""
        return (
            input_tokens * self.input_per_mtok / 1_000_000
            + output_tokens * self.output_per_mtok / 1_000_000
        )


def _build_pricing_table() -> dict[str, ModelPricing]:
    """Derive ``PRICING`` from :data:`kaos_llm_client.cost.MODEL_PRICING`.

    For every entry in the client's canonical table we materialise:

    * the bare model name (``"gpt-5"``)
    * the provider-qualified alias when one can be inferred
      (``"openai:gpt-5"``)

    so callers can look up costs by either form. Models whose provider
    cannot be inferred (rare — covers exotic / custom names) only get
    the bare entry. Failures to infer a provider are silent because the
    rate card itself is still authoritative; the alias is convenience.
    """
    table: dict[str, ModelPricing] = {}
    for model, rates in MODEL_PRICING.items():
        pricing = ModelPricing(
            input_per_mtok=float(rates["input"]),
            output_per_mtok=float(rates["output"]),
        )
        table[model] = pricing
        provider = infer_provider(model)
        if provider is not None:
            table[f"{provider}:{model}"] = pricing
    return table


# Public pricing table — derived from the kaos-llm-client canonical
# source. Both bare names and ``provider:model`` aliases resolve to the
# same ``ModelPricing`` instance.
PRICING: dict[str, ModelPricing] = _build_pricing_table()


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
    elif trace.input_tokens > 0 and trace.model not in {"(program)", ""}:
        _warn_unknown_model(trace.model)

    # Recursive: sum child costs
    child_cost = sum(estimate_cost(child) for child in trace.children)

    return own_cost + child_cost


def _warn_unknown_model(model: str) -> None:
    """Log a one-shot warning for an unknown pricing key.

    The warning is rate-limited to once per model name per process so
    a long-running batch doesn't flood logs. The empty-string model
    name is silently ignored (some trace shapes record it as a
    placeholder).
    """
    if not model or model in _warned_unknown_models:
        return
    _warned_unknown_models.add(model)
    logger.warning(
        "Cost estimation: no pricing entry for model %r — cost_usd will be 0.0. "
        "Update kaos_llm_client.cost.MODEL_PRICING or refresh the lockfile.",
        model,
    )


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
    elif trace.input_tokens > 0 and trace.model not in {"(program)", ""}:
        _warn_unknown_model(trace.model)

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
