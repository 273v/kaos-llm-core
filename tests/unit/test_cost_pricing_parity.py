"""Cross-sibling pricing-table parity regression test.

Closes KC16-11 (and the structural half of KC16-2): the pricing tables
in ``kaos-llm-client`` and ``kaos-llm-core`` must not silently diverge.
``kaos-llm-client.cost.MODEL_PRICING`` is the authoritative source for
base ``(input, output)`` rates; ``kaos-llm-core.observability.cost.PRICING``
is the consumer that ``ExecutionTrace`` cost roll-ups read from.

This test enforces a one-directional subset relationship:

- Every bare model key in ``kaos-llm-client.MODEL_PRICING`` must appear
  in ``kaos-llm-core.PRICING`` (either as the bare key, ``"gpt-5"``, or
  the provider-qualified alias, ``"openai:gpt-5"``).
- The numeric ``input`` / ``output`` per-MTok rates must agree within
  ``pytest.approx`` tolerance — float-rounding from copy-paste is fine,
  silent semantic drift is not.

``kaos-llm-core`` may legitimately track additional models that the
client doesn't yet price (e.g. preview-tier Gemini, the ``gpt-5.4``
family). The reverse direction is intentionally NOT enforced; the
client owns the wire contract for every model it actually dispatches
to.

When this test fails the fix is to mirror the missing / divergent rows
from ``kaos-llm-client.cost.MODEL_PRICING`` into
``kaos-llm-core.observability.cost.PRICING`` and re-run.
"""

from __future__ import annotations

import pytest
from kaos_llm_client.cost import MODEL_PRICING as CLIENT_PRICING

from kaos_llm_core.observability.cost import PRICING as CORE_PRICING


def _resolve_core_rates(bare_model: str) -> tuple[float, float] | None:
    """Look up (input, output) rates for ``bare_model`` in core's PRICING.

    Tries the bare key first (``"gpt-5"``), then each provider-qualified
    alias (``"openai:gpt-5"``, ``"anthropic:gpt-5"``, ``"google:gpt-5"``,
    ``"xai:gpt-5"``). Returns ``None`` if not found under any form.
    """
    if bare_model in CORE_PRICING:
        entry = CORE_PRICING[bare_model]
        return (entry.input_per_mtok, entry.output_per_mtok)
    for provider in ("openai", "anthropic", "google", "xai"):
        qualified = f"{provider}:{bare_model}"
        if qualified in CORE_PRICING:
            entry = CORE_PRICING[qualified]
            return (entry.input_per_mtok, entry.output_per_mtok)
    return None


@pytest.mark.parametrize("bare_model", sorted(CLIENT_PRICING.keys()))
def test_client_model_present_in_core(bare_model: str) -> None:
    """Every bare model in the client table must exist in core's table."""
    rates = _resolve_core_rates(bare_model)
    assert rates is not None, (
        f"Pricing-table divergence: kaos-llm-client.MODEL_PRICING knows "
        f"about '{bare_model}' but kaos-llm-core.observability.cost.PRICING "
        f"does not. Mirror the entry — see "
        f"kaos-llm-client/kaos_llm_client/cost.py for the authoritative "
        f"(input, output, cache_read, cache_creation) row."
    )


@pytest.mark.parametrize("bare_model", sorted(CLIENT_PRICING.keys()))
def test_client_model_rates_agree_with_core(bare_model: str) -> None:
    """Numeric (input, output) rates must agree across client and core."""
    client_entry = CLIENT_PRICING[bare_model]
    client_input = client_entry["input"]
    client_output = client_entry["output"]

    core_rates = _resolve_core_rates(bare_model)
    assert core_rates is not None, (
        f"'{bare_model}' missing from core PRICING — see "
        f"test_client_model_present_in_core for the parity fix."
    )
    core_input, core_output = core_rates

    assert core_input == pytest.approx(client_input), (
        f"Input-rate mismatch for '{bare_model}': "
        f"kaos-llm-client says ${client_input}/MTok, "
        f"kaos-llm-core says ${core_input}/MTok. Mirror the client value."
    )
    assert core_output == pytest.approx(client_output), (
        f"Output-rate mismatch for '{bare_model}': "
        f"kaos-llm-client says ${client_output}/MTok, "
        f"kaos-llm-core says ${core_output}/MTok. Mirror the client value."
    )


def test_gpt_5_5_specifically_present() -> None:
    """KC16-2 specifically: gpt-5.5 must be priced in core.

    A targeted regression test on top of the parametrized scan so that
    the KC16-2 finding has a named test the audit trail can grep for.
    """
    rates = _resolve_core_rates("gpt-5.5")
    assert rates is not None, (
        "gpt-5.5 pricing missing from kaos-llm-core.observability.cost.PRICING "
        "— this is the KC16-2 / PA15 Gap #2 regression. kaos-llm-client "
        "prices gpt-5.5 at $5.00 / $30.00 per MTok (input / output)."
    )
    assert rates == pytest.approx((5.0, 30.0)), (
        f"gpt-5.5 rate drift: core says {rates}, client says (5.0, 30.0)."
    )
