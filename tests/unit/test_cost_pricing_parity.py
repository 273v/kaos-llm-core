"""Pricing-table single-source-of-truth regression test.

``kaos-llm-client.cost.MODEL_PRICING`` is the **single hand-maintained**
rate card for every model the ecosystem knows about.
``kaos-llm-core.observability.cost.PRICING`` is **derived** from it at
import time — both bare names (``"gpt-5"``) and provider-qualified
aliases (``"openai:gpt-5"``) end up in the dict, all pointing at
``ModelPricing`` instances built straight from the canonical rates.

This test guards three properties of the derivation:

1. Every bare model in ``MODEL_PRICING`` resolves to a ``ModelPricing``
   in ``PRICING`` under at least its bare key.
2. The derived ``(input_per_mtok, output_per_mtok)`` exactly equals the
   canonical ``(input, output)`` row — no float drift.
3. Bare and provider-qualified entries share the same ``ModelPricing``
   instance (cheap identity check that proves the derivation collapsed
   the two views, not just produced parallel data).

Originally a parity test (KC16-2 / KC16-11); now a derivation
regression after PA17+PA19 consolidation (task #227). When this test
fails the fix is in the canonical ``MODEL_PRICING`` table, never in a
hand-maintained duplicate — the duplicate no longer exists.
"""

from __future__ import annotations

import pytest
from kaos_llm_client.cost import MODEL_PRICING as CLIENT_PRICING
from kaos_llm_client.profiles import infer_provider

from kaos_llm_core.observability.cost import PRICING as CORE_PRICING


@pytest.mark.parametrize("bare_model", sorted(CLIENT_PRICING.keys()))
def test_bare_model_derived_into_core(bare_model: str) -> None:
    """Every canonical bare model appears in derived PRICING."""
    assert bare_model in CORE_PRICING, (
        f"Derivation lost '{bare_model}' from MODEL_PRICING. The PRICING "
        f"table in kaos_llm_core.observability.cost is built by "
        f"_build_pricing_table(); investigate why the import-time loop "
        f"skipped this entry."
    )


@pytest.mark.parametrize("bare_model", sorted(CLIENT_PRICING.keys()))
def test_derived_rates_match_canonical(bare_model: str) -> None:
    """Derived ``ModelPricing`` carries the canonical numeric rates."""
    client_entry = CLIENT_PRICING[bare_model]
    core_entry = CORE_PRICING[bare_model]

    assert core_entry.input_per_mtok == pytest.approx(client_entry["input"]), (
        f"Input-rate drift for '{bare_model}': MODEL_PRICING says "
        f"${client_entry['input']}/MTok but derived ModelPricing has "
        f"${core_entry.input_per_mtok}/MTok. The derivation must be "
        f"transcribing the canonical row verbatim."
    )
    assert core_entry.output_per_mtok == pytest.approx(client_entry["output"]), (
        f"Output-rate drift for '{bare_model}': MODEL_PRICING says "
        f"${client_entry['output']}/MTok but derived ModelPricing has "
        f"${core_entry.output_per_mtok}/MTok."
    )


@pytest.mark.parametrize("bare_model", sorted(CLIENT_PRICING.keys()))
def test_provider_alias_shares_identity_with_bare(bare_model: str) -> None:
    """``provider:model`` and ``model`` resolve to the *same* instance.

    The derivation builds one ``ModelPricing`` per row and inserts it
    under both keys. Identity sharing is the property that proves there
    is no second copy of the rate to drift against.
    """
    provider = infer_provider(bare_model)
    if provider is None:
        pytest.skip(f"infer_provider returned None for '{bare_model}' — "
                    f"no provider-qualified alias is generated")
    qualified = f"{provider}:{bare_model}"
    assert qualified in CORE_PRICING, (
        f"infer_provider mapped '{bare_model}' to provider '{provider}' "
        f"but '{qualified}' is missing from PRICING. The derivation "
        f"should have inserted both keys."
    )
    assert CORE_PRICING[bare_model] is CORE_PRICING[qualified], (
        f"'{bare_model}' and '{qualified}' resolve to different "
        f"ModelPricing instances — the derivation lost the identity "
        f"sharing it relies on."
    )


def test_gpt_5_5_specifically_present() -> None:
    """KC16-2 named regression: gpt-5.5 must be priced.

    Retained as a named test so the audit trail keeps grepping for it.
    The implementation now passes by the same derivation discipline as
    every other model.
    """
    assert "gpt-5.5" in CORE_PRICING, (
        "gpt-5.5 missing from PRICING — KC16-2 / PA15 Gap #2 regression. "
        "The canonical rate ($5.00 / $30.00 per MTok) lives in "
        "kaos_llm_client.cost.MODEL_PRICING."
    )
    entry = CORE_PRICING["gpt-5.5"]
    assert (entry.input_per_mtok, entry.output_per_mtok) == pytest.approx((5.0, 30.0))
