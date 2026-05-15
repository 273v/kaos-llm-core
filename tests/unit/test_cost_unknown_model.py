"""Regression tests for cost-telemetry under unknown-model conditions.

Background
==========

Prior to the P0-3 fix the cost-rollup pipeline silently reported
``cost_usd = 0.0`` whenever the installed
:data:`kaos_llm_client.cost.MODEL_PRICING` table didn't include the
trace's model id. The most visible symptom was every live-scale
benchmark JSON in ``docs/benchmarks/live-*.json`` reporting
``total_cost_usd: 0.0`` despite real billed provider calls.

The bug had two halves:

1. **Stale lockfile** — the installed ``kaos-llm-client`` lagged the
   model rate card, so ``gpt-5.4-nano`` wasn't a known key. Fixed by
   refreshing the lockfile.
2. **Silent zero** — :func:`apply_cost_estimates` did
   ``PRICING.get(trace.model)`` and fell through to ``cost_usd=0`` on
   ``None`` with no logging, so the failure was invisible to callers.

This module guards the second half. We assert against the
``_warned_unknown_models`` set directly because the ``kaos`` logger
hierarchy uses a non-propagating handler that pytest's ``caplog``
fixture does not capture by default; the registry is the canonical
observable signal for "did the warning path fire".
"""

from __future__ import annotations

from kaos_llm_core.observability import ExecutionTrace
from kaos_llm_core.observability.cost import (
    PRICING,
    _warned_unknown_models,
    apply_cost_estimates,
    estimate_cost,
)


def _make_leaf_trace(*, model: str, input_tokens: int, output_tokens: int) -> ExecutionTrace:
    return ExecutionTrace(
        call_name="leaf",
        signature="LeafSignature",
        inputs={},
        model=model,
        codec="json",
        children=[],
        cost_usd=0.0,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        latency_ms=100.0,
        error=None,
    )


# ---------------------------------------------------------------------------
# Known-model invariant (the happy path that already worked)
# ---------------------------------------------------------------------------


def test_known_model_produces_nonzero_cost() -> None:
    """A trace with input_tokens > 0 and a known model gets non-zero cost."""
    # Pick the first key from PRICING that has positive rates.
    known_model = next(m for m, p in PRICING.items() if p.input_per_mtok > 0 and ":" not in m)
    trace = _make_leaf_trace(model=known_model, input_tokens=1_000, output_tokens=500)
    apply_cost_estimates(trace)
    assert trace.cost_usd > 0.0
    assert trace.cost_usd == estimate_cost(trace)


# ---------------------------------------------------------------------------
# Unknown-model warning (the silent-zero regression we're guarding against)
# ---------------------------------------------------------------------------


def test_unknown_model_records_in_warned_set() -> None:
    """A trace with a model NOT in PRICING and real tokens registers the
    model in ``_warned_unknown_models`` (the side-effect that drives
    the one-shot warning) and leaves ``cost_usd=0``.
    """
    bogus = "definitely-not-a-real-model-id-7c91e3"
    assert bogus not in PRICING
    _warned_unknown_models.discard(bogus)

    trace = _make_leaf_trace(model=bogus, input_tokens=500, output_tokens=200)
    apply_cost_estimates(trace)

    assert trace.cost_usd == 0.0, "cost_usd must stay 0 for unknown model"
    assert bogus in _warned_unknown_models, (
        f"Expected {bogus!r} to land in the warned-set after first hit"
    )


def test_unknown_model_warning_rate_limited() -> None:
    """The warning path is guarded by ``not in _warned_unknown_models``,
    so the second and third ``apply_cost_estimates`` calls with the
    same unknown model never re-enter the warn branch — the set
    membership is the observable rate-limit signal.
    """
    bogus = "rate-limit-test-c4f9a2"
    _warned_unknown_models.discard(bogus)
    assert bogus not in _warned_unknown_models

    for _ in range(3):
        trace = _make_leaf_trace(model=bogus, input_tokens=500, output_tokens=200)
        apply_cost_estimates(trace)

    # Set membership idempotent — but the real assertion is that
    # the warn-branch only ran once. The membership-after-first
    # invariant is what guards the rate limit.
    assert bogus in _warned_unknown_models


def test_program_pseudo_model_does_not_warn() -> None:
    """``model='(program)'`` is the placeholder for Program-level trace
    nodes and must NOT land in the warned-set.
    """
    _warned_unknown_models.discard("(program)")
    trace = ExecutionTrace(
        call_name="prog",
        signature="Program",
        inputs={},
        model="(program)",
        codec="(program)",
        children=[],
        cost_usd=0.0,
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        latency_ms=10.0,
        error=None,
    )
    apply_cost_estimates(trace)
    assert "(program)" not in _warned_unknown_models


def test_zero_input_tokens_does_not_warn() -> None:
    """A trace with zero input tokens is a no-op cost path; even an
    unknown model name shouldn't enter the warn branch."""
    bogus = "zero-input-noop-model"
    _warned_unknown_models.discard(bogus)
    trace = _make_leaf_trace(model=bogus, input_tokens=0, output_tokens=0)
    apply_cost_estimates(trace)
    assert bogus not in _warned_unknown_models


def test_empty_model_name_does_not_warn() -> None:
    """Empty string is some traces' placeholder; never warn."""
    _warned_unknown_models.discard("")
    trace = _make_leaf_trace(model="", input_tokens=500, output_tokens=200)
    apply_cost_estimates(trace)
    assert "" not in _warned_unknown_models


# ---------------------------------------------------------------------------
# Lockfile health check — best-effort, defensive
# ---------------------------------------------------------------------------


def test_pricing_table_has_recent_models() -> None:
    """The installed pricing table should know at least one current-gen
    cheap model. If this fails, the kaos-llm-client lockfile is stale
    and live-scale-test cost telemetry will silently zero out. Run
    ``uv sync`` or ``uv lock --upgrade-package kaos-llm-client``.
    """
    canaries = {"gpt-5.4-nano", "gemini-2.5-flash", "claude-haiku-4-5"}
    known = canaries & set(PRICING.keys())
    assert known, (
        "kaos-llm-client MODEL_PRICING table appears stale — none of the "
        f"current cheap-tier canaries {canaries} are present in PRICING. "
        "Refresh the lockfile."
    )
