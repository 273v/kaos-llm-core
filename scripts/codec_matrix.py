"""Phase 16.3 codec regression matrix.

Runs every (codec x provider x signature) combination against a real
provider, records pass/fail per cell into a JSONL audit log, and
generates ``docs/reference/codec-matrix.md`` as a published grid.

This is the "is JSONCodec actually reliable on Anthropic?" answer
that DSPy ships and we previously did not. There is no mocked mode —
the matrix only means anything if it ran against real APIs.

Usage::

    # Run all available providers + all codecs (default)
    KAOS_LLM_ANTHROPIC_API_KEY=... KAOS_LLM_OPENAI_API_KEY=... \
        uv run python scripts/codec_matrix.py

    # Restrict to one provider for a quick smoke
    uv run python scripts/codec_matrix.py --provider anthropic

    # Restrict to one codec
    uv run python scripts/codec_matrix.py --codec json

    # Emit JSON instead of writing the markdown report
    uv run python scripts/codec_matrix.py --json

    # Custom report location
    uv run python scripts/codec_matrix.py --output ../docs/reference/codec-matrix.md

The script enforces a hard $0.20 cost cap across all cells. Each cell
runs one Call against the selected (codec, provider, model, signature)
tuple and records:

  - whether the call succeeded
  - the latency in milliseconds
  - the cost in USD
  - the validated output (or the exception type + message on failure)

Auditable artifacts:

  - ``docs/reference/codec-matrix.md`` (or --output) — the published
    pass/fail grid that any reader can verify
  - ``benchmarks/codec_matrix.jsonl`` — the per-cell JSONL audit log
    with full timing + cost + output for every cell
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from kaos_llm_core.codecs import ChatCodec, Codec, JSONCodec, XMLCodec
from kaos_llm_core.observability.cost import PRICING
from kaos_llm_core.programs.call import Call
from kaos_llm_core.signatures import InputField, OutputField, Signature

# ---------------------------------------------------------------------------
# Matrix dimensions
# ---------------------------------------------------------------------------

CODECS: dict[str, type[Codec]] = {
    "json": JSONCodec,
    "chat": ChatCodec,
    "xml": XMLCodec,
}

# Provider id → (env-var preflight, default model). The default model
# is the cheapest current-generation model per the kaos-llm-client
# test_live.py landscape.
PROVIDERS: dict[str, tuple[tuple[str, ...], str]] = {
    "anthropic": (
        ("KAOS_LLM_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
        "anthropic:claude-haiku-4-5",
    ),
    "openai": (
        ("KAOS_LLM_OPENAI_API_KEY", "OPENAI_API_KEY"),
        "openai:gpt-5.4-nano",
    ),
    "google": (
        ("KAOS_LLM_GOOGLE_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY"),
        "google:gemini-2.5-flash",
    ),
}

HARD_COST_CAP_USD = 0.20


# ---------------------------------------------------------------------------
# Canonical signature set (small + varied)
# ---------------------------------------------------------------------------


class _Sentiment(Signature):
    """Classify the sentiment of the input text."""

    text: str = InputField(description="A short text snippet")
    sentiment: str = OutputField(description="positive, negative, or neutral")


class _ExtractList(Signature):
    """Extract every distinct entity name mentioned in the text."""

    text: str = InputField(description="Source text")
    entities: list[str] = OutputField(description="List of entity names")


class _Numeric(Signature):
    """Solve a one-line arithmetic problem."""

    question: str = InputField(description="The arithmetic question")
    answer: int = OutputField(description="The numeric answer as an integer")


class _MultiField(Signature):
    """Extract a structured contract triple."""

    text: str = InputField(description="A short contract clause")
    parties: list[str] = OutputField(description="Party names")
    effective_date: str = OutputField(description="Effective date string or 'unknown'")
    obligations: list[str] = OutputField(description="Obligation phrases")


_SIGNATURES: dict[str, tuple[type[Signature], dict[str, Any], dict[str, Any]]] = {
    # name → (Signature, inputs, expected_check_keys)
    "sentiment": (
        _Sentiment,
        {"text": "I absolutely love this product, it's wonderful."},
        {"sentiment": str},
    ),
    "extract_list": (
        _ExtractList,
        {"text": "Acme Corp signed a deal with Beta LLC and Gamma Inc."},
        {"entities": list},
    ),
    "numeric": (
        _Numeric,
        {"question": "What is twelve plus seven?"},
        {"answer": int},
    ),
    "multi_field": (
        _MultiField,
        {
            "text": (
                "Effective July 1, 2026, Acme Corp shall deliver 100 widgets "
                "monthly to Beta LLC and provide 24/7 technical support."
            )
        },
        {"parties": list, "effective_date": str, "obligations": list},
    ),
}


# ---------------------------------------------------------------------------
# Cell result
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _CellResult:
    codec: str
    provider: str
    model: str
    signature: str
    succeeded: bool
    latency_ms: float
    cost_usd: float
    error_type: str | None
    error_message: str | None
    output_keys: list[str]
    timestamp: str


def _has_keys(env_vars: tuple[str, ...]) -> bool:
    return any(os.getenv(v) for v in env_vars)


async def _run_cell(
    *,
    codec_name: str,
    codec_cls: type[Codec],
    provider: str,
    model: str,
    sig_name: str,
) -> _CellResult:
    sig, inputs, expected = _SIGNATURES[sig_name]
    started = time.monotonic()
    error_type: str | None = None
    error_message: str | None = None
    output_keys: list[str] = []
    cost_usd = 0.0
    succeeded = False

    try:
        call = Call(sig, model=model, codec=codec_cls(), max_retries=1)
        invocation = await call.invoke(**inputs)
        output = invocation.output
        # Verify the expected keys exist on the output (the structural check)
        for key, expected_type in expected.items():
            if not hasattr(output, key):
                msg = f"output missing field {key!r}"
                raise ValueError(msg)
            value = getattr(output, key)
            if not isinstance(value, expected_type):
                msg = (
                    f"output field {key!r} has type {type(value).__name__}, "
                    f"expected {expected_type.__name__}"
                )
                raise TypeError(msg)
        output_keys = sorted(expected)
        cost_usd = float(invocation.usage.cost_usd or 0.0)
        succeeded = True
    except Exception as exc:
        error_type = type(exc).__name__
        error_message = str(exc)[:300]

    latency_ms = (time.monotonic() - started) * 1000
    return _CellResult(
        codec=codec_name,
        provider=provider,
        model=model,
        signature=sig_name,
        succeeded=succeeded,
        latency_ms=round(latency_ms, 1),
        cost_usd=round(cost_usd, 6),
        error_type=error_type,
        error_message=error_message,
        output_keys=output_keys,
        timestamp=__import__("datetime")
        .datetime.now(__import__("datetime").UTC)
        .isoformat()
        .replace("+00:00", "Z"),
    )


def _print_grid(results: list[_CellResult]) -> str:
    """Build a Markdown grid of pass/fail per (codec, provider, signature)."""
    lines: list[str] = []
    lines.append("# Codec Regression Matrix")
    lines.append("")
    lines.append("**Phase 16.3** — published pass/fail grid for every")
    lines.append("(codec x provider x signature) combination, regenerated")
    lines.append("by `scripts/codec_matrix.py` against real provider APIs.")
    lines.append("There is no mocked mode.")
    lines.append("")
    lines.append("Last regenerated: " + (results[0].timestamp if results else "n/a"))
    lines.append("")

    # Group by (provider, codec)
    providers_seen: list[str] = []
    codecs_seen: list[str] = []
    sigs_seen: list[str] = []
    for r in results:
        if r.provider not in providers_seen:
            providers_seen.append(r.provider)
        if r.codec not in codecs_seen:
            codecs_seen.append(r.codec)
        if r.signature not in sigs_seen:
            sigs_seen.append(r.signature)

    # One table per provider
    for provider in providers_seen:
        provider_rows = [r for r in results if r.provider == provider]
        if not provider_rows:
            continue
        sample_model = provider_rows[0].model
        lines.append(f"## {provider} (`{sample_model}`)")
        lines.append("")
        header = "| codec \\ signature | " + " | ".join(sigs_seen) + " |"
        sep = "|" + "---|" * (len(sigs_seen) + 1)
        lines.append(header)
        lines.append(sep)
        for codec in codecs_seen:
            cells: list[str] = []
            for sig in sigs_seen:
                cell = next(
                    (r for r in provider_rows if r.codec == codec and r.signature == sig),
                    None,
                )
                if cell is None:
                    cells.append("—")
                elif cell.succeeded:
                    cells.append(f"PASS ({cell.latency_ms:.0f}ms)")
                else:
                    cells.append(f"FAIL ({cell.error_type})")
            lines.append(f"| {codec} | " + " | ".join(cells) + " |")
        lines.append("")

    # Aggregate stats
    total = len(results)
    passed = sum(1 for r in results if r.succeeded)
    cost = sum(r.cost_usd for r in results)
    lines.append("## Aggregate")
    lines.append("")
    lines.append(f"- **Cells**: {total}")
    lines.append(f"- **Pass rate**: {passed} / {total} ({100 * passed / max(total, 1):.1f}%)")
    lines.append(f"- **Total cost**: ${cost:.6f}")
    lines.append("")
    lines.append("Per-cell audit log: `benchmarks/codec_matrix.jsonl`.")
    lines.append("")
    return "\n".join(lines)


async def _run_matrix(args: argparse.Namespace) -> tuple[list[_CellResult], float]:
    # Filter providers
    selected_providers = list(PROVIDERS) if args.provider is None else [args.provider]
    available: list[tuple[str, str]] = []
    for p in selected_providers:
        if p not in PROVIDERS:
            print(f"Unknown provider: {p}", file=sys.stderr)
            sys.exit(2)
        env_vars, model = PROVIDERS[p]
        if not _has_keys(env_vars):
            print(
                f"Skipping {p}: no API key (set one of {env_vars})",
                file=sys.stderr,
            )
            continue
        # Backfill PRICING for the model if not already there — we
        # cannot fail on missing pricing because some models drop in
        # before the table is updated.
        if model.split(":", 1)[-1] not in PRICING:
            from kaos_llm_core.observability.cost import ModelPricing

            PRICING[model.split(":", 1)[-1]] = ModelPricing(input_per_mtok=1.0, output_per_mtok=2.0)
        available.append((p, model))

    if not available:
        print(
            "ERROR: no providers available. Set at least one of "
            "KAOS_LLM_ANTHROPIC_API_KEY / KAOS_LLM_OPENAI_API_KEY / "
            "KAOS_LLM_GOOGLE_API_KEY.",
            file=sys.stderr,
        )
        sys.exit(1)

    selected_codecs = list(CODECS) if args.codec is None else [args.codec]
    if any(c not in CODECS for c in selected_codecs):
        print(f"Unknown codec; valid: {sorted(CODECS)}", file=sys.stderr)
        sys.exit(2)

    selected_sigs = list(_SIGNATURES) if args.signature is None else [args.signature]
    if any(s not in _SIGNATURES for s in selected_sigs):
        print(f"Unknown signature; valid: {sorted(_SIGNATURES)}", file=sys.stderr)
        sys.exit(2)

    cells: list[tuple[str, type[Codec], str, str, str]] = []
    for codec_name in selected_codecs:
        codec_cls = CODECS[codec_name]
        for provider, model in available:
            for sig_name in selected_sigs:
                cells.append((codec_name, codec_cls, provider, model, sig_name))

    print(
        f"[codec_matrix] running {len(cells)} cells "
        f"({len(selected_codecs)} codecs x {len(available)} providers x "
        f"{len(selected_sigs)} signatures)",
        file=sys.stderr,
    )

    results: list[_CellResult] = []
    for codec_name, codec_cls, provider, model, sig_name in cells:
        cell_result = await _run_cell(
            codec_name=codec_name,
            codec_cls=codec_cls,
            provider=provider,
            model=model,
            sig_name=sig_name,
        )
        status = "PASS" if cell_result.succeeded else f"FAIL ({cell_result.error_type})"
        print(
            f"  {codec_name:>5} x {provider:<10} x {sig_name:<13} {status} "
            f"({cell_result.latency_ms:.0f}ms, ${cell_result.cost_usd:.6f})",
            file=sys.stderr,
        )
        results.append(cell_result)
        running_cost = sum(r.cost_usd for r in results)
        if running_cost > HARD_COST_CAP_USD:
            print(
                f"ERROR: hard cost cap ${HARD_COST_CAP_USD} exceeded "
                f"(${running_cost:.6f}). Aborting.",
                file=sys.stderr,
            )
            sys.exit(3)

    return results, sum(r.cost_usd for r in results)


def _write_jsonl(results: list[_CellResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), separators=(",", ":")) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--provider", choices=list(PROVIDERS), help="Restrict to one provider")
    parser.add_argument("--codec", choices=list(CODECS), help="Restrict to one codec")
    parser.add_argument(
        "--signature",
        choices=list(_SIGNATURES),
        help="Restrict to one signature",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent.parent / "docs" / "reference" / "codec-matrix.md",
        help="Where to write the markdown grid",
    )
    parser.add_argument(
        "--audit-log",
        type=Path,
        default=Path(__file__).parent.parent / "benchmarks" / "codec_matrix.jsonl",
        help="Where to write the per-cell JSONL audit log",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON to stdout")
    args = parser.parse_args(argv)

    results, total_cost = asyncio.run(_run_matrix(args))
    _write_jsonl(results, args.audit_log)

    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2))
    else:
        grid = _print_grid(results)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(grid)
        print(f"\n[codec_matrix] wrote {args.output} ({len(results)} cells)", file=sys.stderr)
        print(f"[codec_matrix] audit log: {args.audit_log}", file=sys.stderr)
        print(f"[codec_matrix] total cost: ${total_cost:.6f}", file=sys.stderr)

    return 0 if all(r.succeeded for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
