"""Phase 15.5 benchmark: drive ``batch_run()`` against **real Anthropic Haiku**.

This benchmark hits a real provider end-to-end. It measures what
agents actually pay for: wall-clock throughput including HTTP, retry,
codec decode, validation, cost attribution, JSONL log writes, and
cumulative cost tracking. There is no mocked client. Mocked benchmarks
are worse than nothing because they measure Python overhead rather
than the real system.

Default config is sized to stay under **$0.10 worst-case** against the
cheapest current-generation Anthropic model (claude-haiku-4-5):

  - n = 50 sentiment-classification items (~50 input tokens each)
  - concurrency = 8
  - hard cost cap enforced at the end of every run

Usage::

    # Default 50-item benchmark against Haiku (~$0.005, ~10s)
    KAOS_LLM_ANTHROPIC_API_KEY=... \
        uv run python benchmarks/benchmark_batch_run.py

    # Bigger run; opt-in only.
    KAOS_LLM_ANTHROPIC_API_KEY=... \
        uv run python benchmarks/benchmark_batch_run.py --n 200 --concurrency 16

    # The 1000-item run promised by Phase 15.5. Costs ~$0.10 against Haiku.
    KAOS_LLM_ANTHROPIC_API_KEY=... \
        uv run python benchmarks/benchmark_batch_run.py --full

    # Resume contract at scale: run N, kill the log to half, resume.
    KAOS_LLM_ANTHROPIC_API_KEY=... \
        uv run python benchmarks/benchmark_batch_run.py --resume-test --n 30

    # JSON output for piping into a tracker
    KAOS_LLM_ANTHROPIC_API_KEY=... \
        uv run python benchmarks/benchmark_batch_run.py --json

The benchmark refuses to run without a real API key. There is no
"--mock" mode by design.

This file is excluded from ``ty check`` because the benchmarks dir is
benchmark scripts, not library code (per the project's
benchmark-exclusion policy).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from kaos_llm_core.programs.batch import batch_run, list_input_source
from kaos_llm_core.programs.envelope import from_envelope, program_hash

HAIKU = "claude-haiku-4-5"
HARD_COST_CAP_USD = 0.50  # Belt-and-suspenders. Default config ~$0.005.


# A small, varied corpus of sentiment-classification inputs. Real
# English sentences with mixed valence so the model has to actually
# read them — no degenerate "asdf" inputs that would break a real
# benchmark's representativeness.
_BASE_INPUTS = [
    "I absolutely loved this product, it exceeded every expectation.",
    "Worst customer service experience of my entire life.",
    "It is a chair. It does what a chair does.",
    "Five stars — would buy again in a heartbeat.",
    "I'm not really sure how I feel about this purchase.",
    "The packaging arrived damaged but the product itself works.",
    "Mediocre at best. I expected more for the price.",
    "An absolute disaster from start to finish.",
    "Surprisingly delightful — small details done right.",
    "Functional but uninspired. Gets the job done.",
    "Total scam, do not waste your money.",
    "Fantastic value for the price point.",
    "I have mixed feelings; some great parts and some frustrating.",
    "Everyone in my family loves it.",
    "Returned it within 24 hours. Save yourself the trouble.",
    "Adequate. Not great, not terrible.",
    "Genuinely the best thing I've bought all year.",
    "Hard to recommend at this price.",
    "Decent, but the competition is stronger.",
    "Brilliant design, terrible execution.",
]


def _envelope() -> dict[str, Any]:
    return {
        "kaos_program": "1",
        "name": "bench-sentiment",
        "inputs": {"text": {"type": "string"}},
        "clients": {"default": {"provider": "anthropic", "model": HAIKU}},
        "steps": [
            {
                "id": "classify",
                "kind": "call",
                "client": "default",
                "instruction": (
                    "Classify the sentiment of the input as positive, "
                    "negative, or neutral. Reply with a single word."
                ),
                "inputs": {"text": "$.inputs.text"},
                "output_fields": {
                    "sentiment": {
                        "description": "positive | negative | neutral",
                        "type": {"type": "string"},
                    }
                },
            }
        ],
        "output": {"sentiment": "$.steps.classify.output.sentiment"},
        "capabilities": ["call", "jsonpointer_refs"],
    }


def _build_inputs(n: int) -> list[dict[str, Any]]:
    """Tile the base corpus to length n with a unique index suffix per
    item. The suffix breaks content-addressed deduplication so every
    item is a distinct custom_id — otherwise resume statistics get
    distorted by silent dedup of repeated text."""
    return [{"text": f"[item {i:04d}] {_BASE_INPUTS[i % len(_BASE_INPUTS)]}"} for i in range(n)]


async def _run_once(
    *,
    n: int,
    concurrency: int,
    output_dir: Path,
    resume: bool = False,
) -> tuple[float, Any]:
    envelope = _envelope()
    program = from_envelope(envelope)
    env_hash = program_hash(envelope)
    started = time.monotonic()
    result = await batch_run(
        program,
        list_input_source(_build_inputs(n), program_hash_value=env_hash),
        output_dir=str(output_dir),
        max_concurrency=concurrency,
        resume=resume,
    )
    return time.monotonic() - started, result


async def _bench(args: argparse.Namespace) -> dict[str, Any]:
    if not (os.getenv("KAOS_LLM_ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        raise SystemExit(
            "ERROR: this benchmark requires KAOS_LLM_ANTHROPIC_API_KEY (or "
            "ANTHROPIC_API_KEY). There is no mocked mode — mocked benchmarks "
            "measure Python overhead rather than the real batch system."
        )

    n = 1000 if args.full else args.n
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "bench"
        elapsed, result = await _run_once(n=n, concurrency=args.concurrency, output_dir=out_dir)
        if result.cost_usd > HARD_COST_CAP_USD:
            raise SystemExit(
                f"ERROR: hard cost cap ${HARD_COST_CAP_USD} exceeded "
                f"(${result.cost_usd:.6f}). Aborting."
            )
        log_path = Path(result.log_path)
        log_size = log_path.stat().st_size if log_path.exists() else 0

        report: dict[str, Any] = {
            "model": HAIKU,
            "n": n,
            "concurrency": args.concurrency,
            "runtime_s": round(elapsed, 4),
            "throughput_items_s": round(n / elapsed, 2) if elapsed > 0 else None,
            "log_size_bytes": log_size,
            "cost_usd": round(result.cost_usd, 6),
            "cost_per_item_usd": (round(result.cost_usd / n, 6) if n > 0 else None),
            "tokens_total": result.total_tokens,
            "tokens_input": result.input_tokens,
            "tokens_output": result.output_tokens,
            "n_succeeded": result.n_succeeded,
            "n_errored": result.n_errored,
            "status": result.status,
            "errors_by_type": dict(result.errors_by_type),
        }

        if args.resume_test:
            # Truncate the log to half its lines and resume.
            lines = log_path.read_text().splitlines()
            half = lines[: 1 + (len(lines) - 1) // 2]  # header + half items
            kept_items = max(0, len(half) - 1)
            log_path.write_text("\n".join(half) + "\n")
            elapsed2, result2 = await _run_once(
                n=n, concurrency=args.concurrency, output_dir=out_dir, resume=True
            )
            report["resume"] = {
                "items_in_truncated_log": kept_items,
                "runtime_s": round(elapsed2, 4),
                "n_total": result2.n_total,
                "n_skipped": result2.n_skipped,
                "n_succeeded": result2.n_succeeded,
                "n_errored": result2.n_errored,
                "this_run_cost_usd": round(result2.cost_usd, 6),
            }
        return report


def _print_human(report: dict[str, Any]) -> None:
    print(f"[batch_bench] model:     {report['model']}")
    print(f"[batch_bench] config:    n={report['n']}, concurrency={report['concurrency']}")
    print(f"[batch_bench] runtime:   {report['runtime_s']} s")
    if report.get("throughput_items_s") is not None:
        print(f"[batch_bench] throughput: {report['throughput_items_s']} items/s")
    log_kb = report["log_size_bytes"] / 1024
    print(f"[batch_bench] log size:  {report['log_size_bytes']} bytes ({log_kb:.1f} KB)")
    print(
        f"[batch_bench] cost:      ${report['cost_usd']:.6f} "
        f"(${report['cost_per_item_usd']:.6f}/item, real Haiku)"
    )
    print(
        f"[batch_bench] tokens:    {report['tokens_total']} total "
        f"({report['tokens_input']} in, {report['tokens_output']} out)"
    )
    print(f"[batch_bench] succeeded: {report['n_succeeded']} / {report['n']}")
    print(f"[batch_bench] errored:   {report['n_errored']}")
    if report["errors_by_type"]:
        print(f"[batch_bench] errors:    {report['errors_by_type']}")
    if "resume" in report:
        r = report["resume"]
        print()
        print(f"[batch_bench] resume:    truncated log to {r['items_in_truncated_log']} items")
        print(f"[batch_bench] resume:    re-run took {r['runtime_s']} s")
        print(
            f"[batch_bench] resume:    n_skipped={r['n_skipped']}, "
            f"n_succeeded={r['n_succeeded']} (cumulative)"
        )
        print(
            f"[batch_bench] resume:    this_run_cost=${r['this_run_cost_usd']:.6f} "
            "(should be roughly half of fresh run)"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--n",
        type=int,
        default=50,
        help="Number of items to process (default: 50, ~$0.005 against Haiku)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="The 1000-item Phase 15.5 benchmark (~$0.10 against Haiku)",
    )
    parser.add_argument("--concurrency", type=int, default=8, help="Async semaphore cap")
    parser.add_argument(
        "--resume-test",
        action="store_true",
        help="Truncate the log mid-run and resume — proves the contract at scale",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of human text")
    args = parser.parse_args(argv)

    report = asyncio.run(_bench(args))
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human(report)
    return 0 if report["n_errored"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
