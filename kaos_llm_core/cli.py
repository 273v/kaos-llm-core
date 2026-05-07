"""CLI entry point for kaos-llm-core."""

from __future__ import annotations

import argparse
import sys
from typing import Any


def main(argv: list[str] | None = None) -> None:
    """Entry point for the kaos-llm-core CLI."""
    parser = argparse.ArgumentParser(
        prog="kaos-llm-core",
        description="KAOS LLM Core — LLM programming primitives",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # check command
    check_parser = subparsers.add_parser("check", help="Verify configuration")
    check_parser.add_argument("--json", action="store_true", dest="json_output")

    # examples command
    examples_parser = subparsers.add_parser("examples", help="Run example programs")
    examples_sub = examples_parser.add_subparsers(dest="example", help="Example to run")

    _providers = ["anthropic", "openai", "google"]

    contract_parser = examples_sub.add_parser("contract", help="Contract clause analysis")
    contract_parser.add_argument("--file", type=str, default=None)
    contract_parser.add_argument("--json", action="store_true", dest="json_output")
    contract_parser.add_argument("--provider", choices=_providers, default=None)

    financial_parser = examples_sub.add_parser("financial", help="Earnings transcript extraction")
    financial_parser.add_argument("--file", type=str, default=None)
    financial_parser.add_argument("--json", action="store_true", dest="json_output")
    financial_parser.add_argument("--provider", choices=_providers, default=None)

    cascade_parser = examples_sub.add_parser("cascade", help="Cascade routing with cost report")
    cascade_parser.add_argument("--file", type=str, default=None)
    cascade_parser.add_argument("--json", action="store_true", dest="json_output")
    cascade_parser.add_argument("--threshold", type=float, default=0.8)
    cascade_parser.add_argument("--provider", choices=[*_providers, "cross"], default=None)

    optimize_parser = examples_sub.add_parser(
        "optimize", help="Bootstrap + instruction optimization"
    )
    optimize_parser.add_argument("--json", action="store_true", dest="json_output")
    optimize_parser.add_argument("--provider", choices=_providers, default=None)

    # analyze command — Phase 7.3 prerequisite. Loads a MutationLog JSONL,
    # renders trial cards, computes per-strategy contributions, and prints a
    # human-readable report or a --json envelope per docs/guides/cli-standard.md.
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze a mutation-log JSONL file from an optimization run",
    )
    analyze_parser.add_argument(
        "log",
        type=str,
        help="Path to a mutation-log JSONL file (e.g., ~/.kaos/optimization-runs/<run_id>.jsonl)",
    )
    analyze_parser.add_argument("--json", action="store_true", dest="json_output")
    analyze_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Show only the first N trial cards (default: all)",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "check":
        _cmd_check(args)
    elif args.command == "examples":
        _cmd_examples(args)
    elif args.command == "analyze":
        _cmd_analyze(args)


def _cmd_check(args: argparse.Namespace) -> None:
    """Check kaos-llm-core configuration."""
    import json

    from kaos_llm_core._version import __version__
    from kaos_llm_core.settings import KaosLLMCoreSettings

    settings = KaosLLMCoreSettings()
    result = {
        "command": "check",
        "version": __version__,
        "default_model": settings.default_model,
        "trace_enabled": settings.trace_enabled,
        "status": "ok",
    }

    if args.json_output:
        print(json.dumps(result, indent=2))
    else:
        print(f"kaos-llm-core v{__version__}")
        print(f"  default_model: {settings.default_model or '(not set)'}")
        print(f"  trace_enabled: {settings.trace_enabled}")
        print("  status: ok")


def _cmd_examples(args: argparse.Namespace) -> None:
    """Run an example program."""
    import asyncio
    from pathlib import Path

    if args.example is None:
        print("Available examples: contract, financial, cascade, optimize")
        print("Usage: kaos-llm-core examples <name> [--provider PROVIDER] [--json]")
        sys.exit(1)

    file_path = Path(args.file) if args.file else None

    provider = getattr(args, "provider", None)

    if args.example == "contract":
        from examples.contract_analysis import run

        asyncio.run(run(contract_path=file_path, json_output=args.json_output, provider=provider))

    elif args.example == "financial":
        from examples.financial_extraction import run

        asyncio.run(run(transcript_path=file_path, json_output=args.json_output, provider=provider))

    elif args.example == "cascade":
        from examples.cascade_routing import run

        asyncio.run(
            run(
                doc_path=file_path,
                json_output=args.json_output,
                quality_threshold=args.threshold,
                provider=provider,
            )
        )

    elif args.example == "optimize":
        from examples.optimization_demo import run

        asyncio.run(run(provider=provider, json_output=args.json_output))


def _cmd_analyze(args: argparse.Namespace) -> None:
    """Analyze a mutation-log JSONL file (Phase 7.3 CLI).

    Loads the log via :func:`kaos_llm_core.optimization.analysis.load_mutations`,
    builds trial cards and strategy contributions, and prints either a
    human-readable report (default) or a ``--json`` envelope.

    The ``--json`` envelope shape (per docs/guides/cli-standard.md):

    .. code-block:: json

        {
          "command": "analyze",
          "log": "<path>",
          "summary": {<aggregated metrics>},
          "by_strategy": {<per-strategy contribution>},
          "trials": [<TrialCard records>]
        }

    Errors go to stderr with non-zero exit. Trial numbers in the human
    output are 1-based; the JSON envelope uses the underlying 0-based
    ``trial_id`` from the mutation record.
    """
    import json
    from dataclasses import asdict
    from pathlib import Path

    from kaos_llm_core.optimization.analysis import (
        load_mutations,
        make_trial_cards,
        strategy_contributions,
        summarize_run,
    )

    log_path = Path(args.log).expanduser()
    try:
        mutations = load_mutations(log_path)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(
            f"error: failed to load mutation log {log_path}: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        sys.exit(3)

    if not mutations:
        print(f"error: mutation log {log_path} is empty", file=sys.stderr)
        sys.exit(4)

    summary = summarize_run(mutations)
    contributions = strategy_contributions(mutations)
    cards = make_trial_cards(mutations)
    if args.limit is not None and args.limit > 0:
        cards = cards[: args.limit]

    if args.json_output:
        envelope = {
            "command": "analyze",
            "log": str(log_path),
            "summary": _jsonify(summary),
            "by_strategy": [_jsonify(asdict(c)) for c in contributions],
            "trials": [_jsonify(asdict(card)) for card in cards],
        }
        print(json.dumps(envelope, indent=2, default=str))
        return

    # Human-readable output.
    print(f"Mutation log: {log_path}")
    print(f"  total trials:        {summary.get('total_trials', 0)}")
    print(f"  accepted:            {summary.get('accepted', 0)}")
    print(f"  total cost:          ${summary.get('total_cost_usd', 0.0):.6f}")
    print(f"  total tokens:        {summary.get('total_tokens', 0):,}")
    print(f"  best metric_after:   {summary.get('best_metric_after', 0.0):.4f}")
    print(f"  best improvement:    {summary.get('best_improvement', 0.0):+.4f}")

    if contributions:
        print()
        print("By strategy:")
        for c in contributions:
            print(
                f"  {c.strategy:24s}  trials={c.trials:4d}  "
                f"wins={c.wins:4d}  Δ={c.total_improvement:+.4f}  "
                f"avg_cost=${c.average_cost:.6f}"
            )

    print()
    print(f"Trials ({len(cards)} of {len(mutations)}):")
    for i, card in enumerate(cards, start=1):
        print(f"--- {i} ---")
        print(card.render())


def _jsonify(value: Any) -> Any:
    """Best-effort JSON sanitizer for the analyze envelope.

    Converts ``datetime`` objects to ISO 8601 strings and leaves everything
    else as-is. The CLI ``--json`` envelope must be parseable by anything
    consuming the schema doc.
    """
    from datetime import datetime

    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _jsonify(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonify(v) for v in value]
    return value
