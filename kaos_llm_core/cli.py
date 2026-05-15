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

    # summarize command — plan §7.2 declarative CLI.
    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Summarize a text file using the declarative starter façade",
    )
    summarize_parser.add_argument(
        "file",
        type=str,
        help="Path to a UTF-8 text file to summarize. Use '-' for stdin.",
    )
    summarize_parser.add_argument(
        "--model", type=str, default=None, help="Provider model id (overrides settings)."
    )
    summarize_parser.add_argument(
        "--strategy",
        choices=["auto", "single", "tree", "refine"],
        default="auto",
        help="long_strategy passed to summarize_doc (default: auto).",
    )
    summarize_parser.add_argument(
        "--cited", action="store_true", help="Route the single-call path through CitedSummary."
    )
    summarize_parser.add_argument(
        "--budget-tokens",
        type=int,
        default=None,
        help="Token budget cap; processing halts when reached.",
    )
    summarize_parser.add_argument(
        "--budget-usd",
        type=float,
        default=None,
        help="Cost budget cap in USD; processing halts when reached.",
    )
    summarize_parser.add_argument("--pretty", action="store_true", help="Human-readable output.")
    summarize_parser.add_argument(
        "--cost", action="store_true", help="Print cost / token totals from the Summary metadata."
    )

    # classify command — plan §7.2 declarative CLI.
    classify_parser = subparsers.add_parser(
        "classify",
        help="Classify a text file against a JSON labels file.",
    )
    classify_parser.add_argument(
        "file", type=str, help="Path to a UTF-8 text file to classify. Use '-' for stdin."
    )
    classify_parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Path to a JSON file containing either a list of label names or "
        "a serialized LabelSet (model_dump).",
    )
    classify_parser.add_argument("--model", type=str, default=None)
    classify_parser.add_argument(
        "--strategy",
        choices=["auto", "single", "chunk"],
        default="auto",
        help="long_strategy passed to classify_doc (default: auto).",
    )
    classify_parser.add_argument(
        "--supervision",
        choices=["zero_shot", "few_shot"],
        default="zero_shot",
    )
    classify_parser.add_argument(
        "--budget-tokens", type=int, default=None, help="Token budget cap."
    )
    classify_parser.add_argument(
        "--budget-usd", type=float, default=None, help="Cost budget cap in USD."
    )
    classify_parser.add_argument("--pretty", action="store_true")
    classify_parser.add_argument("--cost", action="store_true")

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
    elif args.command == "summarize":
        _cmd_summarize(args)
    elif args.command == "classify":
        _cmd_classify(args)


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


# ---------------------------------------------------------------------------
# summarize / classify — plan §7.2 declarative CLI
# ---------------------------------------------------------------------------


def _read_text(path: str) -> str:
    """Read UTF-8 text from a file or stdin (``-``)."""
    if path == "-":
        return sys.stdin.read()
    from pathlib import Path

    return Path(path).expanduser().read_text(encoding="utf-8")


def _load_labels(path: str):
    """Load a JSON labels file into a :class:`LabelSet`.

    Accepts either a list of strings or a serialized LabelSet (the
    same shape `LabelSet.model_dump()` produces).
    """
    import json
    from pathlib import Path

    from kaos_llm_core.labels import LabelSet

    raw = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return LabelSet.from_names(raw)
    return LabelSet.model_validate(raw)


def _resolve_budget(tokens: int | None, usd: float | None):
    """Return a :class:`Budget` for the supplied caps, or ``None``."""
    if tokens is None and usd is None:
        return None
    from kaos_llm_core.optimization.budget import Budget

    return Budget(max_tokens=tokens, max_cost_usd=usd)


def _cmd_summarize(args: argparse.Namespace) -> None:
    """``kaos-llm-core summarize <file>`` — plan §7.2."""
    import asyncio
    import json

    from kaos_llm_core.starter import summarize_doc

    text = _read_text(args.file)
    budget = _resolve_budget(args.budget_tokens, args.budget_usd)
    result = asyncio.run(
        summarize_doc(
            text,
            model=args.model,
            long_strategy=args.strategy,
            cited=args.cited,
            budget=budget,
        )
    )

    if args.pretty:
        print(result.text)
        if args.cost:
            print()
            strategy_used = result.metadata.get("starter.long_strategy")
            print(f"strategy: {strategy_used}")
            cost = result.metadata.get("budget.cost_usd")
            tokens = result.metadata.get("budget.tokens")
            if cost is not None or tokens is not None:
                print(f"cost: ${cost or 0.0:.6f}    tokens: {tokens or 0}")
        return
    print(json.dumps(result.model_dump(mode="json"), indent=2))


def _cmd_classify(args: argparse.Namespace) -> None:
    """``kaos-llm-core classify <file> --labels labels.json`` — plan §7.2."""
    import asyncio
    import json

    from kaos_llm_core.starter import classify_doc

    text = _read_text(args.file)
    labels = _load_labels(args.labels)
    budget = _resolve_budget(args.budget_tokens, args.budget_usd)
    result = asyncio.run(
        classify_doc(
            text,
            labels,
            model=args.model,
            supervision=args.supervision,
            long_strategy=args.strategy,
            budget=budget,
        )
    )

    if args.pretty:
        top = result.top_label or "(abstained)"
        print(top)
        for name in result.names:
            score = result.scores.get(name)
            print(f"  - {name}    score={score!r}")
        if args.cost:
            print()
            cost = result.metadata.get("budget.cost_usd")
            tokens = result.metadata.get("budget.tokens")
            print(f"cost: ${cost or 0.0:.6f}    tokens: {tokens or 0}")
        return
    print(json.dumps(result.model_dump(mode="json"), indent=2))
