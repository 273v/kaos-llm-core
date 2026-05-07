"""Optimization Demo — bootstrap + instruction tuning for SEC violation severity.

Runs a real optimization loop on a genuinely hard task: classifying the severity
of SEC enforcement actions from complaint excerpts. The categories are nuanced
enough that a cheap model makes mistakes without examples.

Usage:
    uv run python -m examples.optimization_demo [--provider anthropic|openai|google] [--json]
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from examples.models import get_preset
from kaos_llm_core import Call, InputField, OutputField, Signature
from kaos_llm_core.optimization.bootstrap import BootstrapOptimizer
from kaos_llm_core.optimization.evaluation import evaluate
from kaos_llm_core.optimization.mutations import MutationLog
from kaos_llm_core.types import Example

# --- Signature ---


class ClassifySeverity(Signature):
    """Classify the severity of an SEC enforcement action."""

    text: str = InputField(description="SEC enforcement action excerpt")
    severity: str = OutputField(description="Severity: minor, moderate, or severe")


# --- Labeled Data ---
# These are designed so that cheap models will misclassify some —
# the boundary between moderate and severe is genuinely ambiguous
# without understanding SEC enforcement patterns.

TRAIN_SET = [
    Example(
        inputs={
            "text": "The company failed to file its 10-K annual report within the required "
            "60-day deadline on two occasions in 2024. No material misstatements were found "
            "in the eventually filed reports."
        },
        outputs={"severity": "minor"},
    ),
    Example(
        inputs={
            "text": "The investment adviser charged clients management fees of 2.5% when the "
            "advisory agreement specified 1.8%, resulting in approximately $340,000 in excess "
            "fees collected over 18 months. The adviser self-reported the overcharges and "
            "initiated refunds before the SEC examination."
        },
        outputs={"severity": "moderate"},
    ),
    Example(
        inputs={
            "text": "The CEO directed the CFO to capitalize $45 million in operating expenses "
            "over three quarters to meet analyst revenue targets, while personally selling "
            "$12 million in company stock during the inflated period. The scheme was concealed "
            "from the company's audit committee and external auditors."
        },
        outputs={"severity": "severe"},
    ),
    Example(
        inputs={
            "text": "A broker-dealer failed to maintain minimum net capital requirements on "
            "four trading days during a volatile market period. The shortfalls ranged from "
            "$50,000 to $200,000 and were corrected by the next business day each time."
        },
        outputs={"severity": "minor"},
    ),
    Example(
        inputs={
            "text": "The fund manager allocated profitable trades to personal accounts and "
            "losing trades to client accounts over a two-year period. Total client losses "
            "from cherry-picking were estimated at $8.5 million across 200 affected accounts."
        },
        outputs={"severity": "severe"},
    ),
    Example(
        inputs={
            "text": "The company's investor relations department issued a press release "
            "containing revenue projections that differed materially from internal forecasts. "
            "The VP of IR was unaware of updated Q3 numbers when the release was drafted. "
            "The stock price moved 4% on the release before a correction was issued same day."
        },
        outputs={"severity": "moderate"},
    ),
    Example(
        inputs={
            "text": "An auditing firm issued unqualified opinions for three years while "
            "its lead partner maintained an undisclosed financial relationship with the "
            "client's CEO, including shared real estate investments worth $1.2 million."
        },
        outputs={"severity": "severe"},
    ),
    Example(
        inputs={
            "text": "The municipal adviser failed to register with the SEC as required under "
            "the Dodd-Frank Act but was otherwise conducting its advisory business in "
            "compliance with substantive requirements. The firm registered within 30 days "
            "of receiving the SEC's notice."
        },
        outputs={"severity": "minor"},
    ),
]

VAL_SET = [
    Example(
        inputs={
            "text": "A registered representative made unauthorized trades in 12 client "
            "accounts, generating $85,000 in excess commissions. The trades were unsuitable "
            "for the clients' stated investment objectives and risk tolerances."
        },
        outputs={"severity": "moderate"},
    ),
    Example(
        inputs={
            "text": "The hedge fund manager created fictitious account statements showing "
            "consistent 12% annual returns while the fund had actually lost 40% of assets. "
            "Investor redemption requests were paid with new investor funds for three years "
            "before the scheme collapsed, affecting $500 million in investor capital."
        },
        outputs={"severity": "severe"},
    ),
    Example(
        inputs={
            "text": "The transfer agent processed shareholder name changes without obtaining "
            "required medallion signature guarantees on 45 occasions over six months. "
            "No fraudulent transfers resulted from the procedural lapses."
        },
        outputs={"severity": "minor"},
    ),
    Example(
        inputs={
            "text": "The company's CEO made public statements about a pending FDA approval "
            "that he knew was unlikely based on Phase III trial data shared with him in "
            "confidence. Three board members sold shares worth $4.2 million in the week "
            "following his statements."
        },
        outputs={"severity": "severe"},
    ),
    Example(
        inputs={
            "text": "A mutual fund failed to properly calculate its NAV on 22 occasions due "
            "to a software error in its pricing system. The errors ranged from 0.02% to 0.15% "
            "and affected approximately 3,000 shareholders. The fund corrected NAVs and "
            "reimbursed affected shareholders within 60 days of discovery."
        },
        outputs={"severity": "minor"},
    ),
    Example(
        inputs={
            "text": "The broker-dealer's compliance department identified a pattern of late "
            "trade reporting but failed to file a suspicious activity report. The underlying "
            "trades involved a client who was later charged with insider trading. The "
            "compliance department had no knowledge of the insider trading."
        },
        outputs={"severity": "moderate"},
    ),
]


def exact_match(prediction: Any, gold: dict[str, Any]) -> float:
    """1.0 if severity matches exactly, 0.0 otherwise."""
    return 1.0 if prediction.severity == gold["severity"] else 0.0


async def run(provider: str | None = None, json_output: bool = False) -> dict[str, Any]:
    """Run the optimization demo."""
    preset = get_preset(provider)
    model = preset.cheap

    print("Optimization Demo: SEC Violation Severity Classification")
    print(f"{'=' * 60}")
    print(f"Model: {model}")
    print(f"Training examples: {len(TRAIN_SET)}")
    print(f"Validation examples: {len(VAL_SET)}")
    print()

    call = Call(ClassifySeverity, model=model)

    # Step 1: Baseline
    print("Step 1: Baseline evaluation (no few-shot examples)...")
    baseline = await evaluate(call, VAL_SET, exact_match)
    print(f"  Accuracy: {baseline.score:.0%} ({baseline.n_correct}/{baseline.n_total})")
    if baseline.failures():
        print("  Misclassifications:")
        for f in baseline.failures():
            print(
                f"    expected={f.example.outputs['severity']}, "
                f"got={f.prediction.severity}, "
                f"text={f.example.inputs['text'][:60]}..."
            )
    print()

    # Step 2: Bootstrap
    print("Step 2: Bootstrap optimization...")
    log = MutationLog()
    optimizer = BootstrapOptimizer(
        metric=exact_match,
        max_examples=4,
        score_threshold=1.0,
        mutation_log=log,
    )
    result = await optimizer.optimize(call, TRAIN_SET, VAL_SET)

    print(f"  Demos selected from training: {len(result.examples_added)}")
    print(f"  Accuracy after bootstrap: {result.metric_after:.0%}")
    print(f"  Accepted: {result.accepted}")
    if result.accepted:
        print(f"  Improvement: {result.metric_before:.0%} → {result.metric_after:.0%}")
    print()

    if result.examples_added:
        print("Selected few-shot demonstrations:")
        for ex in result.examples_added:
            print(f"  [{ex.outputs['severity']:8s}] {ex.inputs['text'][:65]}...")
        print()

    # Step 3: Final check
    print("Step 3: Final evaluation...")
    final = await evaluate(call, VAL_SET, exact_match)
    print(f"  Final accuracy: {final.score:.0%} ({final.n_correct}/{final.n_total})")
    if final.failures():
        print("  Remaining misclassifications:")
        for f in final.failures():
            print(f"    expected={f.example.outputs['severity']}, got={f.prediction.severity}")
    print()

    print(log.summary())

    output = {
        "model": model,
        "baseline_accuracy": baseline.score,
        "optimized_accuracy": result.metric_after,
        "final_accuracy": final.score,
        "examples_added": len(result.examples_added),
        "accepted": result.accepted,
    }
    if json_output:
        print()
        print(json.dumps(output, indent=2, default=str))

    return output


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Optimization demo")
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "google"],
        default=None,
    )
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args()

    asyncio.run(run(provider=args.provider, json_output=args.json_output))


if __name__ == "__main__":
    main()
