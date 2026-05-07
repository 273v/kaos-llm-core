"""Cascade Routing — judge-verified multi-model escalation with cost tracking.

Uses a separate judge model to evaluate classification quality instead of
self-reported confidence (which is meaningless). The cascade tries cheap models
first; after each attempt, an independent judge scores the output. If the
score is below threshold, the cascade escalates to the next model.

Demonstrates: CascadeRouter, Judge-based quality gating, cross-provider routing,
ExecutionTrace with per-model cost breakdown.

Usage:
    uv run python -m examples.cascade_routing
    [--provider anthropic|openai|google|cross] [--file PATH] [--json]
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from examples.models import get_preset
from kaos_llm_core import (
    Call,
    InputField,
    OutputField,
    Signature,
)
from kaos_llm_core.observability.cost import apply_cost_estimates, estimate_cost

FIXTURES_DIR = Path(__file__).parent / "fixtures"
DEFAULT_COMPLAINT = FIXTURES_DIR / "sec_complaint.txt"

# Cross-provider cascade: cheapest from each provider
CROSS_PROVIDER_MODELS = [
    "anthropic:claude-haiku-4-5",
    "openai:gpt-5.4-nano",
    "google:gemini-2.5-flash",
]

# The judge is always a stronger model — it evaluates the cheap model's work.
JUDGE_MODEL = "anthropic:claude-sonnet-4-6"


# --- Signatures ---


class ClassifyDocument(Signature):
    """Classify the type and severity of a legal document.

    Determine the document type, the primary legal issue, and the severity
    of the matter.
    """

    text: str = InputField(description="Legal document text")
    document_type: str = OutputField(
        description="Document type: complaint, motion, order, settlement, contract, or other"
    )
    primary_issue: str = OutputField(
        description="Primary legal issue (e.g., securities fraud, breach of contract)"
    )
    severity: str = OutputField(description="Severity: low, medium, high, critical")
    reasoning: str = OutputField(description="Brief explanation of the classification")


class JudgeClassification(Signature):
    """Evaluate the quality of a legal document classification.

    You are an expert legal analyst reviewing another analyst's classification.
    Score the classification based on: correct document type identification,
    accurate primary issue extraction, appropriate severity assessment, and
    quality of reasoning.

    Be critical. A score of 0.8+ means the classification is production-ready.
    A score below 0.8 means it has material errors or omissions.
    """

    document_text: str = InputField(description="The original legal document")
    classification: str = InputField(description="The classification to evaluate (JSON)")
    quality_score: float = OutputField(
        description="Quality score from 0.0 (completely wrong) to 1.0 (perfect)"
    )
    issues: list[str] = OutputField(
        description="List of specific issues found (empty if quality is high)"
    )


# --- Runner ---


def _get_cascade_models(provider: str | None) -> list[str]:
    """Get the cascade model list based on provider selection."""
    if provider == "cross":
        return CROSS_PROVIDER_MODELS
    preset = get_preset(provider)
    return [preset.cheap, preset.balanced]


async def run(
    doc_path: Path | None = None,
    json_output: bool = False,
    quality_threshold: float = 0.8,
    provider: str | None = None,
) -> dict[str, Any]:
    """Run judge-verified cascade classification.

    For each model in the cascade:
    1. Classify the document
    2. Have the judge evaluate the classification
    3. If judge score >= threshold, accept; otherwise escalate
    """
    path = doc_path or DEFAULT_COMPLAINT
    text = path.read_text(encoding="utf-8")
    models = _get_cascade_models(provider)

    judge_call = Call(JudgeClassification, model=JUDGE_MODEL)

    cascade_steps: list[dict[str, Any]] = []
    accepted_result = None
    accepted_model: str | None = None

    for model in models:
        classify_call = Call(ClassifyDocument, model=model)

        # Step 1: Classify
        classify_inv = await classify_call.invoke(text=text)
        result = classify_inv.output
        classify_trace = classify_inv.trace

        # Step 2: Judge the classification
        classification_json = json.dumps(
            {
                "document_type": result.document_type,
                "primary_issue": result.primary_issue,
                "severity": result.severity,
                "reasoning": result.reasoning,
            }
        )
        judge_inv = await judge_call.invoke(
            document_text=text,
            classification=classification_json,
        )
        judgment = judge_inv.output
        judge_trace = judge_inv.trace

        # Record this cascade step
        classify_cost = estimate_cost(classify_trace) if classify_trace else 0.0
        judge_cost = estimate_cost(judge_trace) if judge_trace else 0.0
        if classify_trace:
            apply_cost_estimates(classify_trace)
        if judge_trace:
            apply_cost_estimates(judge_trace)

        step = {
            "model": model,
            "judge_score": judgment.quality_score,
            "judge_issues": judgment.issues,
            "accepted": False,
            "classify_tokens": classify_trace.total_tokens if classify_trace else 0,
            "judge_tokens": judge_trace.total_tokens if judge_trace else 0,
            "classify_cost": classify_cost,
            "judge_cost": judge_cost,
            "total_cost": classify_cost + judge_cost,
            "classify_latency_ms": classify_trace.latency_ms if classify_trace else 0,
            "judge_latency_ms": judge_trace.latency_ms if judge_trace else 0,
        }

        if judgment.quality_score >= quality_threshold:
            step["accepted"] = True
            cascade_steps.append(step)
            accepted_result = result
            accepted_model = model
            break

        cascade_steps.append(step)

    # If no model was accepted, use the last one
    if accepted_result is None:
        accepted_result = result  # type: ignore[possibly-undefined]
        accepted_model = models[-1]
        if cascade_steps:
            cascade_steps[-1]["accepted"] = True

    total_cost = sum(s["total_cost"] for s in cascade_steps)

    output: dict[str, Any] = {
        "file": str(path),
        "provider": provider or "anthropic",
        "judge_model": JUDGE_MODEL,
        "model_used": accepted_model,
        "models_tried": len(cascade_steps),
        "quality_threshold": quality_threshold,
        "result": {
            "document_type": accepted_result.document_type,
            "primary_issue": accepted_result.primary_issue,
            "severity": accepted_result.severity,
            "reasoning": accepted_result.reasoning,
        },
        "cascade": cascade_steps,
        "total_cost_usd": total_cost,
    }

    if json_output:
        print(json.dumps(output, indent=2, default=str))
    else:
        print(f"Judge-Verified Cascade: {path.name} (provider: {provider or 'anthropic'})")
        print(f"{'=' * 60}")
        print(f"Quality threshold: {quality_threshold}")
        print(f"Judge model: {JUDGE_MODEL}")
        print(f"Models tried: {len(cascade_steps)}")
        print(f"Model accepted: {accepted_model}")
        print()

        for step in cascade_steps:
            status = "ACCEPTED" if step["accepted"] else "ESCALATED"
            score = step["judge_score"]
            print(
                f"  {step['model']:40s} {status:10s} "
                f"judge={score:.0%}  "
                f"${step['total_cost']:.4f}  "
                f"{step['classify_tokens'] + step['judge_tokens']:,} tokens"
            )
            if step["judge_issues"]:
                for issue in step["judge_issues"][:3]:
                    print(f"    issue: {issue[:80]}")

        print(f"\n  Total cost (classify + judge): ${total_cost:.4f}")

        print("\nAccepted Classification:")
        r = output["result"]
        print(f"  Document Type:  {r['document_type']}")
        print(f"  Primary Issue:  {r['primary_issue']}")
        print(f"  Severity:       {r['severity'].upper()}")
        print(f"  Reasoning:      {r['reasoning'][:200]}")

    return output


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Classify a legal document with judge-verified cascade routing"
    )
    parser.add_argument("--file", type=Path, default=None)
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--threshold", type=float, default=0.8, help="Judge quality threshold")
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "google", "cross"],
        default=None,
        help="LLM provider, or 'cross' for cross-provider cascade",
    )
    args = parser.parse_args()

    asyncio.run(
        run(
            doc_path=args.file,
            json_output=args.json_output,
            quality_threshold=args.threshold,
            provider=args.provider,
        )
    )


if __name__ == "__main__":
    main()
