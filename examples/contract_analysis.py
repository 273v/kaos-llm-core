"""Contract Analysis — multi-step Program that extracts clauses, classifies risk, summarizes.

Demonstrates: Signature composition, Program with 3 Calls, model tiering
(cheap for classification, balanced for extraction), ExecutionTrace with cost report.

Usage:
    uv run python -m examples.contract_analysis
    [--provider anthropic|openai|google] [--file PATH] [--json]
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from examples.models import get_preset
from kaos_llm_core import (
    Call,
    InputField,
    OutputField,
    Program,
    Signature,
)
from kaos_llm_core.observability.cost import apply_cost_estimates, format_cost_report

FIXTURES_DIR = Path(__file__).parent / "fixtures"
DEFAULT_CONTRACT = FIXTURES_DIR / "employment_contract.txt"


# --- Domain Models ---


class Clause(BaseModel):
    """A contract clause extracted from the document."""

    name: str
    section: str
    summary: str
    risk_level: str  # low, medium, high


# --- Signatures ---


class ExtractClauses(Signature):
    """Extract the key clauses from an employment contract.

    Identify each major clause, its section number, a brief summary,
    and an initial risk assessment (low/medium/high) from the employee's perspective.
    """

    contract_text: str = InputField(description="Full text of the employment contract")
    clauses: list[Clause] = OutputField(
        description="Extracted clauses with risk assessment from employee perspective"
    )


class ClassifyOverallRisk(Signature):
    """Classify the overall risk level of an employment contract from the employee's perspective.

    Consider factors like: non-compete scope, IP assignment breadth,
    termination provisions, compensation structure.
    """

    contract_text: str = InputField(description="Full text of the employment contract")
    clauses_summary: str = InputField(description="Summary of extracted clauses")
    overall_risk: str = OutputField(description="Overall risk level: low, medium, or high")
    risk_factors: list[str] = OutputField(description="List of specific risk factors identified")


class GenerateSummary(Signature):
    """Generate a concise executive summary of the contract analysis.

    Write 2-3 paragraphs covering: parties involved, key terms,
    notable risk factors, and recommended actions.
    """

    contract_text: str = InputField(description="Full text of the employment contract")
    clauses_json: str = InputField(description="JSON of extracted clauses")
    overall_risk: str = InputField(description="Overall risk classification")
    risk_factors_json: str = InputField(description="JSON of risk factors")
    summary: str = OutputField(description="2-3 paragraph executive summary")


# --- Program ---


class ContractAnalyzer(Program):
    """Multi-step contract analysis pipeline.

    1. Extract clauses (balanced model — needs accuracy)
    2. Classify overall risk (cheap model — simpler task)
    3. Generate executive summary (balanced model — needs quality writing)
    """

    def __init__(self, cheap_model: str, balanced_model: str) -> None:
        self.extract = Call(ExtractClauses, model=balanced_model)
        self.classify = Call(ClassifyOverallRisk, model=cheap_model)
        self.summarize = Call(GenerateSummary, model=balanced_model)

    async def forward(self, **kwargs: Any) -> dict[str, Any]:
        contract_text = kwargs["contract_text"]

        # Step 1: Extract clauses
        extraction = await self.extract(contract_text=contract_text)

        # Step 2: Classify risk (needs the clauses summary)
        clauses_summary = "\n".join(
            f"- {c.name}: {c.summary} (risk: {c.risk_level})" for c in extraction.clauses
        )
        classification = await self.classify(
            contract_text=contract_text,
            clauses_summary=clauses_summary,
        )

        # Step 3: Generate summary
        summary_result = await self.summarize(
            contract_text=contract_text,
            clauses_json=json.dumps([c.model_dump() for c in extraction.clauses], indent=2),
            overall_risk=classification.overall_risk,
            risk_factors_json=json.dumps(classification.risk_factors),
        )

        return {
            "clauses": [c.model_dump() for c in extraction.clauses],
            "overall_risk": classification.overall_risk,
            "risk_factors": classification.risk_factors,
            "summary": summary_result.summary,
        }


async def run(
    contract_path: Path | None = None,
    json_output: bool = False,
    provider: str | None = None,
) -> dict[str, Any]:
    """Run the contract analysis pipeline."""
    preset = get_preset(provider)
    path = contract_path or DEFAULT_CONTRACT
    contract_text = path.read_text(encoding="utf-8")

    analyzer = ContractAnalyzer(cheap_model=preset.cheap, balanced_model=preset.balanced)
    invocation = await analyzer.invoke(contract_text=contract_text)
    result = invocation.output
    trace = invocation.trace
    if trace:
        apply_cost_estimates(trace)

    if json_output:
        output: dict[str, Any] = {
            "file": str(path),
            "provider": provider or "anthropic",
            "result": result,
        }
        if trace:
            output["trace"] = trace.to_dict()
            output["cost_report"] = format_cost_report(trace)
        print(json.dumps(output, indent=2, default=str))
    else:
        print(f"Contract Analysis: {path.name} (provider: {provider or 'anthropic'})")
        print(f"{'=' * 60}")
        print(f"\nOverall Risk: {result['overall_risk'].upper()}")
        print("\nRisk Factors:")
        for factor in result["risk_factors"]:
            print(f"  - {factor}")
        print(f"\nClauses Found: {len(result['clauses'])}")
        for clause in result["clauses"]:
            risk = clause.get("risk_level", "?")
            print(f"  [{risk.upper():6s}] {clause['name']}: {clause['summary'][:80]}")
        print("\nExecutive Summary:")
        print(result["summary"])
        if trace:
            print(f"\n{'=' * 60}")
            print(format_cost_report(trace))

    return result


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze an employment contract")
    parser.add_argument("--file", type=Path, default=None, help="Path to contract text file")
    parser.add_argument("--json", action="store_true", dest="json_output", help="JSON output")
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "google"],
        default=None,
        help="LLM provider (default: anthropic)",
    )
    args = parser.parse_args()

    asyncio.run(run(contract_path=args.file, json_output=args.json_output, provider=args.provider))


if __name__ == "__main__":
    main()
