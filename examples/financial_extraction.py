"""Financial Extraction — structured metric extraction from earnings transcripts.

Demonstrates: Complex Pydantic output schema, JSONCodec structured output,
nested models, parallel execution, real financial data.

Usage:
    uv run python -m examples.financial_extraction
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
DEFAULT_TRANSCRIPT = FIXTURES_DIR / "earnings_transcript.txt"


# --- Signatures ---


class ExtractFinancials(Signature):
    """Extract key financial metrics from an earnings call transcript.

    Pull out revenue, margins, growth rates, and guidance. All monetary
    values should be in millions USD. Percentages as decimals (e.g., 0.423 for 42.3%).
    """

    transcript: str = InputField(description="Earnings call transcript text")
    company_name: str = OutputField(description="Company name")
    ticker: str = OutputField(description="Stock ticker symbol")
    quarter: str = OutputField(description="Fiscal quarter (e.g., 'Q4 2025')")
    revenue_millions: float = OutputField(description="Quarterly revenue in millions USD")
    revenue_growth_yoy: float = OutputField(
        description="Year-over-year revenue growth as decimal (e.g., 0.18 for 18%)"
    )
    gross_margin: float = OutputField(description="Gross margin as decimal")
    operating_margin: float = OutputField(description="Operating margin as decimal")
    ebitda_margin: float = OutputField(description="Adjusted EBITDA margin as decimal")
    free_cash_flow_millions: float = OutputField(description="Free cash flow in millions USD")


class Risk(BaseModel):
    """A risk or headwind identified in an earnings call."""

    category: str  # market, regulatory, operational, fx
    description: str
    severity: str  # low, medium, high


class ExtractRisks(Signature):
    """Identify risks and headwinds mentioned in an earnings call transcript."""

    transcript: str = InputField(description="Earnings call transcript text")
    risks: list[Risk] = OutputField(description="Identified risks and headwinds")


class ExtractGuidance(Signature):
    """Extract forward-looking guidance from an earnings call transcript."""

    transcript: str = InputField(description="Earnings call transcript text")
    guidance_quarter: str = OutputField(description="Next quarter being guided (e.g., 'Q1 2026')")
    revenue_low_millions: float = OutputField(description="Revenue guidance low end, millions USD")
    revenue_high_millions: float = OutputField(
        description="Revenue guidance high end, millions USD"
    )
    full_year_revenue_low_millions: float = OutputField(
        description="Full year revenue guidance low end, millions USD"
    )
    full_year_revenue_high_millions: float = OutputField(
        description="Full year revenue guidance high end, millions USD"
    )


# --- Program ---


class EarningsAnalyzer(Program):
    """Multi-step earnings transcript analysis.

    Runs three extraction calls in parallel:
    1. Core financial metrics
    2. Risk identification
    3. Forward guidance
    """

    def __init__(self, model: str) -> None:
        self.financials = Call(ExtractFinancials, model=model)
        self.risks = Call(ExtractRisks, model=model)
        self.guidance = Call(ExtractGuidance, model=model)

    async def forward(self, **kwargs: Any) -> dict[str, Any]:
        transcript = kwargs["transcript"]

        # Run all three in parallel
        fin_result, risk_result, guidance_result = await asyncio.gather(
            self.financials(transcript=transcript),
            self.risks(transcript=transcript),
            self.guidance(transcript=transcript),
        )

        return {
            "company": fin_result.company_name,
            "ticker": fin_result.ticker,
            "quarter": fin_result.quarter,
            "financials": {
                "revenue_millions": fin_result.revenue_millions,
                "revenue_growth_yoy": fin_result.revenue_growth_yoy,
                "gross_margin": fin_result.gross_margin,
                "operating_margin": fin_result.operating_margin,
                "ebitda_margin": fin_result.ebitda_margin,
                "free_cash_flow_millions": fin_result.free_cash_flow_millions,
            },
            "risks": [r.model_dump() for r in risk_result.risks],
            "guidance": {
                "next_quarter": guidance_result.guidance_quarter,
                "revenue_range": [
                    guidance_result.revenue_low_millions,
                    guidance_result.revenue_high_millions,
                ],
                "full_year_range": [
                    guidance_result.full_year_revenue_low_millions,
                    guidance_result.full_year_revenue_high_millions,
                ],
            },
        }


async def run(
    transcript_path: Path | None = None,
    json_output: bool = False,
    provider: str | None = None,
) -> dict[str, Any]:
    """Run the earnings analysis pipeline."""
    preset = get_preset(provider)
    path = transcript_path or DEFAULT_TRANSCRIPT
    transcript = path.read_text(encoding="utf-8")

    analyzer = EarningsAnalyzer(model=preset.cheap)
    invocation = await analyzer.invoke(transcript=transcript)
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
        fin = result["financials"]
        print(
            f"Earnings Analysis: {result['company']} ({result['ticker']}) "
            f"— provider: {provider or 'anthropic'}"
        )
        print(f"{'=' * 60}")
        print(f"Quarter: {result['quarter']}")
        print("\nFinancials:")
        print(f"  Revenue:          ${fin['revenue_millions']:,.0f}M")
        print(f"  Revenue Growth:   {fin['revenue_growth_yoy']:.1%} YoY")
        print(f"  Gross Margin:     {fin['gross_margin']:.1%}")
        print(f"  Operating Margin: {fin['operating_margin']:.1%}")
        print(f"  EBITDA Margin:    {fin['ebitda_margin']:.1%}")
        print(f"  Free Cash Flow:   ${fin['free_cash_flow_millions']:,.0f}M")

        print(f"\nRisks ({len(result['risks'])}):")
        for risk in result["risks"]:
            sev = risk.get("severity", "?")
            cat = risk.get("category", "?")
            desc = risk.get("description", "")
            print(f"  [{sev.upper():6s}] ({cat}) {desc[:70]}")

        g = result["guidance"]
        print("\nGuidance:")
        print(
            f"  {g['next_quarter']}: ${g['revenue_range'][0]:,.0f}M "
            f"- ${g['revenue_range'][1]:,.0f}M"
        )
        print(f"  FY2026: ${g['full_year_range'][0]:,.0f}M - ${g['full_year_range'][1]:,.0f}M")

        if trace:
            print(f"\n{'=' * 60}")
            print(format_cost_report(trace))

    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Analyze earnings transcript")
    parser.add_argument("--file", type=Path, default=None)
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "google"],
        default=None,
    )
    args = parser.parse_args()

    asyncio.run(
        run(transcript_path=args.file, json_output=args.json_output, provider=args.provider)
    )


if __name__ == "__main__":
    main()
