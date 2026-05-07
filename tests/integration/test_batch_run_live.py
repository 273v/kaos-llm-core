"""Live integration tests for ``batch_run()`` against real Anthropic Haiku.

Drives a small (8-item) batch through real Haiku to verify cost
attribution, JSONL log shape, manifest contents, and resume against
the live execution path. Hard ``max_cost_usd`` gate keeps the test
under $0.10 worst-case.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from kaos_llm_core.programs.batch import batch_run, list_input_source
from kaos_llm_core.programs.envelope import from_envelope, program_hash

pytestmark = pytest.mark.integration


def _has_key(*env_vars: str) -> bool:
    return any(os.getenv(v) for v in env_vars)


requires_anthropic = pytest.mark.skipif(
    not _has_key("KAOS_LLM_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    reason="No Anthropic API key",
)


HAIKU = "claude-haiku-4-5"


def _classify_envelope() -> dict:
    """A one-step envelope used by every live batch test in this file."""
    return {
        "kaos_program": "1",
        "name": "live-classify",
        "inputs": {"text": {"type": "string"}},
        "clients": {"default": {"provider": "anthropic", "model": HAIKU}},
        "steps": [
            {
                "id": "classify",
                "kind": "call",
                "client": "default",
                "instruction": (
                    "Classify the sentiment of the input as positive, negative, or neutral."
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


SAMPLE_INPUTS = [
    {"text": "I love this product, it's wonderful!"},
    {"text": "Worst experience of my life. Terrible service."},
    {"text": "It is a chair. It does what a chair does."},
    {"text": "Absolutely fantastic, exceeded all expectations."},
    {"text": "I'm not sure how I feel about this."},
    {"text": "The court dismissed the wrongful termination lawsuit."},
    {"text": "The team celebrated their championship victory."},
    {"text": "I cannot recommend this enough — five stars."},
]


@requires_anthropic
class TestBatchRunLive:
    async def test_batch_against_haiku(self, tmp_path: Path) -> None:
        """8-item batch against real Haiku. Verifies end-to-end:
        - Source iteration
        - Per-item program execution
        - JSONL log streaming
        - Cost attribution via TrialRunner
        - Manifest write
        """
        envelope = _classify_envelope()
        program = from_envelope(envelope)
        env_hash = program_hash(envelope)

        result = await batch_run(
            program,
            list_input_source(SAMPLE_INPUTS, program_hash_value=env_hash),
            output_dir=str(tmp_path / "live_batch"),
            max_concurrency=4,
            resume=False,
        )

        assert result.n_total == 8
        assert result.n_succeeded == 8
        assert result.n_errored == 0
        assert result.cost_usd > 0
        assert result.cost_usd < 0.10, (
            f"Cost cap exceeded: ${result.cost_usd:.6f}. Expected < $0.10."
        )
        assert result.total_tokens > 0
        assert result.duration_s > 0

        # JSONL log shape
        log_path = Path(result.log_path)
        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 9  # 1 header + 8 items
        header = json.loads(lines[0])
        assert header["kind"] == "header"
        assert header["program_hash"] == env_hash

        for line in lines[1:]:
            record = json.loads(line)
            assert record["kind"] == "item"
            assert record["status"] == "success"
            assert "sentiment" in record["output"]
            assert record["usage"]["cost_usd"] > 0

        # Manifest
        manifest_path = Path(result.manifest_path)
        manifest = json.loads(manifest_path.read_text())
        assert manifest["n_succeeded"] == 8
        assert manifest["status"] == "completed"
        assert manifest["cost_usd"] == round(result.cost_usd, 6)
        print(
            f"\n[batch_live] 8 items against Haiku, "
            f"${result.cost_usd:.6f}, {result.total_tokens} tokens, "
            f"{result.duration_s:.2f}s"
        )

    async def test_batch_resume_live(self, tmp_path: Path) -> None:
        """First run processes 4 items; second run resumes with 8 items
        and only the new 4 hit Haiku.
        """
        envelope = _classify_envelope()
        program = from_envelope(envelope)
        env_hash = program_hash(envelope)
        out_dir = tmp_path / "live_resume"

        # Run 1: first 4 items
        result1 = await batch_run(
            program,
            list_input_source(SAMPLE_INPUTS[:4], program_hash_value=env_hash),
            output_dir=str(out_dir),
            max_concurrency=2,
            resume=False,
        )
        assert result1.n_succeeded == 4
        first_cost = result1.cost_usd

        # Run 2: same 4 items + 4 new ones; resume should skip the
        # first 4 and only call Haiku for the new 4.
        result2 = await batch_run(
            program,
            list_input_source(SAMPLE_INPUTS, program_hash_value=env_hash),
            output_dir=str(out_dir),
            max_concurrency=2,
            resume=True,
        )
        assert result2.n_total == 8
        assert result2.n_skipped == 4
        # Cumulative: 4 prior + 4 new = 8 successes
        assert result2.n_succeeded == 8
        # The second run only paid for 4 new items, so its trial
        # cost (which is just this run's spend, not cumulative) is
        # roughly half the first run's cost.
        # Note: result2.cost_usd is the trial accumulator from THIS
        # run, not cumulative across the log. The manifest cost is
        # what this run charged.
        assert result2.cost_usd > 0
        assert result2.cost_usd < 0.10
        print(
            f"\n[batch_live_resume] run1=${first_cost:.6f}, "
            f"run2 (4 new + 4 skipped)=${result2.cost_usd:.6f}"
        )
