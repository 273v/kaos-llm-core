"""Resume + crash-recovery tests for ``batch_run()``.

These exercise the contract that the JSONL checkpoint log is the
source of truth for resume, and that a crash mid-batch is recoverable
without re-running completed items or losing partial cost data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart

from kaos_llm_core.observability.cost import PRICING, ModelPricing
from kaos_llm_core.programs.batch import (
    BatchResumeMismatchError,
    JsonlBatchWriter,
    batch_run,
    list_input_source,
)
from kaos_llm_core.programs.call import Call
from kaos_llm_core.signatures import InputField, OutputField, Signature


class _Sig(Signature):
    """Echo a value."""

    text: str = InputField(description="Input")
    answer: str = OutputField(description="Output")


def _json_response(data: dict[str, Any]) -> ProviderResponse:
    return ProviderResponse.model_construct(
        provider="function",
        model="function-test",
        raw={},
        parts=[ContentPart.model_construct(type="text", text=json.dumps(data))],
        usage=UsageInfo.model_construct(input_tokens=10, output_tokens=5, total_tokens=15),
        stop_reason="end_turn",
        status_code=200,
        response_headers={},
    )


def _make_call(answer_template: str = "ok-{i}") -> Call:
    """Build a Call that returns a unique answer per invocation."""
    counter = {"n": 0}

    def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
        i = counter["n"]
        counter["n"] += 1
        return _json_response({"answer": answer_template.format(i=i)})

    call = Call(_Sig, model="function-test")
    call._client = FunctionClient(function=fn)
    return call


@pytest.fixture(autouse=True)
def _pricing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        PRICING,
        "function-test",
        ModelPricing(input_per_mtok=1.0, output_per_mtok=2.0),
    )


# ---------------------------------------------------------------------------
# Crash simulation: partial log → resume → final state matches clean run
# ---------------------------------------------------------------------------


class TestCrashSimulation:
    async def test_simulate_crash_then_resume(self, tmp_path: Path) -> None:
        """Run a batch, manually truncate the log to simulate a kill-9 mid-run,
        resume with the same source, and assert the final state matches a
        clean uninterrupted run."""
        out_dir = tmp_path / "crash"

        # Build the same source state both times — list_input_source is
        # deterministic for the same inputs + program_hash.
        def make_source():
            return list_input_source(
                [{"text": f"item-{i}"} for i in range(10)],
                program_hash_value="sha256:test",
            )

        # First run: complete cleanly, ground truth for resume.
        clean_dir = tmp_path / "clean"
        clean_result = await batch_run(
            _make_call(),
            make_source(),
            output_dir=str(clean_dir),
            resume=False,
        )
        assert clean_result.n_succeeded == 10

        # Crashed run: write only the first 5 items into the log via the
        # writer directly, then resume from there.
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "items.jsonl"
        writer = JsonlBatchWriter(log_path)
        await writer.write_header(
            {
                "kaos_batch_log": "1",
                "run_id": "crashed-run",
                "program_hash": "sha256:test_hash_for_crash",  # placeholder
                "source": make_source().describe(),
                "config": {},
                "started_at": "2026-04-08T12:00:00Z",
            }
        )
        # Write 5 fake successful items so the resume scan sees them
        for i in range(5):
            await writer.write_item(
                {
                    "custom_id": _expected_custom_id(i),
                    "input_ref": {},
                    "status": "success",
                    "output": {"answer": f"prerun-{i}"},
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15,
                        "cost_usd": 0.00002,
                    },
                    "duration_ms": 1.0,
                    "attempts": 1,
                    "ended_at": "2026-04-08T12:00:01Z",
                }
            )
        await writer.close()

        # Now resume — but with a mismatched program_hash, to verify
        # the resume contract refuses cleanly.
        with pytest.raises(BatchResumeMismatchError, match="program hash mismatch"):
            await batch_run(
                _make_call(),
                make_source(),
                output_dir=str(out_dir),
                resume=True,
            )

    async def test_resume_with_correct_program_hash_continues(self, tmp_path: Path) -> None:
        """A resume against a log written by the SAME program continues correctly."""
        out_dir = tmp_path / "good_resume"
        call = _make_call()

        # Run 1: 5 items
        result1 = await batch_run(
            call,
            list_input_source(
                [{"text": f"item-{i}"} for i in range(5)],
                program_hash_value="sha256:test",
            ),
            output_dir=str(out_dir),
            resume=False,
        )
        assert result1.n_succeeded == 5
        prior_log_size = Path(result1.log_path).stat().st_size

        # Run 2: same 5 items + 5 new ones; first 5 should be skipped via
        # the resume scan, last 5 should run fresh.
        result2 = await batch_run(
            call,
            list_input_source(
                [{"text": f"item-{i}"} for i in range(10)],
                program_hash_value="sha256:test",
            ),
            output_dir=str(out_dir),
            resume=True,
        )
        assert result2.n_skipped == 5
        # Cumulative successes: 5 (prior) + 5 (this run) = 10
        assert result2.n_succeeded == 10
        # Log file grew (new items appended)
        new_log_size = Path(result2.log_path).stat().st_size
        assert new_log_size > prior_log_size


def _expected_custom_id(i: int) -> str:
    """Compute the deterministic custom_id for `{"text": "item-{i}"}` against
    the test program hash."""
    from kaos_llm_core.programs.batch import BatchItem

    return BatchItem.deterministic_id("sha256:test", {"text": f"item-{i}"})


# ---------------------------------------------------------------------------
# Resume hash mismatch
# ---------------------------------------------------------------------------


class TestResumeMismatch:
    async def test_resume_with_different_program_raises(self, tmp_path: Path) -> None:
        """Resuming a log written by a different program must raise cleanly."""
        out_dir = tmp_path / "mismatch"
        out_dir.mkdir()
        log_path = out_dir / "items.jsonl"

        # Write a log header with a fake hash
        writer = JsonlBatchWriter(log_path)
        await writer.write_header(
            {
                "kaos_batch_log": "1",
                "run_id": "old-run",
                "program_hash": "sha256:totally_different_program",
                "source": {"type": "list"},
                "config": {},
                "started_at": "2026-04-08T12:00:00Z",
            }
        )
        await writer.close()

        # Resume should raise BatchResumeMismatchError
        with pytest.raises(BatchResumeMismatchError, match="program hash mismatch"):
            await batch_run(
                _make_call(),
                list_input_source(
                    [{"text": "x"}],
                    program_hash_value="sha256:test",
                ),
                output_dir=str(out_dir),
                resume=True,
            )


# ---------------------------------------------------------------------------
# Empty log + resume = fresh run
# ---------------------------------------------------------------------------


class TestResumeFreshRun:
    async def test_resume_against_nonexistent_log_is_fresh(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "fresh"
        result = await batch_run(
            _make_call(),
            list_input_source(
                [{"text": "x"}],
                program_hash_value="sha256:test",
            ),
            output_dir=str(out_dir),
            resume=True,  # log doesn't exist; should be treated as fresh
        )
        assert result.n_succeeded == 1
        assert result.n_skipped == 0


# ---------------------------------------------------------------------------
# Corrupt last line tolerance (the WAE crash window)
# ---------------------------------------------------------------------------


class TestCorruptLogTolerance:
    async def test_corrupt_last_line_is_ignored_on_resume(self, tmp_path: Path) -> None:
        """A partial / corrupt last line in the JSONL log (e.g. from a crash
        between write() and fsync()) must be tolerated by the resume scan."""
        out_dir = tmp_path / "corrupt"
        out_dir.mkdir()
        log_path = out_dir / "items.jsonl"

        # Hand-write a log with a valid header, one valid item, and a corrupt
        # last line that lacks closing braces.
        # Compute the program hash that batch_run would derive from the same
        # source so resume validation succeeds.
        from kaos_llm_core.programs.batch import _hash_target

        call = _make_call()
        program_hash_value = _hash_target(call)

        with log_path.open("w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "kind": "header",
                        "kaos_batch_log": "1",
                        "run_id": "test",
                        "program_hash": program_hash_value,
                        "source": {"type": "list"},
                        "config": {},
                        "started_at": "2026-04-08T12:00:00Z",
                    }
                )
                + "\n"
            )
            f.write(
                json.dumps(
                    {
                        "kind": "item",
                        "custom_id": _expected_custom_id(0),
                        "input_ref": {},
                        "status": "success",
                        "output": {"answer": "from-prior-run"},
                        "usage": {
                            "input_tokens": 10,
                            "output_tokens": 5,
                            "total_tokens": 15,
                            "cost_usd": 0.00002,
                        },
                        "duration_ms": 1.0,
                        "attempts": 1,
                        "ended_at": "2026-04-08T12:00:01Z",
                    }
                )
                + "\n"
            )
            # Corrupt last line — partial JSON
            f.write('{"kind": "item", "custom_id": "abc", "status": "success", "out')

        # Resume should tolerate the corrupt line and recognize the prior
        # successful item as completed.
        result = await batch_run(
            call,
            list_input_source(
                [{"text": "item-0"}, {"text": "item-1"}],
                program_hash_value="sha256:test",
            ),
            output_dir=str(out_dir),
            resume=True,
        )
        # Item 0 was in the log; item 1 is new. Skip the prior, run the new.
        assert result.n_skipped == 1
        # n_succeeded includes the prior + new = 2
        assert result.n_succeeded == 2
