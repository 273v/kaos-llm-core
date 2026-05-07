"""Tests for the ``kaos-llm-core analyze`` CLI (Phase 7.3 prerequisite).

Verifies that the CLI loads a real mutation log JSONL file, computes a
TrialCard summary, and renders both human-readable and ``--json`` envelopes
per docs/guides/cli-standard.md.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from kaos_llm_core.cli import main
from kaos_llm_core.optimization.mutations import Mutation, MutationLog


def _seed_log(path: Path) -> None:
    """Write a small mutation log fixture using the public MutationLog API."""
    log = MutationLog(path=path)
    log.record(
        Mutation(
            strategy="bootstrap",
            mutation_type="add_examples",
            call_name="DemoCall",
            before={"n_examples": 0},
            after={"n_examples": 4},
            rationale="seeded",
            metric_before=0.5,
            metric_after=0.8,
            tokens_used=120,
            cost_usd=0.001234,
            accepted=True,
            timestamp=datetime.now(UTC),
        )
    )
    log.record(
        Mutation(
            strategy="instruction_tuning",
            mutation_type="change_instructions",
            call_name="DemoCall",
            before={"instructions": "old"},
            after={"instructions": "new and improved"},
            rationale="proposer suggested",
            metric_before=0.8,
            metric_after=0.95,
            tokens_used=240,
            cost_usd=0.004567,
            accepted=True,
            timestamp=datetime.now(UTC),
        )
    )


class TestAnalyzeCLI:
    def test_human_readable_output(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        log_path = tmp_path / "run.jsonl"
        _seed_log(log_path)

        main(["analyze", str(log_path)])

        out = capsys.readouterr().out
        assert f"Mutation log: {log_path}" in out
        assert "total trials:        2" in out
        assert "accepted:            2" in out
        assert "By strategy:" in out
        assert "bootstrap" in out
        assert "instruction_tuning" in out
        assert "ACCEPTED" in out
        # Trial card diffs render the unified diff for the change_instructions
        # mutation, so the new instruction text should appear.
        assert "new and improved" in out

    def test_json_envelope(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        log_path = tmp_path / "run.jsonl"
        _seed_log(log_path)

        main(["analyze", str(log_path), "--json"])

        out = capsys.readouterr().out
        envelope = json.loads(out)

        # Envelope shape per docs/guides/cli-standard.md.
        assert envelope["command"] == "analyze"
        assert envelope["log"] == str(log_path)
        assert "summary" in envelope
        assert "by_strategy" in envelope
        assert "trials" in envelope

        summary = envelope["summary"]
        assert summary["total_trials"] == 2
        assert summary["accepted"] == 2
        # Cost is the sum across both records.
        assert summary["total_cost_usd"] == pytest.approx(0.001234 + 0.004567)

        # Per-strategy contributions are present and well-formed.
        strategies = {s["strategy"] for s in envelope["by_strategy"]}
        assert strategies == {"bootstrap", "instruction_tuning"}

        # Trial cards carry the v2 mutation fields and a diff string.
        trials = envelope["trials"]
        assert len(trials) == 2
        assert trials[0]["strategy"] == "bootstrap"
        assert trials[1]["strategy"] == "instruction_tuning"
        assert "diff" in trials[0]
        assert "metric_after" in trials[0]

    def test_missing_file_exits_nonzero(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        nonexistent = tmp_path / "does_not_exist.jsonl"
        with pytest.raises(SystemExit) as excinfo:
            main(["analyze", str(nonexistent)])
        assert excinfo.value.code != 0
        err = capsys.readouterr().err
        assert "error:" in err
        assert "Mutation log not found" in err

    def test_empty_file_exits_nonzero(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        with pytest.raises(SystemExit) as excinfo:
            main(["analyze", str(empty)])
        assert excinfo.value.code != 0
        err = capsys.readouterr().err
        assert "empty" in err

    def test_limit_truncates_trials(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        log_path = tmp_path / "run.jsonl"
        _seed_log(log_path)
        main(["analyze", str(log_path), "--limit", "1", "--json"])
        envelope = json.loads(capsys.readouterr().out)
        # Only one trial card despite 2 mutations in the log; summary still
        # reflects the full log.
        assert len(envelope["trials"]) == 1
        assert envelope["summary"]["total_trials"] == 2
