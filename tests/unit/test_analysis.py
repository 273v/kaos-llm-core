"""Unit tests for kaos_llm_core.optimization.analysis."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from kaos_llm_core.optimization.analysis import (
    StrategyContribution,
    TrialCard,
    load_mutations,
    make_trial_cards,
    strategy_contributions,
    summarize_run,
)
from kaos_llm_core.optimization.mutations import Mutation, MutationLog


def _mut(
    *,
    strategy: str = "bootstrap",
    mutation_type: str = "add_example",
    before_metric: float = 0.5,
    after_metric: float = 0.7,
    accepted: bool = True,
    cost: float = 0.001,
    tokens: int = 100,
    before: dict | None = None,
    after: dict | None = None,
) -> Mutation:
    return Mutation(
        strategy=strategy,
        mutation_type=mutation_type,
        call_name="DemoCall",
        before=before or {"instructions": "old"},
        after=after or {"instructions": "new"},
        rationale="test",
        metric_before=before_metric,
        metric_after=after_metric,
        accepted=accepted,
        tokens_used=tokens,
        cost_usd=cost,
    )


class TestLoadMutations:
    def test_round_trip(self, tmp_path: Path) -> None:
        log = MutationLog(path=tmp_path / "log.jsonl")
        log.record(_mut())
        log.record(_mut(strategy="instruction", after_metric=0.9))
        loaded = load_mutations(tmp_path / "log.jsonl")
        assert len(loaded) == 2
        assert loaded[0].strategy == "bootstrap"
        assert loaded[1].strategy == "instruction"

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_mutations(tmp_path / "nope.jsonl")

    def test_skips_partial_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "partial.jsonl"
        log = MutationLog(path=path)
        log.record(_mut())
        with path.open("a", encoding="utf-8") as f:
            f.write("\n")  # blank line
            f.write("{not_json")  # partially-written
        loaded = load_mutations(path)
        assert len(loaded) == 1

    def test_valid_invalid_valid_logs_warning_and_keeps_valid(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Bug 10 regression: a non-final invalid line must NOT be silently
        dropped. The previous implementation caught every exception on every
        line; a ``valid-invalid-valid`` JSONL loaded as 2 records with no
        warning, making trial summaries silently inaccurate.

        Expected behavior: load the 2 valid lines, drop the 1 invalid line,
        and emit a WARNING log naming the line number.
        """
        import logging

        path = tmp_path / "valid_invalid_valid.jsonl"
        log = MutationLog(path=path)
        log.record(_mut(strategy="bootstrap"))
        log.record(_mut(strategy="instruction"))
        # Surgically insert a broken line BETWEEN the two valid records.
        text = path.read_text(encoding="utf-8").splitlines(keepends=True)
        assert len(text) == 2
        text.insert(1, "this-is-not-json\n")
        path.write_text("".join(text), encoding="utf-8")

        # The kaos.* logger hierarchy uses get_logger() with its own handler
        # and propagate=False so caplog (which listens on root) doesn't see
        # the records. Capture by attaching a handler directly to the target
        # logger for the duration of the test.
        target_logger = logging.getLogger("kaos.llm_core.optimization.analysis")
        captured: list[logging.LogRecord] = []

        class _CaptureHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                captured.append(record)

        handler = _CaptureHandler(level=logging.WARNING)
        target_logger.addHandler(handler)
        try:
            loaded = load_mutations(path)
        finally:
            target_logger.removeHandler(handler)

        # Both valid records survive.
        assert len(loaded) == 2
        assert loaded[0].strategy == "bootstrap"
        assert loaded[1].strategy == "instruction"
        # The non-final broken line produced a WARNING that names the line
        # number and the file. The previous code emitted nothing.
        warnings = [r.getMessage() for r in captured if r.levelname == "WARNING"]
        assert any("line 2" in w and "non-final" in w for w in warnings), (
            f"Expected a WARNING for non-final line 2, got: {warnings}"
        )

    def test_partial_final_line_stays_silent(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Counterpart to the valid-invalid-valid test: a broken FINAL line is
        the partial-write tolerance path and must NOT emit a warning.
        """
        import logging

        path = tmp_path / "partial_tail.jsonl"
        log = MutationLog(path=path)
        log.record(_mut(strategy="bootstrap"))
        with path.open("a", encoding="utf-8") as f:
            f.write("{partial-write\n")  # final line, broken

        target_logger = logging.getLogger("kaos.llm_core.optimization.analysis")
        captured: list[logging.LogRecord] = []

        class _CaptureHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                captured.append(record)

        handler = _CaptureHandler(level=logging.WARNING)
        target_logger.addHandler(handler)
        try:
            loaded = load_mutations(path)
        finally:
            target_logger.removeHandler(handler)

        assert len(loaded) == 1
        warnings = [r.getMessage() for r in captured if r.levelname == "WARNING"]
        # The partial-tail tolerance path is intentionally silent at WARNING.
        assert not warnings, f"Partial-tail tolerance should not warn; got: {warnings}"


class TestMakeTrialCards:
    def test_renders_cards(self) -> None:
        muts = [_mut(), _mut(strategy="instruction", after_metric=0.9)]
        cards = make_trial_cards(muts)
        assert len(cards) == 2
        assert cards[0].trial_id == 0
        assert cards[1].trial_id == 1
        assert cards[0].improvement == pytest.approx(0.2)
        assert cards[1].improvement == pytest.approx(0.4)

    def test_diff_contains_unified_format(self) -> None:
        muts = [
            _mut(
                before={"instructions": "old text"},
                after={"instructions": "new text"},
            )
        ]
        cards = make_trial_cards(muts)
        assert "---" in cards[0].diff
        assert "+++" in cards[0].diff
        assert "old text" in cards[0].diff
        assert "new text" in cards[0].diff

    def test_render_is_deterministic(self) -> None:
        ts = datetime(2026, 4, 7, 12, 0, 0, tzinfo=UTC)
        card = TrialCard(
            trial_id=0,
            strategy="bootstrap",
            mutation_type="add_example",
            call_name="DemoCall",
            metric_before=0.5,
            metric_after=0.7,
            improvement=0.2,
            accepted=True,
            tokens_used=100,
            cost_usd=0.001,
            timestamp=ts,
            diff="--- before\n+++ after\n",
        )
        rendered = card.render()
        assert "Trial #0 [ACCEPTED]" in rendered
        assert "bootstrap" in rendered
        assert "0.5000 -> 0.7000" in rendered
        assert "+0.2000" in rendered
        # Stable across calls
        assert rendered == card.render()

    def test_rejected_marker(self) -> None:
        muts = [_mut(accepted=False)]
        cards = make_trial_cards(muts)
        assert "rejected" in cards[0].render()


class TestStrategyContributions:
    def test_grouping_and_sort(self) -> None:
        muts = [
            _mut(strategy="bootstrap", before_metric=0.5, after_metric=0.6),
            _mut(strategy="bootstrap", before_metric=0.6, after_metric=0.7),
            _mut(strategy="instruction", before_metric=0.5, after_metric=0.9),
        ]
        contrib = strategy_contributions(muts)
        assert len(contrib) == 2
        # instruction has total improvement 0.4 (highest)
        assert contrib[0].strategy == "instruction"
        assert contrib[0].trials == 1
        assert contrib[0].wins == 1
        assert contrib[0].total_improvement == pytest.approx(0.4)
        assert contrib[1].strategy == "bootstrap"
        assert contrib[1].trials == 2
        assert contrib[1].wins == 2
        assert contrib[1].total_improvement == pytest.approx(0.2)

    def test_empty(self) -> None:
        assert strategy_contributions([]) == []

    def test_negative_improvement_not_counted_as_win(self) -> None:
        muts = [_mut(before_metric=0.7, after_metric=0.5)]
        contrib = strategy_contributions(muts)
        assert contrib[0].wins == 0
        assert contrib[0].total_improvement == 0.0


class TestSummarizeRun:
    def test_empty(self) -> None:
        summary = summarize_run([])
        assert summary["total_trials"] == 0
        assert summary["strategies"] == []

    def test_aggregates(self) -> None:
        muts = [
            _mut(before_metric=0.5, after_metric=0.7, cost=0.001, tokens=100),
            _mut(
                strategy="instruction", before_metric=0.7, after_metric=0.9, cost=0.002, tokens=200
            ),
        ]
        summary = summarize_run(muts)
        assert summary["total_trials"] == 2
        assert summary["accepted"] == 2
        assert summary["wins"] == 2
        assert summary["total_cost_usd"] == pytest.approx(0.003)
        assert summary["total_tokens"] == 300
        assert summary["best_metric_after"] == pytest.approx(0.9)
        assert summary["best_improvement"] == pytest.approx(0.2)
        # Strategies sorted; instruction should come first (same improvement, name tiebreak)
        assert len(summary["strategies"]) == 2
        names = [s["strategy"] for s in summary["strategies"]]
        assert set(names) == {"bootstrap", "instruction"}

    def test_deterministic(self) -> None:
        muts = [_mut(), _mut(strategy="instruction")]
        a = summarize_run(muts)
        b = summarize_run(muts)
        assert json.dumps(a, default=str, sort_keys=True) == json.dumps(
            b, default=str, sort_keys=True
        )


class TestStrategyContributionDataclass:
    def test_fields(self) -> None:
        c = StrategyContribution(
            strategy="x",
            trials=1,
            wins=1,
            total_improvement=0.1,
            average_cost=0.01,
        )
        assert c.strategy == "x"
        assert c.trials == 1
