"""Tests for the §7.2 ``summarize`` / ``classify`` CLI subcommands.

The starter façade is the integration surface that already has
unit-test coverage in ``test_starter_doc_facade.py``; this module
covers the CLI shell:

- arg parsing (``--labels``, ``--strategy``, ``--budget-*``, ``--pretty``)
- file / stdin reading
- labels file loading (list-of-strings vs LabelSet model_dump)
- JSON-vs-pretty output

The CLI handler is invoked in-process via :func:`kaos_llm_core.cli.main`
with a patched ``summarize_doc`` / ``classify_doc`` so no LLM call
fires.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from kaos_llm_core import cli, starter
from kaos_llm_core.labels import Label, LabelSet
from kaos_llm_core.results import Classification, Summary


@pytest.fixture
def short_doc(tmp_path: Path) -> Path:
    p = tmp_path / "doc.txt"
    p.write_text("Hello world. This is a short doc.", encoding="utf-8")
    return p


@pytest.fixture
def labels_file(tmp_path: Path) -> Path:
    p = tmp_path / "labels.json"
    p.write_text(json.dumps(["positive", "negative"]), encoding="utf-8")
    return p


@pytest.fixture
def labelset_file(tmp_path: Path) -> Path:
    p = tmp_path / "labelset.json"
    label_set = LabelSet(
        labels=[Label(name="A", description="ay"), Label(name="B", description="bee")],
        exclusive=True,
    )
    p.write_text(json.dumps(label_set.model_dump()), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


class TestSummarizeCLI:
    def test_summarize_pretty_prints_text(
        self,
        short_doc: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        async def _stub(doc: str, **kwargs: Any) -> Summary[str]:
            return Summary[str](
                text="stub summary",
                method="abstractive",
                metadata={"starter.long_strategy": "single"},
            )

        monkeypatch.setattr(starter, "summarize_doc", _stub)
        # The CLI imports summarize_doc at call time inside the handler,
        # so patching the starter module is enough.
        cli.main(["summarize", str(short_doc), "--pretty"])
        captured = capsys.readouterr()
        assert "stub summary" in captured.out

    def test_summarize_json_output(
        self,
        short_doc: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        async def _stub(doc: str, **kwargs: Any) -> Summary[str]:
            return Summary[str](
                text="json summary",
                method="abstractive",
                metadata={"starter.long_strategy": "single"},
            )

        monkeypatch.setattr(starter, "summarize_doc", _stub)
        cli.main(["summarize", str(short_doc)])
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert payload["text"] == "json summary"
        assert payload["method"] == "abstractive"

    def test_summarize_strategy_flag_forwarded(
        self,
        short_doc: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        seen: dict[str, Any] = {}

        async def _stub(doc: str, **kwargs: Any) -> Summary[str]:
            seen.update(kwargs)
            return Summary[str](text="", method="abstractive")

        monkeypatch.setattr(starter, "summarize_doc", _stub)
        cli.main(["summarize", str(short_doc), "--strategy", "tree"])
        assert seen["long_strategy"] == "tree"

    def test_summarize_budget_flags(
        self,
        short_doc: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        seen: dict[str, Any] = {}

        async def _stub(doc: str, **kwargs: Any) -> Summary[str]:
            seen.update(kwargs)
            return Summary[str](text="", method="abstractive")

        monkeypatch.setattr(starter, "summarize_doc", _stub)
        cli.main(
            [
                "summarize",
                str(short_doc),
                "--budget-tokens",
                "500",
                "--budget-usd",
                "0.10",
            ]
        )
        budget = seen["budget"]
        assert budget is not None
        assert budget.max_tokens == 500
        assert budget.max_cost_usd == 0.10


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------


class TestClassifyCLI:
    def test_classify_reads_label_list(
        self,
        short_doc: Path,
        labels_file: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        captured_labels: list[LabelSet] = []

        async def _stub(doc: str, labels: LabelSet, **kwargs: Any) -> Classification:
            captured_labels.append(labels)
            return Classification(
                labels=[Label(name="positive")],
                scores={"positive": 0.9, "negative": 0.1},
                metadata={"starter.long_strategy": "single"},
            )

        monkeypatch.setattr(starter, "classify_doc", _stub)
        cli.main(
            [
                "classify",
                str(short_doc),
                "--labels",
                str(labels_file),
                "--pretty",
            ]
        )
        # The labels file held ``["positive", "negative"]`` — the CLI
        # built a ``LabelSet.from_names`` and forwarded it.
        assert len(captured_labels) == 1
        assert captured_labels[0].names == ("positive", "negative")
        out = capsys.readouterr().out
        assert "positive" in out

    def test_classify_reads_full_labelset(
        self,
        short_doc: Path,
        labelset_file: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured_labels: list[LabelSet] = []

        async def _stub(doc: str, labels: LabelSet, **kwargs: Any) -> Classification:
            captured_labels.append(labels)
            return Classification(labels=[Label(name="A")], scores={"A": 1.0, "B": 0.0})

        monkeypatch.setattr(starter, "classify_doc", _stub)
        cli.main(
            [
                "classify",
                str(short_doc),
                "--labels",
                str(labelset_file),
            ]
        )
        assert captured_labels[0].names == ("A", "B")
        # Description came through from the serialized LabelSet, not
        # the bare-names path.
        assert captured_labels[0].by_name("A").description == "ay"

    def test_classify_strategy_flag(
        self,
        short_doc: Path,
        labels_file: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        seen: dict[str, Any] = {}

        async def _stub(doc: str, labels: LabelSet, **kwargs: Any) -> Classification:
            seen.update(kwargs)
            return Classification(labels=[Label(name="positive")])

        monkeypatch.setattr(starter, "classify_doc", _stub)
        cli.main(
            [
                "classify",
                str(short_doc),
                "--labels",
                str(labels_file),
                "--strategy",
                "chunk",
            ]
        )
        assert seen["long_strategy"] == "chunk"
