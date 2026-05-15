"""Shared fixtures for live-LLM scale tests over real corpora.

These fixtures load the same HF JSONL corpora that
``kaos-nlp-core/tests/scale/`` uses. They are skipped when:

- The fixtures aren't downloaded (no USC / EDGAR / patents JSONL).
- No live-provider API key is set (Anthropic / OpenAI / Google).

Tests in this directory are marked ``live`` and ``slow``; the
default unit gate (``pytest -m "not live and not network and not
slow"``) skips them.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

# ``slow`` is not declared in this repo's strict-markers list, so we
# rely on ``live`` alone — ``pytest -m live`` runs these; the default
# ``-m "not live and not network and not slow"`` filter skips them.
pytestmark = pytest.mark.live


_FIXTURE_FILES = ("usc.jsonl", "edgar_agreements.jsonl", "patents.jsonl")


def _resolve_fixtures_dir() -> Path | None:
    env = os.environ.get("KAOS_NLP_SCALE_FIXTURES")
    if env:
        path = Path(env).expanduser().resolve()
        if all((path / name).exists() for name in _FIXTURE_FILES):
            return path
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "kaos-modules" / "kaos-nlp-core" / "tests" / "fixtures"
        if all((candidate / name).exists() for name in _FIXTURE_FILES):
            return candidate
    return None


@pytest.fixture(scope="session")
def scale_fixtures_dir() -> Path:
    path = _resolve_fixtures_dir()
    if path is None:
        pytest.skip(
            "Scale fixtures not available. Set KAOS_NLP_SCALE_FIXTURES "
            "or run kaos-nlp-core/tests/fixtures/download_hf_fixtures.py."
        )
    return path


def _load_jsonl(path: Path, *, max_records: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            if max_records is not None and index >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Live-provider gating
# ---------------------------------------------------------------------------


_KEY_ENVS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
)


def _has_any_live_key() -> bool:
    return any(os.environ.get(name) for name in _KEY_ENVS)


@pytest.fixture(scope="session")
def live_model() -> str:
    """Pick the cheapest model whose provider key is in the environment.

    Order chosen to keep the live gate cheap:

    1. ``openai:gpt-5.4-nano`` — $0.10/$0.40 per million tokens.
    2. ``google:gemini-2.5-flash`` — $0.15/$0.60.
    3. ``anthropic:claude-haiku-4-5`` — $0.80/$4.00.

    Override via ``KAOS_LLM_CORE_SCALE_MODEL`` env var.
    """
    if not _has_any_live_key():
        pytest.skip("No live-provider API key available; skipping live test.")
    override = os.environ.get("KAOS_LLM_CORE_SCALE_MODEL")
    if override:
        return override
    if os.environ.get("OPENAI_API_KEY"):
        return "openai:gpt-5.4-nano"
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        return "google:gemini-2.5-flash"
    return "anthropic:claude-haiku-4-5"


# ---------------------------------------------------------------------------
# Sampled corpora — small by default so live-test cost stays under control.
# ---------------------------------------------------------------------------


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


USC_SUMMARIZE_SAMPLE = _int_env("KAOS_LLM_CORE_SCALE_USC_SUMMARIZE", 3)
USC_CLASSIFY_SAMPLE = _int_env("KAOS_LLM_CORE_SCALE_USC_CLASSIFY", 12)
EDGAR_SAMPLE = _int_env("KAOS_LLM_CORE_SCALE_EDGAR_SAMPLE", 2)


def _record_text(record: dict[str, Any]) -> str:
    return str(record.get("text") or record.get("content") or record.get("body") or "")


_USC_TITLE_PATTERN = re.compile(r"usc/\d+/\d+/(\d+)_")


def _parse_title(identifier: str) -> int | None:
    match = _USC_TITLE_PATTERN.search(identifier)
    return int(match.group(1)) if match else None


@pytest.fixture(scope="session")
def usc_summarize_sample(scale_fixtures_dir: Path) -> list[dict[str, Any]]:
    """A small (~3-doc) sample of medium-length USC sections.

    Selected from the middle of the file to skip table-of-contents
    style records at the head.
    """
    records = _load_jsonl(
        scale_fixtures_dir / "usc.jsonl",
        max_records=10_000,
    )
    # Pick sections in a 1500-3000 char window to keep cost down.
    eligible = [r for r in records if 1500 <= len(_record_text(r)) <= 3000]
    return eligible[100 : 100 + USC_SUMMARIZE_SAMPLE]


@pytest.fixture(scope="session")
def usc_classify_sample(scale_fixtures_dir: Path) -> list[dict[str, Any]]:
    """A balanced 3-class USC classification sample.

    Pulls equal numbers of records from titles 10 (Armed Forces),
    18 (Crimes), and 26 (Internal Revenue Code) — three well-separated
    domains where a zero-shot LLM classifier should get most calls
    right.
    """
    target_titles = (10, 18, 26)
    per_title = max(2, USC_CLASSIFY_SAMPLE // len(target_titles))
    buckets: dict[int, list[dict[str, Any]]] = {t: [] for t in target_titles}
    # Stream the file to find balanced examples without loading the
    # whole 321 MB JSONL into memory.
    with (scale_fixtures_dir / "usc.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            title = _parse_title(str(record.get("identifier", "")))
            if title not in target_titles:
                continue
            text = _record_text(record)
            if not (1000 <= len(text) <= 4000):
                continue
            if len(buckets[title]) < per_title:
                record["_label_title"] = title
                buckets[title].append(record)
            if all(len(buckets[t]) >= per_title for t in target_titles):
                break
    sample: list[dict[str, Any]] = []
    for t in target_titles:
        sample.extend(buckets[t])
    return sample


@pytest.fixture(scope="session")
def edgar_long_sample(scale_fixtures_dir: Path) -> list[dict[str, Any]]:
    """A small sample of EDGAR agreements (avg ~13k words each)."""
    return _load_jsonl(
        scale_fixtures_dir / "edgar_agreements.jsonl",
        max_records=EDGAR_SAMPLE,
    )


record_text = _record_text
