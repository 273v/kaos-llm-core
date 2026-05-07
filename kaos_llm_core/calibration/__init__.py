"""Shared calibration harness primitives.

Extracted from ``scripts/grounding_calibration.py`` (WS-1.6) to let the
WS-3.7 ``scripts/multiformat_benchmark.py`` and any future calibration
script reuse the same dataclasses, per-(model, question) runner, artifact
writers, and CLI shape — instead of forking a ~600-LOC script each time.

Public API is the three layers:

- **Dataclasses.** :class:`Question`, :class:`QuestionResult`,
  :class:`ModelReport`, :class:`CalibrationReport`.
- **Runtime.** :func:`run_question`, :func:`run_model`, :func:`dry_run_report`.
- **Artifacts + CLI.** :func:`load_questions_jsonl`, :func:`write_json`,
  :func:`write_markdown`, :func:`build_parser`, :func:`git_commit`,
  :func:`DEFAULT_MODELS`, :func:`DEFAULT_LENIENT_STRATEGIES`,
  :func:`has_api_key_for`.

The module is **corpus-agnostic**. It accepts whatever ``documents``
shape :class:`kaos_llm_core.programs.rag.RAG.invoke` accepts (dict, list,
Corpus Protocol instance) and records per-question outcomes + aggregates
per-model. The calling script is responsible for loading the corpus,
authoring the question set, and emitting the final artifact.
"""

from __future__ import annotations

from kaos_llm_core.calibration.core import (
    CalibrationReport,
    ModelReport,
    Question,
    QuestionResult,
    dry_run_report,
    run_model,
    run_question,
)
from kaos_llm_core.calibration.io import (
    build_parser,
    git_commit,
    load_questions_jsonl,
    write_json,
    write_markdown,
)
from kaos_llm_core.calibration.providers import (
    DEFAULT_MODELS,
    has_api_key_for,
    provider_of,
)
from kaos_llm_core.calibration.strategies import DEFAULT_LENIENT_STRATEGIES

__all__ = [
    "DEFAULT_LENIENT_STRATEGIES",
    "DEFAULT_MODELS",
    "CalibrationReport",
    "ModelReport",
    "Question",
    "QuestionResult",
    "build_parser",
    "dry_run_report",
    "git_commit",
    "has_api_key_for",
    "load_questions_jsonl",
    "provider_of",
    "run_model",
    "run_question",
    "write_json",
    "write_markdown",
]
