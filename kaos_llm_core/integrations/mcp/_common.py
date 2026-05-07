"""Shared helpers for the integrations.mcp tool wrappers.

Phase 14B: every constant and helper function that the per-tool
modules in this package depend on lives here. The split keeps
each tool module focused on its own KaosTool subclass — imports
from this module are the only cross-tool dependency.

Phase 16.5 Tier 2.6: ``BaseLLMCoreTool`` collapses the per-tool
``metadata``-property + ``execute``-try/except boilerplate into
class-level attributes (``_NAME``, ``_DISPLAY_NAME``, ...) and a
single ``_run`` override, saving ~30% LOC across 22 tool files.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from kaos_core import KaosContext, KaosTool, ToolMetadata, ToolResult
from kaos_core.types.annotations import ToolAnnotations
from kaos_core.types.enums import ToolCapability, ToolCategory
from kaos_core.types.parameters import ParameterSchema

_MODULE = "kaos-llm-core"
_VERSION = "0.1.0"

_LLM_CORE_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=True,
)


class BaseLLMCoreTool(KaosTool):
    """Base class for kaos-llm-core MCP tools.

    Phase 16.5 Tier 2.6: collapses the per-tool boilerplate (metadata
    property + execute try/except wrapper) into class-level attributes
    and a single ``_run`` override. Subclasses declare:

    * ``_NAME``           — MCP tool name (``kaos-llm-core-*``)
    * ``_DISPLAY_NAME``   — human-readable display name
    * ``_DESCRIPTION``    — long-form tool description
    * ``_CATEGORY``       — :class:`ToolCategory`
    * ``_CAPABILITY``     — :class:`ToolCapability`
    * ``_ANNOTATIONS``    — :class:`ToolAnnotations`
    * ``_PARAMETERS``     — list of :class:`ParameterSchema`
    * ``_ERROR_HINT``     — appended to the catch-all error message

    and override ``_run(inputs, context)`` with the actual tool logic.
    The base class fills in ``module_name`` / ``version`` from the
    package constants and wraps unhandled exceptions into a
    consistent ``ToolResult.create_error`` envelope. Tools that need
    finer-grained error messages still call
    ``ToolResult.create_error`` directly inside ``_run``.
    """

    _NAME: ClassVar[str]
    _DISPLAY_NAME: ClassVar[str]
    _DESCRIPTION: ClassVar[str]
    _CATEGORY: ClassVar[ToolCategory]
    _CAPABILITY: ClassVar[ToolCapability]
    _ANNOTATIONS: ClassVar[ToolAnnotations]
    _PARAMETERS: ClassVar[list[ParameterSchema]]
    _ERROR_HINT: ClassVar[str] = ""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name=self._NAME,
            display_name=self._DISPLAY_NAME,
            description=self._DESCRIPTION,
            category=self._CATEGORY,
            capability=self._CAPABILITY,
            module_name=_MODULE,
            version=_VERSION,
            annotations=self._ANNOTATIONS,
            input_schema=list(self._PARAMETERS),
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        try:
            return await self._run(inputs, context)
        except Exception as exc:
            hint = f" {self._ERROR_HINT}" if self._ERROR_HINT else ""
            return ToolResult.create_error(f"{self._DISPLAY_NAME} failed: {exc}.{hint}")

    async def _run(self, inputs: dict[str, Any], context: KaosContext | None = None) -> ToolResult:
        raise NotImplementedError


def _settings_for(context: KaosContext | None) -> Any:
    """Return a :class:`KaosLLMCoreSettings` honoring per-request overrides.

    The MCP per-request override mechanism (``_meta.kaos_config``) was
    introduced in Phase 6 but only the four Phase 6/7 tools threaded it
    through. The 13 earlier tools instantiated bare ``KaosLLMCoreSettings()``
    and silently dropped any per-request configuration the caller passed.
    This helper centralizes the right pattern so every tool gets the
    overrides for free.

    Phase 9c follow-up — closes the cross-cutting review's "Settings
    consistency" critical finding.
    """
    from kaos_llm_core.settings import KaosLLMCoreSettings

    return KaosLLMCoreSettings.from_context(context) if context else KaosLLMCoreSettings()


# ---------------------------------------------------------------------------
# Phase 6: Optimizer extension tools
# ---------------------------------------------------------------------------

_PHASE6_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=False,
    openWorldHint=True,
)


# ---------------------------------------------------------------------------
# Phase 7: Observability and tooling
# ---------------------------------------------------------------------------

# Phase 7 tools are local-mostly. The metric tool may invoke an LLM when the
# selected metric is `llm_judge`, so it is conservatively annotated
# `openWorldHint=True`. The analyze-trial tool is fully local JSONL parsing.
_PHASE7_METRIC_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=False,
    openWorldHint=True,
)

_PHASE7_ANALYZE_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)


_PHASE7_METRIC_NAMES = (
    "exact_match",
    "case_insensitive_match",
    "normalized_match",
    "accuracy",
    "numeric_ratio",
    "llm_judge",
)


def _resolve_metric_by_name(name: str) -> Any:
    """Return a metric callable for a named metric.

    MCP cannot transport Python callables, so optimizer tools accept a
    ``metric_name`` string enum. For full flexibility, use the Python API.

    **Default changed (F2):** the default is now ``normalized_match``, not
    ``exact_match``. The previous default was strict ``==`` which scored 0
    when a model returned ``"Termination"`` instead of the gold ``"termination"``,
    or ``"termination."`` with a trailing period, or wrapped the answer in
    quotes — all extremely common LLM behaviors. ``normalized_match`` strips
    whitespace and trailing punctuation, lowercases, and accepts a substring
    match against the gold so prose-wrapped labels score correctly. Pass
    ``metric_name="exact_match"`` explicitly when bit-exact comparison is
    needed (e.g. structured-output schema validation).
    """
    name = (name or "normalized_match").strip().lower()

    def exact_match(prediction: Any, gold: dict[str, Any]) -> float:
        pred = str(getattr(prediction, "answer", "")).strip()
        goldv = str(gold.get("answer", "")).strip()
        return 1.0 if pred == goldv else 0.0

    def case_insensitive_match(prediction: Any, gold: dict[str, Any]) -> float:
        pred = str(getattr(prediction, "answer", "")).strip().lower()
        goldv = str(gold.get("answer", "")).strip().lower()
        return 1.0 if pred == goldv else 0.0

    def normalized_match(prediction: Any, gold: dict[str, Any]) -> float:
        """Lenient comparison for free-text label tasks.

        Lowercases, strips whitespace, strips trailing punctuation
        (``.,;:!?"'``), and accepts a substring match: if the normalized
        gold appears anywhere in the normalized prediction, score 1.0.
        Picks up the F2-fix scenarios: ``"Termination"`` vs ``"termination"``,
        ``"termination."`` vs ``"termination"``, ``"the answer is termination"``
        vs ``"termination"``.
        """
        import re

        def _norm(s: str) -> str:
            s = s.strip().lower()
            # Strip surrounding quotes and trailing punctuation.
            s = s.strip("\"'`")
            s = re.sub(r"[.,;:!?\s]+$", "", s)
            return s

        pred = _norm(str(getattr(prediction, "answer", "")))
        goldv = _norm(str(gold.get("answer", "")))
        if not goldv:
            return 0.0
        if pred == goldv:
            return 1.0
        # Substring containment for prose-wrapped labels.
        return 1.0 if goldv in pred else 0.0

    def length_ratio(prediction: Any, gold: dict[str, Any]) -> float:
        pred = str(getattr(prediction, "answer", ""))
        goldv = str(gold.get("answer", ""))
        if not goldv:
            return 0.0
        ratio = min(len(pred), len(goldv)) / max(len(pred), len(goldv))
        return float(ratio)

    metrics = {
        "exact_match": exact_match,
        "case_insensitive_match": case_insensitive_match,
        "normalized_match": normalized_match,
        "length_ratio": length_ratio,
    }
    if name not in metrics:
        return None
    return metrics[name]


def _build_eval_dataset(examples_raw: Any) -> tuple[Any, Any] | ToolResult:
    """Parse an examples list into (dataset, signature) or a ToolResult error."""
    from pydantic import create_model

    from kaos_llm_core import InputField, OutputField, Signature
    from kaos_llm_core.types import Example

    if isinstance(examples_raw, str):
        try:
            examples_raw = json.loads(examples_raw)
        except json.JSONDecodeError as jde:
            return ToolResult.create_error(
                f"examples is not valid JSON: {jde}. "
                "Provide a list of {input, expected_output} objects."
            )

    if not examples_raw or not isinstance(examples_raw, list):
        return ToolResult.create_error(
            "examples must be a non-empty list of {input, expected_output} objects. "
            'Example: [{"input": "text", "expected_output": "label"}].'
        )

    sig = create_model(
        "DynamicPhase6Signature",
        __base__=Signature,
        __doc__="Answer based on the input.",
        text=(str, InputField(description="Input text")),
        answer=(str, OutputField(description="The answer")),
    )

    dataset: list[Example] = []
    for i, ex in enumerate(examples_raw):
        if not isinstance(ex, dict):
            return ToolResult.create_error(
                f"Each example must be a JSON object with 'input' and "
                f"'expected_output' keys; got {type(ex).__name__} at index {i}. "
                "Pass a list of objects like "
                "[{'input': '...', 'expected_output': '...'}]."
            )
        # Materialize an iteration of (key, value) pairs so ty re-types
        # the result as dict[str, Any] instead of dict[Unknown, Unknown].
        ex_dict: dict[str, Any] = {str(k): v for k, v in ex.items()}
        if "input" not in ex_dict or "expected_output" not in ex_dict:
            return ToolResult.create_error(
                "Each example must have 'input' and 'expected_output' keys. "
                f"Got keys: {list(ex_dict.keys())}."
            )
        dataset.append(
            Example(
                inputs={"text": ex_dict["input"]},
                outputs={"answer": ex_dict["expected_output"]},
            )
        )
    return sig, dataset
