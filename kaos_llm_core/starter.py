"""kaos-llm-core starter API — one-liner convenience functions.

Phase 8.1 of the kaos-llm-core roadmap. The starter API is the lowest-
friction surface for casual use: one-shot scripts, exploration, and
"I just want to classify this string" workflows. It sits alongside
``@llm_call`` (the production surface for typed standalone LLM
functions) and ``Call``/``Program`` (the production surface for
optimizable, composable workflows). The three coexist and starter does
NOT deprecate ``@llm_call``.

Example::

    from kaos_llm_core.starter import text, extract, classify, summarize

    # Set KAOS_LLM_CORE_DEFAULT_MODEL once, then just:
    answer = await text("Name one primary color.")

    person = await extract(
        "John is 32 years old.",
        {"name": str, "age": int},
    )

    label = await classify(
        "I love this product!",
        labels=["positive", "negative", "neutral"],
    )

    summary = await summarize(long_document, max_words=50)

Each function returns plain Python values (string, dict, list, or a
Pydantic ``BaseModel`` instance if the caller passes one for
``extract``). There is no ``Prediction`` wrapper. Sync variants
(``text_sync`` etc.) are provided for scripts that don't want to deal
with ``asyncio``.
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from kaos_core.logging import get_logger
from pydantic import BaseModel

from kaos_llm_core.errors import CallError
from kaos_llm_core.integrations.common.signatures import build_signature
from kaos_llm_core.programs.call import Call
from kaos_llm_core.settings import KaosLLMCoreSettings
from kaos_llm_core.signatures.fields import InputField, OutputField

logger = get_logger(__name__)

SummaryStyle = Literal["concise", "detailed", "bullet"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_default_model(
    explicit: str | None,
    *,
    settings: KaosLLMCoreSettings | None = None,
) -> str:
    """Resolve the model id via the standard settings hierarchy.

    Priority: explicit argument → settings.default_model → raise.
    Never reads ``os.environ`` directly — settings is the edge that
    captures env vars.
    """
    if explicit is not None:
        return explicit
    resolved = settings or KaosLLMCoreSettings()
    if resolved.default_model:
        return resolved.default_model
    raise CallError(
        "No model specified. Pass `model=` explicitly or set the "
        "`KAOS_LLM_CORE_DEFAULT_MODEL` environment variable. "
        "Example models (April 2026): "
        "`anthropic:claude-haiku-4-5`, `openai:gpt-5.4-nano`, "
        "`google:gemini-2.5-flash`."
    )


# Phase 14C: ``_make_signature`` deleted. Starter functions now call
# :func:`kaos_llm_core.integrations.common.signatures.build_signature`
# directly.


def _extract_hyperparameters(
    *,
    max_tokens: int | None,
    temperature: float | None,
) -> dict[str, Any]:
    """Build the generation-parameter kwargs dict for a Call."""
    kwargs: dict[str, Any] = {}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature
    return kwargs


# ---------------------------------------------------------------------------
# text()
# ---------------------------------------------------------------------------


async def text(
    prompt: str,
    *,
    model: str | None = None,
    system: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    settings: KaosLLMCoreSettings | None = None,
) -> str:
    """One-shot text generation. The lowest-friction surface.

    Returns the model's text response as a plain string.

    Args:
        prompt: The user prompt.
        model: Model identifier (e.g., ``"anthropic:claude-haiku-4-5"``).
            Defaults to ``KaosLLMCoreSettings().default_model``.
        system: Optional system instruction. Defaults to a generic
            "You are a helpful assistant." line.
        max_tokens: Optional generation cap.
        temperature: Optional sampling temperature.
        settings: Optional pre-built settings instance. If omitted, a
            fresh ``KaosLLMCoreSettings()`` is constructed (which reads
            env vars).

    Raises:
        ValueError: When no model is specified and none is configured.
    """
    resolved_model = _resolve_default_model(model, settings=settings)
    instruction = system or "You are a helpful assistant. Respond concisely."

    fields = {
        "prompt": (str, InputField(description="The user prompt")),
        "response": (str, OutputField(description="The assistant response")),
    }
    sig_cls = build_signature("StarterText", fields, instruction)

    # Phase 9e: thread the same settings instance into the underlying Call
    # so the model resolution and the runtime settings stay in lockstep.
    # Previously _resolve_default_model used `settings` for the default
    # model lookup but the Call ignored it and instantiated its own
    # KaosLLMCoreSettings(), so explicit and env values could disagree.
    resolved_settings = settings or KaosLLMCoreSettings()
    call = Call(
        sig_cls,
        model=resolved_model,
        instructions=instruction,
        core_settings=resolved_settings,
        **_extract_hyperparameters(max_tokens=max_tokens, temperature=temperature),
    )
    result = await call(prompt=prompt)
    return str(getattr(result, "response", ""))


def text_sync(
    prompt: str,
    *,
    model: str | None = None,
    system: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    settings: KaosLLMCoreSettings | None = None,
) -> str:
    """Synchronous wrapper around :func:`text`."""
    return asyncio.run(
        text(
            prompt,
            model=model,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            settings=settings,
        )
    )


# ---------------------------------------------------------------------------
# extract()
# ---------------------------------------------------------------------------


async def extract(
    text: str,
    fields: dict[str, type] | type[BaseModel],
    *,
    model: str | None = None,
    instruction: str | None = None,
    settings: KaosLLMCoreSettings | None = None,
) -> dict[str, Any] | BaseModel:
    """Extract typed fields from a string.

    ``fields`` is either a dict mapping field names to Python types
    (``str``, ``int``, ``float``, ``bool``, ``list[str]``,
    ``dict[str, Any]``, etc.) or a Pydantic ``BaseModel`` subclass.

    Returns:
        A ``dict`` when ``fields`` is a dict; an instance of the model
        when ``fields`` is a ``BaseModel`` subclass.
    """
    resolved_model = _resolve_default_model(model, settings=settings)
    doc = instruction or "Extract the requested fields from the given text."

    sig_fields: dict[str, tuple[Any, Any]] = {
        "text": (str, InputField(description="Source text")),
    }

    model_cls: type[BaseModel] | None = None
    field_names: list[str] = []
    if isinstance(fields, type) and issubclass(fields, BaseModel):
        model_cls = fields
        for fname, finfo in model_cls.model_fields.items():
            sig_fields[fname] = (
                finfo.annotation,
                OutputField(description=finfo.description or fname),
            )
            field_names.append(fname)
    else:
        # fields is a dict[str, type]
        fields_dict: dict[str, type] = fields  # type: ignore[assignment]
        for fname, ftype in fields_dict.items():
            sig_fields[fname] = (ftype, OutputField(description=fname))
            field_names.append(fname)

    sig_cls = build_signature("StarterExtract", sig_fields, doc)
    # Phase 9e: thread settings through (see text() for rationale).
    resolved_settings = settings or KaosLLMCoreSettings()
    call = Call(sig_cls, model=resolved_model, instructions=doc, core_settings=resolved_settings)
    result: Any = await call(text=text)

    # Build the return value
    output: dict[str, Any] = {}
    for fname in field_names:
        output[fname] = getattr(result, fname, None)
    if model_cls is not None:
        return model_cls.model_validate(output)
    return output


def extract_sync(
    text: str,
    fields: dict[str, type] | type[BaseModel],
    *,
    model: str | None = None,
    instruction: str | None = None,
    settings: KaosLLMCoreSettings | None = None,
) -> dict[str, Any] | BaseModel:
    """Synchronous wrapper around :func:`extract`."""
    return asyncio.run(
        extract(
            text,
            fields,
            model=model,
            instruction=instruction,
            settings=settings,
        )
    )


# ---------------------------------------------------------------------------
# classify()
# ---------------------------------------------------------------------------


async def classify(
    text: str,
    labels: list[str],
    *,
    model: str | None = None,
    multi_label: bool = False,
    instruction: str | None = None,
    settings: KaosLLMCoreSettings | None = None,
) -> str | list[str]:
    """Classify text into one of ``labels`` (or multiple if ``multi_label``).

    Returns:
        The chosen label string, or a list of label strings when
        ``multi_label=True``.

    Raises:
        CallError: When ``labels`` is empty, or when the model returns a
            label that is not in the provided label set. Phase 9e: was
            ``ValueError`` — caught by ``except KaosLLMCoreError`` now.
    """
    if not labels:
        raise CallError(
            "classify() requires a non-empty `labels` list. Pass at least one candidate label."
        )
    resolved_model = _resolve_default_model(model, settings=settings)

    label_set = ", ".join(f"'{label}'" for label in labels)
    base_doc = instruction or (
        f"Classify the given text into one of these labels: {label_set}. "
        f"Respond with the exact label text."
    )

    sig_fields: dict[str, tuple[Any, Any]] = {
        "text": (str, InputField(description="Text to classify")),
    }
    if multi_label:
        sig_fields["labels"] = (
            list[str],
            OutputField(description=f"Applicable labels (subset of {label_set})"),
        )
    else:
        sig_fields["label"] = (
            str,
            OutputField(description=f"Chosen label (one of {label_set})"),
        )

    sig_cls = build_signature("StarterClassify", sig_fields, base_doc)
    # Phase 9e: thread settings through (see text() for rationale).
    resolved_settings = settings or KaosLLMCoreSettings()
    call = Call(
        sig_cls, model=resolved_model, instructions=base_doc, core_settings=resolved_settings
    )
    result = await call(text=text)

    # Validate outputs against the label set
    valid = set(labels)
    if multi_label:
        chosen = list(getattr(result, "labels", []) or [])
        filtered = [label for label in chosen if label in valid]
        if not filtered:
            raise CallError(
                f"classify(multi_label=True) model returned no labels in the "
                f"allowed set. Allowed: {sorted(valid)}. Got: {chosen}. "
                f"Try a stronger model or tighten the instruction."
            )
        return filtered
    chosen_label = str(getattr(result, "label", "")).strip()
    if chosen_label not in valid:
        # Best-effort recovery: check for case-insensitive match
        lower_map = {label.lower(): label for label in labels}
        if chosen_label.lower() in lower_map:
            return lower_map[chosen_label.lower()]
        raise CallError(
            f"classify() model returned label '{chosen_label}' which is not in "
            f"the allowed set {sorted(valid)}. "
            f"Try a stronger model, tighten the instruction, or add the label "
            f"to the `labels` argument."
        )
    return chosen_label


def classify_sync(
    text: str,
    labels: list[str],
    *,
    model: str | None = None,
    multi_label: bool = False,
    instruction: str | None = None,
    settings: KaosLLMCoreSettings | None = None,
) -> str | list[str]:
    """Synchronous wrapper around :func:`classify`."""
    return asyncio.run(
        classify(
            text,
            labels,
            model=model,
            multi_label=multi_label,
            instruction=instruction,
            settings=settings,
        )
    )


# ---------------------------------------------------------------------------
# summarize()
# ---------------------------------------------------------------------------


async def summarize(
    text: str,
    *,
    model: str | None = None,
    max_words: int | None = None,
    style: SummaryStyle = "concise",
    settings: KaosLLMCoreSettings | None = None,
) -> str:
    """Summarize text. Returns a plain string.

    Args:
        text: The text to summarize.
        model: Model identifier.
        max_words: Optional soft cap on summary length (advisory — the
            model may not honor this exactly).
        style: ``"concise"`` (default), ``"detailed"``, or ``"bullet"``.
    """
    resolved_model = _resolve_default_model(model, settings=settings)

    style_hint = {
        "concise": "Produce a concise prose summary.",
        "detailed": "Produce a detailed prose summary covering key points.",
        "bullet": "Produce a bullet-point summary, one point per line.",
    }[style]

    length_hint = f" Keep the summary under approximately {max_words} words." if max_words else ""
    doc = f"Summarize the given text. {style_hint}{length_hint}"

    sig_fields: dict[str, tuple[Any, Any]] = {
        "text": (str, InputField(description="Text to summarize")),
        "summary": (str, OutputField(description="The summary")),
    }
    sig_cls = build_signature("StarterSummarize", sig_fields, doc)
    # Phase 9e: thread settings through (see text() for rationale).
    resolved_settings = settings or KaosLLMCoreSettings()
    call = Call(sig_cls, model=resolved_model, instructions=doc, core_settings=resolved_settings)
    result = await call(text=text)
    return str(getattr(result, "summary", ""))


def summarize_sync(
    text: str,
    *,
    model: str | None = None,
    max_words: int | None = None,
    style: SummaryStyle = "concise",
    settings: KaosLLMCoreSettings | None = None,
) -> str:
    """Synchronous wrapper around :func:`summarize`."""
    return asyncio.run(
        summarize(
            text,
            model=model,
            max_words=max_words,
            style=style,
            settings=settings,
        )
    )


__all__ = [
    "SummaryStyle",
    "classify",
    "classify_sync",
    "extract",
    "extract_sync",
    "summarize",
    "summarize_sync",
    "text",
    "text_sync",
]
