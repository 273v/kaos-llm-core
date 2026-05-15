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
from typing import TYPE_CHECKING, Any, Literal

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


# ---------------------------------------------------------------------------
# summarize_doc() / classify_doc() — Phase 7 declarative façade
# ---------------------------------------------------------------------------
#
# Plan ``docs/summarization-classification-plan.md`` §7.1. These wrap
# the Layer-4 Programs from ``kaos_llm_core.programs.summarize`` /
# ``classify`` so callers get the §1 pyramid endgame: one function,
# typed input + typed output, smart defaults, no Program-construction
# boilerplate.
#
# The simpler ``summarize`` / ``classify`` above return plain strings
# and remain the one-liner surface for casual exploration. ``summarize_doc``
# / ``classify_doc`` return the full ``Summary[str]`` /
# ``Classification[Label]`` Pydantic objects with provenance, metadata,
# and (when ``cited=True``) verified source spans.


LongSummaryStrategy = Literal["auto", "single", "tree", "refine"]
"""Strategy for handling long documents in :func:`summarize_doc`."""

LongClassifyStrategy = Literal["auto", "single", "chunk"]
"""Strategy for handling long documents in :func:`classify_doc`."""


# Auto-strategy threshold: when the input fits inside ``threshold_chars``
# characters (an embedding-free proxy for "fits in one context window"),
# the auto rule picks the single-shot path. Above the threshold it picks
# the tree (summarise) or chunk (classify) strategy. The threshold is
# deliberately conservative — 12k chars ≈ 3k tokens, well below any
# modern context window — because the single-shot path also has to fit
# the model's input plus a generated summary plus retries, and we'd
# rather over-chunk than blow up at runtime.
_LONG_DOC_CHAR_THRESHOLD: int = 12_000


def _resolve_long_summary_strategy(
    text_len: int,
    requested: LongSummaryStrategy,
) -> LongSummaryStrategy:
    """Resolve ``long_strategy="auto"`` for :func:`summarize_doc`.

    Rule:

    - When ``requested != "auto"``: pass through.
    - When ``text_len <= _LONG_DOC_CHAR_THRESHOLD``: ``"single"``.
    - Otherwise: ``"tree"``.
    """
    if requested != "auto":
        return requested
    if text_len <= _LONG_DOC_CHAR_THRESHOLD:
        return "single"
    return "tree"


def _resolve_long_classify_strategy(
    text_len: int,
    requested: LongClassifyStrategy,
) -> LongClassifyStrategy:
    """Resolve ``long_strategy="auto"`` for :func:`classify_doc`."""
    if requested != "auto":
        return requested
    if text_len <= _LONG_DOC_CHAR_THRESHOLD:
        return "single"
    return "chunk"


async def summarize_doc(
    doc: str,
    *,
    model: str | None = None,
    long_strategy: LongSummaryStrategy = "auto",
    cited: bool = False,
    cache: ChunkCache | None = None,
    budget: Budget | None = None,
    chunker: Chunker | None = None,
    settings: KaosLLMCoreSettings | None = None,
) -> Summary[str] | Summary[Any]:
    """Declarative summarization façade (plan §7.1).

    Returns the full
    :class:`~kaos_llm_core.results.Summary` — not a plain string —
    so callers see ``method``, ``source_spans``, and ``metadata``
    (including ``cache.hits``, ``budget.cost_usd``,
    ``chunks.processed``, and ``budget.exhausted`` when relevant)
    alongside the summary text.

    Args:
        doc: Source text.
        model: Provider model identifier. ``None`` resolves to
            ``settings.default_model`` /
            ``KAOS_LLM_CORE_DEFAULT_MODEL``.
        long_strategy: ``"auto"`` (default) picks
            ``"single"`` when ``len(doc) <= 12_000`` characters and
            ``"tree"`` otherwise.  ``"single"`` forces single-call
            abstractive (optionally cited);  ``"tree"`` forces a
            :class:`~kaos_llm_core.programs.summarize.HierarchicalSummary`;
            ``"refine"`` forces a
            :class:`~kaos_llm_core.programs.summarize.RefineSummary`.
        cited: When ``True``, route through
            :class:`~kaos_llm_core.programs.summarize.CitedSummary`
            for the ``"single"`` strategy. Long-doc strategies emit
            ``method="abstractive"``; per-leaf citation is a Phase 6
            feature (``QueryFocusedSummary`` / ``CitedSummary``
            wrapping reducer levels).
        cache: Optional :class:`~kaos_llm_core.cache.ChunkCache`
            threaded into long-doc Programs.
        budget: Optional :class:`~kaos_llm_core.optimization.budget.Budget`
            threaded into long-doc Programs.
        chunker: Optional explicit chunker. ``None`` uses each long-doc
            Program's default (``ParagraphChunker(max_tokens=1024)``).
        settings: Optional :class:`KaosLLMCoreSettings`.
    """
    from kaos_llm_core.programs.summarize import (
        AbstractiveSummary,
        CitedSummary,
        HierarchicalSummary,
        RefineSummary,
    )

    resolved_model = _resolve_default_model(model, settings=settings)
    resolved_settings = settings or KaosLLMCoreSettings()
    strategy = _resolve_long_summary_strategy(len(doc), long_strategy)

    if strategy == "single":
        program_class: type[Any] = CitedSummary if cited else AbstractiveSummary
        program = program_class(model=resolved_model, core_settings=resolved_settings)
        result: Summary[Any] = await program(text=doc)
        # Tag the strategy + façade so downstream code can tell the
        # Summary came through the declarative path.
        return result.model_copy(
            update={
                "metadata": {
                    **dict(result.metadata),
                    "starter.long_strategy": strategy,
                    "starter.facade": "summarize_doc",
                },
            }
        )

    long_kwargs: dict[str, Any] = {
        "model": resolved_model,
        "core_settings": resolved_settings,
    }
    if chunker is not None:
        long_kwargs["chunker"] = chunker
    if cache is not None:
        long_kwargs["cache"] = cache
    if budget is not None:
        long_kwargs["budget"] = budget

    long_program: HierarchicalSummary | RefineSummary
    if strategy == "tree":
        long_program = HierarchicalSummary(**long_kwargs)
    elif strategy == "refine":
        long_program = RefineSummary(**long_kwargs)
    else:  # pragma: no cover - guarded by Literal
        raise CallError(f"unknown long_strategy: {strategy!r}")

    long_result: Summary[str] = await long_program(text=doc)
    return long_result.model_copy(
        update={
            "metadata": {
                **dict(long_result.metadata),
                "starter.long_strategy": strategy,
                "starter.facade": "summarize_doc",
            },
        }
    )


def summarize_doc_sync(
    doc: str,
    *,
    model: str | None = None,
    long_strategy: LongSummaryStrategy = "auto",
    cited: bool = False,
    cache: ChunkCache | None = None,
    budget: Budget | None = None,
    chunker: Chunker | None = None,
    settings: KaosLLMCoreSettings | None = None,
) -> Summary[str] | Summary[Any]:
    """Synchronous wrapper around :func:`summarize_doc`."""
    return asyncio.run(
        summarize_doc(
            doc,
            model=model,
            long_strategy=long_strategy,
            cited=cited,
            cache=cache,
            budget=budget,
            chunker=chunker,
            settings=settings,
        )
    )


async def classify_doc(
    doc: str,
    labels: LabelSet | Sequence[str],
    *,
    model: str | None = None,
    supervision: Literal["zero_shot", "few_shot", "prototype", "retrieval", "nli"] = "zero_shot",
    examples: Sequence[Any] | None = None,
    embedder: Any = None,
    corpus: Sequence[tuple[str, str]] | None = None,
    nli_scorer: Any = None,
    long_strategy: LongClassifyStrategy = "auto",
    aggregator: Aggregator | None = None,
    cache: ChunkCache | None = None,
    budget: Budget | None = None,
    chunker: Chunker | None = None,
    settings: KaosLLMCoreSettings | None = None,
) -> Classification:
    """Declarative classification façade (plan §7.1).

    Returns the full :class:`~kaos_llm_core.results.Classification`,
    not a string, so callers see ``labels``, ``scores``, ``abstained``,
    and the same ``cache.hits`` / ``budget.*`` /
    ``chunks.processed`` metadata as the long-doc Programs.

    Args:
        doc: Source text.
        labels: Either an existing
            :class:`~kaos_llm_core.labels.LabelSet` (carries
            ``exclusive`` / ``allow_abstain`` / ``hierarchical``
            policy flags) or a bare sequence of label names — the
            façade builds a flat ``LabelSet(exclusive=True,
            allow_abstain=True)`` from the names.
        model: Provider model identifier. Required only for LLM
            supervision modes (``zero_shot`` / ``few_shot``).
        supervision: One of five modes (plan §7.1):

            - ``"zero_shot"`` (default) → :class:`ZeroShotClassify`.
            - ``"few_shot"`` → :class:`FewShotClassify`. Requires
              ``examples``.
            - ``"prototype"`` → no-LLM :class:`PrototypeClassify`.
              Requires ``embedder``.
            - ``"retrieval"`` → no-LLM (or LLM-tie-broken)
              :class:`RetrievalClassify`. Requires ``embedder`` +
              ``corpus``.
            - ``"nli"`` → no-LLM
              :class:`ZeroShotNLIClassifier`. Requires ``nli_scorer``.

        examples: Required when ``supervision="few_shot"``.
        embedder: Required when
            ``supervision in {"prototype", "retrieval"}``. Object
            conforming to
            :class:`~kaos_llm_core.programs.classify.Embedder`
            (canonical: ``kaos_nlp_transformers.EmbeddingModel``).
        corpus: Required when ``supervision="retrieval"``. List of
            ``(example_text, label_name)`` pairs.
        nli_scorer: Required when ``supervision="nli"``. Object
            conforming to
            :class:`~kaos_llm_core.programs.classify.NLIScorer`
            (canonical: a future ``kaos_nlp_transformers.NliModel``).
        long_strategy: ``"auto"`` (default) picks ``"single"`` when
            ``len(doc) <= 12_000`` characters and ``"chunk"``
            otherwise. ``"single"`` forces a single classifier call
            on the whole doc; ``"chunk"`` wraps the chosen classifier
            in :class:`~kaos_llm_core.programs.classify.ChunkedClassify`.
        aggregator: Aggregator for the chunked path. ``None`` resolves
            via :meth:`ChunkedClassify._default_aggregator`.
        cache / budget / chunker: Threaded into ``ChunkedClassify``
            when the chunked path is selected. Ignored on the single
            path.
        settings: Optional :class:`KaosLLMCoreSettings`.
    """
    from kaos_llm_core.labels import LabelSet
    from kaos_llm_core.programs.classify import (
        ChunkedClassify,
        FewShotClassify,
        PrototypeClassify,
        RetrievalClassify,
        ZeroShotClassify,
        ZeroShotNLIClassifier,
    )

    resolved_settings = settings or KaosLLMCoreSettings()

    label_set: LabelSet
    if isinstance(labels, LabelSet):
        label_set = labels
    else:
        names = list(labels)
        if not names:
            raise CallError(
                "classify_doc() requires a non-empty `labels` argument "
                "(LabelSet or sequence of names)."
            )
        label_set = LabelSet.from_names(names)

    # Resolve the leaf Program by supervision mode.
    leaf_program: Any
    if supervision in {"zero_shot", "few_shot"}:
        resolved_model = _resolve_default_model(model, settings=settings)
        leaf_kwargs: dict[str, Any] = {
            "labels": label_set,
            "model": resolved_model,
            "core_settings": resolved_settings,
        }
        if supervision == "few_shot":
            if not examples:
                raise CallError(
                    "classify_doc(supervision='few_shot') requires a non-empty "
                    "`examples=` sequence. Pass at least one Example or switch "
                    "to supervision='zero_shot'."
                )
            leaf_kwargs["examples"] = list(examples)
            leaf_program = FewShotClassify(**leaf_kwargs)
        else:
            leaf_program = ZeroShotClassify(**leaf_kwargs)
    elif supervision == "prototype":
        if embedder is None:
            raise CallError(
                "classify_doc(supervision='prototype') requires an `embedder=` "
                "argument (any object conforming to the Embedder protocol)."
            )
        leaf_program = PrototypeClassify(labels=label_set, embedder=embedder)
    elif supervision == "retrieval":
        if embedder is None or corpus is None:
            raise CallError(
                "classify_doc(supervision='retrieval') requires both `embedder=` "
                "and `corpus=` arguments."
            )
        leaf_program = RetrievalClassify(
            labels=label_set,
            embedder=embedder,
            corpus=list(corpus),
        )
    elif supervision == "nli":
        if nli_scorer is None:
            raise CallError(
                "classify_doc(supervision='nli') requires an `nli_scorer=` "
                "argument (any object conforming to the NLIScorer protocol)."
            )
        leaf_program = ZeroShotNLIClassifier(labels=label_set, scorer=nli_scorer)
    else:  # pragma: no cover - guarded by Literal
        raise CallError(f"unknown supervision mode: {supervision!r}")

    strategy = _resolve_long_classify_strategy(len(doc), long_strategy)
    if strategy == "single":
        single_result: Classification = await leaf_program(text=doc)
        return single_result.model_copy(
            update={
                "metadata": {
                    **dict(single_result.metadata),
                    "starter.long_strategy": strategy,
                    "starter.facade": "classify_doc",
                    "starter.supervision": supervision,
                },
            }
        )

    chunked_kwargs: dict[str, Any] = {
        "labels": label_set,
        "per_chunk": leaf_program,
    }
    if chunker is not None:
        chunked_kwargs["chunker"] = chunker
    if aggregator is not None:
        chunked_kwargs["aggregator"] = aggregator
    if cache is not None:
        chunked_kwargs["cache"] = cache
    if budget is not None:
        chunked_kwargs["budget"] = budget
    chunked_program = ChunkedClassify(**chunked_kwargs)
    chunked_result: Classification = await chunked_program(text=doc)
    return chunked_result.model_copy(
        update={
            "metadata": {
                **dict(chunked_result.metadata),
                "starter.long_strategy": strategy,
                "starter.facade": "classify_doc",
                "starter.supervision": supervision,
            },
        }
    )


def classify_doc_sync(
    doc: str,
    labels: LabelSet | Sequence[str],
    *,
    model: str | None = None,
    supervision: Literal["zero_shot", "few_shot", "prototype", "retrieval", "nli"] = "zero_shot",
    examples: Sequence[Any] | None = None,
    embedder: Any = None,
    corpus: Sequence[tuple[str, str]] | None = None,
    nli_scorer: Any = None,
    long_strategy: LongClassifyStrategy = "auto",
    aggregator: Aggregator | None = None,
    cache: ChunkCache | None = None,
    budget: Budget | None = None,
    chunker: Chunker | None = None,
    settings: KaosLLMCoreSettings | None = None,
) -> Classification:
    """Synchronous wrapper around :func:`classify_doc`."""
    return asyncio.run(
        classify_doc(
            doc,
            labels,
            model=model,
            supervision=supervision,
            examples=examples,
            embedder=embedder,
            corpus=corpus,
            nli_scorer=nli_scorer,
            long_strategy=long_strategy,
            aggregator=aggregator,
            cache=cache,
            budget=budget,
            chunker=chunker,
            settings=settings,
        )
    )


# Forward refs imported lazily inside the new façades to avoid pulling
# the long-doc Programs / Budget / ChunkCache at import time. Tests and
# CLI helpers can import these names from the public package surface.
if TYPE_CHECKING:
    from collections.abc import Sequence

    from kaos_nlp_core.chunking import Chunker

    from kaos_llm_core.cache import ChunkCache
    from kaos_llm_core.composition import Aggregator
    from kaos_llm_core.labels import LabelSet
    from kaos_llm_core.optimization.budget import Budget
    from kaos_llm_core.results import Classification, Summary


__all__ = [
    "LongClassifyStrategy",
    "LongSummaryStrategy",
    "SummaryStyle",
    "classify",
    "classify_doc",
    "classify_doc_sync",
    "classify_sync",
    "extract",
    "extract_sync",
    "summarize",
    "summarize_doc",
    "summarize_doc_sync",
    "summarize_sync",
    "text",
    "text_sync",
]
