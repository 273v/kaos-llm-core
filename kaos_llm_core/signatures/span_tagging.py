"""Span tagging — mark source text with citation anchors for LLM prompts.

When source text is passed to an LLM for extraction, tagging it with
anchors lets the LLM reference specific passages by ID rather than
guessing character offsets.  This is the same pattern Anthropic's
"cite your sources" prompts use, made reusable infrastructure.

Two tagging modes:

- **passage**: Tag each passage in a multi-passage context with a
  ``=== SOURCE: <uri> ===`` header.  This is what RAG already does.
- **paragraph**: Tag each paragraph (double-newline-separated block)
  within a single document with ``[SRC:<id>]`` markers.

After the LLM generates output with ``Cited[T]`` fields, use
:func:`validate_cited_output` to verify every span resolves to
the original source text.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from kaos_core.logging import get_logger

from kaos_llm_core.signatures.grounding import (
    DEFAULT_MATCH_STRATEGIES,
    Cited,
    MatchStrategy,
    SpanError,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Paragraph-level tagging
# ---------------------------------------------------------------------------


def tag_paragraphs(
    text: str,
    *,
    source_uri: str = "doc:input",
    min_paragraph_chars: int = 20,
) -> tuple[str, dict[str, tuple[int, int]]]:
    """Tag each paragraph in ``text`` with a source anchor.

    Paragraphs are delimited by two or more consecutive newlines.
    Each paragraph is prefixed with ``[SRC:<source_uri>#p<N>]``
    where N is the 0-indexed paragraph number.

    Args:
        text: The source text to tag.
        source_uri: Base URI for the source document.
        min_paragraph_chars: Minimum characters for a paragraph to be
            tagged.  Shorter blocks (headers, blank lines) are included
            but not tagged.

    Returns:
        A 2-tuple ``(tagged_text, span_map)`` where:
        - ``tagged_text`` is the text with ``[SRC:...]`` markers
        - ``span_map`` maps each marker URI to ``(start, end)``
          character offsets in the **original** (untagged) text

    Example::

        tagged, spans = tag_paragraphs(
            "First paragraph.\\n\\nSecond paragraph.",
            source_uri="doc:contract",
        )
        # tagged = "[SRC:doc:contract#p0] First paragraph.\\n\\n"
        #          "[SRC:doc:contract#p1] Second paragraph."
        # spans = {"doc:contract#p0": (0, 16), "doc:contract#p1": (18, 35)}
    """
    import re

    # Split on 2+ newlines while preserving the delimiter
    parts = re.split(r"(\n{2,})", text)
    span_map: dict[str, tuple[int, int]] = {}
    tagged_parts: list[str] = []
    para_idx = 0
    char_offset = 0

    for part in parts:
        if not part.strip():
            # Delimiter (blank lines) — pass through
            tagged_parts.append(part)
            char_offset += len(part)
            continue

        # Content paragraph
        start = char_offset
        end = char_offset + len(part)

        if len(part.strip()) >= min_paragraph_chars:
            uri = f"{source_uri}#p{para_idx}"
            span_map[uri] = (start, end)
            tagged_parts.append(f"[SRC:{uri}] {part}")
            para_idx += 1
        else:
            tagged_parts.append(part)

        char_offset = end

    tagged_text = "".join(tagged_parts)
    return tagged_text, span_map


# ---------------------------------------------------------------------------
# Passage-level tagging (multi-document)
# ---------------------------------------------------------------------------


def tag_passages(
    passages: dict[str, str],
) -> tuple[str, dict[str, str]]:
    """Tag multiple passages with ``=== SOURCE: <uri> ===`` headers.

    This is the same format RAG uses. Each passage gets a header line
    with its URI, making it citable by the LLM.

    Args:
        passages: Mapping of ``source_uri`` to passage text.

    Returns:
        A 2-tuple ``(tagged_text, source_map)`` where:
        - ``tagged_text`` is the assembled context with SOURCE headers
        - ``source_map`` is the same ``passages`` dict (for verification)
    """
    parts: list[str] = []
    for uri, text in passages.items():
        parts.append(f"=== SOURCE: {uri} ===")
        parts.append(text)
    return "\n\n".join(parts), dict(passages)


# ---------------------------------------------------------------------------
# Post-decode citation validation
# ---------------------------------------------------------------------------


def validate_cited_output(
    output: Any,
    corpus: dict[str, str] | Callable[[str], str],
    *,
    strategies: Iterable[MatchStrategy] = DEFAULT_MATCH_STRATEGIES,
    threshold: float = 0.9,
    span_map: dict[str, tuple[int, int]] | None = None,
) -> list[SpanError]:
    """Validate all Cited[T] fields in a decoded output.

    Walks the output object looking for ``Cited`` instances and
    verifies each one's spans against the corpus.

    If ``span_map`` is provided (from :func:`tag_paragraphs`), it is
    used to expand paragraph-level URIs (``doc:x#p0``) into the
    actual source text for verification.

    Args:
        output: The decoded Signature output (a Pydantic model).
        corpus: Source text lookup — dict or callable.
        strategies: Match strategies for verification.
        threshold: Similarity threshold for fuzzy strategies.
        span_map: Optional mapping from paragraph URIs to char spans
            in the original text (from :func:`tag_paragraphs`).

    Returns:
        A list of :class:`SpanError` — empty if all citations verified.
    """
    all_errors: list[SpanError] = []

    # Build an enriched corpus that resolves paragraph URIs
    effective_corpus: dict[str, str] | Callable[[str], str]
    if span_map and isinstance(corpus, dict):
        # For paragraph-tagged text, resolve #pN URIs to their text.
        # Cast to typed dict to help ty resolve the union.
        corpus_dict: dict[str, str] = corpus  # ty: ignore[invalid-assignment]
        enriched = dict(corpus_dict)
        base_uri = None
        for uri, (start, end) in span_map.items():
            if base_uri is None:
                base_uri = uri.split("#")[0]
            if base_uri in corpus_dict:
                enriched[uri] = corpus_dict[base_uri][start:end]
        effective_corpus = enriched
    else:
        effective_corpus = corpus

    # Walk the output looking for Cited instances
    cited_values = _extract_cited_values(output)
    for cited in cited_values:
        errors = cited.verify(effective_corpus, strategies=strategies, threshold=threshold)
        all_errors.extend(errors)

    if all_errors:
        logger.debug(
            "validate_cited_output: %d span error(s) across %d Cited field(s)",
            len(all_errors),
            len(cited_values),
        )

    return all_errors


def _extract_cited_values(obj: Any) -> list[Cited[Any]]:
    """Recursively extract all Cited instances from a Pydantic model."""
    results: list[Cited[Any]] = []

    if isinstance(obj, Cited):
        results.append(obj)
        return results

    # Walk Pydantic model fields (access on class, not instance — Pydantic 2.11+)
    if hasattr(type(obj), "model_fields"):
        for field_name in type(obj).model_fields:
            value = getattr(obj, field_name, None)
            if value is not None:
                results.extend(_extract_cited_values(value))
    elif isinstance(obj, list | tuple):
        for item in obj:
            results.extend(_extract_cited_values(item))
    elif isinstance(obj, dict):
        for v in obj.values():
            results.extend(_extract_cited_values(v))

    return results
