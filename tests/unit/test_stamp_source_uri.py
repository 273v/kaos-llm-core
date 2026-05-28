"""Pin the dispatcher-owns-source_uri contract.

`Cited[T]`'s Pydantic validation enforces `source_uri` is non-empty
per `Span` but cannot enforce that the value MATCHES the document
the LLM was shown. Extraction-time literature documents 3-13%
URL-hallucination rates on uncontrolled `source_uri` outputs.

`stamp_source_uri(cited, source_uri=...)` closes the gap by giving
the dispatcher a typed helper that overrides every span's
`source_uri` with the authoritative identifier the dispatcher
knows. The LLM can emit whatever it wants for `source_uri` during
extraction; the dispatcher overwrites it before the value enters
the agent's memory or the user's answer.

These tests pin:

1. Every span's `source_uri` is overridden (including across
   multi-span cited values).
2. `value` and `confidence` are preserved verbatim.
3. `quote`, `char_span`, `page` are preserved (`Span`'s own
   validators recompute `quote_hash`).
4. The returned `Cited` is a NEW instance (input not mutated;
   helps callers reason about ownership).
5. Empty `source_uri` is rejected (per `Span` `min_length=1`).
6. Unicode source_uri works.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from kaos_llm_core.signatures.grounding import Cited, Span, stamp_source_uri


def _make_span(
    source_uri: str = "doc-A.docx",
    quote: str = "the agreement is governed by Delaware law",
    char_span: tuple[int, int] = (100, 142),
    page: int | None = None,
) -> Span:
    return Span(
        source_uri=source_uri,
        page=page,
        char_span=char_span,
        quote=quote,
    )


class TestStampOverridesSourceUri:
    def test_single_span_overridden(self) -> None:
        original = Cited[str](
            value="Delaware",
            spans=[_make_span(source_uri="LLM-emitted-wrong.docx")],
            confidence=0.9,
        )
        stamped = stamp_source_uri(original, source_uri="EMNA Mutual NDA.docx")
        assert stamped.spans[0].source_uri == "EMNA Mutual NDA.docx"

    def test_multi_span_all_overridden(self) -> None:
        original = Cited[str](
            value="Delaware",
            spans=[
                _make_span(source_uri="LLM-said-A.docx", quote="span 1", char_span=(0, 6)),
                _make_span(source_uri="LLM-said-B.docx", quote="span 2", char_span=(10, 16)),
                _make_span(source_uri="LLM-said-C.docx", quote="span 3", char_span=(20, 26)),
            ],
        )
        stamped = stamp_source_uri(original, source_uri="dispatcher-truth.docx")
        for span in stamped.spans:
            assert span.source_uri == "dispatcher-truth.docx", (
                "every span must be stamped; cross-doc citations are "
                "the threat model (LLM mixes attribution across N "
                "documents in a single fan-out turn)"
            )

    def test_stamp_overrides_even_when_llm_emitted_correct_value(self) -> None:
        """The override is unconditional; we don't trust the LLM
        emitted value even when it happens to be correct. The
        dispatcher is the authoritative source of identity."""
        original = Cited[str](
            value="Delaware",
            spans=[_make_span(source_uri="emna.docx")],
        )
        stamped = stamp_source_uri(original, source_uri="EMNA Mutual NDA.docx")
        assert stamped.spans[0].source_uri == "EMNA Mutual NDA.docx"


class TestStampPreservesEverythingElse:
    def test_value_preserved(self) -> None:
        original = Cited[str](
            value="Michigan governing law",
            spans=[_make_span()],
            confidence=0.85,
        )
        stamped = stamp_source_uri(original, source_uri="new.docx")
        assert stamped.value == "Michigan governing law"

    def test_confidence_preserved(self) -> None:
        original = Cited[str](
            value="X",
            spans=[_make_span()],
            confidence=0.42,
        )
        stamped = stamp_source_uri(original, source_uri="new.docx")
        assert stamped.confidence == 0.42

    def test_quote_preserved(self) -> None:
        original = Cited[str](
            value="X",
            spans=[_make_span(quote="the exact verbatim text of the clause")],
        )
        stamped = stamp_source_uri(original, source_uri="new.docx")
        assert stamped.spans[0].quote == "the exact verbatim text of the clause"

    def test_char_span_preserved(self) -> None:
        original = Cited[str](
            value="X",
            spans=[_make_span(char_span=(123, 456))],
        )
        stamped = stamp_source_uri(original, source_uri="new.docx")
        assert stamped.spans[0].char_span == (123, 456)

    def test_page_preserved(self) -> None:
        original = Cited[str](
            value="X",
            spans=[_make_span(page=7)],
        )
        stamped = stamp_source_uri(original, source_uri="new.docx")
        assert stamped.spans[0].page == 7

    def test_quote_hash_consistent_after_stamp(self) -> None:
        """quote_hash is force-recomputed by Span's own validator on
        every construction; the stamped span should have a hash that
        matches the preserved quote."""
        original = Cited[str](
            value="X",
            spans=[_make_span(quote="some quoted text")],
        )
        stamped = stamp_source_uri(original, source_uri="new.docx")
        # quote_hash exists and matches the quote (not empty / not stale)
        assert stamped.spans[0].quote_hash
        assert stamped.spans[0].quote_hash == original.spans[0].quote_hash


class TestStampOwnership:
    def test_returns_new_instance(self) -> None:
        original = Cited[str](
            value="X",
            spans=[_make_span()],
        )
        stamped = stamp_source_uri(original, source_uri="new.docx")
        assert stamped is not original

    def test_input_spans_not_mutated(self) -> None:
        """Span is frozen so this should be impossible to violate,
        but pin the contract explicitly."""
        original = Cited[str](
            value="X",
            spans=[_make_span(source_uri="original.docx")],
        )
        _ = stamp_source_uri(original, source_uri="new.docx")
        assert original.spans[0].source_uri == "original.docx"


class TestStampValidation:
    def test_empty_source_uri_rejected(self) -> None:
        original = Cited[str](
            value="X",
            spans=[_make_span()],
        )
        with pytest.raises(ValidationError):
            stamp_source_uri(original, source_uri="")

    def test_unicode_source_uri_accepted(self) -> None:
        original = Cited[str](
            value="X",
            spans=[_make_span()],
        )
        stamped = stamp_source_uri(original, source_uri="Vertrag — München & Köln 中文.docx")
        assert stamped.spans[0].source_uri == "Vertrag — München & Köln 中文.docx"

    def test_complex_value_type_round_trips(self) -> None:
        """Cited[T] is generic; the value can be any type. The helper
        must preserve complex values verbatim."""
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class Clause:
            kind: str
            text: str

        original = Cited[Clause](
            value=Clause(kind="indemnification", text="hold harmless"),
            spans=[_make_span()],
        )
        stamped = stamp_source_uri(original, source_uri="new.docx")
        assert stamped.value.kind == "indemnification"
        assert stamped.value.text == "hold harmless"


class TestStampPublicSurface:
    def test_importable_from_signatures_package(self) -> None:
        """stamp_source_uri is re-exported from kaos_llm_core.signatures."""
        from kaos_llm_core.signatures import stamp_source_uri as exported

        assert exported is stamp_source_uri
