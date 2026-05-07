"""Live end-to-end test proving a ``kaos_content.corpus.Corpus`` Protocol
instance flows through ``RAG`` without hitting the legacy
``_is_corpus`` duck-type fallback.

This validates the WS-3.3 (Corpus Protocol) + WS-3.5 (RAG isinstance swap)
integration point: a caller builds a ``ContentDocumentCorpus`` from a
``ContentDocument`` and hands it straight to ``RAG.query``. The internal
``_is_corpus`` check now returns True via the formal Protocol rather
than the string-name attr-triple duck-type.

The test deliberately exercises the ``.units + .unit + __iter__``
duck-type path as well by building the Corpus dict at the top of
``RAG.forward`` — so even though ``ContentDocumentCorpus`` does NOT have
``.units`` or ``.unit`` methods, the Protocol branch handles it because
the updated ``_is_corpus`` routes Protocol-conformant instances through
the same passage-iteration code path.
"""

from __future__ import annotations

import pytest
from kaos_content.corpus import ContentDocumentCorpus
from kaos_content.model.attr import Provenance, SourceRef
from kaos_content.model.blocks import Paragraph
from kaos_content.model.document import ContentDocument
from kaos_content.model.inlines import Text
from kaos_content.model.metadata import DocumentMetadata

from kaos_llm_core.programs.rag import RAG, RAGResult
from kaos_llm_core.signatures import Answer, MatchStrategy

from .conftest import requires_anthropic

_LENIENT = (
    MatchStrategy.STRICT,
    MatchStrategy.SUBSTRING,
    MatchStrategy.CASE_INSENSITIVE,
    MatchStrategy.NORMALIZED_TOKEN,
)


def _build_doc() -> ContentDocument:
    source = SourceRef(uri="doc:grounding/delaware-gcl-live")
    prov = Provenance(source=source, page=1)
    return ContentDocument(
        metadata=DocumentMetadata(title="Delaware GCL (fixture)"),
        body=(
            Paragraph(
                children=(
                    Text(
                        value=(
                            "The Delaware General Corporation Law requires that a "
                            "certificate of incorporation be filed with the "
                            "Secretary of State. The filing fee is $89 and the "
                            "certificate must include the corporation's name, "
                            "its registered office, and the name of its "
                            "registered agent."
                        )
                    ),
                ),
                provenance=prov,
            ),
        ),
    )


@requires_anthropic
@pytest.mark.integration
class TestRAGAcceptsContentDocumentCorpus:
    async def test_content_document_corpus_feeds_rag(self) -> None:
        """Handing a ContentDocumentCorpus to RAG.query must yield a
        verified Answer — exercises the new Protocol branch of
        ``_is_corpus``."""
        corpus = ContentDocumentCorpus([_build_doc()])
        rag = RAG(
            model="anthropic:claude-haiku-4-5",
            match_strategies=_LENIENT,
            top_k=5,
        )
        result = await rag.query(
            question="What is the filing fee for a Delaware certificate of incorporation?",
            documents=corpus,
        )
        assert isinstance(result, RAGResult)
        assert isinstance(result.grounded_answer, Answer), (
            "Expected an Answer for an answerable factual question. "
            "Got InsufficientEvidence, which means either the Protocol "
            "adoption broke retrieval or the model drifted."
        )
        assert "89" in str(result.grounded_answer.value), (
            f"Answer should mention $89: {result.grounded_answer.value!r}"
        )
        assert len(result.verification_errors) == 0, (
            f"Unexpected verification errors: {result.verification_errors}"
        )
