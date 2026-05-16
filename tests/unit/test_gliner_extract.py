"""Tests for :class:`~kaos_llm_core.programs.ner.GLiNERExtract`.

Offline + deterministic: a stub :class:`NerExtractor` returns
hand-crafted entity dicts so the Program's marshalling logic is
covered without a real GLiNER model. The real
``kaos_nlp_transformers.GLiNERExtractor`` satisfies the same Protocol
at runtime once it ships (plan §4.2.4, Phase 8).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from kaos_llm_core.programs.ner import GLiNERExtract, NerExtractor
from kaos_llm_core.results import Entities, EntitySpan


@dataclass(frozen=True, slots=True)
class _StubEntity:
    start: int
    end: int
    text: str
    label: str
    score: float


class _StubExtractor:
    def __init__(self, fixed: list[_StubEntity]) -> None:
        self._fixed = fixed
        self.calls: list[tuple[list[str], list[str], dict]] = []

    def extract(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
        *,
        threshold: float = 0.5,
        max_width: int = 12,
        flat_ner: bool = True,
        dup_label: bool = False,
        multi_label: bool = False,
    ) -> Sequence[Sequence[_StubEntity]]:
        self.calls.append(
            (
                list(texts),
                list(labels),
                {
                    "threshold": threshold,
                    "max_width": max_width,
                    "flat_ner": flat_ner,
                    "dup_label": dup_label,
                    "multi_label": multi_label,
                },
            )
        )
        return [list(self._fixed)]


class TestGLiNERExtract:
    @pytest.mark.asyncio
    async def test_basic_extraction_marshals_into_entities(self) -> None:
        stub = _StubExtractor(
            [
                _StubEntity(0, 12, "Barack Obama", "person", 0.99),
                _StubEntity(25, 31, "Hawaii", "place", 0.97),
            ]
        )
        program = GLiNERExtract(
            extractor=stub,
            labels=["person", "place"],
        )
        result = await program(text="Barack Obama was born in Hawaii.")
        assert isinstance(result, Entities)
        assert len(result.spans) == 2
        assert all(isinstance(s, EntitySpan) for s in result.spans)
        # Source order — sorted by (start, end).
        assert result.spans[0].text == "Barack Obama"
        assert result.spans[1].text == "Hawaii"
        assert result.labels == ["person", "place"]
        assert result.metadata["program"] == "GLiNERExtract"
        assert result.metadata["extractor"] == "_StubExtractor"
        assert result.metadata["n_spans"] == 2

    @pytest.mark.asyncio
    async def test_empty_input_short_circuits_no_backend_call(self) -> None:
        stub = _StubExtractor([])
        program = GLiNERExtract(extractor=stub, labels=["person"])
        result = await program(text="")
        assert result.spans == []
        assert result.metadata["skip_reason"] == "empty_input"
        assert stub.calls == []

    @pytest.mark.asyncio
    async def test_forwarded_params_match_init(self) -> None:
        stub = _StubExtractor([])
        program = GLiNERExtract(
            extractor=stub,
            labels=["person"],
            threshold=0.3,
            max_width=5,
            flat_ner=False,
            dup_label=True,
            multi_label=True,
        )
        await program(text="hello world")
        assert len(stub.calls) == 1
        texts, labels, params = stub.calls[0]
        assert texts == ["hello world"]
        assert labels == ["person"]
        assert params == {
            "threshold": 0.3,
            "max_width": 5,
            "flat_ner": False,
            "dup_label": True,
            "multi_label": True,
        }

    @pytest.mark.asyncio
    async def test_sort_order_is_source_byte_offset(self) -> None:
        # Backend returns entities out of order; Program must sort them.
        stub = _StubExtractor(
            [
                _StubEntity(25, 31, "Hawaii", "place", 0.97),
                _StubEntity(0, 12, "Barack Obama", "person", 0.99),
            ]
        )
        program = GLiNERExtract(extractor=stub, labels=["person", "place"])
        result = await program(text="Barack Obama was born in Hawaii.")
        starts = [s.start for s in result.spans]
        assert starts == sorted(starts)

    def test_rejects_empty_labels(self) -> None:
        with pytest.raises(ValueError):
            GLiNERExtract(extractor=_StubExtractor([]), labels=[])

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_rejects_out_of_range_threshold(self, bad: float) -> None:
        with pytest.raises(ValueError):
            GLiNERExtract(extractor=_StubExtractor([]), labels=["x"], threshold=bad)

    def test_rejects_zero_max_width(self) -> None:
        with pytest.raises(ValueError):
            GLiNERExtract(extractor=_StubExtractor([]), labels=["x"], max_width=0)

    def test_stub_satisfies_ner_extractor_protocol(self) -> None:
        """The Protocol is runtime-checkable — a structurally compatible
        stub must pass ``isinstance``."""
        stub = _StubExtractor([])
        assert isinstance(stub, NerExtractor)
