"""Unit tests for FUND-10 cross-document dedup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from kaos_llm_core.dedup import DedupResult, dedup_cells


@dataclass
class FakeCell:
    doc_id: str
    column_id: str
    ai_value: Any
    status: str = "extracted"


class TestDedupCells:
    def test_empty_input(self) -> None:
        result = dedup_cells([])
        assert result.cells == ()
        assert result.removed_count == 0

    def test_no_duplicates(self) -> None:
        cells = [
            FakeCell("d1", "parties", "Acme Corp"),
            FakeCell("d2", "parties", "Beta LLC"),
        ]
        result = dedup_cells(cells)
        assert len(result.cells) == 2
        assert result.removed_count == 0

    def test_exact_duplicate_removed(self) -> None:
        cells = [
            FakeCell("d1", "parties", "Acme Corp"),
            FakeCell("d2", "parties", "Acme Corp"),
            FakeCell("d3", "parties", "Acme Corp"),
        ]
        result = dedup_cells(cells)
        assert len(result.cells) == 1
        assert result.removed_count == 2
        assert result.cells[0].doc_id == "d1"

    def test_whitespace_variation_deduped(self) -> None:
        cells = [
            FakeCell("d1", "parties", "Acme Corp"),
            FakeCell("d2", "parties", "  Acme   Corp  "),
        ]
        result = dedup_cells(cells)
        assert len(result.cells) == 1
        assert result.removed_count == 1

    def test_case_variation_deduped(self) -> None:
        cells = [
            FakeCell("d1", "parties", "ACME CORP"),
            FakeCell("d2", "parties", "Acme Corp"),
        ]
        result = dedup_cells(cells)
        assert len(result.cells) == 1
        assert result.removed_count == 1

    def test_entity_suffix_variation_deduped(self) -> None:
        cells = [
            FakeCell("d1", "parties", "Acme Inc."),
            FakeCell("d2", "parties", "Acme Incorporated"),
            FakeCell("d3", "parties", "Acme Corporation"),
        ]
        result = dedup_cells(cells, column_types={"parties": "entity_role"})
        assert len(result.cells) == 1
        assert result.removed_count == 2

    def test_different_columns_not_deduped(self) -> None:
        cells = [
            FakeCell("d1", "parties", "Acme Corp"),
            FakeCell("d1", "governing_law", "Acme Corp"),
        ]
        result = dedup_cells(cells)
        assert len(result.cells) == 2
        assert result.removed_count == 0

    def test_non_extracted_cells_pass_through(self) -> None:
        cells = [
            FakeCell("d1", "parties", None, status="not_in_document"),
            FakeCell("d2", "parties", None, status="not_in_document"),
        ]
        result = dedup_cells(cells)
        assert len(result.cells) == 2
        assert result.removed_count == 0

    def test_date_normalization(self) -> None:
        cells = [
            FakeCell("d1", "effective_date", "2024-01-15"),
            FakeCell("d2", "effective_date", "01/15/2024"),
            FakeCell("d3", "effective_date", "15 January 2024"),
        ]
        result = dedup_cells(cells, column_types={"effective_date": "date"})
        assert len(result.cells) == 1
        assert result.removed_count == 2

    def test_money_normalization(self) -> None:
        cells = [
            FakeCell("d1", "cap", "$1,000,000"),
            FakeCell("d2", "cap", "1000000"),
            FakeCell("d3", "cap", "$1000000"),
        ]
        result = dedup_cells(cells, column_types={"cap": "money"})
        assert len(result.cells) == 1
        assert result.removed_count == 2

    def test_lineage_tracks_doc_ids(self) -> None:
        cells = [
            FakeCell("d1", "parties", "Acme Corp"),
            FakeCell("d2", "parties", "Acme Corp"),
            FakeCell("d3", "parties", "Acme Corp"),
        ]
        result = dedup_cells(cells)
        parties_lineage = result.lineage["parties"]
        assert len(parties_lineage) == 1
        content_hash = next(iter(parties_lineage))
        assert sorted(parties_lineage[content_hash]) == ["d1", "d2", "d3"]

    def test_mixed_columns_and_types(self) -> None:
        cells = [
            FakeCell("d1", "parties", "Acme Inc."),
            FakeCell("d2", "parties", "Acme Corporation"),
            FakeCell("d1", "effective_date", "2024-03-15"),
            FakeCell("d2", "effective_date", "03/15/2024"),
            FakeCell("d1", "governing_law", "Delaware"),
            FakeCell("d2", "governing_law", "New York"),
        ]
        result = dedup_cells(
            cells,
            column_types={
                "parties": "entity_role",
                "effective_date": "date",
                "governing_law": "string",
            },
        )
        assert result.removed_count == 2
        assert len(result.cells) == 4

    def test_result_is_frozen(self) -> None:
        result = dedup_cells([])
        assert isinstance(result, DedupResult)
        with pytest.raises(AttributeError):
            result.__setattr__("removed_count", 5)
