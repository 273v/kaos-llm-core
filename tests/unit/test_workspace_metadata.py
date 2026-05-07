"""Unit tests for the Phase 15.3 workspace SQLite metadata store."""

from __future__ import annotations

from pathlib import Path

import pytest

from kaos_llm_core.programs.batch import BatchProgress
from kaos_llm_core.workspace import (
    BatchAlreadyExistsError,
    BatchRecord,
    WorkspaceMetadata,
)
from kaos_llm_core.workspace.metadata import _now_iso


def _record(batch_id: str = "batch-test-1", **overrides) -> BatchRecord:
    base: dict = {
        "batch_id": batch_id,
        "name": "smoke",
        "created_at": _now_iso(),
        "created_by": None,
        "envelope_path": "/vfs/p.json",
        "program_hash": "sha256:abc",
        "program_name": "p",
        "inputs_source": {"type": "list", "n_items_hint": 3},
        "inputs_count_hint": 3,
        "output_dir": "/vfs/runs/test-1",
        "log_path": "/abs/items.jsonl",
        "manifest_path": "/abs/manifest.json",
    }
    base.update(overrides)
    return BatchRecord(**base)


class TestWorkspaceCRUD:
    def test_create_and_get_round_trip(self, tmp_path: Path) -> None:
        ws = WorkspaceMetadata(tmp_path / "ws.sqlite")
        rec = _record()
        ws.create_batch(rec)
        got = ws.get_batch("batch-test-1")
        assert got is not None
        assert got.batch_id == "batch-test-1"
        assert got.inputs_source == {"type": "list", "n_items_hint": 3}
        assert got.status == "pending"
        assert got.cost_usd == 0.0
        assert got.n_succeeded == 0

    def test_get_missing_returns_none(self, tmp_path: Path) -> None:
        ws = WorkspaceMetadata(tmp_path / "ws.sqlite")
        assert ws.get_batch("does-not-exist") is None

    def test_create_collision_raises(self, tmp_path: Path) -> None:
        ws = WorkspaceMetadata(tmp_path / "ws.sqlite")
        ws.create_batch(_record())
        with pytest.raises(BatchAlreadyExistsError):
            ws.create_batch(_record())

    def test_list_batches_filter_by_status(self, tmp_path: Path) -> None:
        ws = WorkspaceMetadata(tmp_path / "ws.sqlite")
        ws.create_batch(_record("a"))
        ws.create_batch(_record("b"))
        ws.create_batch(_record("c"))
        ws.update_status("b", "running", started_at=_now_iso())
        ws.update_status("c", "completed", completed_at=_now_iso())
        all_batches = ws.list_batches()
        assert {b.batch_id for b in all_batches} == {"a", "b", "c"}
        running = ws.list_batches(status="running")
        assert [b.batch_id for b in running] == ["b"]
        done = ws.list_batches(status="completed")
        assert [b.batch_id for b in done] == ["c"]

    def test_update_progress_writes_progress_columns(self, tmp_path: Path) -> None:
        ws = WorkspaceMetadata(tmp_path / "ws.sqlite")
        ws.create_batch(_record())
        prog = BatchProgress(
            n_done=3,
            n_total=10,
            n_succeeded=3,
            n_errored=0,
            n_skipped=0,
            cost_usd_so_far=0.0042,
            tokens_so_far=420,
            last_custom_id="abc",
            last_status="success",
        )
        ws.update_progress("batch-test-1", prog)
        got = ws.get_batch("batch-test-1")
        assert got is not None
        assert got.n_succeeded == 3
        assert got.cost_usd == pytest.approx(0.0042)
        assert got.total_tokens == 420
        assert got.last_progress_at is not None

    def test_update_status_with_terminal_fields(self, tmp_path: Path) -> None:
        ws = WorkspaceMetadata(tmp_path / "ws.sqlite")
        ws.create_batch(_record())
        ws.update_status(
            "batch-test-1",
            "completed",
            completed_at=_now_iso(),
            duration_s=12.5,
            n_total=10,
            n_succeeded=10,
            cost_usd=0.99,
        )
        got = ws.get_batch("batch-test-1")
        assert got is not None
        assert got.status == "completed"
        assert got.duration_s == pytest.approx(12.5)
        assert got.n_total == 10
        assert got.cost_usd == pytest.approx(0.99)

    def test_update_status_rejects_unknown_field(self, tmp_path: Path) -> None:
        ws = WorkspaceMetadata(tmp_path / "ws.sqlite")
        ws.create_batch(_record())
        with pytest.raises(Exception, match="unknown terminal field"):
            ws.update_status("batch-test-1", "completed", bogus_field="x")

    def test_error_summary_round_trip(self, tmp_path: Path) -> None:
        ws = WorkspaceMetadata(tmp_path / "ws.sqlite")
        ws.create_batch(_record())
        ws.update_status(
            "batch-test-1",
            "failed",
            completed_at=_now_iso(),
            error_summary={"errors_by_type": {"CallError": 3}, "first": "boom"},
        )
        got = ws.get_batch("batch-test-1")
        assert got is not None
        assert got.error_summary == {
            "errors_by_type": {"CallError": 3},
            "first": "boom",
        }

    def test_cancel_batch_sets_cancel_requested_at(self, tmp_path: Path) -> None:
        ws = WorkspaceMetadata(tmp_path / "ws.sqlite")
        ws.create_batch(_record())
        ws.cancel_batch("batch-test-1")
        got = ws.get_batch("batch-test-1")
        assert got is not None
        assert got.cancel_requested_at is not None


class TestSchemaInit:
    def test_schema_created_on_first_open(self, tmp_path: Path) -> None:
        db = tmp_path / "ws.sqlite"
        assert not db.exists()
        WorkspaceMetadata(db)
        assert db.exists()

    def test_two_instances_share_state(self, tmp_path: Path) -> None:
        db = tmp_path / "ws.sqlite"
        ws1 = WorkspaceMetadata(db)
        ws1.create_batch(_record("shared"))
        ws2 = WorkspaceMetadata(db)
        got = ws2.get_batch("shared")
        assert got is not None
        assert got.batch_id == "shared"

    def test_wal_mode_enabled(self, tmp_path: Path) -> None:
        ws = WorkspaceMetadata(tmp_path / "ws.sqlite")
        cur = ws._con.execute("PRAGMA journal_mode")
        mode = cur.fetchone()[0]
        assert mode.lower() == "wal"
