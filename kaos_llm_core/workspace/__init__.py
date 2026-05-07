"""Phase 15.3 workspace metadata for kaos-llm-core.

The workspace is a per-VFS SQLite database that persists batch job
state across MCP requests. It is narrowly scoped to batch jobs only —
the broader programs/datasets/runs concepts belong to the future
kaos-agents and are deferred per the §5.9 scoping decision in the
canonical roadmap.

Sub-design: ``docs/internal/design/batch-workspace-schema.md``.
"""

from __future__ import annotations

from kaos_llm_core.workspace.metadata import (
    BatchAlreadyExistsError,
    BatchRecord,
    WorkspaceMetadata,
    WorkspaceUnsupportedBackendError,
)

__all__ = [
    "BatchAlreadyExistsError",
    "BatchRecord",
    "WorkspaceMetadata",
    "WorkspaceUnsupportedBackendError",
]
