"""Batch error types — leaf module, no internal batch deps."""

from __future__ import annotations

from kaos_llm_core.errors import KaosLLMCoreError


class BatchError(KaosLLMCoreError):
    """Base class for batch-execution errors."""


class BatchResumeMismatchError(BatchError):
    """``resume=True`` failed because the existing log was for a different program."""


class BatchStopAfterNError(BatchError):
    """``error_policy='stop_after_n'`` triggered: too many per-item failures."""


class BatchUnsupportedBackendError(BatchError):
    """The configured VFS backend does not support the batch log requirements."""
