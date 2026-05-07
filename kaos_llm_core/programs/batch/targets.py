"""Target normalization helpers for ``batch_run``.

The batch primitive accepts a Call, Program, ProgramEnvelope, or raw
envelope dict. These helpers normalize all four shapes into a runnable
program + a stable resume hash + a real disk path for the JSONL log.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

from kaos_core.logging import get_logger

from kaos_llm_core.programs.base import Program
from kaos_llm_core.programs.batch._errors import (
    BatchError,
    BatchUnsupportedBackendError,
)
from kaos_llm_core.programs.call import Call
from kaos_llm_core.programs.envelope import (
    ProgramEnvelope,
    from_envelope,
    program_hash,
)

logger = get_logger(__name__)


def _hash_target(target: Call | Program | ProgramEnvelope | dict[str, Any]) -> str:
    """Return a stable hash for any target shape that batch_run accepts.

    Resume identity contract: a target built via ``from_envelope(env)``
    must hash the same as the source envelope so a batch built from the
    Python API and a batch built from the MCP wire (which always goes
    through an envelope) share resume state. ``_EnvelopeProgram`` exposes
    its envelope via the ``envelope`` property; we hash that.
    """
    if isinstance(target, ProgramEnvelope):
        return program_hash(target)
    if isinstance(target, dict):
        # ``program_hash`` accepts both ProgramEnvelope and dict; the dict
        # branch is the wire path. ty narrows dict to dict[Unknown, Unknown]
        # but the runtime overload accepts dict[str, Any], so a cast is
        # the simplest fix here.
        return program_hash(cast("dict[str, Any]", target))
    # Programs built via from_envelope expose the original envelope so
    # the hash matches the wire-side program_hash call. Capture the
    # value through the isinstance narrow so ty sees a ProgramEnvelope.
    candidate_envelope = getattr(target, "envelope", None)
    if isinstance(candidate_envelope, ProgramEnvelope):
        return program_hash(candidate_envelope)
    # Call / Program: hash the type name + serialized learnable state.
    state: dict[str, Any] = {"_type": type(target).__name__}
    if hasattr(target, "get_learnable_state"):
        try:
            state["learnable"] = target.get_learnable_state()
        except Exception:  # pragma: no cover - defensive
            logger.debug("Failed to compute learnable hash", exc_info=True)
            state["learnable"] = None
    canonical = json.dumps(state, sort_keys=True, separators=(",", ":"), default=str)
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _resolve_target(
    target: Call | Program | ProgramEnvelope | dict[str, Any],
) -> Program | Call:
    if isinstance(target, (Call, Program)):
        return target
    if isinstance(target, (ProgramEnvelope, dict)):
        return from_envelope(target)
    msg = (
        f"batch_run: target must be a Call, Program, ProgramEnvelope, or "
        f"envelope dict. Got {type(target).__name__}."
    )
    raise BatchError(msg)


def _resolve_output_disk_path(
    output_dir: str,
    runtime: Any | None,
    context: Any | None,
) -> Path:
    """Map a VFS path to a disk path the JSONL writer can append to.

    v1 limitation: requires the disk backend. Memory / S3 backends raise
    :class:`BatchUnsupportedBackendError`. Phase 16 should add
    ``VirtualFileSystem.append()`` so other backends can ship.
    """
    # Fast-path: if output_dir is already a real disk path (absolute path that
    # exists or is creatable), use it directly. This makes batch_run usable
    # in unit tests without a runtime.
    if runtime is None or not hasattr(runtime, "vfs"):
        return Path(output_dir).expanduser().resolve()
    try:
        session_id = getattr(context, "session_id", "default") if context is not None else "default"
        disk_path = runtime.vfs.resolve_disk_path(output_dir, context_id=session_id)
        return Path(disk_path)
    except (AttributeError, NotImplementedError) as exc:
        msg = (
            f"batch_run: VFS backend does not support resolve_disk_path "
            f"({exc}). v1 batch_run requires the disk backend for the JSONL "
            f"checkpoint log. Workaround: pass an absolute disk path as "
            "output_dir, or configure the runtime with the disk backend."
        )
        raise BatchUnsupportedBackendError(msg) from exc
