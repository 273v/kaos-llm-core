"""Cross-process telemetry recorder activated by env var.

When ``KAOS_LLM_CORE_RECORDER_DIR`` is set in the environment at
process start, kaos-llm-core monkey-patches
:py:meth:`kaos_llm_core.programs.call.Call._execute` to append each
:class:`~kaos_llm_core.programs._invocation.Invocation` to a
per-PID JSONL file under that directory.

The patch is installed once at import time via
:func:`install_from_env`. It is idempotent (sets a sentinel on
``Call``) and additive — if another recorder (e.g. the kaos-agents
test recorder) is also installed, both fire in order. The patch
never raises into the caller: any write or serialization error is
swallowed, because telemetry must not break execution.

The kaos-agents test recorder
(``kaos-agents/tests/integration/_recorder.py``) is the canonical
parent. It sets the env var before spawning subprocesses so LLM
calls inside ``kaos-agents-serve`` and CLI shells are captured into
the parent's run directory, then stitches the subprocess JSONLs
into its own at test cleanup time. This closes the F2 finding's
"subprocess/MCP-surface recorder gap" for KC6.

Disabling: unset the env var, or set it to empty. Installation is
skipped silently. No behavior change in production environments
that never set the var.

Crash-safe contract (schema_version=4):

* Each ``install_from_env()`` call writes a ``kind="header"`` line
  to its per-PID JSONL **before** the patch goes live. The header
  carries the PID, schema version, and a ``streaming=true`` flag so
  the parent recorder's stitcher can branch on it.
* Each ``_append_invocation`` flushes the Python buffer + calls
  ``os.fsync()`` on the underlying fd **before** returning. This
  bounds the audit-trail loss window to "the line currently being
  written" instead of "everything since process start".
* The file has **no explicit trailer**. Subprocesses can be killed
  abruptly (SIGTERM from the parent, OOM-killer, container restart);
  consumers MUST treat the JSONL as ``header + N invocations`` with
  no end marker. The kaos-agents stitcher's
  ``_collect_subprocess_records`` already filters by
  ``kind=="invocation"`` so old (header-less) files keep working.

Transparency redaction (schema_version=4 — KC16-4):

Mirrors the kaos-agents recorder. Every string value longer than
``KAOS_RECORDER_REDACTION_THRESHOLD`` chars (default 2048) in the
serialized Invocation is replaced with
``<first 200 chars> ... [TRUNCATED N chars; sha256=<16 hex>]``.
Opt-out for local dev via ``KAOS_RECORDER_FULL_TEXT=1``. The
header advertises ``redaction_enabled`` + ``redaction_threshold_chars``
so the parent stitcher can verify what shape it's reading. There
is no trailer here (the subprocess JSONL deliberately has none —
see above), so the per-PID file does NOT carry a ``redacted_count``;
the parent kaos-agents recorder's trailer counts redactions for
its in-process invocations only.

Linux fsync caveats — see the matching note in
``kaos-agents/tests/integration/_recorder.py``: ``TextIOWrapper``
buffers above the OS layer, so we must call ``fh.flush()`` first;
``os.fsync`` wants a fileno, not a file object; on tmpfs / some
network mounts fsync is best-effort.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime
import hashlib
import json
import os
import secrets
import threading
from pathlib import Path
from typing import Any

ENV_VAR = "KAOS_LLM_CORE_RECORDER_DIR"
SENTINEL_ATTR = "_kaos_env_recorder_installed"

# Schema version for the per-PID JSONL files. Bumped to 4 alongside
# the kaos-agents recorder bump for KC16-4 transparency redaction.
SCHEMA_VERSION = 4

# KC16-4 — redaction tunables. Kept in sync with
# ``kaos-agents/tests/integration/_recorder.py`` so a stitcher reading
# a parent-and-subprocess pair gets uniform formatting.
_REDACTION_DEFAULT_THRESHOLD_CHARS = 2048
_REDACTION_PREFIX_CHARS = 200
_REDACTION_HASH_HEX_LEN = 16
_FULL_TEXT_ENV_VAR = "KAOS_RECORDER_FULL_TEXT"
_THRESHOLD_ENV_VAR = "KAOS_RECORDER_REDACTION_THRESHOLD"


def _redaction_config() -> tuple[bool, int]:
    """Read redaction config from env. Returns (enabled, threshold_chars).

    Mirrors the kaos-agents recorder helper of the same name so the
    semantics are identical across recorders. Invalid threshold
    values fall back to the default rather than erroring.
    """
    full_text = os.environ.get(_FULL_TEXT_ENV_VAR, "").strip().lower()
    enabled = full_text not in {"1", "true", "yes", "on"}
    raw = os.environ.get(_THRESHOLD_ENV_VAR, "").strip()
    threshold = _REDACTION_DEFAULT_THRESHOLD_CHARS
    if raw:
        try:
            parsed = int(raw)
            if parsed > 0:
                threshold = parsed
        except ValueError:
            pass
    return enabled, threshold


def _truncate_string(value: str, threshold: int) -> str:
    """Redact a single string. Returns the value unchanged when short."""
    if len(value) <= threshold:
        return value
    full_len = len(value)
    digest = hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()
    prefix = value[:_REDACTION_PREFIX_CHARS]
    return f"{prefix} ... [TRUNCATED {full_len} chars; sha256={digest[:_REDACTION_HASH_HEX_LEN]}]"


def _redact(value: Any, threshold: int) -> Any:
    """Recursively redact long strings in a JSON-ready value.

    Walks ``dict`` and ``list`` containers; coerces nothing else
    (the caller already produced JSON-ready primitives via Pydantic
    ``model_dump`` or ``dataclasses.asdict``).
    """
    if isinstance(value, str):
        return _truncate_string(value, threshold)
    if isinstance(value, list):
        return [_redact(item, threshold) for item in value]
    if isinstance(value, dict):
        return {k: _redact(v, threshold) for k, v in value.items()}
    return value


# Process-wide lock so multiple threads (or multiple Calls running
# concurrently inside an asyncio event loop) don't interleave bytes
# inside the same JSONL line. Multi-process safety comes from the
# per-PID file naming convention.
_WRITE_LOCK = threading.Lock()
_HEADER_WRITTEN: set[str] = set()


def install_from_env() -> bool:
    """Install the Call._execute patch when ``ENV_VAR`` is set.

    Returns True if the patch was installed, False otherwise. Safe
    to call repeatedly — idempotent via a sentinel attribute on
    ``Call``.

    No-op when:
      - ``ENV_VAR`` is unset or empty.
      - The recorder is already installed on ``Call``.
      - The target directory cannot be created.
    """
    out_dir_str = os.environ.get(ENV_VAR, "").strip()
    if not out_dir_str:
        return False

    from kaos_llm_core.programs.call import Call

    if getattr(Call, SENTINEL_ATTR, False):
        return False

    out_dir = Path(out_dir_str)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False

    pid = os.getpid()
    suffix = secrets.token_hex(4)
    jsonl_path = out_dir / f"subprocess-{pid}-{suffix}.jsonl"

    # Write the streaming header BEFORE the patch goes live so any
    # subsequent ``_append_invocation`` call has the header
    # already on disk. Crash between install and first call is
    # then recoverable (the stitcher sees header + 0 invocations).
    _write_header(jsonl_path, pid=pid)

    original_execute = Call._execute

    async def patched_execute(self_call: Any, inputs: dict[str, Any]) -> Any:
        try:
            invocation = await original_execute(self_call, inputs)
        except BaseException as exc:
            partial = getattr(exc, "invocation", None)
            if partial is not None:
                _append_invocation(jsonl_path, partial)
            raise
        _append_invocation(jsonl_path, invocation)
        return invocation

    Call._execute = patched_execute  # ty: ignore[invalid-assignment]
    setattr(Call, SENTINEL_ATTR, True)
    return True


def _write_header(path: Path, *, pid: int) -> None:
    """Write a streaming header to ``path`` if one hasn't been written yet.

    Idempotent within a process — multiple ``install_from_env``
    calls produce only one header line per JSONL. Errors are
    swallowed: telemetry must not break execution.
    """
    key = str(path)
    with _WRITE_LOCK:
        if key in _HEADER_WRITTEN:
            return
        try:
            redaction_enabled, redaction_threshold_chars = _redaction_config()
            header = {
                "kind": "header",
                "pid": pid,
                "start_ts_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                "schema_version": SCHEMA_VERSION,
                "streaming": True,
                "trailer_optional": True,
                "partial_last_line_tolerated": True,
                # KC16-4: advertise redaction policy so the parent
                # stitcher can verify what shape the per-PID file
                # was written in (full-text local dev vs redacted CI).
                "redaction_enabled": redaction_enabled,
                "redaction_threshold_chars": redaction_threshold_chars,
            }
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(header) + "\n")
                f.flush()
                with contextlib.suppress(OSError, ValueError):
                    os.fsync(f.fileno())
            _HEADER_WRITTEN.add(key)
        except Exception:
            pass


def _append_invocation(path: Path, invocation: Any) -> None:
    """Append one JSONL record + flush + fsync. Swallows all errors.

    Crash-safe contract: the write, the ``flush()``, and the
    ``fsync()`` all happen before this function returns. ``TextIOWrapper``
    buffers Python-side writes; ``flush()`` drains that buffer to the
    fd, and ``fsync(fileno())`` asks the OS to push the page cache to
    disk. On Linux this is the strongest portable durability guarantee
    we have access to. It is **not** absolute under hard kill — tmpfs
    and some network filesystems treat fsync as a no-op or only data-
    sync without the directory entry — but it is dramatically better
    than the pre-streaming "write everything at exit" pattern.
    """
    try:
        record = _serialize(invocation)
        # KC16-4: redact long string payloads inside the serialized
        # record. ``_serialize`` produces JSON-ready primitives so the
        # walk only needs to handle dict/list/str. Metadata fields
        # (model, invocation_id, pid, usage numbers, error class) are
        # short and pass through untouched. Redaction config is read
        # per-call from env so a long-lived subprocess sees toggle
        # changes too — matters more in dev than CI.
        enabled, threshold = _redaction_config()
        redacted = _redact(record, threshold) if enabled else record
        line = json.dumps(redacted) + "\n"
        # Combined `with` keeps SIM117 happy; we hold the process-wide
        # write lock for the full open/write/flush/fsync to prevent
        # interleaving between concurrent in-process Calls.
        with _WRITE_LOCK, path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            # OSError: fsync not supported (some tmpfs).
            # ValueError: fd closed during shutdown race.
            with contextlib.suppress(OSError, ValueError):
                os.fsync(f.fileno())
    except Exception:
        # Intentional: a recorder that fails on bad data would break
        # production flows that happen to set the env var. Drop the
        # record silently.
        pass


def _serialize(invocation: Any) -> dict[str, Any]:
    """Convert an Invocation to a JSON-ready dict.

    Minimal serialization — full ExecutionTrace dump via
    ``model_dump(mode="json")`` when available. Mirrors the shape
    the kaos-agents test recorder writes so stitching is a straight
    concatenation, no schema reshape.
    """
    out: dict[str, Any] = {
        "kind": "invocation",
        "pid": os.getpid(),
        "invocation_id": getattr(invocation, "id", None),
        "model": getattr(invocation, "model", None),
    }
    err = getattr(invocation, "error", None)
    out["error"] = repr(err) if err is not None else None

    trace = getattr(invocation, "trace", None)
    if trace is not None and hasattr(trace, "model_dump"):
        try:
            out["trace"] = trace.model_dump(mode="json")
        except Exception:
            out["trace"] = None
    else:
        out["trace"] = None

    out["usage"] = _coerce_to_dict(getattr(invocation, "usage", None))

    output = getattr(invocation, "output", None)
    if output is not None and hasattr(output, "model_dump"):
        try:
            out["output"] = output.model_dump(mode="json")
        except Exception:
            out["output"] = repr(output)
    elif output is not None:
        out["output"] = repr(output)
    else:
        out["output"] = None

    return out


def _coerce_to_dict(obj: Any) -> Any:
    """Best-effort conversion to a JSON-friendly dict.

    Order: Pydantic model_dump → dataclass → public attrs from
    __dict__ → public attrs read individually for slotted classes.
    Returns None when nothing usable comes out.
    """
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")
        except Exception:
            pass
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        try:
            return dataclasses.asdict(obj)
        except Exception:
            # Some dataclasses contain unpicklable refs; fall through.
            pass
    if hasattr(obj, "__dict__"):
        try:
            return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
        except Exception:
            pass
    # Slotted non-dataclass: enumerate __slots__ if present.
    slots = getattr(type(obj), "__slots__", None)
    if slots:
        result: dict[str, Any] = {}
        for name in slots:
            if name.startswith("_"):
                continue
            try:
                result[name] = getattr(obj, name)
            except AttributeError:
                continue
        return result if result else None
    return None
