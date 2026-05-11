"""Durability tests for the kaos-llm-core cross-process env recorder.

Sprint-3 #8 (transparency lens): the env_recorder used to write
each line at function-exit only. Under SIGTERM / OOM-kill /
container-restart the per-PID JSONL had nothing for any in-flight
Call.

Schema-v3 streaming contract verified here:

1. ``install_from_env()`` writes a ``kind="header"`` line + fsync
   BEFORE returning, so a SIGTERM-after-install crash still leaves
   a parseable file on disk.
2. Each ``_append_invocation`` flushes + fsyncs before returning.
3. A real ``multiprocessing.Process`` SIGTERM'd after one
   ``Call._execute`` leaves at minimum the header + 1 invocation
   line. The file has no explicit trailer (subprocesses never
   write one — the parent kaos-agents recorder stitches them).
4. Concurrent writers across multiple subprocesses each land in
   their own per-PID file without interleaving.
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
import signal
import time
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Subprocess workers (top-level so multiprocessing can pickle them).
# ---------------------------------------------------------------------------


def _run_env_recorder_child(
    recorder_dir_str: str,
    sentinel_path: str,
    *,
    sleep_after_call_seconds: float = 30.0,
) -> None:
    """Child target: install env recorder, run ONE fake Call, idle.

    The kaos_llm_core ``__init__.py`` auto-installs ``install_from_env``
    on first import, so we set the env var BEFORE importing
    kaos_llm_core and let the auto-install path do the patching.

    Streaming contract: header line is written + fsync'd inside
    ``install_from_env`` (i.e. before any Call). Each invocation
    line is written + fsync'd inside ``_append_invocation`` (i.e.
    before the caller observes the patched ``_execute`` returning).
    """
    os.environ["KAOS_LLM_CORE_RECORDER_DIR"] = recorder_dir_str

    # Importing kaos_llm_core fires the auto-install path.
    import kaos_llm_core  # noqa: F401
    from kaos_llm_core.observability.env_recorder import SENTINEL_ATTR
    from kaos_llm_core.programs._invocation import Invocation, TokenUsage
    from kaos_llm_core.programs.call import Call

    if not getattr(Call, SENTINEL_ATTR, False):
        # Auto-install didn't fire — that's a setup error.
        Path(sentinel_path).write_text("install-failed", encoding="utf-8")
        return

    # The env_recorder's wrapper closes over the **original**
    # ``Call._execute`` captured at install time. We need that
    # captured original to be our fake. Reach into the wrapper's
    # closure and swap it.
    async def fake_execute(self_call: Any, inputs: dict[str, Any]) -> Any:
        return Invocation(
            id="inv-env-durability",
            client=None,
            model="function-env-durability",
            context=None,
            extras={},
            output={"answer": "streamed-env"},
            trace=None,
            usage=TokenUsage(input_tokens=7, output_tokens=4, total_tokens=11, cost_usd=0.0),
            error=None,
        )

    wrapper = Call._execute
    swapped = False
    if hasattr(wrapper, "__closure__") and wrapper.__closure__ is not None:
        for cell in wrapper.__closure__:
            try:
                cur = cell.cell_contents
            except ValueError:
                continue
            if callable(cur):
                name = getattr(cur, "__qualname__", "") or getattr(cur, "__name__", "")
                if "Call._execute" in name or name == "_execute":
                    cell.cell_contents = fake_execute  # type: ignore[misc]
                    swapped = True
                    break
    if not swapped:
        Path(sentinel_path).write_text("swap-failed", encoding="utf-8")
        return

    async def body() -> None:
        # Trigger one invocation through the env recorder's wrapper.
        inv = await Call._execute(None, {})  # ty: ignore[invalid-argument-type]
        assert inv is not None
        Path(sentinel_path).write_text("ok", encoding="utf-8")
        await asyncio.sleep(sleep_after_call_seconds)

    asyncio.run(body())


def _run_install_only_child(recorder_dir_str: str, sentinel_path: str) -> None:
    """Child target: trigger env-recorder auto-install, touch sentinel, idle."""
    os.environ["KAOS_LLM_CORE_RECORDER_DIR"] = recorder_dir_str

    import kaos_llm_core  # noqa: F401  -- auto-installs env_recorder
    from kaos_llm_core.observability.env_recorder import SENTINEL_ATTR
    from kaos_llm_core.programs.call import Call

    installed = bool(getattr(Call, SENTINEL_ATTR, False))
    Path(sentinel_path).write_text("installed" if installed else "failed", encoding="utf-8")
    time.sleep(30.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wait_for_path(path: Path, timeout_s: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.05)
    return False


def _spawn(method: str, target: Any, args: tuple[Any, ...]) -> multiprocessing.Process:
    ctx = multiprocessing.get_context(method)
    proc: multiprocessing.Process = ctx.Process(target=target, args=args)  # ty: ignore[unresolved-attribute]
    proc.start()
    return proc


def _require_pid(proc: multiprocessing.Process) -> int:
    """``Process.pid`` is ``int | None`` until ``start()`` returns.

    All callers here have already ``start()``ed; this assertion
    makes the invariant explicit for the type checker.
    """
    pid = proc.pid
    assert pid is not None, "process has not been started yet"
    return pid


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEnvRecorderHeaderOnInstall:
    """The header must hit disk inside ``install_from_env``."""

    def test_header_present_before_any_call(self, tmp_path: Path) -> None:
        """SIGTERM after install (but before any Call) leaves a parseable file."""
        recorder_dir = tmp_path / "recorder"
        recorder_dir.mkdir()
        sentinel = tmp_path / "installed.sentinel"

        proc = _spawn("spawn", _run_install_only_child, (str(recorder_dir), str(sentinel)))
        try:
            assert _wait_for_path(sentinel), "install_from_env never completed"
            assert sentinel.read_text() == "installed"

            jsonl_files = list(recorder_dir.glob("subprocess-*.jsonl"))
            assert len(jsonl_files) == 1, f"expected 1 jsonl, got {jsonl_files}"
            content = jsonl_files[0].read_text(encoding="utf-8")
            lines = [ln for ln in content.split("\n") if ln.strip()]
            assert len(lines) == 1, f"expected header-only, got {len(lines)} lines"
            header = json.loads(lines[0])
            assert header["kind"] == "header"
            assert header["streaming"] is True
            # Schema bumped to 4 in KC16-4 alongside redaction.
            assert header["schema_version"] == 4
            assert "pid" in header
            assert header["pid"] == _require_pid(proc)
        finally:
            if proc.is_alive():
                os.kill(_require_pid(proc), signal.SIGTERM)
            proc.join(timeout=10.0)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5.0)


class TestEnvRecorderInvocationOnCompletion:
    """A completed Invocation must hit disk before the next line of the body runs."""

    def test_invocation_persists_across_sigterm(self, tmp_path: Path) -> None:
        """Acceptance gate: header + 1 invocation survives SIGTERM."""
        recorder_dir = tmp_path / "recorder"
        recorder_dir.mkdir()
        sentinel = tmp_path / "after-call.sentinel"

        proc = _spawn(
            "spawn",
            _run_env_recorder_child,
            (str(recorder_dir), str(sentinel)),
        )
        try:
            assert _wait_for_path(sentinel), "child failed to write post-call sentinel"
            assert sentinel.read_text() == "ok", "child install/setup failed"

            time.sleep(0.05)  # tmpfs scheduling cushion

            os.kill(_require_pid(proc), signal.SIGTERM)
            proc.join(timeout=10.0)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5.0)
                pytest.fail("child did not exit on SIGTERM")

            jsonl_files = list(recorder_dir.glob("subprocess-*.jsonl"))
            assert len(jsonl_files) == 1, f"expected 1 jsonl, got {jsonl_files}"
            raw = jsonl_files[0].read_text(encoding="utf-8")
            lines = [ln for ln in raw.split("\n") if ln.strip()]

            # Acceptance: header + 1 invocation, no trailer.
            assert len(lines) >= 2, f"expected header + 1 invocation, got {lines}"

            header = json.loads(lines[0])
            assert header["kind"] == "header"
            assert header["streaming"] is True
            # Schema bumped to 4 in KC16-4 alongside redaction.
            assert header["schema_version"] == 4
            assert header["pid"] == _require_pid(proc)

            # Every subsequent line should be well-formed JSON.
            for i, line in enumerate(lines[1:], start=1):
                parsed = json.loads(line)
                assert "kind" in parsed, f"line {i} not well-formed: {line!r}"

            invocations = [
                json.loads(ln) for ln in lines[1:] if json.loads(ln).get("kind") == "invocation"
            ]
            assert len(invocations) >= 1, f"expected >=1 invocation, got 0 in {lines!r}"
            inv = invocations[0]
            assert inv["model"] == "function-env-durability"
            assert inv["invocation_id"] == "inv-env-durability"
            assert inv["usage"]["total_tokens"] == 11

            # Env recorder writes NO trailer — that's the parent's job.
            trailer_lines = [ln for ln in lines if json.loads(ln).get("kind") == "trailer"]
            assert trailer_lines == [], "env_recorder must not write trailer lines"
        finally:
            if proc.is_alive():
                os.kill(_require_pid(proc), signal.SIGKILL)
                proc.join(timeout=5.0)


class TestConcurrentSubprocessWriters:
    """Per-PID file naming prevents inter-process interleaving."""

    def test_two_pids_get_two_files(self, tmp_path: Path) -> None:
        recorder_dir = tmp_path / "recorder"
        recorder_dir.mkdir()
        sentinel_a = tmp_path / "a.sentinel"
        sentinel_b = tmp_path / "b.sentinel"

        proc_a = _spawn("spawn", _run_install_only_child, (str(recorder_dir), str(sentinel_a)))
        proc_b = _spawn("spawn", _run_install_only_child, (str(recorder_dir), str(sentinel_b)))
        try:
            assert _wait_for_path(sentinel_a)
            assert _wait_for_path(sentinel_b)

            jsonl_files = sorted(recorder_dir.glob("subprocess-*.jsonl"))
            assert len(jsonl_files) == 2, f"expected 2 per-PID files, got {jsonl_files}"

            pids_seen: set[int] = set()
            for jp in jsonl_files:
                line = jp.read_text(encoding="utf-8").splitlines()[0]
                header = json.loads(line)
                assert header["kind"] == "header"
                pids_seen.add(int(header["pid"]))

            assert pids_seen == {proc_a.pid, proc_b.pid}
        finally:
            for p in (proc_a, proc_b):
                if p.is_alive():
                    os.kill(_require_pid(p), signal.SIGTERM)
                p.join(timeout=5.0)
                if p.is_alive():
                    p.kill()
                    p.join(timeout=2.0)
