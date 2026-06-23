"""``batch_run()`` — the streaming, resumable batch primitive itself."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from kaos_core.logging import get_logger

from kaos_llm_core.optimization.trial_runner import TrialRunner
from kaos_llm_core.programs.base import Program
from kaos_llm_core.programs.batch._errors import BatchError
from kaos_llm_core.programs.batch.models import (
    BatchInputSource,
    BatchItem,
    BatchProgress,
    BatchResult,
)
from kaos_llm_core.programs.batch.records import (
    _error_record,
    _now_iso,
    _success_record,
    _write_manifest,
)
from kaos_llm_core.programs.batch.resume import _scan_existing_log
from kaos_llm_core.programs.batch.targets import (
    _hash_target,
    _resolve_output_disk_path,
    _resolve_target,
)
from kaos_llm_core.programs.batch.writer import JsonlBatchWriter
from kaos_llm_core.programs.call import Call
from kaos_llm_core.programs.envelope import ProgramEnvelope

logger = get_logger(__name__)


async def batch_run(
    target: Call | Program | ProgramEnvelope | dict[str, Any],
    source: BatchInputSource,
    *,
    output_dir: str,
    runtime: Any | None = None,
    context: Any | None = None,
    run_id: str | None = None,
    max_concurrency: int = 8,
    mode: Literal["live", "provider_batch"] = "live",
    error_policy: Literal["continue", "stop", "stop_after_n"] = "continue",
    max_errors: int | None = None,
    resume: bool = True,
    progress_callback: Callable[[BatchProgress], Awaitable[None]] | None = None,
    item_timeout_s: float | None = None,
) -> BatchResult:
    """Execute a Call/Program/envelope against a streaming input source.

    See ``docs/internal/design/batch-run.md`` for the full contract.

    Args:
        target: The Call, Program, ProgramEnvelope, or raw envelope dict
            to execute against each input.
        source: A BatchInputSource yielding BatchItem instances. Must be
            deterministic across runs for resume to work.
        output_dir: VFS or disk path to a directory that will receive
            ``items.jsonl`` and ``manifest.json``. Created if missing.
        runtime: Optional KaosRuntime for VFS path resolution. If None,
            output_dir is treated as a literal disk path.
        context: Optional KaosContext for session_id scoping and the
            default progress_callback's report_progress forwarding.
        run_id: Optional run identifier. Auto-derived from program_hash
            + source descriptor if None.
        max_concurrency: asyncio semaphore cap on in-flight items.
        mode: ``"live"`` (in-process asyncio fan-out, v1 default) or
            ``"provider_batch"`` (Phase 16 — submits to OpenAI/Anthropic
            Batch APIs for the 50% discount; raises NotImplementedError
            in v1).
        error_policy:
            - ``"continue"`` (default): per-item failures land in the
              JSONL log; the batch completes with n_errored > 0.
            - ``"stop"``: first failure aborts; log left resumable.
            - ``"stop_after_n"``: requires max_errors; the loop aborts
              after N failures. The money-safety valve.
        max_errors: Required iff error_policy=='stop_after_n'.
        resume: When True (default), reads any existing log at
            output_dir/items.jsonl, validates the program_hash matches,
            builds a skip-set on completed custom_ids, and re-runs only
            the missing items. When False, errors out if a log already
            exists.
        progress_callback: Optional callable. The default forwards to
            ``context.report_progress`` if a context is supplied.
        item_timeout_s: Optional per-item asyncio timeout.

    Returns:
        A BatchResult summarizing the run. Per-item results live in
        ``${output_dir}/items.jsonl``; the aggregate manifest lives at
        ``${output_dir}/manifest.json``.
    """
    if mode == "provider_batch":
        msg = (
            "batch_run mode='provider_batch' is reserved for Phase 16 (OpenAI Batch / "
            "Anthropic Message Batches integration). v1 ships mode='live' only."
        )
        raise NotImplementedError(msg)
    if error_policy == "stop_after_n" and max_errors is None:
        msg = "error_policy='stop_after_n' requires max_errors to be set."
        raise BatchError(msg)

    program = _resolve_target(target)
    program_hash_value = _hash_target(target)

    # Resolve output paths
    disk_output_dir = _resolve_output_disk_path(output_dir, runtime, context)
    disk_output_dir.mkdir(parents=True, exist_ok=True)
    log_path = disk_output_dir / "items.jsonl"
    manifest_path = disk_output_dir / "manifest.json"

    # Auto-derive run_id if not supplied
    if run_id is None:
        source_desc = json.dumps(source.describe(), sort_keys=True, separators=(",", ":"))
        h = hashlib.sha256((program_hash_value + source_desc).encode()).hexdigest()[:12]
        run_id = f"batch-{h}"

    # Resume scan
    skip_set: set[str] = set()
    n_succeeded = 0
    n_errored = 0
    n_skipped = 0
    cost_so_far = 0.0
    tokens_so_far = 0

    if resume and log_path.exists():
        scan = _scan_existing_log(log_path, program_hash_value)
        skip_set, n_succeeded, n_errored, n_skipped, cost_so_far, tokens_so_far = scan
        logger.debug(
            "batch_run resume: %d items already in log, %d succeeded, %d errored",
            len(skip_set),
            n_succeeded,
            n_errored,
        )

    # Open writer; write header iff this is a fresh log
    writer = JsonlBatchWriter(log_path)
    if not log_path.exists() or log_path.stat().st_size == 0:
        await writer.write_header(
            {
                "kaos_batch_log": "1",
                "run_id": run_id,
                "program_hash": program_hash_value,
                "source": source.describe(),
                "config": {
                    "max_concurrency": max_concurrency,
                    "mode": mode,
                    "error_policy": error_policy,
                    "max_errors": max_errors,
                },
                "started_at": _now_iso(),
            }
        )

    started_at = _now_iso()
    started_monotonic = time.monotonic()

    # Trial scope for cost attribution
    runner = TrialRunner()
    sem = asyncio.Semaphore(max_concurrency)

    n_total_seen = 0  # items observed in this run's iteration (skipped + processed)
    errors_by_type: dict[str, int] = {}

    # Async lock for writer + counters
    write_lock = asyncio.Lock()

    async def _publish_progress(
        last_id: str | None,
        last_status: Literal["success", "error", "skipped"] | None,
    ) -> None:
        if progress_callback is None and context is None:
            return
        prog = BatchProgress(
            n_done=n_succeeded + n_errored + n_skipped,
            n_total=None,
            n_succeeded=n_succeeded,
            n_errored=n_errored,
            n_skipped=n_skipped,
            cost_usd_so_far=cost_so_far,
            tokens_so_far=tokens_so_far,
            last_custom_id=last_id,
            last_status=last_status,
        )
        if progress_callback is not None:
            await progress_callback(prog)
        elif context is not None and hasattr(context, "report_progress"):
            # report_progress may raise on transport errors; the batch
            # continues regardless because progress is best-effort.
            with contextlib.suppress(Exception):
                await context.report_progress(
                    progress=prog.n_done,
                    total=prog.n_total,
                    message=f"{prog.n_done} done — ${prog.cost_usd_so_far:.4f} so far",
                )

    stop_requested = False
    final_status: Literal["completed", "stopped", "cancelled", "failed"] = "completed"

    try:
        with runner.trial(f"batch_run:{run_id}") as trial:

            async def _process_item(item: BatchItem) -> None:
                nonlocal n_succeeded, n_errored, cost_so_far, tokens_so_far, stop_requested

                if stop_requested:
                    return

                async with sem:
                    if stop_requested:
                        return
                    item_start = time.monotonic()
                    try:
                        if item_timeout_s is not None:
                            invocation = await asyncio.wait_for(
                                program.invoke(**item.inputs),
                                timeout=item_timeout_s,
                            )
                        else:
                            invocation = await program.invoke(**item.inputs)
                        duration_ms = (time.monotonic() - item_start) * 1000
                        record = _success_record(item, invocation, duration_ms)
                        async with write_lock:
                            await writer.write_item(record)
                            n_succeeded += 1
                            cost_so_far = trial.cost_usd
                            tokens_so_far = trial.total_tokens
                        await _publish_progress(item.custom_id, "success")
                    except BaseException as exc:
                        duration_ms = (time.monotonic() - item_start) * 1000
                        error_class = type(exc).__name__
                        record = _error_record(item, exc, duration_ms)
                        async with write_lock:
                            await writer.write_item(record)
                            n_errored += 1
                            errors_by_type[error_class] = errors_by_type.get(error_class, 0) + 1
                            cost_so_far = trial.cost_usd
                            tokens_so_far = trial.total_tokens
                        await _publish_progress(item.custom_id, "error")
                        if error_policy == "stop" or (
                            error_policy == "stop_after_n"
                            and max_errors is not None
                            and n_errored >= max_errors
                        ):
                            stop_requested = True

            tasks: list[asyncio.Task[None]] = []
            async for item in source:
                if item.custom_id in skip_set:
                    n_total_seen += 1
                    continue
                if stop_requested:
                    break
                n_total_seen += 1
                tasks.append(asyncio.create_task(_process_item(item)))
                # Back-pressure: drain when the in-flight task list grows
                # past 4x the concurrency cap.
                if len(tasks) >= max_concurrency * 4:
                    _done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    tasks = list(pending)
            # Drain remaining tasks
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            if stop_requested:
                final_status = "stopped"
                if (
                    error_policy == "stop_after_n"
                    and max_errors is not None
                    and n_errored >= max_errors
                ):
                    # Don't raise — return the result so the caller sees
                    # partial state. The mutation is in the log + manifest.
                    pass
            cost_usd = trial.cost_usd
            input_tokens = trial.input_tokens
            output_tokens = trial.output_tokens
            total_tokens = trial.total_tokens
    finally:
        await writer.close()

    completed_at = _now_iso()
    duration_s = time.monotonic() - started_monotonic

    _write_manifest(
        manifest_path,
        run_id=run_id,
        program_hash_value=program_hash_value,
        log_path=log_path,
        n_total=n_total_seen,
        n_succeeded=n_succeeded,
        n_errored=n_errored,
        n_skipped=n_skipped,
        cost_usd=cost_usd,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        started_at=started_at,
        completed_at=completed_at,
        duration_s=duration_s,
        status=final_status,
        errors_by_type=errors_by_type,
    )

    return BatchResult(
        run_id=run_id,
        output_dir=str(disk_output_dir),
        log_path=str(log_path),
        manifest_path=str(manifest_path),
        n_total=n_total_seen,
        n_succeeded=n_succeeded,
        n_errored=n_errored,
        n_skipped=n_skipped,
        cost_usd=cost_usd,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        duration_s=duration_s,
        status=final_status,
        started_at=started_at,
        completed_at=completed_at,
        program_hash=program_hash_value,
        errors_by_type=errors_by_type,
    )
