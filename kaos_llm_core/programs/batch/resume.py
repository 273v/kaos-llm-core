"""Resume scan over an existing JSONL batch log."""

from __future__ import annotations

import json
from pathlib import Path

from kaos_core.logging import get_logger

from kaos_llm_core.programs.batch._errors import BatchResumeMismatchError

logger = get_logger(__name__)


def _scan_existing_log(
    log_path: Path,
    expected_program_hash: str,
) -> tuple[set[str], int, int, int, float, int]:
    """Read an existing JSONL log and return (skip_set, n_succeeded, n_errored,
    n_skipped, cost_so_far, tokens_so_far).

    Validates the header's program_hash matches the expected one. Raises
    :class:`BatchResumeMismatchError` if not.
    """
    skip_set: set[str] = set()
    n_succeeded = 0
    n_errored = 0
    cost_so_far = 0.0
    tokens_so_far = 0

    if not log_path.exists():
        return skip_set, 0, 0, 0, 0.0, 0

    with log_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # Tolerate a partial last line from a crashed write — it
                # was never fsync'd. Ignore and continue.
                logger.warning(
                    "batch_run resume: ignoring corrupt line %d in %s",
                    line_no + 1,
                    log_path,
                )
                continue
            kind = record.get("kind")
            if kind == "header":
                stored_hash = record.get("program_hash")
                if stored_hash and stored_hash != expected_program_hash:
                    msg = (
                        f"Cannot resume {log_path}: program hash mismatch. "
                        f"Existing log was for program {stored_hash}, current "
                        f"program is {expected_program_hash}. Either start a "
                        f"fresh run with a new output_dir, or restore the "
                        "original program."
                    )
                    raise BatchResumeMismatchError(msg)
            elif kind == "item":
                cid = record.get("custom_id")
                if cid:
                    skip_set.add(cid)
                if record.get("status") == "success":
                    n_succeeded += 1
                else:
                    n_errored += 1
                usage = record.get("usage", {})
                cost_so_far += float(usage.get("cost_usd", 0.0) or 0.0)
                tokens_so_far += int(usage.get("total_tokens", 0) or 0)

    return skip_set, n_succeeded, n_errored, len(skip_set), cost_so_far, tokens_so_far
