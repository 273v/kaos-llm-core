"""JSONL export and import for execution traces."""

from __future__ import annotations

from pathlib import Path

from kaos_llm_core.observability.traces import ExecutionTrace


def export_jsonl(traces: list[ExecutionTrace], path: str | Path) -> None:
    """Export execution traces to a JSONL file.

    Each line is a JSON-serialized trace. Appends to existing file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        for trace in traces:
            f.write(trace.model_dump_json() + "\n")


def load_jsonl(path: str | Path) -> list[ExecutionTrace]:
    """Load execution traces from a JSONL file."""
    path = Path(path)
    if not path.exists():
        return []

    traces: list[ExecutionTrace] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(ExecutionTrace.model_validate_json(line))
    return traces
