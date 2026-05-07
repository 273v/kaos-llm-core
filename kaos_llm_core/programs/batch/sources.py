"""Reference :class:`BatchInputSource` implementations.

ListInputSource for in-memory inputs and tests; JsonlInputSource for
streaming files. Cross-module bridges (PdfPagesInputSource,
TabularRowsInputSource) live in :mod:`kaos_llm_core.programs.batch_bridges`
and are gated on optional extras.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable
from pathlib import Path
from typing import Any

from kaos_llm_core.programs.batch._errors import BatchError
from kaos_llm_core.programs.batch.models import BatchItem


class ListInputSource:
    """Wrap an iterable of (inputs, metadata) tuples or BatchItems.

    Used for tests and small batches. Auto-derives ``custom_id`` from
    the program hash if a raw dict is supplied without one.
    """

    def __init__(
        self,
        items: Iterable[BatchItem | dict[str, Any]],
        *,
        program_hash_value: str | None = None,
    ) -> None:
        self._items = list(items)
        self._program_hash = program_hash_value

    def __aiter__(self) -> AsyncIterator[BatchItem]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[BatchItem]:
        for raw in self._items:
            if isinstance(raw, BatchItem):
                yield raw
                continue
            inputs = dict(raw)
            metadata = inputs.pop("__metadata__", {})
            cid = inputs.pop("__custom_id__", None)
            if cid is None:
                if self._program_hash is None:
                    msg = (
                        "ListInputSource: cannot derive custom_id without "
                        "a program_hash. Pass program_hash_value= to the "
                        "constructor or supply __custom_id__ in each input dict."
                    )
                    raise BatchError(msg)
                cid = BatchItem.deterministic_id(self._program_hash, inputs)
            yield BatchItem(custom_id=cid, inputs=inputs, metadata=metadata)

    def describe(self) -> dict[str, Any]:
        return {"type": "list", "n_items_hint": len(self._items)}


def list_input_source(
    items: Iterable[dict[str, Any]],
    *,
    program_hash_value: str,
) -> ListInputSource:
    """Build a ListInputSource that auto-derives custom_ids from inputs."""
    return ListInputSource(items, program_hash_value=program_hash_value)


class JsonlInputSource:
    """Stream BatchItems from a JSONL file.

    Each line is a JSON object. Recognized fields:
      - ``custom_id`` (optional; auto-derived if missing)
      - ``inputs`` (required) — kwargs for ``program.invoke``
      - ``metadata`` (optional) — caller-defined provenance

    For the simplest usage, each line can also be a flat input dict —
    the entire dict becomes the ``inputs``, no ``custom_id`` field
    required (auto-derived from the program hash).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        program_hash_value: str,
    ) -> None:
        self._path = Path(path)
        self._program_hash = program_hash_value

    def __aiter__(self) -> AsyncIterator[BatchItem]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[BatchItem]:
        if not self._path.exists():
            msg = (
                f"JsonlInputSource: file {self._path} does not exist. "
                "Provide a path to an existing JSONL file or use ListInputSource "
                "for in-memory inputs."
            )
            raise BatchError(msg)
        with self._path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    msg = (
                        f"JsonlInputSource: line {line_no + 1} of {self._path} "
                        f"is not valid JSON: {exc}. Use jq or `python -m json.tool` "
                        "to validate the file."
                    )
                    raise BatchError(msg) from exc
                if not isinstance(raw, dict):
                    msg = (
                        f"JsonlInputSource: line {line_no + 1} is a {type(raw).__name__}, "
                        "expected a JSON object."
                    )
                    raise BatchError(msg)
                # Two shapes: structured (with explicit custom_id/inputs/metadata)
                # or flat (the whole dict is the inputs).
                if "inputs" in raw and isinstance(raw["inputs"], dict):
                    inputs = raw["inputs"]
                    metadata = raw.get("metadata", {})
                    cid = raw.get(
                        "custom_id",
                        BatchItem.deterministic_id(self._program_hash, inputs),
                    )
                else:
                    inputs = raw
                    metadata = {"source_line": line_no + 1}
                    cid = BatchItem.deterministic_id(self._program_hash, inputs)
                yield BatchItem(custom_id=cid, inputs=inputs, metadata=metadata)

    def describe(self) -> dict[str, Any]:
        return {"type": "jsonl", "path": str(self._path)}


def jsonl_input_source(
    path: str | Path,
    *,
    program_hash_value: str,
) -> JsonlInputSource:
    """Build a JsonlInputSource for a streaming file-backed batch."""
    return JsonlInputSource(path, program_hash_value=program_hash_value)
