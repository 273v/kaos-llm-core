"""Stable program-hash function for batch resume identity.

Used by Phase 15.2 :func:`batch_run` for resume-correctness: a
batch's checkpoint includes the program hash, and resume refuses to
continue if the current program hash differs.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from kaos_llm_core.programs.envelope.models import ProgramEnvelope


def program_hash(envelope: ProgramEnvelope | dict[str, Any]) -> str:
    """Return a stable sha256 hash of an envelope.

    Both raw dicts and parsed :class:`ProgramEnvelope` instances are
    accepted. Both are normalized through ``ProgramEnvelope`` first
    so the hash always covers the same canonical form (defaults
    filled in, fields sorted) — calling ``program_hash`` on a partial
    dict that lacks optional fields produces the same hash as
    calling it on the parsed envelope.
    """
    if isinstance(envelope, ProgramEnvelope):
        parsed = envelope
    else:
        parsed = ProgramEnvelope.model_validate(envelope)
    env_dict = parsed.model_dump(by_alias=True, exclude_none=False)
    canonical = json.dumps(env_dict, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"
