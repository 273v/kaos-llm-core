"""Program v3 envelope — JSON-declarable multi-step LLM program composition.

Phase 15.1. The envelope is the JSON-declarable equivalent of
``class MyProgram(Program): def forward(self, ...)``: an MCP-driven
agent (Claude Code, Codex CLI) constructs a v3 envelope, writes it
to the VFS via ``kaos-core-vfs-write``, and executes it via the
``kaos-llm-core-program-execute`` MCP tool.

Sub-design: ``docs/internal/design/program-envelope-v3.md``.

Constraints (load-bearing, from the sub-design):

- **Linear DAG only.** No ``if`` / ``for`` / ``branch`` constructs.
  Loops are first-class via the existing ``react`` / ``refine`` /
  ``best_of_n`` step kinds (and the Phase 16.2
  ``multi_chain_comparison`` / ``program_of_thought`` step kinds).
- **No embedded callables ever.** The envelope is data only.
- **Closed ``kind:`` enum.** Eight step kinds; unknown kinds fail
  at envelope validation, not at execution.
- **Symbolic clients.** Credentials never appear in the envelope —
  they resolve from ``KaosLLMSettings`` at run time.
- **DSPy-style tunable / author split.** Every step has a
  ``tunable: {instruction, demos, hyperparameters, codec, model}``
  sub-object reserved from day one. Optimizers write here without
  touching author fields.
- **Self-describing.** Top-level ``inputs:`` / ``output:`` shapes
  let kaos-mcp auto-generate an MCP tool from any envelope.

The envelope IS the program — there is no server-side state.
``from_envelope(env)`` builds a real ``Program`` instance whose
``forward()`` walks the steps in declaration order, resolves inputs
against the live context, and assembles outputs via the top-level
``output:`` mapping.

Package layout (Phase 16.5 — was a single 1012-LOC file pre-split):

  - :mod:`._constants` — ``StepKind`` Literal, ``SUPPORTED_CAPABILITIES``,
    naming regexes.
  - :mod:`._errors` — :class:`ProgramEnvelopeError`.
  - :mod:`.models` — Pydantic models (``InputSpec``, ``ClientSpec``,
    ``Tunable``, ``JudgeSpec``, ``ToolSpec``, ``StepOutputField``,
    ``EnvelopeStep``, ``ProgramEnvelope``).
  - :mod:`.validation` — Cross-step validation helpers.
  - :mod:`.hashing` — :func:`program_hash`.
  - :mod:`.substitution` — JSON-pointer resolution.
  - :mod:`.signatures` — Per-step Signature builder.
  - :mod:`.builders` — Step builder registry + dispatch.
  - :mod:`.runtime` — :class:`_EnvelopeProgram`.

Public API is re-exported from this :mod:`__init__` so callers can
keep importing from ``kaos_llm_core.programs.envelope`` unchanged.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from kaos_llm_core.programs.base import Program
from kaos_llm_core.programs.envelope._constants import (
    SUPPORTED_CAPABILITIES,
    StepKind,
)
from kaos_llm_core.programs.envelope._errors import ProgramEnvelopeError
from kaos_llm_core.programs.envelope.builders import _build_step_program
from kaos_llm_core.programs.envelope.hashing import program_hash
from kaos_llm_core.programs.envelope.models import (
    ClientSpec,
    EnvelopeStep,
    InputSpec,
    JudgeSpec,
    ProgramEnvelope,
    StepOutputField,
    ToolSpec,
    Tunable,
)
from kaos_llm_core.programs.envelope.runtime import _EnvelopeProgram
from kaos_llm_core.programs.envelope.substitution import (
    _resolve_output_mapping,
    _resolve_pointer,
    _resolve_step_inputs,
)
from kaos_llm_core.programs.envelope.validation import (
    _validate_pointer,
    _validate_type_refs,
)


def from_envelope(envelope: ProgramEnvelope | dict[str, Any]) -> Program:
    """Build an executable :class:`Program` from a v3 envelope.

    Accepts either a parsed :class:`ProgramEnvelope` or a raw dict.
    Raises :class:`ProgramEnvelopeError` on validation failure with
    an agent-friendly recovery message.

    Example::

        envelope = json.loads(envelope_json)
        program = from_envelope(envelope)
        invocation = await program.invoke(text="...")

    See ``docs/internal/design/program-envelope-v3.md`` for the
    full envelope spec.
    """
    if isinstance(envelope, dict):
        try:
            parsed = ProgramEnvelope.model_validate(envelope)
        except ValidationError as exc:
            msg = (
                f"Envelope failed schema validation: {exc.error_count()} error(s).\n"
                + "\n".join(
                    f"  - {'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
                )
                + "\nSee docs/internal/design/program-envelope-v3.md for the schema."
            )
            raise ProgramEnvelopeError(msg) from exc
    elif isinstance(envelope, ProgramEnvelope):
        parsed = envelope
    else:
        msg = f"from_envelope expected a dict or ProgramEnvelope, got {type(envelope).__name__}."
        raise ProgramEnvelopeError(msg)
    return _EnvelopeProgram(parsed)


__all__ = [
    "SUPPORTED_CAPABILITIES",
    "ClientSpec",
    "EnvelopeStep",
    "InputSpec",
    "JudgeSpec",
    "ProgramEnvelope",
    "ProgramEnvelopeError",
    "StepKind",
    "StepOutputField",
    "ToolSpec",
    "Tunable",
    # Private but re-exported for tests + internal callers (kaos-llm-core
    # has at least one test that imports `_EnvelopeProgram` directly,
    # and the cross-module bridges import the validation helpers).
    "_EnvelopeProgram",
    "_build_step_program",
    "_resolve_output_mapping",
    "_resolve_pointer",
    "_resolve_step_inputs",
    "_validate_pointer",
    "_validate_type_refs",
    # Public functions
    "from_envelope",
    "program_hash",
]
