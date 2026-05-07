"""Concrete envelope-backed Program. Run-time executor.

Walks the steps in declaration order, resolves inputs against the
live execution context, calls the underlying program for each step,
and assembles outputs via the top-level ``output:`` mapping.
"""

from __future__ import annotations

from typing import Any

from kaos_llm_core.programs.base import Program
from kaos_llm_core.programs.envelope.builders import _build_step_program
from kaos_llm_core.programs.envelope.models import EnvelopeStep, ProgramEnvelope
from kaos_llm_core.programs.envelope.substitution import (
    _resolve_output_mapping,
    _resolve_step_inputs,
)
from kaos_llm_core.programs.result_mixin import HasOutputs


class _EnvelopeProgram(Program):
    """Concrete :class:`Program` built from a :class:`ProgramEnvelope`.

    Sub-call discovery (Phase 11 ``ProgramGraph`` auto-registration)
    picks up each step under its declaration name (``self.<step_id>``)
    so optimizers can target individual steps.
    """

    def __init__(self, envelope: ProgramEnvelope) -> None:
        self._envelope = envelope
        self._step_specs: dict[str, EnvelopeStep] = {s.id: s for s in envelope.steps}
        # Build each step's underlying Program/Call.
        for step in envelope.steps:
            instance = _build_step_program(step, envelope)
            # Setting as a public attribute auto-registers via Phase 11
            # ProgramGraph __setattr__.
            setattr(self, step.id, instance)

    @property
    def envelope(self) -> ProgramEnvelope:
        """The envelope this program was built from."""
        return self._envelope

    async def forward(self, **inputs: Any) -> dict[str, Any]:
        """Execute the steps in declaration order."""
        ctx: dict[str, Any] = {"inputs": inputs, "steps": {}}
        for step in self._envelope.steps:
            step_inputs = _resolve_step_inputs(step.inputs, ctx)
            step_program = getattr(self, step.id)
            invocation = await step_program.invoke(**step_inputs)
            output = invocation.output
            # Result wrappers from BestOfN, MultiChainComparison, and
            # ProgramOfThought hold the underlying pydantic model on
            # their ``outputs`` attribute. Unwrap so downstream pointer
            # resolution sees a uniform shape regardless of which step
            # kind produced it. Phase 16.5: switched from hasattr to
            # isinstance(HasOutputs) so the check is type-narrowed and
            # any future result wrapper that forgets the ``outputs``
            # slot is caught at definition time.
            if isinstance(output, HasOutputs) and not isinstance(output, dict):
                output = output.outputs
            # Normalize: every step's output looks like a dict in ctx so
            # downstream pointer resolution works uniformly. Pydantic
            # signatures dump to dict cleanly via model_dump.
            if hasattr(output, "model_dump"):
                output_dict = output.model_dump()
            elif isinstance(output, dict):
                output_dict = output
            else:
                output_dict = {"result": output}
            ctx["steps"][step.id] = {"output": output_dict}
        return _resolve_output_mapping(self._envelope.output, ctx)
