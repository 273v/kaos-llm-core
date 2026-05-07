"""Step builder registry — replaces the pre-Phase-16.5 if/elif ladder.

Each step kind has one focused builder function. Adding a new kind
is one entry in :data:`_STEP_KIND_BUILDERS` plus one builder
function — no edits to the dispatch ladder.

Builders take a :class:`_StepBuildContext` (the pre-resolved model /
instruction / kwargs / examples / signature) so the per-kind code
is focused on its own arguments. Builders MAY raise
:class:`ProgramEnvelopeError` for kind-specific validation; the
dispatch layer is just a lookup.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from kaos_llm_core.programs.call import Call
from kaos_llm_core.programs.chain_of_thought import ChainOfThought
from kaos_llm_core.programs.envelope._errors import ProgramEnvelopeError
from kaos_llm_core.programs.envelope.models import EnvelopeStep, ProgramEnvelope
from kaos_llm_core.programs.envelope.signatures import _build_signature_for_step
from kaos_llm_core.programs.judge import Judge
from kaos_llm_core.types import Example

if TYPE_CHECKING:
    from kaos_llm_core.programs.base import Program
    from kaos_llm_core.signatures.signature import Signature


@dataclass(frozen=True, slots=True)
class _StepBuildContext:
    """Pre-resolved values shared across every step builder."""

    step: EnvelopeStep
    envelope: ProgramEnvelope
    sig: type[Signature]
    effective_model: str
    effective_instruction: str
    base_kwargs: dict[str, Any]


def _build_call_step(ctx: _StepBuildContext) -> Call | Program:
    return Call(ctx.sig, **ctx.base_kwargs)


def _build_reason_step(ctx: _StepBuildContext) -> Call | Program:
    return ChainOfThought(ctx.sig, **ctx.base_kwargs)


def _build_judge_step(ctx: _StepBuildContext) -> Call | Program:
    # Judge takes producer_model + judge_model. The envelope's
    # 'judge' step kind builds a Judge whose producer Signature is
    # the step's Signature; the judge_model defaults to the same
    # client as the step.
    return Judge(
        ctx.sig,
        producer_model=ctx.effective_model,
        judge_model=ctx.effective_model,
        criteria=ctx.effective_instruction,
    )


def _build_multi_chain_comparison_step(ctx: _StepBuildContext) -> Call | Program:
    """Phase 16.2: sample N reasoning chains and synthesize via aggregator LM.

    Phase 16.5: ``ctx.step`` is the discriminated union ``EnvelopeStep``;
    we ``cast`` to the per-kind sub-class so ty narrows the per-kind
    field accesses (``aggregator_client``, ``n``).
    """
    from kaos_llm_core.programs.envelope.models import MultiChainComparisonStep
    from kaos_llm_core.programs.multi_chain_comparison import MultiChainComparison

    step = cast("MultiChainComparisonStep", ctx.step)
    agg_model = ctx.effective_model
    if step.aggregator_client is not None:
        agg_client = ctx.envelope.clients.get(step.aggregator_client)
        if agg_client is None:
            msg = (
                f"Step {step.id!r}: aggregator_client={step.aggregator_client!r} "
                f"is not declared in the envelope's 'clients' block."
            )
            raise ProgramEnvelopeError(msg)
        agg_model = agg_client.model_string
    return MultiChainComparison(
        ctx.sig,
        n=step.n,
        producer_model=ctx.effective_model,
        aggregator_model=agg_model,
        instruction=ctx.effective_instruction,
    )


def _build_program_of_thought_step(ctx: _StepBuildContext) -> Call | Program:
    """Phase 16.2: code-as-reasoning with subprocess sandbox.

    ``allow_code_execution`` was validated at parse time by the
    discriminated-union ``ProgramOfThoughtStep`` model — by the time
    we get here, the field is guaranteed to be ``True``.
    """
    from kaos_llm_core.programs.envelope.models import ProgramOfThoughtStep
    from kaos_llm_core.programs.program_of_thought import ProgramOfThought

    step = cast("ProgramOfThoughtStep", ctx.step)
    pot_kwargs: dict[str, Any] = {
        "producer_model": ctx.effective_model,
        "interpreter_model": ctx.effective_model,
        "allow_code_execution": True,
    }
    if step.timeout_s is not None:
        pot_kwargs["timeout_s"] = step.timeout_s
    return ProgramOfThought(ctx.sig, **pot_kwargs)


_RESERVED_ALTERNATIVES: dict[str, str] = {
    "react": (
        "For ReAct loops, use the kaos-llm-core-react MCP tool directly "
        "(dry-run only) or the kaos_llm_core.programs.react.ReAct Python API "
        "for real tool execution."
    ),
    "refine": (
        "For produce-critique-refine loops, use the kaos-llm-core-refine MCP "
        "tool directly, or compose kind='call' + kind='judge' steps in the "
        "envelope as a manual alternative."
    ),
    "best_of_n": (
        "For N-sample selection, use the kaos-llm-core-best-of-n MCP tool "
        "directly, or run multiple kind='call' steps and select externally."
    ),
}


def _build_reserved_step(ctx: _StepBuildContext) -> Call | Program:
    """Builder for kinds reserved in the schema but not yet executable.

    Currently: ``react``, ``refine``, ``best_of_n``. The schema
    validates them but the executor refuses to run them until each
    kind's wiring lands (tool dispatch, judge sub-spec, sample
    fan-out — see docs/internal/design/program-envelope-v3.md).
    """
    step = ctx.step
    alternative = _RESERVED_ALTERNATIVES.get(
        step.kind,
        "Use kind='call', kind='reason', kind='judge', "
        "kind='multi_chain_comparison', or kind='program_of_thought' instead.",
    )
    msg = (
        f"Step {step.id!r}: kind={step.kind!r} is NOT EXECUTABLE in the v3 "
        f"envelope. The envelope schema accepts {step.kind!r} for forward "
        f"compatibility but the executor cannot run it yet — the wiring for "
        f"tool dispatch, judge sub-specs, and sample fan-out has not landed. "
        f"How to fix: replace kind={step.kind!r} with an executable kind "
        f"(call, reason, judge, multi_chain_comparison, program_of_thought). "
        f"Alternative: {alternative}"
    )
    raise ProgramEnvelopeError(msg)


# The single source of truth for step-kind dispatch. New step kinds are
# added by (1) extending the StepKind Literal in _constants.py, (2)
# adding the kind to SUPPORTED_CAPABILITIES, (3) writing a per-kind
# validator branch in validation.py if needed, and (4) registering the
# builder here.
_STEP_KIND_BUILDERS: dict[str, Callable[[_StepBuildContext], Call | Program]] = {
    "call": _build_call_step,
    "reason": _build_reason_step,
    "judge": _build_judge_step,
    "multi_chain_comparison": _build_multi_chain_comparison_step,
    "program_of_thought": _build_program_of_thought_step,
    "react": _build_reserved_step,
    "refine": _build_reserved_step,
    "best_of_n": _build_reserved_step,
}


def _build_step_program(
    step: EnvelopeStep,
    envelope: ProgramEnvelope,
) -> Call | Program:
    """Construct the underlying program for one envelope step.

    Dispatches via :data:`_STEP_KIND_BUILDERS` instead of an if/elif
    ladder. Each builder is a small focused function; new step kinds
    are one registry entry plus one function.
    """
    sig = _build_signature_for_step(step, envelope.types)
    client_spec = envelope.clients[step.client]

    # Resolve model: tunable.model > step.tunable.model > client.model
    effective_model = step.tunable.model or client_spec.model_string

    # Resolve instructions: tunable.instruction > step.instruction
    effective_instruction = (
        step.tunable.instruction if step.tunable.instruction is not None else step.instruction
    )

    # Combine kwargs: client.kwargs + step.tunable.hyperparameters
    combined_kwargs = {**client_spec.kwargs, **step.tunable.hyperparameters}

    # Build per-step Examples from tunable.demos
    examples = [
        Example(inputs=d.get("inputs", {}), outputs=d.get("outputs", {}))
        for d in step.tunable.demos
    ]

    base_kwargs: dict[str, Any] = {
        "model": effective_model,
        "instructions": effective_instruction,
        "examples": examples or None,
        **combined_kwargs,
    }

    ctx = _StepBuildContext(
        step=step,
        envelope=envelope,
        sig=sig,
        effective_model=effective_model,
        effective_instruction=effective_instruction,
        base_kwargs=base_kwargs,
    )

    builder = _STEP_KIND_BUILDERS.get(step.kind)
    if builder is None:
        msg = f"Step {step.id!r}: unknown kind {step.kind!r}."
        raise ProgramEnvelopeError(msg)
    return builder(ctx)
