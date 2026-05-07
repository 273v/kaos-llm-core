"""MultiChainComparison — sample N reasoning chains and synthesize.

Phase 16.2 step kind. Inspired by DSPy's MultiChainComparison module:
sample the same producer ``ChainOfThought`` ``n`` times in parallel
(varying the seed / temperature so the chains diverge), then send all
``n`` candidate chains to an aggregator Call that synthesizes the
final output.

This is distinct from :class:`BestOfN` and :class:`Ensemble`:

  - ``BestOfN`` samples N times against ONE model and **picks** one
    candidate (by metric or judge). The losers are discarded.
  - ``Ensemble`` runs the same task across N **different** models in
    parallel and majority-votes on the output.
  - ``MultiChainComparison`` samples N times against ONE model and
    **synthesizes** a new answer from all N reasoning chains via an
    aggregator LM. No candidate is discarded; the aggregator sees
    every chain and produces a fresh answer informed by all of them.

The synthesis path is the load-bearing difference: it lets the
aggregator catch the case where every individual chain reaches a
plausible-but-wrong conclusion that, when read together, reveals the
right answer. Vote-based and pick-best strategies cannot do this.

Audit notes inherited from BestOfN
----------------------------------

Sampling fans out via :func:`asyncio.gather`. Producer hyperparameters
are never mutated: each sample runs against a transient deep clone
whose mutable internals are isolated from the original (same pattern
as :class:`BestOfN`, audit finding #7).

The trace tree uses the push-based collector so each sample's clone
appends its trace to the active collector. The final parent trace
lists every sample as a child plus the aggregator's trace as the
final child.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from kaos_core.logging import get_logger
from pydantic import create_model

from kaos_llm_core.errors import CallError
from kaos_llm_core.programs.base import Program
from kaos_llm_core.programs.call import Call
from kaos_llm_core.programs.chain_of_thought import ChainOfThought
from kaos_llm_core.programs.cloning import clone_call
from kaos_llm_core.programs.result_mixin import OutputForwardingMixin
from kaos_llm_core.signatures.fields import InputField, OutputField
from kaos_llm_core.signatures.signature import Signature

logger = get_logger(__name__)


@dataclass
class _ChainSample:
    """One reasoning chain produced by a producer sample."""

    index: int
    output: Any  # validated pydantic model
    reasoning: str


@dataclass(frozen=True, slots=True)
class MultiChainComparisonResult(OutputForwardingMixin):
    """Final result of a MultiChainComparison run.

    ``outputs`` is the aggregator's synthesized answer (the same shape
    as the producer signature). The constituent chains are accessible
    via ``chains`` for inspection / logging / optimizer feedback.
    Attribute access transparently forwards to ``outputs`` via the
    :class:`OutputForwardingMixin`, matching the convention from
    :class:`BestOfN`.
    """

    outputs: Any
    chains: list[_ChainSample] = field(default_factory=list)
    failed_count: int = 0
    errors: list[str] = field(default_factory=list)


class MultiChainComparison(Program):
    """Sample N reasoning chains and synthesize via an aggregator LM.

    Args:
        signature: The producer Signature. The aggregator's output
            shape is the same as this signature.
        n: Number of reasoning chains to sample. Default 5.
        producer_model: Model for the N producer samples. If None,
            falls back to the per-Call default.
        aggregator_model: Model for the synthesis step. Defaults to
            ``producer_model`` so callers can use one model for both
            phases.
        temperature: Temperature for the producer samples. Slight
            variation across samples comes from sampling, not from
            randomizing temperature per-sample.
        instruction: Optional override for the producer instruction.
            The aggregator's instruction is fixed: "synthesize a final
            answer informed by every reasoning chain below."
    """

    # Hard cap on ``n`` — see ReAct.MAX_ITERATIONS for rationale.
    MAX_N: int = 100

    def __init__(
        self,
        signature: type[Signature],
        *,
        n: int = 5,
        producer_model: str | None = None,
        aggregator_model: str | None = None,
        temperature: float = 0.7,
        instruction: str | None = None,
    ) -> None:
        super().__init__()
        if n < 2:
            msg = (
                f"MultiChainComparison requires n >= 2 (got {n}). "
                "For a single sample, use ChainOfThought directly."
            )
            raise ValueError(msg)
        if n > self.MAX_N:
            msg = (
                f"MultiChainComparison: n={n} exceeds the hard cap of "
                f"{self.MAX_N}. The cap protects against runaway API spend "
                "(each chain is a full LLM call); pass a smaller n, or "
                "override MAX_N in a subclass."
            )
            raise ValueError(msg)
        self.signature = signature
        self.n = n
        self.producer_model = producer_model
        self.aggregator_model = aggregator_model or producer_model
        self.temperature = temperature
        self.instruction = instruction

        # Producer: ChainOfThought of the original signature.
        cot_kwargs: dict[str, Any] = {"temperature": temperature}
        if producer_model is not None:
            cot_kwargs["model"] = producer_model
        if instruction is not None:
            cot_kwargs["instructions"] = instruction
        self.producer = ChainOfThought(signature, **cot_kwargs)

        # Aggregator: Call against a Signature that takes
        # (original_inputs..., chains: list[str]) and produces the
        # original output fields.
        self.aggregator = self._build_aggregator()

    def _build_aggregator(self) -> Call:
        """Build the aggregator Call dynamically.

        The aggregator Signature mirrors the producer's input fields,
        adds a ``chains`` input that holds the N reasoning chains as a
        list of strings, and produces every output field of the
        producer signature.
        """
        from kaos_llm_core.signatures.introspection import (
            get_input_fields,
            get_output_fields,
        )

        input_fields = get_input_fields(self.signature)
        output_fields = get_output_fields(self.signature)

        # Build the aggregator's field defs
        agg_field_defs: dict[str, Any] = {}
        for name, field_info in input_fields.items():
            # Mirror the producer's input fields with their original types
            agg_field_defs[name] = (
                self.signature.model_fields[name].annotation,
                InputField(description=field_info.description or name),
            )
        agg_field_defs["chains"] = (
            list[str],
            InputField(
                description=(
                    "The N candidate reasoning chains from the producer "
                    "samples. Each chain is one independent attempt at "
                    "solving the task. Synthesize a final answer that "
                    "is informed by all of them — not by majority vote, "
                    "but by reasoning across the chains as a whole."
                ),
            ),
        )
        for name, field_info in output_fields.items():
            agg_field_defs[name] = (
                self.signature.model_fields[name].annotation,
                OutputField(description=field_info.description or name),
            )

        agg_sig = create_model(
            f"{self.signature.__name__}Aggregator",
            __base__=Signature,
            __doc__=(
                f"Synthesize a final answer for {self.signature.__name__} "
                "by reasoning across N candidate reasoning chains."
            ),
            **agg_field_defs,
        )

        agg_kwargs: dict[str, Any] = {}
        if self.aggregator_model is not None:
            agg_kwargs["model"] = self.aggregator_model
        return Call(agg_sig, **agg_kwargs)

    async def forward(self, **inputs: Any) -> Any:
        """Sample N chains, then synthesize."""
        # 1. Fan out N producer samples in parallel via deep clones.
        clones = [clone_call(self.producer) for _ in range(self.n)]

        async def _run_one(idx: int, clone: Any) -> _ChainSample | Exception:
            try:
                invocation = await clone.invoke(**inputs)
                # ChainOfThought puts reasoning either on the output (prompt
                # mode) or in invocation.extras['native_thinking']
                # (native mode). Try both.
                reasoning = ""
                output = invocation.output
                if hasattr(output, "reasoning"):
                    reasoning = str(getattr(output, "reasoning", "") or "")
                if not reasoning and invocation.extras:
                    reasoning = str(invocation.extras.get("native_thinking", "") or "")
                if not reasoning:
                    reasoning = "(no explicit reasoning captured)"
                return _ChainSample(index=idx, output=output, reasoning=reasoning)
            except Exception as exc:
                return exc

        results = await asyncio.gather(
            *(_run_one(i, c) for i, c in enumerate(clones)),
            return_exceptions=False,
        )

        chains: list[_ChainSample] = []
        errors: list[str] = []
        for r in results:
            if isinstance(r, _ChainSample):
                chains.append(r)
            else:
                errors.append(f"{type(r).__name__}: {r}")

        if not chains:
            err_summary = ", ".join(errors) or "no samples succeeded"
            msg = (
                f"MultiChainComparison: all {self.n} producer samples failed. Errors: {err_summary}"
            )
            raise CallError(msg)

        # 2. Aggregator synthesis.
        chain_texts = [f"Chain {c.index + 1}: {c.reasoning}" for c in chains]
        agg_inputs: dict[str, Any] = dict(inputs)
        agg_inputs["chains"] = chain_texts
        agg_invocation = await self.aggregator.invoke(**agg_inputs)

        result = MultiChainComparisonResult(
            outputs=agg_invocation.output,
            chains=chains,
            failed_count=len(errors),
            errors=errors,
        )
        return result
