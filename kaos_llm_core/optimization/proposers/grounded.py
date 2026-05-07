"""GroundedInstructionProposer — KAOS port of DSPy's GroundedProposer.

Phase 17.1: the meta-LM-driven instruction proposer that
:class:`MiproV2Optimizer` uses to generate ``N`` stylistically
diverse candidate instructions per predictor.

Reference: ``stanfordnlp/dspy/main/dspy/propose/grounded_proposer.py``
(SHA ``4ece27f7494e3ae3ac4311034481f9c697edb69a``). The KAOS port
mirrors the three sub-Signatures verbatim and the ``TIPS`` bank
verbatim — these are hand-tuned and changing them would invalidate
benchmark comparisons against DSPy.

Architecture
============

A grounded instruction proposal needs context to be useful. The
proposer wires together up to five context sources, conditional on
configuration flags:

* **Dataset summary** — a one-shot natural-language description of
  the training set, computed once via :class:`DescribeDataset` and
  reused across every draw. (``data_aware`` toggle.)
* **Program code** — a pseudocode rendering of the LLM program
  being optimized, used by :class:`DescribeProgram` to summarize
  what the program does. (``program_aware`` toggle.)
* **Module description** — within a multi-predictor program, the
  proposer generates one instruction per predictor; the
  per-module description gives the proposer enough context to
  write instructions specific to that sub-call. (``program_aware``
  also gates this.)
* **Task demos** — a few-shot exemplar of inputs / outputs for the
  predictor being targeted, sampled from the bootstrap candidate
  sets that step 1 of MIPROv2 produced. (``fewshot_aware`` toggle.)
* **Tip** — one of six hand-tuned stylistic nudges sampled
  uniformly per draw, to inject variety. (``tip_aware`` toggle.)

The proposer's primary entry point is
:meth:`GroundedInstructionProposer.propose_n_instructions_for_call`,
which returns ``N`` candidate instructions for a single Call.
Phase 17.1 restricts MIPROv2 to single-predictor targets, so the
multi-predictor variant from DSPy's ``propose_instructions_for_program``
is not yet exposed.

Cost characteristics
====================

For one Call with N draws and ``program_aware=True``, this module
makes:

* 1 dataset-summary call (one-shot, on construction).
* 1 ``DescribeProgram`` call (one-shot, on construction).
* 1 ``DescribeModule`` call (one-shot, on construction).
* N ``GenerateSingleModuleInstruction`` calls (one per draw).

Total per Call: ``3 + N`` proposer-LM invocations. The MIPROv2
optimizer wraps the construction calls in their own ``runner.trial``
scopes so the cost is attributed correctly in the mutation log.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from kaos_llm_core.programs.call import Call
from kaos_llm_core.signatures import InputField, OutputField, Signature
from kaos_llm_core.types import Example

__all__ = [
    "TIPS",
    "DescribeDataset",
    "DescribeModule",
    "DescribeProgram",
    "GenerateSingleModuleInstruction",
    "GroundedInstructionProposer",
    "ProposedInstruction",
]


# ---------------------------------------------------------------------------
# TIPS bank — verbatim from DSPy grounded_proposer.py:17 for parity
# ---------------------------------------------------------------------------


TIPS: dict[str, str] = {
    "none": "",
    "creative": ("Don't be afraid to be creative when creating the new instruction!"),
    "simple": "Keep the instruction clear and concise.",
    "description": "Make sure your instruction is very informative and descriptive.",
    "high_stakes": (
        "The instruction should include a high stakes scenario in which the LM must solve the task!"
    ),
    "persona": (
        'Include a persona that is relevant to the task in the instruction (ie. "You are a ...")'
    ),
}


# ---------------------------------------------------------------------------
# Sub-signatures — mirrored verbatim from DSPy grounded_proposer.py
# ---------------------------------------------------------------------------


class DescribeDataset(Signature):
    """Below are several examples from a dataset. Please write a short
    natural-language description of what kind of inputs the dataset
    contains and what the expected outputs look like. Be specific about
    the domain and the structure of the task.
    """

    examples_text: str = InputField(
        description=(
            "A serialized batch of training examples (one per line, each as 'inputs -> outputs')."
        )
    )
    dataset_description: str = OutputField(
        description=(
            "A short natural-language description of the dataset and the task it represents."
        )
    )


class DescribeProgram(Signature):
    """Below is some pseudocode for a pipeline that solves tasks with
    calls to language models. Please describe what type of task this
    program appears to be designed to solve, and how it appears to
    work.
    """

    program_code: str = InputField(description="Pseudocode for a language model program.")
    program_example: str = InputField(
        description="An example of the program in use (one input/output pair)."
    )
    program_description: str = OutputField(
        description=(
            "Describe what task the program is designed to solve, and how "
            "it goes about solving this task."
        )
    )


class DescribeModule(Signature):
    """Below is some pseudocode for a pipeline that solves tasks with
    calls to language models. Please describe the purpose of the
    specified module within this pipeline.
    """

    program_code: str = InputField(description="Pseudocode for a language model program.")
    program_example: str = InputField(description="An example of the program in use.")
    program_description: str = InputField(
        description=(
            "Summary of the task the program is designed to solve, and how "
            "it goes about solving it."
        )
    )
    module: str = InputField(description="The module in the program that we want to describe.")
    module_description: str = OutputField(
        description="Description of the module's role in the broader program."
    )


class GenerateSingleModuleInstruction(Signature):
    """Use the information below to learn about a task that we are
    trying to solve using calls to a language model, then generate a
    new instruction that will be used to prompt a language model to
    better solve the task.
    """

    dataset_description: str = InputField(
        description="A description of the dataset that we are using."
    )
    program_code: str = InputField(
        description="Language model program designed to solve a particular task."
    )
    program_description: str = InputField(
        description=(
            "Summary of the task the program is designed to solve, and how "
            "it goes about solving it."
        )
    )
    module: str = InputField(description="The module to create an instruction for.")
    module_description: str = InputField(
        description="Description of the module to create an instruction for."
    )
    task_demos: str = InputField(description="Example inputs and outputs of the module.")
    basic_instruction: str = InputField(description="The current basic instruction for the module.")
    tip: str = InputField(
        description=("A short suggestion for how to go about generating the new instruction.")
    )
    proposed_instruction: str = OutputField(
        description=(
            "Propose an instruction that will be used to prompt a language "
            "model to perform this task. Be self-contained and complete."
        )
    )
    rationale: str = OutputField(
        description=(
            "A one-sentence justification for the chosen instruction (used by the mutation log)."
        )
    )


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProposedInstruction:
    """One instruction draw from the grounded proposer.

    Attributes
    ----------
    instruction:
        The proposed instruction string.
    tip:
        Which TIPS key was sampled for this draw (e.g. ``"creative"``).
    rationale:
        The proposer's own one-sentence justification.
    verbatim_copy:
        ``True`` if the proposer returned the basic instruction
        unchanged. The mutation log uses this flag for post-hoc
        analysis (cheap LLMs sometimes copy the input field).
    """

    instruction: str
    tip: str
    rationale: str
    verbatim_copy: bool


# ---------------------------------------------------------------------------
# The proposer module
# ---------------------------------------------------------------------------


class GroundedInstructionProposer:
    """Generate ``N`` candidate instructions for an LLM call, grounded
    in dataset / program / module / demo / tip context.

    Parameters
    ----------
    proposer_model:
        Meta-LM used for every sub-call. Should be a strong model
        (defaults to ``"anthropic:claude-sonnet-4-6"``).
    program_code:
        Optional pseudocode rendering of the user's program. If
        ``None``, falls back to the call's signature docstring.
    program_aware:
        Toggle the ``DescribeProgram`` / ``DescribeModule`` /
        ``program_*`` input fields. Default ``True``.
    data_aware:
        Toggle the ``DescribeDataset`` summary. Default ``True``.
    tip_aware:
        Toggle the random tip sampling. Default ``True``.
    fewshot_aware:
        Toggle the ``task_demos`` input field. Default ``True``.
    init_temperature:
        Sampling temperature for the proposer LM. DSPy default 1.0.
    view_data_batch_size:
        How many training examples to feed the dataset summarizer.
    seed:
        Random seed for tip sampling.

    Notes
    -----
    The dataset summary, program description, and module description
    are computed eagerly via :meth:`build_static_context`. Call
    :meth:`build_static_context` once with the train_set and the
    target call before issuing any draws via
    :meth:`propose_n_instructions_for_call`. The MIPROv2 optimizer
    wraps both calls in :class:`TrialRunner` trial scopes so the cost
    is attributed correctly.
    """

    def __init__(
        self,
        *,
        proposer_model: str = "anthropic:claude-sonnet-4-6",
        program_code: str | None = None,
        program_aware: bool = True,
        data_aware: bool = True,
        tip_aware: bool = True,
        fewshot_aware: bool = True,
        init_temperature: float = 1.0,
        view_data_batch_size: int = 10,
        seed: int = 9,
    ) -> None:
        self.proposer_model = proposer_model
        self.program_code = program_code
        self.program_aware = program_aware
        self.data_aware = data_aware
        self.tip_aware = tip_aware
        self.fewshot_aware = fewshot_aware
        self.init_temperature = init_temperature
        self.view_data_batch_size = view_data_batch_size
        self._rng = random.Random(seed)

        # Filled by build_static_context.
        self.dataset_description: str = ""
        self.program_description: str = ""
        self.module_description: str = ""
        self._static_built: bool = False

    # ------------------------------------------------------------------
    # Static context (called once per optimize() pipeline)
    # ------------------------------------------------------------------

    async def build_static_context(
        self,
        *,
        call: Call,
        train_set: list[Example],
        program_example: str | None = None,
    ) -> None:
        """Compute the one-shot dataset / program / module summaries.

        Must be called before :meth:`propose_n_instructions_for_call`.
        Each sub-call is independent and runs sequentially. Cost
        attribution is the caller's responsibility — wrap this method
        in a ``runner.trial("mipro_v2_summarize_dataset")`` scope from
        the optimizer.

        Parameters
        ----------
        call:
            The Call being optimized. Used to derive the program code
            (if not provided at construction time) and the module
            label.
        train_set:
            The training set, used by the dataset summarizer. The
            first ``view_data_batch_size`` examples are serialized
            into a single text blob and sent to the summarizer.
        program_example:
            Optional rendering of one example input/output for the
            program. If ``None``, the first ``train_set`` example is
            used.
        """
        program_code = self.program_code or _render_call_pseudocode(call)
        if program_example is None:
            program_example = _render_example(train_set[0]) if train_set else ""

        # 1. Dataset summary (data_aware path).
        if self.data_aware:
            sample = train_set[: self.view_data_batch_size]
            examples_text = "\n".join(_render_example(ex) for ex in sample)
            describe_ds = Call(
                DescribeDataset, model=self.proposer_model, temperature=self.init_temperature
            )
            ds_result = await describe_ds(examples_text=examples_text)
            self.dataset_description = str(ds_result.dataset_description).strip()
        else:
            self.dataset_description = ""

        # 2. Program description (program_aware path).
        if self.program_aware:
            describe_prog = Call(
                DescribeProgram, model=self.proposer_model, temperature=self.init_temperature
            )
            prog_result = await describe_prog(
                program_code=program_code,
                program_example=program_example,
            )
            self.program_description = str(prog_result.program_description).strip()

            # 3. Module description.
            describe_mod = Call(
                DescribeModule, model=self.proposer_model, temperature=self.init_temperature
            )
            mod_result = await describe_mod(
                program_code=program_code,
                program_example=program_example,
                program_description=self.program_description,
                module=call.signature.__name__,
            )
            self.module_description = str(mod_result.module_description).strip()
        else:
            self.program_description = ""
            self.module_description = ""

        self._static_built = True

    # ------------------------------------------------------------------
    # Per-call instruction proposal (the per-draw path)
    # ------------------------------------------------------------------

    async def propose_n_instructions_for_call(
        self,
        *,
        call: Call,
        n: int,
        demo_candidates: list[list[Example]] | None = None,
    ) -> list[ProposedInstruction]:
        """Generate ``n`` candidate instructions for the given call.

        Parameters
        ----------
        call:
            The Call being optimized. Its current ``instructions`` is
            passed as ``basic_instruction`` and its docstring is used
            for the program code fallback.
        n:
            Number of instructions to generate.
        demo_candidates:
            Optional list of demo sets (each set is a ``list[Example]``)
            from MIPROv2 step 1 bootstrap. Each draw samples one demo
            set as ``task_demos``. If ``None`` or empty,
            ``task_demos`` is the empty string and ``fewshot_aware``
            is implicitly off for these draws.
        """
        if not self._static_built:
            msg = (
                "GroundedInstructionProposer.propose_n_instructions_for_call: "
                "must call build_static_context() first."
            )
            raise RuntimeError(msg)
        if n < 1:
            msg = f"n must be >= 1; got {n}"
            raise ValueError(msg)

        program_code = self.program_code or _render_call_pseudocode(call)
        basic_instruction = call.instructions or (call.signature.__doc__ or "").strip()
        tip_keys = list(TIPS.keys())

        results: list[ProposedInstruction] = []
        generator = Call(
            GenerateSingleModuleInstruction,
            model=self.proposer_model,
            temperature=self.init_temperature,
        )

        for draw_index in range(n):
            tip_key = self._rng.choice(tip_keys) if self.tip_aware else "none"
            tip_text = TIPS[tip_key]

            if demo_candidates and self.fewshot_aware:
                demo_set = demo_candidates[draw_index % len(demo_candidates)]
                task_demos = "\n".join(_render_example(ex) for ex in demo_set)
            else:
                task_demos = ""

            proposal = await generator(
                dataset_description=self.dataset_description,
                program_code=program_code,
                program_description=self.program_description,
                module=call.signature.__name__,
                module_description=self.module_description,
                task_demos=task_demos,
                basic_instruction=basic_instruction,
                tip=tip_text,
            )
            instruction = str(proposal.proposed_instruction).strip()
            rationale = str(getattr(proposal, "rationale", "")).strip()
            verbatim = instruction == basic_instruction.strip()
            results.append(
                ProposedInstruction(
                    instruction=instruction,
                    tip=tip_key,
                    rationale=rationale,
                    verbatim_copy=verbatim,
                )
            )

        return results


# ---------------------------------------------------------------------------
# Helpers — call/example rendering for the meta-LM context
# ---------------------------------------------------------------------------


def _render_call_pseudocode(call: Call) -> str:
    """Render a single Call as one-line pseudocode.

    DSPy's ``get_dspy_source_code`` produces a multi-line dump of the
    user's program. KAOS Call objects are simpler — one Signature in,
    one Signature out — so a one-line synopsis is enough context for
    the proposer.
    """
    sig_name = call.signature.__name__
    docstring = (call.signature.__doc__ or "").strip().splitlines()
    summary = docstring[0] if docstring else "(no docstring)"
    fields_in = ", ".join(_input_field_names(call.signature))
    fields_out = ", ".join(_output_field_names(call.signature))
    return (
        f"def {sig_name}({fields_in}) -> ({fields_out}):\n"
        f"    # {summary}\n"
        f"    return llm_call(signature={sig_name!r})"
    )


def _render_example(example: Example) -> str:
    """Render one Example as 'inputs -> outputs' for context blobs."""
    inputs_str = ", ".join(f"{k}={v!r}" for k, v in example.inputs.items())
    outputs_str = ", ".join(f"{k}={v!r}" for k, v in example.outputs.items())
    return f"{inputs_str} -> {outputs_str}"


def _input_field_names(signature: type[Signature]) -> list[str]:
    from kaos_llm_core.signatures.introspection import get_input_fields

    return list(get_input_fields(signature).keys())


def _output_field_names(signature: type[Signature]) -> list[str]:
    from kaos_llm_core.signatures.introspection import get_output_fields

    return list(get_output_fields(signature).keys())


# Re-exported for callers that want to inspect the helpers (e.g. tests).
_helpers: dict[str, Any] = {
    "_render_call_pseudocode": _render_call_pseudocode,
    "_render_example": _render_example,
}
