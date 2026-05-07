"""Instruction proposers — Phase 17.1.

Proposer modules generate candidate instruction strings for an LLM
program. They are used by optimizers (e.g. ``MiproV2Optimizer``) that
need to search over instruction candidates rather than relying on a
single hand-written instruction.

The first member is :class:`GroundedInstructionProposer`, a port of
DSPy's ``GroundedProposer`` (`stanfordnlp/dspy/main/dspy/propose/
grounded_proposer.py`). It composes three sub-Signatures
(``DescribeProgram``, ``DescribeModule``, ``GenerateSingleModuleInstruction``)
under one meta-LM and produces N stylistically diverse instructions
per call by sampling from a hard-coded ``TIPS`` bank.
"""

from __future__ import annotations

from kaos_llm_core.optimization.proposers.grounded import (
    TIPS,
    DescribeDataset,
    DescribeModule,
    DescribeProgram,
    GenerateSingleModuleInstruction,
    GroundedInstructionProposer,
    ProposedInstruction,
)

__all__ = [
    "TIPS",
    "DescribeDataset",
    "DescribeModule",
    "DescribeProgram",
    "GenerateSingleModuleInstruction",
    "GroundedInstructionProposer",
    "ProposedInstruction",
]
