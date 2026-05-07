"""Envelope constants — leaf module, no internal envelope deps.

Defines the closed step-kind enum, the supported-capability set, and
the regex patterns for program names and step ids. Importing from this
module is always cheap and side-effect-free.
"""

from __future__ import annotations

import re
from typing import Literal

# Program name and step id naming rules.
_NAME_RE = re.compile(r"^[a-z][a-z0-9-]*$")
_STEP_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")

# Closed enum of step kinds. v1 (kaos_program="1") supported six;
# v1.1 (kaos_program="1.1") adds two more from Phase 16.2:
# multi_chain_comparison (sample N reasoning chains and synthesize via
# an aggregator LM) and program_of_thought (code-as-reasoning with a
# subprocess sandbox). The schema bump is additive — every v1 envelope
# parses unchanged under v1.1.
StepKind = Literal[
    "call",
    "reason",
    "judge",
    "react",
    "refine",
    "best_of_n",
    "multi_chain_comparison",
    "program_of_thought",
]

# Capabilities the executor implements. Envelopes that declare a
# capability not in this set fail validation.
SUPPORTED_CAPABILITIES: frozenset[str] = frozenset(
    {
        "call",
        "reason",
        "judge",
        "react",
        "refine",
        "best_of_n",
        "multi_chain_comparison",
        "program_of_thought",
        "jsonpointer_refs",
        "jinja2_prompts",
    }
)
