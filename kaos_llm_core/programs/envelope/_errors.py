"""Envelope error type — leaf module, no internal envelope deps."""

from __future__ import annotations

from kaos_llm_core.errors import KaosLLMCoreError


class ProgramEnvelopeError(KaosLLMCoreError):
    """Envelope failed validation or could not be built into a Program.

    Subclass of :class:`KaosLLMCoreError` so callers doing
    ``except KaosLLMCoreError:`` catch envelope failures alongside
    every other library error. Messages follow the agent-friendly
    contract: what went wrong, where (envelope path or step id),
    how to fix it.
    """
