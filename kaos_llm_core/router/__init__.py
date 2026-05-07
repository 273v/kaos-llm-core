"""Router system — model selection for Calls."""

from kaos_llm_core.router.base import Router
from kaos_llm_core.router.cascade import CascadeRouter
from kaos_llm_core.router.rules import Rule, RuleRouter

__all__ = [
    "CascadeRouter",
    "Router",
    "Rule",
    "RuleRouter",
]
