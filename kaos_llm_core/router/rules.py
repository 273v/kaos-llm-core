"""RuleRouter — rule-based model selection.

Maps task characteristics to models via a list of rules. Each rule specifies
conditions on the Signature name and/or input values, and a target model.
First matching rule wins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kaos_llm_core.errors import CallError
from kaos_llm_core.signatures.signature import Signature


@dataclass(frozen=True, slots=True)
class Rule:
    """A single routing rule.

    Matches when:
    - ``signature_name`` matches the Signature class name (if set)
    - ``input_matches`` is a dict where all key-value pairs are present in inputs (if set)

    At least one of ``signature_name`` or ``input_matches`` must be set.
    """

    model: str
    signature_name: str | None = None
    input_matches: dict[str, Any] = field(default_factory=dict)

    def matches(self, signature: type[Signature], inputs: dict[str, Any]) -> bool:
        """Return True if this rule matches the given signature and inputs."""
        if self.signature_name is not None and signature.__name__ != self.signature_name:
            return False
        if self.input_matches:
            for key, value in self.input_matches.items():
                if key not in inputs or inputs[key] != value:
                    return False
        return True


class RuleRouter:
    """Route to models based on matching rules.

    Rules are evaluated in order; the first match wins. If no rule matches,
    ``default_model`` is used (if set), otherwise an error is raised.

    Example::

        router = RuleRouter(
            rules=[
                Rule(model="anthropic:claude-haiku-4-5", signature_name="ClassifyDoc"),
                Rule(model="anthropic:claude-sonnet-4-6", signature_name="ExtractEntities"),
            ],
            default_model="anthropic:claude-haiku-4-5",
        )
    """

    def __init__(
        self,
        rules: list[Rule],
        default_model: str | None = None,
    ) -> None:
        self.rules = rules
        self.default_model = default_model

    async def select_model(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
        context: Any = None,
    ) -> str:
        """Select a model by evaluating rules in order."""
        for rule in self.rules:
            if rule.matches(signature, inputs):
                return rule.model

        if self.default_model:
            return self.default_model

        raise CallError(
            f"RuleRouter: no rule matched for {signature.__name__} "
            f"and no default_model is set. "
            f"Add a catch-all rule or set default_model."
        )
