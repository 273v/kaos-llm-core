"""ChainOfThought — Call with automatic step-by-step reasoning.

Two modes, selected automatically based on model capabilities:

1. **Native thinking** — model supports it (Anthropic ``thinking``, OpenAI
   ``reasoning``, Google ``thinkingConfig``). The original Signature is sent
   unchanged; reasoning comes from ``response.thinking``.
2. **Prompt-based** — injects a ``reasoning`` OutputField into the Signature
   so the LLM writes reasoning in the JSON output.

Both modes go through Call's standard pipeline (retry, trace, error handling)
by overriding step methods, never by copying the pipeline.

Phase 10 redesign: per-execution scratch (``native_thinking``) lives in
``Invocation.extras``, populated by overriding
:meth:`Call._build_invocation_extras`. Step methods read it via
:func:`current_invocation` instead of from instance attributes. There is
no ``_native_thinking`` instance attribute, no ``_execute`` override,
and no ``_is_native_thinking`` helper that falls back to instance state.
"""

from __future__ import annotations

from typing import Any

from kaos_core.logging import get_logger
from kaos_llm_client import BaseProviderClient, ProviderResponse
from kaos_llm_client.settings import KaosLLMSettings
from pydantic import BaseModel

from kaos_llm_core.codecs.base import Codec
from kaos_llm_core.programs.call import Call, current_invocation
from kaos_llm_core.settings import KaosLLMCoreSettings
from kaos_llm_core.signatures.fields import OutputField
from kaos_llm_core.signatures.introspection import (
    create_output_model,
    get_instruction,
)
from kaos_llm_core.signatures.signature import Signature
from kaos_llm_core.types import Example

logger = get_logger(__name__)


def _inject_reasoning_field(sig: type[Signature]) -> type[Signature]:
    """Create a Signature subclass with a 'reasoning' OutputField added."""
    if "reasoning" in sig.model_fields:
        return sig

    from pydantic import create_model

    field_defs: dict[str, Any] = {
        "reasoning": (
            str,
            OutputField(
                description=(
                    "Step-by-step reasoning before answering. Think through the problem carefully."
                ),
            ),
        ),
    }
    for name, field_info in sig.model_fields.items():
        field_defs[name] = (field_info.annotation, field_info)

    return create_model(
        sig.__name__ + "WithReasoning",
        __base__=Signature,
        __doc__=get_instruction(sig),
        **field_defs,
    )


class ChainOfThought(Call):
    """Call with automatic chain-of-thought reasoning.

    Example::

        cot = ChainOfThought(ClassifyRisk, model="anthropic:claude-sonnet-4-6")
        result = await cot(text="Complex scenario...")
        print(result.reasoning)  # Step-by-step thinking
        print(result.level)     # Final classification
    """

    def __init__(
        self,
        signature: type[Signature],
        *,
        model: str | None = None,
        codec: Codec | None = None,
        client: BaseProviderClient | None = None,
        settings: KaosLLMSettings | None = None,
        core_settings: KaosLLMCoreSettings | None = None,
        examples: list[Example] | None = None,
        instructions: str | None = None,
        max_retries: int | None = None,
        **kwargs: Any,
    ) -> None:
        self._original_signature = signature

        cot_sig = _inject_reasoning_field(signature)
        base_instruction = instructions or get_instruction(signature)
        self._base_instruction = base_instruction  # preserved for native-thinking path
        cot_instruction = (
            f"{base_instruction}\n\n"
            "Think step-by-step. First explain your reasoning in the 'reasoning' field, "
            "then provide your final answer in the remaining fields."
        )

        super().__init__(
            cot_sig,
            model=model,
            codec=codec,
            client=client,
            settings=settings,
            core_settings=core_settings,
            examples=examples,
            instructions=cot_instruction,
            max_retries=max_retries,
            **kwargs,
        )

        # Cache output models for both paths
        self._original_output_model = create_output_model(signature)
        self._combined_output_model = self._output_model  # includes reasoning

    # ---------------------------------------------------------------
    # Per-execution scratch — populated into Invocation.extras
    # ---------------------------------------------------------------

    def _build_invocation_extras(self, *, client: Any, model: str) -> dict[str, Any]:
        """Detect native-thinking support from the resolved client.

        Called once at the top of ``Call._run_pipeline`` (and once at the
        top of ``prepare_call`` so the plan reflects the same effective
        signature). The result lands on ``Invocation.extras["native_thinking"]``,
        accessible from step methods via :func:`current_invocation`.
        """
        return {
            "native_thinking": bool(
                getattr(client, "profile", None) and client.profile.supports_thinking
            ),
        }

    def _is_native_thinking(self) -> bool:
        """Read native-thinking flag from the active Invocation extras.

        Returns ``False`` outside an active execution scope (a defensive
        default for direct step-method invocation that should not happen
        in practice).
        """
        invocation = current_invocation()
        if invocation is None:
            return False
        return bool(invocation.extras.get("native_thinking", False))

    def _active_client(self) -> Any:
        """Read the resolved client from the active Invocation."""
        invocation = current_invocation()
        if invocation is None:
            return None
        return invocation.client

    # ---------------------------------------------------------------
    # Step-method overrides — read from Invocation extras
    # ---------------------------------------------------------------

    def _prepare_call_kwargs(self) -> dict[str, Any]:
        """Inject thinking parameters when native thinking is available."""
        kwargs = super()._prepare_call_kwargs()
        if not self._is_native_thinking():
            return kwargs
        client = self._active_client()
        if client is None:
            return kwargs

        param = client.profile.thinking_parameter
        # Reasoning models in 2026 burn 30-100K+ tokens on hidden CoT;
        # 8K caps were a Sonnet-3.5 / o1-preview-era default. 32K gives
        # the model real room to think while still leaving headroom for
        # the visible answer under the per-profile max_tokens ceiling.
        if param == "thinking":
            kwargs["thinking"] = True
            kwargs.setdefault("max_tokens", 32_768)
        elif param == "reasoning":
            kwargs.setdefault("reasoning", {"effort": "medium"})
            kwargs.setdefault("max_completion_tokens", 32_768)
        elif param == "thinkingConfig":
            kwargs.setdefault("thinkingConfig", {"thinkingBudget": 32_768})
        elif param:
            logger.warning("Unknown thinking parameter %r; passing %s=True", param, param)
            kwargs[param] = True

        return kwargs

    def _get_effective_signature(self) -> type[Signature]:
        """Use original signature for native thinking (no reasoning in schema)."""
        if self._is_native_thinking():
            return self._original_signature
        return self.signature

    def _get_effective_instructions(self) -> str:
        """Use caller-supplied base instructions for native thinking (no CoT suffix)."""
        if self._is_native_thinking():
            return self._base_instruction
        return self.instructions

    def _get_output_model(self) -> type[BaseModel]:
        """Use original output model for native thinking."""
        if self._is_native_thinking():
            return self._original_output_model
        return self._combined_output_model

    def _post_process(
        self, result: Any, response: ProviderResponse, output_dict: dict[str, Any]
    ) -> Any:
        """Attach native thinking text as reasoning when using native path."""
        if self._is_native_thinking():
            reasoning_text = response.thinking or ""
            combined_dict = {**output_dict, "reasoning": reasoning_text}
            return self._combined_output_model.model_validate(combined_dict)
        return result
