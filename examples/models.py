"""Model presets per provider.

Centralizes model selection so examples can switch providers with --provider.
Models verified against kaos-llm-client/tests/integration/test_live.py (April 2026).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelPreset:
    cheap: str
    balanced: str


PROVIDERS: dict[str, ModelPreset] = {
    "anthropic": ModelPreset(
        cheap="anthropic:claude-haiku-4-5",
        balanced="anthropic:claude-sonnet-4-6",
    ),
    "openai": ModelPreset(
        cheap="openai:gpt-5.4-nano",
        balanced="openai:gpt-5.4-mini",
    ),
    "google": ModelPreset(
        cheap="google:gemini-2.5-flash",
        balanced="google:gemini-2.5-pro",
    ),
}

DEFAULT_PROVIDER = "anthropic"


def get_preset(provider: str | None = None) -> ModelPreset:
    """Get model preset for a provider name."""
    name = provider or DEFAULT_PROVIDER
    if name not in PROVIDERS:
        available = ", ".join(sorted(PROVIDERS.keys()))
        raise ValueError(f"Unknown provider {name!r}. Available: {available}")
    return PROVIDERS[name]
