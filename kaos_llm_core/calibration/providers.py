"""Provider-envvar mapping + default model sweep for calibration scripts.

The single source of truth for "which model id maps to which API key env
var" across calibration harnesses. Mirrors the ``requires_*`` skip
markers in ``tests/integration/conftest.py`` but is callable from a
script (which runs outside pytest).
"""

from __future__ import annotations

import os

# Default model sweep — cheapest current-gen per provider.
# Keep in sync with ``kaos-llm-client/tests/integration/test_live.py``.
DEFAULT_MODELS: tuple[str, ...] = (
    "anthropic:claude-haiku-4-5",
    "openai:gpt-5.4-nano",
    "google:gemini-2.5-flash",
)

PROVIDER_ENV: dict[str, tuple[str, ...]] = {
    "anthropic": ("KAOS_LLM_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    "openai": ("KAOS_LLM_OPENAI_API_KEY", "OPENAI_API_KEY"),
    "google": (
        "KAOS_LLM_GOOGLE_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_GENERATIVE_AI_API_KEY",
    ),
}


def provider_of(model: str) -> str:
    """Extract the provider prefix from a ``provider:model`` id."""
    return model.split(":", 1)[0] if ":" in model else model


def has_api_key_for(model: str) -> bool:
    """Return True if the provider's API key is set in the environment.

    Checks both the ``KAOS_LLM_*`` canonical names and the legacy
    provider-native names (``OPENAI_API_KEY``, etc.).
    """
    provider = provider_of(model)
    env_names = PROVIDER_ENV.get(provider, ())
    return any(os.getenv(name) for name in env_names)


__all__ = ["DEFAULT_MODELS", "PROVIDER_ENV", "has_api_key_for", "provider_of"]
