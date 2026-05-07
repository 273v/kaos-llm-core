"""Typed settings for kaos-llm-core.

Resolution priority:
1. Explicit overrides via ``from_context()``
2. ``KaosContext._config`` entries
3. Environment variables (``KAOS_LLM_CORE_`` prefix)
4. ``.env`` file
5. Field defaults

Phase 8.3 adds a ``profile`` field (``default`` / ``dev`` / ``staging`` /
``prod``). The profile is advisory — downstream consumers may branch on
``settings.profile`` if they want profile-specific behavior. The global
``KAOS_PROFILE`` environment variable is honored as a fallback when the
module-specific ``KAOS_LLM_CORE_PROFILE`` is unset, so one global env var
can set the profile for every KAOS module that opts in.
"""

from __future__ import annotations

import os
from typing import Any, Literal

from kaos_core.config import ModuleSettings
from pydantic import model_validator
from pydantic_settings import SettingsConfigDict

ProfileName = Literal["default", "dev", "staging", "prod"]


class KaosLLMCoreSettings(ModuleSettings):
    """Configuration for kaos-llm-core.

    Env vars use the ``KAOS_LLM_CORE_`` prefix:
    - ``KAOS_LLM_CORE_DEFAULT_MODEL`` — default model for Calls without explicit model
    - ``KAOS_LLM_CORE_DEFAULT_MAX_RETRIES`` — validation retry attempts
    - ``KAOS_LLM_CORE_TRACE_ENABLED`` — enable execution tracing
    - ``KAOS_LLM_CORE_PROFILE`` — one of ``default``/``dev``/``staging``/``prod``
      (falls back to the global ``KAOS_PROFILE`` env var when unset)

    API keys and provider config live in ``KaosLLMSettings`` (kaos-llm-client).
    This settings class covers only kaos-llm-core-specific concerns.
    """

    # Default model for Calls that don't specify one
    default_model: str | None = None

    # Validation retry (distinct from transport retry in kaos-llm-client)
    default_max_retries: int = 1

    # Tracing — enable/disable. Trace export sinks are passed explicitly
    # at the call site (see ``observability.export``); there is no
    # settings-driven default trace directory in this release.
    trace_enabled: bool = True

    # Phase 8.3: deployment profile. Advisory — downstream consumers may
    # branch on this field for profile-specific behavior. The global
    # ``KAOS_PROFILE`` env var is honored as a fallback via the validator
    # below when ``KAOS_LLM_CORE_PROFILE`` is not set.
    profile: ProfileName = "default"

    model_config = SettingsConfigDict(
        env_prefix="KAOS_LLM_CORE_",
        env_file=".env",
        extra="ignore",
    )

    @model_validator(mode="before")
    @classmethod
    def _global_profile_fallback(cls, values: Any) -> Any:
        """Fall back to the global ``KAOS_PROFILE`` env var when the
        module-specific ``KAOS_LLM_CORE_PROFILE`` is not set.

        Precedence (highest first):
        1. Explicit ``profile=`` kwarg (already in ``values``)
        2. ``KAOS_LLM_CORE_PROFILE`` env var (picked up by pydantic-settings
           via ``env_prefix`` before this validator runs — but only when the
           value reaches the validator as a dict entry)
        3. ``KAOS_PROFILE`` env var (handled here)
        4. Field default ``"default"``
        """
        if not isinstance(values, dict):
            return values
        if values.get("profile"):
            return values
        module_specific = os.environ.get("KAOS_LLM_CORE_PROFILE")
        if module_specific:
            # Let pydantic-settings handle this via env_prefix; do nothing.
            return values
        global_profile = os.environ.get("KAOS_PROFILE")
        if global_profile:
            values["profile"] = global_profile
        return values
