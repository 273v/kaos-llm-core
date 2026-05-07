"""Tests for KaosLLMCoreSettings profile support (Phase 8.3)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from kaos_llm_core.settings import KaosLLMCoreSettings


class TestProfileDefaults:
    def test_default_profile(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("KAOS_LLM_CORE_PROFILE", raising=False)
        monkeypatch.delenv("KAOS_PROFILE", raising=False)
        settings = KaosLLMCoreSettings()
        assert settings.profile == "default"

    def test_explicit_kwarg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("KAOS_LLM_CORE_PROFILE", raising=False)
        monkeypatch.delenv("KAOS_PROFILE", raising=False)
        settings = KaosLLMCoreSettings(profile="prod")
        assert settings.profile == "prod"


class TestModuleSpecificEnvVar:
    def test_module_specific_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("KAOS_PROFILE", raising=False)
        monkeypatch.setenv("KAOS_LLM_CORE_PROFILE", "dev")
        settings = KaosLLMCoreSettings()
        assert settings.profile == "dev"

    def test_module_specific_wins_over_global(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAOS_PROFILE", "staging")
        monkeypatch.setenv("KAOS_LLM_CORE_PROFILE", "prod")
        settings = KaosLLMCoreSettings()
        assert settings.profile == "prod"


class TestGlobalFallback:
    def test_global_profile_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("KAOS_LLM_CORE_PROFILE", raising=False)
        monkeypatch.setenv("KAOS_PROFILE", "staging")
        settings = KaosLLMCoreSettings()
        assert settings.profile == "staging"

    def test_global_profile_dev(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("KAOS_LLM_CORE_PROFILE", raising=False)
        monkeypatch.setenv("KAOS_PROFILE", "dev")
        settings = KaosLLMCoreSettings()
        assert settings.profile == "dev"


class TestInvalidProfile:
    def test_invalid_profile_value_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAOS_LLM_CORE_PROFILE", "turbo")  # not in Literal
        with pytest.raises(ValidationError):
            KaosLLMCoreSettings()
