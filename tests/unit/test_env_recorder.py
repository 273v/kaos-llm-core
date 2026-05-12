"""Unit tests for the cross-process env recorder (KC6).

Verifies that:
1. ``install_from_env()`` returns False when ``ENV_VAR`` is unset.
2. ``install_from_env()`` is idempotent — the second call is a no-op.
3. A patched ``Call._execute`` writes a JSONL line per Invocation.
4. The patch is additive — wrapping a pre-existing patch composes.
5. Serialization errors do not propagate (telemetry must not break
   execution).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from kaos_llm_core.observability.env_recorder import (
    ENV_VAR,
    SENTINEL_ATTR,
    _serialize,
    install_from_env,
)


@pytest.fixture
def restore_call_execute() -> Any:
    """Ensure Call._execute is restored after each test."""
    from kaos_llm_core.programs.call import Call

    original = Call._execute
    sentinel_was_set = getattr(Call, SENTINEL_ATTR, False)
    yield Call
    Call._execute = original
    if not sentinel_was_set and hasattr(Call, SENTINEL_ATTR):
        delattr(Call, SENTINEL_ATTR)


class TestInstallFromEnv:
    def test_returns_false_when_env_var_unset(
        self, monkeypatch: pytest.MonkeyPatch, restore_call_execute: Any
    ) -> None:
        monkeypatch.delenv(ENV_VAR, raising=False)
        assert install_from_env() is False

    def test_returns_false_when_env_var_empty(
        self, monkeypatch: pytest.MonkeyPatch, restore_call_execute: Any
    ) -> None:
        monkeypatch.setenv(ENV_VAR, "")
        assert install_from_env() is False

    def test_returns_false_when_env_var_whitespace(
        self, monkeypatch: pytest.MonkeyPatch, restore_call_execute: Any
    ) -> None:
        monkeypatch.setenv(ENV_VAR, "   ")
        assert install_from_env() is False

    def test_returns_true_when_env_var_set(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        restore_call_execute: Any,
    ) -> None:
        monkeypatch.setenv(ENV_VAR, str(tmp_path))
        assert install_from_env() is True
        assert getattr(restore_call_execute, SENTINEL_ATTR, False) is True

    def test_idempotent(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        restore_call_execute: Any,
    ) -> None:
        """Second call returns False — installation only happens once."""
        monkeypatch.setenv(ENV_VAR, str(tmp_path))
        assert install_from_env() is True
        assert install_from_env() is False

    def test_returns_false_when_dir_uncreatable(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        restore_call_execute: Any,
    ) -> None:
        def _raise_os_error(*_args: Any, **_kwargs: Any) -> None:
            raise OSError("cannot create recorder dir")

        monkeypatch.setenv(ENV_VAR, str(tmp_path / "recorder"))
        monkeypatch.setattr(Path, "mkdir", _raise_os_error)
        assert install_from_env() is False


class TestSerialize:
    def test_handles_invocation_with_trace(self) -> None:
        from dataclasses import dataclass

        @dataclass(slots=True)
        class _MockUsage:
            input_tokens: int = 10
            output_tokens: int = 5
            total_tokens: int = 15
            cost_usd: float = 0.001

        class _MockInvocation:
            id = "inv-123"
            model = "anthropic:claude-haiku-4-5"
            output = "hello"
            error = None
            trace = None
            usage = _MockUsage()

        record = _serialize(_MockInvocation())
        assert record["kind"] == "invocation"
        assert record["invocation_id"] == "inv-123"
        assert record["model"] == "anthropic:claude-haiku-4-5"
        assert record["error"] is None
        assert record["usage"]["input_tokens"] == 10
        assert record["usage"]["cost_usd"] == 0.001

    def test_handles_pydantic_output(self) -> None:
        from pydantic import BaseModel

        class _Out(BaseModel):
            text: str = "hi"

        class _MockInvocation:
            id = "inv-1"
            model = "test"
            output = _Out()
            error = None
            trace = None
            usage = None

        record = _serialize(_MockInvocation())
        assert record["output"] == {"text": "hi"}

    def test_swallows_serialization_failure_in_trace(self) -> None:
        """A broken model_dump on the trace must NOT raise."""

        class _BadTrace:
            def model_dump(self, mode: str = "python") -> dict[str, Any]:
                raise RuntimeError("boom")

        class _MockInvocation:
            id = "inv-1"
            model = "test"
            output = None
            error = None
            trace = _BadTrace()
            usage = None

        # Should not raise — sets trace to None and continues.
        record = _serialize(_MockInvocation())
        assert record["trace"] is None


class TestEndToEndPatch:
    @pytest.mark.asyncio
    async def test_patched_execute_writes_jsonl_on_success(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        restore_call_execute: Any,
    ) -> None:
        """When the env var is set + an Invocation runs, a JSONL line lands in the dir."""
        from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
        from kaos_llm_client.providers.function import FunctionClient
        from kaos_llm_client.types import ContentPart

        import kaos_llm_core.programs.call as call_mod
        from kaos_llm_core import InputField, OutputField, Signature
        from kaos_llm_core.programs.call import Call

        monkeypatch.setenv(ENV_VAR, str(tmp_path))
        assert install_from_env() is True

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return ProviderResponse.model_construct(
                provider="function",
                model="function-test",
                raw={},
                parts=[ContentPart.model_construct(type="text", text='{"answer": "ok"}')],
                usage=UsageInfo.model_construct(input_tokens=10, output_tokens=5, total_tokens=15),
                stop_reason="end_turn",
                status_code=200,
                response_headers={},
            )

        client = FunctionClient(model="function-test", function=handler)
        original = call_mod.create_client
        call_mod.create_client = lambda *a, **kw: client  # ty: ignore[invalid-assignment]
        try:

            class _Sig(Signature):
                """Test."""

                text: str = InputField(description="in")
                answer: str = OutputField(description="out")

            call = Call(_Sig, model="function-test")
            result = await call(text="hi")
            assert getattr(result, "answer", None) == "ok"
        finally:
            call_mod.create_client = original

        # Schema-v4 streaming: header line at install_from_env() +
        # one invocation line per completed Call. Schema bumped from
        # 3 to 4 in KC16-4 alongside transparency-lens redaction.
        jsonl_files = list(tmp_path.glob("subprocess-*.jsonl"))
        assert len(jsonl_files) == 1, f"expected 1 jsonl, got {jsonl_files}"
        lines = jsonl_files[0].read_text().splitlines()
        assert len(lines) == 2
        header = json.loads(lines[0])
        assert header["kind"] == "header"
        assert header["streaming"] is True
        assert header["schema_version"] == 4
        # KC16-4: redaction policy advertised in header.
        assert header["redaction_enabled"] is True
        assert header["redaction_threshold_chars"] == 2048
        record = json.loads(lines[1])
        assert record["kind"] == "invocation"
        assert record["model"] == "function-test"
        assert record["usage"]["total_tokens"] == 15
