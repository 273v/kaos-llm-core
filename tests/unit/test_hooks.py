"""Unit tests for CallHooks — lifecycle hooks around Call._execute."""

from __future__ import annotations

import contextlib
import json
from typing import Any

from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart

from kaos_llm_core.programs.call import Call
from kaos_llm_core.programs.hooks import CallHooks, fire_hook
from kaos_llm_core.signatures import InputField, OutputField, Signature


class ClassifyRisk(Signature):
    """Classify risk."""

    text: str = InputField(description="Input text")
    level: str = OutputField(description="Risk level")


def _json_response(data: dict[str, Any]) -> ProviderResponse:
    text = json.dumps(data)
    return ProviderResponse.model_construct(
        provider="function",
        model="function-test",
        raw={},
        parts=[ContentPart.model_construct(type="text", text=text)],
        usage=UsageInfo.model_construct(input_tokens=10, output_tokens=5, total_tokens=15),
        stop_reason="end_turn",
        response_id="test-id",
        status_code=200,
        response_headers={},
        latency_ms=10.0,
    )


class TestHookFiringOrder:
    async def test_success_path_fires_start_and_end(self) -> None:
        events: list[str] = []

        def on_start(call: Any, inputs: Any) -> None:
            events.append("start")

        def on_end(call: Any, inputs: Any, invocation: Any) -> None:
            events.append("end")

        def on_error(*args: Any, **kwargs: Any) -> None:
            events.append("error")

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"level": "high"})

        client = FunctionClient(function=fn)
        hooks = CallHooks(on_call_start=on_start, on_call_end=on_end, on_call_error=on_error)
        call = Call(ClassifyRisk, model="function-test", client=client, hooks=hooks)
        result = await call(text="test")
        assert result.level == "high"  # type: ignore[attr-defined]
        assert events == ["start", "end"]

    async def test_error_path_fires_error(self) -> None:
        events: list[str] = []

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            raise RuntimeError("provider broke")

        client = FunctionClient(function=fn)
        hooks = CallHooks(
            on_call_start=lambda *a, **k: events.append("start"),
            on_call_end=lambda *a, **k: events.append("end"),
            on_call_error=lambda *a, **k: events.append("error"),
        )
        call = Call(ClassifyRisk, model="function-test", client=client, hooks=hooks)
        with contextlib.suppress(Exception):
            await call(text="test")
        assert "start" in events
        assert "error" in events
        assert "end" not in events


class TestValidationRetryHook:
    async def test_fires_on_retry(self) -> None:
        events: list[tuple[str, Any]] = []
        call_count = {"n": 0}

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _json_response({"not_level": "bad"})  # decode failure
            return _json_response({"level": "high"})

        client = FunctionClient(function=fn)

        def on_retry(call: Any, inputs: Any, attempt: int, err: Any) -> None:
            events.append(("retry", attempt))

        hooks = CallHooks(on_validation_retry=on_retry)
        call = Call(
            ClassifyRisk,
            model="function-test",
            client=client,
            hooks=hooks,
            max_retries=2,
        )
        result = await call(text="test")
        assert result.level == "high"  # type: ignore[attr-defined]
        assert any(e[0] == "retry" for e in events)


class TestHookErrorIsolation:
    async def test_hook_raising_does_not_crash_call(self) -> None:
        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"level": "high"})

        client = FunctionClient(function=fn)

        def bad_hook(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("hook explode")

        hooks = CallHooks(on_call_start=bad_hook, on_call_end=bad_hook)
        call = Call(ClassifyRisk, model="function-test", client=client, hooks=hooks)
        result = await call(text="test")
        assert result.level == "high"  # type: ignore[attr-defined]

    def test_fire_hook_none(self) -> None:
        # fire_hook with None is a no-op
        fire_hook(None)
        fire_hook(None, 1, 2, kw="x")


class TestHookContextPropagation:
    """§5.5 design decision: hooks receive a KaosContext kwarg for correlation."""

    async def test_context_kwarg_flows_through_all_hooks(self) -> None:
        """A context set via ``set_call_context`` reaches every hook firing."""
        from kaos_llm_core.programs.call import _call_context_var, set_call_context

        seen_contexts: dict[str, Any] = {}

        def on_start(call: Any, inputs: Any, *, context: Any = None) -> None:
            seen_contexts["start"] = context

        def on_end(call: Any, inputs: Any, invocation: Any, *, context: Any = None) -> None:
            seen_contexts["end"] = context

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"level": "low"})

        client = FunctionClient(function=fn)
        hooks = CallHooks(on_call_start=on_start, on_call_end=on_end)
        call = Call(ClassifyRisk, model="function-test", client=client, hooks=hooks)

        sentinel = object()
        token = set_call_context(sentinel)
        try:
            await call(text="test")
        finally:
            _call_context_var.reset(token)

        assert seen_contexts["start"] is sentinel
        assert seen_contexts["end"] is sentinel

    async def test_legacy_callback_without_context_still_fires(self) -> None:
        """Old-style callbacks (no ``context`` kwarg) must keep working.

        ``fire_hook`` falls back to invoking without ``context=`` when the
        callback rejects the kwarg, so existing callers don't break.
        """
        events: list[str] = []

        def legacy_on_start(call: Any, inputs: Any) -> None:
            # Note: NO **kwargs and NO context= parameter.
            events.append("start")

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"level": "low"})

        client = FunctionClient(function=fn)
        hooks = CallHooks(on_call_start=legacy_on_start)
        call = Call(ClassifyRisk, model="function-test", client=client, hooks=hooks)
        await call(text="test")
        assert events == ["start"]

    async def test_validation_retry_hook_receives_context(self) -> None:
        """on_validation_retry must also receive the context kwarg."""
        from kaos_llm_core.programs.call import _call_context_var, set_call_context

        retry_contexts: list[Any] = []
        call_count = {"n": 0}

        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _json_response({"not_level": "bad"})
            return _json_response({"level": "high"})

        def on_retry(
            call: Any, inputs: Any, attempt: int, err: Any, *, context: Any = None
        ) -> None:
            retry_contexts.append(context)

        client = FunctionClient(function=fn)
        hooks = CallHooks(on_validation_retry=on_retry)
        call = Call(
            ClassifyRisk,
            model="function-test",
            client=client,
            hooks=hooks,
            max_retries=2,
        )

        sentinel = object()
        token = set_call_context(sentinel)
        try:
            await call(text="test")
        finally:
            _call_context_var.reset(token)

        assert retry_contexts and retry_contexts[0] is sentinel


class TestHookNonInterference:
    async def test_trace_preserved(self) -> None:
        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"level": "high"})

        client = FunctionClient(function=fn)
        hooks = CallHooks(on_call_start=lambda *a, **k: None)
        call = Call(ClassifyRisk, model="function-test", client=client, hooks=hooks)
        invocation = await call.invoke(text="test")
        assert invocation.trace is not None
        assert invocation.trace.call_name == "ClassifyRisk"

    async def test_no_hooks_still_works(self) -> None:
        def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"level": "high"})

        client = FunctionClient(function=fn)
        call = Call(ClassifyRisk, model="function-test", client=client)
        result = await call(text="test")
        assert result.level == "high"  # type: ignore[attr-defined]
