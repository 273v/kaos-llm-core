"""Unit tests for the ReAct program (kaos-llm-core 0.1.0, Phase 5.2).

Implements the 9 tests pinned in ``docs/internal/design/react-loop.md`` §7.
``FunctionClient`` from ``kaos-llm-client`` is used to deterministically
simulate model responses without HTTP. Real ``ToolDefinition``,
``ToolCall``, ``ContentPart``, and ``ProviderResponse`` objects are used
end-to-end so the tests exercise the same wire format as live providers.

``asyncio_mode = "auto"`` (configured in pyproject.toml) lets ``async``
tests run without explicit ``@pytest.mark.asyncio`` decorators.
"""

from __future__ import annotations

import json
from typing import Any, Literal

import pytest
from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart, ToolCall

from kaos_llm_core.programs.react import (
    Iteration,
    ReAct,
    ReActResult,
    ToolObservation,
)
from kaos_llm_core.programs.tool import Tool
from kaos_llm_core.signatures import InputField, OutputField, Signature

# ---------------------------------------------------------------------------
# Test signatures
# ---------------------------------------------------------------------------


class AnswerSignature(Signature):
    """Answer the user's question, optionally calling tools."""

    question: str = InputField(description="The question to answer")
    answer: str = OutputField(description="The final answer")


# ---------------------------------------------------------------------------
# Helpers — build deterministic ProviderResponses for FunctionClient
# ---------------------------------------------------------------------------


def _text_response(text: str) -> ProviderResponse:
    """Build a text-only response (no tool calls)."""
    return ProviderResponse.model_construct(
        provider="function",
        model="function-test",
        raw={},
        parts=[ContentPart.model_construct(type="text", text=text)],
        usage=UsageInfo.model_construct(input_tokens=10, output_tokens=5, total_tokens=15),
        stop_reason="end_turn",
        status_code=200,
        response_headers={},
    )


def _json_response(data: dict[str, Any]) -> ProviderResponse:
    """Build a text response whose body is the JSON encoding of ``data``.

    Used to satisfy ``JSONCodec.decode`` on the final turn.
    """
    return _text_response(json.dumps(data))


def _tool_call_response(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    call_id: str = "call_1",
    text: str = "",
) -> ProviderResponse:
    """Build a response containing a single tool_use part."""
    parts: list[ContentPart] = []
    if text:
        parts.append(ContentPart.model_construct(type="text", text=text))
    parts.append(
        ContentPart.model_construct(
            type="tool_use",
            tool_call=ToolCall.model_construct(id=call_id, name=tool_name, arguments=arguments),
        )
    )
    return ProviderResponse.model_construct(
        provider="function",
        model="function-test",
        raw={},
        parts=parts,
        usage=UsageInfo.model_construct(input_tokens=12, output_tokens=8, total_tokens=20),
        stop_reason="tool_use",
        status_code=200,
        response_headers={},
    )


def _multi_tool_call_response(
    calls: list[tuple[str, dict[str, Any], str]],
) -> ProviderResponse:
    """Build a response containing multiple tool_use parts in order.

    Each entry is ``(tool_name, arguments, call_id)``.
    """
    parts: list[ContentPart] = []
    for name, args, cid in calls:
        parts.append(
            ContentPart.model_construct(
                type="tool_use",
                tool_call=ToolCall.model_construct(id=cid, name=name, arguments=args),
            )
        )
    return ProviderResponse.model_construct(
        provider="function",
        model="function-test",
        raw={},
        parts=parts,
        usage=UsageInfo.model_construct(input_tokens=15, output_tokens=12, total_tokens=27),
        stop_reason="tool_use",
        status_code=200,
        response_headers={},
    )


def _make_react(
    *,
    handler: Any,
    tools: list[Tool] | None = None,
    max_iterations: int = 10,
    on_tool_error: Literal["continue", "raise"] = "continue",
) -> ReAct:
    """Build a ReAct wired to a FunctionClient running ``handler``."""
    client = FunctionClient(model="function-test", function=handler)
    return ReAct(
        AnswerSignature,
        tools=tools or [],
        model="function-test",
        max_iterations=max_iterations,
        on_tool_error=on_tool_error,
        client=client,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReActNoTools:
    async def test_terminates_immediately_with_valid_answer(self) -> None:
        """No tools, valid Signature output on first call -> single iteration."""
        call_count = {"n": 0}

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            call_count["n"] += 1
            return _json_response({"answer": "42"})

        react = _make_react(handler=handler, tools=[])
        result = await react(question="What is the meaning of life?")

        assert isinstance(result, ReActResult)
        assert result.outputs == {"answer": "42"}
        assert result.stop_reason == "TERMINATED"
        assert result.iterations_used == 1
        assert call_count["n"] == 1
        assert len(result.trajectory) == 1
        assert result.trajectory[0].tool_calls == []
        assert result.trajectory[0].tool_results == []
        assert result.trajectory[0].error is None


class TestReActSingleToolSingleCall:
    async def test_single_tool_invoked_then_terminates(self) -> None:
        """Tool call on turn 1, final answer on turn 2."""
        call_count = {"n": 0}

        def add(a: int, b: int) -> int:
            """Add two integers."""
            return a + b

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _tool_call_response("add", {"a": 17, "b": 23}, call_id="call_1")
            return _json_response({"answer": "40"})

        react = _make_react(handler=handler, tools=[Tool.from_callable(add)])
        result = await react(question="What is 17 + 23?")

        assert result.outputs == {"answer": "40"}
        assert result.stop_reason == "TERMINATED"
        assert result.iterations_used == 2
        assert call_count["n"] == 2

        # First iteration ran the tool
        first = result.trajectory[0]
        assert len(first.tool_calls) == 1
        assert first.tool_calls[0].name == "add"
        assert len(first.tool_results) == 1
        obs = first.tool_results[0]
        assert obs.tool_name == "add"
        assert obs.arguments == {"a": 17, "b": 23}
        assert obs.result == 40
        assert obs.is_error is False

        # Second iteration finalized
        assert result.trajectory[1].tool_calls == []
        assert result.trajectory[1].tool_results == []


class TestReActToolErrorContinue:
    async def test_tool_raises_error_fed_back_loop_continues(self) -> None:
        """on_tool_error='continue' (default): exception becomes error envelope."""
        call_count = {"n": 0}

        def boom(x: int) -> int:
            """Always raises."""
            raise ValueError("simulated tool failure")

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _tool_call_response("boom", {"x": 1})
            # On turn 2, the model has seen the error envelope and answers.
            return _json_response({"answer": "tool failed, sorry"})

        react = _make_react(handler=handler, tools=[Tool.from_callable(boom)])
        result = await react(question="Run boom")

        assert result.stop_reason == "TERMINATED"
        assert result.iterations_used == 2

        obs = result.trajectory[0].tool_results[0]
        assert obs.is_error is True
        assert isinstance(obs.result, dict)
        assert obs.result.get("error") is True
        assert "simulated tool failure" in obs.result["message"]
        assert "ValueError" in obs.result["message"]


class TestReActToolErrorRaise:
    async def test_tool_raises_propagates_with_raise_mode(self) -> None:
        """on_tool_error='raise' lets the exception propagate to the caller."""

        def boom(x: int) -> int:
            """Always raises."""
            raise ValueError("fatal")

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _tool_call_response("boom", {"x": 1})

        react = _make_react(
            handler=handler,
            tools=[Tool.from_callable(boom)],
            on_tool_error="raise",
        )
        with pytest.raises(ValueError, match="fatal"):
            await react(question="Run boom")


class TestReActUnknownToolName:
    async def test_unknown_tool_name_becomes_error_observation(self) -> None:
        """Model hallucinates a tool name -> error envelope, loop continues."""
        call_count = {"n": 0}

        def known(x: int) -> int:
            """A real tool."""
            return x

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _tool_call_response("does_not_exist", {"x": 1})
            return _json_response({"answer": "I made up that tool, sorry"})

        react = _make_react(handler=handler, tools=[Tool.from_callable(known)])
        result = await react(question="Use a fake tool")

        assert result.stop_reason == "TERMINATED"
        assert result.iterations_used == 2

        obs = result.trajectory[0].tool_results[0]
        assert obs.tool_name == "does_not_exist"
        assert obs.is_error is True
        assert "Unknown tool" in obs.result["message"]
        assert "known" in obs.result["message"]


class TestReActMaxIterations:
    async def test_max_iterations_returns_partial_trajectory(self) -> None:
        """Loop overrun -> stop_reason=MAX_ITERATIONS, never raises."""
        call_count = {"n": 0}

        def loop_tool() -> str:
            """A no-op tool."""
            return "ok"

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            call_count["n"] += 1
            # Always return a tool call, never terminate.
            return _tool_call_response("loop_tool", {}, call_id=f"call_{call_count['n']}")

        react = _make_react(
            handler=handler,
            tools=[Tool.from_callable(loop_tool)],
            max_iterations=3,
        )
        result = await react(question="Loop forever")

        assert result.stop_reason == "MAX_ITERATIONS"
        assert result.iterations_used == 3
        assert len(result.trajectory) == 3
        # Best-effort decode failed because final response was a tool call
        assert result.outputs is None
        assert call_count["n"] == 3


class TestReActDecodeFailureRecovery:
    async def test_decode_failure_on_final_turn_then_retry_succeeds(self) -> None:
        """Malformed output -> corrective UserMessage -> next turn validates."""
        call_count = {"n": 0}

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            call_count["n"] += 1
            if call_count["n"] == 1:
                # Not valid JSON for the AnswerSignature
                return _text_response("this is not json at all")
            return _json_response({"answer": "fixed"})

        react = _make_react(handler=handler, tools=[])
        result = await react(question="Decode failure test")

        assert result.stop_reason == "TERMINATED"
        assert result.iterations_used == 2
        assert result.outputs == {"answer": "fixed"}
        # First iteration recorded an error
        assert result.trajectory[0].error is not None
        assert result.trajectory[1].error is None


class TestReActMultipleToolCallsPerTurn:
    async def test_multiple_tool_calls_executed_in_order(self) -> None:
        """Two tool calls in one turn -> both executed sequentially in order."""
        call_log: list[tuple[str, dict[str, Any]]] = []

        def first_tool(x: int) -> int:
            """First tool."""
            call_log.append(("first_tool", {"x": x}))
            return x * 2

        def second_tool(y: int) -> int:
            """Second tool."""
            call_log.append(("second_tool", {"y": y}))
            return y + 100

        call_count = {"n": 0}

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _multi_tool_call_response(
                    [
                        ("first_tool", {"x": 5}, "call_a"),
                        ("second_tool", {"y": 7}, "call_b"),
                    ]
                )
            return _json_response({"answer": "done"})

        react = _make_react(
            handler=handler,
            tools=[
                Tool.from_callable(first_tool),
                Tool.from_callable(second_tool),
            ],
        )
        result = await react(question="Run two tools")

        assert result.stop_reason == "TERMINATED"
        assert result.iterations_used == 2
        # Tools were called in the order the model emitted them
        assert call_log == [
            ("first_tool", {"x": 5}),
            ("second_tool", {"y": 7}),
        ]
        # Both observations recorded in order
        first_iter = result.trajectory[0]
        assert len(first_iter.tool_results) == 2
        assert first_iter.tool_results[0].tool_name == "first_tool"
        assert first_iter.tool_results[0].result == 10
        assert first_iter.tool_results[0].tool_call_id == "call_a"
        assert first_iter.tool_results[1].tool_name == "second_tool"
        assert first_iter.tool_results[1].result == 107
        assert first_iter.tool_results[1].tool_call_id == "call_b"


class TestReActResultAttributeAccess:
    async def test_attribute_access_forwards_to_outputs(self) -> None:
        """``result.answer`` works when ``outputs`` contains ``answer``."""

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _json_response({"answer": "forwarded"})

        react = _make_react(handler=handler, tools=[])
        result = await react(question="Test attribute access")

        # Forwarded to outputs
        assert result.answer == "forwarded"  # type: ignore[attr-defined]
        # Direct dataclass field still works
        assert result.outputs == {"answer": "forwarded"}
        # Missing attribute raises
        with pytest.raises(AttributeError):
            _ = result.does_not_exist  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Sanity checks on the dataclasses themselves
# ---------------------------------------------------------------------------


class TestDataclassShape:
    def test_iteration_defaults(self) -> None:
        it = Iteration(iteration=0, text="")
        assert it.tool_calls == []
        assert it.tool_results == []
        assert it.error is None

    def test_tool_observation_fields(self) -> None:
        obs = ToolObservation(
            tool_call_id="x",
            tool_name="t",
            arguments={"k": 1},
            result={"v": 2},
            is_error=False,
        )
        assert obs.tool_call_id == "x"
        assert obs.tool_name == "t"
        assert obs.arguments == {"k": 1}
        assert obs.result == {"v": 2}
        assert obs.is_error is False

    def test_react_result_fields(self) -> None:
        r = ReActResult(
            outputs={"answer": "a"},
            trajectory=[],
            stop_reason="TERMINATED",
            iterations_used=0,
        )
        assert r.outputs == {"answer": "a"}
        assert r.stop_reason == "TERMINATED"
        assert r.iterations_used == 0
