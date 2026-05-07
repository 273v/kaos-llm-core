"""Live integration tests for the ReAct program (Phase 5.2).

These hit real LLM provider APIs and verify the full ReAct loop:
``user prompt → tool call → tool result → final answer``. Each test is
gated on the corresponding API key. Pattern matches
``kaos-llm-client/tests/integration/test_live.py``.

Models used (April 2026 cheapest current-generation):
- Anthropic: ``claude-haiku-4-5``
- OpenAI:    ``gpt-5.4-nano``
- Google:    ``gemini-2.5-flash``

The calculator scenario asks the model to compute ``17 * 23`` (= 391) using
a ``multiply(a, b)`` tool. The retry scenario forces the tool to fail once
with ``{"error": True, ...}`` and verifies the model recovers.
"""

from __future__ import annotations

import os

import pytest
from kaos_llm_client.types import ToolDefinition

from kaos_llm_core.programs.react import ReAct, ReActResult
from kaos_llm_core.programs.tool import Tool
from kaos_llm_core.signatures import InputField, OutputField, Signature

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Skip markers — local copies so this file can be moved without dragging
# the conftest fixture along.
# ---------------------------------------------------------------------------


def _has_key(*env_vars: str) -> bool:
    return any(os.getenv(v) for v in env_vars)


requires_anthropic = pytest.mark.skipif(
    not _has_key("KAOS_LLM_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    reason="No Anthropic API key",
)
requires_openai = pytest.mark.skipif(
    not _has_key("KAOS_LLM_OPENAI_API_KEY", "OPENAI_API_KEY"),
    reason="No OpenAI API key",
)
requires_google = pytest.mark.skipif(
    not _has_key("KAOS_LLM_GOOGLE_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY"),
    reason="No Google API key",
)


# ---------------------------------------------------------------------------
# Shared signature + tool
# ---------------------------------------------------------------------------


class CalculatorAnswer(Signature):
    """You are a careful arithmetic assistant. When the user asks a math question,
    call the multiply tool to compute the product, then return the final answer
    as a string containing the numeric result.
    """

    question: str = InputField(description="A math question from the user")
    answer: str = OutputField(
        description="The final answer string. Must contain the numeric result."
    )


def _make_multiply_tool(*, google_compatible: bool = False) -> tuple[Tool, dict[str, int]]:
    """Build a deterministic multiply tool plus a call counter.

    Returns the Tool and a counter dict so tests can assert how many
    times the model invoked it.

    When ``google_compatible=True``, builds the ``ToolDefinition`` by hand
    so the parameters schema omits ``additionalProperties: false`` —
    Google's Gemini API rejects schemas containing that key.
    """
    counter = {"calls": 0}

    def multiply(a: float, b: float) -> dict[str, float]:
        """Multiply two numbers and return the product."""
        counter["calls"] += 1
        return {"product": a * b}

    if not google_compatible:
        return Tool.from_callable(multiply), counter

    definition = ToolDefinition(
        name="multiply",
        description="Multiply two numbers and return the product.",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        },
    )
    return Tool(definition=definition, executor=multiply), counter


def _assert_calculator_result(result: ReActResult) -> None:
    """Shared assertions for the 17*23 = 391 scenario."""
    assert result.outputs is not None, (
        f"ReAct returned no outputs (stop_reason={result.stop_reason}). "
        f"Trajectory had {len(result.trajectory)} iterations."
    )
    assert result.stop_reason == "TERMINATED", (
        f"Expected TERMINATED, got {result.stop_reason}. Trajectory: {result.trajectory!r}"
    )
    # At least one iteration should have a successful tool result
    success_iters = [
        it
        for it in result.trajectory
        if it.tool_results and any(not obs.is_error for obs in it.tool_results)
    ]
    assert success_iters, (
        f"Expected at least one iteration with a successful tool result. "
        f"Trajectory: {result.trajectory!r}"
    )
    # Final answer should contain the right product
    answer = str(result.outputs.get("answer", ""))
    assert "391" in answer, (
        f"Expected '391' in final answer, got {answer!r}. Trajectory: {result.trajectory!r}"
    )


# ---------------------------------------------------------------------------
# Live tests — one per provider, calculator scenario
# ---------------------------------------------------------------------------


@requires_anthropic
async def test_react_anthropic_calculator() -> None:
    """Anthropic claude-haiku-4-5 calls multiply(17, 23) and answers 391."""
    tool, counter = _make_multiply_tool()
    react = ReAct(
        CalculatorAnswer,
        tools=[tool],
        model="anthropic:claude-haiku-4-5",
        max_iterations=6,
    )
    result = await react(question="What is 17 multiplied by 23?")
    _assert_calculator_result(result)
    assert counter["calls"] >= 1, "Expected at least one call to multiply()"


@requires_openai
async def test_react_openai_calculator() -> None:
    """OpenAI gpt-5.4-nano calls multiply(17, 23) and answers 391."""
    tool, counter = _make_multiply_tool()
    react = ReAct(
        CalculatorAnswer,
        tools=[tool],
        model="openai:gpt-5.4-nano",
        max_iterations=6,
    )
    result = await react(question="What is 17 multiplied by 23?")
    _assert_calculator_result(result)
    assert counter["calls"] >= 1, "Expected at least one call to multiply()"


@requires_google
async def test_react_google_calculator() -> None:
    """Google gemini-2.5-flash calls multiply(17, 23) and answers 391."""
    tool, counter = _make_multiply_tool(google_compatible=True)
    react = ReAct(
        CalculatorAnswer,
        tools=[tool],
        model="google:gemini-2.5-flash",
        max_iterations=6,
    )
    result = await react(question="What is 17 multiplied by 23?")
    _assert_calculator_result(result)
    assert counter["calls"] >= 1, "Expected at least one call to multiply()"


# ---------------------------------------------------------------------------
# Live test — tool returns an error envelope first, then succeeds
# ---------------------------------------------------------------------------


@requires_anthropic
async def test_react_anthropic_tool_error_then_retry() -> None:
    """Tool RAISES first, succeeds on retry; model recovers via ReAct's
    exception envelope.

    Phase 9 audit fix #3 — ReAct's ``is_error`` flag is set by ReAct's
    dispatcher when a tool *raises*, NOT by inspecting the returned dict.
    A tool that legitimately returns ``{"error": True, "details": ...}``
    (e.g. wrapping a real API) is forwarded verbatim with
    ``is_error=False``. This test exercises the real exception path so
    we verify the dispatcher actually catches raises and the model
    recovers from the error envelope ReAct constructs.
    """
    counter = {"calls": 0}

    def multiply(a: float, b: float) -> float:
        """Multiply two numbers."""
        counter["calls"] += 1
        if counter["calls"] == 1:
            raise RuntimeError("rate limited; please retry the multiply call")
        return a * b

    tool = Tool.from_callable(multiply)
    react = ReAct(
        CalculatorAnswer,
        tools=[tool],
        model="anthropic:claude-haiku-4-5",
        max_iterations=8,
    )
    result = await react(question="What is 17 multiplied by 23?")

    assert result.stop_reason == "TERMINATED", (
        f"Expected TERMINATED, got {result.stop_reason}. Trajectory: {result.trajectory!r}"
    )
    assert result.outputs is not None
    answer = str(result.outputs.get("answer", ""))
    assert "391" in answer, (
        f"Expected '391' in final answer, got {answer!r}. Trajectory: {result.trajectory!r}"
    )

    # ReAct caught the raise -> is_error=True with the dispatcher's envelope
    error_obs = [obs for it in result.trajectory for obs in it.tool_results if obs.is_error]
    success_obs = [obs for it in result.trajectory for obs in it.tool_results if not obs.is_error]
    assert error_obs, "Expected ReAct to flag the first call as is_error after the raise"
    assert success_obs, "Expected at least one successful observation after retry"
    assert counter["calls"] >= 2, (
        f"Expected the tool to be invoked at least twice (got {counter['calls']})"
    )
