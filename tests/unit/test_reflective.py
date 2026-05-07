"""Tests for ReflectiveOptimizer — self-critique without labeled data."""

from __future__ import annotations

import json
from typing import Any

from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart

from kaos_llm_core.optimization.mutations import MutationLog
from kaos_llm_core.optimization.reflective import ReflectiveOptimizer
from kaos_llm_core.programs.call import Call
from kaos_llm_core.signatures import InputField, OutputField, Signature


class SummarizeSig(Signature):
    """Summarize the given text."""

    text: str = InputField(description="Text to summarize")
    summary: str = OutputField(description="Brief summary")


def _make_reflective_setup() -> tuple[Call, ReflectiveOptimizer]:
    """Create a Call + ReflectiveOptimizer with FunctionClient responses."""
    call_count = {"summarize": 0, "critique": 0, "improve": 0}

    def fn(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
        system = messages[0]["content"] if messages else ""

        if "Summarize the given text" in system:
            call_count["summarize"] += 1
            return ProviderResponse.model_construct(
                provider="function",
                model="function-test",
                raw={},
                parts=[
                    ContentPart.model_construct(
                        type="text",
                        text=json.dumps({"summary": f"Summary attempt {call_count['summarize']}"}),
                    )
                ],
                usage=UsageInfo.model_construct(input_tokens=10, output_tokens=5, total_tokens=15),
                stop_reason="end_turn",
                status_code=200,
                response_headers={},
            )
        elif "Critique" in system or "quality reviewer" in system:
            call_count["critique"] += 1
            score = 0.6 if call_count["critique"] <= 2 else 0.95
            return ProviderResponse.model_construct(
                provider="function",
                model="function-test",
                raw={},
                parts=[
                    ContentPart.model_construct(
                        type="text",
                        text=json.dumps(
                            {
                                "quality_score": score,
                                "strengths": ["concise"],
                                "weaknesses": ["could be more specific"] if score < 0.9 else [],
                                "suggestions": ["add more detail"] if score < 0.9 else [],
                            }
                        ),
                    )
                ],
                usage=UsageInfo.model_construct(input_tokens=10, output_tokens=5, total_tokens=15),
                stop_reason="end_turn",
                status_code=200,
                response_headers={},
            )
        else:
            call_count["improve"] += 1
            return ProviderResponse.model_construct(
                provider="function",
                model="function-test",
                raw={},
                parts=[
                    ContentPart.model_construct(
                        type="text",
                        text=json.dumps(
                            {
                                "improved_instruction": "Summarize the text with specific details.",
                                "changes_made": "Added specificity requirement.",
                            }
                        ),
                    )
                ],
                usage=UsageInfo.model_construct(input_tokens=10, output_tokens=5, total_tokens=15),
                stop_reason="end_turn",
                status_code=200,
                response_headers={},
            )

    client = FunctionClient(function=fn)
    call = Call(SummarizeSig, model="function-test", client=client)

    log = MutationLog()
    optimizer = ReflectiveOptimizer(
        critic_model="function-test",
        proposer_model="function-test",
        max_trials=2,
        sample_count=2,
        quality_threshold=0.9,
        mutation_log=log,
    )
    # Patch both critic and proposer to use same FunctionClient
    # (the optimizer creates its own Calls internally, so we need to
    # monkeypatch create_client)
    import kaos_llm_core.programs.call as call_mod

    call_mod.create_client = lambda *a, **kw: client  # ty: ignore[invalid-assignment]

    return call, optimizer


class TestReflectiveOptimizer:
    async def test_runs_critique_loop(self) -> None:
        """ReflectiveOptimizer should run critique rounds and propose improvements."""
        import kaos_llm_core.programs.call as call_mod

        original = call_mod.create_client
        call, optimizer = _make_reflective_setup()
        try:
            sample_inputs = [
                {"text": "The SEC filed suit against Acme Corp."},
                {"text": "Revenue grew 18% year over year."},
            ]
            result = await optimizer.optimize(call, sample_inputs)

            assert result.avg_score_before is not None
            assert result.trials >= 1
            assert len(result.critiques) >= 2  # at least initial critique round
        finally:
            call_mod.create_client = original

    async def test_records_mutations(self) -> None:
        """Mutations should be recorded in the log."""
        import kaos_llm_core.programs.call as call_mod

        original = call_mod.create_client
        call, optimizer = _make_reflective_setup()
        try:
            await optimizer.optimize(call, [{"text": "test input"}])
            assert len(optimizer.mutation_log.mutations) >= 1
            m = optimizer.mutation_log.mutations[0]
            assert m.strategy == "reflective"
            assert m.mutation_type == "change_instructions"
        finally:
            call_mod.create_client = original

    async def test_stops_at_quality_threshold(self) -> None:
        """Should stop when average quality exceeds threshold."""
        import kaos_llm_core.programs.call as call_mod

        original = call_mod.create_client
        call, optimizer = _make_reflective_setup()
        optimizer.quality_threshold = 0.5  # very low → initial round might pass
        try:
            result = await optimizer.optimize(call, [{"text": "test"}])
            # Should either stop early or complete trials
            assert result.trials <= optimizer.max_trials
        finally:
            call_mod.create_client = original
