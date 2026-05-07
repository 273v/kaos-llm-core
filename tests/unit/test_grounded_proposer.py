"""Tests for GroundedInstructionProposer (Phase 17.1).

Mocked offline test — wires a :class:`FunctionClient` that returns
canned JSON responses for each sub-Signature, then asserts the
proposer (a) calls the right Signatures in the right order during
``build_static_context``, (b) emits ``n`` distinct draws via
``propose_n_instructions_for_call``, (c) samples a tip per draw,
(d) flags verbatim copies correctly, and (e) wires every enabled
context field into the generator inputs.

The canonical pattern for routing kaos-llm-core proposer Calls
through a FunctionClient is to ``patch("kaos_llm_client.create_client",
return_value=client)`` plus ``patch(
"kaos_llm_core.programs.call.create_client", return_value=client)``.
This is what test_mipro_lite.py uses for its own proposer flow.

Live integration coverage is in
``tests/integration/test_mipro_v2_live.py`` (Phase 17.1 C5).
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest
from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart

from kaos_llm_core.optimization.proposers.grounded import (
    TIPS,
    DescribeDataset,
    DescribeModule,
    DescribeProgram,
    GenerateSingleModuleInstruction,
    GroundedInstructionProposer,
)
from kaos_llm_core.programs.call import Call
from kaos_llm_core.signatures import InputField, OutputField, Signature
from kaos_llm_core.types import Example

# ---------------------------------------------------------------------------
# Test signature for the call we are "optimizing"
# ---------------------------------------------------------------------------


class _ClassifySig(Signature):
    """Classify a question into one of six TREC-6 categories."""

    question: str = InputField(description="The question to classify")
    category: str = OutputField(description="One of ABBR/DESC/ENTY/HUM/LOC/NUM")


# ---------------------------------------------------------------------------
# Recorder: routes proposer requests to canned responses by signature name
# ---------------------------------------------------------------------------


def _make_response(payload: dict[str, Any]) -> ProviderResponse:
    return ProviderResponse.model_construct(
        provider="function",
        model="function-test",
        raw={},
        parts=[ContentPart.model_construct(type="text", text=json.dumps(payload))],
        usage=UsageInfo.model_construct(input_tokens=20, output_tokens=10, total_tokens=30),
        stop_reason="end_turn",
        status_code=200,
        response_headers={},
    )


class _Recorder:
    """Routes proposer requests to canned responses by signature name.

    The kaos-llm-core JSONCodec includes the signature class name in
    the system prompt, so we can disambiguate which sub-Signature is
    being invoked.
    """

    def __init__(self) -> None:
        self.calls_by_sig: dict[str, int] = {}
        self.last_messages: list[list[dict[str, Any]]] = []
        self.canned: dict[str, dict[str, Any]] = {
            "DescribeDataset": {
                "dataset_description": (
                    "TREC-6 question classification: questions in English "
                    "labeled with one of six coarse categories."
                ),
            },
            "DescribeProgram": {
                "program_description": (
                    "A single LLM call that takes a question and returns "
                    "one of six TREC-6 category labels."
                ),
            },
            "DescribeModule": {
                "module_description": (
                    "The classification module that maps a single question to a category label."
                ),
            },
        }
        self._gen_counter = 0

    def __call__(self, messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
        self.last_messages.append(messages)
        system = next((m["content"] for m in messages if m.get("role") == "system"), "")
        for sig_name in (
            "DescribeDataset",
            "DescribeProgram",
            "DescribeModule",
            "GenerateSingleModuleInstruction",
        ):
            if sig_name in system:
                self.calls_by_sig[sig_name] = self.calls_by_sig.get(sig_name, 0) + 1
                if sig_name == "GenerateSingleModuleInstruction":
                    self._gen_counter += 1
                    return _make_response(
                        {
                            "proposed_instruction": (
                                f"Classify the question into one of six TREC-6 "
                                f"categories. Variant {self._gen_counter}."
                            ),
                            "rationale": f"Variant {self._gen_counter}.",
                        }
                    )
                return _make_response(self.canned[sig_name])
        # Fallback (the target Call is not patched, so its requests
        # also flow through here when the test wires the client up).
        return _make_response({"category": "DESC"})


def _proposer(**overrides: Any) -> GroundedInstructionProposer:
    """Construct a proposer wired to the function-test model id."""
    kwargs: dict[str, Any] = {
        "proposer_model": "function-test",
        "program_aware": True,
        "data_aware": True,
        "tip_aware": True,
        "fewshot_aware": True,
        "view_data_batch_size": 3,
        "seed": 7,
    }
    kwargs.update(overrides)
    return GroundedInstructionProposer(**kwargs)


def _patch_client(client: FunctionClient) -> Any:
    """Return a context manager that routes every Call through the
    provided FunctionClient. Mirrors the test_mipro_lite.py pattern.
    """
    from contextlib import ExitStack

    stack = ExitStack()
    stack.enter_context(patch("kaos_llm_client.create_client", return_value=client))
    stack.enter_context(patch("kaos_llm_core.programs.call.create_client", return_value=client))
    return stack


@pytest.fixture
def train_set() -> list[Example]:
    return [
        Example(inputs={"question": "What is the capital of France?"}, outputs={"category": "LOC"}),
        Example(inputs={"question": "Who wrote 1984?"}, outputs={"category": "HUM"}),
        Example(inputs={"question": "When did WWII end?"}, outputs={"category": "NUM"}),
        Example(inputs={"question": "What does NASA stand for?"}, outputs={"category": "ABBR"}),
        Example(inputs={"question": "Define entropy."}, outputs={"category": "DESC"}),
    ]


def _target_call() -> Call:
    """A bare Call we treat as the optimization target. The proposer
    reads its signature/instructions but does not invoke it."""
    return Call(_ClassifySig, model="function-test", instructions="Pick a category.")


# ---------------------------------------------------------------------------
# Static context build
# ---------------------------------------------------------------------------


class TestStaticContext:
    async def test_full_context_build_invokes_all_three(self, train_set: list[Example]) -> None:
        rec = _Recorder()
        client = FunctionClient(function=rec)
        proposer = _proposer()
        with _patch_client(client):
            await proposer.build_static_context(call=_target_call(), train_set=train_set)

        assert rec.calls_by_sig.get("DescribeDataset") == 1
        assert rec.calls_by_sig.get("DescribeProgram") == 1
        assert rec.calls_by_sig.get("DescribeModule") == 1
        assert rec.calls_by_sig.get("GenerateSingleModuleInstruction", 0) == 0
        assert "TREC-6" in proposer.dataset_description
        assert "LLM call" in proposer.program_description
        assert "classification" in proposer.module_description

    async def test_data_aware_off_skips_dataset_summary(self, train_set: list[Example]) -> None:
        rec = _Recorder()
        client = FunctionClient(function=rec)
        proposer = _proposer(data_aware=False)
        with _patch_client(client):
            await proposer.build_static_context(call=_target_call(), train_set=train_set)
        assert "DescribeDataset" not in rec.calls_by_sig
        assert proposer.dataset_description == ""

    async def test_program_aware_off_skips_program_and_module(
        self, train_set: list[Example]
    ) -> None:
        rec = _Recorder()
        client = FunctionClient(function=rec)
        proposer = _proposer(program_aware=False)
        with _patch_client(client):
            await proposer.build_static_context(call=_target_call(), train_set=train_set)
        assert "DescribeProgram" not in rec.calls_by_sig
        assert "DescribeModule" not in rec.calls_by_sig
        assert proposer.program_description == ""
        assert proposer.module_description == ""


# ---------------------------------------------------------------------------
# Per-draw instruction proposal
# ---------------------------------------------------------------------------


class TestProposeNInstructions:
    async def test_n_draws_returns_n_results(self, train_set: list[Example]) -> None:
        rec = _Recorder()
        client = FunctionClient(function=rec)
        proposer = _proposer()
        target = _target_call()
        with _patch_client(client):
            await proposer.build_static_context(call=target, train_set=train_set)
            results = await proposer.propose_n_instructions_for_call(call=target, n=4)

        assert len(results) == 4
        assert rec.calls_by_sig.get("GenerateSingleModuleInstruction") == 4
        # Each draw produces a distinct instruction (the recorder
        # increments a counter, so each one ends with "Variant N.").
        instructions = [r.instruction for r in results]
        assert len(set(instructions)) == 4
        # All draws have a tip from the bank.
        for r in results:
            assert r.tip in TIPS
        # Rationale is captured.
        for r in results:
            assert "variant" in r.rationale.lower()
        # None should be marked as verbatim copies (the recorder
        # always returns a fresh string different from basic_instruction).
        assert not any(r.verbatim_copy for r in results)

    async def test_must_build_static_context_first(self) -> None:
        proposer = _proposer()
        with pytest.raises(RuntimeError, match="build_static_context"):
            await proposer.propose_n_instructions_for_call(call=_target_call(), n=2)

    async def test_n_zero_rejected(self, train_set: list[Example]) -> None:
        rec = _Recorder()
        client = FunctionClient(function=rec)
        proposer = _proposer()
        target = _target_call()
        with _patch_client(client):
            await proposer.build_static_context(call=target, train_set=train_set)
            with pytest.raises(ValueError, match="n must be"):
                await proposer.propose_n_instructions_for_call(call=target, n=0)


# ---------------------------------------------------------------------------
# Tip sampling: tip_aware off => always 'none'
# ---------------------------------------------------------------------------


class TestTipSampling:
    async def test_tip_aware_off_uses_none(self, train_set: list[Example]) -> None:
        rec = _Recorder()
        client = FunctionClient(function=rec)
        proposer = _proposer(tip_aware=False)
        target = _target_call()
        with _patch_client(client):
            await proposer.build_static_context(call=target, train_set=train_set)
            results = await proposer.propose_n_instructions_for_call(call=target, n=3)

        for r in results:
            assert r.tip == "none"


# ---------------------------------------------------------------------------
# Verbatim-copy detection
# ---------------------------------------------------------------------------


class TestVerbatimCopy:
    async def test_verbatim_copy_detected(self, train_set: list[Example]) -> None:
        """If the proposer returns the basic_instruction unchanged,
        ``ProposedInstruction.verbatim_copy`` must be True so the
        mutation log can flag it for post-hoc analysis."""

        target = _target_call()
        basic = target.instructions

        class _CopyRec(_Recorder):
            def __call__(
                self, messages: list[dict[str, Any]], profile: ModelProfile
            ) -> ProviderResponse:
                self.last_messages.append(messages)
                system = next((m["content"] for m in messages if m.get("role") == "system"), "")
                if "GenerateSingleModuleInstruction" in system:
                    return _make_response({"proposed_instruction": basic, "rationale": "verbatim"})
                # Static context: fall through to canned responses.
                return super().__call__(messages, profile)

        rec = _CopyRec()
        client = FunctionClient(function=rec)
        proposer = _proposer()
        with _patch_client(client):
            await proposer.build_static_context(call=target, train_set=train_set)
            results = await proposer.propose_n_instructions_for_call(call=target, n=2)

        assert all(r.verbatim_copy for r in results)


# ---------------------------------------------------------------------------
# Sub-signature wiring sanity (no LLM calls — pure schema check)
# ---------------------------------------------------------------------------


class TestSignatureWiring:
    """Verify the four Signatures expose the field set the design doc
    requires. These are pure schema introspection — no LLM calls."""

    def test_describe_dataset_fields(self) -> None:
        from kaos_llm_core.signatures.introspection import (
            get_input_fields,
            get_output_fields,
        )

        assert set(get_input_fields(DescribeDataset)) == {"examples_text"}
        assert set(get_output_fields(DescribeDataset)) == {"dataset_description"}

    def test_describe_program_fields(self) -> None:
        from kaos_llm_core.signatures.introspection import (
            get_input_fields,
            get_output_fields,
        )

        assert set(get_input_fields(DescribeProgram)) == {
            "program_code",
            "program_example",
        }
        assert set(get_output_fields(DescribeProgram)) == {"program_description"}

    def test_describe_module_fields(self) -> None:
        from kaos_llm_core.signatures.introspection import (
            get_input_fields,
            get_output_fields,
        )

        assert set(get_input_fields(DescribeModule)) == {
            "program_code",
            "program_example",
            "program_description",
            "module",
        }
        assert set(get_output_fields(DescribeModule)) == {"module_description"}

    def test_generate_single_module_instruction_fields(self) -> None:
        from kaos_llm_core.signatures.introspection import (
            get_input_fields,
            get_output_fields,
        )

        inputs = set(get_input_fields(GenerateSingleModuleInstruction))
        outputs = set(get_output_fields(GenerateSingleModuleInstruction))
        # Every context source from the design doc table is wired in.
        assert inputs == {
            "dataset_description",
            "program_code",
            "program_description",
            "module",
            "module_description",
            "task_demos",
            "basic_instruction",
            "tip",
        }
        assert outputs == {"proposed_instruction", "rationale"}


# ---------------------------------------------------------------------------
# TIPS bank sanity
# ---------------------------------------------------------------------------


class TestTipsBank:
    def test_six_tips_present(self) -> None:
        # Verbatim parity with DSPy grounded_proposer.py:17.
        assert set(TIPS.keys()) == {
            "none",
            "creative",
            "simple",
            "description",
            "high_stakes",
            "persona",
        }

    def test_none_is_empty(self) -> None:
        assert TIPS["none"] == ""

    def test_other_tips_nonempty(self) -> None:
        for key, value in TIPS.items():
            if key == "none":
                continue
            assert value, f"TIP {key!r} unexpectedly empty"
