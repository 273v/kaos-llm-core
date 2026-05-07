"""Unit tests for the Phase 11 :class:`ProgramGraph` and the auto-registration
machinery on :meth:`Program.__setattr__`.

The audit flagged that the only existing coverage for ProgramGraph was a
single declaration-order assertion buried in test_audit_phase9.py. This file
exercises the contract directly: registry build, primary() ordering,
unregister-on-None, multi-name aliasing, private attribute skip, and the
program_hooks bypass.
"""

from __future__ import annotations

from typing import Any

import pytest

from kaos_llm_core.programs.base import Program
from kaos_llm_core.programs.call import Call
from kaos_llm_core.programs.graph import ProgramGraph
from kaos_llm_core.programs.program_hooks import ProgramHooks
from kaos_llm_core.signatures import InputField, OutputField, Signature


class _Sig(Signature):
    """Trivial signature."""

    text: str = InputField(description="Input")
    answer: str = OutputField(description="Output")


def _make_call() -> Call:
    return Call(_Sig, model="function-test")


# ---------------------------------------------------------------------------
# Auto-registration via __setattr__
# ---------------------------------------------------------------------------


class TestAutoRegistration:
    def test_public_call_attribute_is_registered(self) -> None:
        class P(Program):
            def __init__(self) -> None:
                self.extract = _make_call()

        prog = P()
        assert "extract" in prog.named_calls()
        assert prog.named_calls()["extract"] is prog.extract

    def test_public_program_attribute_is_registered(self) -> None:
        class Inner(Program):
            async def forward(self, **kwargs: Any) -> Any:
                return None

        class Outer(Program):
            def __init__(self) -> None:
                self.inner = Inner()

        outer = Outer()
        assert "inner" in outer.named_calls()
        assert isinstance(outer.named_calls()["inner"], Program)

    def test_private_attribute_is_not_registered(self) -> None:
        class P(Program):
            def __init__(self) -> None:
                self._hidden = _make_call()

        prog = P()
        assert prog.named_calls() == {}

    def test_program_hooks_attribute_does_not_pollute_registry(self) -> None:
        class P(Program):
            def __init__(self) -> None:
                self.program_hooks = ProgramHooks()
                self.real = _make_call()

        prog = P()
        names = prog.named_calls()
        assert "program_hooks" not in names
        assert "real" in names

    def test_non_call_public_attribute_is_not_registered(self) -> None:
        class P(Program):
            def __init__(self) -> None:
                self.label = "hello"
                self.numbers = [1, 2, 3]
                self.work = _make_call()

        prog = P()
        names = prog.named_calls()
        assert "label" not in names
        assert "numbers" not in names
        assert "work" in names

    def test_self_reference_is_skipped(self) -> None:
        class P(Program):
            def __init__(self) -> None:
                self.me = self  # type: ignore[assignment]
                self.real = _make_call()

        prog = P()
        names = prog.named_calls()
        assert "me" not in names
        assert "real" in names


# ---------------------------------------------------------------------------
# Reassignment semantics
# ---------------------------------------------------------------------------


class TestReassignment:
    def test_reassigning_to_none_unregisters(self) -> None:
        class P(Program):
            def __init__(self) -> None:
                self.extract = _make_call()

        prog = P()
        assert "extract" in prog.named_calls()
        prog.extract = None  # type: ignore[assignment]
        assert "extract" not in prog.named_calls()

    def test_reassigning_to_string_unregisters(self) -> None:
        class P(Program):
            def __init__(self) -> None:
                self.extract = _make_call()

        prog = P()
        prog.extract = "no longer a call"  # type: ignore[assignment]
        assert "extract" not in prog.named_calls()

    def test_reassigning_call_to_call_replaces(self) -> None:
        class P(Program):
            def __init__(self) -> None:
                self.extract = _make_call()

        prog = P()
        original = prog.extract
        new_call = _make_call()
        prog.extract = new_call
        assert prog.named_calls()["extract"] is new_call
        assert prog.named_calls()["extract"] is not original


# ---------------------------------------------------------------------------
# Same Call under multiple names
# ---------------------------------------------------------------------------


class TestMultipleAliases:
    def test_same_call_assigned_to_two_attributes(self) -> None:
        """The ReAct ``self.call = self._inner_call`` aliasing pattern."""
        shared = _make_call()

        class P(Program):
            def __init__(self) -> None:
                self._inner_call = shared
                self.call = shared

        prog = P()
        names = prog.named_calls()
        # The private name is skipped; the public alias picks it up.
        assert "call" in names
        assert "_inner_call" not in names
        assert prog.named_calls()["call"] is shared


# ---------------------------------------------------------------------------
# ProgramGraph.primary() ordering
# ---------------------------------------------------------------------------


class TestPrimaryOrdering:
    def test_first_declared_call_wins(self) -> None:
        class P(Program):
            def __init__(self) -> None:
                self.zeta = _make_call()
                self.alpha = _make_call()

        prog = P()
        primary = prog.graph.primary()
        # Declaration order ⇒ zeta, not alphabetical alpha.
        assert primary is prog.zeta

    def test_primary_skips_sub_programs(self) -> None:
        class Inner(Program):
            async def forward(self, **kwargs: Any) -> Any:
                return None

        class P(Program):
            def __init__(self) -> None:
                self.judge = Inner()  # Program, not Call — skip
                self.producer = _make_call()  # second declared, but first Call

        prog = P()
        primary = prog.graph.primary()
        assert primary is prog.producer

    def test_primary_raises_when_no_call_registered(self) -> None:
        class Inner(Program):
            async def forward(self, **kwargs: Any) -> Any:
                return None

        class P(Program):
            def __init__(self) -> None:
                self.only_inner = Inner()

        prog = P()
        with pytest.raises(TypeError, match="primary"):
            prog.graph.primary()

    def test_primary_raises_on_empty_registry(self) -> None:
        class P(Program):
            def __init__(self) -> None:
                self._hidden = _make_call()  # private — not registered

        prog = P()
        with pytest.raises(TypeError, match="primary"):
            prog.graph.primary()


# ---------------------------------------------------------------------------
# ProgramGraph view object
# ---------------------------------------------------------------------------


class TestProgramGraphView:
    def test_graph_property_returns_program_graph_instance(self) -> None:
        class P(Program):
            def __init__(self) -> None:
                self.work = _make_call()

        prog = P()
        assert isinstance(prog.graph, ProgramGraph)

    def test_children_returns_snapshot(self) -> None:
        class P(Program):
            def __init__(self) -> None:
                self.first = _make_call()
                self.second = _make_call()

        prog = P()
        snap = prog.graph.children()
        assert list(snap.keys()) == ["first", "second"]
        # Snapshot is a copy — mutating it does not affect the registry.
        snap["bogus"] = _make_call()
        assert "bogus" not in prog.named_calls()

    def test_named_calls_preserves_declaration_order(self) -> None:
        class P(Program):
            def __init__(self) -> None:
                self.gamma = _make_call()
                self.alpha = _make_call()
                self.beta = _make_call()

        prog = P()
        assert list(prog.named_calls().keys()) == ["gamma", "alpha", "beta"]
