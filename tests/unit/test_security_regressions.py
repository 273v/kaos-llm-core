"""Regression tests for the kaos-llm-core 0.1.0a1 audit findings.

Each test is named ``test_kllc_<id>_<short>`` and pairs with a finding in
``docs/oss/`` audit notes (KLLC-01 .. KLLC-07). Failure of any of these
indicates a security regression.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from kaos_llm_core.codecs.json_codec import JSONCodec
from kaos_llm_core.errors import CallError
from kaos_llm_core.programs.best_of_n import BestOfN
from kaos_llm_core.programs.call import Call, _resolve_codec_class
from kaos_llm_core.programs.judge import Judge
from kaos_llm_core.programs.multi_chain_comparison import MultiChainComparison
from kaos_llm_core.programs.react import ReAct
from kaos_llm_core.programs.refine import Refine
from kaos_llm_core.signatures import InputField, OutputField, Signature


def _err_text(result: object) -> str:
    """Extract text from a ToolResult.content[0] in a ty-friendly way."""
    content = getattr(result, "content", None)
    if not content:
        return ""
    return getattr(content[0], "text", "") or ""


class _Sig(Signature):
    """Minimal signature for instantiating Calls in these tests."""

    text: str = InputField(description="input text")
    answer: str = OutputField(description="model answer")


# ---------------------------------------------------------------------------
# KLLC-01: arbitrary importlib.import_module via saved-state JSON
# ---------------------------------------------------------------------------


def test_kllc_01_resolve_codec_class_rejects_non_allowlisted_module() -> None:
    """`_resolve_codec_class` must reject any module not under
    ``kaos_llm_core.codecs.``.

    The fix: allowlist check happens BEFORE ``importlib.import_module`` so
    that module-level code from arbitrary packages cannot run."""
    with patch("importlib.import_module") as mock_import:
        result = _resolve_codec_class("antigravity.AntigravityCodec")
        assert result is None
        mock_import.assert_not_called(), ("non-allowlisted module name must not trigger import")


def test_kllc_01_resolve_codec_class_rejects_stdlib_module() -> None:
    """A stdlib module name with importable side-effect-free contents
    must still be refused."""
    with patch("importlib.import_module") as mock_import:
        result = _resolve_codec_class("os.path.join")
        assert result is None
        mock_import.assert_not_called()


def test_kllc_01_resolve_codec_class_accepts_allowlisted_module() -> None:
    """Real first-party codecs still resolve correctly."""
    cls = _resolve_codec_class("kaos_llm_core.codecs.json_codec.JSONCodec")
    assert cls is JSONCodec


def test_kllc_01_set_learnable_state_does_not_import_arbitrary_module() -> None:
    """Round-trip via the user-facing ``set_learnable_state`` API.

    Even if a malicious envelope sets ``codec`` to a non-allowlisted
    module, no import happens."""
    call = Call(_Sig, model="anthropic:claude-haiku-4-5")
    with patch("importlib.import_module") as mock_import:
        call.set_learnable_state({"codec": "subprocess.Popen"})
        # The non-allowlisted name was rejected without importing anything.
        for module_arg in (c.args[0] for c in mock_import.call_args_list):
            assert module_arg.startswith("kaos_llm_core.codecs."), (
                f"set_learnable_state imported non-allowlisted module {module_arg!r}"
            )


# ---------------------------------------------------------------------------
# KLLC-02: absolute paths in batch helpers bypass VFS containment
# ---------------------------------------------------------------------------


def test_kllc_02_resolve_vfs_to_disk_rejects_absolute_path() -> None:
    """``_resolve_vfs_to_disk`` must reject absolute paths so that an MCP
    caller cannot escape the VFS root by passing ``/etc/cron.d`` etc."""
    from kaos_core import ToolResult

    from kaos_llm_core.integrations.mcp._batch_helpers import _resolve_vfs_to_disk

    result = _resolve_vfs_to_disk("/etc/cron.d/escape-test", context=None)
    assert isinstance(result, ToolResult)
    assert result.isError
    assert "absolute" in _err_text(result).lower()


def test_kllc_02_resolve_output_dir_rejects_absolute_path(tmp_path: Path) -> None:
    """``resolve_output_dir`` must reject absolute paths and must not
    create the directory side effect."""
    from kaos_core import ToolResult

    from kaos_llm_core.integrations.mcp._batch_helpers import resolve_output_dir

    target = tmp_path / "kllc02-escape"
    result = resolve_output_dir(str(target), context=None)
    assert isinstance(result, ToolResult)
    assert result.isError
    assert "absolute" in _err_text(result).lower()
    assert not target.exists(), "absolute-path output_dir must not be created"


# ---------------------------------------------------------------------------
# KLLC-03: save_load reads arbitrary disk paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kllc_03_save_load_load_rejects_absolute_path() -> None:
    """`KaosLLMCoreSaveLoadTool` must reject absolute paths regardless
    of mode (load and round-trip both)."""
    from kaos_llm_core.integrations.mcp.save_load import KaosLLMCoreSaveLoadTool

    tool = KaosLLMCoreSaveLoadTool()
    result = await tool._run({"mode": "load", "path": "/etc/hostname"}, context=None)
    assert result.isError
    assert "absolute" in _err_text(result).lower()


@pytest.mark.asyncio
async def test_kllc_03_save_load_round_trip_rejects_absolute_path() -> None:
    from kaos_llm_core.integrations.mcp.save_load import KaosLLMCoreSaveLoadTool

    tool = KaosLLMCoreSaveLoadTool()
    result = await tool._run({"mode": "round-trip", "path": "/etc/hostname"}, context=None)
    assert result.isError
    assert "absolute" in _err_text(result).lower()


# ---------------------------------------------------------------------------
# KLLC-05: _build_eval_dataset accepts non-dict examples
# ---------------------------------------------------------------------------


def test_kllc_05_build_eval_dataset_rejects_non_dict_example() -> None:
    """A list, None, or string in the examples list must surface an
    agent-friendly ``ToolResult`` error, not an uncaught ``TypeError``."""
    from kaos_core import ToolResult

    from kaos_llm_core.integrations.mcp._common import _build_eval_dataset

    result = _build_eval_dataset([["input", "expected"]])
    assert isinstance(result, ToolResult)
    assert result.isError
    text = _err_text(result).lower()
    assert "json object" in text or "must be" in text


def test_kllc_05_build_eval_dataset_rejects_none_example() -> None:
    from kaos_core import ToolResult

    from kaos_llm_core.integrations.mcp._common import _build_eval_dataset

    result = _build_eval_dataset([None])
    assert isinstance(result, ToolResult)
    assert result.isError


def test_kllc_05_build_eval_dataset_accepts_well_formed_examples() -> None:
    """The validation must not break the happy path."""
    from kaos_llm_core.integrations.mcp._common import _build_eval_dataset

    _, dataset = _build_eval_dataset([{"input": "hello", "expected_output": "world"}])
    assert len(dataset) == 1


# ---------------------------------------------------------------------------
# KLLC-07: iteration counts have no upper bound (Python API)
# ---------------------------------------------------------------------------


def test_kllc_07_react_max_iterations_above_cap_rejected() -> None:
    """ReAct construction must reject ``max_iterations`` exceeding the
    class-level ``MAX_ITERATIONS`` cap to prevent runaway API spend."""
    with pytest.raises(CallError, match=r"hard cap of \d+"):
        ReAct(
            _Sig,
            tools=[],
            max_iterations=ReAct.MAX_ITERATIONS + 1,
        )


def test_kllc_07_refine_max_iterations_above_cap_rejected() -> None:
    """Refine construction must reject ``max_iterations`` exceeding the cap."""
    producer = Call(_Sig, model="anthropic:claude-haiku-4-5")
    judge = Judge(_Sig, judge_model="anthropic:claude-haiku-4-5")
    with pytest.raises(CallError, match=r"hard cap of \d+"):
        Refine(
            producer,
            judge,
            max_iterations=Refine.MAX_ITERATIONS + 1,
        )


def test_kllc_07_best_of_n_n_above_cap_rejected() -> None:
    """BestOfN construction must reject ``n`` exceeding the cap."""
    producer = Call(_Sig, model="anthropic:claude-haiku-4-5")

    def selector(output: object, inputs: dict[str, object]) -> float:
        return 1.0

    with pytest.raises(CallError, match=r"hard cap of \d+"):
        BestOfN(
            producer,
            n=BestOfN.MAX_N + 1,
            selector=selector,
        )


def test_kllc_07_multi_chain_comparison_n_above_cap_rejected() -> None:
    """MultiChainComparison construction must reject ``n`` exceeding the cap."""
    with pytest.raises(ValueError, match=r"hard cap of \d+"):
        MultiChainComparison(_Sig, n=MultiChainComparison.MAX_N + 1)


def test_kllc_07_react_max_iterations_at_cap_accepted() -> None:
    """Boundary case — ``MAX_ITERATIONS`` itself must still be accepted."""
    # No exception expected.
    ReAct(_Sig, tools=[], max_iterations=ReAct.MAX_ITERATIONS)
