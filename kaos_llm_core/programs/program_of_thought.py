"""ProgramOfThought — code-as-reasoning.

Phase 16.2 step kind. Inspired by DSPy's ProgramOfThought module: the
LLM writes a small Python program that solves the task, the program
is executed in an isolated subprocess sandbox, and a second LLM call
interprets the captured stdout to produce the final structured
answer.

This is the "let the model use code as a calculator" pattern. It is
particularly effective for tasks where the answer requires arithmetic,
list manipulation, or any operation an LLM is unreliable at executing
in its head.

## Security model

**Executing model-generated code is dangerous by default.** This
module ships an opt-in `allow_code_execution=True` flag that the
caller MUST set explicitly. Constructing a ``ProgramOfThought`` with
the default ``allow_code_execution=False`` and then calling it raises
``ProgramOfThoughtError`` before any subprocess is spawned.

The v1 sandbox is a fresh Python interpreter spawned via ``subprocess.run``:

  - Isolated mode (``-I``) — ignores PYTHONPATH, PYTHONSTARTUP, user site
  - No site (``-S``) — does not import the site module
  - No environment variables — only PATH is preserved (so ``python``
    can find shared libs but the model can't read API keys from env)
  - Hard timeout via ``subprocess.run(timeout=...)``
  - POSIX resource limits via a ``preexec_fn``: RLIMIT_AS (memory),
    RLIMIT_CPU (CPU seconds), RLIMIT_FSIZE (max file size).
  - stdin closed; stdout + stderr captured into byte buffers
  - working directory = a fresh ``tempfile.mkdtemp()`` that is
    removed after the call (so the model's code can write to its
    own scratch dir but not pollute the host fs)

**v1 sandbox is best-effort.** It blocks the obvious mistakes
(memory bombs, infinite loops, accidental large file writes,
environment-variable exfiltration) but it does not stop a determined
adversary. Production deployments that take untrusted task input
should swap in a real container sandbox (gVisor, firecracker,
seccomp-bpf) by subclassing ``ProgramOfThought`` and overriding
``_execute_code``. The default implementation is documented enough
that the swap is local.

**ALWAYS audit the generated code before enabling this in production.**
Set ``audit_callback=...`` to receive every generated code block
before it executes; the callback can raise to refuse the run.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# `resource` is POSIX-only — importing it at module top level breaks
# every test that transitively imports kaos_llm_core on Windows
# (ModuleNotFoundError: No module named 'resource'). The sandbox
# preexec is also POSIX-only (no `fork()` on Windows), so we gate the
# whole rlimit machinery behind `os.name`.
if os.name == "posix":
    import resource as _resource
else:  # pragma: no cover - exercised on Windows CI legs only
    _resource = None  # type: ignore[assignment]

from kaos_core.logging import get_logger
from pydantic import create_model

from kaos_llm_core.errors import KaosLLMCoreError
from kaos_llm_core.programs.base import Program
from kaos_llm_core.programs.call import Call
from kaos_llm_core.programs.result_mixin import OutputForwardingMixin
from kaos_llm_core.signatures.fields import InputField, OutputField
from kaos_llm_core.signatures.introspection import (
    get_input_fields,
    get_output_fields,
)
from kaos_llm_core.signatures.signature import Signature
from kaos_llm_core.types import Example

logger = get_logger(__name__)


class ProgramOfThoughtError(KaosLLMCoreError):
    """Raised when the ProgramOfThought sandbox refuses or fails a run."""


@dataclass
class _CodeExecutionResult:
    """Captured output from one sandbox execution."""

    stdout: str
    stderr: str
    return_code: int
    timed_out: bool
    code: str


@dataclass(frozen=True, slots=True)
class ProgramOfThoughtResult(OutputForwardingMixin):
    """Final result of a ProgramOfThought run.

    ``outputs`` is the interpreter's structured answer (the same shape
    as the producer signature). The intermediate code + raw subprocess
    output are accessible via ``execution`` for debugging / logging.
    Matches the convention from :class:`BestOfN`: attribute access
    forwards to ``outputs`` via :class:`OutputForwardingMixin`.
    """

    outputs: Any
    execution: _CodeExecutionResult


# Default sandbox limits — enforced via setrlimit in the subprocess preexec.
_DEFAULT_MEM_BYTES = 256 * 1024 * 1024  # 256 MB
_DEFAULT_CPU_SECONDS = 5
_DEFAULT_FSIZE_BYTES = 4 * 1024 * 1024  # 4 MB
_DEFAULT_WALL_TIMEOUT_S = 8.0


def _apply_rlimits(
    *,
    mem_bytes: int = _DEFAULT_MEM_BYTES,
    cpu_seconds: int = _DEFAULT_CPU_SECONDS,
    fsize_bytes: int = _DEFAULT_FSIZE_BYTES,
) -> None:
    """preexec_fn for the sandboxed subprocess.

    Sets POSIX resource limits BEFORE the Python interpreter starts.
    Hardens against the most obvious "model writes a fork bomb / mem
    bomb / writes the entire dictionary to disk" classes of mistake.

    Darwin caveat: ``RLIMIT_AS`` is not supported on macOS and
    ``setrlimit`` raises ``OSError(EINVAL)`` for it. We deliberately
    skip RLIMIT_AS on Darwin so the other two limits still take
    effect; failing the whole preexec would propagate up as
    ``SubprocessError: Exception occurred in preexec_fn`` and kill
    every subprocess on the macOS test leg.
    """
    assert _resource is not None  # caller gated by os.name == "posix"
    is_darwin = sys.platform == "darwin"
    try:
        if not is_darwin:
            _resource.setrlimit(_resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        _resource.setrlimit(_resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        _resource.setrlimit(_resource.RLIMIT_FSIZE, (fsize_bytes, fsize_bytes))
    except (ValueError, OSError):
        # If we can't set the rlimits, do NOT silently proceed — the
        # subprocess will run unbounded. Raise so the parent process
        # sees the failure.
        raise


class ProgramOfThought(Program):
    """Code-as-reasoning.

    Two-phase pipeline:

      1. **Code generation** — the producer Call is asked to write a
         self-contained Python program whose stdout is the answer.
         The code is generated against a synthetic signature whose
         output is a single ``code`` string field.
      2. **Sandboxed execution** — the generated code runs in an
         isolated Python subprocess with rlimits + timeout.
      3. **Interpretation** — the interpreter Call takes the original
         inputs plus the captured stdout and produces the final
         structured answer.

    Args:
        signature: The original Signature describing the task.
        producer_model: Model for code generation. If None, falls back
            to per-Call default.
        interpreter_model: Model for interpreting the captured output.
            Defaults to ``producer_model``.
        allow_code_execution: **Required to be True**. Constructing
            with the default ``False`` and then calling raises a
            ``ProgramOfThoughtError`` before any subprocess is
            spawned. Forces the caller to make a deliberate choice.
        timeout_s: Wall-clock timeout for the subprocess. Default 8s.
        memory_bytes: RLIMIT_AS cap. Default 256 MB.
        cpu_seconds: RLIMIT_CPU cap. Default 5.
        audit_callback: Optional ``(code: str) -> None`` callback that
            is invoked with every generated code block before
            execution. Raise from the callback to refuse the run.
    """

    def __init__(
        self,
        signature: type[Signature],
        *,
        producer_model: str | None = None,
        interpreter_model: str | None = None,
        allow_code_execution: bool = False,
        timeout_s: float = _DEFAULT_WALL_TIMEOUT_S,
        memory_bytes: int = _DEFAULT_MEM_BYTES,
        cpu_seconds: int = _DEFAULT_CPU_SECONDS,
        audit_callback: Callable[[str], None] | None = None,
        code_writer_examples: list[Example] | None = None,
        interpreter_examples: list[Example] | None = None,
        core_settings: Any = None,
    ) -> None:
        """Construct a ProgramOfThought.

        See the module docstring for the full security model. ``examples``
        kwargs ground the two inner Calls; they are independent because
        the writer and interpreter operate on different synthesised
        Signatures (writer outputs ``code``; interpreter outputs the
        original Signature's output fields). Pass ``None`` to either to
        opt out of grounding for that step.

        ``core_settings`` is forwarded to BOTH inner Calls so
        ``KAOS_LLM_CORE_TRACE_ENABLED`` and per-request
        ``_meta.kaos_config`` overrides reach the writer + interpreter.
        Without it, per-request config silently drops — mirrors the
        Phase 9d wiring in :class:`Judge`.
        """
        super().__init__()
        self.signature = signature
        self.producer_model = producer_model
        self.interpreter_model = interpreter_model or producer_model
        self.allow_code_execution = allow_code_execution
        self.timeout_s = timeout_s
        self.memory_bytes = memory_bytes
        self.cpu_seconds = cpu_seconds
        self.audit_callback = audit_callback
        self._code_writer_examples = code_writer_examples
        self._interpreter_examples = interpreter_examples
        self._core_settings = core_settings

        self.code_writer = self._build_code_writer()
        self.interpreter = self._build_interpreter()

    def _build_code_writer(self) -> Call:
        """Synthetic Signature whose output is a single ``code`` field."""
        input_fields = get_input_fields(self.signature)
        field_defs: dict[str, Any] = {}
        for name in input_fields:
            field_defs[name] = (
                self.signature.model_fields[name].annotation,
                InputField(description=input_fields[name].description or name),
            )
        field_defs["code"] = (
            str,
            OutputField(
                description=(
                    "A self-contained Python program that solves the task. "
                    "Print the answer to stdout. The program must be valid "
                    "Python 3, must not import network libraries, and must "
                    "complete in a few seconds. Use only the standard library."
                ),
            ),
        )
        writer_sig = create_model(
            f"{self.signature.__name__}CodeWriter",
            __base__=Signature,
            __doc__=(
                f"Write a Python program that solves a {self.signature.__name__} "
                "task. The program's stdout will be interpreted as the answer."
            ),
            **field_defs,
        )
        kwargs: dict[str, Any] = {}
        if self.producer_model is not None:
            kwargs["model"] = self.producer_model
        if self._code_writer_examples is not None:
            kwargs["examples"] = self._code_writer_examples
        if self._core_settings is not None:
            kwargs["core_settings"] = self._core_settings
        return Call(writer_sig, **kwargs)

    def _build_interpreter(self) -> Call:
        """Signature: original_inputs + raw_output → original_outputs."""
        input_fields = get_input_fields(self.signature)
        output_fields = get_output_fields(self.signature)
        field_defs: dict[str, Any] = {}
        for name in input_fields:
            field_defs[name] = (
                self.signature.model_fields[name].annotation,
                InputField(description=input_fields[name].description or name),
            )
        field_defs["raw_output"] = (
            str,
            InputField(
                description=(
                    "The raw stdout captured from a Python program that "
                    "solved the task. Parse this output into the structured "
                    "answer fields below."
                ),
            ),
        )
        for name in output_fields:
            field_defs[name] = (
                self.signature.model_fields[name].annotation,
                OutputField(description=output_fields[name].description or name),
            )
        interp_sig = create_model(
            f"{self.signature.__name__}Interpreter",
            __base__=Signature,
            __doc__=(
                f"Interpret raw program output into a structured {self.signature.__name__} answer."
            ),
            **field_defs,
        )
        kwargs: dict[str, Any] = {}
        if self.interpreter_model is not None:
            kwargs["model"] = self.interpreter_model
        if self._interpreter_examples is not None:
            kwargs["examples"] = self._interpreter_examples
        if self._core_settings is not None:
            kwargs["core_settings"] = self._core_settings
        return Call(interp_sig, **kwargs)

    def _execute_code(self, code: str) -> _CodeExecutionResult:
        """Run the generated code in a sandboxed subprocess.

        Subclasses can override this to swap in a stronger sandbox
        (gVisor, firecracker, seccomp-bpf). The default uses an
        isolated Python interpreter with rlimits + tempdir cwd +
        wall-clock timeout.
        """
        if not self.allow_code_execution:
            msg = (
                "ProgramOfThought refuses to execute model-generated code "
                "because allow_code_execution=False (the default). Pass "
                "allow_code_execution=True at construction time to opt in. "
                "Read the security model in "
                "kaos_llm_core/programs/program_of_thought.py before doing so."
            )
            raise ProgramOfThoughtError(msg)

        if self.audit_callback is not None:
            # The callback can raise to refuse the run.
            self.audit_callback(code)

        # Fresh tempdir for the subprocess cwd.
        with tempfile.TemporaryDirectory(prefix="kaos-pot-") as scratch:
            # Minimal env: only PATH so the interpreter can find its libs.
            env: dict[str, str] = {}
            if "PATH" in os.environ:
                env["PATH"] = os.environ["PATH"]

            mem = self.memory_bytes
            cpu = self.cpu_seconds

            # POSIX-only: subprocess.run on Windows raises if
            # preexec_fn is passed. On Windows the sandbox is
            # weaker — only the wall-clock timeout and isolated
            # interpreter flags (-I -S) apply. The Windows path is
            # explicitly NOT a defence-in-depth boundary in
            # production; see the module docstring. We keep the
            # subprocess path running so test coverage on the
            # Windows leg still exercises the code-generation +
            # interpretation flow.
            preexec: Callable[[], None] | None
            if os.name == "posix":
                preexec = lambda: _apply_rlimits(mem_bytes=mem, cpu_seconds=cpu)  # noqa: E731
            else:
                preexec = None

            try:
                completed = subprocess.run(
                    [sys.executable, "-I", "-S", "-c", code],
                    capture_output=True,
                    timeout=self.timeout_s,
                    cwd=scratch,
                    env=env,
                    preexec_fn=preexec,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                stdout = exc.stdout.decode("utf-8", errors="replace") if exc.stdout else ""
                stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
                return _CodeExecutionResult(
                    stdout=stdout,
                    stderr=stderr,
                    return_code=-1,
                    timed_out=True,
                    code=code,
                )

            return _CodeExecutionResult(
                stdout=completed.stdout.decode("utf-8", errors="replace"),
                stderr=completed.stderr.decode("utf-8", errors="replace"),
                return_code=completed.returncode,
                timed_out=False,
                code=code,
            )

    async def forward(self, **inputs: Any) -> Any:
        # 1. Generate code
        writer_invocation = await self.code_writer.invoke(**inputs)
        code = str(getattr(writer_invocation.output, "code", "")).strip()
        if not code:
            msg = (
                "ProgramOfThought: code writer returned an empty 'code' field. "
                "Either the producer model failed to generate code, or the "
                "JSON codec dropped the field. Inspect the trace."
            )
            raise ProgramOfThoughtError(msg)

        # Strip markdown code fences if the model wrapped the code.
        code = _strip_code_fence(code)

        # 2. Execute in the sandbox
        execution = await asyncio.to_thread(self._execute_code, code)

        # 3. Interpret the captured output
        interp_inputs: dict[str, Any] = dict(inputs)
        interp_inputs["raw_output"] = execution.stdout
        interp_invocation = await self.interpreter.invoke(**interp_inputs)

        return ProgramOfThoughtResult(
            outputs=interp_invocation.output,
            execution=execution,
        )


def _strip_code_fence(code: str) -> str:
    """If the model wrapped the code in ```python ... ``` fences, strip them."""
    code = code.strip()
    if code.startswith("```"):
        # Drop the opening fence line
        lines = code.splitlines()
        # Remove first line (```python or just ```)
        lines = lines[1:]
        # Remove trailing fence if present
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)
    return code


# Silence ruff: Path is imported for subclass type hints in docstrings
_ = Path
