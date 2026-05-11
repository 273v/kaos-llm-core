"""Cross-platform regression tests for ProgramOfThought.

The sandbox itself is POSIX-only — `preexec_fn` doesn't exist on
Windows and `RLIMIT_AS` isn't supported on Darwin. The module
nevertheless has to:

1. Import cleanly on Windows (previously `import resource` at module
   top level made every test that transitively imported
   `kaos_llm_core` fail to collect with `ModuleNotFoundError: No
   module named 'resource'`).
2. `_apply_rlimits` must skip `RLIMIT_AS` on Darwin (otherwise the
   preexec raises `OSError(EINVAL)` and every macOS sandbox test
   fails with `SubprocessError: Exception occurred in preexec_fn`).

These tests run on every platform — they don't carry the
POSIX-only marker that the rest of the sandbox suite uses.
"""

from __future__ import annotations

import sys
from unittest.mock import patch


def test_module_imports_on_windows() -> None:
    """Importing kaos_llm_core must not require POSIX `resource`."""
    # If module init pulls in `resource` unconditionally, this test
    # never gets to run on Windows because the file fails to collect.
    # Reaching the function body is itself proof on Windows; on POSIX
    # it's a trivial assertion that the module is reachable.
    import kaos_llm_core.programs.program_of_thought as pot

    assert hasattr(pot, "ProgramOfThought")
    assert hasattr(pot, "_apply_rlimits")

    if sys.platform == "win32":
        # On Windows we expect the lazy `_resource` symbol to be None.
        assert pot._resource is None
    else:
        # On POSIX, `_resource` should be the real `resource` stdlib
        # module so the preexec can set rlimits.
        assert pot._resource is not None
        assert hasattr(pot._resource, "setrlimit")


def test_apply_rlimits_skips_rlimit_as_on_darwin() -> None:
    """Darwin: `setrlimit(RLIMIT_AS, ...)` would raise; must be skipped.

    Regression for the macOS-arm64 CI leg failing with
    ``subprocess.SubprocessError: Exception occurred in preexec_fn``.
    The macOS kernel rejects RLIMIT_AS; the implementation must skip
    it on Darwin and still apply RLIMIT_CPU / RLIMIT_FSIZE.
    """
    if sys.platform == "win32":
        # On Windows `_resource` is None and `_apply_rlimits` is
        # never invoked (preexec is None). Nothing to assert.
        return

    import kaos_llm_core.programs.program_of_thought as pot

    calls: list[int] = []

    class _FakeResource:
        RLIMIT_AS = 1
        RLIMIT_CPU = 2
        RLIMIT_FSIZE = 3

        @staticmethod
        def setrlimit(which: int, _limits: tuple[int, int]) -> None:
            calls.append(which)

    with (
        patch.object(pot, "_resource", _FakeResource),
        patch.object(pot.sys, "platform", "darwin"),
    ):
        pot._apply_rlimits(mem_bytes=1024, cpu_seconds=1, fsize_bytes=1024)

    # RLIMIT_AS (1) must NOT be in calls — Darwin doesn't support it.
    # RLIMIT_CPU (2) and RLIMIT_FSIZE (3) must both fire.
    assert _FakeResource.RLIMIT_AS not in calls, (
        "RLIMIT_AS must be skipped on Darwin; setrlimit(RLIMIT_AS, ...) "
        "raises OSError(EINVAL) on macOS and kills the preexec."
    )
    assert _FakeResource.RLIMIT_CPU in calls
    assert _FakeResource.RLIMIT_FSIZE in calls


def test_apply_rlimits_applies_all_three_on_linux() -> None:
    """Linux: all three POSIX rlimits must be set."""
    if sys.platform == "win32":
        return

    import kaos_llm_core.programs.program_of_thought as pot

    calls: list[int] = []

    class _FakeResource:
        RLIMIT_AS = 1
        RLIMIT_CPU = 2
        RLIMIT_FSIZE = 3

        @staticmethod
        def setrlimit(which: int, _limits: tuple[int, int]) -> None:
            calls.append(which)

    with (
        patch.object(pot, "_resource", _FakeResource),
        patch.object(pot.sys, "platform", "linux"),
    ):
        pot._apply_rlimits(mem_bytes=1024, cpu_seconds=1, fsize_bytes=1024)

    assert calls == [
        _FakeResource.RLIMIT_AS,
        _FakeResource.RLIMIT_CPU,
        _FakeResource.RLIMIT_FSIZE,
    ]
