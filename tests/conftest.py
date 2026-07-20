import os
import re
import subprocess
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

TEST_ROOT = Path(__file__).resolve().parent
TEST_CATEGORY_BY_DIRECTORY = {
    "unit": "unit",
    "integration": "integration",
    "hpc": "hpc",
}

NATIVE_THREAD_ENV_VARS = (
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
EXTERNAL_SCHEDULER_COMMANDS = frozenset(
    {"sacct", "sbatch", "scancel", "scontrol", "sinfo", "squeue", "srun"}
)
_SHELL_COMMANDS = frozenset({"bash", "dash", "sh", "zsh"})
_EXTERNAL_SCHEDULER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_-])(?:[^\s'\"]*/)?("
    + "|".join(sorted(EXTERNAL_SCHEDULER_COMMANDS))
    + r")(?![A-Za-z0-9_-])"
)

for name in NATIVE_THREAD_ENV_VARS:
    os.environ.setdefault(name, "1")


def _command_arguments(arguments: object) -> tuple[str, ...]:
    if isinstance(arguments, (str, bytes, os.PathLike)):
        return (os.fsdecode(arguments),)
    if isinstance(arguments, Sequence):
        return tuple(
            os.fsdecode(argument)
            if isinstance(argument, (str, bytes, os.PathLike))
            else repr(argument)
            for argument in arguments
        )
    return (repr(arguments),)


def _external_scheduler_command(
    arguments: object,
    *,
    shell: bool = False,
) -> str | None:
    tokens = _command_arguments(arguments)
    if not tokens:
        return None
    executable = Path(tokens[0]).name
    if executable in EXTERNAL_SCHEDULER_COMMANDS:
        return executable
    if shell:
        match = _EXTERNAL_SCHEDULER_PATTERN.search(" ".join(tokens))
        return None if match is None else match.group(1)
    if executable == "env":
        for token in tokens[1:]:
            if token == "--" or token.startswith("-") or "=" in token:
                continue
            candidate = Path(token).name
            return candidate if candidate in EXTERNAL_SCHEDULER_COMMANDS else None
    if executable in _SHELL_COMMANDS and "-c" in tokens:
        command_index = tokens.index("-c") + 1
        if command_index < len(tokens):
            match = _EXTERNAL_SCHEDULER_PATTERN.search(tokens[command_index])
            return None if match is None else match.group(1)
    return None


@pytest.fixture(autouse=True)
def prevent_portable_scheduler_commands(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Fail closed before a portable test can invoke a SLURM command."""
    if request.node.get_closest_marker("hpc") is not None:
        yield
        return

    popen = subprocess.Popen

    def guarded_popen(*args: Any, **kwargs: Any) -> subprocess.Popen[Any]:
        arguments = kwargs.get("args", args[0] if args else ())
        command = _external_scheduler_command(
            kwargs.get("executable", arguments),
            shell=kwargs.get("shell") is True,
        )
        if command is None and "executable" in kwargs:
            command = _external_scheduler_command(
                arguments,
                shell=kwargs.get("shell") is True,
            )
        if command is not None:
            raise RuntimeError(
                f"portable tests must not invoke external scheduler command {command!r}"
            )
        return popen(*args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", guarded_popen)
    yield


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    violations: dict[str, tuple[str, ...]] = {}
    for item in items:
        relative_path = Path(item.path).resolve().relative_to(TEST_ROOT)
        directory = relative_path.parts[0]
        expected = TEST_CATEGORY_BY_DIRECTORY.get(directory)
        selected = tuple(
            marker
            for marker in TEST_CATEGORY_BY_DIRECTORY.values()
            if item.get_closest_marker(marker) is not None
        )
        if expected is None or selected != (expected,):
            violations.setdefault(str(relative_path), selected)

    if violations:
        details = ", ".join(
            f"{path}: {markers or ('<none>',)}"
            for path, markers in sorted(violations.items())
        )
        raise pytest.UsageError(
            "Each collected test must have exactly one category marker matching its "
            f"first directory: {details}"
        )
