import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from _scheduler_firewall import (
    SCHEDULER_COMMANDS,
    SCHEDULER_FIREWALL_ENV,
    create_scheduler_shims,
    install_process_guards,
    scheduler_command,
)

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
EXTERNAL_SCHEDULER_COMMANDS = SCHEDULER_COMMANDS
_external_scheduler_command = scheduler_command

for name in NATIVE_THREAD_ENV_VARS:
    os.environ.setdefault(name, "1")


def _activate_scheduler_firewall(
    patcher: pytest.MonkeyPatch,
    root: Path,
) -> None:
    shim_root = create_scheduler_shims(root / "cyexm-scheduler-firewall-bin")
    patcher.setenv(
        "PATH",
        f"{shim_root}{os.pathsep}{os.environ.get('PATH', '')}",
    )
    patcher.setenv(
        "PYTHONPATH",
        f"{TEST_ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
    )
    patcher.setenv(SCHEDULER_FIREWALL_ENV, "1")
    install_process_guards(patcher.setattr)


@pytest.fixture(autouse=True)
def prevent_portable_scheduler_commands(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> Iterator[None]:
    """Fail closed before a portable test can invoke a SLURM command."""
    if request.node.get_closest_marker("hpc") is not None:
        yield
        return

    patcher = pytest.MonkeyPatch()
    _activate_scheduler_firewall(patcher, tmp_path)
    try:
        yield
    finally:
        patcher.undo()


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
