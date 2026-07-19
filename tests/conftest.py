import os
from pathlib import Path

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

for name in NATIVE_THREAD_ENV_VARS:
    os.environ.setdefault(name, "1")


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
