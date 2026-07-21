from __future__ import annotations

from pathlib import Path

import pytest

from hpc.environment import (
    HpcAcceptanceContext,
    prepare_acceptance_context,
    require_coordinator_native_threads,
    require_explicit_hpc_selection,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CLEAREX_ROOT = Path("/project/bioinformatics/Danuser_lab/Dean/dean/git/clearex")
LIVE_SPHEROID_ROOTS = (
    Path(
        "/endosome/archive/bioinformatics/Danuser_lab/Dean/dean/"
        "2026-07-18-spheroid-registration"
    ),
    Path(
        "/archive/bioinformatics/Danuser_lab/Dean/dean/2026-07-18-spheroid-registration"
    ),
)


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("cyexm-hpc", "explicit Cy-ExM HPC acceptance")
    group.addoption(
        "--cyexm-dataset",
        action="store",
        default=None,
        metavar="PATH",
        help="approved strict dataset manifest",
    )
    group.addoption(
        "--cyexm-execution",
        action="store",
        default=None,
        metavar="PATH",
        help="approved strict SLURM execution profile",
    )
    group.addoption(
        "--cyexm-evidence-dir",
        action="store",
        default=None,
        metavar="PATH",
        help="existing empty caller-owned acceptance evidence directory",
    )


@pytest.fixture(scope="session", autouse=True)
def _required_hpc_context(
    request: pytest.FixtureRequest,
) -> HpcAcceptanceContext:
    try:
        require_explicit_hpc_selection(request.config.invocation_params.args)
        context = prepare_acceptance_context(
            request.config.getoption("--cyexm-dataset"),
            request.config.getoption("--cyexm-execution"),
            request.config.getoption("--cyexm-evidence-dir"),
            repository_root=REPOSITORY_ROOT,
            protected_roots=(CLEAREX_ROOT, *LIVE_SPHEROID_ROOTS),
        )
        require_coordinator_native_threads(context.profile)
        return context
    except Exception as error:
        raise pytest.UsageError(
            "Cy-ExM HPC acceptance prerequisites failed before scheduler access: "
            f"{type(error).__name__}: {error}"
        ) from error


@pytest.fixture(scope="session")
def hpc_context(
    _required_hpc_context: HpcAcceptanceContext,
) -> HpcAcceptanceContext:
    return _required_hpc_context
