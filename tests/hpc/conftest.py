from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from cyexm.config.models import ExecutionProfile
from hpc.environment import (
    HpcAcceptanceContext,
    prepare_acceptance_context,
    require_coordinator_native_threads,
    require_explicit_hpc_selection,
    require_hpc_opt_in,
    validate_additional_acceptance_profile,
)
from hpc.fine_workflow_acceptance import fine_acceptance_profiles
from hpc.fixed_affine_chunk_characterization import (
    FixedAffineCharacterizationSpec,
    ValidatedFixedAffineSource,
    load_characterization_spec,
    validate_fixed_affine_source,
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


@dataclass(frozen=True, slots=True)
class FineHpcProfiles:
    """Two exact operator-approved profiles for fine acceptance."""

    coarse_path: Path
    fine_path: Path
    coarse: ExecutionProfile
    fine: ExecutionProfile


@dataclass(frozen=True, slots=True)
class FixedAffineCharacterizationContext:
    """Validated operator inputs for the fixed-affine characterization."""

    path: Path
    spec: FixedAffineCharacterizationSpec
    source: ValidatedFixedAffineSource


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("cyexm-hpc", "explicit Cy-ExM HPC acceptance")
    group.addoption(
        "--run-hpc",
        action="store_true",
        default=False,
        help="explicitly authorize selected real-SLURM acceptance tests",
    )
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
    group.addoption(
        "--cyexm-coarse-execution",
        action="store",
        default=None,
        metavar="PATH",
        help="approved strict coarse SLURM execution profile for fine acceptance",
    )
    group.addoption(
        "--cyexm-fine-execution",
        action="store",
        default=None,
        metavar="PATH",
        help="approved strict fine-grained SLURM execution profile",
    )
    group.addoption(
        "--cyexm-characterization",
        action="store",
        default=None,
        metavar="PATH",
        help="approved fixed-affine chunk characterization configuration",
    )


@pytest.fixture(scope="session", autouse=True)
def _required_hpc_context(
    request: pytest.FixtureRequest,
) -> HpcAcceptanceContext:
    try:
        require_hpc_opt_in(request.config.getoption("--run-hpc"))
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


@pytest.fixture(scope="session")
def fine_hpc_profiles(
    request: pytest.FixtureRequest,
    hpc_context: HpcAcceptanceContext,
) -> FineHpcProfiles:
    """Load both fine-acceptance profiles only when the fine test requests them."""
    raw_coarse = request.config.getoption("--cyexm-coarse-execution")
    raw_fine = request.config.getoption("--cyexm-fine-execution")
    if not isinstance(raw_coarse, str) or not raw_coarse.strip():
        raise pytest.UsageError(
            "fine HPC acceptance requires --cyexm-coarse-execution PATH"
        )
    if not isinstance(raw_fine, str) or not raw_fine.strip():
        raise pytest.UsageError(
            "fine HPC acceptance requires --cyexm-fine-execution PATH"
        )
    try:
        coarse_path = Path(raw_coarse).resolve(strict=True)
        fine_path = Path(raw_fine).resolve(strict=True)
    except OSError as error:
        raise pytest.UsageError(
            "fine HPC acceptance profile paths must be existing files"
        ) from error
    if (
        not coarse_path.is_file()
        or not fine_path.is_file()
        or coarse_path != hpc_context.execution_path
    ):
        raise pytest.UsageError(
            "--cyexm-execution and --cyexm-coarse-execution must identify the "
            "same approved coarse profile; both profile paths must be files"
        )
    try:
        validated_fine_path, validated_fine = validate_additional_acceptance_profile(
            hpc_context,
            fine_path,
            repository_root=REPOSITORY_ROOT,
            protected_roots=(CLEAREX_ROOT, *LIVE_SPHEROID_ROOTS),
        )
        if validated_fine_path != fine_path:
            raise ValueError("fine profile identity changed during validation")
        coarse, fine = fine_acceptance_profiles(
            hpc_context.profile,
            validated_fine,
        )
        if fine.slurm is None or fine.slurm.task_retries < 1:
            raise ValueError(
                "fine acceptance profile requires at least one task retry "
                "to prove worker replacement liveness"
            )
        coarse_dataset = coarse.paths.dataset_root.resolve(strict=True)
        fine_dataset = fine.paths.dataset_root.resolve(strict=True)
        coarse_output = coarse.paths.output_root.resolve(strict=True)
        fine_output = fine.paths.output_root.resolve(strict=True)
        archive = Path("/archive").resolve(strict=True)
        if (
            fine.source_path.resolve(strict=True) != fine_path
            or fine_dataset != coarse_dataset
            or not fine_output.is_relative_to(archive)
            or fine_output.is_relative_to(coarse_output)
            or coarse_output.is_relative_to(fine_output)
            or fine.paths.output_root.is_symlink()
            or not fine_output.is_dir()
        ):
            raise ValueError(
                "fine profile must preserve the approved dataset root and use a "
                "distinct existing real output root below /archive"
            )
        require_coordinator_native_threads(fine)
    except Exception as error:
        raise pytest.UsageError(
            "fine HPC acceptance profiles failed strict validation: "
            f"{type(error).__name__}: {error}"
        ) from error
    return FineHpcProfiles(
        coarse_path=coarse_path,
        fine_path=fine_path,
        coarse=coarse,
        fine=fine,
    )


@pytest.fixture(scope="session")
def fixed_affine_characterization_context(
    request: pytest.FixtureRequest,
    hpc_context: HpcAcceptanceContext,
) -> FixedAffineCharacterizationContext:
    """Validate the characterization source and fixed four-node profile."""
    raw_path = request.config.getoption("--cyexm-characterization")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise pytest.UsageError(
            "fixed-affine characterization requires --cyexm-characterization PATH"
        )
    try:
        path = Path(raw_path).resolve(strict=True)
        if not path.is_file():
            raise ValueError("characterization path must be a file")
        spec = load_characterization_spec(path)
        source = validate_fixed_affine_source(spec.source_workflow_root)
        profile = hpc_context.profile
        slurm = profile.slurm
        if (
            profile.executor != "slurm"
            or profile.parallelism.mode != "fine_grained"
            or slurm is None
            or slurm.workers != 4
            or slurm.cores_per_worker != 30
            or slurm.processes_per_worker != 1
            or slurm.memory != "500GB"
            or "--exclusive" not in slurm.job_extra_directives
            or profile.native_threads.itk != 30
            or profile.native_threads.blas != 30
        ):
            raise ValueError(
                "characterization requires fine SLURM with four exclusive "
                "500GB workers, one process, 30 cores, and 30/30 threads"
            )
        output = profile.paths.output_root.resolve(strict=True)
        source_root = source.workflow_root.resolve(strict=True)
        if (
            output == source_root
            or output.is_relative_to(source_root)
            or source_root.is_relative_to(output)
        ):
            raise ValueError(
                "characterization output and source workflow roots must be disjoint"
            )
    except Exception as error:
        raise pytest.UsageError(
            "fixed-affine characterization prerequisites failed before scheduler "
            f"access: {type(error).__name__}: {error}"
        ) from error
    return FixedAffineCharacterizationContext(
        path=path,
        spec=spec,
        source=source,
    )
