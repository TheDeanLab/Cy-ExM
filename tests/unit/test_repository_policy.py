from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
THREAD_ENVIRONMENT = (
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
PHASE_TWO_EXAMPLE_PATHS = (
    "examples/spheroid-dataset.yaml",
    "examples/execution-sequential.yaml",
    "examples/execution-local-dask.yaml",
    "examples/execution-slurm-512gb.yaml",
)
AGENT_GUIDANCE_PATHS = frozenset(
    {
        "AGENTS.md",
        ".github/AGENTS.md",
        ".github/workflows/AGENTS.md",
        "docs/AGENTS.md",
        "docs/superpowers/AGENTS.md",
        "docs/superpowers/plans/AGENTS.md",
        "docs/superpowers/specs/AGENTS.md",
        "examples/AGENTS.md",
        "examples/notebooks/AGENTS.md",
        "examples/scripts/AGENTS.md",
        "src/AGENTS.md",
        "src/cyexm/AGENTS.md",
        "src/cyexm/cli/AGENTS.md",
        "src/cyexm/config/AGENTS.md",
        "src/cyexm/execution/AGENTS.md",
        "src/cyexm/io/AGENTS.md",
        "src/cyexm/registration/AGENTS.md",
        "src/cyexm/runtime/AGENTS.md",
        "src/cyexm/workflow/AGENTS.md",
        "tests/AGENTS.md",
        "tests/integration/AGENTS.md",
        "tests/unit/AGENTS.md",
    }
)
PHASE_TWO_GUIDANCE_REQUIRED_FRAGMENTS = {
    "src/cyexm/cli/AGENTS.md": (
        "phase 2a",
        "parsing, logging setup, lifecycle, formatting, and exit-code mapping",
        "public workflow api",
        "diagnostics go to stderr",
        "structured json and human results to stdout",
        "no scientific algorithms",
        "scheduler construction",
        "filesystem or network activity",
        "environment mutation at import time",
        "0 success",
        "2 config/plan",
        "3 startup/preflight",
        "4 round failure",
        "130 interruption",
    ),
    "src/cyexm/config/AGENTS.md": (
        "phase 2a",
        "strict safe yaml decoding",
        "frozen scientific-manifest and execution-profile models",
        "two documents separate",
        "unknown and duplicate keys",
        "non-scalar yaml keys",
        "aliases",
        "custom tags",
        "additional documents",
        "unsafe paths",
        "scientific and execution roots are disjoint, neither containing the other",
        "must not import ants, numpy, scipy, dask, distributed, or dask-jobqueue",
        "must not mutate the filesystem or environment",
        "initialize a scheduler",
    ),
    "src/cyexm/execution/AGENTS.md": (
        "phase 2d implements sequential, bounded local-dask, and slurm execution",
        "executor protocol",
        "sequential, dask, and slurm adapter lifecycle",
        "never registration algorithms or artifact-validity decisions",
        "scheduler imports occur only inside the selected operation",
        "process-isolated workers",
        "min(pending rounds, configured workers)",
        "one coarse task per round",
        "each worker advertises one `round_slot`",
        "every submitted round task requests and consumes one",
        "default retries are zero",
        "no nested chunk scheduling or silent fallback",
        "all connected workers pass preflight before any registration task is "
        "submitted",
        "post-preflight round failure does not cancel independent round tasks",
        "cancel unfinished futures",
        "client before its owned cluster",
        "success, failure, and cancellation",
        "operational evidence",
        "never authorizes artifact reuse",
        "real slurm submission and hpc acceptance remain unaccepted until phase 2e",
    ),
    "src/cyexm/workflow/AGENTS.md": (
        "phase 2b binds sequential execution to the real round worker",
        "immutable plans, tasks, and results",
        "deterministic path and header validation",
        "output layout, status, fingerprinting, and public orchestration",
        "planning creates no files, directories, clients, clusters, or workers",
        "workflow must not mutate a plan or create a scheduler",
        "phase 1 completion records remain the only restart authority",
        "scientific fingerprints exclude executor, resource, and logging settings",
        "force never bypasses a scientific-plan fingerprint mismatch",
        "no module-scope scientific or scheduler imports",
        "only real registration execution boundary in workflow",
        "configures the task's native threads before its lazy "
        "registration/scientific imports",
        "must not create a dask client, submit child tasks, schedule other rounds, "
        "or start multiprocessing",
    ),
}


def _read_repository_file(path: str) -> str:
    return (REPOSITORY_ROOT / path).read_text(encoding="utf-8")


def _normalized_repository_file(path: str) -> str:
    return " ".join(_read_repository_file(path).split()).lower()


def _load_yaml(path: str, *, github_actions: bool = False) -> object:
    loader = yaml.BaseLoader if github_actions else yaml.SafeLoader
    return yaml.load(_read_repository_file(path), Loader=loader)


def test_pre_commit_runs_the_locked_portable_gates_through_uv() -> None:
    assert _load_yaml(".pre-commit-config.yaml") == {
        "repos": [
            {
                "repo": "local",
                "hooks": [
                    {
                        "id": "ruff-check",
                        "name": "Ruff check",
                        "entry": "uv run ruff check",
                        "language": "system",
                        "types_or": ["python", "pyi"],
                    },
                    {
                        "id": "ruff-format",
                        "name": "Ruff format check",
                        "entry": "uv run ruff format --check",
                        "language": "system",
                        "types_or": ["python", "pyi"],
                    },
                    {
                        "id": "basedpyright",
                        "name": "BasedPyright",
                        "entry": "uv run basedpyright",
                        "language": "system",
                        "pass_filenames": False,
                    },
                    {
                        "id": "pytest-unit",
                        "name": "Unit tests",
                        "entry": 'uv run pytest -m "unit and not hpc" tests/unit -q',
                        "language": "system",
                        "pass_filenames": False,
                    },
                ],
            }
        ]
    }


def test_ci_runs_portable_phase_two_gates_with_hpc_dependencies() -> None:
    assert _load_yaml(".github/workflows/ci.yml", github_actions=True) == {
        "name": "CI",
        "on": {"push": "", "pull_request": ""},
        "permissions": {"contents": "read"},
        "jobs": {
            "portable": {
                "name": "Portable Phase 2 repository gates",
                "runs-on": "ubuntu-latest",
                "timeout-minutes": "30",
                "env": dict.fromkeys(THREAD_ENVIRONMENT, "1"),
                "steps": [
                    {
                        "name": "Check out source",
                        "uses": "actions/checkout@v7",
                    },
                    {
                        "name": "Install uv and Python",
                        "uses": "astral-sh/setup-uv@v8.3.2",
                        "with": {
                            "version": "0.7.13",
                            "python-version": "3.12",
                            "enable-cache": "true",
                        },
                    },
                    {
                        "name": "Sync locked environment",
                        "run": (
                            "uv sync --locked --group dev --extra examples --extra hpc"
                        ),
                    },
                    {"name": "Ruff check", "run": "uv run ruff check ."},
                    {
                        "name": "Ruff format check",
                        "run": "uv run ruff format --check .",
                    },
                    {"name": "BasedPyright", "run": "uv run basedpyright"},
                    {
                        "name": "Unit tests",
                        "run": ('uv run pytest -m "unit and not hpc" tests/unit -v'),
                    },
                    {
                        "name": "Bounded integration tests",
                        "run": (
                            'uv run pytest -m "integration and not hpc" '
                            "tests/integration -v"
                        ),
                    },
                ],
            }
        },
    }

    workflow = _read_repository_file(".github/workflows/ci.yml").lower()
    for forbidden in (
        "clearex",
        "/archive",
        "/project",
        "-m hpc",
        '-m "hpc"',
        "--cyexm-dataset",
        "--cyexm-execution",
        "--cyexm-evidence-dir",
        "sbatch",
        "squeue",
        "scontrol",
        "scancel",
        "sacct",
        "sinfo",
        "srun",
    ):
        assert forbidden not in workflow

    configuration = tomllib.loads(_read_repository_file("pyproject.toml"))
    distributed = configuration["project"]["optional-dependencies"]["distributed"]
    assert "dask==2025.1.0" in distributed
    assert "distributed==2025.1.0" in distributed


def test_root_guidance_tracks_current_phase_two_runtime() -> None:
    root = _normalized_repository_file("AGENTS.md")
    for required in (
        "phase 2 is the current workflow runtime",
        "phase 1 api",
        "phase 1 artifact store remains the only restart authority",
        "approved phase 2 runtime design and implementation plans",
        "strict scientific dataset manifests",
        "separate strict execution profiles",
        "deterministic planning",
        "`cyexm workflow validate`",
        "`cyexm workflow plan`",
        "`cyexm workflow run`",
        "`cyexm workflow status`",
        "sequential, bounded process-based local dask, and slurm executors",
        "real scheduler-neutral round runner",
        "restart-safe transformed-channel artifacts",
        "one coarse task per moving round",
        "one `round_slot` per worker",
        "one process per slurm job",
        "one node per round",
        "all-worker preflight",
        "portable ci installs the hpc python extra",
        "never contacts an external scheduler",
        "explicitly selected `hpc` tests",
        "serial chunk workflow remains",
        "nested chunk multiprocessing",
        "dask chunk graph",
        "uv lock --check",
        "uv sync --locked --group dev --extra examples --extra hpc",
    ):
        assert required in root

    root_lines = set(_read_repository_file("AGENTS.md").splitlines())
    assert "uv lock" not in root_lines
    assert "uv sync --group dev --extra examples" not in root_lines


def test_phase_two_optional_dependencies_are_cumulative_and_pinned() -> None:
    configuration = tomllib.loads(_read_repository_file("pyproject.toml"))
    extras = configuration["project"]["optional-dependencies"]
    assert extras["workflow"] == ["PyYAML>=6,<7"]
    assert extras["distributed"] == [
        "PyYAML>=6,<7",
        "dask==2025.1.0",
        "distributed==2025.1.0",
    ]
    assert extras["hpc"] == [
        "PyYAML>=6,<7",
        "dask==2025.1.0",
        "distributed==2025.1.0",
        "dask-jobqueue==0.9.0",
    ]
    assert configuration["project"]["scripts"] == {"cyexm": "cyexm.cli:main"}

    direct = "\n".join(configuration["project"]["dependencies"]).lower()
    for forbidden in ("pyyaml", "dask", "distributed", "dask-jobqueue"):
        assert forbidden not in direct


def test_documentation_and_examples_track_the_current_phase_two_boundary() -> None:
    documentation = _normalized_repository_file("docs/AGENTS.md")
    for required in (
        "merged phase 1 behavior",
        "phase 2d portable runtime available on this branch",
        "strict yaml validation",
        "deterministic planning",
        "real sequential round execution",
        "bounded process-based local dask execution",
        "restart-safe transformed channels",
        "slurmcluster construction",
        "all-worker preflight",
        "real slurm submission and hpc acceptance",
        "phase 2e",
    ):
        assert required in documentation

    examples = _normalized_repository_file("examples/AGENTS.md")
    for required in (
        "phase 2 is the current workflow runtime",
        "strict yaml validation",
        "deterministic planning",
        "real sequential round execution",
        "bounded process-based local dask execution",
        "restart-safe transformed channels",
        "workflow cli",
        "examples never lead production behavior",
        "scientific manifest",
        "execution profile",
        "slurmcluster adapter",
        "explicitly selected hpc acceptance",
    ):
        assert required in examples
    for path in PHASE_TWO_EXAMPLE_PATHS:
        assert path in examples


def test_all_agent_guidance_is_free_of_stale_phase_two_c_slurm_claims() -> None:
    stale_claims = (
        "phase 2c is the current",
        "phase 2c runtime available on this branch",
        "phase 2c implements sequential and bounded local-dask execution",
        "slurm execution remains proposed for phase 2d",
        "slurm execution and hpc acceptance remain deferred",
        "slurm execution and hpc acceptance remain later milestones",
    )
    for path in AGENT_GUIDANCE_PATHS:
        guidance = _normalized_repository_file(path)
        for claim in stale_claims:
            assert claim not in guidance, (path, claim)


@pytest.mark.parametrize(
    "command",
    ("sbatch", "squeue", "scontrol", "scancel", "sacct", "sinfo", "srun"),
)
def test_portable_test_scheduler_firewall_rejects_slurm_commands(
    command: str,
) -> None:
    with pytest.raises(
        RuntimeError,
        match=rf"portable tests must not invoke external scheduler command {command!r}",
    ):
        subprocess.run([command], check=False)


@pytest.mark.parametrize(
    ("arguments", "command"),
    (
        (("bash", "-c", "squeue"), "squeue"),
        (("env", "CYEXM_TEST=1", "scontrol"), "scontrol"),
    ),
)
def test_portable_test_scheduler_firewall_rejects_wrapped_slurm_commands(
    arguments: tuple[str, ...],
    command: str,
) -> None:
    with pytest.raises(
        RuntimeError,
        match=rf"portable tests must not invoke external scheduler command {command!r}",
    ):
        subprocess.run(arguments, check=False)


def test_portable_test_scheduler_firewall_allows_command_names_as_data() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "value = 'sbatch squeue scontrol scancel sacct sinfo srun'; assert value",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_source_guidance_keeps_execution_out_of_registration_and_runtime() -> None:
    source = _normalized_repository_file("src/AGENTS.md")
    for required in (
        "scheduler-enabled execution adapters",
        "scheduler imports",
        "module scope",
        "no import-time cluster",
    ):
        assert required in source

    package = _normalized_repository_file("src/cyexm/AGENTS.md")
    for required in (
        "configuration, workflow, execution, and cli namespaces implement the "
        "phase 2 workflow surface",
        "must not become package-root exports",
        "phase 2b channel primitive usable without dask, distributed, "
        "dask-jobqueue, or yaml workflow dependencies",
    ):
        assert required in package
    assert "scheduler concerns do not belong in `cyexm.registration`" in package

    registration = _normalized_repository_file("src/cyexm/registration/AGENTS.md")
    for required in (
        "backend artifact writer methods",
        "must not own workflow scheduling",
        "completion-record primitives remain in `artifacts.py`",
        "`whole_image.py`, `chunked.py`, and `channels.py` each own orchestration "
        "of their scoped artifacts",
        "`channels.py` owns only one source channel's restart-safe transform "
        "artifact lifecycle",
        "configures native threads before requesting the lazy scientific backend",
        "must not schedule rounds, create clients or child tasks, or import "
        "workflow/executor code",
    ):
        assert required in registration

    runtime = _normalized_repository_file("src/cyexm/runtime/AGENTS.md")
    for required in (
        "worker process",
        "before importing ants, numpy, or scipy",
        "must not initialize or own a scheduler",
    ):
        assert required in runtime


def test_test_guidance_separates_boundary_objects_local_schedulers_and_hpc() -> None:
    tests = _normalized_repository_file("tests/AGENTS.md")
    for required in (
        "narrow concrete-image exception",
        "real registration belongs in integration",
        "real slurm submission",
        "external scheduler or cluster services",
        "bounded in-process or local-scheduler checks",
        "portable integration",
        "hpc",
        "portable scheduler firewall",
        "subprocess, `os.system`, and representative `os.exec*`/`os.spawn*`",
        "hpc-marked tests are explicitly exempt",
    ):
        assert required in tests

    unit = _normalized_repository_file("tests/unit/AGENTS.md")
    for required in (
        "phase 2 manifest, execution-profile, planning, sequential-executor, "
        "workflow, and cli policy",
        "plain python data",
        "channel and round-task modules defer scheduler and scientific runtimes",
        "real registration remains in integration tests",
        "inert scheduler shims",
        "red tests never reach host scheduler clients",
    ):
        assert required in unit

    integration = _normalized_repository_file("tests/integration/AGENTS.md")
    for required in (
        "approved scheduler dependency",
        "bounded local-scheduler",
        "access to an external scheduler or cluster service",
        "bounded in-process or local-scheduler checks",
        "portable integration",
        "actual slurm submission remains `hpc`",
    ):
        assert required in integration
    assert (
        "larger data, representative lab inputs, and scheduler access belong"
        not in integration
    )


def test_completed_guidance_plan_records_execution_state() -> None:
    plan_path = "docs/superpowers/plans/2026-07-19-phase-2-ready-agents-guidance.md"
    plan = _read_repository_file(plan_path)
    normalized_plan = " ".join(plan.split()).lower()

    assert "execution status: implementation complete" in normalized_plan
    assert "integration pending" in normalized_plan
    assert "- [ ]" not in plan
    assert plan.count("- [x]") == 39


def test_every_maintained_directory_has_local_agent_guidance() -> None:
    actual = {
        str(path.relative_to(REPOSITORY_ROOT))
        for path in REPOSITORY_ROOT.rglob("AGENTS.md")
        if not {".git", ".worktrees", ".venv"}
        & set(path.relative_to(REPOSITORY_ROOT).parts)
    }
    assert actual == AGENT_GUIDANCE_PATHS

    required_by_path = {
        ".github/AGENTS.md": (
            "least-privilege",
            "version-pinned",
            "public github-hosted runners",
            "lab paths",
        ),
        ".github/workflows/AGENTS.md": (
            "uv sync --locked --group dev --extra examples --extra hpc",
            "python 3.12",
            "all six native-thread",
            "unit and not hpc",
            "integration and not hpc",
            "bounded process-based localcluster",
            "portable slurm import, type, fake-construction, and preflight checks",
            "never invoke external scheduler commands",
            "explicitly selected hpc tests",
        ),
        "docs/superpowers/AGENTS.md": (
            "dated",
            "design",
            "implementation plan",
            "generated review packages",
        ),
        "docs/superpowers/plans/AGENTS.md": (
            "exact files",
            "tdd",
            "evidence",
            "historical plans",
        ),
        "docs/superpowers/specs/AGENTS.md": (
            "current behavior",
            "alternatives",
            "failure handling",
            "acceptance criteria",
            "erratum",
        ),
    }
    for path, required_fragments in required_by_path.items():
        guidance = _normalized_repository_file(path)
        for fragment in required_fragments:
            assert fragment in guidance, (path, fragment)


@pytest.mark.parametrize(
    ("path", "required_fragment"),
    tuple(
        (path, required_fragment)
        for path, required_fragments in PHASE_TWO_GUIDANCE_REQUIRED_FRAGMENTS.items()
        for required_fragment in required_fragments
    ),
)
def test_phase_two_package_guidance_enforces_boundaries(
    path: str, required_fragment: str
) -> None:
    guidance = _normalized_repository_file(path)
    assert required_fragment in guidance, (path, required_fragment)


def test_acceptance_audit_records_the_implementation_baseline_and_scope() -> None:
    configuration = tomllib.loads(_read_repository_file("pyproject.toml"))
    assert configuration["tool"]["uv"]["required-version"] == "==0.7.13"
    addopts = configuration["tool"]["pytest"]["ini_options"]["addopts"]
    assert "--strict-config" in addopts
    assert "--strict-markers" in addopts

    direct_dependencies = "\n".join(configuration["project"]["dependencies"]).lower()
    for forbidden in (
        "clearex",
        "dask",
        "distributed",
        "dask-jobqueue",
        "pyyaml",
    ):
        assert forbidden not in direct_dependencies

    acceptance = _read_repository_file("docs/phase-1-acceptance.md")
    assert "5365d184fb3c712e22e79b4b211c474cbebbbb57" in acceptance
    assert "cannot contain its own commit hash" in acceptance
    normalized_acceptance = " ".join(acceptance.split()).lower()
    for required_evidence in (
        "ANTSpyX",
        "ITK",
        "NumPy",
        "SciPy",
        "tifffile",
        "legacy parity",
        "import side effects",
        "thread defaults and overrides",
        "strict explicit thread counts",
        "external ZYX geometry boundary",
        "fail-closed TIFF validation",
        "source snapshot identity",
        "affine resume",
        "chunk resume",
        "serial chunk scalar parity",
        "transform provenance chain",
        "forbidden dependencies and exports",
        "Phase 2",
    ):
        assert required_evidence.lower() in normalized_acceptance
    assert "required uv version is pinned to exactly 0.7.13" in normalized_acceptance
    for required_snapshot in (
        "| Python | 3.12.11 |",
        "| ANTSpyX | 0.6.3 |",
        "| ITK | 5.4.5 |",
        "604 passed, 0 skipped, in 6.72 s",
        "40 passed, 0 skipped, in 4.07 s",
        "59 files already formatted",
        "644 passed in 8.77 s",
        "16 unit modules and 7 integration modules",
        "whole-image and serial-chunk workflow cache requests",
        "allowed grayscale ZYX, QYX, or singleton-Z layout",
        "affine, nonlinear, and optional fixed/moving crop records",
        "XYZ pixel spacing",
        "object-based affine inspection",
    ):
        assert required_snapshot.lower() in normalized_acceptance
    for stale_snapshot in (
        "3768882870f1b163fbbef8c8447c409d98c44394",
        "0fb61ba42f86cc29c5beb6e0fe5b4b7aaef197a3",
        "416 passed",
        "36 passed",
        "452 passed",
        "58 files already formatted",
        "6 integration modules",
    ):
        assert stale_snapshot not in acceptance

    readme = _read_repository_file("README.md")
    assert "[Phase 1 acceptance audit](docs/phase-1-acceptance.md)" in readme
    unit_command = 'uv run pytest -m "unit and not hpc" tests/unit -v'
    integration_command = (
        'uv run pytest -m "integration and not hpc" tests/integration -v'
    )
    assert unit_command in readme
    assert integration_command in readme
    root_guidance = _read_repository_file("AGENTS.md")
    assert unit_command in root_guidance
    assert integration_command in root_guidance

    ignored_paths = set(_read_repository_file(".gitignore").splitlines())
    assert {"__pycache__/", "*.py[cod]"} <= ignored_paths
