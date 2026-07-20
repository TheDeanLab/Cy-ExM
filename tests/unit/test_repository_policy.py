from __future__ import annotations

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
        "src/cyexm/io/AGENTS.md",
        "src/cyexm/registration/AGENTS.md",
        "src/cyexm/runtime/AGENTS.md",
        "tests/AGENTS.md",
        "tests/integration/AGENTS.md",
        "tests/unit/AGENTS.md",
    }
)


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


def test_ci_runs_only_portable_phase_one_gates_with_bounded_native_threads() -> None:
    assert _load_yaml(".github/workflows/ci.yml", github_actions=True) == {
        "name": "CI",
        "on": {"push": "", "pull_request": ""},
        "permissions": {"contents": "read"},
        "jobs": {
            "portable": {
                "name": "Portable Phase 1 gates",
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
                        "run": "uv sync --locked --group dev --extra examples",
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
    for forbidden in ("clearex", "dask", "slurm", "/archive", "/project"):
        assert forbidden not in workflow


def test_root_guidance_tracks_stable_phase_one_and_gates_phase_two() -> None:
    root = _normalized_repository_file("AGENTS.md")
    for required in (
        "phase 1 is the current stable runtime",
        "dedicated phase 2 runtime design and implementation plan",
        "scientific dataset manifests",
        "execution profiles",
        "serial chunk workflow remains",
        "nested chunk multiprocessing",
        "uv lock --check",
        "uv sync --locked --group dev --extra examples",
    ):
        assert required in root

    root_lines = set(_read_repository_file("AGENTS.md").splitlines())
    assert "uv lock" not in root_lines
    assert "uv sync --group dev --extra examples" not in root_lines


def test_documentation_and_examples_cannot_lead_phase_two_runtime() -> None:
    documentation = _normalized_repository_file("docs/AGENTS.md")
    for required in (
        "merged phase 1 behavior",
        "proposed phase 2",
        "not installable behavior",
        "dedicated runtime design",
    ):
        assert required in documentation

    examples = _normalized_repository_file("examples/AGENTS.md")
    for required in (
        "phase 1 examples are the current runnable surface",
        "runtime api is implemented and accepted",
        "examples never lead production behavior",
        "scientific manifest",
        "execution profile",
    ):
        assert required in examples


def test_source_guidance_keeps_execution_out_of_registration_and_runtime() -> None:
    source = _normalized_repository_file("src/AGENTS.md")
    for required in (
        "future execution boundary",
        "scheduler imports",
        "module scope",
        "no import-time cluster",
    ):
        assert required in source

    package = _normalized_repository_file("src/cyexm/AGENTS.md")
    assert "workflow or configuration namespace" in package
    assert "scheduler concerns do not belong in `cyexm.registration`" in package

    registration = _normalized_repository_file("src/cyexm/registration/AGENTS.md")
    for required in (
        "backend artifact writer methods",
        "`artifacts.py`, `whole_image.py`, and `chunked.py`",
        "must not own workflow scheduling",
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
        "hpc",
    ):
        assert required in tests

    unit = _normalized_repository_file("tests/unit/AGENTS.md")
    for required in (
        "future manifest and planning policy",
        "plain python data",
        "must not import scheduler",
    ):
        assert required in unit

    integration = _normalized_repository_file("tests/integration/AGENTS.md")
    for required in (
        "approved scheduler dependency",
        "bounded local-scheduler",
        "actual slurm submission remains `hpc`",
    ):
        assert required in integration


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
            "uv sync --locked --group dev --extra examples",
            "python 3.12",
            "all six native-thread",
            "unit and not hpc",
            "integration and not hpc",
            "slurm",
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
