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


def _read_repository_file(path: str) -> str:
    return (REPOSITORY_ROOT / path).read_text(encoding="utf-8")


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
    assert "3768882870f1b163fbbef8c8447c409d98c44394" in acceptance
    assert "cannot contain its own commit hash" in acceptance
    for required_evidence in (
        "ANTSpyX",
        "ITK",
        "NumPy",
        "SciPy",
        "tifffile",
        "legacy parity",
        "import side effects",
        "thread defaults and overrides",
        "affine resume",
        "chunk resume",
        "forbidden dependencies and exports",
        "Phase 2",
    ):
        assert required_evidence.lower() in acceptance.lower()
    assert "required uv version is pinned to exactly 0.7.13" in acceptance.lower()

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
