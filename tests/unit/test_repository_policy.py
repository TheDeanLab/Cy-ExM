from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

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


def test_pre_commit_runs_the_locked_portable_gates_through_uv() -> None:
    configuration = _read_repository_file(".pre-commit-config.yaml")

    assert re.findall(r"^  - repo: (.+)$", configuration, flags=re.MULTILINE) == [
        "local"
    ]
    for hook_id, entry in (
        ("ruff-check", "uv run ruff check"),
        ("ruff-format", "uv run ruff format --check"),
        ("basedpyright", "uv run basedpyright"),
        ("pytest-unit", "uv run pytest tests/unit -q"),
    ):
        hook = re.search(
            rf"^      - id: {re.escape(hook_id)}\n(?P<body>(?:^        .+\n?)+)",
            configuration,
            flags=re.MULTILINE,
        )
        assert hook is not None, hook_id
        assert f"entry: {entry}" in hook["body"]
        assert "language: system" in hook["body"]


def test_ci_runs_only_portable_phase_one_gates_with_bounded_native_threads() -> None:
    workflow = _read_repository_file(".github/workflows/ci.yml")

    assert "runs-on: ubuntu-latest" in workflow
    assert "timeout-minutes:" in workflow
    assert "uses: actions/checkout@v7" in workflow
    assert "uses: astral-sh/setup-uv@v8.3.2" in workflow
    assert 'python-version: "3.12"' in workflow
    assert "uv sync --locked --group dev --extra examples" in workflow
    for command in (
        "uv run ruff check .",
        "uv run ruff format --check .",
        "uv run basedpyright",
        "uv run pytest tests/unit -v",
        'uv run pytest -m "integration and not hpc" tests/integration -v',
    ):
        assert command in workflow

    job_environment = re.search(
        r"^    env:\n(?P<body>(?:^      [A-Z0-9_]+: [^\n]+\n)+)",
        workflow,
        flags=re.MULTILINE,
    )
    assert job_environment is not None
    configured_threads = dict(
        re.findall(
            r'^      ([A-Z0-9_]+): "?([^"\n]+)"?$',
            job_environment["body"],
            flags=re.MULTILINE,
        )
    )
    assert configured_threads == dict.fromkeys(THREAD_ENVIRONMENT, "1")

    lowered = workflow.lower()
    for forbidden in (
        "/archive/bioinformatics",
        "/project/bioinformatics",
        "clearex",
        "dask",
        "slurm",
    ):
        assert forbidden not in lowered


def test_acceptance_audit_records_the_implementation_baseline_and_scope() -> None:
    configuration = tomllib.loads(_read_repository_file("pyproject.toml"))
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

    readme = _read_repository_file("README.md")
    assert "[Phase 1 acceptance audit](docs/phase-1-acceptance.md)" in readme
