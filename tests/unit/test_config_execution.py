from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from cyexm.config import (
    ExecutionPaths,
    ExecutionProfile,
    LocalDaskSpec,
    NativeThreadSpec,
    SlurmSpec,
    load_execution_profile,
)
from cyexm.config.execution import _resolve_inherited_native_threads
from cyexm.errors import ConfigurationValidationError

pytestmark = pytest.mark.unit

_NATIVE_THREAD_ENV_VARS = (
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

SEQUENTIAL_YAML = """\
schema_version: 1
executor: sequential
paths:
  dataset_root: ../data
  output_root: ../results
  worker_local_directory: null
native_threads:
  itk: 1
  blas: 1
"""

LOCAL_DASK_YAML = """\
schema_version: 1
executor: local_dask
paths:
  dataset_root: ../data
  output_root: ../results
  worker_local_directory: ../scratch
native_threads:
  itk: 2
  blas: 3
local_dask:
  workers: 4
  threads_per_worker: 1
  processes: false
  memory_limit: 24GB
  startup_timeout: "30s"
  task_retries: 2
"""

SLURM_YAML = """\
schema_version: 1
executor: slurm
paths:
  dataset_root: /archive/path/to/source
  output_root: /archive/path/to/results
  worker_local_directory: /project/path/to/scratch
native_threads:
  itk: 1
  blas: 1
slurm:
  queue: 512GB
  workers: 3
  cores_per_worker: 72
  processes_per_worker: 1
  memory: 500GB
  walltime: "72:00:00"
  interface: ib0
  scheduler_interface: ib0
  startup_timeout: "15m"
  task_retries: 0
  job_name: cyexm-registration
  job_extra_directives: []
"""


def _valid_data(executor: str) -> dict[str, object]:
    content = {
        "sequential": SEQUENTIAL_YAML,
        "local_dask": LOCAL_DASK_YAML,
        "slurm": SLURM_YAML,
    }[executor]
    value = yaml.safe_load(content)
    assert isinstance(value, dict)
    return value


def _write_profile(tmp_path: Path, data: dict[str, object]) -> Path:
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _load_profile(tmp_path: Path, data: dict[str, object]) -> ExecutionProfile:
    return load_execution_profile(_write_profile(tmp_path, data))


def _mapping_at(data: dict[str, object], key: str) -> dict[str, object]:
    value = data[key]
    assert isinstance(value, dict)
    return value


def test_execution_paths_resolve_relative_to_profile(tmp_path: Path) -> None:
    profile_path = tmp_path / "profiles" / "sequential.yaml"
    profile_path.parent.mkdir()
    profile_path.write_text(SEQUENTIAL_YAML, encoding="utf-8")

    profile = load_execution_profile(profile_path)

    assert profile.executor == "sequential"
    assert profile.paths == ExecutionPaths(
        dataset_root=(profile_path.parent / "../data").resolve(),
        output_root=(profile_path.parent / "../results").resolve(),
        worker_local_directory=None,
    )
    assert profile.native_threads == NativeThreadSpec(itk=1, blas=1)
    assert profile.local_dask is None
    assert profile.slurm is None
    assert profile.source_path == profile_path.resolve()
    assert not profile.paths.dataset_root.exists()
    assert not profile.paths.output_root.exists()


def test_local_dask_profile_binds_exact_values(tmp_path: Path) -> None:
    profile_path = tmp_path / "profiles" / "local.yaml"
    profile_path.parent.mkdir()
    profile_path.write_text(LOCAL_DASK_YAML, encoding="utf-8")

    profile = load_execution_profile(profile_path)

    assert profile.executor == "local_dask"
    assert profile.paths.worker_local_directory == (tmp_path / "scratch").resolve()
    assert profile.native_threads == NativeThreadSpec(itk=2, blas=3)
    assert profile.local_dask == LocalDaskSpec(
        workers=4,
        threads_per_worker=1,
        processes=False,
        memory_limit="24GB",
        startup_timeout="30s",
        task_retries=2,
    )
    assert profile.slurm is None


def test_slurm_profile_binds_approved_512gb_values(tmp_path: Path) -> None:
    profile = _load_profile(tmp_path, _valid_data("slurm"))

    assert profile.executor == "slurm"
    assert profile.paths == ExecutionPaths(
        dataset_root=Path("/archive/path/to/source").resolve(),
        output_root=Path("/archive/path/to/results").resolve(),
        worker_local_directory=Path("/project/path/to/scratch").resolve(),
    )
    assert profile.slurm == SlurmSpec(
        queue="512GB",
        workers=3,
        cores_per_worker=72,
        processes_per_worker=1,
        memory="500GB",
        walltime="72:00:00",
        interface="ib0",
        scheduler_interface="ib0",
        startup_timeout="15m",
        task_retries=0,
        job_name="cyexm-registration",
        job_extra_directives=(),
    )
    assert profile.local_dask is None


def test_execution_models_are_frozen(tmp_path: Path) -> None:
    profile = _load_profile(tmp_path, _valid_data("local_dask"))

    with pytest.raises(FrozenInstanceError):
        profile.native_threads.itk = 8  # type: ignore[misc]


@pytest.mark.parametrize(
    ("executor", "extra_section"),
    [
        ("sequential", "local_dask"),
        ("sequential", "slurm"),
        ("local_dask", "slurm"),
        ("slurm", "local_dask"),
    ],
)
def test_executor_rejects_mismatched_sections(
    tmp_path: Path, executor: str, extra_section: str
) -> None:
    data = _valid_data(executor)
    data[extra_section] = _valid_data(extra_section)[extra_section]

    with pytest.raises(ConfigurationValidationError, match="execution keys"):
        _load_profile(tmp_path, data)


@pytest.mark.parametrize("executor", ["local_dask", "slurm"])
def test_executor_requires_its_matching_section(tmp_path: Path, executor: str) -> None:
    data = _valid_data(executor)
    del data[executor]

    with pytest.raises(ConfigurationValidationError, match="execution keys"):
        _load_profile(tmp_path, data)


@pytest.mark.parametrize(
    ("section", "executor"),
    [
        (None, "sequential"),
        ("paths", "sequential"),
        ("native_threads", "sequential"),
        ("local_dask", "local_dask"),
        ("slurm", "slurm"),
    ],
)
def test_execution_profile_rejects_unknown_keys(
    tmp_path: Path, section: str | None, executor: str
) -> None:
    data = _valid_data(executor)
    target = data if section is None else _mapping_at(data, section)
    target["unexpected"] = "value"

    with pytest.raises(ConfigurationValidationError, match="keys must be exactly"):
        _load_profile(tmp_path, data)


@pytest.mark.parametrize("value", [True, 1.0, 0, 2])
def test_schema_version_requires_builtin_integer_one(
    tmp_path: Path, value: object
) -> None:
    data = _valid_data("sequential")
    data["schema_version"] = value

    with pytest.raises(
        ConfigurationValidationError,
        match="schema_version must be the built-in integer 1",
    ):
        _load_profile(tmp_path, data)


def test_executor_rejects_unknown_name(tmp_path: Path) -> None:
    data = _valid_data("sequential")
    data["executor"] = "threads"

    with pytest.raises(ConfigurationValidationError, match="executor must be one of"):
        _load_profile(tmp_path, data)


@pytest.mark.parametrize(
    ("dataset_root", "output_root"),
    [
        ("data", "data"),
        ("data", "data/results"),
        ("data/source", "data"),
    ],
)
def test_dataset_and_output_roots_must_be_disjoint(
    tmp_path: Path, dataset_root: str, output_root: str
) -> None:
    data = _valid_data("sequential")
    paths = _mapping_at(data, "paths")
    paths["dataset_root"] = dataset_root
    paths["output_root"] = output_root

    with pytest.raises(
        ConfigurationValidationError,
        match="dataset_root and output_root must be disjoint",
    ):
        _load_profile(tmp_path, data)


@pytest.mark.parametrize("scratch", ["/archive", "/archive/cyexm/scratch"])
def test_worker_scratch_rejects_archive_paths(tmp_path: Path, scratch: str) -> None:
    data = _valid_data("sequential")
    _mapping_at(data, "paths")["worker_local_directory"] = scratch

    with pytest.raises(
        ConfigurationValidationError,
        match="worker_local_directory must not be /archive or below",
    ):
        _load_profile(tmp_path, data)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset_root", ""),
        ("output_root", "results\nother"),
        ("worker_local_directory", "scratch\x00bad"),
        ("dataset_root", 1),
    ],
)
def test_paths_require_nonempty_single_line_strings(
    tmp_path: Path, field: str, value: object
) -> None:
    data = _valid_data("sequential")
    _mapping_at(data, "paths")[field] = value

    with pytest.raises(
        ConfigurationValidationError,
        match="must be a nonempty single-line string",
    ):
        _load_profile(tmp_path, data)


@pytest.mark.parametrize("field", ["itk", "blas"])
@pytest.mark.parametrize("value", [True, 1.0, 0, -1])
def test_native_thread_counts_require_positive_builtin_integers(
    tmp_path: Path, field: str, value: object
) -> None:
    data = _valid_data("sequential")
    _mapping_at(data, "native_threads")[field] = value

    with pytest.raises(
        ConfigurationValidationError,
        match=rf"native_threads\.{field} must be a built-in integer >= 1",
    ):
        _load_profile(tmp_path, data)


def test_omitted_native_threads_default_to_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in _NATIVE_THREAD_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    data = _valid_data("sequential")
    del data["native_threads"]

    profile = _load_profile(tmp_path, data)

    assert profile.native_threads == NativeThreadSpec(itk=1, blas=1)


def test_inherited_native_threads_accept_uniform_blas_values() -> None:
    environ = {
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": "2",
        "OMP_NUM_THREADS": "3",
        "OPENBLAS_NUM_THREADS": "3",
        "MKL_NUM_THREADS": "3",
        "NUMEXPR_NUM_THREADS": "3",
        "VECLIB_MAXIMUM_THREADS": "3",
    }

    resolved = _resolve_inherited_native_threads(environ)

    assert resolved == NativeThreadSpec(itk=2, blas=3)
    assert environ["OMP_NUM_THREADS"] == "3"


def test_inherited_native_threads_reject_unequal_blas_values() -> None:
    environ = {
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "3",
        "MKL_NUM_THREADS": "2",
        "NUMEXPR_NUM_THREADS": "2",
        "VECLIB_MAXIMUM_THREADS": "2",
    }

    with pytest.raises(
        ConfigurationValidationError,
        match=r"set explicit native_threads\.blas",
    ):
        _resolve_inherited_native_threads(environ)


def test_loading_inherited_threads_does_not_mutate_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in _NATIVE_THREAD_ENV_VARS:
        monkeypatch.setenv(name, "4")
    before = dict(os.environ)
    data = _valid_data("sequential")
    del data["native_threads"]

    profile = _load_profile(tmp_path, data)

    assert profile.native_threads == NativeThreadSpec(itk=4, blas=4)
    assert dict(os.environ) == before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("workers", True),
        ("workers", 0),
        ("threads_per_worker", 1.0),
        ("threads_per_worker", -1),
    ],
)
def test_local_worker_counts_require_positive_builtin_integers(
    tmp_path: Path, field: str, value: object
) -> None:
    data = _valid_data("local_dask")
    _mapping_at(data, "local_dask")[field] = value

    with pytest.raises(
        ConfigurationValidationError,
        match=rf"local_dask\.{field} must be a built-in integer >= 1",
    ):
        _load_profile(tmp_path, data)


@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_local_processes_requires_a_builtin_boolean(
    tmp_path: Path, value: object
) -> None:
    data = _valid_data("local_dask")
    _mapping_at(data, "local_dask")["processes"] = value

    with pytest.raises(
        ConfigurationValidationError,
        match=r"local_dask\.processes must be a built-in boolean",
    ):
        _load_profile(tmp_path, data)


@pytest.mark.parametrize(
    ("executor", "field"),
    [("local_dask", "startup_timeout"), ("slurm", "startup_timeout")],
)
@pytest.mark.parametrize("value", [0, "", "10m\n20m", "10m\x00bad"])
def test_startup_timeouts_require_nonempty_single_line_strings(
    tmp_path: Path, executor: str, field: str, value: object
) -> None:
    data = _valid_data(executor)
    _mapping_at(data, executor)[field] = value

    with pytest.raises(
        ConfigurationValidationError,
        match=rf"{executor}\.{field} must be a nonempty single-line string",
    ):
        _load_profile(tmp_path, data)


@pytest.mark.parametrize("executor", ["local_dask", "slurm"])
@pytest.mark.parametrize("value", [True, 1.0, -1])
def test_task_retries_require_nonnegative_builtin_integers(
    tmp_path: Path, executor: str, value: object
) -> None:
    data = _valid_data(executor)
    _mapping_at(data, executor)["task_retries"] = value

    with pytest.raises(
        ConfigurationValidationError,
        match=rf"{executor}\.task_retries must be a built-in integer >= 0",
    ):
        _load_profile(tmp_path, data)


@pytest.mark.parametrize("field", ["workers", "cores_per_worker"])
@pytest.mark.parametrize("value", [True, 1.0, 0])
def test_slurm_resource_counts_require_positive_builtin_integers(
    tmp_path: Path, field: str, value: object
) -> None:
    data = _valid_data("slurm")
    _mapping_at(data, "slurm")[field] = value

    with pytest.raises(
        ConfigurationValidationError,
        match=rf"slurm\.{field} must be a built-in integer >= 1",
    ):
        _load_profile(tmp_path, data)


@pytest.mark.parametrize("value", [True, 0, 2, 1.0])
def test_slurm_process_count_must_be_builtin_integer_one(
    tmp_path: Path, value: object
) -> None:
    data = _valid_data("slurm")
    _mapping_at(data, "slurm")["processes_per_worker"] = value

    with pytest.raises(
        ConfigurationValidationError,
        match=r"slurm\.processes_per_worker must be the built-in integer 1",
    ):
        _load_profile(tmp_path, data)


@pytest.mark.parametrize(
    "field",
    [
        "queue",
        "memory",
        "walltime",
        "interface",
        "scheduler_interface",
        "job_name",
    ],
)
@pytest.mark.parametrize("value", [1, "", "bad\nvalue", "bad\x00value"])
def test_slurm_text_fields_require_nonempty_single_line_strings(
    tmp_path: Path, field: str, value: object
) -> None:
    data = _valid_data("slurm")
    _mapping_at(data, "slurm")[field] = value

    with pytest.raises(
        ConfigurationValidationError,
        match=rf"slurm\.{field} must be a nonempty single-line string",
    ):
        _load_profile(tmp_path, data)


@pytest.mark.parametrize(
    "directive",
    [
        "--partition=512GB",
        "-p 512GB",
        "--mem=500GB",
        "--mem-per-cpu=8GB",
        "--cpus-per-task=72",
        "-c 72",
        "--ntasks=1",
        "-n 1",
        "--output=job.out",
        "-o job.out",
        "--error=job.err",
        "-e job.err",
    ],
)
def test_slurm_extra_directives_reject_managed_options(
    tmp_path: Path, directive: str
) -> None:
    data = _valid_data("slurm")
    _mapping_at(data, "slurm")["job_extra_directives"] = [f"  {directive}"]

    with pytest.raises(
        ConfigurationValidationError,
        match="conflicts with managed SLURM option",
    ):
        _load_profile(tmp_path, data)


@pytest.mark.parametrize("directive", ["", "bad\nvalue", "bad\x00value", 1])
def test_slurm_extra_directives_require_single_line_strings(
    tmp_path: Path, directive: object
) -> None:
    data = _valid_data("slurm")
    _mapping_at(data, "slurm")["job_extra_directives"] = [directive]

    with pytest.raises(
        ConfigurationValidationError,
        match=r"slurm\.job_extra_directives\[0\] must be a nonempty single-line string",
    ):
        _load_profile(tmp_path, data)


def test_slurm_extra_directives_bind_as_an_immutable_tuple(tmp_path: Path) -> None:
    data = _valid_data("slurm")
    _mapping_at(data, "slurm")["job_extra_directives"] = [
        "--exclusive",
        "--mail-type=FAIL",
    ]

    profile = _load_profile(tmp_path, data)

    assert profile.slurm is not None
    assert profile.slurm.job_extra_directives == (
        "--exclusive",
        "--mail-type=FAIL",
    )


def test_slurm_extra_directives_requires_a_list(tmp_path: Path) -> None:
    data = _valid_data("slurm")
    _mapping_at(data, "slurm")["job_extra_directives"] = "--exclusive"

    with pytest.raises(
        ConfigurationValidationError,
        match=r"slurm\.job_extra_directives must be a list",
    ):
        _load_profile(tmp_path, data)


def test_importing_config_does_not_load_runtimes_or_mutate_environment() -> None:
    code = """
import os
import sys

before = dict(os.environ)
import cyexm.config
after = dict(os.environ)

assert before == after
for root in ("ants", "numpy", "scipy", "dask", "distributed", "dask_jobqueue"):
    assert not any(name == root or name.startswith(root + ".") for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
