from __future__ import annotations

import os
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from cyexm.config._validation import (
    built_in_int,
    nonempty_string,
    require_exact_keys,
)
from cyexm.config._yaml import load_yaml_mapping
from cyexm.config.models import (
    ExecutionPaths,
    ExecutionProfile,
    ExecutorName,
    LocalDaskSpec,
    NativeThreadSpec,
    SlurmSpec,
)
from cyexm.errors import ConfigurationValidationError

_EXECUTORS = frozenset({"sequential", "local_dask", "slurm"})
_BASE_EXECUTION_KEYS = frozenset({"schema_version", "executor", "paths"})
_PATH_KEYS = frozenset({"dataset_root", "output_root", "worker_local_directory"})
_NATIVE_THREAD_KEYS = frozenset({"itk", "blas"})
_LOCAL_DASK_KEYS = frozenset(
    {
        "workers",
        "threads_per_worker",
        "processes",
        "memory_limit",
        "startup_timeout",
        "task_retries",
    }
)
_SLURM_KEYS = frozenset(
    {
        "queue",
        "workers",
        "cores_per_worker",
        "processes_per_worker",
        "memory",
        "walltime",
        "interface",
        "scheduler_interface",
        "startup_timeout",
        "task_retries",
        "job_name",
        "job_extra_directives",
    }
)

_ITK_THREAD_ENV_VAR = "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"
_BLAS_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

_MANAGED_SLURM_OPTIONS = (
    "--partition",
    "-p",
    "--mem",
    "--mem-per-cpu",
    "--cpus-per-task",
    "-c",
    "--ntasks",
    "-n",
    "--output",
    "-o",
    "--error",
    "-e",
)
_MANAGED_SLURM_LONG_OPTIONS = frozenset(
    option for option in _MANAGED_SLURM_OPTIONS if option.startswith("--")
)
_MANAGED_SLURM_SHORT_OPTIONS = frozenset(
    option for option in _MANAGED_SLURM_OPTIONS if not option.startswith("--")
)


def _choice(value: object, accepted: frozenset[str], yaml_path: str) -> str:
    if not isinstance(value, str) or value not in accepted:
        raise ConfigurationValidationError(
            f"{yaml_path} must be one of {sorted(accepted)!r}; got {value!r}"
        )
    return value


def _list(value: object, yaml_path: str) -> list[object]:
    if not isinstance(value, list):
        raise ConfigurationValidationError(f"{yaml_path} must be a list; got {value!r}")
    return value


def _resolved_path(value: object, yaml_path: str, source_directory: Path) -> Path:
    text = nonempty_string(value, yaml_path)
    path = Path(text)
    if not path.is_absolute():
        path = source_directory / path
    return path.resolve(strict=False)


def _paths(
    value: object,
    source_directory: Path,
    *,
    require_worker_scratch: bool,
) -> ExecutionPaths:
    mapping = require_exact_keys(value, _PATH_KEYS, "paths")
    dataset_root = _resolved_path(
        mapping["dataset_root"], "paths.dataset_root", source_directory
    )
    output_root = _resolved_path(
        mapping["output_root"], "paths.output_root", source_directory
    )
    if dataset_root.is_relative_to(output_root) or output_root.is_relative_to(
        dataset_root
    ):
        raise ConfigurationValidationError(
            "paths.dataset_root and output_root must be disjoint after resolution; "
            f"got {dataset_root} and {output_root}"
        )

    raw_worker_directory = mapping["worker_local_directory"]
    if require_worker_scratch and raw_worker_directory is None:
        raise ConfigurationValidationError(
            "paths.worker_local_directory must be set for worker-backed executors; "
            "got None"
        )
    worker_local_directory = (
        None
        if raw_worker_directory is None
        else _resolved_path(
            raw_worker_directory,
            "paths.worker_local_directory",
            source_directory,
        )
    )
    archive_root = Path("/archive").resolve(strict=False)
    if worker_local_directory is not None and worker_local_directory.is_relative_to(
        archive_root
    ):
        raise ConfigurationValidationError(
            "paths.worker_local_directory must not be /archive or below; "
            f"got {worker_local_directory}"
        )
    return ExecutionPaths(
        dataset_root=dataset_root,
        output_root=output_root,
        worker_local_directory=worker_local_directory,
    )


def _environment_thread_count(
    environ: Mapping[str, str], name: str, default: int = 1
) -> int:
    raw_value = environ.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value, 10)
    except (TypeError, ValueError) as error:
        raise ConfigurationValidationError(
            f"environment variable {name} must be a positive integer; got {raw_value!r}"
        ) from error
    if value < 1:
        raise ConfigurationValidationError(
            f"environment variable {name} must be a positive integer; got {raw_value!r}"
        )
    return value


def _resolve_inherited_native_threads(
    environ: Mapping[str, str],
) -> NativeThreadSpec:
    itk = _environment_thread_count(environ, _ITK_THREAD_ENV_VAR)
    blas_values = tuple(
        _environment_thread_count(environ, name) for name in _BLAS_THREAD_ENV_VARS
    )
    if len(set(blas_values)) != 1:
        details = ", ".join(
            f"{name}={value}"
            for name, value in zip(_BLAS_THREAD_ENV_VARS, blas_values, strict=True)
        )
        raise ConfigurationValidationError(
            "inherited BLAS thread values do not agree "
            f"({details}); set explicit native_threads.blas"
        )
    return NativeThreadSpec(itk=itk, blas=blas_values[0])


def _native_threads(value: object) -> NativeThreadSpec:
    mapping = require_exact_keys(value, _NATIVE_THREAD_KEYS, "native_threads")
    return NativeThreadSpec(
        itk=built_in_int(mapping["itk"], "native_threads.itk", minimum=1),
        blas=built_in_int(mapping["blas"], "native_threads.blas", minimum=1),
    )


def _local_dask(value: object) -> LocalDaskSpec:
    mapping = require_exact_keys(value, _LOCAL_DASK_KEYS, "local_dask")
    processes = mapping["processes"]
    if type(processes) is not bool:
        raise ConfigurationValidationError(
            f"local_dask.processes must be a built-in boolean; got {processes!r}"
        )
    return LocalDaskSpec(
        workers=built_in_int(mapping["workers"], "local_dask.workers", minimum=1),
        threads_per_worker=built_in_int(
            mapping["threads_per_worker"],
            "local_dask.threads_per_worker",
            minimum=1,
        ),
        processes=processes,
        memory_limit=nonempty_string(
            mapping["memory_limit"], "local_dask.memory_limit"
        ),
        startup_timeout=nonempty_string(
            mapping["startup_timeout"], "local_dask.startup_timeout"
        ),
        task_retries=built_in_int(mapping["task_retries"], "local_dask.task_retries"),
    )


def _job_extra_directives(value: object) -> tuple[str, ...]:
    directives: list[str] = []
    for index, item in enumerate(_list(value, "slurm.job_extra_directives")):
        yaml_path = f"slurm.job_extra_directives[{index}]"
        directive = nonempty_string(item, yaml_path)
        managed_option = _managed_slurm_option(directive, yaml_path)
        if managed_option is not None:
            raise ConfigurationValidationError(
                f"{yaml_path} conflicts with managed SLURM option "
                f"{managed_option}; got {directive!r}"
            )
        directives.append(directive)
    return tuple(directives)


def _managed_slurm_option(directive: str, yaml_path: str) -> str | None:
    try:
        tokens = shlex.split(directive)
    except ValueError as error:
        raise ConfigurationValidationError(
            f"{yaml_path} must contain valid shell-style tokens; got {directive!r}"
        ) from error

    for token in tokens:
        if token == "--":
            break
        if token.startswith("--"):
            option = token.partition("=")[0]
            if option in _MANAGED_SLURM_LONG_OPTIONS:
                return option
        elif token.startswith("-"):
            for name in token[1:]:
                option = f"-{name}"
                if option in _MANAGED_SLURM_SHORT_OPTIONS:
                    return option
    return None


def _slurm(value: object) -> SlurmSpec:
    mapping = require_exact_keys(value, _SLURM_KEYS, "slurm")
    processes_per_worker = mapping["processes_per_worker"]
    if type(processes_per_worker) is not int or processes_per_worker != 1:
        raise ConfigurationValidationError(
            "slurm.processes_per_worker must be the built-in integer 1; "
            f"got {processes_per_worker!r}"
        )
    return SlurmSpec(
        queue=nonempty_string(mapping["queue"], "slurm.queue"),
        workers=built_in_int(mapping["workers"], "slurm.workers", minimum=1),
        cores_per_worker=built_in_int(
            mapping["cores_per_worker"], "slurm.cores_per_worker", minimum=1
        ),
        processes_per_worker=processes_per_worker,
        memory=nonempty_string(mapping["memory"], "slurm.memory"),
        walltime=nonempty_string(mapping["walltime"], "slurm.walltime"),
        interface=nonempty_string(mapping["interface"], "slurm.interface"),
        scheduler_interface=nonempty_string(
            mapping["scheduler_interface"], "slurm.scheduler_interface"
        ),
        startup_timeout=nonempty_string(
            mapping["startup_timeout"], "slurm.startup_timeout"
        ),
        task_retries=built_in_int(mapping["task_retries"], "slurm.task_retries"),
        job_name=nonempty_string(mapping["job_name"], "slurm.job_name"),
        job_extra_directives=_job_extra_directives(mapping["job_extra_directives"]),
    )


def load_execution_profile(path: str | Path) -> ExecutionProfile:
    source_path = Path(path).resolve(strict=False)
    raw_mapping = load_yaml_mapping(source_path)
    raw_executor = raw_mapping.get("executor")
    executor = cast(
        ExecutorName,
        _choice(raw_executor, _EXECUTORS, "executor"),
    )
    expected_keys = _BASE_EXECUTION_KEYS
    if "native_threads" in raw_mapping:
        expected_keys |= frozenset({"native_threads"})
    if executor != "sequential":
        expected_keys |= frozenset({executor})
    mapping = require_exact_keys(raw_mapping, expected_keys, "execution")

    schema_version = mapping["schema_version"]
    if type(schema_version) is not int or schema_version != 1:
        raise ConfigurationValidationError(
            f"schema_version must be the built-in integer 1; got {schema_version!r}"
        )
    native_threads = (
        _native_threads(mapping["native_threads"])
        if "native_threads" in mapping
        else _resolve_inherited_native_threads(os.environ)
    )
    return ExecutionProfile(
        schema_version=schema_version,
        executor=executor,
        paths=_paths(
            mapping["paths"],
            source_path.parent,
            require_worker_scratch=executor != "sequential",
        ),
        native_threads=native_threads,
        local_dask=(
            _local_dask(mapping["local_dask"]) if executor == "local_dask" else None
        ),
        slurm=_slurm(mapping["slurm"]) if executor == "slurm" else None,
        source_path=source_path,
    )
