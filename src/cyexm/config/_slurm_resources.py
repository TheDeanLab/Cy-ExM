"""Derived SLURM job and process resources."""

from __future__ import annotations

from dataclasses import dataclass

from cyexm.config.memory import SlurmMemoryQuantity, normalize_slurm_memory
from cyexm.config.models import NativeThreadSpec, ParallelismSpec, SlurmSpec
from cyexm.errors import ConfigurationValidationError

_BINARY_GIB = 1024**3


@dataclass(frozen=True, slots=True)
class SlurmJobResources:
    """Validated resources for one owned SLURM job and its worker processes."""

    job_cap: int
    processes_per_job: int
    cores_per_job: int
    cores_per_process: int
    task_slots_per_job: int
    max_task_slots: int
    job_memory: SlurmMemoryQuantity
    process_memory_gib: int
    process_memory_bytes: int
    process_memory: str
    dask_managed_memory: str


def derive_slurm_job_resources(
    spec: SlurmSpec,
    native_threads: NativeThreadSpec,
    parallelism: ParallelismSpec,
) -> SlurmJobResources:
    """Validate and derive exact per-job and per-process SLURM resources."""
    processes = spec.processes_per_worker
    if parallelism.mode == "coarse" and processes != 1:
        raise ConfigurationValidationError(
            "coarse SLURM execution requires processes_per_worker == 1"
        )
    if spec.cores_per_worker % processes:
        raise ConfigurationValidationError(
            "slurm.cores_per_worker must be evenly divisible by "
            "slurm.processes_per_worker"
        )
    cores_per_process = spec.cores_per_worker // processes
    for name, value in (
        ("native_threads.itk", native_threads.itk),
        ("native_threads.blas", native_threads.blas),
    ):
        if value > cores_per_process:
            raise ConfigurationValidationError(
                f"{name} must not exceed {cores_per_process} cores per process"
            )

    memory = normalize_slurm_memory(spec.memory)
    normalized_gib = memory.scheduler_bytes // _BINARY_GIB
    process_gib = normalized_gib // processes
    if process_gib < 1:
        raise ConfigurationValidationError(
            "per-process Dask memory must be at least 1GiB"
        )
    return SlurmJobResources(
        job_cap=spec.workers,
        processes_per_job=processes,
        cores_per_job=spec.cores_per_worker,
        cores_per_process=cores_per_process,
        task_slots_per_job=processes,
        max_task_slots=spec.workers * processes,
        job_memory=memory,
        process_memory_gib=process_gib,
        process_memory_bytes=process_gib * _BINARY_GIB,
        process_memory=f"{process_gib}GiB",
        dask_managed_memory=f"{process_gib * processes}GiB",
    )


__all__ = ["SlurmJobResources", "derive_slurm_job_resources"]
