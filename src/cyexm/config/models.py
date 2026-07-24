from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from cyexm._nonlinear_metric import NonlinearMetricName

type PixelSpacingXYZ = tuple[float, float, float]
type ChunkSizeZYX = tuple[int, int, int]
type RegistrationStrategy = Literal["whole_image", "serial_chunked"]
type ChannelInterpolation = Literal["linear", "nearestNeighbor"]
type ExecutorName = Literal["sequential", "local_dask", "slurm"]
type ParallelismMode = Literal["coarse", "fine_grained"]


@dataclass(frozen=True, slots=True)
class ChannelSpec:
    file: PurePosixPath
    interpolation: ChannelInterpolation


@dataclass(frozen=True, slots=True)
class RoundSpec:
    round: int
    strategy: RegistrationStrategy
    registration_channel: PurePosixPath
    channels: tuple[ChannelSpec, ...]
    chunk_size_zyx: ChunkSizeZYX | None = None
    overlap_fraction: float | None = None


@dataclass(frozen=True, slots=True)
class ReferenceChannelSpec:
    file: PurePosixPath


@dataclass(frozen=True, slots=True)
class ReferenceSpec:
    round: int
    registration_channel: PurePosixPath
    channels: tuple[ReferenceChannelSpec, ...]


@dataclass(frozen=True, slots=True)
class RegistrationStageSpec:
    type: str
    accuracy: str


@dataclass(frozen=True, slots=True)
class NonlinearRegistrationStageSpec:
    type: str
    accuracy: str
    metric: NonlinearMetricName
    radius: int | None


@dataclass(frozen=True, slots=True)
class RegistrationSpec:
    crop: bool
    linear: RegistrationStageSpec
    nonlinear: NonlinearRegistrationStageSpec


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    schema_version: int
    dataset_id: str
    voxel_spacing_um: PixelSpacingXYZ
    reference: ReferenceSpec
    registration: RegistrationSpec
    rounds: tuple[RoundSpec, ...]


@dataclass(frozen=True, slots=True)
class ExecutionPaths:
    dataset_root: Path
    output_root: Path
    worker_local_directory: Path | None


@dataclass(frozen=True, slots=True)
class NativeThreadSpec:
    itk: int
    blas: int


@dataclass(frozen=True, slots=True)
class LocalDaskSpec:
    workers: int
    threads_per_worker: int
    processes: bool
    memory_limit: str
    startup_timeout: str
    task_retries: int


@dataclass(frozen=True, slots=True)
class SlurmSpec:
    queue: str
    workers: int
    cores_per_worker: int
    processes_per_worker: int
    memory: str
    walltime: str
    interface: str
    scheduler_interface: str
    startup_timeout: str
    task_retries: int
    job_name: str
    job_extra_directives: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ParallelismSpec:
    mode: ParallelismMode = "coarse"


@dataclass(frozen=True, slots=True)
class ExecutionProfile:
    schema_version: int
    executor: ExecutorName
    paths: ExecutionPaths
    native_threads: NativeThreadSpec
    local_dask: LocalDaskSpec | None
    slurm: SlurmSpec | None
    source_path: Path
    parallelism: ParallelismSpec = ParallelismSpec()
