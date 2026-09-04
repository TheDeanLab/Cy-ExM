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
class ExplicitXYRotationSpec:
    """Explicit physical XY rotation correction for one moving round."""

    mode: Literal["explicit"]
    angle_deg: float


@dataclass(frozen=True, slots=True)
class EstimatedXYRotationSpec:
    """Target-blind XY rotation estimation settings for one moving round."""

    mode: Literal["estimate"]
    search_range_deg: tuple[float, float] = (-45.0, 45.0)
    coarse_step_deg: float = 1.0
    refinement_step_deg: float = 0.1
    minimum_score_gain: float = 0.01
    minimum_peak_margin: float = 0.01


type XYRotationSpec = ExplicitXYRotationSpec | EstimatedXYRotationSpec


@dataclass(frozen=True, slots=True)
class InitialAlignmentSpec:
    """Optional geometric initialization for one moving round."""

    xy_rotation: XYRotationSpec


@dataclass(frozen=True, slots=True)
class RoundSpec:
    round: int
    strategy: RegistrationStrategy
    registration_channel: PurePosixPath
    channels: tuple[ChannelSpec, ...]
    chunk_size_zyx: ChunkSizeZYX | None = None
    overlap_fraction: float | None = None
    initial_alignment: InitialAlignmentSpec | None = None


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
class FeatureMaskSpec:
    """Optional physical morphology for an asinh registration-feature mask."""

    min_component_volume_um3: float = 1.0
    dilation_radius_um: float = 1.0


@dataclass(frozen=True, slots=True)
class RegistrationPreprocessingSpec:
    """Optional transform-estimation feature settings.

    ``filter_sigma_zyx`` is a Gaussian sigma in NumPy ZYX voxel units. YAML may
    supply a scalar or three values; the normalized Python model always stores the
    canonical three-value tuple.
    """

    filter_sigma_zyx: tuple[float, float, float] = (12.0, 12.0, 12.0)
    feature_mask: FeatureMaskSpec | None = None


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
    preprocessing: RegistrationPreprocessingSpec | None = None


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
