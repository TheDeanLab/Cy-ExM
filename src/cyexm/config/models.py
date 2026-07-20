from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

type PixelSpacingXYZ = tuple[float, float, float]
type ChunkSizeZYX = tuple[int, int, int]
type RegistrationStrategy = Literal["whole_image", "serial_chunked"]
type ChannelInterpolation = Literal["linear", "nearestNeighbor"]


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
class ReferenceSpec:
    round: int
    channel: PurePosixPath


@dataclass(frozen=True, slots=True)
class RegistrationStageSpec:
    type: str
    accuracy: str


@dataclass(frozen=True, slots=True)
class RegistrationSpec:
    crop: bool
    linear: RegistrationStageSpec
    nonlinear: RegistrationStageSpec


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    schema_version: int
    dataset_id: str
    voxel_spacing_um: PixelSpacingXYZ
    reference: ReferenceSpec
    registration: RegistrationSpec
    rounds: tuple[RoundSpec, ...]
