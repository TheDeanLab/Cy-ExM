from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from cyexm.config.models import (
    ChannelInterpolation,
    ChunkSizeZYX,
    DatasetManifest,
    ExecutionProfile,
    NativeThreadSpec,
    PixelSpacingXYZ,
    RegistrationSpec,
    RegistrationStrategy,
)
from cyexm.spatial import ImageGeometry


def canonical_json(value: object) -> str:
    """Encode a JSON-compatible value in the workflow's canonical form."""
    return json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_fingerprint(encoded: str) -> str:
    """Hash a canonical UTF-8 JSON string with SHA-256."""
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _detached_json_value(value: object, *, location: str) -> object:
    if isinstance(value, Mapping):
        detached: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(
                    f"{location} keys must be exact built-in strings; got {key!r}"
                )
            detached[key] = _detached_json_value(
                item,
                location=f"{location}.{key}",
            )
        return detached
    if type(value) is list:
        return [
            _detached_json_value(item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{location} must contain only finite floats")
        return value
    raise TypeError(f"{location} contains a non-JSON-compatible value: {value!r}")


def _canonical_execution_evidence(value: object) -> str:
    """Detach and canonically encode executor-owned operational evidence."""
    detached = _detached_json_value(value, location="execution_evidence")
    if type(detached) is not dict:
        raise TypeError("execution_evidence must be a JSON object")
    return canonical_json(detached)


@dataclass(frozen=True, slots=True)
class ChannelTask:
    source: Path
    output: Path
    interpolation: ChannelInterpolation


@dataclass(frozen=True, slots=True)
class RoundTask:
    dataset_id: str
    round: int
    strategy: RegistrationStrategy
    fixed_image: Path
    moving_image: Path
    registration_directory: Path
    log_path: Path
    voxel_spacing_xyz: PixelSpacingXYZ
    registration: RegistrationSpec
    native_threads: NativeThreadSpec
    channels: tuple[ChannelTask, ...]
    chunk_size_zyx: ChunkSizeZYX | None
    overlap_fraction: float | None
    force: bool = False


@dataclass(frozen=True, slots=True)
class ImageHeader:
    path: Path
    shape_zyx: tuple[int, int, int]
    dtype: str
    axes: str
    geometry: ImageGeometry | None
    st_size: int
    st_mtime_ns: int


@dataclass(frozen=True, slots=True)
class RoundFailure:
    stage: str
    exception_type: str
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "exception_type": self.exception_type,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class RoundResult:
    dataset_id: str
    round: int
    status: Literal["reused", "completed", "failed", "cancelled"]
    started_at: str
    ended_at: str
    duration_seconds: float
    hostname: str
    pid: int
    dask_task_key: str | None
    slurm_job_id: str | None
    strategy: RegistrationStrategy
    native_threads: tuple[tuple[str, int], ...]
    artifact_paths: tuple[str, ...]
    reused_artifacts: tuple[str, ...]
    log_path: str
    failure: RoundFailure | None

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "round": self.round,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": self.duration_seconds,
            "hostname": self.hostname,
            "pid": self.pid,
            "dask_task_key": self.dask_task_key,
            "slurm_job_id": self.slurm_job_id,
            "strategy": self.strategy,
            "native_threads": [list(item) for item in self.native_threads],
            "artifact_paths": list(self.artifact_paths),
            "reused_artifacts": list(self.reused_artifacts),
            "log_path": self.log_path,
            "failure": None if self.failure is None else self.failure.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class WorkflowPlan:
    schema_version: int
    dataset_id: str
    manifest: DatasetManifest
    execution: ExecutionProfile
    fixed_image: Path
    tasks: tuple[RoundTask, ...]
    expected_artifacts: tuple[Path, ...]
    scientific_json: str
    scientific_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tasks",
            tuple(sorted(self.tasks, key=lambda task: task.round)),
        )


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    dataset_id: str
    scientific_fingerprint: str
    status: Literal["completed", "failed", "cancelled"]
    started_at: str
    ended_at: str
    rounds: tuple[RoundResult, ...]
    run_record: str
    execution_evidence_json: str = "{}"

    @property
    def succeeded(self) -> bool:
        return self.status == "completed"

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "scientific_fingerprint": self.scientific_fingerprint,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "rounds": [result.to_dict() for result in self.rounds],
            "run_record": self.run_record,
            "execution_evidence_json": self.execution_evidence_json,
        }


@dataclass(frozen=True, slots=True)
class RoundStatus:
    round: int
    status: Literal["pending", "partial", "complete", "invalid"]
    complete_artifacts: tuple[str, ...]
    incomplete_artifacts: tuple[str, ...]
    invalid_artifacts: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "round": self.round,
            "status": self.status,
            "complete_artifacts": list(self.complete_artifacts),
            "incomplete_artifacts": list(self.incomplete_artifacts),
            "invalid_artifacts": list(self.invalid_artifacts),
        }


@dataclass(frozen=True, slots=True)
class WorkflowStatus:
    dataset_id: str
    scientific_fingerprint: str | None
    rounds: tuple[RoundStatus, ...]
    complete: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "scientific_fingerprint": self.scientific_fingerprint,
            "rounds": [status.to_dict() for status in self.rounds],
            "complete": self.complete,
        }


__all__ = (
    "ChannelTask",
    "ImageHeader",
    "RoundFailure",
    "RoundResult",
    "RoundStatus",
    "RoundTask",
    "WorkflowPlan",
    "WorkflowResult",
    "WorkflowStatus",
    "canonical_json",
    "sha256_fingerprint",
)
