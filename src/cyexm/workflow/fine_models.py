"""Immutable internal task and result values for fine-grained execution."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

from cyexm.chunking import ChunkInfo

if TYPE_CHECKING:
    from cyexm.config.models import NativeThreadSpec, RegistrationStrategy
    from cyexm.registration.prepared import PreparedRoundPaths
    from cyexm.workflow._round_provenance import RegistrationArtifactPaths
    from cyexm.workflow.models import ChannelTask, RoundFailure, RoundTask

type FineTaskStage = Literal[
    "preparation",
    "chunk",
    "assembly",
    "whole-image",
    "channel",
    "validation",
]
type FineTaskStatus = Literal["reused", "completed", "failed", "cancelled"]

_STAGES = frozenset(
    {"preparation", "chunk", "assembly", "whole-image", "channel", "validation"}
)
_INDEXED_STAGES = frozenset({"chunk", "channel"})
_INDEX_PATTERN = re.compile(r"[0-9]{4}")


def _json_object(value: object, name: str) -> dict[str, object]:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a strict JSON-compatible object") from error
    if not isinstance(decoded, dict) or not all(
        isinstance(key, str) for key in decoded
    ):
        raise ValueError(f"{name} must be a strict JSON-compatible object")
    return decoded


@dataclass(frozen=True, slots=True)
class FineTaskKey:
    """Stable identity for one internal workflow task."""

    round: int
    stage: FineTaskStage
    item: str

    def __post_init__(self) -> None:
        if isinstance(self.round, bool) or not isinstance(self.round, int):
            raise TypeError("round must be a non-negative integer")
        if self.round < 0:
            raise ValueError("round must be a non-negative integer")
        if self.stage not in _STAGES:
            raise ValueError(f"unsupported fine task stage: {self.stage!r}")
        expected_index = self.stage in _INDEXED_STAGES
        if (expected_index and _INDEX_PATTERN.fullmatch(self.item) is None) or (
            not expected_index and self.item != "main"
        ):
            raise ValueError(
                f"item is malformed for fine task stage {self.stage!r}: {self.item!r}"
            )

    @property
    def encoded(self) -> str:
        return f"round-{self.round:03d}:{self.stage}:{self.item}"

    @property
    def log_name(self) -> str:
        return f"round-{self.round:03d}-{self.stage}-{self.item}.log"


class _FineTask:
    round_task: RoundTask
    key: FineTaskKey
    dependencies: tuple[FineTaskKey, ...]
    output_paths: tuple[Path, ...]
    log_path: Path

    @property
    def round(self) -> int:
        return self.round_task.round

    @property
    def dataset_id(self) -> str:
        return self.round_task.dataset_id

    @property
    def strategy(self) -> RegistrationStrategy:
        return self.round_task.strategy

    @property
    def native_threads(self) -> NativeThreadSpec:
        return self.round_task.native_threads

    @property
    def registration_directory(self) -> Path:
        return self.round_task.registration_directory

    @property
    def channels(self) -> tuple[ChannelTask, ...]:
        return self.round_task.channels


def _validate_task(
    task: _FineTask,
    *,
    stage: FineTaskStage,
    item: str,
    strategy: Literal["whole_image", "serial_chunked"] | None = None,
) -> None:
    from cyexm.workflow.models import RoundTask

    if type(task.round_task) is not RoundTask:
        raise TypeError("round_task must be an exact RoundTask")
    expected = FineTaskKey(round=task.round_task.round, stage=stage, item=item)
    if task.key.round != task.round_task.round:
        raise ValueError("task key and embedded round must match")
    if task.key != expected:
        raise ValueError(f"task key must use the {stage!r} stage and canonical item")
    if strategy is not None and task.round_task.strategy != strategy:
        label = "whole-image" if strategy == "whole_image" else "serial-chunked"
        raise ValueError(f"task requires a {label} round")
    if len(set(task.dependencies)) != len(task.dependencies):
        raise ValueError("task dependencies must be unique")
    if any(key.round != task.round for key in task.dependencies):
        raise ValueError("task dependencies must belong to the embedded round")
    paths = tuple(Path(path) for path in task.output_paths)
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("task output paths must be non-empty and unique")
    expected_log = task.round_task.log_path.parent / task.key.log_name
    if task.log_path != expected_log:
        raise ValueError(f"log_path must be the canonical task log: {expected_log}")
    if task.log_path == task.round_task.log_path:
        raise ValueError("fine tasks must not share the coarse round log")


def _prepared_paths(round_task: RoundTask) -> PreparedRoundPaths:
    from cyexm.registration.prepared import PreparedRoundPaths

    return PreparedRoundPaths.for_round(
        round_task.registration_directory, round_task.round
    )


def _registration_paths(round_task: RoundTask) -> RegistrationArtifactPaths:
    from cyexm.workflow._round_provenance import RegistrationArtifactPaths

    return RegistrationArtifactPaths.for_round(
        round_task.registration_directory, round_task.round
    )


@dataclass(frozen=True, slots=True)
class RoundPreparationTask(_FineTask):
    """Prepare affine state and persist one fixed-grid nonlinear bundle."""

    round_task: RoundTask
    key: FineTaskKey
    dependencies: tuple[FineTaskKey, ...]
    output_paths: tuple[Path, ...]
    log_path: Path

    def __post_init__(self) -> None:
        _validate_task(
            self,
            stage="preparation",
            item="main",
            strategy="serial_chunked",
        )

    @classmethod
    def create(cls, round_task: RoundTask) -> Self:
        key = FineTaskKey(round_task.round, "preparation", "main")
        prepared = _prepared_paths(round_task)
        registration = _registration_paths(round_task)
        return cls(
            round_task=round_task,
            key=key,
            dependencies=(),
            output_paths=(
                registration.affine,
                registration.pre_metrics,
                registration.linear_metrics,
                *((registration.moving_crop,) if round_task.registration.crop else ()),
                *prepared.array_paths,
                prepared.manifest,
            ),
            log_path=round_task.log_path.parent / key.log_name,
        )


@dataclass(frozen=True, slots=True)
class NonlinearChunkTask(_FineTask):
    """Estimate and persist one canonical nonlinear chunk."""

    round_task: RoundTask
    key: FineTaskKey
    chunk: ChunkInfo
    prepared_descriptor_json: str
    dependencies: tuple[FineTaskKey, ...]
    output_paths: tuple[Path, ...]
    log_path: Path

    def __post_init__(self) -> None:
        if type(self.chunk) is not ChunkInfo:
            raise TypeError("chunk must be an exact ChunkInfo")
        _validate_task(
            self,
            stage="chunk",
            item=f"{self.chunk.chunk_id:04d}",
            strategy="serial_chunked",
        )
        _json_object(json.loads(self.prepared_descriptor_json), "prepared descriptor")

    @property
    def prepared_descriptor(self) -> dict[str, object]:
        return _json_object(
            json.loads(self.prepared_descriptor_json), "prepared descriptor"
        )

    @classmethod
    def create(
        cls,
        round_task: RoundTask,
        chunk: ChunkInfo,
        *,
        prepared_descriptor: object,
        dependencies: tuple[FineTaskKey, ...],
    ) -> Self:
        key = FineTaskKey(round_task.round, "chunk", f"{chunk.chunk_id:04d}")
        chunk_root = (
            round_task.registration_directory / f"round_{round_task.round}_chunks"
        )
        descriptor = _json_object(prepared_descriptor, "prepared descriptor")
        return cls(
            round_task=round_task,
            key=key,
            chunk=chunk,
            prepared_descriptor_json=json.dumps(
                descriptor,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            dependencies=dependencies,
            output_paths=(
                chunk_root / f"chunk_{chunk.chunk_id:04d}_warp.nii.gz",
                chunk_root / f"chunk_{chunk.chunk_id:04d}_registered_core.npy",
            ),
            log_path=round_task.log_path.parent / key.log_name,
        )


@dataclass(frozen=True, slots=True)
class ChunkAssemblyTask(_FineTask):
    """Assemble every accepted chunk in canonical ascending-ID order."""

    round_task: RoundTask
    key: FineTaskKey
    prepared_descriptor_json: str
    dependencies: tuple[FineTaskKey, ...]
    output_paths: tuple[Path, ...]
    log_path: Path

    def __post_init__(self) -> None:
        _validate_task(
            self,
            stage="assembly",
            item="main",
            strategy="serial_chunked",
        )
        if not self.dependencies or any(
            key.stage != "chunk" for key in self.dependencies
        ):
            raise ValueError("assembly dependencies must be nonlinear chunk keys")
        _json_object(json.loads(self.prepared_descriptor_json), "prepared descriptor")

    @property
    def prepared_descriptor(self) -> dict[str, object]:
        return _json_object(
            json.loads(self.prepared_descriptor_json), "prepared descriptor"
        )

    @classmethod
    def create(
        cls,
        round_task: RoundTask,
        *,
        prepared_descriptor: object,
        dependencies: tuple[FineTaskKey, ...],
    ) -> Self:
        key = FineTaskKey(round_task.round, "assembly", "main")
        paths = _registration_paths(round_task)
        chunk_manifest = (
            round_task.registration_directory
            / f"round_{round_task.round}_chunks"
            / "manifest.json"
        )
        descriptor = _json_object(prepared_descriptor, "prepared descriptor")
        return cls(
            round_task=round_task,
            key=key,
            prepared_descriptor_json=json.dumps(
                descriptor,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            dependencies=dependencies,
            output_paths=(
                paths.warp,
                paths.registered,
                paths.nonlinear_metrics,
                chunk_manifest,
            ),
            log_path=round_task.log_path.parent / key.log_name,
        )


@dataclass(frozen=True, slots=True)
class WholeImageRegistrationTask(_FineTask):
    """Run the accepted whole-image registration without channel transforms."""

    round_task: RoundTask
    key: FineTaskKey
    dependencies: tuple[FineTaskKey, ...]
    output_paths: tuple[Path, ...]
    log_path: Path

    def __post_init__(self) -> None:
        _validate_task(
            self,
            stage="whole-image",
            item="main",
            strategy="whole_image",
        )

    @classmethod
    def create(cls, round_task: RoundTask) -> Self:
        key = FineTaskKey(round_task.round, "whole-image", "main")
        paths = _registration_paths(round_task)
        return cls(
            round_task=round_task,
            key=key,
            dependencies=(),
            output_paths=(
                paths.affine,
                paths.warp,
                paths.registered,
                paths.pre_metrics,
                paths.linear_metrics,
                paths.nonlinear_metrics,
                *((paths.moving_crop,) if round_task.registration.crop else ()),
            ),
            log_path=round_task.log_path.parent / key.log_name,
        )


@dataclass(frozen=True, slots=True)
class ChannelTransformTask(_FineTask):
    """Transform one declared channel through an accepted transform chain."""

    round_task: RoundTask
    key: FineTaskKey
    channel_index: int
    dependencies: tuple[FineTaskKey, ...]
    output_paths: tuple[Path, ...]
    log_path: Path

    def __post_init__(self) -> None:
        if (
            isinstance(self.channel_index, bool)
            or not isinstance(self.channel_index, int)
            or self.channel_index < 0
            or self.channel_index >= len(self.round_task.channels)
        ):
            raise ValueError("channel_index must identify one declared channel")
        _validate_task(
            self,
            stage="channel",
            item=f"{self.channel_index:04d}",
        )
        if len(self.output_paths) != 1 or (
            self.output_paths[0] != self.round_task.channels[self.channel_index].output
        ):
            raise ValueError("channel task output must match its declared channel")

    @classmethod
    def create(
        cls,
        round_task: RoundTask,
        channel_index: int,
        *,
        dependencies: tuple[FineTaskKey, ...],
    ) -> Self:
        if isinstance(channel_index, bool) or not isinstance(channel_index, int):
            raise ValueError("channel_index must identify one declared channel")
        if channel_index < 0 or channel_index >= len(round_task.channels):
            raise ValueError("channel_index must identify one declared channel")
        key = FineTaskKey(round_task.round, "channel", f"{channel_index:04d}")
        return cls(
            round_task=round_task,
            key=key,
            channel_index=channel_index,
            dependencies=dependencies,
            output_paths=(round_task.channels[channel_index].output,),
            log_path=round_task.log_path.parent / key.log_name,
        )


@dataclass(frozen=True, slots=True)
class RoundValidationTask(_FineTask):
    """Validate the complete public artifact chain for one moving round."""

    round_task: RoundTask
    key: FineTaskKey
    dependencies: tuple[FineTaskKey, ...]
    output_paths: tuple[Path, ...]
    log_path: Path

    def __post_init__(self) -> None:
        _validate_task(self, stage="validation", item="main")
        if not self.dependencies or any(
            key.stage != "channel" for key in self.dependencies
        ):
            raise ValueError("validation dependencies must be channel keys")

    @classmethod
    def create(
        cls,
        round_task: RoundTask,
        *,
        dependencies: tuple[FineTaskKey, ...],
    ) -> Self:
        key = FineTaskKey(round_task.round, "validation", "main")
        paths = _registration_paths(round_task)
        return cls(
            round_task=round_task,
            key=key,
            dependencies=dependencies,
            output_paths=(
                *paths.public_registration_artifacts,
                *(channel.output for channel in round_task.channels),
            ),
            log_path=round_task.log_path.parent / key.log_name,
        )


type FineWorkflowTask = (
    RoundPreparationTask
    | NonlinearChunkTask
    | ChunkAssemblyTask
    | WholeImageRegistrationTask
    | ChannelTransformTask
    | RoundValidationTask
)


@dataclass(frozen=True, slots=True)
class FineTaskResult:
    """Terminal operational evidence for one exact fine-grained task."""

    task: FineWorkflowTask
    status: FineTaskStatus
    started_at: str
    ended_at: str
    duration_seconds: float
    hostname: str
    pid: int
    dask_task_key: str | None
    slurm_job_id: str | None
    native_threads: tuple[tuple[str, int], ...]
    artifact_paths: tuple[str, ...]
    reused_artifacts: tuple[str, ...]
    log_path: str
    failure: RoundFailure | None
    prepared_descriptor: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if type(self.task) not in (
            RoundPreparationTask,
            NonlinearChunkTask,
            ChunkAssemblyTask,
            WholeImageRegistrationTask,
            ChannelTransformTask,
            RoundValidationTask,
        ):
            raise TypeError("task must be an exact fine workflow task")
        if self.status not in {"reused", "completed", "failed", "cancelled"}:
            raise ValueError(f"unsupported fine task status: {self.status!r}")
        if not math.isfinite(self.duration_seconds) or self.duration_seconds < 0:
            raise ValueError("duration_seconds must be finite and non-negative")
        if isinstance(self.pid, bool) or not isinstance(self.pid, int) or self.pid < 1:
            raise ValueError("pid must be a positive integer")
        if self.log_path != str(self.task.log_path):
            raise ValueError("log_path must match the exact task log")
        if len(set(self.artifact_paths)) != len(self.artifact_paths):
            raise ValueError("artifact paths must be unique")
        if not set(self.reused_artifacts).issubset(self.artifact_paths):
            raise ValueError("reused artifacts must be completed artifacts")
        if any(
            not isinstance(name, str)
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
            for name, value in self.native_threads
        ):
            raise ValueError("native thread evidence must contain positive integers")
        if self.status == "failed" and self.failure is None:
            raise ValueError("failed fine task results require failure evidence")
        if self.status != "failed" and self.failure is not None:
            raise ValueError("only failed fine task results may contain a failure")
        if self.prepared_descriptor is not None:
            descriptor = _json_object(self.prepared_descriptor, "prepared descriptor")
            object.__setattr__(self, "prepared_descriptor", descriptor)
        if type(self.task) is RoundPreparationTask:
            if (
                self.status in {"completed", "reused"}
                and self.prepared_descriptor is None
            ):
                raise ValueError(
                    "successful preparation requires a prepared descriptor"
                )
        elif self.prepared_descriptor is not None:
            raise ValueError(
                "only preparation results may contain a prepared descriptor"
            )

    @property
    def key(self) -> FineTaskKey:
        return self.task.key

    @property
    def round(self) -> int:
        return self.task.round

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key.encoded,
            "round": self.round,
            "stage": self.key.stage,
            "item": self.key.item,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": self.duration_seconds,
            "hostname": self.hostname,
            "pid": self.pid,
            "dask_task_key": self.dask_task_key,
            "slurm_job_id": self.slurm_job_id,
            "native_threads": [list(item) for item in self.native_threads],
            "artifact_paths": list(self.artifact_paths),
            "reused_artifacts": list(self.reused_artifacts),
            "log_path": self.log_path,
            "failure": None if self.failure is None else self.failure.to_dict(),
            "prepared_descriptor": self.prepared_descriptor,
        }


type FineTask = FineWorkflowTask

__all__ = (
    "ChannelTransformTask",
    "ChunkAssemblyTask",
    "FineTask",
    "FineTaskKey",
    "FineTaskResult",
    "FineWorkflowTask",
    "NonlinearChunkTask",
    "RoundPreparationTask",
    "RoundValidationTask",
    "WholeImageRegistrationTask",
)
