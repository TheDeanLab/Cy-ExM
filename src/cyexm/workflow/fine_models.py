"""Immutable internal task and result values for fine-grained execution."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self, cast

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
_INDEX_PATTERN = re.compile(r"(?:[0-9]{4}|[1-9][0-9]{4,})")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_thaw_json(item) for item in value]
    return value


class _FrozenJsonObject(Mapping[str, object]):
    def __init__(self, value: Mapping[str, object]) -> None:
        self._values = {key: _freeze_json(item) for key, item in value.items()}

    def __getitem__(self, key: str) -> object:
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return _FrozenJsonObject(value)
    if isinstance(value, tuple | list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _json_object(value: object, name: str) -> dict[str, object]:
    try:
        encoded = json.dumps(
            _thaw_json(value),
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

    @classmethod
    def from_encoded(cls, value: str) -> Self:
        """Decode one exact canonical task identity."""
        if type(value) is not str:
            raise TypeError("encoded fine task key must be an exact string")
        try:
            round_label, stage, item = value.split(":", 2)
            if not round_label.startswith("round-"):
                raise ValueError
            round_number = int(round_label.removeprefix("round-"))
            key = cls(round_number, cast(FineTaskStage, stage), item)
        except (TypeError, ValueError) as error:
            raise ValueError(f"malformed fine task key: {value!r}") from error
        if key.encoded != value:
            raise ValueError(f"fine task key is not canonically encoded: {value!r}")
        return key

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


def _require_outputs(task: _FineTask, expected: tuple[Path, ...]) -> None:
    if task.output_paths != expected:
        raise ValueError("task output_paths must equal the exact canonical outputs")


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


def _preparation_outputs(round_task: RoundTask) -> tuple[Path, ...]:
    prepared = _prepared_paths(round_task)
    registration = _registration_paths(round_task)
    return (
        registration.affine,
        registration.pre_metrics,
        registration.linear_metrics,
        *((registration.moving_crop,) if round_task.registration.crop else ()),
        *prepared.array_paths,
        prepared.manifest,
    )


def _whole_outputs(round_task: RoundTask) -> tuple[Path, ...]:
    paths = _registration_paths(round_task)
    return (
        paths.affine,
        paths.warp,
        paths.registered,
        paths.pre_metrics,
        paths.linear_metrics,
        paths.nonlinear_metrics,
        *((paths.moving_crop,) if round_task.registration.crop else ()),
    )


def _validation_outputs(round_task: RoundTask) -> tuple[Path, ...]:
    paths = _registration_paths(round_task)
    return (
        *paths.public_registration_artifacts,
        *(channel.output for channel in round_task.channels),
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
        if self.dependencies:
            raise ValueError("preparation task must not have dependencies")
        _require_outputs(self, _preparation_outputs(self.round_task))

    @classmethod
    def create(cls, round_task: RoundTask) -> Self:
        key = FineTaskKey(round_task.round, "preparation", "main")
        return cls(
            round_task=round_task,
            key=key,
            dependencies=(),
            output_paths=_preparation_outputs(round_task),
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
        expected_dependency = FineTaskKey(self.round, "preparation", "main")
        if self.dependencies != (expected_dependency,):
            raise ValueError("chunk task must depend exactly on round preparation")
        chunk_root = (
            self.round_task.registration_directory / f"round_{self.round}_chunks"
        )
        _require_outputs(
            self,
            (
                chunk_root / f"chunk_{self.chunk.chunk_id:04d}_warp.nii.gz",
                chunk_root / f"chunk_{self.chunk.chunk_id:04d}_registered_core.npy",
            ),
        )
        descriptor = _json_object(
            json.loads(self.prepared_descriptor_json), "prepared descriptor"
        )
        from cyexm.chunking import compute_chunk_grid
        from cyexm.registration.prepared import PreparedManifest

        manifest = PreparedManifest.from_dict(descriptor)
        if (
            self.round_task.chunk_size_zyx is None
            or self.round_task.overlap_fraction is None
        ):
            raise ValueError("chunk task requires chunk geometry")
        chunks = compute_chunk_grid(
            manifest.geometry.shape,
            self.round_task.chunk_size_zyx,
            self.round_task.overlap_fraction,
        )
        if self.chunk.chunk_id >= len(chunks) or (
            chunks[self.chunk.chunk_id] != self.chunk
        ):
            raise ValueError("chunk must equal its canonical prepared-grid member")

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
        descriptor = _json_object(
            json.loads(self.prepared_descriptor_json), "prepared descriptor"
        )
        from cyexm.chunking import compute_chunk_grid
        from cyexm.registration.prepared import PreparedManifest

        manifest = PreparedManifest.from_dict(descriptor)
        if (
            self.round_task.chunk_size_zyx is None
            or self.round_task.overlap_fraction is None
        ):
            raise ValueError("assembly requires chunk geometry")
        chunks = compute_chunk_grid(
            manifest.geometry.shape,
            self.round_task.chunk_size_zyx,
            self.round_task.overlap_fraction,
        )
        expected = tuple(
            FineTaskKey(self.round, "chunk", f"{chunk.chunk_id:04d}")
            for chunk in chunks
        )
        if self.dependencies != expected:
            raise ValueError(
                "assembly dependencies must equal every canonical chunk key"
            )
        paths = _registration_paths(self.round_task)
        chunk_manifest = (
            self.round_task.registration_directory
            / f"round_{self.round}_chunks"
            / "manifest.json"
        )
        _require_outputs(
            self,
            (
                paths.warp,
                paths.registered,
                paths.nonlinear_metrics,
                chunk_manifest,
            ),
        )

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
        if self.dependencies:
            raise ValueError("whole-image task must not have dependencies")
        _require_outputs(self, _whole_outputs(self.round_task))

    @classmethod
    def create(cls, round_task: RoundTask) -> Self:
        key = FineTaskKey(round_task.round, "whole-image", "main")
        return cls(
            round_task=round_task,
            key=key,
            dependencies=(),
            output_paths=_whole_outputs(round_task),
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
        expected_stage = (
            "assembly"
            if self.round_task.strategy == "serial_chunked"
            else "whole-image"
        )
        expected = FineTaskKey(self.round, expected_stage, "main")
        if self.dependencies != (expected,):
            raise ValueError("channel task dependency does not match round strategy")

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
        expected = tuple(
            FineTaskKey(self.round, "channel", f"{index:04d}")
            for index in range(len(self.round_task.channels))
        )
        if self.dependencies != expected:
            raise ValueError(
                "validation dependencies must equal every declared channel"
            )
        _require_outputs(self, _validation_outputs(self.round_task))

    @classmethod
    def create(
        cls,
        round_task: RoundTask,
        *,
        dependencies: tuple[FineTaskKey, ...],
    ) -> Self:
        key = FineTaskKey(round_task.round, "validation", "main")
        return cls(
            round_task=round_task,
            key=key,
            dependencies=dependencies,
            output_paths=_validation_outputs(round_task),
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

_FINE_TASK_RESULT_KEYS = frozenset(
    {
        "key",
        "round",
        "stage",
        "item",
        "status",
        "started_at",
        "ended_at",
        "duration_seconds",
        "hostname",
        "pid",
        "dask_task_key",
        "slurm_job_id",
        "native_threads",
        "artifact_paths",
        "reused_artifacts",
        "log_path",
        "failure",
        "prepared_descriptor",
    }
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
    prepared_descriptor: Mapping[str, object] | None = None

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
            object.__setattr__(
                self, "prepared_descriptor", _FrozenJsonObject(descriptor)
            )
        planned = tuple(str(path) for path in self.task.output_paths)
        if self.status in {"completed", "reused"} and self.artifact_paths != planned:
            raise ValueError(
                "successful artifact_paths must equal exact planned outputs"
            )
        if self.status in {"failed", "cancelled"} and not set(
            self.artifact_paths
        ).issubset(planned):
            raise ValueError("partial artifact_paths must be planned outputs")
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
            "prepared_descriptor": (
                None
                if self.prepared_descriptor is None
                else _json_object(self.prepared_descriptor, "prepared descriptor")
            ),
        }

    @classmethod
    def from_dict(
        cls,
        task: FineWorkflowTask,
        payload: object,
    ) -> Self:
        """Decode the exact JSON representation for one known immutable task."""
        if type(payload) is not dict or any(type(key) is not str for key in payload):
            raise TypeError("fine task result must be an exact JSON object")
        raw = cast(dict[str, object], payload)
        if frozenset(raw) != _FINE_TASK_RESULT_KEYS:
            raise ValueError(
                "fine task result keys must be exactly "
                f"{sorted(_FINE_TASK_RESULT_KEYS)!r}"
            )
        if type(raw["round"]) is not int:
            raise TypeError("fine task result round must be an exact integer")
        for name in ("key", "stage", "item"):
            if type(raw[name]) is not str:
                raise TypeError(f"fine task result {name} must be an exact string")
        if (
            raw["key"] != task.key.encoded
            or raw["round"] != task.round
            or raw["stage"] != task.key.stage
            or raw["item"] != task.key.item
        ):
            raise ValueError("fine task result identity must match the exact task")
        for name in ("status", "started_at", "ended_at", "hostname", "log_path"):
            if type(raw[name]) is not str:
                raise TypeError(f"fine task result {name} must be an exact string")
        if type(raw["pid"]) is not int:
            raise TypeError("fine task result pid must be an exact integer")
        if type(raw["duration_seconds"]) is not float:
            raise TypeError("fine task result duration_seconds must be an exact float")
        for name in ("dask_task_key", "slurm_job_id"):
            if raw[name] is not None and type(raw[name]) is not str:
                raise TypeError(
                    f"fine task result {name} must be null or an exact string"
                )

        def decode_strings(name: str) -> tuple[str, ...]:
            value = raw[name]
            if type(value) is not list or any(type(item) is not str for item in value):
                raise TypeError(
                    f"fine task result {name} must be an exact string array"
                )
            return tuple(cast(list[str], value))

        native_raw = raw["native_threads"]
        if type(native_raw) is not list:
            raise TypeError("fine task result native_threads must be an exact array")
        native_threads: list[tuple[str, int]] = []
        for pair in cast(list[object], native_raw):
            if (
                type(pair) is not list
                or len(pair) != 2
                or type(pair[0]) is not str
                or type(pair[1]) is not int
            ):
                raise TypeError(
                    "fine task result native_threads entries must be exact pairs"
                )
            native_threads.append((cast(str, pair[0]), cast(int, pair[1])))

        from cyexm.workflow.models import RoundFailure

        failure_raw = raw["failure"]
        failure: RoundFailure | None
        if failure_raw is None:
            failure = None
        else:
            if type(failure_raw) is not dict or frozenset(failure_raw) != frozenset(
                {"stage", "exception_type", "message"}
            ):
                raise ValueError(
                    "fine task result failure must use the exact failure schema"
                )
            failure_values = cast(dict[str, object], failure_raw)
            if any(
                type(failure_values[name]) is not str
                for name in ("stage", "exception_type", "message")
            ):
                raise TypeError("fine task result failure values must be exact strings")
            failure = RoundFailure(
                stage=cast(str, failure_values["stage"]),
                exception_type=cast(str, failure_values["exception_type"]),
                message=cast(str, failure_values["message"]),
            )
        descriptor = raw["prepared_descriptor"]
        if descriptor is not None and type(descriptor) is not dict:
            raise TypeError(
                "fine task result prepared_descriptor must be null or an exact object"
            )
        return cls(
            task=task,
            status=cast(FineTaskStatus, raw["status"]),
            started_at=cast(str, raw["started_at"]),
            ended_at=cast(str, raw["ended_at"]),
            duration_seconds=cast(float, raw["duration_seconds"]),
            hostname=cast(str, raw["hostname"]),
            pid=cast(int, raw["pid"]),
            dask_task_key=cast(str | None, raw["dask_task_key"]),
            slurm_job_id=cast(str | None, raw["slurm_job_id"]),
            native_threads=tuple(native_threads),
            artifact_paths=decode_strings("artifact_paths"),
            reused_artifacts=decode_strings("reused_artifacts"),
            log_path=cast(str, raw["log_path"]),
            failure=failure,
            prepared_descriptor=cast(dict[str, object] | None, descriptor),
        )


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
