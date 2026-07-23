"""Lightweight validation shared by workflow APIs and execution adapters."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from cyexm.config.models import (
    NativeThreadSpec,
    NonlinearRegistrationStageSpec,
    RegistrationSpec,
    RegistrationStageSpec,
)
from cyexm.errors import RoundExecutionError, WorkflowPlanError
from cyexm.registration.prepared import PreparedManifest
from cyexm.runtime import NATIVE_THREAD_ENV_VARS
from cyexm.workflow.fine_models import (
    ChannelTransformTask,
    ChunkAssemblyTask,
    FineTaskResult,
    FineWorkflowTask,
    NonlinearChunkTask,
    RoundPreparationTask,
    RoundValidationTask,
    WholeImageRegistrationTask,
)
from cyexm.workflow.models import (
    ChannelTask,
    ReferenceChannelTask,
    ReferenceTask,
    RoundFailure,
    RoundResult,
    RoundTask,
    WorkflowTask,
)


@dataclass(frozen=True, slots=True)
class _FineExecutionEvidence:
    """Detached scheduler-neutral evidence for one fine graph execution."""

    executor: str
    worker_cap: int
    tasks: tuple[FineWorkflowTask, ...]
    task_graph: tuple[tuple[str, tuple[str, ...]], ...]
    task_results: tuple[FineTaskResult, ...]
    attempt_counts: tuple[tuple[str, int], ...]
    blocked_descendants: tuple[str, ...]
    cancelled: bool
    cleanup: tuple[tuple[str, bool, str | None, str | None], ...]


def _canonical_path(path: object, location: str) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise WorkflowPlanError(
            f"{location} must be an absolute pathlib.Path; got {path!r}"
        )
    try:
        resolved = path.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise WorkflowPlanError(
            f"{location} cannot be resolved safely; got {path}"
        ) from error
    if resolved != path:
        raise WorkflowPlanError(
            f"{location} must be canonical; got {path}, resolved {resolved}"
        )
    return path


def _task_path(
    path: object,
    location: str,
    error_type: type[RoundExecutionError] | type[WorkflowPlanError],
) -> None:
    if not isinstance(path, Path) or not path.is_absolute():
        raise error_type(f"{location} must be an absolute pathlib.Path; got {path!r}")
    try:
        resolved = path.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise error_type(f"{location} cannot be resolved safely; got {path}") from error
    if resolved != path:
        raise error_type(
            f"{location} must be canonical; got {path}, resolved {resolved}"
        )


def _validate_round_task_value(
    task: object,
    *,
    location: str,
    error_type: type[RoundExecutionError] | type[WorkflowPlanError],
) -> None:
    if type(task) is not RoundTask:
        raise error_type(f"{location} must be a RoundTask; got {task!r}")
    if type(task.dataset_id) is not str or not task.dataset_id:
        raise error_type(f"{location}.dataset_id must be a nonempty built-in string")
    if type(task.round) is not int or task.round < 0:
        raise error_type(f"{location}.round must be a nonnegative built-in integer")
    if type(task.strategy) is not str or task.strategy not in {
        "whole_image",
        "serial_chunked",
    }:
        raise error_type(
            f"{location}.strategy must be an exact supported built-in string"
        )
    for field_name, path in (
        ("fixed_image", task.fixed_image),
        ("moving_image", task.moving_image),
        ("registration_directory", task.registration_directory),
        ("log_path", task.log_path),
    ):
        _task_path(path, f"{location}.{field_name}", error_type)

    spacing = task.voxel_spacing_xyz
    if (
        type(spacing) is not tuple
        or len(spacing) != 3
        or any(
            type(value) is not float or not math.isfinite(value) or value <= 0.0
            for value in spacing
        )
    ):
        raise error_type(
            f"{location}.voxel_spacing_xyz must be an exact tuple of three "
            "positive finite built-in floats"
        )

    registration = task.registration
    if type(registration) is not RegistrationSpec:
        raise error_type(f"{location}.registration must be a RegistrationSpec")
    if type(registration.crop) is not bool:
        raise error_type(f"{location}.registration.crop must be a built-in boolean")
    for stage_name, stage, expected_type in (
        ("linear", registration.linear, RegistrationStageSpec),
        ("nonlinear", registration.nonlinear, NonlinearRegistrationStageSpec),
    ):
        stage_location = f"{location}.registration.{stage_name}"
        if type(stage) is not expected_type:
            raise error_type(f"{stage_location} must be a {expected_type.__name__}")
        if type(stage.type) is not str or not stage.type:
            raise error_type(
                f"{stage_location}.type must be a nonempty built-in string"
            )
        if type(stage.accuracy) is not str or not stage.accuracy:
            raise error_type(
                f"{stage_location}.accuracy must be a nonempty built-in string"
            )
    nonlinear = registration.nonlinear
    if type(nonlinear.metric) is not str or nonlinear.metric not in {"CC", "mattes"}:
        raise error_type(
            f"{location}.registration.nonlinear.metric must be exactly CC or mattes"
        )
    if nonlinear.metric == "CC":
        if type(nonlinear.radius) is not int or nonlinear.radius < 1:
            raise error_type(
                f"{location}.registration.nonlinear.radius must be a positive "
                "built-in integer for CC"
            )
    elif nonlinear.radius is not None:
        raise error_type(
            f"{location}.registration.nonlinear.radius must be null for mattes"
        )

    native_threads = task.native_threads
    if type(native_threads) is not NativeThreadSpec:
        raise error_type(f"{location}.native_threads must be a NativeThreadSpec")
    for field_name, value in (
        ("itk", native_threads.itk),
        ("blas", native_threads.blas),
    ):
        if type(value) is not int or value < 1:
            raise error_type(
                f"{location}.native_threads.{field_name} must be a positive "
                "built-in integer"
            )

    if type(task.channels) is not tuple or not task.channels:
        raise error_type(f"{location}.channels must be a nonempty exact tuple")
    for channel_index, channel in enumerate(task.channels):
        channel_location = f"{location}.channels[{channel_index}]"
        if type(channel) is not ChannelTask:
            raise error_type(f"{channel_location} must be a ChannelTask")
        _task_path(channel.source, f"{channel_location}.source", error_type)
        _task_path(channel.output, f"{channel_location}.output", error_type)
        if type(channel.interpolation) is not str or channel.interpolation not in {
            "linear",
            "nearestNeighbor",
        }:
            raise error_type(
                f"{channel_location}.interpolation must be an exact supported "
                "built-in string"
            )

    chunk_size = task.chunk_size_zyx
    if chunk_size is not None:
        if type(chunk_size) is not tuple or len(chunk_size) != 3:
            raise error_type(
                f"{location}.chunk_size_zyx must be null or an exact tuple of "
                "three positive built-in integers"
            )
        for chunk_index, value in enumerate(chunk_size):
            if type(value) is not int or value < 1:
                raise error_type(
                    f"{location}.chunk_size_zyx[{chunk_index}] must be a positive "
                    "built-in integer"
                )
    overlap = task.overlap_fraction
    if overlap is not None and (
        type(overlap) is not float
        or not math.isfinite(overlap)
        or not 0.0 <= overlap < 1.0
    ):
        raise error_type(
            f"{location}.overlap_fraction must be null or a finite built-in "
            "float from zero up to one"
        )
    if task.strategy == "whole_image" and (
        chunk_size is not None or overlap is not None
    ):
        raise error_type(
            f"{location}.chunk_size_zyx and {location}.overlap_fraction must be "
            "null for whole_image"
        )
    if task.strategy == "serial_chunked" and (chunk_size is None or overlap is None):
        raise error_type(
            f"{location}.chunk_size_zyx and {location}.overlap_fraction are "
            "required for serial_chunked"
        )
    if type(task.prepared_fixed) is not bool:
        raise error_type(f"{location}.prepared_fixed must be a built-in boolean")
    if type(task.force) is not bool:
        raise error_type(f"{location}.force must be a built-in boolean")


def _validate_round_task_structure(task: object, index: int) -> None:
    _validate_round_task_value(
        task,
        location=f"WorkflowPlan.tasks[{index}]",
        error_type=WorkflowPlanError,
    )


def _validate_executor_round_task(task: object, index: int) -> None:
    _validate_round_task_value(
        task,
        location=f"LocalDaskExecutor.tasks[{index}]",
        error_type=RoundExecutionError,
    )


def _validate_executor_reference_task(task: object) -> None:
    location = "DistributedExecutor.reference_task"
    if type(task) is not ReferenceTask:
        raise RoundExecutionError(f"{location} must be an exact ReferenceTask")
    if type(task.dataset_id) is not str or not task.dataset_id:
        raise RoundExecutionError(f"{location}.dataset_id must be nonempty")
    if type(task.round) is not int or task.round < 0:
        raise RoundExecutionError(f"{location}.round must be nonnegative")
    for field_name, path in (
        ("registration_channel_source", task.registration_channel_source),
        ("registration_channel_output", task.registration_channel_output),
        ("registration_directory", task.registration_directory),
        ("log_path", task.log_path),
    ):
        _task_path(path, f"{location}.{field_name}", RoundExecutionError)
    if task.crop_bounds_path is not None:
        _task_path(
            task.crop_bounds_path,
            f"{location}.crop_bounds_path",
            RoundExecutionError,
        )
    if type(task.crop) is not bool or task.crop != (task.crop_bounds_path is not None):
        raise RoundExecutionError(f"{location}.crop must match crop_bounds_path")
    if type(task.force) is not bool:
        raise RoundExecutionError(f"{location}.force must be a boolean")
    if type(task.native_threads) is not NativeThreadSpec:
        raise RoundExecutionError(f"{location}.native_threads must be exact")
    if task.native_threads.itk < 1 or task.native_threads.blas < 1:
        raise RoundExecutionError(f"{location}.native_threads must be positive")
    spacing = task.voxel_spacing_xyz
    if (
        type(spacing) is not tuple
        or len(spacing) != 3
        or any(
            type(value) is not float or not math.isfinite(value) or value <= 0.0
            for value in spacing
        )
    ):
        raise RoundExecutionError(f"{location}.voxel_spacing_xyz is invalid")
    if type(task.channels) is not tuple or not task.channels:
        raise RoundExecutionError(f"{location}.channels must be a nonempty tuple")
    sources: list[Path] = []
    outputs: list[Path] = []
    for index, channel in enumerate(task.channels):
        if type(channel) is not ReferenceChannelTask:
            raise RoundExecutionError(
                f"{location}.channels[{index}] must be exact ReferenceChannelTask"
            )
        _task_path(
            channel.source,
            f"{location}.channels[{index}].source",
            RoundExecutionError,
        )
        _task_path(
            channel.output,
            f"{location}.channels[{index}].output",
            RoundExecutionError,
        )
        sources.append(channel.source)
        outputs.append(channel.output)
    if len(set(sources)) != len(sources) or len(set(outputs)) != len(outputs):
        raise RoundExecutionError(f"{location} channel paths must be unique")
    if task.registration_channel_source not in sources:
        raise RoundExecutionError(f"{location} registration source is undeclared")
    if task.registration_channel_output not in outputs:
        raise RoundExecutionError(f"{location} registration output is undeclared")


def _planned_task_artifacts(task: WorkflowTask) -> tuple[str, ...]:
    if type(task) is ReferenceTask:
        return (
            *(() if task.crop_bounds_path is None else (str(task.crop_bounds_path),)),
            *(str(channel.output) for channel in task.channels),
        )
    return (
        str(task.registration_directory / f"linear_transform_{task.round}.mat"),
        str(task.registration_directory / f"nonlinear_warp_{task.round}.nii.gz"),
        str(task.registration_directory / f"nonlinear_registered_{task.round}.tif"),
        *(str(channel.output) for channel in task.channels),
    )


def _parse_utc_timestamp(value: object, location: str) -> datetime:
    if type(value) is not str or "T" not in value or not value.endswith("Z"):
        raise RoundExecutionError(
            f"{location} must be an ISO 8601 UTC timestamp ending in Z"
        )
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise RoundExecutionError(
            f"{location} must be an ISO 8601 UTC timestamp ending in Z"
        ) from error
    if parsed.utcoffset() != UTC.utcoffset(None):
        raise RoundExecutionError(
            f"{location} must be an ISO 8601 UTC timestamp ending in Z"
        )
    return parsed


def _validate_round_result_structure(result: object, location: str) -> None:
    if type(result) is not RoundResult:
        raise RoundExecutionError(f"{location} must be an exact RoundResult")
    for field_name, value in (
        ("dataset_id", result.dataset_id),
        ("status", result.status),
        ("started_at", result.started_at),
        ("ended_at", result.ended_at),
        ("hostname", result.hostname),
        ("strategy", result.strategy),
        ("log_path", result.log_path),
    ):
        if type(value) is not str or not value:
            raise RoundExecutionError(
                f"{location} {field_name} must be a nonempty built-in string"
            )
    if type(result.round) is not int or result.round < 0:
        raise RoundExecutionError(
            f"{location} round must be a nonnegative built-in integer"
        )
    if (
        type(result.duration_seconds) is not float
        or not math.isfinite(result.duration_seconds)
        or result.duration_seconds < 0.0
    ):
        raise RoundExecutionError(
            f"{location} duration_seconds must be a nonnegative finite built-in float"
        )
    if type(result.pid) is not int or result.pid < 1:
        raise RoundExecutionError(f"{location} pid must be a positive built-in integer")
    for field_name, value in (
        ("dask_task_key", result.dask_task_key),
        ("slurm_job_id", result.slurm_job_id),
    ):
        if value is not None and (type(value) is not str or not value):
            raise RoundExecutionError(
                f"{location} {field_name} must be null or a nonempty built-in string"
            )

    native_threads = result.native_threads
    if type(native_threads) is not tuple:
        raise RoundExecutionError(f"{location} native_threads must be an exact tuple")
    allowed_names = frozenset({"itk", "blas", *NATIVE_THREAD_ENV_VARS})
    names: list[str] = []
    for index, entry in enumerate(native_threads):
        entry_location = f"{location} native_threads[{index}]"
        if type(entry) is not tuple or len(entry) != 2:
            raise RoundExecutionError(
                f"{entry_location} must be an exact (name, count) tuple"
            )
        name, count = entry
        if type(name) is not str or name not in allowed_names:
            raise RoundExecutionError(
                f"{entry_location} name must be an exact expected built-in string"
            )
        if type(count) is not int or count < 1:
            raise RoundExecutionError(
                f"{entry_location} count must be a positive built-in integer"
            )
        names.append(name)
    if len(set(names)) != len(names):
        raise RoundExecutionError(f"{location} native_threads names must be unique")

    for field_name, values in (
        ("artifact_paths", result.artifact_paths),
        ("reused_artifacts", result.reused_artifacts),
    ):
        if type(values) is not tuple:
            raise RoundExecutionError(f"{location} {field_name} must be an exact tuple")
        if any(type(value) is not str for value in values):
            raise RoundExecutionError(
                f"{location} {field_name} entries must be built-in strings"
            )

    failure = result.failure
    if failure is not None:
        if type(failure) is not RoundFailure:
            raise RoundExecutionError(
                f"{location} failure must be null or an exact RoundFailure"
            )
        for field_name, value in (
            ("stage", failure.stage),
            ("exception_type", failure.exception_type),
            ("message", failure.message),
        ):
            if type(value) is not str or (field_name != "message" and not value):
                raise RoundExecutionError(
                    f"{location} failure.{field_name} must be a built-in string"
                )


def _validate_round_result(
    task: WorkflowTask,
    result: RoundResult,
    *,
    fine_mode: bool = False,
) -> None:
    location = f"Executor result for round {task.round}"
    _validate_round_result_structure(result, location)
    if result.dataset_id != task.dataset_id:
        raise RoundExecutionError(f"{location} dataset_id does not match the task")
    if type(result.round) is not int or result.round != task.round:
        raise RoundExecutionError(f"{location} round does not match the task")
    expected_strategy = (
        "reference-preparation" if isinstance(task, ReferenceTask) else task.strategy
    )
    if result.strategy != expected_strategy:
        raise RoundExecutionError(f"{location} strategy does not match the task")
    if result.status not in {"reused", "completed", "failed", "cancelled"}:
        raise RoundExecutionError(f"{location} has unknown status {result.status!r}")
    if result.status == "failed" and result.failure is None:
        raise RoundExecutionError(f"{location} failure is required for failed status")
    if result.status != "failed" and result.failure is not None:
        raise RoundExecutionError(
            f"{location} failure must be absent unless status is failed"
        )
    compact_threads = (
        ("itk", task.native_threads.itk),
        ("blas", task.native_threads.blas),
    )
    expanded_threads = tuple(
        (
            name,
            task.native_threads.itk
            if name == NATIVE_THREAD_ENV_VARS[0]
            else task.native_threads.blas,
        )
        for name in NATIVE_THREAD_ENV_VARS
    )
    if (
        result.native_threads != compact_threads
        and result.native_threads != expanded_threads
    ):
        raise RoundExecutionError(f"{location} native_threads do not match the task")
    started_at = _parse_utc_timestamp(result.started_at, f"{location} started_at")
    ended_at = _parse_utc_timestamp(result.ended_at, f"{location} ended_at")
    if ended_at < started_at:
        raise RoundExecutionError(f"{location} ended_at must not precede started_at")
    expected_log_path = task.log_path
    if fine_mode and type(task) is RoundTask:
        from cyexm.workflow.fine_models import FineTaskKey

        if result.status in {"completed", "reused"}:
            expected_key = FineTaskKey(task.round, "validation", "main")
        else:
            name = Path(result.log_path).name
            prefix = f"round-{task.round:03d}-"
            suffix = ".log"
            if not name.startswith(prefix) or not name.endswith(suffix):
                raise RoundExecutionError(
                    f"{location} log_path is not a canonical fine task log"
                )
            encoded_item = name[len(prefix) : -len(suffix)]
            stage: str
            item: str
            if encoded_item.startswith("whole-image-"):
                stage, item = "whole-image", encoded_item.removeprefix("whole-image-")
            else:
                try:
                    stage, item = encoded_item.rsplit("-", 1)
                except ValueError as error:
                    raise RoundExecutionError(
                        f"{location} log_path is not a canonical fine task log"
                    ) from error
            try:
                expected_key = FineTaskKey.from_encoded(
                    f"round-{task.round:03d}:{stage}:{item}"
                )
            except (TypeError, ValueError) as error:
                raise RoundExecutionError(
                    f"{location} log_path is not a canonical fine task log"
                ) from error
            if (
                result.status == "failed"
                and result.failure is not None
                and result.failure.stage != expected_key.stage
            ):
                raise RoundExecutionError(
                    f"{location} log_path stage does not match failure evidence"
                )
        expected_log_path = task.log_path.parent / expected_key.log_name
    if result.log_path != str(expected_log_path):
        raise RoundExecutionError(f"{location} log_path does not match the task")

    planned = _planned_task_artifacts(task)
    if len(set(result.artifact_paths)) != len(result.artifact_paths) or any(
        path not in planned for path in result.artifact_paths
    ):
        raise RoundExecutionError(
            f"{location} artifact_paths must be unique planned task artifacts"
        )
    if result.status in {"completed", "reused"} and result.artifact_paths != planned:
        raise RoundExecutionError(
            f"{location} artifact_paths must contain every planned artifact"
        )
    if len(set(result.reused_artifacts)) != len(result.reused_artifacts) or any(
        path not in result.artifact_paths for path in result.reused_artifacts
    ):
        raise RoundExecutionError(
            f"{location} reused_artifacts must be a unique artifact_paths subset"
        )
    if result.status == "reused" and (result.reused_artifacts != result.artifact_paths):
        raise RoundExecutionError(
            f"{location} reused status must identify every artifact as reused"
        )


def _parse_aware_timestamp(value: object, location: str) -> datetime:
    if type(value) is not str or "T" not in value:
        raise RoundExecutionError(
            f"{location} must be a timezone-aware ISO 8601 timestamp"
        )
    encoded = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(encoded)
    except ValueError as error:
        raise RoundExecutionError(
            f"{location} must be a timezone-aware ISO 8601 timestamp"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RoundExecutionError(
            f"{location} must be a timezone-aware ISO 8601 timestamp"
        )
    return parsed


def _validate_fine_task_result(
    task: FineWorkflowTask,
    result: object,
) -> FineTaskResult:
    """Validate scheduler-neutral evidence for one exact fine task result."""
    location = f"Fine executor result for {task.key.encoded}"
    if type(result) is not FineTaskResult:
        raise RoundExecutionError(f"{location} must be an exact FineTaskResult")
    if type(result.task) is not type(task) or result.task != task:
        raise RoundExecutionError(f"{location} does not match the submitted task")
    try:
        FineTaskResult(
            task=result.task,
            status=result.status,
            started_at=result.started_at,
            ended_at=result.ended_at,
            duration_seconds=result.duration_seconds,
            hostname=result.hostname,
            pid=result.pid,
            dask_task_key=result.dask_task_key,
            slurm_job_id=result.slurm_job_id,
            native_threads=result.native_threads,
            artifact_paths=result.artifact_paths,
            reused_artifacts=result.reused_artifacts,
            log_path=result.log_path,
            failure=result.failure,
            prepared_descriptor=result.prepared_descriptor,
        )
    except (TypeError, ValueError) as error:
        raise RoundExecutionError(
            f"{location} violates the exact FineTaskResult structure: {error}"
        ) from error
    if type(result.status) is not str or result.status not in {
        "reused",
        "completed",
        "failed",
        "cancelled",
    }:
        raise RoundExecutionError(f"{location} has an unsupported status")
    if type(result.hostname) is not str or not result.hostname:
        raise RoundExecutionError(f"{location} hostname must be nonempty")
    if type(result.pid) is not int or result.pid < 1:
        raise RoundExecutionError(f"{location} pid must be a positive integer")
    if (
        type(result.duration_seconds) is not float
        or not math.isfinite(result.duration_seconds)
        or result.duration_seconds < 0.0
    ):
        raise RoundExecutionError(
            f"{location} duration_seconds must be finite and non-negative"
        )
    for field_name, value in (
        ("dask_task_key", result.dask_task_key),
        ("slurm_job_id", result.slurm_job_id),
    ):
        if value is not None and (type(value) is not str or not value):
            raise RoundExecutionError(
                f"{location} {field_name} must be null or nonempty"
            )

    expected_threads = tuple(
        (
            name,
            (
                task.native_threads.itk
                if name == NATIVE_THREAD_ENV_VARS[0]
                else task.native_threads.blas
            ),
        )
        for name in NATIVE_THREAD_ENV_VARS
    )
    if result.native_threads != expected_threads:
        raise RoundExecutionError(
            f"{location} native_threads must exactly match configured expanded "
            "thread evidence"
        )

    started = _parse_aware_timestamp(result.started_at, f"{location} started_at")
    ended = _parse_aware_timestamp(result.ended_at, f"{location} ended_at")
    if ended < started:
        raise RoundExecutionError(f"{location} ended_at precedes started_at")
    if result.log_path != str(task.log_path):
        raise RoundExecutionError(f"{location} log_path does not match the task")

    planned = tuple(str(path) for path in task.output_paths)
    if type(result.artifact_paths) is not tuple or len(
        set(result.artifact_paths)
    ) != len(result.artifact_paths):
        raise RoundExecutionError(
            f"{location} artifact_paths must be a unique exact tuple"
        )
    if any(
        type(path) is not str or path not in planned for path in result.artifact_paths
    ):
        raise RoundExecutionError(
            f"{location} artifact_paths must contain only planned outputs"
        )
    if result.status in {"completed", "reused"} and result.artifact_paths != planned:
        raise RoundExecutionError(
            f"{location} successful artifact_paths must equal planned outputs"
        )
    if (
        type(result.reused_artifacts) is not tuple
        or len(set(result.reused_artifacts)) != len(result.reused_artifacts)
        or any(
            type(path) is not str or path not in result.artifact_paths
            for path in result.reused_artifacts
        )
    ):
        raise RoundExecutionError(
            f"{location} reused_artifacts must be a unique artifact subset"
        )
    if result.status == "reused" and result.reused_artifacts != planned:
        raise RoundExecutionError(
            f"{location} reused status must identify every output as reused"
        )

    failure = result.failure
    if result.status == "failed":
        if type(failure) is not RoundFailure:
            raise RoundExecutionError(f"{location} requires exact failure evidence")
        if failure.stage != task.key.stage:
            raise RoundExecutionError(
                f"{location} failure stage does not match the task"
            )
        if type(failure.exception_type) is not str or not failure.exception_type:
            raise RoundExecutionError(
                f"{location} failure exception_type must be nonempty"
            )
        if type(failure.message) is not str:
            raise RoundExecutionError(
                f"{location} failure message must be a built-in string"
            )
    elif failure is not None:
        raise RoundExecutionError(
            f"{location} failure must be absent unless status is failed"
        )

    if type(task) is RoundPreparationTask:
        if (
            result.status in {"completed", "reused"}
            and result.prepared_descriptor is None
        ):
            raise RoundExecutionError(
                f"{location} successful preparation requires its descriptor"
            )
        if result.status in {"completed", "reused"}:
            descriptor = result.to_dict()["prepared_descriptor"]
            if not isinstance(descriptor, Mapping):
                raise RoundExecutionError(
                    f"{location} prepared descriptor must be a JSON object"
                )
            try:
                PreparedManifest.from_dict(descriptor)
            except (TypeError, ValueError) as error:
                raise RoundExecutionError(
                    f"{location} prepared descriptor violates PreparedManifest: {error}"
                ) from error
    elif result.prepared_descriptor is not None:
        raise RoundExecutionError(
            f"{location} prepared descriptor belongs only to preparation"
        )
    return result


def _canonical_fine_execution_evidence(
    value: object,
    *,
    expected_executor: str,
    expected_round_tasks: tuple[RoundTask, ...],
    expected_round_results: tuple[RoundResult, ...],
    expected_worker_cap: int,
    task_retries: int,
) -> str:
    """Validate and canonically encode scheduler-neutral fine execution evidence."""
    from cyexm.execution.dispatcher import (
        canonical_fine_task_key,
        validate_fine_graph_description,
    )
    from cyexm.workflow.fine_planning import (
        channel_tasks,
        chunk_tasks,
        initial_fine_tasks,
    )
    from cyexm.workflow.models import canonical_json

    location = "fine execution evidence"
    if type(value) is not _FineExecutionEvidence:
        raise RoundExecutionError(f"{location} must be an exact _FineExecutionEvidence")
    if type(value.executor) is not str or value.executor != expected_executor:
        raise RoundExecutionError(
            f"{location} executor must equal {expected_executor!r}"
        )
    if (
        type(value.worker_cap) is not int
        or value.worker_cap != expected_worker_cap
        or value.worker_cap < 1
    ):
        raise RoundExecutionError(
            f"{location} worker_cap must equal the normalized profile"
        )
    if type(value.cancelled) is not bool:
        raise RoundExecutionError(f"{location} cancelled must be an exact boolean")
    exact_task_types = (
        RoundPreparationTask,
        NonlinearChunkTask,
        ChunkAssemblyTask,
        WholeImageRegistrationTask,
        ChannelTransformTask,
        RoundValidationTask,
    )
    if type(value.tasks) is not tuple or any(
        type(task) not in exact_task_types for task in value.tasks
    ):
        raise RoundExecutionError(f"{location} tasks must be exact fine task values")
    expected_by_round = {task.round: task for task in expected_round_tasks}
    task_by_key: dict[str, FineWorkflowTask] = {}
    for index, task in enumerate(value.tasks):
        if task.round_task != expected_by_round.get(task.round):
            raise RoundExecutionError(
                f"{location} tasks[{index}] is not bound to the exact submitted "
                "RoundTask"
            )
        if task.key.encoded in task_by_key:
            raise RoundExecutionError(f"{location} task keys must be unique")
        task_by_key[task.key.encoded] = task
    derived_graph = tuple(
        (
            task.key.encoded,
            tuple(dependency.encoded for dependency in task.dependencies),
        )
        for task in value.tasks
    )
    if value.task_graph != derived_graph:
        raise RoundExecutionError(
            f"{location} task_graph must exactly describe immutable tasks"
        )
    initial = initial_fine_tasks(expected_round_tasks)
    if tuple(task for task in value.tasks if not task.dependencies) != initial:
        raise RoundExecutionError(
            f"{location} initial tasks must match submitted moving rounds"
        )
    if type(value.task_graph) is not tuple or any(
        type(item) is not tuple
        or len(item) != 2
        or type(item[0]) is not str
        or type(item[1]) is not tuple
        or any(type(dependency) is not str for dependency in item[1])
        for item in value.task_graph
    ):
        raise RoundExecutionError(f"{location} task_graph has an invalid structure")
    if type(value.task_results) is not tuple:
        raise RoundExecutionError(f"{location} task_results must be an exact tuple")
    results_by_key: dict[str, FineTaskResult] = {}
    terminal_statuses: dict[str, str] = {}
    for index, result in enumerate(value.task_results):
        if type(result) is not FineTaskResult:
            raise RoundExecutionError(
                f"{location} task_results[{index}] must be an exact FineTaskResult"
            )
        validated = _validate_fine_task_result(result.task, result)
        if validated.task != task_by_key.get(validated.key.encoded):
            raise RoundExecutionError(
                f"{location} task_results[{index}] is not bound to the exact task"
            )
        key = validated.key.encoded
        if key in results_by_key:
            raise RoundExecutionError(f"{location} task result keys must be unique")
        results_by_key[key] = validated
        terminal_statuses[key] = validated.status
    for result in results_by_key.values():
        if result.status not in {"completed", "reused"}:
            continue
        if type(result.task) is RoundPreparationTask:
            successors = chunk_tasks(result)
        elif type(result.task) in {
            ChunkAssemblyTask,
            WholeImageRegistrationTask,
        }:
            successors = channel_tasks(result)
        else:
            continue
        actual_successors = tuple(
            task_by_key.get(task.key.encoded) for task in successors
        )
        if actual_successors != successors:
            raise RoundExecutionError(
                f"{location} successor tasks must match exact producer expansion"
            )

    summaries = {
        result.round: result
        for result in expected_round_results
        if result.round in {task.round for task in expected_round_tasks}
    }
    if len(summaries) != len(
        tuple(
            result
            for result in expected_round_results
            if result.round in {task.round for task in expected_round_tasks}
        )
    ):
        raise RoundExecutionError(f"{location} round summaries must be unique")
    if not value.cancelled and set(summaries) != {
        task.round for task in expected_round_tasks
    }:
        raise RoundExecutionError(
            f"{location} requires one summary per submitted moving round"
        )
    for round_number, summary in summaries.items():
        actual = tuple(
            result for result in results_by_key.values() if result.round == round_number
        )
        failed = tuple(result for result in actual if result.status == "failed")
        cancelled = tuple(result for result in actual if result.status == "cancelled")
        validation = next(
            (
                result
                for result in actual
                if result.key.stage == "validation"
                and result.status in {"completed", "reused"}
            ),
            None,
        )
        if failed:
            representative = min(
                failed,
                key=lambda result: canonical_fine_task_key(result.key.encoded),
            )
            expected_status = "failed"
        elif cancelled:
            representative = min(
                cancelled,
                key=lambda result: canonical_fine_task_key(result.key.encoded),
            )
            expected_status = "cancelled"
        elif validation is not None:
            representative = validation
            producer_reuse = {
                path
                for result in actual
                if result.key.stage != "validation"
                for path in result.reused_artifacts
            }
            expected_status = (
                "reused"
                if set(validation.artifact_paths).issubset(producer_reuse)
                else "completed"
            )
        else:
            if value.cancelled:
                continue
            raise RoundExecutionError(
                f"{location} successful round lacks validation task evidence"
            )
        if (
            summary.status != expected_status
            or summary.log_path != representative.log_path
            or summary.failure != representative.failure
            or summary.hostname != representative.hostname
            or summary.pid != representative.pid
            or summary.dask_task_key != representative.dask_task_key
            or summary.slurm_job_id != representative.slurm_job_id
        ):
            raise RoundExecutionError(
                f"{location} round summary disagrees with canonical task evidence"
            )

    try:
        task_keys, dependencies = validate_fine_graph_description(
            value.task_graph,
            expected_round_tasks,
            allow_partial=value.cancelled,
            terminal_statuses=terminal_statuses,
        )
    except (TypeError, ValueError) as error:
        raise RoundExecutionError(
            f"{location} task_graph is invalid: {error}"
        ) from error
    if set(results_by_key) - set(task_keys):
        raise RoundExecutionError(
            f"{location} task results must name materialized graph tasks"
        )
    if tuple(results_by_key) != tuple(
        key for key in task_keys if key in results_by_key
    ):
        raise RoundExecutionError(
            f"{location} task results must use canonical graph order"
        )
    for key, result in results_by_key.items():
        result_dependencies = tuple(
            dependency.encoded for dependency in result.task.dependencies
        )
        if dependencies[key] != result_dependencies:
            raise RoundExecutionError(
                f"{location} result task dependencies do not match task_graph"
            )

    if type(value.attempt_counts) is not tuple:
        raise RoundExecutionError(f"{location} attempt_counts must be an exact tuple")
    attempts: dict[str, int] = {}
    for index, item in enumerate(value.attempt_counts):
        if (
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not int
            or item[1] < 1
            or item[1] > task_retries + 1
        ):
            raise RoundExecutionError(f"{location} attempt_counts[{index}] is invalid")
        key, count = item
        if key in attempts or key not in dependencies:
            raise RoundExecutionError(
                f"{location} attempt keys must be unique graph task keys"
            )
        attempts[key] = count
    expected_attempt_order = tuple(key for key in task_keys if key in attempts)
    if tuple(attempts) != expected_attempt_order:
        raise RoundExecutionError(
            f"{location} attempt counts must use canonical graph order"
        )
    if not set(results_by_key) <= set(attempts):
        raise RoundExecutionError(
            f"{location} every task result requires attempt evidence"
        )
    if not value.cancelled and set(results_by_key) != set(attempts):
        raise RoundExecutionError(
            f"{location} settled execution attempts must have task results"
        )

    if type(value.blocked_descendants) is not tuple or any(
        type(key) is not str for key in value.blocked_descendants
    ):
        raise RoundExecutionError(
            f"{location} blocked_descendants must be an exact string tuple"
        )
    if value.blocked_descendants != tuple(
        sorted(value.blocked_descendants, key=canonical_fine_task_key)
    ):
        raise RoundExecutionError(
            f"{location} blocked descendants must use canonical order"
        )
    if (
        len(set(value.blocked_descendants)) != len(value.blocked_descendants)
        or not set(value.blocked_descendants) <= set(task_keys)
        or set(value.blocked_descendants) & set(attempts)
    ):
        raise RoundExecutionError(
            f"{location} blocked descendants must be unique unattempted graph tasks"
        )
    if not value.cancelled and set(value.blocked_descendants) != (
        set(task_keys) - set(attempts)
    ):
        raise RoundExecutionError(
            f"{location} blocked descendants must exactly explain unattempted tasks"
        )

    if type(value.cleanup) is not tuple:
        raise RoundExecutionError(f"{location} cleanup must be an exact tuple")
    cleanup: dict[str, dict[str, object]] = {}
    for index, item in enumerate(value.cleanup):
        if (
            type(item) is not tuple
            or len(item) != 4
            or type(item[0]) is not str
            or not item[0]
            or type(item[1]) is not bool
            or item[0] in cleanup
        ):
            raise RoundExecutionError(f"{location} cleanup[{index}] is invalid")
        name, succeeded, error_type, error_message = item
        if succeeded:
            if error_type is not None or error_message is not None:
                raise RoundExecutionError(
                    f"{location} successful cleanup cannot contain an error"
                )
            cleanup[name] = {"attempted": True, "succeeded": True}
        else:
            if (
                type(error_type) is not str
                or not error_type
                or type(error_message) is not str
            ):
                raise RoundExecutionError(
                    f"{location} failed cleanup requires exact error evidence"
                )
            cleanup[name] = {
                "attempted": True,
                "succeeded": False,
                "error": {
                    "exception_type": error_type,
                    "message": error_message,
                },
            }
    expected_cleanup = (
        ()
        if expected_executor == "sequential"
        else ("client", "cluster", "configuration")
    )
    if tuple(cleanup) != expected_cleanup:
        raise RoundExecutionError(
            f"{location} cleanup must match the executor-owned resources"
        )

    return canonical_json(
        {
            "schema_version": 1,
            "executor": value.executor,
            "mode": "fine_grained",
            "worker_cap": value.worker_cap,
            "task_graph": [
                {"key": key, "dependencies": list(task_dependencies)}
                for key, task_dependencies in value.task_graph
            ],
            "task_results": [
                results_by_key[key].to_dict()
                for key in task_keys
                if key in results_by_key
            ],
            "attempt_counts": [
                {"task_key": key, "attempts": attempts[key]}
                for key in task_keys
                if key in attempts
            ],
            "blocked_descendants": list(value.blocked_descendants),
            "cancelled": value.cancelled,
            "cleanup": cleanup,
        }
    )


__all__ = []
