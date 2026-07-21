"""Lightweight validation shared by workflow APIs and execution adapters."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

from cyexm.config.models import (
    NativeThreadSpec,
    RegistrationSpec,
    RegistrationStageSpec,
)
from cyexm.errors import RoundExecutionError, WorkflowPlanError
from cyexm.runtime import NATIVE_THREAD_ENV_VARS
from cyexm.workflow.models import (
    ChannelTask,
    ReferenceChannelTask,
    ReferenceTask,
    RoundFailure,
    RoundResult,
    RoundTask,
    WorkflowTask,
)


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
    for stage_name, stage in (
        ("linear", registration.linear),
        ("nonlinear", registration.nonlinear),
    ):
        stage_location = f"{location}.registration.{stage_name}"
        if type(stage) is not RegistrationStageSpec:
            raise error_type(f"{stage_location} must be a RegistrationStageSpec")
        if type(stage.type) is not str or not stage.type:
            raise error_type(
                f"{stage_location}.type must be a nonempty built-in string"
            )
        if type(stage.accuracy) is not str or not stage.accuracy:
            raise error_type(
                f"{stage_location}.accuracy must be a nonempty built-in string"
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


def _validate_round_result(task: WorkflowTask, result: RoundResult) -> None:
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
    if result.log_path != str(task.log_path):
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


__all__ = []
