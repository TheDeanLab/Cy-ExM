"""Execute one scheduler-neutral registration round."""

from __future__ import annotations

import math
import os
import socket
import sys
import time
import traceback
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from cyexm.registration.artifacts import CompletionRecord, FileIdentity, StageRequest
from cyexm.runtime import (
    NATIVE_THREAD_ENV_VARS,
    NativeThreadConfig,
    ResolvedNativeThreads,
    configure_native_threads,
)
from cyexm.spatial import AXIS_CONVENTION_VERSION, ImageGeometry
from cyexm.workflow._registration_policy import (
    CHANNEL_POSTPROCESSING_POLICY,
    OUTPUT_FORMAT_POLICY,
    linear_parameters,
    nonlinear_parameters,
    scalar_policy,
)
from cyexm.workflow.models import (
    RoundFailure,
    RoundResult,
    RoundTask,
    canonical_json,
)

type _ArtifactSnapshot = tuple[CompletionRecord, FileIdentity]

_CHUNK_KEYS = frozenset(
    {
        "chunk_id",
        "z_start",
        "z_end",
        "y_start",
        "y_end",
        "x_start",
        "x_end",
        "z_start_ext",
        "z_end_ext",
        "y_start_ext",
        "y_end_ext",
        "x_start_ext",
        "x_end_ext",
    }
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _runtime() -> tuple[Any, Any, Any, Any]:
    """Load registration operations only after native threads are configured."""
    from cyexm.registration.channels import transform_channel
    from cyexm.registration.chunked import ChunkedImageRegistration
    from cyexm.registration.whole_image import ImageRegistration, RegistrationPaths

    return (
        ImageRegistration,
        ChunkedImageRegistration,
        RegistrationPaths,
        transform_channel,
    )


def _dask_task_key() -> str | None:
    """Read a current task key without importing or initializing Dask."""
    distributed = sys.modules.get("distributed")
    if distributed is None:
        return None
    try:
        worker = distributed.get_worker()
        key = worker.get_current_task()
        if key is None:
            return None
        encoded = str(key)
    except Exception:
        return None
    return encoded or None


def _planned_artifacts(task: RoundTask) -> tuple[Path, ...]:
    return (
        task.registration_directory / f"linear_transform_{task.round}.mat",
        task.registration_directory / f"nonlinear_warp_{task.round}.nii.gz",
        task.registration_directory / f"nonlinear_registered_{task.round}.tif",
        *(channel.output for channel in task.channels),
    )


def _completion_snapshot(path: Path) -> _ArtifactSnapshot | None:
    record_path = path.with_name(f"{path.name}.complete.json")
    try:
        if not path.is_file() or not record_path.is_file():
            return None
        text = record_path.read_text(encoding="utf-8")
        record = CompletionRecord.read(record_path)
        identity = FileIdentity.from_path(path)
    except (OSError, RuntimeError, TypeError, UnicodeError, ValueError):
        return None
    if record.to_json() != text or record.artifact != identity:
        return None
    return record, identity


def _expected_native_threads(task: RoundTask) -> dict[str, int]:
    itk_name, *blas_names = NATIVE_THREAD_ENV_VARS
    return {
        itk_name: task.native_threads.itk,
        **{name: task.native_threads.blas for name in blas_names},
    }


def _registration_inputs(task: RoundTask) -> tuple[FileIdentity, FileIdentity] | None:
    try:
        return (
            FileIdentity.from_path(task.fixed_image),
            FileIdentity.from_path(task.moving_image),
        )
    except (OSError, RuntimeError):
        return None


def _common_parameters(task: RoundTask) -> dict[str, object]:
    return {
        "axis_convention_version": AXIS_CONVENTION_VERSION,
        "crop": task.registration.crop,
        "imaging_round": task.round,
        "pixel_size_xyz": list(task.voxel_spacing_xyz),
    }


def _mapping_geometry(value: object) -> ImageGeometry | None:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        return None
    try:
        return ImageGeometry.from_dict(dict(value))
    except (TypeError, ValueError):
        return None


def _same_json(left: object, right: object) -> bool:
    try:
        return canonical_json(left) == canonical_json(right)
    except (OverflowError, TypeError, ValueError):
        return False


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _request_basics_valid(
    request: object,
    *,
    inputs: tuple[FileIdentity, ...],
    task: RoundTask,
) -> bool:
    return (
        isinstance(request, StageRequest)
        and request.inputs == inputs
        and request.native_threads == _expected_native_threads(task)
        and request.geometry.spacing == task.voxel_spacing_xyz
    )


def _static_parameters_valid(
    parameters: Mapping[str, object],
    expected: Mapping[str, object],
    *,
    dynamic: frozenset[str] = frozenset(),
) -> bool:
    if frozenset(parameters) != frozenset({*expected, *dynamic}):
        return False
    static = {key: value for key, value in parameters.items() if key not in dynamic}
    return _same_json(static, expected)


def _affine_snapshot(task: RoundTask, path: Path) -> _ArtifactSnapshot | None:
    snapshot = _completion_snapshot(path)
    inputs = _registration_inputs(task)
    if snapshot is None or inputs is None:
        return None
    record, _identity = snapshot
    request = record.request
    expected = {
        **_common_parameters(task),
        "image_interpolation": "linear",
        "mask_interpolation": "nearestNeighbor",
        "registration_parameters": linear_parameters(
            task.registration.linear.type,
            task.registration.linear.accuracy,
        ),
    }
    moving_geometry = _mapping_geometry(request.parameters.get("moving_geometry"))
    if (
        not _request_basics_valid(request, inputs=inputs, task=task)
        or request.stage != "affine"
        or request.method != task.registration.linear.type
        or request.accuracy != task.registration.linear.accuracy
        or request.upstream is not None
        or moving_geometry is None
        or moving_geometry.spacing != task.voxel_spacing_xyz
        or not _static_parameters_valid(
            request.parameters,
            expected,
            dynamic=frozenset({"moving_geometry"}),
        )
    ):
        return None
    return snapshot


def _identity_from_evidence(value: object) -> FileIdentity | None:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        return None
    try:
        identity = FileIdentity.from_dict(dict(value))
        current = FileIdentity.from_path(Path(identity.path))
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    return identity if identity == current else None


def _serial_chunk_evidence_valid(task: RoundTask, request: object) -> bool:
    if not isinstance(request, StageRequest):
        return False
    parameters = request.parameters
    chunk_grid = parameters.get("chunk_grid")
    local_values = parameters.get("local_warps")
    manifest = _identity_from_evidence(parameters.get("chunk_manifest"))
    if (
        task.chunk_size_zyx is None
        or task.overlap_fraction is None
        or parameters.get("chunk_size_zyx") != list(task.chunk_size_zyx)
        or parameters.get("overlap_fraction") != task.overlap_fraction
        or not isinstance(chunk_grid, list)
        or not chunk_grid
        or not isinstance(local_values, list)
        or len(local_values) != len(chunk_grid)
        or manifest is None
    ):
        return False
    chunk_directory = task.registration_directory / f"round_{task.round}_chunks"
    if Path(manifest.path) != (chunk_directory / "manifest.json").resolve():
        return False
    seen_ids: list[int] = []
    for index, (chunk, local_value) in enumerate(
        zip(chunk_grid, local_values, strict=True)
    ):
        if not isinstance(chunk, Mapping) or frozenset(chunk) != _CHUNK_KEYS:
            return False
        chunk_values: dict[str, int] = {}
        for key in _CHUNK_KEYS:
            value = chunk.get(key)
            if isinstance(value, bool) or not isinstance(value, int):
                return False
            chunk_values[key] = value
        chunk_id = chunk_values["chunk_id"]
        if chunk_id != index:
            return False
        seen_ids.append(chunk_id)
        for axis, size in zip("zyx", request.geometry.shape, strict=True):
            start = chunk_values[f"{axis}_start"]
            end = chunk_values[f"{axis}_end"]
            start_ext = chunk_values[f"{axis}_start_ext"]
            end_ext = chunk_values[f"{axis}_end_ext"]
            if not (0 <= start_ext <= start < end <= end_ext <= size):
                return False
        local = _identity_from_evidence(local_value)
        if (
            local is None
            or Path(local.path)
            != (chunk_directory / f"chunk_{chunk_id:04d}_warp.nii.gz").resolve()
        ):
            return False
    return seen_ids == list(range(len(chunk_grid)))


def _warp_snapshot(
    task: RoundTask,
    path: Path,
    affine: _ArtifactSnapshot,
) -> _ArtifactSnapshot | None:
    snapshot = _completion_snapshot(path)
    inputs = _registration_inputs(task)
    if snapshot is None or inputs is None:
        return None
    record, _identity = snapshot
    request = record.request
    expected_scalar_policy = scalar_policy(task.strategy)
    expected = {
        **_common_parameters(task),
        "image_interpolation": "linear",
        "mask_interpolation": "nearestNeighbor",
        "registration_mode": (
            "whole-image" if task.strategy == "whole_image" else "serial-chunked"
        ),
        "registration_parameters": nonlinear_parameters(
            task.registration.nonlinear.accuracy
        ),
        "scalar_postprocessing_policy": expected_scalar_policy,
    }
    dynamic = {"moving_geometry"}
    if task.strategy == "serial_chunked":
        dynamic.update(
            {
                "chunk_grid",
                "chunk_manifest",
                "chunk_size_zyx",
                "local_warps",
                "overlap_fraction",
            }
        )
    moving_geometry = _mapping_geometry(request.parameters.get("moving_geometry"))
    if (
        not _request_basics_valid(request, inputs=inputs, task=task)
        or request.geometry != affine[0].request.geometry
        or request.stage != "nonlinear"
        or request.method != task.registration.nonlinear.type
        or request.accuracy != task.registration.nonlinear.accuracy
        or request.upstream != affine[1]
        or moving_geometry != request.geometry
        or not _static_parameters_valid(
            request.parameters,
            expected,
            dynamic=frozenset(dynamic),
        )
        or (
            task.strategy == "serial_chunked"
            and not _serial_chunk_evidence_valid(task, request)
        )
    ):
        return None
    return snapshot


def _registered_snapshot(
    task: RoundTask,
    path: Path,
    warp: _ArtifactSnapshot,
) -> _ArtifactSnapshot | None:
    snapshot = _completion_snapshot(path)
    inputs = _registration_inputs(task)
    if snapshot is None or inputs is None:
        return None
    record, _identity = snapshot
    request = record.request
    expected_scalar_policy = scalar_policy(task.strategy)
    expected = {
        **_common_parameters(task),
        "format": "uint16-tiff",
        "scalar_postprocessing_policy": expected_scalar_policy,
    }
    minimum = request.parameters.get("minimum")
    maximum = request.parameters.get("maximum")
    minimum_value = _finite_number(minimum)
    maximum_value = _finite_number(maximum)
    if (
        not _request_basics_valid(request, inputs=inputs, task=task)
        or request.geometry != warp[0].request.geometry
        or request.stage != "registered-image"
        or request.method != task.registration.nonlinear.type
        or request.accuracy != task.registration.nonlinear.accuracy
        or request.upstream != warp[1]
        or minimum_value is None
        or maximum_value is None
        or minimum_value > maximum_value
        or not _static_parameters_valid(
            request.parameters,
            expected,
            dynamic=frozenset({"minimum", "maximum"}),
        )
    ):
        return None
    return snapshot


def _channel_snapshot(
    task: RoundTask,
    path: Path,
    channel_index: int,
    affine: _ArtifactSnapshot,
    warp: _ArtifactSnapshot,
) -> _ArtifactSnapshot | None:
    snapshot = _completion_snapshot(path)
    channel = task.channels[channel_index]
    try:
        inputs = (
            FileIdentity.from_path(task.fixed_image),
            FileIdentity.from_path(channel.source),
            affine[1],
            warp[1],
        )
    except (OSError, RuntimeError):
        return None
    if snapshot is None:
        return None
    request = snapshot[0].request
    expected = {
        "pixel_size_xyz_um": list(task.voxel_spacing_xyz),
        "interpolation": channel.interpolation,
        "transform_order": ["warp", "affine"],
        "whichtoinvert": [False, False],
        "axis_convention": AXIS_CONVENTION_VERSION,
        "output_format": OUTPUT_FORMAT_POLICY,
        "postprocessing": CHANNEL_POSTPROCESSING_POLICY,
    }
    if (
        not _request_basics_valid(request, inputs=inputs, task=task)
        or request.stage != "channel-transform-v1"
        or request.method != "warp-affine"
        or request.accuracy != "not-applicable"
        or request.upstream != warp[1]
        or not _same_json(request.parameters, expected)
    ):
        return None
    return snapshot


def _completed_artifacts(task: RoundTask, paths: tuple[Path, ...]) -> tuple[Path, ...]:
    complete: list[Path] = []
    affine = _affine_snapshot(task, paths[0])
    if affine is None:
        return ()
    complete.append(paths[0])
    warp = _warp_snapshot(task, paths[1], affine)
    if warp is None:
        return tuple(complete)
    complete.append(paths[1])
    registered = _registered_snapshot(task, paths[2], warp)
    if registered is not None:
        complete.append(paths[2])
    for index, path in enumerate(paths[3:]):
        if _channel_snapshot(task, path, index, affine, warp) is not None:
            complete.append(path)
    return tuple(complete)


def _reused_artifacts(
    paths: tuple[Path, ...],
    before: dict[Path, _ArtifactSnapshot],
    *,
    forced: bool,
) -> tuple[str, ...]:
    if forced:
        return ()
    reused: list[str] = []
    for path in paths:
        original = before.get(path)
        if original is not None and _completion_snapshot(path) == original:
            reused.append(str(path))
    return tuple(reused)


def _thread_evidence(
    task: RoundTask, resolved: ResolvedNativeThreads | None
) -> tuple[tuple[str, int], ...]:
    if resolved is None:
        itk_name, *blas_names = NATIVE_THREAD_ENV_VARS
        return (
            (itk_name, task.native_threads.itk),
            *((name, task.native_threads.blas) for name in blas_names),
        )
    return tuple((name, resolved.values[name]) for name in NATIVE_THREAD_ENV_VARS)


def _write_log(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"{_utc_now()} {message}\n")


def _best_effort_failure_log(path: Path, message: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_log(path, message)
    except OSError:
        pass


def _result(
    *,
    task: RoundTask,
    status: Literal["completed", "failed"],
    started_at: str,
    started: float,
    hostname: str,
    pid: int,
    dask_task_key: str | None,
    slurm_job_id: str | None,
    resolved: ResolvedNativeThreads | None,
    completed_paths: tuple[Path, ...],
    reused_artifacts: tuple[str, ...],
    failure: RoundFailure | None,
) -> RoundResult:
    return RoundResult(
        dataset_id=task.dataset_id,
        round=task.round,
        status=status,
        started_at=started_at,
        ended_at=_utc_now(),
        duration_seconds=float(max(0.0, time.monotonic() - started)),
        hostname=hostname,
        pid=pid,
        dask_task_key=dask_task_key,
        slurm_job_id=slurm_job_id,
        strategy=task.strategy,
        native_threads=_thread_evidence(task, resolved),
        artifact_paths=tuple(str(path) for path in completed_paths),
        reused_artifacts=reused_artifacts,
        log_path=str(task.log_path),
        failure=failure,
    )


def run_round_task(task: RoundTask) -> RoundResult:
    """Run registration and channel transforms serially for one immutable task."""
    started_at = _utc_now()
    started = time.monotonic()
    planned = _planned_artifacts(task)
    hostname = "unknown"
    pid = 1
    dask_task_key: str | None = None
    slurm_job_id: str | None = None
    before: dict[Path, _ArtifactSnapshot] = {}
    resolved: ResolvedNativeThreads | None = None
    stage = "initialization"

    try:
        hostname = socket.gethostname()
        pid = os.getpid()
        dask_task_key = _dask_task_key()
        slurm_job_id = os.environ.get("SLURM_JOB_ID") or None
        initially_complete = _completed_artifacts(task, planned)
        before = {
            path: snapshot
            for path in initially_complete
            if (snapshot := _completion_snapshot(path)) is not None
        }

        stage = "native-threads"
        threads = NativeThreadConfig(
            itk_threads=task.native_threads.itk,
            blas_threads=task.native_threads.blas,
        )
        resolved = configure_native_threads(threads)

        stage = "setup"
        task.log_path.parent.mkdir(parents=True, exist_ok=True)
        task.registration_directory.mkdir(parents=True, exist_ok=True)
        task.log_path.touch(exist_ok=True)
        _write_log(task.log_path, f"round {task.round} started")

        stage = "runtime"
        (
            ImageRegistration,
            ChunkedImageRegistration,
            RegistrationPaths,
            transform_channel,
        ) = _runtime()

        registration_arguments: dict[str, object] = {
            "fixed_image_path": task.fixed_image,
            "moving_image_path": task.moving_image,
            "save_directory": task.registration_directory,
            "imaging_round": task.round,
            "pixel_size": task.voxel_spacing_xyz,
            "crop": task.registration.crop,
            "enable_logging": True,
            "force_override": task.force,
            "native_threads": threads,
            "linear_type": task.registration.linear.type,
            "linear_accuracy": task.registration.linear.accuracy,
            "nonlinear_type": task.registration.nonlinear.type,
            "nonlinear_accuracy": task.registration.nonlinear.accuracy,
        }
        stage = "registration"
        if task.strategy == "whole_image":
            registration = ImageRegistration(**registration_arguments)
        elif task.strategy == "serial_chunked":
            registration = ChunkedImageRegistration(
                **registration_arguments,
                chunk_size=task.chunk_size_zyx,
                overlap_fraction=task.overlap_fraction,
            )
        else:
            raise ValueError(f"Unsupported registration strategy: {task.strategy!r}")
        registration.register()

        registration_paths = RegistrationPaths.for_round(
            task.registration_directory, task.round
        )
        runtime_registration_paths = (
            Path(registration_paths.affine),
            Path(registration_paths.warp),
            Path(registration_paths.registered),
        )
        if runtime_registration_paths != planned[:3]:
            raise RuntimeError("registration paths do not match the planned artifacts")
        affine_path, warp_path, _registered_path = runtime_registration_paths
        completed = _completed_artifacts(task, planned)
        invalid_registration = tuple(
            path for path in runtime_registration_paths if path not in completed
        )
        if invalid_registration:
            raise RuntimeError(
                "registration artifacts lack task-consistent completion evidence: "
                f"{tuple(str(path) for path in invalid_registration)!r}"
            )
        _write_log(task.log_path, f"round {task.round} registration completed")

        for channel in task.channels:
            stage = f"channel:{channel.source.name}"
            channel.output.parent.mkdir(parents=True, exist_ok=True)
            transformed = transform_channel(
                fixed_image_path=task.fixed_image,
                moving_image_path=channel.source,
                affine_path=affine_path,
                warp_path=warp_path,
                output_path=channel.output,
                pixel_size=task.voxel_spacing_xyz,
                interpolation=channel.interpolation,
                native_threads=threads,
                force_override=task.force,
            )
            if Path(transformed) != channel.output:
                raise RuntimeError(
                    "channel transform returned a path other than its planned output"
                )
            if channel.output not in _completed_artifacts(task, planned):
                raise RuntimeError(
                    "channel artifact lacks task-consistent completion evidence: "
                    f"{channel.output}"
                )
            _write_log(task.log_path, f"channel completed: {channel.source.name}")

        stage = "validation"
        completed = _completed_artifacts(task, planned)
        if completed != planned:
            missing = tuple(str(path) for path in planned if path not in completed)
            raise RuntimeError(f"round completion validation failed: {missing!r}")
        reused = _reused_artifacts(completed, before, forced=task.force)
        _write_log(task.log_path, f"round {task.round} completed")
        return _result(
            task=task,
            status="completed",
            started_at=started_at,
            started=started,
            hostname=hostname,
            pid=pid,
            dask_task_key=dask_task_key,
            slurm_job_id=slurm_job_id,
            resolved=resolved,
            completed_paths=completed,
            reused_artifacts=reused,
            failure=None,
        )
    except Exception as error:
        formatted_traceback = traceback.format_exc()
        try:
            completed = _completed_artifacts(task, planned)
            reused = _reused_artifacts(completed, before, forced=task.force)
        except Exception:
            completed = ()
            reused = ()
        _best_effort_failure_log(
            task.log_path,
            f"round {task.round} failed at {stage}: "
            f"{type(error).__name__}: {error}\n{formatted_traceback}",
        )
        return _result(
            task=task,
            status="failed",
            started_at=started_at,
            started=started,
            hostname=hostname,
            pid=pid,
            dask_task_key=dask_task_key,
            slurm_job_id=slurm_job_id,
            resolved=resolved,
            completed_paths=completed,
            reused_artifacts=reused,
            failure=RoundFailure(
                stage=stage,
                exception_type=type(error).__name__,
                message=str(error),
            ),
        )


__all__ = ["run_round_task"]
