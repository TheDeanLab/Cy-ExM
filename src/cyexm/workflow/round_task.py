"""Execute one scheduler-neutral registration round."""

from __future__ import annotations

import os
import socket
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from cyexm.runtime import (
    NATIVE_THREAD_ENV_VARS,
    NativeThreadConfig,
    ResolvedNativeThreads,
    configure_native_threads,
)
from cyexm.spatial import ImageGeometry, PixelSize
from cyexm.workflow._round_provenance import (
    ArtifactSnapshot as _ArtifactSnapshot,
)
from cyexm.workflow._round_provenance import (
    RegistrationArtifactPaths,
    SourceGeometries,
    channel_artifact_valid,
    completion_evidence,
    completion_snapshot,
)
from cyexm.workflow._round_provenance import (
    RoundCompletionEvidence as _RoundCompletionEvidence,
)
from cyexm.workflow.models import RoundFailure, RoundResult, RoundTask


class _RuntimeRegistrationPaths(Protocol):
    output: str | os.PathLike[str]
    affine: str | os.PathLike[str]
    warp: str | os.PathLike[str]
    registered: str | os.PathLike[str]
    pre_metrics: str | os.PathLike[str]
    linear_metrics: str | os.PathLike[str]
    nonlinear_metrics: str | os.PathLike[str]
    moving_crop: str | os.PathLike[str]


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


def _planned_artifacts(
    task: RoundTask, paths: RegistrationArtifactPaths
) -> tuple[Path, ...]:
    return (
        *paths.public_registration_artifacts,
        *(channel.output for channel in task.channels),
    )


def _completion_snapshot(path: Path) -> _ArtifactSnapshot | None:
    return completion_snapshot(path)


def _inspect_image_geometry(path: Path, spacing: PixelSize) -> ImageGeometry:
    """Inspect authoritative TIFF geometry without loading voxel data."""
    from cyexm.workflow.planning import _inspect_tiff_header_in_process

    header = _inspect_tiff_header_in_process(path)
    if header.geometry is not None:
        return header.geometry
    return ImageGeometry(
        shape=header.shape_zyx,
        origin=(0.0, 0.0, 0.0),
        spacing=spacing,
        direction=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )


def _completed_artifacts(
    task: RoundTask,
    paths: RegistrationArtifactPaths,
    source_geometries: SourceGeometries,
) -> _RoundCompletionEvidence:
    return completion_evidence(
        task,
        paths,
        source_geometries,
        snapshot_reader=_completion_snapshot,
        geometry_reader=_inspect_image_geometry,
    )


def _registration_paths(
    value: _RuntimeRegistrationPaths, round_number: int
) -> RegistrationArtifactPaths:
    paths = RegistrationArtifactPaths.for_round(Path(value.output), round_number)
    if (
        paths.affine != Path(value.affine)
        or paths.warp != Path(value.warp)
        or paths.registered != Path(value.registered)
        or paths.pre_metrics != Path(value.pre_metrics)
        or paths.linear_metrics != Path(value.linear_metrics)
        or paths.nonlinear_metrics != Path(value.nonlinear_metrics)
        or paths.moving_crop != Path(value.moving_crop)
    ):
        raise RuntimeError(
            "runtime registration paths do not match the workflow layout"
        )
    return paths


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
    registration_paths = RegistrationArtifactPaths.for_round(
        task.registration_directory, task.round
    )
    planned = _planned_artifacts(task, registration_paths)
    hostname = "unknown"
    pid = 1
    dask_task_key: str | None = None
    slurm_job_id: str | None = None
    before: dict[Path, _ArtifactSnapshot] = {}
    resolved: ResolvedNativeThreads | None = None
    source_geometries: SourceGeometries | None = None
    stage = "initialization"

    try:
        if type(task.prepared_fixed) is not bool or not task.prepared_fixed:
            raise ValueError("workflow round task requires prepared_fixed=True")
        hostname = socket.gethostname()
        pid = os.getpid()
        dask_task_key = _dask_task_key()
        slurm_job_id = os.environ.get("SLURM_JOB_ID") or None

        stage = "native-threads"
        threads = NativeThreadConfig(
            itk_threads=task.native_threads.itk,
            blas_threads=task.native_threads.blas,
        )
        resolved = configure_native_threads(threads)

        stage = "source-geometry"
        source_geometries = SourceGeometries(
            fixed=_inspect_image_geometry(task.fixed_image, task.voxel_spacing_xyz),
            moving=_inspect_image_geometry(task.moving_image, task.voxel_spacing_xyz),
        )

        stage = "runtime"
        (
            ImageRegistration,
            ChunkedImageRegistration,
            RegistrationPaths,
            transform_channel,
        ) = _runtime()
        initially_complete = _completed_artifacts(
            task, registration_paths, source_geometries
        ).artifacts
        before = {
            path: snapshot
            for path in initially_complete
            if (snapshot := _completion_snapshot(path)) is not None
        }

        stage = "setup"
        task.log_path.parent.mkdir(parents=True, exist_ok=True)
        task.registration_directory.mkdir(parents=True, exist_ok=True)
        task.log_path.touch(exist_ok=True)
        _write_log(task.log_path, f"round {task.round} started")

        registration_arguments: dict[str, object] = {
            "fixed_image_path": task.fixed_image,
            "moving_image_path": task.moving_image,
            "save_directory": task.registration_directory,
            "imaging_round": task.round,
            "pixel_size": task.voxel_spacing_xyz,
            "crop": task.registration.crop,
            "enable_logging": True,
            "force_override": task.force,
            "prepared_fixed": task.prepared_fixed,
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

        runtime_paths_value = RegistrationPaths.for_round(
            task.registration_directory, task.round
        )
        if _registration_paths(runtime_paths_value, task.round) != registration_paths:
            raise RuntimeError("registration paths do not match the planned artifacts")

        affine_path, warp_path, _registered_path = (
            registration_paths.public_registration_artifacts
        )
        registration_evidence = _completed_artifacts(
            task, registration_paths, source_geometries
        )
        completed = registration_evidence.artifacts
        invalid_registration = tuple(
            path
            for path in registration_paths.public_registration_artifacts
            if path not in completed
        )
        if invalid_registration:
            raise RuntimeError(
                "registration artifacts lack task-consistent completion evidence: "
                f"{tuple(str(path) for path in invalid_registration)!r}"
            )
        affine_snapshot = registration_evidence.affine
        warp_snapshot = registration_evidence.warp
        if affine_snapshot is None or warp_snapshot is None:
            raise RuntimeError(
                "registration transform completion evidence was not accepted"
            )
        _write_log(task.log_path, f"round {task.round} registration completed")

        for channel_index, channel in enumerate(task.channels):
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
            if not channel_artifact_valid(
                task,
                channel_index,
                source_geometries.fixed,
                affine_snapshot,
                warp_snapshot,
                snapshot_reader=_completion_snapshot,
            ):
                raise RuntimeError(
                    "channel artifact lacks task-consistent completion evidence: "
                    f"{channel.output}"
                )
            _write_log(task.log_path, f"channel completed: {channel.source.name}")

        stage = "validation"
        completed = _completed_artifacts(
            task, registration_paths, source_geometries
        ).artifacts
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
            if source_geometries is None:
                completed = ()
                reused = ()
            else:
                completed = _completed_artifacts(
                    task, registration_paths, source_geometries
                ).artifacts
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
