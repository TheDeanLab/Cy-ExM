"""Execute one scheduler-neutral registration round."""

from __future__ import annotations

import os
import socket
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from cyexm.registration.artifacts import CompletionRecord, FileIdentity
from cyexm.runtime import (
    NATIVE_THREAD_ENV_VARS,
    NativeThreadConfig,
    ResolvedNativeThreads,
    configure_native_threads,
)
from cyexm.workflow.models import RoundFailure, RoundResult, RoundTask

type _ArtifactSnapshot = tuple[CompletionRecord, FileIdentity]


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
    except Exception:
        return None
    if key is None:
        return None
    encoded = str(key)
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
        record = CompletionRecord.read(record_path)
        identity = FileIdentity.from_path(path)
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    if record.artifact != identity:
        return None
    return record, identity


def _completed_artifacts(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    return tuple(path for path in paths if _completion_snapshot(path) is not None)


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
    hostname = socket.gethostname()
    pid = os.getpid()
    dask_task_key = _dask_task_key()
    slurm_job_id = os.environ.get("SLURM_JOB_ID") or None
    planned = _planned_artifacts(task)
    before = {
        path: snapshot
        for path in planned
        if (snapshot := _completion_snapshot(path)) is not None
    }
    resolved: ResolvedNativeThreads | None = None
    stage = "native-threads"

    try:
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
        affine_path = Path(registration_paths.affine)
        warp_path = Path(registration_paths.warp)
        if (affine_path, warp_path) != planned[:2]:
            raise RuntimeError("registration paths do not match the planned artifacts")
        for path in planned[:3]:
            if _completion_snapshot(path) is None:
                raise RuntimeError(
                    f"registration artifact lacks a valid completion record: {path}"
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
            if _completion_snapshot(channel.output) is None:
                raise RuntimeError(
                    "channel artifact lacks a valid completion record: "
                    f"{channel.output}"
                )
            _write_log(task.log_path, f"channel completed: {channel.source.name}")

        stage = "validation"
        completed = _completed_artifacts(planned)
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
        completed = _completed_artifacts(planned)
        reused = _reused_artifacts(completed, before, forced=task.force)
        _best_effort_failure_log(
            task.log_path,
            f"round {task.round} failed at {stage}: {type(error).__name__}: {error}",
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
