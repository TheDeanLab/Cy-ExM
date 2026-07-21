"""Prepare every declared reference-round channel on one shared grid."""

from __future__ import annotations

import os
import socket
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from cyexm.registration._request_policy import (
    reference_channel_parameters,
    reference_crop_parameters,
)
from cyexm.registration.artifacts import ArtifactStore, FileIdentity, StageRequest
from cyexm.registration.crop import bounds_text, estimate_bounds, read_bounds
from cyexm.runtime import (
    NATIVE_THREAD_ENV_VARS,
    NativeThreadConfig,
    ResolvedNativeThreads,
    configure_native_threads,
)
from cyexm.spatial import ImageGeometry, geometry_compatible
from cyexm.workflow.models import ReferenceTask, RoundFailure, RoundResult


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _backend(threads: NativeThreadConfig) -> Any:
    from cyexm.registration.backend import AntsRegistrationBackend

    return AntsRegistrationBackend(threads)


def _thread_evidence(
    task: ReferenceTask, resolved: ResolvedNativeThreads | None
) -> tuple[tuple[str, int], ...]:
    if resolved is None:
        itk_name, *blas_names = NATIVE_THREAD_ENV_VARS
        return (
            (itk_name, task.native_threads.itk),
            *((name, task.native_threads.blas) for name in blas_names),
        )
    return tuple((name, resolved.values[name]) for name in NATIVE_THREAD_ENV_VARS)


def _result(
    task: ReferenceTask,
    *,
    status: Literal["completed", "failed"],
    started_at: str,
    started: float,
    resolved: ResolvedNativeThreads | None,
    artifacts: tuple[Path, ...],
    reused: tuple[str, ...],
    failure: RoundFailure | None,
) -> RoundResult:
    return RoundResult(
        dataset_id=task.dataset_id,
        round=task.round,
        status=status,
        started_at=started_at,
        ended_at=_utc_now(),
        duration_seconds=max(0.0, time.monotonic() - started),
        hostname=socket.gethostname(),
        pid=os.getpid(),
        dask_task_key=None,
        slurm_job_id=os.environ.get("SLURM_JOB_ID") or None,
        strategy="reference-preparation",
        native_threads=_thread_evidence(task, resolved),
        artifact_paths=tuple(str(path) for path in artifacts),
        reused_artifacts=reused,
        log_path=str(task.log_path),
        failure=failure,
    )


def _validate_task(task: ReferenceTask) -> None:
    if type(task) is not ReferenceTask or not task.channels:
        raise ValueError("reference task must be exact and declare channels")
    if task.crop != (task.crop_bounds_path is not None):
        raise ValueError("reference crop path must match crop policy")
    sources = tuple(channel.source.resolve(strict=True) for channel in task.channels)
    outputs = tuple(channel.output.resolve(strict=False) for channel in task.channels)
    if len(set(sources)) != len(sources) or len(set(outputs)) != len(outputs):
        raise ValueError("reference source and output paths must be unique")
    if set(sources) & set(outputs):
        raise ValueError("reference outputs must not alias source images")
    if task.registration_channel_source.resolve(strict=True) not in sources:
        raise ValueError("reference registration source must be declared")
    if task.registration_channel_output.resolve(strict=False) not in outputs:
        raise ValueError("reference registration output must be declared")


def run_reference_task(
    task: ReferenceTask,
    *,
    _backend: Any | None = None,
    _artifact_store: ArtifactStore | None = None,
) -> RoundResult:
    started_at = _utc_now()
    started = time.monotonic()
    resolved: ResolvedNativeThreads | None = None
    completed: list[Path] = []
    reused: list[str] = []
    stage = "validation"
    try:
        _validate_task(task)
        threads = NativeThreadConfig(
            itk_threads=task.native_threads.itk,
            blas_threads=task.native_threads.blas,
        )
        stage = "native-threads"
        resolved = configure_native_threads(threads)
        backend = _backend if _backend is not None else globals()["_backend"](threads)
        store = _artifact_store or ArtifactStore()
        stage = "source-geometry"
        identities = tuple(
            FileIdentity.from_path(channel.source) for channel in task.channels
        )
        images = tuple(
            backend.load_image(channel.source, task.voxel_spacing_xyz)
            for channel in task.channels
        )
        geometries = tuple(backend.geometry(image) for image in images)
        registration_index = next(
            index
            for index, channel in enumerate(task.channels)
            if channel.source == task.registration_channel_source
        )
        expected = geometries[registration_index]
        if any(not geometry_compatible(geometry, expected) for geometry in geometries):
            raise ValueError("declared reference channel geometries do not match")

        crop_identity = None
        bounds = None
        if task.crop_bounds_path is not None:
            stage = "crop"
            crop_request = StageRequest.for_identities(
                stage="reference-crop",
                inputs=(identities[registration_index],),
                geometry=expected,
                method="robust-bounding-box",
                accuracy="legacy-robust",
                native_threads=resolved.values,
                upstream=None,
                parameters=reference_crop_parameters(),
            )
            if task.force:
                store.mark_stale(task.crop_bounds_path)
            else:
                bounds = store.reuse(
                    task.crop_bounds_path,
                    crop_request,
                    lambda path: read_bounds(path, expected.shape),
                )
            if bounds is None:
                registration_image = images[registration_index]
                bounds = estimate_bounds(backend, registration_image)
                text = bounds_text(bounds)

                def write_crop(path: Path) -> None:
                    path.write_text(text, encoding="utf-8")

                store.commit(
                    task.crop_bounds_path,
                    crop_request,
                    write_crop,
                    lambda path: read_bounds(path, expected.shape) == bounds,
                )
            else:
                reused.append(str(task.crop_bounds_path))
            completed.append(task.crop_bounds_path)
            crop_identity = FileIdentity.from_path(task.crop_bounds_path)

        for index, (channel, image, identity) in enumerate(
            zip(task.channels, images, identities, strict=True)
        ):
            stage = f"channel:{channel.source.name}"
            prepared = (
                backend.crop_image(image, bounds) if bounds is not None else image
            )
            output_geometry = backend.geometry(prepared)
            request = StageRequest.for_identities(
                stage="reference-channel",
                inputs=(identity,),
                geometry=output_geometry,
                method="uint16-tiff",
                accuracy="exact",
                native_threads=resolved.values,
                upstream=crop_identity,
                parameters=reference_channel_parameters(
                    index=index, cropped=bounds is not None
                ),
            )
            if task.force:
                store.mark_stale(channel.output)
                cached = None
            else:
                cached = store.reuse(
                    channel.output,
                    request,
                    lambda path: backend.load_image(path, task.voxel_spacing_xyz),
                )
            if cached is None:
                if FileIdentity.from_path(channel.source) != identity:
                    raise RuntimeError("reference source changed during preparation")

                def write_channel(
                    path: Path, image_to_write: object = prepared
                ) -> None:
                    backend.write_clipped_tiff(image_to_write, path)

                def validate_channel(
                    path: Path, expected_geometry: ImageGeometry = output_geometry
                ) -> bool:
                    return geometry_compatible(
                        backend.geometry(
                            backend.load_image(path, task.voxel_spacing_xyz)
                        ),
                        expected_geometry,
                    )

                store.commit(
                    channel.output,
                    request,
                    write_channel,
                    validate_channel,
                )
            else:
                reused.append(str(channel.output))
            completed.append(channel.output)
        task.log_path.parent.mkdir(parents=True, exist_ok=True)
        task.log_path.touch(exist_ok=True)
        return _result(
            task,
            status="completed",
            started_at=started_at,
            started=started,
            resolved=resolved,
            artifacts=tuple(completed),
            reused=tuple(reused),
            failure=None,
        )
    except Exception as error:
        return _result(
            task,
            status="failed",
            started_at=started_at,
            started=started,
            resolved=resolved,
            artifacts=tuple(completed),
            reused=tuple(reused),
            failure=RoundFailure(
                stage=stage,
                exception_type=type(error).__name__,
                message=f"{error}\n{traceback.format_exc()}",
            ),
        )


__all__ = ("run_reference_task",)
