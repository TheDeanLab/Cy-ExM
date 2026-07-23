"""Execute one immutable fine-grained task without owning a scheduler."""

from __future__ import annotations

import os
import socket
import sys
import time
import traceback
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from cyexm.runtime import (
    NATIVE_THREAD_ENV_VARS,
    NativeThreadConfig,
    ResolvedNativeThreads,
    configure_native_threads,
)
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
from cyexm.workflow.models import RoundFailure, RoundTask


@dataclass(frozen=True, slots=True)
class _StageOutcome:
    artifact_paths: tuple[Path, ...]
    reused_artifacts: tuple[Path, ...]
    prepared_descriptor: dict[str, object] | None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _dask_task_key() -> str | None:
    distributed = sys.modules.get("distributed")
    if distributed is None:
        return None
    try:
        worker = distributed.get_worker()
        value = worker.get_current_task()
    except Exception:
        return None
    if value is None:
        return None
    encoded = str(value)
    return encoded or None


def _write_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"{_utc_now()} {message}\n")


def _best_effort_log(path: Path, message: str) -> None:
    with suppress(OSError):
        _write_log(path, message)


def _thread_evidence(
    task: FineWorkflowTask,
    resolved: ResolvedNativeThreads | None,
) -> tuple[tuple[str, int], ...]:
    if resolved is None:
        itk_name, *blas_names = NATIVE_THREAD_ENV_VARS
        return (
            (itk_name, task.round_task.native_threads.itk),
            *((name, task.round_task.native_threads.blas) for name in blas_names),
        )
    return tuple((name, resolved.values[name]) for name in NATIVE_THREAD_ENV_VARS)


def _new_image_registration(task: RoundTask, threads: NativeThreadConfig) -> object:
    from cyexm.registration.whole_image import ImageRegistration

    return ImageRegistration(
        fixed_image_path=task.fixed_image,
        moving_image_path=task.moving_image,
        save_directory=task.registration_directory,
        imaging_round=task.round,
        pixel_size=task.voxel_spacing_xyz,
        crop=task.registration.crop,
        enable_logging=False,
        force_override=task.force,
        prepared_fixed=task.prepared_fixed,
        native_threads=threads,
        linear_type=task.registration.linear.type,
        linear_accuracy=task.registration.linear.accuracy,
        nonlinear_type=task.registration.nonlinear.type,
        nonlinear_accuracy=task.registration.nonlinear.accuracy,
        nonlinear_metric=task.registration.nonlinear.metric,
        nonlinear_radius=task.registration.nonlinear.radius,
    )


def _snapshots(paths: tuple[Path, ...]) -> dict[Path, object]:
    from cyexm.workflow._round_provenance import completion_snapshot

    return {
        path: snapshot
        for path in paths
        if (snapshot := completion_snapshot(path)) is not None
    }


def _unchanged(
    paths: tuple[Path, ...], before: dict[Path, object], *, forced: bool
) -> tuple[Path, ...]:
    if forced:
        return ()
    from cyexm.workflow._round_provenance import completion_snapshot

    return tuple(
        path
        for path in paths
        if path in before and completion_snapshot(path) == before[path]
    )


def _run_preparation(task: RoundPreparationTask) -> _StageOutcome:
    from cyexm.registration.prepared import commit_prepared_round
    from cyexm.registration.whole_image import ImageRegistration

    threads = NativeThreadConfig(
        itk_threads=task.round_task.native_threads.itk,
        blas_threads=task.round_task.native_threads.blas,
    )
    before = _snapshots(task.output_paths)
    registration = cast(
        ImageRegistration,
        _new_image_registration(task.round_task, threads),
    )
    manifest = commit_prepared_round(registration)
    if any(not path.is_file() for path in task.output_paths):
        raise RuntimeError("prepared round did not produce every planned artifact")
    return _StageOutcome(
        artifact_paths=task.output_paths,
        reused_artifacts=_unchanged(
            task.output_paths, before, forced=task.round_task.force
        ),
        prepared_descriptor=manifest.to_dict(),
    )


def _verify_identity(identity: object, role: str) -> None:
    from cyexm.registration.artifacts import FileIdentity

    if type(identity) is not FileIdentity:
        raise TypeError(f"{role} must be an exact FileIdentity")
    if FileIdentity.from_path(Path(identity.path)) != identity:
        raise RuntimeError(f"{role} identity changed")


def _whole_chunk(shape: tuple[int, int, int]) -> object:
    from cyexm.chunking import ChunkInfo

    z, y, x = shape
    return ChunkInfo(0, 0, z, 0, y, 0, x, 0, z, 0, y, 0, x)


def _chunk_context(
    task: NonlinearChunkTask | ChunkAssemblyTask,
    *,
    assembly: bool,
) -> tuple[Any, Any | None, tuple[Any, ...]]:
    from cyexm.chunking import compute_chunk_grid
    from cyexm.registration.artifacts import ArtifactStore
    from cyexm.registration.backend import AntsRegistrationBackend
    from cyexm.registration.chunk_stages import ChunkRegion, ChunkStageContext
    from cyexm.registration.prepared import (
        PreparedManifest,
        read_prepared_region,
    )

    round_task = task.round_task
    if round_task.chunk_size_zyx is None or round_task.overlap_fraction is None:
        raise ValueError("chunk task requires complete chunk geometry")
    manifest = PreparedManifest.from_dict(task.prepared_descriptor)
    chunks = compute_chunk_grid(
        manifest.geometry.shape,
        round_task.chunk_size_zyx,
        round_task.overlap_fraction,
    )
    threads = NativeThreadConfig(
        itk_threads=round_task.native_threads.itk,
        blas_threads=round_task.native_threads.blas,
    )
    backend = AntsRegistrationBackend(threads)
    store = ArtifactStore()
    source_identities = (manifest.native_fixed, manifest.native_moving)

    def verify_sources() -> None:
        _verify_identity(manifest.native_fixed, "fixed source")
        _verify_identity(manifest.native_moving, "moving source")
        _verify_identity(manifest.affine, "affine")

    def reuse_artifact(path: Path, request: object, reader: object) -> object | None:
        verify_sources()
        return store.reuse(path, request, reader)  # type: ignore[arg-type]

    def commit_artifact(
        path: Path, request: object, writer: object, validator: object
    ) -> None:
        verify_sources()
        store.commit(
            path,
            request,  # type: ignore[arg-type]
            writer,  # type: ignore[arg-type]
            validator,  # type: ignore[arg-type]
        )
        verify_sources()

    def mark_chunk_stale(chunk_id: int) -> None:
        chunk_root = (
            round_task.registration_directory / f"round_{round_task.round}_chunks"
        )
        for path in (
            chunk_root / f"chunk_{chunk_id:04d}_warp.nii.gz",
            chunk_root / f"chunk_{chunk_id:04d}_registered_core.npy",
            chunk_root / "manifest.json",
            *ChunkAssemblyTask.create(
                round_task,
                prepared_descriptor=task.prepared_descriptor,
                dependencies=tuple(
                    NonlinearChunkTask.create(
                        round_task,
                        chunk,
                        prepared_descriptor=task.prepared_descriptor,
                        dependencies=(),
                    ).key
                    for chunk in chunks
                ),
            ).output_paths[:3],
        ):
            store.mark_stale(path)

    fixed = backend.load_image(round_task.fixed_image, round_task.voxel_spacing_xyz)
    if backend.geometry(fixed) != manifest.geometry:
        raise ValueError("prepared fixed TIFF geometry does not match the manifest")
    if assembly:
        selected = _whole_chunk(manifest.geometry.shape)
    else:
        if type(task) is not NonlinearChunkTask:
            raise TypeError("regional chunk context requires a nonlinear chunk task")
        selected = task.chunk
    region = read_prepared_region(manifest, selected)  # type: ignore[arg-type]
    if assembly:
        reference = fixed
    else:
        if type(task) is not NonlinearChunkTask:
            raise TypeError("regional chunk context requires a nonlinear chunk task")
        reference = backend.crop_image(
            fixed,
            (
                task.chunk.z_start_ext,
                task.chunk.z_end_ext,
                task.chunk.y_start_ext,
                task.chunk.y_end_ext,
                task.chunk.x_start_ext,
                task.chunk.x_end_ext,
            ),
        )
    fixed_region = backend.from_array(region.fixed_scalar, reference=reference)
    moving_region = backend.from_array(region.moving_linear_scalar, reference=reference)
    fixed_mask_region = backend.from_array(region.fixed_mask, reference=reference)
    moving_mask_region = backend.from_array(
        region.moving_linear_mask, reference=reference
    )
    context = ChunkStageContext(
        source_identities=source_identities,
        affine_identity=manifest.affine,
        fixed_geometry=manifest.geometry,
        fixed=fixed_region if assembly else fixed,
        moving_linear=moving_region,
        fixed_mask=fixed_mask_region,
        moving_linear_mask=moving_mask_region,
        nonlinear_type=round_task.registration.nonlinear.type,
        nonlinear_accuracy=round_task.registration.nonlinear.accuracy,
        nonlinear_metric=round_task.registration.nonlinear.metric,
        nonlinear_radius=round_task.registration.nonlinear.radius,
        chunks=chunks,
        chunk_size=round_task.chunk_size_zyx,
        overlap_fraction=round_task.overlap_fraction,
        native_threads=dict(manifest.native_threads),
        force=round_task.force,
        output_directory=round_task.registration_directory,
        round_number=round_task.round,
        pixel_size=round_task.voxel_spacing_xyz,
        crop=round_task.registration.crop,
        prepared_fixed=round_task.prepared_fixed,
        backend=backend,
        store=store,
        verify_sources=verify_sources,
        reuse_artifact=reuse_artifact,
        commit_artifact=commit_artifact,
        mark_chunk_stale=mark_chunk_stale,
    )
    chunk_region = (
        None
        if assembly
        else ChunkRegion(
            fixed=fixed_region,
            moving=moving_region,
            fixed_mask=fixed_mask_region,
            moving_mask=moving_mask_region,
            fixed_array=region.fixed_scalar,
            moving_array=region.moving_linear_scalar,
        )
    )
    return context, chunk_region, chunks


def _run_chunk(task: NonlinearChunkTask) -> _StageOutcome:
    from cyexm.registration.chunk_stages import run_nonlinear_chunk

    context, region, _chunks = _chunk_context(task, assembly=False)
    if region is None:
        raise RuntimeError("chunk region was not constructed")
    artifact = run_nonlinear_chunk(context, region, task.chunk)
    paths = (artifact.warp_path, artifact.scalar_core_path)
    return _StageOutcome(
        artifact_paths=paths,
        reused_artifacts=paths if artifact.reused else (),
        prepared_descriptor=None,
    )


def _run_assembly(task: ChunkAssemblyTask) -> _StageOutcome:
    from cyexm.registration.artifacts import FileIdentity
    from cyexm.registration.chunk_stages import (
        ChunkStageArtifacts,
        assemble_nonlinear_chunks,
    )

    context, _region, chunks = _chunk_context(task, assembly=True)
    artifacts = tuple(
        ChunkStageArtifacts(
            warp_path=context.warp_path(chunk.chunk_id),
            scalar_core_path=context.scalar_core_path(chunk.chunk_id),
            warp_identity=FileIdentity.from_path(context.warp_path(chunk.chunk_id)),
            scalar_core_identity=FileIdentity.from_path(
                context.scalar_core_path(chunk.chunk_id)
            ),
            reused=True,
        )
        for chunk in chunks
    )
    before = _snapshots(task.output_paths)
    assembled = assemble_nonlinear_chunks(context, artifacts)
    paths = (
        assembled.warp_path,
        assembled.registered_path,
        assembled.nonlinear_metrics_path,
        assembled.chunk_manifest_path,
    )
    return _StageOutcome(
        artifact_paths=paths,
        reused_artifacts=_unchanged(paths, before, forced=task.round_task.force),
        prepared_descriptor=None,
    )


def _source_geometries(task: RoundTask) -> Any:
    from cyexm.workflow._round_provenance import SourceGeometries
    from cyexm.workflow.round_task import _inspect_image_geometry

    return SourceGeometries(
        fixed=_inspect_image_geometry(task.fixed_image, task.voxel_spacing_xyz),
        moving=_inspect_image_geometry(task.moving_image, task.voxel_spacing_xyz),
    )


def _accepted_registration(task: RoundTask) -> Any:
    from cyexm.workflow._round_provenance import (
        RegistrationArtifactPaths,
        completion_evidence,
        completion_snapshot,
    )
    from cyexm.workflow.round_task import _inspect_image_geometry

    paths = RegistrationArtifactPaths.for_round(task.registration_directory, task.round)
    evidence = completion_evidence(
        task,
        paths,
        _source_geometries(task),
        snapshot_reader=completion_snapshot,
        geometry_reader=_inspect_image_geometry,
    )
    if any(
        path not in evidence.artifacts for path in paths.public_registration_artifacts
    ):
        raise RuntimeError(
            "registration artifacts lack task-consistent completion evidence"
        )
    if evidence.affine is None or evidence.warp is None:
        raise RuntimeError(
            "registration transform completion evidence was not accepted"
        )
    return evidence


def _run_whole_image(task: WholeImageRegistrationTask) -> _StageOutcome:
    from cyexm.registration.whole_image import ImageRegistration

    round_task = task.round_task
    threads = NativeThreadConfig(
        itk_threads=round_task.native_threads.itk,
        blas_threads=round_task.native_threads.blas,
    )
    before = _snapshots(task.output_paths)
    registration = cast(
        ImageRegistration,
        _new_image_registration(round_task, threads),
    )
    registration.register()
    _accepted_registration(round_task)
    if any(not path.is_file() for path in task.output_paths):
        raise RuntimeError("whole-image registration omitted a planned artifact")
    return _StageOutcome(
        artifact_paths=task.output_paths,
        reused_artifacts=_unchanged(task.output_paths, before, forced=round_task.force),
        prepared_descriptor=None,
    )


def _run_channel(task: ChannelTransformTask) -> _StageOutcome:
    from cyexm.registration.channels import transform_channel
    from cyexm.workflow._round_provenance import (
        RegistrationArtifactPaths,
        channel_artifact_valid,
    )

    round_task = task.round_task
    evidence = _accepted_registration(round_task)
    paths = RegistrationArtifactPaths.for_round(
        round_task.registration_directory, round_task.round
    )
    channel = round_task.channels[task.channel_index]
    before = _snapshots(task.output_paths)
    threads = NativeThreadConfig(
        itk_threads=round_task.native_threads.itk,
        blas_threads=round_task.native_threads.blas,
    )
    transformed = transform_channel(
        fixed_image_path=round_task.fixed_image,
        moving_image_path=channel.source,
        affine_path=paths.affine,
        warp_path=paths.warp,
        output_path=channel.output,
        pixel_size=round_task.voxel_spacing_xyz,
        interpolation=channel.interpolation,
        native_threads=threads,
        force_override=round_task.force,
    )
    if Path(transformed) != channel.output:
        raise RuntimeError("channel transform returned an unplanned output")
    if not channel_artifact_valid(
        round_task,
        task.channel_index,
        _source_geometries(round_task).fixed,
        evidence.affine,
        evidence.warp,
    ):
        raise RuntimeError("channel lacks task-consistent completion evidence")
    return _StageOutcome(
        artifact_paths=task.output_paths,
        reused_artifacts=_unchanged(task.output_paths, before, forced=round_task.force),
        prepared_descriptor=None,
    )


def _run_validation(task: RoundValidationTask) -> _StageOutcome:
    evidence = _accepted_registration(task.round_task)
    if tuple(evidence.artifacts) != task.output_paths:
        missing = tuple(
            str(path) for path in task.output_paths if path not in evidence.artifacts
        )
        raise RuntimeError(f"round completion validation failed: {missing!r}")
    return _StageOutcome(
        artifact_paths=task.output_paths,
        reused_artifacts=task.output_paths,
        prepared_descriptor=None,
    )


def _execute_fine_task(task: FineWorkflowTask) -> _StageOutcome:
    if type(task) is RoundPreparationTask:
        return _run_preparation(task)
    if type(task) is NonlinearChunkTask:
        return _run_chunk(task)
    if type(task) is ChunkAssemblyTask:
        return _run_assembly(task)
    if type(task) is WholeImageRegistrationTask:
        return _run_whole_image(task)
    if type(task) is ChannelTransformTask:
        return _run_channel(task)
    if type(task) is RoundValidationTask:
        return _run_validation(task)
    raise TypeError(f"fine workflow task must use an exact task type; got {task!r}")


def run_fine_task(task: FineWorkflowTask) -> FineTaskResult:
    """Execute one exact fine task and return terminal operational evidence."""
    exact_types = (
        RoundPreparationTask,
        NonlinearChunkTask,
        ChunkAssemblyTask,
        WholeImageRegistrationTask,
        ChannelTransformTask,
        RoundValidationTask,
    )
    if type(task) not in exact_types:
        raise TypeError(f"fine workflow task must use an exact task type; got {task!r}")
    started_at = _utc_now()
    started = time.monotonic()
    resolved: ResolvedNativeThreads | None = None
    hostname = "unknown"
    pid = 1
    dask_key: str | None = None
    slurm_job: str | None = None
    try:
        hostname = socket.gethostname()
        pid = os.getpid()
        dask_key = _dask_task_key()
        slurm_job = os.environ.get("SLURM_JOB_ID") or None
        threads = NativeThreadConfig(
            itk_threads=task.round_task.native_threads.itk,
            blas_threads=task.round_task.native_threads.blas,
        )
        resolved = configure_native_threads(threads)
        _write_log(task.log_path, f"{task.key.encoded} started")
        outcome = _execute_fine_task(task)
        _write_log(task.log_path, f"{task.key.encoded} completed")
        status = (
            "reused"
            if outcome.artifact_paths
            and set(outcome.reused_artifacts) == set(outcome.artifact_paths)
            else "completed"
        )
        return FineTaskResult(
            task=task,
            status=status,
            started_at=started_at,
            ended_at=_utc_now(),
            duration_seconds=max(0.0, time.monotonic() - started),
            hostname=hostname,
            pid=pid,
            dask_task_key=dask_key,
            slurm_job_id=slurm_job,
            native_threads=_thread_evidence(task, resolved),
            artifact_paths=tuple(str(path) for path in outcome.artifact_paths),
            reused_artifacts=tuple(str(path) for path in outcome.reused_artifacts),
            log_path=str(task.log_path),
            failure=None,
            prepared_descriptor=outcome.prepared_descriptor,
        )
    except Exception as error:
        _best_effort_log(
            task.log_path,
            f"{task.key.encoded} failed: {type(error).__name__}: {error}\n"
            f"{traceback.format_exc()}",
        )
        return FineTaskResult(
            task=task,
            status="failed",
            started_at=started_at,
            ended_at=_utc_now(),
            duration_seconds=max(0.0, time.monotonic() - started),
            hostname=hostname,
            pid=pid,
            dask_task_key=dask_key,
            slurm_job_id=slurm_job,
            native_threads=_thread_evidence(task, resolved),
            artifact_paths=(),
            reused_artifacts=(),
            log_path=str(task.log_path),
            failure=RoundFailure(
                stage=task.key.stage,
                exception_type=type(error).__name__,
                message=str(error),
            ),
            prepared_descriptor=None,
        )


__all__ = ("run_fine_task",)
