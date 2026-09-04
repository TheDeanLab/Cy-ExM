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
from cyexm.workflow.round_task import _initial_alignment


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

    arguments: dict[str, object] = {
        "fixed_image_path": task.fixed_image,
        "moving_image_path": task.moving_image,
        "save_directory": task.registration_directory,
        "imaging_round": task.round,
        "pixel_size": task.voxel_spacing_xyz,
        "crop": task.registration.crop,
        "enable_logging": False,
        "force_override": task.force,
        "prepared_fixed": task.prepared_fixed,
        "native_threads": threads,
        "linear_type": task.registration.linear.type,
        "linear_accuracy": task.registration.linear.accuracy,
        "nonlinear_type": task.registration.nonlinear.type,
        "nonlinear_accuracy": task.registration.nonlinear.accuracy,
        "nonlinear_metric": task.registration.nonlinear.metric,
        "nonlinear_radius": task.registration.nonlinear.radius,
        "preprocess_filter_sigma_zyx": (
            None
            if task.registration.preprocessing is None
            else task.registration.preprocessing.filter_sigma_zyx
        ),
        "preprocess_feature_mask": (
            None
            if task.registration.preprocessing is None
            else task.registration.preprocessing.feature_mask
        ),
    }
    initial_alignment = _initial_alignment(task)
    if initial_alignment is not None:
        arguments["initial_alignment"] = initial_alignment
    registration_type: Any = ImageRegistration
    return registration_type(**arguments)


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


def _durable_progress(
    task: FineWorkflowTask,
    actual_threads: dict[str, int],
    accepted_progress: dict[Path, object],
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    from cyexm.registration.artifacts import (
        AcceptedArtifactProgress,
        ArtifactStore,
        CompletionRecord,
        FileIdentity,
    )

    store = ArtifactStore()
    accepted: list[Path] = []
    reused: list[Path] = []
    for path in task.output_paths:
        try:
            progress = accepted_progress.get(path)
            if type(progress) is not AcceptedArtifactProgress:
                continue
            record_path = store.record_path(path)
            text = record_path.read_text(encoding="utf-8")
            record = CompletionRecord.read(record_path)
            if (
                text != record.to_json()
                or record != progress.record
                or record.artifact != FileIdentity.from_path(path)
                or record.request.native_threads != actual_threads
                or any(
                    FileIdentity.from_path(Path(identity.path)) != identity
                    for identity in record.request.inputs
                )
                or (
                    record.request.upstream is not None
                    and FileIdentity.from_path(Path(record.request.upstream.path))
                    != record.request.upstream
                )
            ):
                continue
        except Exception:
            continue
        accepted.append(path)
        if progress.reused and not task.round_task.force:
            reused.append(path)
    return tuple(accepted), tuple(reused)


def _progress_signature(
    task: FineWorkflowTask,
    path: Path,
) -> tuple[str, str, str] | None:
    from cyexm.registration.prepared import PreparedRoundPaths
    from cyexm.workflow._round_provenance import RegistrationArtifactPaths

    round_task = task.round_task
    registration = RegistrationArtifactPaths.for_round(
        round_task.registration_directory, round_task.round
    )
    common: dict[Path, tuple[str, str, str]] = {
        registration.affine: (
            "affine",
            round_task.registration.linear.type,
            round_task.registration.linear.accuracy,
        ),
        registration.warp: (
            "nonlinear",
            round_task.registration.nonlinear.type,
            round_task.registration.nonlinear.accuracy,
        ),
        registration.registered: (
            "registered-image",
            round_task.registration.nonlinear.type,
            round_task.registration.nonlinear.accuracy,
        ),
        registration.pre_metrics: (
            "metrics:pre-registration",
            "Correlation+MattesMutualInformation",
            "full",
        ),
        registration.linear_metrics: (
            "metrics:linearly-registered",
            "Correlation+MattesMutualInformation",
            "full",
        ),
        registration.nonlinear_metrics: (
            "metrics:nonlinear-registered",
            "Correlation+MattesMutualInformation",
            "full",
        ),
        registration.moving_crop: (
            "crop:moving",
            "robust-bounding-box",
            "legacy-robust",
        ),
    }
    if round_task.initial_alignment is not None:
        common.update(
            {
                registration.initial_alignment_json: (
                    "initial-alignment:evidence",
                    "xy-rotation",
                    "deterministic",
                ),
                registration.initial_alignment_qc: (
                    "initial-alignment:qc",
                    "xy-rotation-montage",
                    "deterministic",
                ),
                registration.initial_alignment_transform: (
                    "initial-alignment:transform",
                    "xy-rotation",
                    "exact",
                ),
            }
        )
    if type(task) is RoundPreparationTask:
        prepared = PreparedRoundPaths.for_round(
            round_task.registration_directory, round_task.round
        )
        roles = (
            "fixed_scalar",
            "moving_linear_scalar",
            "fixed_mask",
            "moving_linear_mask",
        )
        common.update(
            {
                prepared_path: (
                    f"prepared:{role}",
                    round_task.registration.linear.type,
                    round_task.registration.linear.accuracy,
                )
                for prepared_path, role in zip(prepared.array_paths, roles, strict=True)
            }
        )
        common[prepared.manifest] = (
            "prepared:manifest",
            round_task.registration.linear.type,
            round_task.registration.linear.accuracy,
        )
    elif type(task) is NonlinearChunkTask:
        common = {
            task.output_paths[0]: (
                f"nonlinear-chunk:{task.chunk.chunk_id}",
                round_task.registration.nonlinear.type,
                round_task.registration.nonlinear.accuracy,
            ),
            task.output_paths[1]: (
                f"nonlinear-chunk-scalar-core:{task.chunk.chunk_id}",
                "local-warp-reconstruction",
                "exact",
            ),
        }
    elif type(task) is ChunkAssemblyTask:
        common[task.output_paths[3]] = (
            "nonlinear-chunk-manifest",
            "serial-chunk-grid",
            "exact",
        )
    elif type(task) is ChannelTransformTask:
        common = {
            task.output_paths[0]: (
                "channel-transform-v1",
                "warp-affine",
                "not-applicable",
            )
        }
    return common.get(path)


def _accept_progress_for_task(
    task: FineWorkflowTask,
    progress: object,
) -> bool:
    from cyexm.registration.artifacts import AcceptedArtifactProgress

    if type(progress) is not AcceptedArtifactProgress:
        return False
    signature = _progress_signature(task, progress.path)
    request = progress.record.request
    return signature == (request.stage, request.method, request.accuracy)


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


def _validated_prepared_manifest(
    task: RoundTask,
    descriptor: object,
    actual_threads: dict[str, int],
) -> Any:
    import json

    from cyexm.registration._request_policy import (
        PREPARED_SCALAR_POLICY,
        common_parameters,
        feature_mask_config_parameters,
        preprocessing_config_parameters,
        registration_feature_parameters,
    )
    from cyexm.registration.artifacts import (
        ArtifactStore,
        CompletionRecord,
        FileIdentity,
        StageRequest,
    )
    from cyexm.registration.prepared import PreparedManifest, PreparedRoundPaths
    from cyexm.workflow._round_provenance import (
        RegistrationArtifactPaths,
        SourceGeometries,
        initial_alignment_artifacts,
    )
    from cyexm.workflow.round_task import _inspect_image_geometry

    if not isinstance(descriptor, dict):
        descriptor = json.loads(
            json.dumps(descriptor, allow_nan=False, default=lambda value: dict(value))
        )
    if not isinstance(descriptor, dict):
        raise TypeError("prepared descriptor must be a JSON object")
    manifest = PreparedManifest.from_dict(descriptor)
    configured_preprocessing = task.registration.preprocessing
    expected_format = (
        2
        if configured_preprocessing is None
        else (4 if configured_preprocessing.feature_mask is not None else 3)
    )
    if task.initial_alignment is not None:
        expected_format += 3
    if manifest.format_version != expected_format:
        raise ValueError(
            "prepared manifest format does not match registration preprocessing"
        )
    preprocessing: dict[str, object] | None = None
    if configured_preprocessing is not None:
        expected_static = preprocessing_config_parameters(
            configured_preprocessing.filter_sigma_zyx
        )
        actual_static = dict(manifest.preprocessing or {})
        actual_mask = actual_static.pop("feature_mask", None)
        if actual_static != expected_static:
            raise ValueError("prepared preprocessing does not match the embedded round")
        configured_mask = configured_preprocessing.feature_mask
        if configured_mask is None:
            if actual_mask is not None:
                raise ValueError("prepared feature mask is unexpectedly enabled")
        else:
            if not isinstance(actual_mask, dict):
                raise ValueError("prepared feature mask evidence is missing")
            expected_mask = feature_mask_config_parameters(
                min_component_volume_um3=(configured_mask.min_component_volume_um3),
                dilation_radius_um=configured_mask.dilation_radius_um,
            )
            if any(
                actual_mask.get(key) != value for key, value in expected_mask.items()
            ):
                raise ValueError(
                    "prepared feature mask does not match the embedded round"
                )
        if (
            manifest.fixed_feature_scale is None
            or manifest.moving_feature_scale is None
        ):
            raise ValueError("prepared feature scales are missing")
        preprocessing = registration_feature_parameters(
            filter_sigma_zyx=configured_preprocessing.filter_sigma_zyx,
            fixed_scale=manifest.fixed_feature_scale,
            moving_scale=manifest.moving_feature_scale,
        )
        if actual_mask is not None:
            preprocessing["feature_mask"] = dict(actual_mask)
    fixed = FileIdentity.from_path(task.fixed_image)
    moving = FileIdentity.from_path(task.moving_image)
    if (manifest.native_fixed, manifest.native_moving) != (fixed, moving):
        raise ValueError("prepared manifest sources do not match the embedded round")
    prepared = PreparedRoundPaths.for_round(task.registration_directory, task.round)
    for identity, path in zip(
        manifest.array_identities, prepared.array_paths, strict=True
    ):
        if Path(identity.path) != path.resolve(strict=True):
            raise ValueError("prepared array path is not canonical for the round")
    registration = RegistrationArtifactPaths.for_round(
        task.registration_directory, task.round
    )
    initial_alignment: dict[str, object] | None = None
    initial_alignment_identities: tuple[FileIdentity, ...] = ()
    if task.initial_alignment is not None:
        from cyexm.registration.initial_alignment import (
            InitialAlignmentPolicy,
            initial_alignment_config_parameters,
            parse_initial_alignment_result,
        )

        runtime_policy = _initial_alignment(task)
        if type(runtime_policy) is not InitialAlignmentPolicy:
            raise ValueError("prepared initial alignment policy is missing")
        initial_alignment = manifest.initial_alignment
        if not isinstance(initial_alignment, dict) or initial_alignment.get(
            "configured"
        ) != initial_alignment_config_parameters(runtime_policy):
            raise ValueError(
                "prepared initial alignment policy does not match the round"
            )
        parse_initial_alignment_result(
            initial_alignment.get("realized"), runtime_policy
        )
        identity_paths = (
            (manifest.initial_alignment_json, registration.initial_alignment_json),
            (
                manifest.initial_alignment_transform,
                registration.initial_alignment_transform,
            ),
            (manifest.initial_alignment_qc, registration.initial_alignment_qc),
        )
        for identity, path in identity_paths:
            if (
                identity is None
                or Path(identity.path) != path.resolve(strict=True)
                or FileIdentity.from_path(path) != identity
            ):
                raise ValueError("prepared initial alignment identity is not current")
        initial_alignment_identities = tuple(
            identity for identity, _path in identity_paths if identity is not None
        )
        authenticated = initial_alignment_artifacts(
            task,
            registration,
            SourceGeometries(
                fixed=_inspect_image_geometry(task.fixed_image, task.voxel_spacing_xyz),
                moving=_inspect_image_geometry(
                    task.moving_image, task.voxel_spacing_xyz
                ),
            ),
        )
        if (
            authenticated is None
            or authenticated[0] != initial_alignment
            or authenticated[1] != manifest.initial_alignment_transform
        ):
            raise ValueError(
                "prepared initial alignment completion evidence is not current"
            )
    if Path(manifest.affine.path) != registration.affine.resolve(strict=True):
        raise ValueError("prepared affine path is not canonical for the round")
    if FileIdentity.from_path(registration.affine) != manifest.affine:
        raise ValueError("prepared affine identity does not match the round")
    if task.registration.crop:
        if manifest.moving_crop is None:
            raise ValueError("cropped preparation requires moving-crop evidence")
        if (
            Path(manifest.moving_crop.path)
            != registration.moving_crop.resolve(strict=True)
            or FileIdentity.from_path(registration.moving_crop) != manifest.moving_crop
        ):
            raise ValueError("prepared moving crop does not match the round")
    elif manifest.moving_crop is not None:
        raise ValueError("uncropped preparation must not declare a moving crop")
    if manifest.native_threads != dict(sorted(actual_threads.items())):
        raise ValueError(
            "prepared native threads do not match the configured worker runtime"
        )
    if (
        _inspect_image_geometry(task.fixed_image, task.voxel_spacing_xyz)
        != manifest.geometry
    ):
        raise ValueError("prepared geometry does not match the fixed round geometry")
    common = common_parameters(
        crop=task.registration.crop,
        imaging_round=task.round,
        pixel_size_xyz=task.voxel_spacing_xyz,
    )
    store = ArtifactStore()
    for role, identity, path in zip(
        (
            "fixed_scalar",
            "moving_linear_scalar",
            "fixed_mask",
            "moving_linear_mask",
        ),
        manifest.array_identities,
        prepared.array_paths,
        strict=True,
    ):
        parameters = {
            **common,
            "format_version": manifest.format_version,
            "prepared_role": role,
            "scalar_policy": PREPARED_SCALAR_POLICY,
            "mask_policy": manifest.mask_policy,
            "ants_pixeltype": manifest.pixeltypes[role],
            **({} if preprocessing is None else {"preprocessing": preprocessing}),
            **(
                {}
                if initial_alignment is None
                else {"initial_alignment": initial_alignment}
            ),
        }
        expected = StageRequest.for_identities(
            stage=f"prepared:{role}",
            inputs=(fixed, moving),
            geometry=manifest.geometry,
            method=task.registration.linear.type,
            accuracy=task.registration.linear.accuracy,
            native_threads=actual_threads,
            upstream=manifest.affine,
            parameters=parameters,
        )
        record = CompletionRecord.read(store.record_path(path))
        if (
            record.artifact != identity
            or FileIdentity.from_path(path) != identity
            or record.request != expected
        ):
            raise ValueError("prepared completion evidence does not match runtime")
    manifest_identity = FileIdentity.from_path(prepared.manifest)
    manifest_inputs = (
        fixed,
        moving,
        *manifest.array_identities,
        *(() if manifest.moving_crop is None else (manifest.moving_crop,)),
        *initial_alignment_identities,
    )
    expected_manifest = StageRequest.for_identities(
        stage="prepared:manifest",
        inputs=manifest_inputs,
        geometry=manifest.geometry,
        method=task.registration.linear.type,
        accuracy=task.registration.linear.accuracy,
        native_threads=actual_threads,
        upstream=manifest.affine,
        parameters={
            "format_version": manifest.format_version,
            "scalar_policy": PREPARED_SCALAR_POLICY,
            "mask_policy": manifest.mask_policy,
            "pixeltypes": manifest.pixeltypes,
            **({} if preprocessing is None else {"preprocessing": preprocessing}),
            **(
                {}
                if initial_alignment is None
                else {"initial_alignment": initial_alignment}
            ),
        },
    )
    manifest_record = CompletionRecord.read(store.record_path(prepared.manifest))
    if (
        manifest_record.artifact != manifest_identity
        or manifest_record.request != expected_manifest
    ):
        raise ValueError("prepared manifest completion evidence does not match runtime")
    payload = json.loads(prepared.manifest.read_text(encoding="utf-8"))
    if PreparedManifest.from_dict(payload) != manifest:
        raise ValueError("committed prepared manifest does not match the descriptor")
    return manifest


def _chunk_context(
    task: NonlinearChunkTask | ChunkAssemblyTask,
    *,
    assembly: bool,
    actual_threads: dict[str, int],
) -> tuple[Any, Any | None, tuple[Any, ...]]:
    from cyexm.chunking import compute_chunk_grid
    from cyexm.registration._request_policy import (
        CHUNK_SCALAR_POLICY,
        RAW_RECONSTRUCTED_SCALAR_POLICY,
        registration_feature_parameters,
    )
    from cyexm.registration.artifacts import ArtifactStore
    from cyexm.registration.backend import AntsRegistrationBackend
    from cyexm.registration.chunk_stages import ChunkRegion, ChunkStageContext
    from cyexm.registration.crop import read_bounds
    from cyexm.registration.prepared import read_prepared_region

    round_task = task.round_task
    if round_task.chunk_size_zyx is None or round_task.overlap_fraction is None:
        raise ValueError("chunk task requires complete chunk geometry")
    manifest = _validated_prepared_manifest(
        round_task, task.prepared_descriptor, actual_threads
    )
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
        if manifest.moving_crop is not None:
            _verify_identity(manifest.moving_crop, "moving crop")

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
            round_task.registration_directory
            / f"nonlinear_warp_{round_task.round}.nii.gz",
            round_task.registration_directory
            / f"nonlinear_registered_{round_task.round}.tif",
            round_task.registration_directory
            / "image_metrics_nonlinear-registered.txt",
        ):
            store.mark_stale(path)

    if assembly:
        verify_sources()
        raw_fixed = backend.load_image(
            round_task.fixed_image, round_task.voxel_spacing_xyz
        )
        if backend.geometry(raw_fixed) != manifest.geometry:
            raise ValueError("prepared fixed TIFF geometry does not match the manifest")
        raw_moving = backend.load_image(
            round_task.moving_image, round_task.voxel_spacing_xyz
        )
        if round_task.registration.crop:
            from cyexm.workflow._round_provenance import RegistrationArtifactPaths

            registration_paths = RegistrationArtifactPaths.for_round(
                round_task.registration_directory,
                round_task.round,
            )
            bounds = read_bounds(
                registration_paths.moving_crop,
                backend.geometry(raw_moving).shape,
            )
            raw_moving = backend.crop_image(raw_moving, bounds)
        raw_moving_linear = backend.apply_affine(
            fixed=raw_fixed,
            moving=raw_moving,
            transform=backend.read_affine(Path(manifest.affine.path)),
            interpolation="linear",
        )
        if backend.geometry(raw_moving_linear) != manifest.geometry:
            raise ValueError(
                "prepared raw affine result geometry does not match the manifest"
            )
        selected = _whole_chunk(manifest.geometry.shape)
    else:
        if type(task) is not NonlinearChunkTask:
            raise TypeError("regional chunk context requires a nonlinear chunk task")
        selected = task.chunk
    region = read_prepared_region(manifest, selected)  # type: ignore[arg-type]
    fixed_region = backend.from_geometry(
        region.fixed_scalar,
        geometry=region.geometry,
        pixeltype=region.pixeltypes["fixed_scalar"],
    )
    moving_region = backend.from_geometry(
        region.moving_linear_scalar,
        geometry=region.geometry,
        pixeltype=region.pixeltypes["moving_linear_scalar"],
    )
    fixed_mask_region = backend.from_geometry(
        region.fixed_mask,
        geometry=region.geometry,
        pixeltype=region.pixeltypes["fixed_mask"],
    )
    moving_mask_region = backend.from_geometry(
        region.moving_linear_mask,
        geometry=region.geometry,
        pixeltype=region.pixeltypes["moving_linear_mask"],
    )
    preprocessing = None
    if getattr(manifest, "format_version", 2) in {3, 4, 6, 7}:
        configured = round_task.registration.preprocessing
        if (
            configured is None
            or manifest.fixed_feature_scale is None
            or manifest.moving_feature_scale is None
        ):
            raise ValueError("prepared feature evidence is incomplete")
        preprocessing = registration_feature_parameters(
            filter_sigma_zyx=configured.filter_sigma_zyx,
            fixed_scale=manifest.fixed_feature_scale,
            moving_scale=manifest.moving_feature_scale,
        )
        if manifest.format_version in {4, 7}:
            if not isinstance(manifest.preprocessing, dict):
                raise ValueError("prepared feature mask evidence is incomplete")
            feature_mask = manifest.preprocessing.get("feature_mask")
            if not isinstance(feature_mask, dict):
                raise ValueError("prepared feature mask evidence is incomplete")
            preprocessing["feature_mask"] = dict(feature_mask)
    if not assembly:
        raw_fixed = fixed_region
        raw_moving_linear = moving_region
    context = ChunkStageContext(
        source_identities=source_identities,
        affine_identity=manifest.affine,
        fixed_geometry=manifest.geometry,
        fixed=fixed_region,
        moving_linear=moving_region,
        raw_fixed=raw_fixed,
        raw_moving_linear=raw_moving_linear,
        fixed_mask=fixed_mask_region,
        moving_linear_mask=moving_mask_region,
        preprocessing=preprocessing,
        registered_scalar_policy=(
            CHUNK_SCALAR_POLICY
            if preprocessing is None
            else RAW_RECONSTRUCTED_SCALAR_POLICY
        ),
        nonlinear_type=round_task.registration.nonlinear.type,
        nonlinear_accuracy=round_task.registration.nonlinear.accuracy,
        nonlinear_metric=round_task.registration.nonlinear.metric,
        nonlinear_radius=round_task.registration.nonlinear.radius,
        chunks=chunks,
        chunk_size=round_task.chunk_size_zyx,
        overlap_fraction=round_task.overlap_fraction,
        native_threads=dict(actual_threads),
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


def _run_chunk(
    task: NonlinearChunkTask, actual_threads: dict[str, int]
) -> _StageOutcome:
    from cyexm.registration.chunk_stages import run_nonlinear_chunk

    context, region, _chunks = _chunk_context(
        task, assembly=False, actual_threads=actual_threads
    )
    if region is None:
        raise RuntimeError("chunk region was not constructed")
    artifact = run_nonlinear_chunk(context, region, task.chunk)
    paths = (artifact.warp_path, artifact.scalar_core_path)
    return _StageOutcome(
        artifact_paths=paths,
        reused_artifacts=paths if artifact.reused else (),
        prepared_descriptor=None,
    )


def _run_assembly(
    task: ChunkAssemblyTask, actual_threads: dict[str, int]
) -> _StageOutcome:
    from cyexm.registration.artifacts import FileIdentity
    from cyexm.registration.chunk_stages import (
        ChunkStageArtifacts,
        assemble_nonlinear_chunks,
    )

    context, _region, chunks = _chunk_context(
        task, assembly=True, actual_threads=actual_threads
    )
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
        # Validation executes afresh even when every accepted artifact is reused.
        reused_artifacts=(),
        prepared_descriptor=None,
    )


def _execute_fine_task(
    task: FineWorkflowTask,
    resolved: ResolvedNativeThreads,
) -> _StageOutcome:
    if type(task) is RoundPreparationTask:
        return _run_preparation(task)
    if type(task) is NonlinearChunkTask:
        return _run_chunk(task, dict(resolved.values))
    if type(task) is ChunkAssemblyTask:
        return _run_assembly(task, dict(resolved.values))
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
    accepted_progress: dict[Path, object] = {}
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
        from cyexm.registration.artifacts import (
            AcceptedArtifactProgress,
            observe_artifact_progress,
        )

        def accept_progress(progress: AcceptedArtifactProgress) -> None:
            if progress.path in task.output_paths and _accept_progress_for_task(
                task, progress
            ):
                accepted_progress[progress.path] = progress

        with observe_artifact_progress(accept_progress):
            outcome = _execute_fine_task(task, resolved)
        if outcome.artifact_paths != task.output_paths:
            raise RuntimeError(
                "fine task stage returned artifacts other than exact planned outputs"
            )
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
        actual_threads = (
            {
                NATIVE_THREAD_ENV_VARS[0]: task.round_task.native_threads.itk,
                **{
                    name: task.round_task.native_threads.blas
                    for name in NATIVE_THREAD_ENV_VARS[1:]
                },
            }
            if resolved is None
            else dict(resolved.values)
        )
        completed, reused = _durable_progress(task, actual_threads, accepted_progress)
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
            artifact_paths=tuple(str(path) for path in completed),
            reused_artifacts=tuple(str(path) for path in reused),
            log_path=str(task.log_path),
            failure=RoundFailure(
                stage=task.key.stage,
                exception_type=type(error).__name__,
                message=str(error),
            ),
            prepared_descriptor=None,
        )


__all__ = ("run_fine_task",)
