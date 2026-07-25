from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from cyexm.chunking import ChunkInfo, compute_chunk_grid
from cyexm.errors import (
    CacheValidationError,
    ChunkRegistrationError,
    RegistrationStageError,
)
from cyexm.metrics import read_metrics, write_metrics
from cyexm.registration._native_paths import discard_path
from cyexm.registration._request_policy import (
    CHUNK_SCALAR_POLICY,
    canonical_compact_json,
    chunk_manifest_parameters,
    chunk_manifest_payload,
    common_parameters,
    crop_geometry,
    local_chunk_parameters,
    nonlinear_stage_parameters,
    registered_image_parameters,
)
from cyexm.registration.artifacts import (
    ArtifactStore,
    CompletionRecord,
    FileIdentity,
    StageRequest,
)
from cyexm.registration.backend import RegistrationBackend
from cyexm.spatial import ImageGeometry, PixelSize, geometry_compatible


class ChunkOutputValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ChunkRegion:
    fixed: object
    moving: object
    fixed_mask: object | None
    moving_mask: object | None
    fixed_array: object
    moving_array: object


@dataclass(frozen=True, slots=True)
class ChunkStageContext:
    source_identities: tuple[FileIdentity, ...]
    affine_identity: FileIdentity
    fixed_geometry: ImageGeometry
    fixed: object
    moving_linear: object
    raw_fixed: object
    raw_moving_linear: object
    fixed_mask: object
    moving_linear_mask: object
    preprocessing: dict[str, object] | None
    registered_scalar_policy: str
    nonlinear_type: str
    nonlinear_accuracy: str
    nonlinear_metric: object
    nonlinear_radius: object
    chunks: tuple[ChunkInfo, ...]
    chunk_size: tuple[int, int, int]
    overlap_fraction: float
    native_threads: dict[str, int]
    force: bool
    output_directory: Path
    round_number: int
    pixel_size: PixelSize
    crop: bool
    prepared_fixed: bool
    backend: RegistrationBackend = field(repr=False, compare=False)
    store: ArtifactStore = field(repr=False, compare=False)
    verify_sources: Callable[[], None] = field(repr=False, compare=False)
    reuse_artifact: Callable[
        [Path, StageRequest, Callable[[Path], Any]], Any | None
    ] = field(repr=False, compare=False)
    commit_artifact: Callable[
        [Path, StageRequest, Callable[[Path], None], Callable[[Path], bool]], None
    ] = field(repr=False, compare=False)
    mark_chunk_stale: Callable[[int], None] = field(repr=False, compare=False)
    require_existing: bool = False

    def __post_init__(self) -> None:
        canonical = compute_chunk_grid(
            self.fixed_geometry.shape,
            self.chunk_size,
            self.overlap_fraction,
        )
        if self.chunks != canonical:
            raise ValueError("chunks must equal the canonical legacy chunk grid")

    @property
    def chunk_directory(self) -> Path:
        return self.output_directory / f"round_{self.round_number}_chunks"

    def warp_path(self, chunk_id: int) -> Path:
        return self.chunk_directory / f"chunk_{chunk_id:04d}_warp.nii.gz"

    def scalar_core_path(self, chunk_id: int) -> Path:
        return self.chunk_directory / f"chunk_{chunk_id:04d}_registered_core.npy"

    @property
    def chunk_manifest_path(self) -> Path:
        return self.chunk_directory / "manifest.json"


@dataclass(frozen=True, slots=True)
class ChunkStageArtifacts:
    warp_path: Path
    scalar_core_path: Path
    warp_identity: FileIdentity
    scalar_core_identity: FileIdentity
    reused: bool


@dataclass(frozen=True, slots=True)
class AssembledNonlinear:
    warp_path: Path
    registered_path: Path
    nonlinear_metrics_path: Path
    chunk_manifest_path: Path
    reused: bool
    _manifest_identity: FileIdentity = field(repr=False, compare=False)


def _chunk_dict(chunk: ChunkInfo) -> dict[str, int]:
    return {item.name: int(getattr(chunk, item.name)) for item in fields(chunk)}


def _geometry_compatible(
    np: Any, actual: ImageGeometry, expected: ImageGeometry
) -> bool:
    if actual.shape != expected.shape:
        return False
    return bool(
        np.allclose(actual.origin, expected.origin, rtol=2e-7, atol=1e-6)
        and np.allclose(actual.spacing, expected.spacing, rtol=2e-7, atol=1e-6)
        and np.allclose(actual.direction, expected.direction, rtol=2e-7, atol=1e-6)
    )


def _validated_scalar_array(
    np: Any,
    backend: RegistrationBackend,
    image: object,
    expected_shape: tuple[int, int, int],
    role: str,
    *,
    finite: bool = True,
) -> Any:
    array = np.asarray(backend.to_array(image))
    if array.shape != expected_shape:
        raise ChunkOutputValidationError(
            f"{role} scalar shape {array.shape} does not match {expected_shape}"
        )
    if finite and not bool(np.isfinite(array).all()):
        raise ChunkOutputValidationError(f"{role} scalar values must all be finite")
    return array


def _read_local_warp(
    np: Any,
    backend: RegistrationBackend,
    path: Path,
    expected_geometry: ImageGeometry,
) -> tuple[object, Any]:
    warp = backend.read_warp(path)
    geometry = backend.geometry(warp)
    if not _geometry_compatible(np, geometry, expected_geometry):
        raise ValueError("local warp physical geometry does not match the fixed chunk")
    array = np.asarray(backend.to_array(warp))
    expected_shape = (*expected_geometry.shape, 3)
    if array.shape != expected_shape:
        if array.ndim != 4 or array.shape[-1] != 3:
            raise ValueError("local warp must have exactly three components")
        raise ValueError(
            f"local warp shape {array.shape} does not match {expected_shape}"
        )
    if not bool(np.isfinite(array).all()):
        raise ValueError("local warp values must all be finite")
    return warp, array


def _float32_core(np: Any, array: Any, chunk: ChunkInfo, role: str) -> Any:
    extended = np.asarray(array)
    if extended.shape[:3] != chunk.extended_shape or extended.ndim not in {3, 4}:
        raise ChunkOutputValidationError(
            f"{role} extended chunk shape {extended.shape} does not match "
            f"{chunk.extended_shape}"
        )
    if extended.ndim == 4 and extended.shape[-1] != 3:
        raise ChunkOutputValidationError(
            f"{role} warp data must have exactly three components"
        )
    core = extended[chunk.core_within_extended]
    expected = (*chunk.core_shape, *extended.shape[3:])
    if core.shape != expected:
        raise ChunkOutputValidationError(
            f"{role} core shape {core.shape} does not match {expected}"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        converted = np.asarray(core, dtype=np.float32)
    if not bool(np.isfinite(converted).all()):
        raise ChunkOutputValidationError(
            f"{role} core values must remain finite after float32 conversion"
        )
    return converted


def _local_request(
    context: ChunkStageContext,
    region: ChunkRegion,
    chunk: ChunkInfo,
) -> StageRequest:
    context.verify_sources()
    parameters = local_chunk_parameters(
        chunk=_chunk_dict(chunk),
        chunks=[_chunk_dict(item) for item in context.chunks],
        chunk_size_zyx=context.chunk_size,
        fixed_mask_geometry=(
            None
            if region.fixed_mask is None
            else context.backend.geometry(region.fixed_mask)
        ),
        image_shape_zyx=context.fixed_geometry.shape,
        moving_geometry=context.backend.geometry(region.moving),
        moving_mask_geometry=(
            None
            if region.moving_mask is None
            else context.backend.geometry(region.moving_mask)
        ),
        overlap_fraction=context.overlap_fraction,
        nonlinear_accuracy=context.nonlinear_accuracy,
        nonlinear_metric=context.nonlinear_metric,
        nonlinear_radius=context.nonlinear_radius,
        scalar_postprocessing_policy=CHUNK_SCALAR_POLICY,
    )
    if context.preprocessing is not None:
        parameters["preprocessing"] = context.preprocessing
    return StageRequest.for_identities(
        stage=f"nonlinear-chunk:{chunk.chunk_id}",
        inputs=context.source_identities,
        geometry=context.backend.geometry(region.fixed),
        method=context.nonlinear_type,
        accuracy=context.nonlinear_accuracy,
        native_threads=context.native_threads,
        upstream=context.affine_identity,
        parameters=parameters,
    )


def _scalar_core_request(
    context: ChunkStageContext,
    region: ChunkRegion,
    chunk: ChunkInfo,
    warp_identity: FileIdentity,
) -> StageRequest:
    context.verify_sources()
    parameters: dict[str, object] = {
        "scalar_postprocessing_policy": CHUNK_SCALAR_POLICY
    }
    if context.preprocessing is not None:
        parameters["preprocessing"] = context.preprocessing
    return StageRequest.for_identities(
        stage=f"nonlinear-chunk-scalar-core:{chunk.chunk_id}",
        inputs=context.source_identities,
        geometry=context.backend.geometry(region.fixed),
        method="local-warp-reconstruction",
        accuracy="exact",
        native_threads=context.native_threads,
        upstream=warp_identity,
        parameters=parameters,
    )


def _load_core(np: Any, path: Path, expected_shape: tuple[int, int, int]) -> Any:
    array = np.load(path, allow_pickle=False)
    if array.shape != expected_shape:
        raise ValueError(
            "registered scalar core shape "
            f"{array.shape} does not match {expected_shape}"
        )
    if array.dtype != np.dtype(np.float32):
        raise ValueError("registered scalar core dtype must be exactly float32")
    if not bool(np.isfinite(array).all()):
        raise ValueError("registered scalar core values must all be finite")
    return array


def run_nonlinear_chunk(
    context: ChunkStageContext,
    region: ChunkRegion,
    chunk: ChunkInfo,
) -> ChunkStageArtifacts:
    import numpy as np

    if chunk.chunk_id >= len(context.chunks) or context.chunks[chunk.chunk_id] != chunk:
        raise ValueError("chunk must belong exactly to the canonical legacy grid")
    backend = context.backend
    context.chunk_directory.mkdir(parents=True, exist_ok=True)
    fixed_array = np.asarray(region.fixed_array)
    moving_array = np.asarray(region.moving_array)
    if fixed_array.size == 0 or moving_array.size == 0:
        raise ValueError("extracted scalar chunk must not be empty")
    if not bool(np.isfinite(fixed_array).all()) or not bool(
        np.isfinite(moving_array).all()
    ):
        raise ValueError("fixed and moving chunk values must all be finite")
    low_variance = bool(np.std(fixed_array) < 1e-6 or np.std(moving_array) < 1e-6)
    warp_path = context.warp_path(chunk.chunk_id)
    expected_geometry = backend.geometry(region.fixed)
    request = _local_request(context, region, chunk)

    def read_warp(path: Path) -> tuple[object, Any]:
        return _read_local_warp(np, backend, path, expected_geometry)

    cached_warp = None
    if not context.force:
        cached_warp = context.reuse_artifact(warp_path, request, read_warp)
    warp_reused = cached_warp is not None
    if cached_warp is None:
        if context.require_existing:
            from cyexm.errors import CacheValidationError

            raise CacheValidationError(
                f"Required local chunk cache entry is missing: {warp_path}"
            )
        temporary_warp: Path | None = None
        try:
            if low_variance:

                def writer(path: Path) -> None:
                    backend.write_identity_warp(region.fixed, path)

            else:
                _candidate_scalar, temporary_warp = backend.nonlinear(
                    moving_image=region.moving,
                    fixed_image=region.fixed,
                    fixed_mask=region.fixed_mask,
                    moving_mask=region.moving_mask,
                    registration_type=context.nonlinear_type,
                    accuracy=context.nonlinear_accuracy,
                    metric=context.nonlinear_metric,
                    radius=context.nonlinear_radius,
                    verbose=True,
                )
                read_warp(temporary_warp)

                def writer(path: Path) -> None:
                    if temporary_warp is None:
                        raise RuntimeError("local warp candidate was not prepared")
                    shutil.copyfile(temporary_warp, path)

            context.commit_artifact(
                warp_path,
                request,
                writer,
                lambda path: _valid_reader(read_warp, path),
            )
            try:
                cached_warp = read_warp(warp_path)
            except Exception:
                context.store.mark_stale(warp_path)
                raise
            context.store.mark_stale(context.chunk_manifest_path)
        finally:
            discard_path(temporary_warp)

    warp_identity = FileIdentity.from_path(warp_path)
    scalar_path = context.scalar_core_path(chunk.chunk_id)
    if not warp_reused:
        context.store.mark_stale(scalar_path)
    try:
        core_reused = _load_or_commit_scalar_core(
            np, context, region, chunk, warp_identity
        )
    except ChunkOutputValidationError:
        context.store.mark_stale(scalar_path)
        raise
    except Exception:
        if not warp_reused:
            context.store.mark_stale(warp_path)
        raise
    return ChunkStageArtifacts(
        warp_path=warp_path,
        scalar_core_path=scalar_path,
        warp_identity=warp_identity,
        scalar_core_identity=FileIdentity.from_path(scalar_path),
        reused=warp_reused and core_reused,
    )


def _load_or_commit_scalar_core(
    np: Any,
    context: ChunkStageContext,
    region: ChunkRegion,
    chunk: ChunkInfo,
    warp_identity: FileIdentity,
) -> bool:
    backend = context.backend
    scalar_path = context.scalar_core_path(chunk.chunk_id)
    scalar_request = _scalar_core_request(context, region, chunk, warp_identity)
    cached_core = None
    if not context.force:
        cached_core = context.reuse_artifact(
            scalar_path,
            scalar_request,
            lambda path: _load_core(np, path, chunk.core_shape),
        )
    if cached_core is not None:
        return True
    raw_scalar = backend.apply_warp(
        fixed=region.fixed,
        moving=region.moving,
        warp_path=context.warp_path(chunk.chunk_id),
        interpolation="linear",
    )
    scalar = backend.histogram_match(source=raw_scalar, reference=region.moving)
    scalar_array = _validated_scalar_array(
        np, backend, scalar, chunk.extended_shape, "registered chunk"
    )
    if region.moving_mask is not None:
        raw_mask = backend.apply_warp(
            fixed=region.fixed,
            moving=region.moving_mask,
            warp_path=context.warp_path(chunk.chunk_id),
            interpolation="nearestNeighbor",
        )
        mask = backend.threshold_mask(raw_mask)
        _validated_scalar_array(
            np,
            backend,
            mask,
            chunk.extended_shape,
            "registered mask chunk",
        )
    scalar_core = _float32_core(np, scalar_array, chunk, "registered scalar")

    def write_core(path: Path) -> None:
        with path.open("wb") as stream:
            np.save(stream, scalar_core, allow_pickle=False)

    context.commit_artifact(
        scalar_path,
        scalar_request,
        write_core,
        lambda path: _valid_core(np, path, chunk.core_shape),
    )
    return False


def _valid_reader(reader: Callable[[Path], object], path: Path) -> bool:
    try:
        reader(path)
    except Exception:
        return False
    return True


def _valid_core(np: Any, path: Path, shape: tuple[int, int, int]) -> bool:
    try:
        _load_core(np, path, shape)
    except Exception:
        return False
    return True


def _commit_manifest(
    context: ChunkStageContext,
    artifacts: Sequence[ChunkStageArtifacts],
) -> FileIdentity:
    warp_identities = tuple(item.warp_identity for item in artifacts)
    payload = chunk_manifest_payload(
        round_number=context.round_number,
        image_shape_zyx=context.fixed_geometry.shape,
        chunk_size_zyx=context.chunk_size,
        overlap_fraction=context.overlap_fraction,
        chunks=[_chunk_dict(chunk) for chunk in context.chunks],
        local_warps=[identity.to_dict() for identity in warp_identities],
    )
    text = canonical_compact_json(payload)
    parameters = chunk_manifest_parameters(
        chunk_size_zyx=context.chunk_size,
        image_shape_zyx=context.fixed_geometry.shape,
        local_warps=[identity.to_dict() for identity in warp_identities],
        nonlinear_accuracy=context.nonlinear_accuracy,
        nonlinear_metric=context.nonlinear_metric,
        nonlinear_radius=context.nonlinear_radius,
        nonlinear_type=context.nonlinear_type,
        overlap_fraction=context.overlap_fraction,
    )
    if context.preprocessing is not None:
        parameters["preprocessing"] = context.preprocessing
    request = StageRequest.for_identities(
        stage="nonlinear-chunk-manifest",
        inputs=context.source_identities,
        geometry=context.fixed_geometry,
        method="serial-chunk-grid",
        accuracy="exact",
        native_threads=context.native_threads,
        upstream=context.affine_identity,
        parameters=parameters,
    )

    def reader(path: Path) -> dict[str, object]:
        candidate_text = path.read_text(encoding="utf-8")
        candidate = json.loads(candidate_text)
        if not isinstance(candidate, dict) or canonical_compact_json(candidate) != (
            candidate_text
        ):
            raise ValueError("chunk manifest is not canonical")
        if candidate != payload:
            raise ValueError("chunk manifest contents do not match the grid")
        return candidate

    cached = None
    if not context.force:
        cached = context.reuse_artifact(context.chunk_manifest_path, request, reader)
    if cached is None:
        if context.require_existing:
            from cyexm.errors import CacheValidationError

            raise CacheValidationError(
                "Required chunk manifest cache entry is missing: "
                f"{context.chunk_manifest_path}"
            )

        def writer(path: Path) -> None:
            path.write_text(text, encoding="utf-8")

        context.commit_artifact(
            context.chunk_manifest_path,
            request,
            writer,
            lambda path: _valid_reader(reader, path),
        )
    return FileIdentity.from_path(context.chunk_manifest_path)


def _validate_chunk_records(
    context: ChunkStageContext,
    chunk: ChunkInfo,
    artifact: ChunkStageArtifacts,
) -> None:
    bounds = (
        chunk.z_start_ext,
        chunk.z_end_ext,
        chunk.y_start_ext,
        chunk.y_end_ext,
        chunk.x_start_ext,
        chunk.x_end_ext,
    )
    geometry = crop_geometry(context.fixed_geometry, bounds)
    warp_parameters = local_chunk_parameters(
        chunk=_chunk_dict(chunk),
        chunks=[_chunk_dict(item) for item in context.chunks],
        chunk_size_zyx=context.chunk_size,
        fixed_mask_geometry=geometry,
        image_shape_zyx=context.fixed_geometry.shape,
        moving_geometry=geometry,
        moving_mask_geometry=geometry,
        overlap_fraction=context.overlap_fraction,
        nonlinear_accuracy=context.nonlinear_accuracy,
        nonlinear_metric=context.nonlinear_metric,
        nonlinear_radius=context.nonlinear_radius,
        scalar_postprocessing_policy=CHUNK_SCALAR_POLICY,
    )
    if context.preprocessing is not None:
        warp_parameters["preprocessing"] = context.preprocessing
    expected_warp = StageRequest.for_identities(
        stage=f"nonlinear-chunk:{chunk.chunk_id}",
        inputs=context.source_identities,
        geometry=geometry,
        method=context.nonlinear_type,
        accuracy=context.nonlinear_accuracy,
        native_threads=context.native_threads,
        upstream=context.affine_identity,
        parameters=warp_parameters,
    )
    warp_record = CompletionRecord.read(context.store.record_path(artifact.warp_path))
    if warp_record.request != expected_warp or warp_record.artifact != (
        artifact.warp_identity
    ):
        raise ValueError("local warp completion evidence does not match assembly")
    scalar_parameters: dict[str, object] = {
        "scalar_postprocessing_policy": CHUNK_SCALAR_POLICY
    }
    if context.preprocessing is not None:
        scalar_parameters["preprocessing"] = context.preprocessing
    expected_scalar = StageRequest.for_identities(
        stage=f"nonlinear-chunk-scalar-core:{chunk.chunk_id}",
        inputs=context.source_identities,
        geometry=geometry,
        method="local-warp-reconstruction",
        accuracy="exact",
        native_threads=context.native_threads,
        upstream=artifact.warp_identity,
        parameters=scalar_parameters,
    )
    scalar_record = CompletionRecord.read(
        context.store.record_path(artifact.scalar_core_path)
    )
    if scalar_record.request != expected_scalar or scalar_record.artifact != (
        artifact.scalar_core_identity
    ):
        raise ValueError("scalar core completion evidence does not match assembly")


def _stage_request(
    context: ChunkStageContext,
    image: object,
    *,
    stage: str,
    method: str,
    accuracy: str,
    upstream: FileIdentity | None,
    parameters: dict[str, object],
) -> StageRequest:
    context.verify_sources()
    evidence = common_parameters(
        crop=context.crop,
        imaging_round=context.round_number,
        pixel_size_xyz=context.pixel_size,
    )
    evidence["prepared_fixed"] = context.prepared_fixed
    evidence.update(parameters)
    return StageRequest.for_identities(
        stage=stage,
        inputs=context.source_identities,
        geometry=context.backend.geometry(image),
        method=method,
        accuracy=accuracy,
        native_threads=context.native_threads,
        upstream=upstream,
        parameters=evidence,
    )


def _finalize_assembled_nonlinear(
    context: ChunkStageContext,
    registered_image: object,
    candidate: Path,
    evidence: dict[str, object],
) -> bool:
    backend = context.backend
    output = context.output_directory
    warp_path = output / f"nonlinear_warp_{context.round_number}.nii.gz"
    metrics_path = output / "image_metrics_nonlinear-registered.txt"
    registered_path = output / f"nonlinear_registered_{context.round_number}.tif"
    reused = False
    committed = False
    commit_attempted = False
    try:
        request = _stage_request(
            context,
            context.fixed,
            stage="nonlinear",
            method=context.nonlinear_type,
            accuracy=context.nonlinear_accuracy,
            upstream=context.affine_identity,
            parameters=nonlinear_stage_parameters(
                moving_geometry=context.backend.geometry(context.moving_linear),
                registration_mode="serial-chunked",
                nonlinear_accuracy=context.nonlinear_accuracy,
                nonlinear_metric=context.nonlinear_metric,
                nonlinear_radius=context.nonlinear_radius,
                scalar_postprocessing_policy=context.registered_scalar_policy,
                evidence=evidence,
            ),
        )
        if not context.force:
            reused = (
                context.reuse_artifact(warp_path, request, backend.read_warp)
                is not None
            )
        if not reused:

            def write_warp(path: Path) -> None:
                shutil.copyfile(candidate, path)

            commit_attempted = True
            context.commit_artifact(
                warp_path,
                request,
                write_warp,
                lambda path: _valid_reader(backend.read_warp, path),
            )
            committed = True
        raw_mask = backend.apply_warp(
            fixed=context.raw_fixed,
            moving=context.moving_linear_mask,
            warp_path=warp_path,
            interpolation="nearestNeighbor",
        )
        registered_mask = backend.threshold_mask(raw_mask)
        if context.preprocessing is not None:
            registered_image = backend.apply_warp(
                fixed=context.raw_fixed,
                moving=context.raw_moving_linear,
                warp_path=warp_path,
                interpolation="linear",
            )
    except RegistrationStageError:
        if committed:
            context.store.mark_stale(warp_path)
        raise
    except CacheValidationError as error:
        if committed:
            context.store.mark_stale(warp_path)
        if commit_attempted:
            raise RegistrationStageError(
                f"nonlinear registration failed for round {context.round_number}"
            ) from error
        raise
    except Exception as error:
        if committed:
            context.store.mark_stale(warp_path)
        raise RegistrationStageError(
            f"nonlinear registration failed for round {context.round_number}"
        ) from error
    finally:
        discard_path(candidate)

    try:
        metrics_request = _stage_request(
            context,
            context.raw_fixed,
            stage="metrics:nonlinear-registered",
            method="Correlation+MattesMutualInformation",
            accuracy="full",
            upstream=FileIdentity.from_path(warp_path),
            parameters={
                "moving_geometry": backend.geometry(registered_image).to_dict(),
                "source_moving_geometry": backend.geometry(registered_image).to_dict(),
                "resampled_to_fixed": False,
                "sampling": "regular",
                "sampling_pct": 1.0,
                "scalar_postprocessing_policy": context.registered_scalar_policy,
                **(
                    {}
                    if context.preprocessing is None
                    else {"preprocessing": context.preprocessing}
                ),
            },
        )
        metrics_cached = None
        if not context.force:
            metrics_cached = context.reuse_artifact(
                metrics_path,
                metrics_request,
                read_metrics,
            )
        if metrics_cached is None:
            metrics = backend.calculate_metrics(
                context.raw_fixed,
                registered_image,
                fixed_mask=context.fixed_mask,
                moving_mask=registered_mask,
                sampling="regular",
                sampling_pct=1.0,
            )

            def write_nonlinear_metrics(path: Path) -> None:
                write_metrics(metrics, path)

            context.commit_artifact(
                metrics_path,
                metrics_request,
                write_nonlinear_metrics,
                lambda path: read_metrics(path) == metrics,
            )
    except RegistrationStageError:
        raise
    except Exception as error:
        raise RegistrationStageError(
            "nonlinear-registered metrics stage failed"
        ) from error

    try:
        array: Any = backend.to_array(registered_image)
        minimum, maximum = float(array.min()), float(array.max())
        expected_geometry = backend.geometry(registered_image)
        registered_request = _stage_request(
            context,
            registered_image,
            stage="registered-image",
            method=context.nonlinear_type,
            accuracy=context.nonlinear_accuracy,
            upstream=FileIdentity.from_path(warp_path),
            parameters={
                **registered_image_parameters(
                    minimum=minimum,
                    maximum=maximum,
                    scalar_postprocessing_policy=context.registered_scalar_policy,
                ),
                **(
                    {}
                    if context.preprocessing is None
                    else {"preprocessing": context.preprocessing}
                ),
            },
        )

        def read_registered(path: Path) -> object:
            image = backend.load_image(path, context.pixel_size)
            if not geometry_compatible(backend.geometry(image), expected_geometry):
                raise ValueError(
                    "registered TIFF geometry does not match the nonlinear image"
                )
            return image

        registered_cached = None
        if not context.force:
            registered_cached = context.reuse_artifact(
                registered_path,
                registered_request,
                read_registered,
            )
        if registered_cached is None:

            def write_registered(path: Path) -> None:
                if context.preprocessing is not None:
                    backend.write_clipped_tiff(registered_image, path)
                    return
                backend.write_tiff(
                    registered_image,
                    path,
                    min_value=minimum,
                    max_value=maximum,
                )

            context.commit_artifact(
                registered_path,
                registered_request,
                write_registered,
                lambda path: _valid_reader(read_registered, path),
            )
    except RegistrationStageError:
        raise
    except Exception as error:
        raise RegistrationStageError(
            f"registered image persistence failed for round {context.round_number}"
        ) from error
    return reused


def assemble_nonlinear_chunks(
    context: ChunkStageContext,
    manifest: Sequence[ChunkStageArtifacts],
) -> AssembledNonlinear:
    import numpy as np

    by_warp = {item.warp_identity.path: item for item in manifest}
    ordered = sorted(manifest, key=lambda item: _chunk_id_for_path(item.warp_path))
    if len(ordered) != len(context.chunks) or len(by_warp) != len(ordered):
        raise ValueError(
            "chunk artifacts must contain every expected chunk exactly once"
        )
    registered = np.zeros(context.fixed_geometry.shape, dtype=np.float32)
    displacement = np.zeros((*context.fixed_geometry.shape, 3), dtype=np.float32)
    coverage = np.zeros(context.fixed_geometry.shape, dtype=np.uint32)
    for chunk, artifact in zip(context.chunks, ordered, strict=True):
        try:
            _validate_chunk_records(context, chunk, artifact)
            if _chunk_id_for_path(artifact.warp_path) != chunk.chunk_id:
                raise ValueError("chunk artifact IDs do not match the canonical grid")
            if FileIdentity.from_path(artifact.warp_path) != artifact.warp_identity:
                raise ValueError("local warp identity changed before assembly")
            if FileIdentity.from_path(artifact.scalar_core_path) != (
                artifact.scalar_core_identity
            ):
                raise ValueError("scalar core identity changed before assembly")
            scalar_core = _load_core(np, artifact.scalar_core_path, chunk.core_shape)
            warp = context.backend.read_warp(artifact.warp_path)
            warp_array = np.asarray(context.backend.to_array(warp))
            warp_core = _float32_core(np, warp_array, chunk, "displacement warp")
        except ChunkOutputValidationError as error:
            context.mark_chunk_stale(chunk.chunk_id)
            raise ChunkRegistrationError(chunk.chunk_id, str(error)) from error
        except Exception as error:
            raise ChunkRegistrationError(chunk.chunk_id, str(error)) from error
        coverage[chunk.core_slices] += 1
        registered[chunk.core_slices] = scalar_core
        displacement[chunk.core_slices] = warp_core
    _require_complete_coverage(coverage)
    manifest_identity = _commit_manifest(context, ordered)
    registered_image = context.backend.from_array(registered, reference=context.fixed)
    warp = context.backend.from_array(
        displacement, reference=context.fixed, has_components=True
    )
    descriptor, candidate_name = tempfile.mkstemp(
        dir=context.output_directory,
        prefix=".chunked-full-warp-",
        suffix=".nii.gz",
    )
    os.close(descriptor)
    candidate = Path(candidate_name)
    discard_path(candidate)
    try:
        context.backend.write_warp(warp, candidate)
        read = context.backend.read_warp(candidate)
        if not _geometry_compatible(
            np, context.backend.geometry(read), context.fixed_geometry
        ):
            raise ValueError("assembled warp geometry does not match the fixed image")
        array = np.asarray(context.backend.to_array(read))
        if array.shape != (*context.fixed_geometry.shape, 3):
            raise ValueError("assembled warp must have shape (z, y, x, 3)")
        if array.dtype != np.dtype(np.float32):
            raise ValueError("assembled warp dtype must be exactly float32")
        if not bool(np.isfinite(array).all()):
            raise ValueError("assembled warp values must all be finite")
    except BaseException:
        discard_path(candidate)
        raise
    warp_identities = tuple(item.warp_identity for item in ordered)
    evidence = {
        "chunk_grid": [_chunk_dict(chunk) for chunk in context.chunks],
        "chunk_manifest": manifest_identity.to_dict(),
        "chunk_size_zyx": context.chunk_size,
        "local_warps": [identity.to_dict() for identity in warp_identities],
        "overlap_fraction": context.overlap_fraction,
    }
    if context.preprocessing is not None:
        evidence["preprocessing"] = context.preprocessing
    reused = _finalize_assembled_nonlinear(
        context,
        registered_image,
        candidate,
        evidence,
    )
    output = context.output_directory
    return AssembledNonlinear(
        warp_path=output / f"nonlinear_warp_{context.round_number}.nii.gz",
        registered_path=output / f"nonlinear_registered_{context.round_number}.tif",
        nonlinear_metrics_path=output / "image_metrics_nonlinear-registered.txt",
        chunk_manifest_path=context.chunk_manifest_path,
        reused=reused and all(item.reused for item in ordered),
        _manifest_identity=manifest_identity,
    )


def _require_complete_coverage(coverage: object) -> None:
    import numpy as np

    if not bool(np.all(np.asarray(coverage) >= 1)):
        raise ValueError("chunk core coverage contains a gap")


def _chunk_id_for_path(path: Path) -> int:
    try:
        return int(path.name.split("_", 2)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"invalid chunk artifact path: {path}") from error


__all__ = [
    "AssembledNonlinear",
    "ChunkRegion",
    "ChunkStageArtifacts",
    "ChunkStageContext",
    "assemble_nonlinear_chunks",
    "run_nonlinear_chunk",
]
