"""Authoritative Phase 1 completion-chain validation for one round task."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from cyexm.registration._request_policy import (
    affine_parameters,
    canonical_compact_json,
    channel_transform_parameters,
    chunk_grid,
    chunk_manifest_parameters,
    chunk_manifest_payload,
    common_parameters,
    crop_geometry,
    crop_parameters,
    fixed_reference_parameters,
    local_chunk_parameters,
    nonlinear_stage_parameters,
    registered_image_parameters,
    scalar_policy,
)
from cyexm.registration.artifacts import CompletionRecord, FileIdentity, StageRequest
from cyexm.runtime import NATIVE_THREAD_ENV_VARS
from cyexm.spatial import ImageGeometry, PixelSize, geometry_compatible
from cyexm.workflow.models import RoundTask

type ArtifactSnapshot = tuple[CompletionRecord, FileIdentity]
type SnapshotReader = Callable[[Path], ArtifactSnapshot | None]
type GeometryReader = Callable[[Path, PixelSize], ImageGeometry]


@dataclass(frozen=True, slots=True)
class RegistrationArtifactPaths:
    """The complete Phase 1 path contract used by a round registration."""

    output: Path
    affine: Path
    warp: Path
    registered: Path
    fixed_reference: Path
    pre_metrics: Path
    linear_metrics: Path
    nonlinear_metrics: Path
    fixed_crop: Path
    moving_crop: Path

    @classmethod
    def for_round(cls, output: Path, round_number: int) -> Self:
        return cls(
            output=output,
            affine=output / f"linear_transform_{round_number}.mat",
            warp=output / f"nonlinear_warp_{round_number}.nii.gz",
            registered=output / f"nonlinear_registered_{round_number}.tif",
            fixed_reference=output / "fixed_reference.tif",
            pre_metrics=output / "image_metrics_pre-registration.txt",
            linear_metrics=output / "image_metrics_linearly-registered.txt",
            nonlinear_metrics=output / "image_metrics_nonlinear-registered.txt",
            fixed_crop=output / f"fixed_crop_indices_{round_number}.json",
            moving_crop=output / f"moving_crop_indices_{round_number}.json",
        )

    @property
    def public_registration_artifacts(self) -> tuple[Path, Path, Path]:
        return self.affine, self.warp, self.registered


@dataclass(frozen=True, slots=True)
class SourceGeometries:
    fixed: ImageGeometry
    moving: ImageGeometry


def completion_snapshot(path: Path) -> ArtifactSnapshot | None:
    """Return a canonical current completion snapshot, or ``None``."""
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


def _common(task: RoundTask) -> dict[str, object]:
    return common_parameters(
        crop=task.registration.crop,
        imaging_round=task.round,
        pixel_size_xyz=task.voxel_spacing_xyz,
    )


def _request_matches(
    snapshot: ArtifactSnapshot | None,
    *,
    stage: str,
    inputs: tuple[FileIdentity, ...],
    geometry: ImageGeometry,
    method: str,
    accuracy: str,
    task: RoundTask,
    upstream: FileIdentity | None,
    parameters: Mapping[str, object],
) -> bool:
    if snapshot is None:
        return False
    expected = StageRequest.for_identities(
        stage=stage,
        inputs=inputs,
        geometry=geometry,
        method=method,
        accuracy=accuracy,
        native_threads=_expected_native_threads(task),
        upstream=upstream,
        parameters=parameters,
    )
    return snapshot[0].request == expected


def _finite_range(parameters: Mapping[str, object]) -> tuple[float, float] | None:
    minimum = parameters.get("minimum")
    maximum = parameters.get("maximum")
    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, int | float)
        or isinstance(maximum, bool)
        or not isinstance(maximum, int | float)
    ):
        return None
    lower = float(minimum)
    upper = float(maximum)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        return None
    return lower, upper


def _read_crop_bounds(
    path: Path, geometry: ImageGeometry
) -> tuple[int, int, int, int, int, int] | None:
    try:
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text)
        if (
            not isinstance(payload, list)
            or len(payload) != 6
            or any(
                isinstance(item, bool) or not isinstance(item, int) for item in payload
            )
        ):
            return None
        bounds = (
            payload[0],
            payload[1],
            payload[2],
            payload[3],
            payload[4],
            payload[5],
        )
        crop_geometry(geometry, bounds)
        if canonical_compact_json(list(bounds)) != text:
            return None
    except (OSError, OverflowError, TypeError, UnicodeError, ValueError):
        return None
    return bounds


def _crop_snapshot(
    task: RoundTask,
    path: Path,
    role: Literal["fixed", "moving"],
    geometry: ImageGeometry,
    inputs: tuple[FileIdentity, FileIdentity],
    snapshot_reader: SnapshotReader,
) -> tuple[ArtifactSnapshot, ImageGeometry] | None:
    snapshot = snapshot_reader(path)
    if not _request_matches(
        snapshot,
        stage=f"crop:{role}",
        inputs=inputs,
        geometry=geometry,
        method="robust-bounding-box",
        accuracy="legacy-robust",
        task=task,
        upstream=None,
        parameters={**_common(task), **crop_parameters(role)},
    ):
        return None
    bounds = _read_crop_bounds(path, geometry)
    if snapshot is None or bounds is None:
        return None
    return snapshot, crop_geometry(geometry, bounds)


def _effective_geometries(
    task: RoundTask,
    paths: RegistrationArtifactPaths,
    source_geometries: SourceGeometries,
    inputs: tuple[FileIdentity, FileIdentity],
    *,
    snapshot_reader: SnapshotReader,
) -> SourceGeometries | None:
    if not task.registration.crop:
        return source_geometries
    fixed = _crop_snapshot(
        task,
        paths.fixed_crop,
        "fixed",
        source_geometries.fixed,
        inputs,
        snapshot_reader,
    )
    moving = _crop_snapshot(
        task,
        paths.moving_crop,
        "moving",
        source_geometries.moving,
        inputs,
        snapshot_reader,
    )
    if fixed is None or moving is None:
        return None
    return SourceGeometries(fixed=fixed[1], moving=moving[1])


def _fixed_reference_valid(
    task: RoundTask,
    path: Path,
    geometry: ImageGeometry,
    inputs: tuple[FileIdentity, FileIdentity],
    *,
    snapshot_reader: SnapshotReader,
    geometry_reader: GeometryReader,
) -> bool:
    snapshot = snapshot_reader(path)
    if snapshot is None:
        return False
    value_range = _finite_range(snapshot[0].request.parameters)
    if value_range is None:
        return False
    minimum, maximum = value_range
    if not _request_matches(
        snapshot,
        stage="fixed-reference",
        inputs=inputs,
        geometry=geometry,
        method="uint16-tiff",
        accuracy="exact",
        task=task,
        upstream=None,
        parameters={
            **_common(task),
            **fixed_reference_parameters(minimum, maximum),
        },
    ):
        return False
    try:
        actual = geometry_reader(path, task.voxel_spacing_xyz)
    except Exception:
        return False
    return geometry_compatible(actual, geometry)


def _affine_snapshot(
    task: RoundTask,
    path: Path,
    geometries: SourceGeometries,
    inputs: tuple[FileIdentity, FileIdentity],
    snapshot_reader: SnapshotReader,
) -> ArtifactSnapshot | None:
    snapshot = snapshot_reader(path)
    if not _request_matches(
        snapshot,
        stage="affine",
        inputs=inputs,
        geometry=geometries.fixed,
        method=task.registration.linear.type,
        accuracy=task.registration.linear.accuracy,
        task=task,
        upstream=None,
        parameters={
            **_common(task),
            **affine_parameters(
                moving_geometry=geometries.moving,
                linear_type=task.registration.linear.type,
                accuracy=task.registration.linear.accuracy,
            ),
        },
    ):
        return None
    return snapshot


def _serial_chunk_evidence(
    task: RoundTask,
    paths: RegistrationArtifactPaths,
    geometry: ImageGeometry,
    inputs: tuple[FileIdentity, FileIdentity],
    affine_identity: FileIdentity,
    *,
    snapshot_reader: SnapshotReader,
) -> dict[str, object] | None:
    if task.chunk_size_zyx is None or task.overlap_fraction is None:
        return None
    grid = chunk_grid(
        geometry.shape,
        task.chunk_size_zyx,
        task.overlap_fraction,
    )
    chunk_directory = paths.output / f"round_{task.round}_chunks"
    local_identities: list[FileIdentity] = []
    for chunk in grid:
        local_path = chunk_directory / f"chunk_{chunk['chunk_id']:04d}_warp.nii.gz"
        snapshot = snapshot_reader(local_path)
        local_geometry = crop_geometry(
            geometry,
            (
                chunk["z_start_ext"],
                chunk["z_end_ext"],
                chunk["y_start_ext"],
                chunk["y_end_ext"],
                chunk["x_start_ext"],
                chunk["x_end_ext"],
            ),
        )
        if not _request_matches(
            snapshot,
            stage=f"nonlinear-chunk:{chunk['chunk_id']}",
            inputs=inputs,
            geometry=local_geometry,
            method=task.registration.nonlinear.type,
            accuracy=task.registration.nonlinear.accuracy,
            task=task,
            upstream=affine_identity,
            parameters=local_chunk_parameters(
                chunk=chunk,
                chunks=grid,
                chunk_size_zyx=task.chunk_size_zyx,
                fixed_mask_geometry=local_geometry,
                image_shape_zyx=geometry.shape,
                moving_geometry=local_geometry,
                moving_mask_geometry=local_geometry,
                overlap_fraction=task.overlap_fraction,
                nonlinear_accuracy=task.registration.nonlinear.accuracy,
                scalar_postprocessing_policy=scalar_policy(task.strategy),
            ),
        ):
            return None
        if snapshot is None or Path(snapshot[1].path) != local_path.resolve():
            return None
        local_identities.append(snapshot[1])

    local_dicts = [identity.to_dict() for identity in local_identities]
    manifest_path = chunk_directory / "manifest.json"
    manifest_snapshot = snapshot_reader(manifest_path)
    if not _request_matches(
        manifest_snapshot,
        stage="nonlinear-chunk-manifest",
        inputs=inputs,
        geometry=geometry,
        method="serial-chunk-grid",
        accuracy="exact",
        task=task,
        upstream=affine_identity,
        parameters=chunk_manifest_parameters(
            chunk_size_zyx=task.chunk_size_zyx,
            image_shape_zyx=geometry.shape,
            local_warps=local_dicts,
            nonlinear_accuracy=task.registration.nonlinear.accuracy,
            nonlinear_type=task.registration.nonlinear.type,
            overlap_fraction=task.overlap_fraction,
        ),
    ):
        return None
    if (
        manifest_snapshot is None
        or Path(manifest_snapshot[1].path) != manifest_path.resolve()
    ):
        return None
    payload = chunk_manifest_payload(
        round_number=task.round,
        image_shape_zyx=geometry.shape,
        chunk_size_zyx=task.chunk_size_zyx,
        overlap_fraction=task.overlap_fraction,
        chunks=grid,
        local_warps=local_dicts,
    )
    try:
        if manifest_path.read_text(encoding="utf-8") != canonical_compact_json(payload):
            return None
    except (OSError, OverflowError, TypeError, UnicodeError, ValueError):
        return None
    return {
        "chunk_grid": [dict(chunk) for chunk in grid],
        "chunk_manifest": manifest_snapshot[1].to_dict(),
        "chunk_size_zyx": list(task.chunk_size_zyx),
        "local_warps": local_dicts,
        "overlap_fraction": task.overlap_fraction,
    }


def _warp_snapshot(
    task: RoundTask,
    path: Path,
    geometry: ImageGeometry,
    inputs: tuple[FileIdentity, FileIdentity],
    affine: ArtifactSnapshot,
    paths: RegistrationArtifactPaths,
    snapshot_reader: SnapshotReader,
) -> ArtifactSnapshot | None:
    evidence: dict[str, object] | None = None
    registration_mode: Literal["whole-image", "serial-chunked"] = "whole-image"
    if task.strategy == "serial_chunked":
        registration_mode = "serial-chunked"
        evidence = _serial_chunk_evidence(
            task,
            paths,
            geometry,
            inputs,
            affine[1],
            snapshot_reader=snapshot_reader,
        )
        if evidence is None:
            return None
    snapshot = snapshot_reader(path)
    if not _request_matches(
        snapshot,
        stage="nonlinear",
        inputs=inputs,
        geometry=geometry,
        method=task.registration.nonlinear.type,
        accuracy=task.registration.nonlinear.accuracy,
        task=task,
        upstream=affine[1],
        parameters={
            **_common(task),
            **nonlinear_stage_parameters(
                moving_geometry=geometry,
                registration_mode=registration_mode,
                nonlinear_accuracy=task.registration.nonlinear.accuracy,
                scalar_postprocessing_policy=scalar_policy(task.strategy),
                evidence=evidence,
            ),
        },
    ):
        return None
    return snapshot


def _registered_snapshot(
    task: RoundTask,
    path: Path,
    geometry: ImageGeometry,
    inputs: tuple[FileIdentity, FileIdentity],
    warp: ArtifactSnapshot,
    snapshot_reader: SnapshotReader,
) -> ArtifactSnapshot | None:
    snapshot = snapshot_reader(path)
    if snapshot is None:
        return None
    value_range = _finite_range(snapshot[0].request.parameters)
    if value_range is None:
        return None
    minimum, maximum = value_range
    if not _request_matches(
        snapshot,
        stage="registered-image",
        inputs=inputs,
        geometry=geometry,
        method=task.registration.nonlinear.type,
        accuracy=task.registration.nonlinear.accuracy,
        task=task,
        upstream=warp[1],
        parameters={
            **_common(task),
            **registered_image_parameters(
                minimum=minimum,
                maximum=maximum,
                scalar_postprocessing_policy=scalar_policy(task.strategy),
            ),
        },
    ):
        return None
    return snapshot


def _channel_snapshot(
    task: RoundTask,
    path: Path,
    channel_index: int,
    geometry: ImageGeometry,
    affine: ArtifactSnapshot,
    warp: ArtifactSnapshot,
    snapshot_reader: SnapshotReader,
) -> ArtifactSnapshot | None:
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
    snapshot = snapshot_reader(path)
    if not _request_matches(
        snapshot,
        stage="channel-transform-v1",
        inputs=inputs,
        geometry=geometry,
        method="warp-affine",
        accuracy="not-applicable",
        task=task,
        upstream=warp[1],
        parameters=channel_transform_parameters(
            pixel_size_xyz=task.voxel_spacing_xyz,
            interpolation=channel.interpolation,
        ),
    ):
        return None
    return snapshot


def completed_artifacts(
    task: RoundTask,
    paths: RegistrationArtifactPaths,
    source_geometries: SourceGeometries,
    *,
    snapshot_reader: SnapshotReader = completion_snapshot,
    geometry_reader: GeometryReader,
) -> tuple[Path, ...]:
    """Return public artifacts backed by the exact authoritative Phase 1 chain."""
    inputs = _registration_inputs(task)
    if inputs is None:
        return ()
    effective = _effective_geometries(
        task,
        paths,
        source_geometries,
        inputs,
        snapshot_reader=snapshot_reader,
    )
    if effective is None or not _fixed_reference_valid(
        task,
        paths.fixed_reference,
        effective.fixed,
        inputs,
        snapshot_reader=snapshot_reader,
        geometry_reader=geometry_reader,
    ):
        return ()
    complete: list[Path] = []
    affine = _affine_snapshot(
        task,
        paths.affine,
        effective,
        inputs,
        snapshot_reader,
    )
    if affine is None:
        return ()
    complete.append(paths.affine)
    warp = _warp_snapshot(
        task,
        paths.warp,
        effective.fixed,
        inputs,
        affine,
        paths,
        snapshot_reader,
    )
    if warp is None:
        return tuple(complete)
    complete.append(paths.warp)
    registered = _registered_snapshot(
        task,
        paths.registered,
        effective.fixed,
        inputs,
        warp,
        snapshot_reader,
    )
    if registered is not None:
        complete.append(paths.registered)
    for index, channel in enumerate(task.channels):
        if (
            _channel_snapshot(
                task,
                channel.output,
                index,
                source_geometries.fixed,
                affine,
                warp,
                snapshot_reader,
            )
            is not None
        ):
            complete.append(channel.output)
    return tuple(complete)


__all__ = [
    "ArtifactSnapshot",
    "RegistrationArtifactPaths",
    "SourceGeometries",
    "completed_artifacts",
    "completion_snapshot",
]
