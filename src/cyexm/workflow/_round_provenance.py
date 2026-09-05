"""Authoritative Phase 1 completion-chain validation for one round task."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from cyexm.registration._request_policy import (
    RAW_RECONSTRUCTED_SCALAR_POLICY,
    affine_parameters,
    canonical_compact_json,
    channel_transform_parameters,
    chunk_grid,
    chunk_manifest_parameters,
    chunk_manifest_payload,
    common_parameters,
    crop_geometry,
    crop_parameters,
    feature_mask_parameters,
    fixed_reference_parameters,
    local_chunk_parameters,
    nonlinear_parameters,
    nonlinear_stage_parameters,
    registered_image_parameters,
    registration_feature_parameters,
    scalar_policy,
    validate_insufficient_chunk_support_parameters,
)
from cyexm.registration.artifacts import CompletionRecord, FileIdentity, StageRequest
from cyexm.runtime import NATIVE_THREAD_ENV_VARS
from cyexm.spatial import ImageGeometry, PixelSize, geometry_compatible
from cyexm.workflow.models import RoundTask

type ArtifactSnapshot = tuple[CompletionRecord, FileIdentity]
type SnapshotReader = Callable[[Path], ArtifactSnapshot | None]
type GeometryReader = Callable[[Path, PixelSize], ImageGeometry]

_GEOMETRY_PARAMETER_NAMES = frozenset(
    {
        "fixed_mask_geometry",
        "moving_geometry",
        "moving_mask_geometry",
    }
)


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
    initial_alignment_transform: Path
    initial_alignment_json: Path
    initial_alignment_qc: Path

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
            initial_alignment_transform=(
                output / f"initial_alignment_{round_number}.mat"
            ),
            initial_alignment_json=output / f"initial_alignment_{round_number}.json",
            initial_alignment_qc=(output / f"initial_alignment_qc_{round_number}.png"),
        )

    @property
    def public_registration_artifacts(self) -> tuple[Path, Path, Path]:
        return self.affine, self.warp, self.registered

    @property
    def initial_alignment_artifacts(self) -> tuple[Path, Path, Path]:
        return (
            self.initial_alignment_transform,
            self.initial_alignment_json,
            self.initial_alignment_qc,
        )


@dataclass(frozen=True, slots=True)
class SourceGeometries:
    fixed: ImageGeometry
    moving: ImageGeometry


@dataclass(frozen=True, slots=True)
class RoundCompletionEvidence:
    """Accepted public artifacts and transform snapshots from one full scan."""

    artifacts: tuple[Path, ...]
    affine: ArtifactSnapshot | None
    warp: ArtifactSnapshot | None


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
    parameters = common_parameters(
        crop=task.registration.crop,
        imaging_round=task.round,
        pixel_size_xyz=task.voxel_spacing_xyz,
    )
    parameters["prepared_fixed"] = task.prepared_fixed
    return parameters


def _registration_scalar_policy(task: RoundTask) -> str:
    if task.registration.preprocessing is not None:
        return RAW_RECONSTRUCTED_SCALAR_POLICY
    return scalar_policy(task.strategy)


def _feature_evidence(
    task: RoundTask,
    snapshot: ArtifactSnapshot | None,
) -> dict[str, object] | None:
    configured = task.registration.preprocessing
    if configured is None:
        return None
    if snapshot is None:
        return None
    value = snapshot[0].request.parameters.get("preprocessing")
    if not isinstance(value, Mapping):
        return None
    fixed_scale = value.get("fixed_robust_scale")
    moving_scale = value.get("moving_robust_scale")
    for scale in (fixed_scale, moving_scale):
        if (
            isinstance(scale, bool)
            or not isinstance(scale, int | float)
            or not math.isfinite(float(scale))
            or float(scale) <= 0.0
        ):
            return None
    expected = registration_feature_parameters(
        filter_sigma_zyx=configured.filter_sigma_zyx,
        fixed_scale=float(fixed_scale),  # type: ignore[arg-type]
        moving_scale=float(moving_scale),  # type: ignore[arg-type]
    )
    configured_mask = configured.feature_mask
    actual_mask = value.get("feature_mask")
    if configured_mask is None:
        if actual_mask is not None:
            return None
    else:
        if not isinstance(actual_mask, Mapping):
            return None
        try:
            canonical_mask = feature_mask_parameters(
                min_component_volume_um3=(configured_mask.min_component_volume_um3),
                dilation_radius_um=configured_mask.dilation_radius_um,
                fixed_foreground_voxels=actual_mask["fixed_foreground_voxels"],  # type: ignore[arg-type]
                fixed_total_voxels=actual_mask["fixed_total_voxels"],  # type: ignore[arg-type]
                moving_foreground_voxels=actual_mask["moving_foreground_voxels"],  # type: ignore[arg-type]
                moving_total_voxels=actual_mask["moving_total_voxels"],  # type: ignore[arg-type]
            )
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return None
        if dict(actual_mask) != canonical_mask:
            return None
        expected["feature_mask"] = canonical_mask
    if not _parameters_match(dict(value), expected):
        return None
    return expected


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
    actual = snapshot[0].request
    return (
        actual.stage == expected.stage
        and actual.inputs == expected.inputs
        and geometry_compatible(actual.geometry, expected.geometry)
        and actual.method == expected.method
        and actual.accuracy == expected.accuracy
        and actual.native_threads == expected.native_threads
        and actual.upstream == expected.upstream
        and _parameters_match(actual.parameters, expected.parameters)
    )


def _parameter_geometry(value: object) -> ImageGeometry | None:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        return None
    try:
        return ImageGeometry.from_dict(dict(value))
    except (TypeError, ValueError):
        return None


def _parameters_match(
    actual: Mapping[str, object],
    expected: Mapping[str, object],
) -> bool:
    if set(actual) != set(expected):
        return False
    for name, expected_value in expected.items():
        actual_value = actual[name]
        if name in _GEOMETRY_PARAMETER_NAMES and isinstance(expected_value, Mapping):
            actual_geometry = _parameter_geometry(actual_value)
            expected_geometry = _parameter_geometry(expected_value)
            if (
                actual_geometry is None
                or expected_geometry is None
                or not geometry_compatible(actual_geometry, expected_geometry)
            ):
                return False
        elif canonical_compact_json(actual_value) != canonical_compact_json(
            expected_value
        ):
            return False
    return True


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
    moving = _crop_snapshot(
        task,
        paths.moving_crop,
        "moving",
        source_geometries.moving,
        inputs,
        snapshot_reader,
    )
    if moving is None:
        return None
    if task.prepared_fixed:
        return SourceGeometries(fixed=source_geometries.fixed, moving=moving[1])
    fixed = _crop_snapshot(
        task,
        paths.fixed_crop,
        "fixed",
        source_geometries.fixed,
        inputs,
        snapshot_reader,
    )
    if fixed is None:
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
    initial_alignment: tuple[dict[str, object], FileIdentity] | None = None,
) -> ArtifactSnapshot | None:
    snapshot = snapshot_reader(path)
    preprocessing = _feature_evidence(task, snapshot)
    if task.registration.preprocessing is not None and preprocessing is None:
        return None
    parameters = affine_parameters(
        moving_geometry=geometries.moving,
        linear_type=task.registration.linear.type,
        accuracy=task.registration.linear.accuracy,
    )
    if preprocessing is not None:
        parameters["preprocessing"] = preprocessing
    if initial_alignment is not None:
        parameters["initial_alignment"] = initial_alignment[0]
    if not _request_matches(
        snapshot,
        stage="affine",
        inputs=inputs,
        geometry=geometries.fixed,
        method=task.registration.linear.type,
        accuracy=task.registration.linear.accuracy,
        task=task,
        upstream=None if initial_alignment is None else initial_alignment[1],
        parameters={
            **_common(task),
            **parameters,
        },
    ):
        return None
    return snapshot


def _validated_initial_alignment_evidence(
    task: RoundTask,
    paths: RegistrationArtifactPaths,
    geometries: SourceGeometries,
    inputs: tuple[FileIdentity, FileIdentity],
    snapshot_reader: SnapshotReader,
) -> tuple[dict[str, object], FileIdentity, bool] | None:
    configured = task.initial_alignment
    if configured is None:
        return None
    from cyexm.workflow.models import _initial_alignment_mapping

    configured_mapping = _initial_alignment_mapping(configured)
    from cyexm.config.models import EstimatedXYRotationSpec, ExplicitXYRotationSpec
    from cyexm.registration.initial_alignment import (
        EXPLICIT_ROTATION_POLICY,
        ROTATION_ESTIMATOR_POLICY,
        EstimatedXYRotationPolicy,
        ExplicitXYRotationPolicy,
        InitialAlignmentPolicy,
        parse_initial_alignment_result,
    )

    configured_rotation = configured.xy_rotation
    if type(configured_rotation) is ExplicitXYRotationSpec:
        algorithm = EXPLICIT_ROTATION_POLICY
        runtime_rotation = ExplicitXYRotationPolicy(configured_rotation.angle_deg)
    elif type(configured_rotation) is EstimatedXYRotationSpec:
        algorithm = ROTATION_ESTIMATOR_POLICY
        runtime_rotation = EstimatedXYRotationPolicy(
            search_range_deg=configured_rotation.search_range_deg,  # type: ignore[union-attr]
            coarse_step_deg=configured_rotation.coarse_step_deg,  # type: ignore[union-attr]
            refinement_step_deg=configured_rotation.refinement_step_deg,  # type: ignore[union-attr]
            minimum_score_gain=configured_rotation.minimum_score_gain,  # type: ignore[union-attr]
            minimum_peak_margin=configured_rotation.minimum_peak_margin,  # type: ignore[union-attr]
        )
    else:  # pragma: no cover - RoundTask validation owns this boundary
        return None
    runtime_policy = InitialAlignmentPolicy(runtime_rotation)
    crop_evidence: dict[str, object] = {"enabled": task.registration.crop}
    if task.registration.crop:
        try:
            text = paths.moving_crop.read_text(encoding="utf-8")
            bounds = json.loads(text)
            if (
                not isinstance(bounds, list)
                or len(bounds) != 6
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in bounds
                )
                or canonical_compact_json(bounds) != text
            ):
                return None
            crop_evidence["moving"] = {
                "bounds": bounds,
                "identity": FileIdentity.from_path(paths.moving_crop).to_dict(),
            }
        except (OSError, TypeError, ValueError):
            return None
        crop_evidence["fixed"] = {"prepared_fixed": True}
    context = {
        "sources": {
            "fixed": inputs[0].to_dict(),
            "moving": inputs[1].to_dict(),
        },
        "geometries": {
            "fixed": geometries.fixed.to_dict(),
            "moving": geometries.moving.to_dict(),
        },
        "crop": crop_evidence,
    }
    evidence_snapshot = snapshot_reader(paths.initial_alignment_json)
    if not _request_matches(
        evidence_snapshot,
        stage="initial-alignment:evidence",
        inputs=inputs,
        geometry=geometries.fixed,
        method="xy-rotation",
        accuracy="deterministic",
        task=task,
        upstream=None,
        parameters={
            **_common(task),
            "configured": configured_mapping,
            "algorithm": algorithm,
            **context,
        },
    ):
        return None
    try:
        evidence = json.loads(paths.initial_alignment_json.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if (
        not isinstance(evidence, dict)
        or set(evidence) != {"configured", "sources", "geometries", "crop", "realized"}
        or evidence.get("configured") != configured_mapping
        or any(evidence.get(key) != value for key, value in context.items())
    ):
        return None
    try:
        realized = parse_initial_alignment_result(
            evidence.get("realized"), runtime_policy
        )
    except (KeyError, OverflowError, TypeError, ValueError):
        return None
    evidence_identity = evidence_snapshot[1]  # type: ignore[index]
    return evidence, evidence_identity, realized.accepted


def _initial_alignment_snapshots(
    task: RoundTask,
    paths: RegistrationArtifactPaths,
    geometries: SourceGeometries,
    inputs: tuple[FileIdentity, FileIdentity],
    snapshot_reader: SnapshotReader,
) -> tuple[dict[str, object], FileIdentity] | None:
    validated = _validated_initial_alignment_evidence(
        task, paths, geometries, inputs, snapshot_reader
    )
    if validated is None:
        return None
    evidence, evidence_identity, accepted = validated
    if not accepted:
        return None
    for path, stage, method, geometry in (
        (
            paths.initial_alignment_qc,
            "initial-alignment:qc",
            "xy-rotation-montage",
            geometries.fixed,
        ),
        (
            paths.initial_alignment_transform,
            "initial-alignment:transform",
            "xy-rotation",
            geometries.moving,
        ),
    ):
        if not _request_matches(
            snapshot_reader(path),
            stage=stage,
            inputs=inputs,
            geometry=geometry,
            method=method,
            accuracy="deterministic" if stage.endswith(":qc") else "exact",
            task=task,
            upstream=evidence_identity,
            parameters={**_common(task), **evidence},
        ):
            return None
    transform = snapshot_reader(paths.initial_alignment_transform)
    if transform is None:
        return None
    return evidence, transform[1]


def initial_alignment_evidence(
    task: RoundTask,
    paths: RegistrationArtifactPaths,
    source_geometries: SourceGeometries,
    *,
    snapshot_reader: SnapshotReader = completion_snapshot,
) -> dict[str, object] | None:
    """Return request-authenticated alignment evidence, including rejections."""
    inputs = _registration_inputs(task)
    if inputs is None:
        return None
    effective = _effective_geometries(
        task,
        paths,
        source_geometries,
        inputs,
        snapshot_reader=snapshot_reader,
    )
    if effective is None:
        return None
    validated = _validated_initial_alignment_evidence(
        task, paths, effective, inputs, snapshot_reader
    )
    return None if validated is None else validated[0]


def initial_alignment_artifacts(
    task: RoundTask,
    paths: RegistrationArtifactPaths,
    source_geometries: SourceGeometries,
    *,
    snapshot_reader: SnapshotReader = completion_snapshot,
) -> tuple[dict[str, object], FileIdentity] | None:
    """Return a request-authenticated accepted alignment artifact bundle."""
    inputs = _registration_inputs(task)
    if inputs is None:
        return None
    effective = _effective_geometries(
        task,
        paths,
        source_geometries,
        inputs,
        snapshot_reader=snapshot_reader,
    )
    if effective is None:
        return None
    return _initial_alignment_snapshots(task, paths, effective, inputs, snapshot_reader)


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
    preprocessing = _feature_evidence(
        task,
        snapshot_reader(Path(affine_identity.path)),
    )
    if task.registration.preprocessing is not None and preprocessing is None:
        return None
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
        parameters = local_chunk_parameters(
            chunk=chunk,
            chunks=grid,
            chunk_size_zyx=task.chunk_size_zyx,
            fixed_mask_geometry=local_geometry,
            image_shape_zyx=geometry.shape,
            moving_geometry=local_geometry,
            moving_mask_geometry=local_geometry,
            overlap_fraction=task.overlap_fraction,
            nonlinear_accuracy=task.registration.nonlinear.accuracy,
            nonlinear_metric=task.registration.nonlinear.metric,
            nonlinear_radius=task.registration.nonlinear.radius,
            scalar_postprocessing_policy=scalar_policy(task.strategy),
        )
        if snapshot is not None:
            fallback = snapshot[0].request.parameters.get("insufficient_support")
            if fallback is not None:
                iterations = nonlinear_parameters(
                    task.registration.nonlinear.accuracy,
                    task.registration.nonlinear.metric,
                    task.registration.nonlinear.radius,
                )["reg_iterations"]
                if not isinstance(iterations, list) or not iterations:
                    return None
                try:
                    parameters["insufficient_support"] = (
                        validate_insufficient_chunk_support_parameters(
                            fallback,
                            expected_shrink_factor=2 ** (len(iterations) - 1),
                        )
                    )
                except (TypeError, ValueError):
                    return None
        if preprocessing is not None:
            parameters["preprocessing"] = preprocessing
        if not _request_matches(
            snapshot,
            stage=f"nonlinear-chunk:{chunk['chunk_id']}",
            inputs=inputs,
            geometry=local_geometry,
            method=task.registration.nonlinear.type,
            accuracy=task.registration.nonlinear.accuracy,
            task=task,
            upstream=affine_identity,
            parameters=parameters,
        ):
            return None
        if snapshot is None or Path(snapshot[1].path) != local_path.resolve():
            return None
        local_identities.append(snapshot[1])

    local_dicts = [identity.to_dict() for identity in local_identities]
    manifest_path = chunk_directory / "manifest.json"
    manifest_snapshot = snapshot_reader(manifest_path)
    manifest_parameters = chunk_manifest_parameters(
        chunk_size_zyx=task.chunk_size_zyx,
        image_shape_zyx=geometry.shape,
        local_warps=local_dicts,
        nonlinear_accuracy=task.registration.nonlinear.accuracy,
        nonlinear_metric=task.registration.nonlinear.metric,
        nonlinear_radius=task.registration.nonlinear.radius,
        nonlinear_type=task.registration.nonlinear.type,
        overlap_fraction=task.overlap_fraction,
    )
    if preprocessing is not None:
        manifest_parameters["preprocessing"] = preprocessing
    if not _request_matches(
        manifest_snapshot,
        stage="nonlinear-chunk-manifest",
        inputs=inputs,
        geometry=geometry,
        method="serial-chunk-grid",
        accuracy="exact",
        task=task,
        upstream=affine_identity,
        parameters=manifest_parameters,
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
    evidence: dict[str, object] = {
        "chunk_grid": [dict(chunk) for chunk in grid],
        "chunk_manifest": manifest_snapshot[1].to_dict(),
        "chunk_size_zyx": list(task.chunk_size_zyx),
        "local_warps": local_dicts,
        "overlap_fraction": task.overlap_fraction,
    }
    if preprocessing is not None:
        evidence["preprocessing"] = preprocessing
    return evidence


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
    preprocessing = _feature_evidence(task, affine)
    if task.registration.preprocessing is not None and preprocessing is None:
        return None
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
    elif preprocessing is not None:
        evidence = {"preprocessing": preprocessing}
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
                nonlinear_metric=task.registration.nonlinear.metric,
                nonlinear_radius=task.registration.nonlinear.radius,
                scalar_postprocessing_policy=_registration_scalar_policy(task),
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
    preprocessing = _feature_evidence(task, warp)
    if task.registration.preprocessing is not None and preprocessing is None:
        return None
    parameters = registered_image_parameters(
        minimum=minimum,
        maximum=maximum,
        scalar_postprocessing_policy=_registration_scalar_policy(task),
    )
    if preprocessing is not None:
        parameters["preprocessing"] = preprocessing
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
            **parameters,
        },
    ):
        return None
    return snapshot


def _channel_snapshot(
    task: RoundTask,
    path: Path,
    channel_index: int,
    geometry: ImageGeometry,
    affine_identity: FileIdentity,
    warp_identity: FileIdentity,
    snapshot_reader: SnapshotReader,
) -> ArtifactSnapshot | None:
    channel = task.channels[channel_index]
    try:
        inputs = (
            FileIdentity.from_path(task.fixed_image),
            FileIdentity.from_path(channel.source),
            affine_identity,
            warp_identity,
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
        upstream=warp_identity,
        parameters=channel_transform_parameters(
            pixel_size_xyz=task.voxel_spacing_xyz,
            interpolation=channel.interpolation,
        ),
    ):
        return None
    return snapshot


def channel_artifact_valid(
    task: RoundTask,
    channel_index: int,
    geometry: ImageGeometry,
    affine: ArtifactSnapshot,
    warp: ArtifactSnapshot,
    *,
    snapshot_reader: SnapshotReader = completion_snapshot,
) -> bool:
    """Validate one channel against an already accepted transform chain."""
    channel = task.channels[channel_index]
    current_affine = snapshot_reader(Path(affine[1].path))
    current_warp = snapshot_reader(Path(warp[1].path))
    return (
        current_affine == affine
        and current_warp == warp
        and _channel_snapshot(
            task,
            channel.output,
            channel_index,
            geometry,
            affine[1],
            warp[1],
            snapshot_reader,
        )
        is not None
    )


def completion_evidence(
    task: RoundTask,
    paths: RegistrationArtifactPaths,
    source_geometries: SourceGeometries,
    *,
    snapshot_reader: SnapshotReader = completion_snapshot,
    geometry_reader: GeometryReader,
) -> RoundCompletionEvidence:
    """Return artifacts and transform snapshots accepted by one full-chain scan."""
    inputs = _registration_inputs(task)
    if inputs is None:
        return RoundCompletionEvidence(artifacts=(), affine=None, warp=None)
    effective = _effective_geometries(
        task,
        paths,
        source_geometries,
        inputs,
        snapshot_reader=snapshot_reader,
    )
    if effective is None:
        return RoundCompletionEvidence(artifacts=(), affine=None, warp=None)
    if not task.prepared_fixed and not _fixed_reference_valid(
        task,
        paths.fixed_reference,
        effective.fixed,
        inputs,
        snapshot_reader=snapshot_reader,
        geometry_reader=geometry_reader,
    ):
        return RoundCompletionEvidence(artifacts=(), affine=None, warp=None)
    complete: list[Path] = []
    initial_alignment = _initial_alignment_snapshots(
        task,
        paths,
        effective,
        inputs,
        snapshot_reader,
    )
    if task.initial_alignment is not None:
        if initial_alignment is None:
            return RoundCompletionEvidence(artifacts=(), affine=None, warp=None)
        complete.extend(paths.initial_alignment_artifacts)
    affine = _affine_snapshot(
        task,
        paths.affine,
        effective,
        inputs,
        snapshot_reader,
        initial_alignment,
    )
    if affine is None:
        return RoundCompletionEvidence(artifacts=(), affine=None, warp=None)
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
        return RoundCompletionEvidence(
            artifacts=tuple(complete),
            affine=affine,
            warp=None,
        )
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
                affine[1],
                warp[1],
                snapshot_reader,
            )
            is not None
        ):
            complete.append(channel.output)
    return RoundCompletionEvidence(
        artifacts=tuple(complete),
        affine=affine,
        warp=warp,
    )


def completed_artifacts(
    task: RoundTask,
    paths: RegistrationArtifactPaths,
    source_geometries: SourceGeometries,
    *,
    snapshot_reader: SnapshotReader = completion_snapshot,
    geometry_reader: GeometryReader,
) -> tuple[Path, ...]:
    """Return public artifacts backed by the exact authoritative Phase 1 chain."""
    return completion_evidence(
        task,
        paths,
        source_geometries,
        snapshot_reader=snapshot_reader,
        geometry_reader=geometry_reader,
    ).artifacts


__all__ = [
    "ArtifactSnapshot",
    "RegistrationArtifactPaths",
    "RoundCompletionEvidence",
    "SourceGeometries",
    "channel_artifact_valid",
    "completed_artifacts",
    "completion_evidence",
    "completion_snapshot",
]
