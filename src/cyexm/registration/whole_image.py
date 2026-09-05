from __future__ import annotations

import json
import logging
import math
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Literal

from cyexm._nonlinear_metric import resolve_nonlinear_metric
from cyexm.config.models import FeatureMaskSpec
from cyexm.errors import (
    CacheValidationError,
    ChunkRegistrationError,
    RegistrationStageError,
)
from cyexm.metrics import read_metrics, write_metrics
from cyexm.registration._native_paths import discard_path
from cyexm.registration._request_policy import (
    CHUNK_SCALAR_POLICY,
    RAW_RECONSTRUCTED_SCALAR_POLICY,
    WHOLE_SCALAR_POLICY,
    affine_parameters,
    canonical_compact_json,
    common_parameters,
    crop_parameters,
    feature_mask_config_parameters,
    feature_mask_parameters,
    fixed_reference_parameters,
    linear_parameters,
    nonlinear_parameters,
    nonlinear_stage_parameters,
    preprocessing_config_parameters,
    registered_image_parameters,
    registration_feature_parameters,
)
from cyexm.registration.artifacts import ArtifactStore, FileIdentity, StageRequest
from cyexm.registration.backend import AntsRegistrationBackend, RegistrationBackend
from cyexm.registration.crop import (
    bounds_text,
    estimate_bounds,
    read_bounds,
    validate_bounds,
)
from cyexm.registration.initial_alignment import (
    EXPLICIT_ROTATION_POLICY,
    ROTATION_ESTIMATOR_POLICY,
    SIMILARITY_ESTIMATOR_POLICY,
    EstimatedXYRotationPolicy,
    EstimatedXYSimilarityPolicy,
    ExplicitXYRotationPolicy,
    InitialAlignmentPolicy,
    XYRotationEstimate,
    estimate_xy_rotation,
    estimate_xy_similarity,
    initial_alignment_config_parameters,
    initial_alignment_result_parameters,
    parse_initial_alignment_result,
    render_initial_alignment_qc,
)
from cyexm.runtime import (
    NativeThreadConfig,
    ResolvedNativeThreads,
    configure_native_threads,
)
from cyexm.spatial import PixelSize, geometry_compatible

logger = logging.getLogger(__name__)

_LINEAR_TYPES = frozenset({"Translation", "Rigid", "Similarity", "Affine", "TRSAA"})
_NONLINEAR_TYPES = frozenset({"Elastic", "SyNOnly", "SyN"})
_ACCURACIES = frozenset({"high", "medium", "low", "dry run"})


def _validated_pixel_size(value: object) -> PixelSize:
    if isinstance(value, str | bytes):
        raise TypeError("pixel_size must contain exactly three numeric values")
    try:
        values = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(
            "pixel_size must contain exactly three numeric values"
        ) from error
    if len(values) != 3:
        raise ValueError("pixel_size must contain exactly three numeric values")
    normalized: list[float] = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, Real):
            raise TypeError("pixel_size values must be positive and finite")
        number = float(item)
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError("pixel_size values must be positive and finite")
        normalized.append(number)
    return normalized[0], normalized[1], normalized[2]


def _validated_choice(value: object, name: str, choices: frozenset[str]) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if value not in choices:
        supported = ", ".join(sorted(choices))
        raise ValueError(f"Unsupported {name}: {value}. Supported values: {supported}")
    return value


def _validated_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _validated_preprocessing_sigma(
    value: tuple[float, float, float] | None,
) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, tuple):
        raise TypeError(
            "preprocess_filter_sigma_zyx must be a canonical three-value tuple"
        )
    if len(value) != 3:
        raise ValueError(
            "preprocess_filter_sigma_zyx must contain exactly three values"
        )
    normalized: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Real):
            raise TypeError(
                "preprocess_filter_sigma_zyx values must be positive and finite"
            )
        number = float(item)
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(
                "preprocess_filter_sigma_zyx values must be positive and finite"
            )
        normalized.append(number)
    return normalized[0], normalized[1], normalized[2]


def _validated_feature_mask(value: object) -> FeatureMaskSpec | None:
    if value is None:
        return None
    if type(value) is not FeatureMaskSpec:
        raise TypeError("preprocess_feature_mask must be a FeatureMaskSpec or None")
    for name in ("min_component_volume_um3", "dilation_radius_um"):
        item = getattr(value, name)
        if type(item) is not float or not math.isfinite(item) or item <= 0.0:
            raise ValueError(
                f"preprocess_feature_mask.{name} must be a positive finite float"
            )
    return value


@dataclass(frozen=True, slots=True)
class RegistrationPaths:
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
    def for_round(cls, output: Path, round_number: int) -> RegistrationPaths:
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


@dataclass(frozen=True, slots=True)
class _LinearPreparation:
    backend: RegistrationBackend
    paths: RegistrationPaths
    resolved: ResolvedNativeThreads
    fixed: object
    moving_linear: object
    raw_fixed: object
    raw_moving_linear: object
    raw_fixed_mask: object
    raw_moving_linear_mask: object
    registration_fixed_mask: object
    registration_moving_linear_mask: object
    affine: object
    fixed_feature_scale: float | None
    moving_feature_scale: float | None


class ImageRegistration:
    """Register one moving image to one fixed image.

    ``preprocess_filter_sigma_zyx`` is an optional canonical three-value tuple in
    NumPy ZYX voxel units. When present, cropped feature images estimate transforms
    that are applied to the raw images. ``preprocess_feature_mask`` can additionally
    derive transform-estimation masks from positive feature support; raw-image masks
    remain diagnostic, and the final registered output remains raw intensity data.
    YAML scalar expansion occurs before this Python API.

    ``initial_alignment`` optionally resolves a physical XY rotation or uniform
    similarity correction from raw cropped registration-channel MIPs before feature
    construction. The seed is supplied once to the existing 3-D linear registrar;
    dependent channels retain the established warp-plus-affine resampling path.

    Parameters
    ----------
    fixed_image_path, moving_image_path : str or path-like
        Fixed and moving scalar 3-D TIFF inputs.
    save_directory : str or path-like
        Dedicated artifact directory, created when needed.
    imaging_round : int, optional
        Nonnegative moving-round identifier used in artifact names.
    pixel_size : tuple of float, optional
        Positive physical spacing in ``(x, y, z)`` micrometers.
    crop : bool, optional
        Estimate and apply foreground crops before registration.
    enable_logging : bool, optional
        Emit stage information through the module logger.
    force_override : bool, optional
        Recompute this request and invalidate its descendants.
    prepared_fixed : bool, optional
        Treat the supplied fixed image as an already prepared reference artifact.
    native_threads : NativeThreadConfig, optional
        Explicit ITK and BLAS-family native thread limits.
    linear_type, linear_accuracy : str, optional
        Supported linear transform family and schedule.
    nonlinear_type, nonlinear_accuracy : str, optional
        Supported nonlinear transform family and schedule.
    nonlinear_metric, nonlinear_radius : object, optional
        Exact nonlinear metric and optional CC voxel radius.
    preprocess_filter_sigma_zyx : tuple of float, optional
        Gaussian feature sigma in NumPy ``(z, y, x)`` voxel units.
    preprocess_feature_mask : FeatureMaskSpec, optional
        Physical component filtering and dilation for feature-derived masks.
    initial_alignment : InitialAlignmentPolicy, optional
        Per-moving-image physical XY rotation or uniform-similarity policy. Positive
        angles follow the right-hand rule about physical +Z and appear clockwise in
        an identity-direction row-down display. Similarity scale is visible uniform
        XY magnification about the moving physical center; Z scale remains one.
        Automatic confidence rejection writes JSON and QC evidence, then raises
        before linear registration.

    Raises
    ------
    TypeError, ValueError
        If configuration does not satisfy the strict public contract.
    RegistrationStageError
        If a registration stage or automatic initialization fails.
    CacheValidationError
        If an existing artifact cannot be authenticated for this request.

    See Also
    --------
    register_round :
        Convenience entry point that runs this complete lifecycle once.
    cyexm.registration.ChunkedImageRegistration :
        Reuse the same affine preparation with serial nonlinear chunks.

    Notes
    -----
    Artifacts and completion records are committed atomically. Omission of
    ``initial_alignment`` preserves the historical request and artifact set.
    """

    def __init__(
        self,
        fixed_image_path: str | os.PathLike[str],
        moving_image_path: str | os.PathLike[str],
        save_directory: str | os.PathLike[str],
        imaging_round: int = 0,
        pixel_size: PixelSize = (0.15, 0.15, 0.15),
        crop: bool = False,
        enable_logging: bool = True,
        force_override: bool = False,
        *,
        prepared_fixed: bool = False,
        native_threads: NativeThreadConfig | None = None,
        linear_type: str = "TRSAA",
        linear_accuracy: str = "low",
        nonlinear_type: str = "SyNOnly",
        nonlinear_accuracy: str = "high",
        nonlinear_metric: object = "mattes",
        nonlinear_radius: object = None,
        preprocess_filter_sigma_zyx: tuple[float, float, float] | None = None,
        preprocess_feature_mask: FeatureMaskSpec | None = None,
        initial_alignment: InitialAlignmentPolicy | None = None,
        _backend: RegistrationBackend | None = None,
        _artifact_store: ArtifactStore | None = None,
    ) -> None:
        if isinstance(imaging_round, bool) or not isinstance(imaging_round, int):
            raise TypeError("imaging_round must be a non-negative integer")
        if imaging_round < 0:
            raise ValueError("imaging_round must be a non-negative integer")
        self.fixed_image_path = fixed_image_path
        self.moving_image_path = moving_image_path
        self.save_directory = save_directory
        self.imaging_round = imaging_round
        self.pixel_size = _validated_pixel_size(pixel_size)
        self.crop = _validated_bool(crop, "crop")
        self.enable_logging = _validated_bool(enable_logging, "enable_logging")
        self.force_override = _validated_bool(force_override, "force_override")
        self.prepared_fixed = _validated_bool(prepared_fixed, "prepared_fixed")
        self.native_threads = native_threads
        self.linear_type = _validated_choice(linear_type, "linear_type", _LINEAR_TYPES)
        self.linear_accuracy = _validated_choice(
            linear_accuracy, "linear_accuracy", _ACCURACIES
        )
        self.nonlinear_type = _validated_choice(
            nonlinear_type, "nonlinear_type", _NONLINEAR_TYPES
        )
        self.nonlinear_accuracy = _validated_choice(
            nonlinear_accuracy, "nonlinear_accuracy", _ACCURACIES
        )
        resolved_metric = resolve_nonlinear_metric(nonlinear_metric, nonlinear_radius)
        self.nonlinear_metric = resolved_metric.metric
        self.nonlinear_radius = resolved_metric.radius
        self.preprocess_filter_sigma_zyx = _validated_preprocessing_sigma(
            preprocess_filter_sigma_zyx
        )
        self.preprocess_feature_mask = _validated_feature_mask(preprocess_feature_mask)
        if (
            initial_alignment is not None
            and type(initial_alignment) is not InitialAlignmentPolicy
        ):
            raise TypeError(
                "initial_alignment must be an InitialAlignmentPolicy or None"
            )
        self.initial_alignment = initial_alignment
        if (
            self.preprocess_feature_mask is not None
            and self.preprocess_filter_sigma_zyx is None
        ):
            raise ValueError(
                "preprocess_feature_mask requires registration preprocessing"
            )
        self._backend = _backend
        self._artifact_store = _artifact_store
        self._log = logger
        self._nonlinear_mode: Literal["whole-image", "serial-chunked"] = "whole-image"
        self._scalar_postprocessing_policy = WHOLE_SCALAR_POLICY
        self._nonlinear_request_evidence: dict[str, object] = {}
        self._registration_feature_evidence: dict[str, object] | None = None
        self._prepare_nonlinear_before_reuse = False
        self._source_paths: tuple[Path, Path] | None = None
        self._source_identities: tuple[FileIdentity, FileIdentity] | None = None
        self._initial_alignment_evidence: dict[str, object] | None = None
        self._initial_transform_path: Path | None = None

    def _prepare_linear_state(self) -> _LinearPreparation:
        resolved = configure_native_threads(self.native_threads)
        backend = self._backend or AntsRegistrationBackend(self.native_threads)
        paths = RegistrationPaths.for_round(
            Path(self.save_directory), self.imaging_round
        )
        paths.output.mkdir(parents=True, exist_ok=True)
        self._artifact_store = self._artifact_store or ArtifactStore()
        if self.enable_logging:
            self._log.info("Native thread settings: %s", resolved.describe())

        self._source_paths = None
        self._source_identities = None
        source_paths = (
            Path(self.fixed_image_path),
            Path(self.moving_image_path),
        )
        source_identities = (
            FileIdentity.from_path(source_paths[0]),
            FileIdentity.from_path(source_paths[1]),
        )
        self._source_paths = source_paths
        self._source_identities = source_identities
        try:
            fixed = backend.load_image(source_paths[0], self.pixel_size)
            moving = backend.load_image(source_paths[1], self.pixel_size)
        except Exception:
            self._verify_source_identities()
            raise
        self._verify_source_identities()
        raw_fixed, raw_moving = self._crop_if_requested(
            backend, fixed, moving, paths, resolved
        )
        self._initial_transform_path = self._load_or_resolve_initial_alignment(
            backend, raw_fixed, raw_moving, paths, resolved
        )
        raw_fixed_mask = backend.zero_mask(raw_fixed)
        raw_moving_mask = backend.zero_mask(raw_moving)

        if not self.prepared_fixed:
            self._commit_fixed_reference(backend, raw_fixed, paths, resolved)
        self._commit_metrics(
            "pre-registration",
            backend,
            raw_fixed,
            raw_moving,
            raw_fixed_mask,
            raw_moving_mask,
            paths,
            resolved,
        )

        fixed = raw_fixed
        moving = raw_moving
        fixed_feature_scale: float | None = None
        moving_feature_scale: float | None = None
        registration_fixed_mask = raw_fixed_mask
        registration_moving_mask = raw_moving_mask
        self._registration_feature_evidence = None
        if self.preprocess_filter_sigma_zyx is not None:
            fixed, fixed_feature_scale = self._registration_feature(
                backend, raw_fixed, "fixed"
            )
            moving, moving_feature_scale = self._registration_feature(
                backend, raw_moving, "moving"
            )
            feature_evidence = registration_feature_parameters(
                filter_sigma_zyx=self.preprocess_filter_sigma_zyx,
                fixed_scale=fixed_feature_scale,
                moving_scale=moving_feature_scale,
            )
            if self.preprocess_feature_mask is not None:
                registration_fixed_mask, fixed_mask_result = self._registration_mask(
                    backend, fixed, "fixed"
                )
                registration_moving_mask, moving_mask_result = self._registration_mask(
                    backend, moving, "moving"
                )
                policy = self.preprocess_feature_mask
                feature_evidence["feature_mask"] = feature_mask_parameters(
                    min_component_volume_um3=policy.min_component_volume_um3,
                    dilation_radius_um=policy.dilation_radius_um,
                    fixed_foreground_voxels=fixed_mask_result.foreground_voxels,
                    fixed_total_voxels=fixed_mask_result.total_voxels,
                    moving_foreground_voxels=moving_mask_result.foreground_voxels,
                    moving_total_voxels=moving_mask_result.total_voxels,
                )
                if self.enable_logging:
                    self._log.info(
                        "Registration feature masks: fixed=%d/%d, moving=%d/%d",
                        fixed_mask_result.foreground_voxels,
                        fixed_mask_result.total_voxels,
                        moving_mask_result.foreground_voxels,
                        moving_mask_result.total_voxels,
                    )
            self._registration_feature_evidence = feature_evidence
            self._scalar_postprocessing_policy = RAW_RECONSTRUCTED_SCALAR_POLICY
            if self.enable_logging:
                self._log.info(
                    "Registration preprocessing robust scales: fixed=%g, moving=%g",
                    fixed_feature_scale,
                    moving_feature_scale,
                )

        affine, moving_linear, moving_linear_mask = self._load_or_estimate_affine(
            backend,
            fixed,
            moving,
            registration_fixed_mask,
            registration_moving_mask,
            paths,
            resolved,
        )
        raw_moving_linear = moving_linear
        raw_moving_linear_mask = moving_linear_mask
        if self.preprocess_filter_sigma_zyx is not None:
            try:
                raw_moving_linear = backend.apply_affine(
                    fixed=raw_fixed,
                    moving=raw_moving,
                    transform=affine,
                    interpolation="linear",
                )
                self._validate_affine_result_geometry(
                    backend,
                    raw_fixed,
                    raw_moving_linear,
                    moving_linear_mask,
                )
            except RegistrationStageError:
                raise
            except Exception as error:
                raise RegistrationStageError(
                    f"raw affine reconstruction failed for round {self.imaging_round}"
                ) from error
        if self.preprocess_feature_mask is not None:
            try:
                raw_moving_linear_mask = backend.threshold_mask(
                    backend.apply_affine(
                        fixed=raw_fixed,
                        moving=raw_moving_mask,
                        transform=affine,
                        interpolation="nearestNeighbor",
                    )
                )
                self._validate_affine_result_geometry(
                    backend,
                    raw_fixed,
                    raw_moving_linear,
                    raw_moving_linear_mask,
                )
            except RegistrationStageError:
                raise
            except Exception as error:
                raise RegistrationStageError(
                    "raw mask affine reconstruction failed "
                    f"for round {self.imaging_round}"
                ) from error
        self._commit_metrics(
            "linearly-registered",
            backend,
            raw_fixed,
            raw_moving_linear,
            raw_fixed_mask,
            raw_moving_linear_mask,
            paths,
            resolved,
        )

        return _LinearPreparation(
            backend=backend,
            paths=paths,
            resolved=resolved,
            fixed=fixed,
            moving_linear=moving_linear,
            raw_fixed=raw_fixed,
            raw_moving_linear=raw_moving_linear,
            raw_fixed_mask=raw_fixed_mask,
            raw_moving_linear_mask=raw_moving_linear_mask,
            registration_fixed_mask=registration_fixed_mask,
            registration_moving_linear_mask=moving_linear_mask,
            affine=affine,
            fixed_feature_scale=fixed_feature_scale,
            moving_feature_scale=moving_feature_scale,
        )

    def register(self) -> None:
        state = self._prepare_linear_state()
        warp_path, moving_nonlinear, moving_nonlinear_mask = (
            self._load_or_estimate_nonlinear(
                state.backend,
                state.fixed,
                state.moving_linear,
                state.registration_fixed_mask,
                state.registration_moving_linear_mask,
                state.affine,
                state.paths,
                state.resolved,
            )
        )
        raw_moving_nonlinear_mask = moving_nonlinear_mask
        if self.preprocess_filter_sigma_zyx is not None:
            try:
                moving_nonlinear = state.backend.apply_warp(
                    fixed=state.raw_fixed,
                    moving=state.raw_moving_linear,
                    warp_path=warp_path,
                    interpolation="linear",
                )
            except Exception as error:
                raise RegistrationStageError(
                    "raw nonlinear reconstruction failed "
                    f"for round {self.imaging_round}"
                ) from error
        if self.preprocess_feature_mask is not None:
            try:
                raw_moving_nonlinear_mask = state.backend.threshold_mask(
                    state.backend.apply_warp(
                        fixed=state.raw_fixed,
                        moving=state.raw_moving_linear_mask,
                        warp_path=warp_path,
                        interpolation="nearestNeighbor",
                    )
                )
            except Exception as error:
                raise RegistrationStageError(
                    "raw mask nonlinear reconstruction failed "
                    f"for round {self.imaging_round}"
                ) from error
        self._commit_metrics(
            "nonlinear-registered",
            state.backend,
            state.raw_fixed,
            moving_nonlinear,
            state.raw_fixed_mask,
            raw_moving_nonlinear_mask,
            state.paths,
            state.resolved,
        )
        self._commit_registered_image(
            state.backend,
            moving_nonlinear,
            state.paths,
            state.resolved,
            warp_path,
        )

    def _registration_feature(
        self,
        backend: RegistrationBackend,
        raw: object,
        role: Literal["fixed", "moving"],
    ) -> tuple[object, float]:
        sigma = self.preprocess_filter_sigma_zyx
        if sigma is None:
            raise RuntimeError("registration preprocessing is disabled")
        try:
            from cyexm.registration.preprocessing import background_subtract_asinh

            result = background_subtract_asinh(backend.to_array(raw), sigma)
            feature = backend.from_array(result.array, reference=raw)
            if not geometry_compatible(
                backend.geometry(feature),
                backend.geometry(raw),
            ):
                raise ValueError(
                    "registration feature geometry does not match its raw source"
                )
            return feature, result.robust_scale
        except Exception as error:
            raise RegistrationStageError(
                f"{role} preprocessing stage failed for round {self.imaging_round}"
            ) from error

    def _registration_mask(
        self,
        backend: RegistrationBackend,
        feature: object,
        role: Literal["fixed", "moving"],
    ) -> tuple[object, Any]:
        policy = self.preprocess_feature_mask
        if policy is None:
            raise RuntimeError("feature-derived registration masking is disabled")
        try:
            from cyexm.registration.preprocessing import feature_mask

            spacing_xyz = backend.geometry(feature).spacing
            spacing_zyx = (
                spacing_xyz[2],
                spacing_xyz[1],
                spacing_xyz[0],
            )
            result = feature_mask(
                backend.to_array(feature),
                spacing_zyx,
                min_component_volume_um3=policy.min_component_volume_um3,
                dilation_radius_um=policy.dilation_radius_um,
            )
            mask = backend.from_array(result.array, reference=feature)
            if not geometry_compatible(
                backend.geometry(mask),
                backend.geometry(feature),
            ):
                raise ValueError(
                    "registration feature mask geometry does not match its feature"
                )
            return mask, result
        except Exception as error:
            raise RegistrationStageError(
                f"{role} feature-mask stage failed for round {self.imaging_round}"
            ) from error

    def _preprocessing_parameters(self, *, derived: bool) -> dict[str, object] | None:
        sigma = self.preprocess_filter_sigma_zyx
        if sigma is None:
            return None
        if derived:
            if self._registration_feature_evidence is None:
                raise RuntimeError("registration feature scales are not available")
            return dict(self._registration_feature_evidence)
        parameters = preprocessing_config_parameters(sigma)
        policy = self.preprocess_feature_mask
        if policy is not None:
            parameters["feature_mask"] = feature_mask_config_parameters(
                min_component_volume_um3=policy.min_component_volume_um3,
                dilation_radius_um=policy.dilation_radius_um,
            )
        return parameters

    def _store(self) -> ArtifactStore:
        if self._artifact_store is None:
            raise RuntimeError("Artifact store is available only during registration")
        return self._artifact_store

    def _input_identity_snapshot(self) -> tuple[FileIdentity, FileIdentity]:
        if self._source_identities is None:
            raise RuntimeError(
                "Source identities are available only during registration"
            )
        return self._source_identities

    def _verify_source_identities(self) -> None:
        if self._source_paths is None:
            raise RuntimeError("Source paths are available only during registration")
        expected_identities = self._input_identity_snapshot()
        for path, expected in zip(
            self._source_paths,
            expected_identities,
            strict=True,
        ):
            try:
                current = FileIdentity.from_path(path)
            except (OSError, RuntimeError) as error:
                validation_error = CacheValidationError(
                    f"Source input is no longer readable: {path}"
                )
                validation_error.__cause__ = error
                raise RegistrationStageError(
                    "source input identity changed during registration "
                    f"for round {self.imaging_round}: {path}"
                ) from validation_error
            if current != expected:
                raise RegistrationStageError(
                    "source input identity changed during registration "
                    f"for round {self.imaging_round}: {path}"
                ) from CacheValidationError(
                    f"Source input identity no longer matches the snapshot: {path}"
                )

    @staticmethod
    def _completion_record_snapshot(
        store: ArtifactStore,
        artifact: Path,
    ) -> tuple[int, int, int, int, bytes] | None:
        try:
            with store.record_path(artifact).open("rb") as stream:
                contents = stream.read()
                metadata = os.fstat(stream.fileno())
        except FileNotFoundError:
            return None
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            contents,
        )

    def _reuse_artifact[T](
        self,
        artifact: Path,
        request: StageRequest,
        reader: Callable[[Path], T],
    ) -> T | None:
        self._verify_source_identities()
        try:
            value = self._store().reuse(artifact, request, reader)
        except Exception:
            self._verify_source_identities()
            raise
        self._verify_source_identities()
        return value

    def _commit_artifact(
        self,
        artifact: Path,
        request: StageRequest,
        writer: Callable[[Path], None],
        validator: Callable[[Path], bool],
    ) -> None:
        store = self._store()
        record_before = self._completion_record_snapshot(store, artifact)
        self._verify_source_identities()
        try:
            store.commit(artifact, request, writer, validator)
        except Exception:
            try:
                self._verify_source_identities()
            except Exception:
                record_after = self._completion_record_snapshot(store, artifact)
                if record_after != record_before:
                    store.mark_stale(artifact)
                raise
            raise
        try:
            self._verify_source_identities()
        except Exception:
            store.mark_stale(artifact)
            raise

    def _request(
        self,
        backend: RegistrationBackend,
        image: object,
        resolved: ResolvedNativeThreads,
        *,
        stage: str,
        method: str,
        accuracy: str,
        upstream: FileIdentity | None,
        parameters: dict[str, object],
    ) -> StageRequest:
        self._verify_source_identities()
        evidence = common_parameters(
            crop=self.crop,
            imaging_round=self.imaging_round,
            pixel_size_xyz=self.pixel_size,
        )
        evidence["prepared_fixed"] = self.prepared_fixed
        evidence.update(parameters)
        return StageRequest.for_identities(
            stage=stage,
            inputs=self._input_identity_snapshot(),
            geometry=backend.geometry(image),
            method=method,
            accuracy=accuracy,
            native_threads=resolved.values,
            upstream=upstream,
            parameters=evidence,
        )

    def _crop_if_requested(
        self,
        backend: RegistrationBackend,
        fixed: object,
        moving: object,
        paths: RegistrationPaths,
        resolved: ResolvedNativeThreads,
    ) -> tuple[object, object]:
        if not self.crop:
            return fixed, moving
        fixed_cropped = fixed
        if not self.prepared_fixed:
            fixed_cropped = self._crop_image(
                backend,
                fixed,
                paths.fixed_crop,
                "fixed",
                resolved,
            )
        moving_cropped = self._crop_image(
            backend,
            moving,
            paths.moving_crop,
            "moving",
            resolved,
        )
        return fixed_cropped, moving_cropped

    _validated_bounds = staticmethod(validate_bounds)
    _bounds_text = staticmethod(bounds_text)
    _read_bounds = staticmethod(read_bounds)

    def _crop_image(
        self,
        backend: RegistrationBackend,
        image: object,
        path: Path,
        role: Literal["fixed", "moving"],
        resolved: ResolvedNativeThreads,
    ) -> object:
        try:
            geometry = backend.geometry(image)
            request = self._request(
                backend,
                image,
                resolved,
                stage=f"crop:{role}",
                method="robust-bounding-box",
                accuracy="legacy-robust",
                upstream=None,
                parameters=crop_parameters(role),
            )
        except RegistrationStageError:
            raise
        except Exception as error:
            raise RegistrationStageError(
                f"{role} crop stage failed for round {self.imaging_round}"
            ) from error
        bounds = None
        if not self.force_override:
            bounds = self._reuse_artifact(
                path,
                request,
                lambda source: self._read_bounds(source, geometry.shape),
            )
        if bounds is None:
            try:
                bounds = estimate_bounds(backend, image)
                text = self._bounds_text(bounds)

                def writer(destination: Path) -> None:
                    destination.write_text(text, encoding="utf-8")

                self._commit_artifact(
                    path,
                    request,
                    writer=writer,
                    validator=lambda candidate: (
                        self._read_bounds(candidate, geometry.shape) == bounds
                    ),
                )
            except RegistrationStageError:
                raise
            except Exception as error:
                raise RegistrationStageError(
                    f"{role} crop stage failed for round {self.imaging_round}"
                ) from error
        try:
            return backend.crop_image(image, bounds)
        except Exception as error:
            raise RegistrationStageError(
                f"{role} crop stage failed for round {self.imaging_round}"
            ) from error

    @staticmethod
    def _image_range(
        backend: RegistrationBackend, image: object
    ) -> tuple[float, float]:
        array: Any = backend.to_array(image)
        return float(array.min()), float(array.max())

    def _linear_parameters(self) -> dict[str, object]:
        return linear_parameters(self.linear_type, self.linear_accuracy)

    def _nonlinear_parameters(self) -> dict[str, object]:
        return nonlinear_parameters(
            self.nonlinear_accuracy,
            self.nonlinear_metric,
            self.nonlinear_radius,
        )

    def _commit_fixed_reference(
        self,
        backend: RegistrationBackend,
        fixed: object,
        paths: RegistrationPaths,
        resolved: ResolvedNativeThreads,
    ) -> None:
        try:
            minimum, maximum = self._image_range(backend, fixed)
            expected_geometry = backend.geometry(fixed)
            request = self._request(
                backend,
                fixed,
                resolved,
                stage="fixed-reference",
                method="uint16-tiff",
                accuracy="exact",
                upstream=None,
                parameters=fixed_reference_parameters(minimum, maximum),
            )
        except RegistrationStageError:
            raise
        except Exception as error:
            raise RegistrationStageError(
                f"fixed reference persistence failed for round {self.imaging_round}"
            ) from error

        def reader(path: Path) -> object:
            image = backend.load_image(path, self.pixel_size)
            if not geometry_compatible(backend.geometry(image), expected_geometry):
                raise ValueError(
                    "fixed reference TIFF geometry does not match the source image"
                )
            return image

        if not self.force_override:
            cached = self._reuse_artifact(paths.fixed_reference, request, reader)
            if cached is not None:
                return

        def validator(path: Path) -> bool:
            try:
                reader(path)
            except Exception:
                return False
            return True

        try:
            self._commit_artifact(
                paths.fixed_reference,
                request,
                writer=lambda path: backend.write_tiff(
                    fixed, path, min_value=minimum, max_value=maximum
                ),
                validator=validator,
            )
        except RegistrationStageError:
            raise
        except Exception as error:
            raise RegistrationStageError(
                f"fixed reference persistence failed for round {self.imaging_round}"
            ) from error

    def _commit_metrics(
        self,
        stage: str,
        backend: RegistrationBackend,
        fixed: object,
        moving: object,
        fixed_mask: object,
        moving_mask: object,
        paths: RegistrationPaths,
        resolved: ResolvedNativeThreads,
    ) -> None:
        path_by_stage = {
            "pre-registration": paths.pre_metrics,
            "linearly-registered": paths.linear_metrics,
            "nonlinear-registered": paths.nonlinear_metrics,
        }
        try:
            source_moving_geometry = backend.geometry(moving)
            resampled_to_fixed = not geometry_compatible(
                backend.geometry(fixed), source_moving_geometry
            )
            if resampled_to_fixed:
                moving = backend.resample_to_target(
                    image=moving,
                    target=fixed,
                    interpolation="linear",
                )
                moving_mask = backend.threshold_mask(
                    backend.resample_to_target(
                        image=moving_mask,
                        target=fixed_mask,
                        interpolation="nearestNeighbor",
                    )
                )
            upstream = None
            if stage == "linearly-registered":
                upstream = FileIdentity.from_path(paths.affine)
            elif stage == "nonlinear-registered":
                upstream = FileIdentity.from_path(paths.warp)
            parameters: dict[str, object] = {
                "moving_geometry": backend.geometry(moving).to_dict(),
                "source_moving_geometry": source_moving_geometry.to_dict(),
                "resampled_to_fixed": resampled_to_fixed,
                "sampling": "regular",
                "sampling_pct": 1.0,
            }
            if stage == "nonlinear-registered":
                parameters["scalar_postprocessing_policy"] = (
                    self._scalar_postprocessing_policy
                )
            preprocessing = self._preprocessing_parameters(
                derived=stage != "pre-registration"
            )
            if preprocessing is not None:
                parameters["preprocessing"] = preprocessing
            if self._initial_alignment_evidence is not None:
                parameters["initial_alignment"] = self._initial_alignment_evidence
            request = self._request(
                backend,
                fixed,
                resolved,
                stage=f"metrics:{stage}",
                method="Correlation+MattesMutualInformation",
                accuracy="full",
                upstream=upstream,
                parameters=parameters,
            )
        except RegistrationStageError:
            raise
        except Exception as error:
            raise RegistrationStageError(f"{stage} metrics stage failed") from error
        destination = path_by_stage[stage]
        if not self.force_override:
            cached = self._reuse_artifact(destination, request, read_metrics)
            if cached is not None:
                return
        try:
            metrics = backend.calculate_metrics(
                fixed,
                moving,
                fixed_mask=fixed_mask,
                moving_mask=moving_mask,
                sampling="regular",
                sampling_pct=1.0,
            )
            self._commit_artifact(
                destination,
                request,
                writer=lambda path: write_metrics(metrics, path),
                validator=lambda path: read_metrics(path) == metrics,
            )
        except RegistrationStageError:
            raise
        except Exception as error:
            raise RegistrationStageError(f"{stage} metrics stage failed") from error

    def _load_or_resolve_initial_alignment(
        self,
        backend: RegistrationBackend,
        fixed: object,
        moving: object,
        paths: RegistrationPaths,
        resolved: ResolvedNativeThreads,
    ) -> Path | None:
        policy = self.initial_alignment
        self._initial_alignment_evidence = None
        if policy is None:
            return None
        configured = initial_alignment_config_parameters(policy)
        rotation = policy.xy_rotation
        similarity = policy.xy_similarity
        if type(rotation) is ExplicitXYRotationPolicy:
            algorithm = EXPLICIT_ROTATION_POLICY
            method = "xy-rotation"
        elif type(rotation) is EstimatedXYRotationPolicy:
            algorithm = ROTATION_ESTIMATOR_POLICY
            method = "xy-rotation"
        elif type(similarity) is EstimatedXYSimilarityPolicy:
            algorithm = SIMILARITY_ESTIMATOR_POLICY
            method = "xy-similarity"
        else:  # pragma: no cover - InitialAlignmentPolicy owns this invariant
            raise TypeError("unsupported initial alignment policy")
        source = (
            "explicit" if type(rotation) is ExplicitXYRotationPolicy else "estimated"
        )
        fixed_geometry = backend.geometry(fixed)
        moving_geometry = backend.geometry(moving)
        if type(rotation) is ExplicitXYRotationPolicy and any(
            not math.isclose(fixed_value, moving_value, rel_tol=0.0, abs_tol=1e-8)
            for fixed_row, moving_row in zip(
                fixed_geometry.direction, moving_geometry.direction, strict=True
            )
            for fixed_value, moving_value in zip(fixed_row, moving_row, strict=True)
        ):
            raise RegistrationStageError(
                "explicit XY rotation requires matching fixed and moving "
                "direction matrices"
            )
        fixed_source, moving_source = self._input_identity_snapshot()
        crop_evidence: dict[str, object] = {"enabled": self.crop}
        if self.crop:

            def crop_artifact(path: Path) -> dict[str, object]:
                text = path.read_text(encoding="utf-8")
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
                    raise ValueError("initial alignment crop evidence is invalid")
                return {
                    "bounds": bounds,
                    "identity": FileIdentity.from_path(path).to_dict(),
                }

            crop_evidence["moving"] = {
                **crop_artifact(paths.moving_crop),
            }
            crop_evidence["fixed"] = (
                {"prepared_fixed": True}
                if self.prepared_fixed
                else crop_artifact(paths.fixed_crop)
            )
        context: dict[str, object] = {
            "sources": {
                "fixed": fixed_source.to_dict(),
                "moving": moving_source.to_dict(),
            },
            "geometries": {
                "fixed": fixed_geometry.to_dict(),
                "moving": moving_geometry.to_dict(),
            },
            "crop": crop_evidence,
        }
        evidence_request = self._request(
            backend,
            fixed,
            resolved,
            stage="initial-alignment:evidence",
            method=method,
            accuracy="deterministic",
            upstream=None,
            parameters={
                "configured": configured,
                "algorithm": algorithm,
                **context,
            },
        )

        def read_evidence(path: Path) -> dict[str, object]:
            value = json.loads(path.read_text(encoding="utf-8"))
            if (
                not isinstance(value, dict)
                or set(value)
                != {"configured", "sources", "geometries", "crop", "realized"}
                or value.get("configured") != configured
                or any(value.get(key) != expected for key, expected in context.items())
            ):
                raise ValueError("initial alignment evidence is not canonical")
            result = parse_initial_alignment_result(value.get("realized"), policy)
            if (
                result.policy != algorithm
                or value["realized"].get("source") != source
                or initial_alignment_result_parameters(result) != value.get("realized")
            ):
                raise ValueError("initial alignment result is not canonical")
            if path.read_text(encoding="utf-8") != canonical_compact_json(value):
                raise ValueError("initial alignment evidence JSON is not canonical")
            return value

        evidence: dict[str, object] | None = None
        if not self.force_override:
            evidence = self._reuse_artifact(
                paths.initial_alignment_json, evidence_request, read_evidence
            )
        if evidence is None:
            if type(rotation) is ExplicitXYRotationPolicy:
                result = XYRotationEstimate(
                    policy=EXPLICIT_ROTATION_POLICY,
                    accepted=True,
                    rejection_reason=None,
                    angle_deg=rotation.angle_deg,
                    translation_yx_px=(0.0, 0.0),
                    zero_angle_score=None,
                    score_gain=None,
                    peak_margin=None,
                    valid_overlap_fraction=None,
                    coarse_scores=(),
                    refined_scores=(),
                )
            elif type(rotation) is EstimatedXYRotationPolicy:
                result = estimate_xy_rotation(
                    backend.to_array(fixed),
                    backend.to_array(moving),
                    rotation,
                    fixed_geometry=fixed_geometry,
                    moving_geometry=moving_geometry,
                )
            elif type(similarity) is EstimatedXYSimilarityPolicy:
                result = estimate_xy_similarity(
                    backend.to_array(fixed),
                    backend.to_array(moving),
                    similarity,
                    fixed_geometry=fixed_geometry,
                    moving_geometry=moving_geometry,
                )
            else:  # pragma: no cover - protected by InitialAlignmentPolicy
                raise TypeError("unsupported initial alignment policy")
            evidence = {
                "configured": configured,
                **context,
                "realized": initial_alignment_result_parameters(result),
            }

            def write_evidence(path: Path) -> None:
                path.write_text(canonical_compact_json(evidence), encoding="utf-8")

            self._commit_artifact(
                paths.initial_alignment_json,
                evidence_request,
                writer=write_evidence,
                validator=lambda path: read_evidence(path) == evidence,
            )
        result = parse_initial_alignment_result(evidence["realized"], policy)
        evidence_identity = FileIdentity.from_path(paths.initial_alignment_json)
        qc_request = self._request(
            backend,
            fixed,
            resolved,
            stage="initial-alignment:qc",
            method=f"{method}-montage",
            accuracy="deterministic",
            upstream=evidence_identity,
            parameters=evidence,
        )

        def read_qc(path: Path) -> tuple[str, tuple[int, int]]:
            from PIL import Image

            with Image.open(path) as image:
                image.load()
                return image.mode, image.size

        if (
            self.force_override
            or self._reuse_artifact(paths.initial_alignment_qc, qc_request, read_qc)
            is None
        ):
            qc = render_initial_alignment_qc(
                backend.to_array(fixed),
                backend.to_array(moving),
                result,
                fixed_geometry=backend.geometry(fixed),
                moving_geometry=backend.geometry(moving),
            )

            def write_qc(path: Path) -> None:
                from PIL import Image

                Image.fromarray(qc, mode="L").save(path, format="PNG")

            self._commit_artifact(
                paths.initial_alignment_qc,
                qc_request,
                writer=write_qc,
                validator=lambda path: read_qc(path) == ("L", (768, 512)),
            )
        self._initial_alignment_evidence = evidence
        if not result.accepted or result.angle_deg is None:
            raise RegistrationStageError(
                "initial-alignment rejected for round "
                f"{self.imaging_round}; inspect {paths.initial_alignment_json}"
            )
        transform_request = self._request(
            backend,
            moving,
            resolved,
            stage="initial-alignment:transform",
            method=method,
            accuracy="exact",
            upstream=evidence_identity,
            parameters=evidence,
        )
        if not self.force_override:
            cached = self._reuse_artifact(
                paths.initial_alignment_transform,
                transform_request,
                backend.read_affine,
            )
            if cached is not None:
                return paths.initial_alignment_transform
        if type(similarity) is EstimatedXYSimilarityPolicy:
            if getattr(result, "scale", None) is None:
                raise RegistrationStageError("accepted XY similarity has no scale")
            transform = backend.create_xy_similarity_seed(
                fixed_geometry=fixed_geometry,
                geometry=moving_geometry,
                angle_deg=result.angle_deg,
                scale=result.scale,  # type: ignore[union-attr]
                translation_yx_px=result.translation_yx_px or (0.0, 0.0),
            )
        else:
            transform = backend.create_xy_rotation_seed(
                fixed_geometry=fixed_geometry,
                geometry=moving_geometry,
                angle_deg=result.angle_deg,
                translation_yx_px=result.translation_yx_px or (0.0, 0.0),
            )
        self._commit_artifact(
            paths.initial_alignment_transform,
            transform_request,
            writer=lambda path: backend.write_affine(transform, path),
            validator=lambda path: bool(backend.read_affine(path)),
        )
        return paths.initial_alignment_transform

    def _load_or_estimate_affine(
        self,
        backend: RegistrationBackend,
        fixed: object,
        moving: object,
        fixed_mask: object,
        moving_mask: object,
        paths: RegistrationPaths,
        resolved: ResolvedNativeThreads,
    ) -> tuple[object, object, object]:
        try:
            parameters = affine_parameters(
                moving_geometry=backend.geometry(moving),
                linear_type=self.linear_type,
                accuracy=self.linear_accuracy,
            )
            preprocessing = self._preprocessing_parameters(derived=True)
            if preprocessing is not None:
                parameters["preprocessing"] = preprocessing
            if self._initial_alignment_evidence is not None:
                parameters["initial_alignment"] = self._initial_alignment_evidence
            request = self._request(
                backend,
                fixed,
                resolved,
                stage="affine",
                method=self.linear_type,
                accuracy=self.linear_accuracy,
                upstream=(
                    None
                    if self._initial_transform_path is None
                    else FileIdentity.from_path(self._initial_transform_path)
                ),
                parameters=parameters,
            )
        except RegistrationStageError:
            raise
        except Exception as error:
            raise RegistrationStageError(
                f"affine registration failed for round {self.imaging_round}"
            ) from error
        if not self.force_override:
            transform = self._reuse_artifact(paths.affine, request, backend.read_affine)
            if transform is not None:
                try:
                    moving_linear = backend.apply_affine(
                        fixed=fixed,
                        moving=moving,
                        transform=transform,
                        interpolation="linear",
                    )
                    moving_linear_mask = backend.threshold_mask(
                        backend.apply_affine(
                            fixed=fixed,
                            moving=moving_mask,
                            transform=transform,
                            interpolation="nearestNeighbor",
                        )
                    )
                except Exception as error:
                    raise RegistrationStageError(
                        "cached affine reconstruction failed "
                        f"for round {self.imaging_round}"
                    ) from error
                self._validate_affine_result_geometry(
                    backend,
                    fixed,
                    moving_linear,
                    moving_linear_mask,
                )
                return transform, moving_linear, moving_linear_mask

        try:
            linear_arguments: dict[str, object] = {
                "moving_image": moving,
                "fixed_image": fixed,
                "fixed_mask": fixed_mask,
                "moving_mask": moving_mask,
                "registration_type": self.linear_type,
                "accuracy": self.linear_accuracy,
                "verbose": True,
            }
            if self._initial_transform_path is not None:
                linear_arguments["initial_transform"] = self._initial_transform_path
            moving_linear, moving_linear_mask, transform = backend.linear(
                **linear_arguments
            )
            if moving_linear_mask is None:
                moving_linear_mask = backend.threshold_mask(
                    backend.apply_affine(
                        fixed=fixed,
                        moving=moving_mask,
                        transform=transform,
                        interpolation="nearestNeighbor",
                    )
                )
            self._validate_affine_result_geometry(
                backend,
                fixed,
                moving_linear,
                moving_linear_mask,
            )

            def validator(path: Path) -> bool:
                try:
                    backend.read_affine(path)
                except Exception:
                    return False
                return True

            self._commit_artifact(
                paths.affine,
                request,
                writer=lambda path: backend.write_affine(transform, path),
                validator=validator,
            )
            if self.force_override:
                for descendant in (
                    paths.linear_metrics,
                    paths.warp,
                    paths.nonlinear_metrics,
                    paths.registered,
                ):
                    self._store().mark_stale(descendant)
        except RegistrationStageError:
            raise
        except Exception as error:
            raise RegistrationStageError(
                f"affine registration failed for round {self.imaging_round}"
            ) from error
        return transform, moving_linear, moving_linear_mask

    def _validate_affine_result_geometry(
        self,
        backend: RegistrationBackend,
        fixed: object,
        moving_linear: object,
        moving_linear_mask: object,
    ) -> None:
        fixed_geometry = backend.geometry(fixed)
        for role, image in (
            ("scalar", moving_linear),
            ("mask", moving_linear_mask),
        ):
            if not geometry_compatible(backend.geometry(image), fixed_geometry):
                raise RegistrationStageError(
                    f"affine result geometry for {role} does not match the fixed "
                    f"reference for round {self.imaging_round}"
                )

    def _load_or_estimate_nonlinear(
        self,
        backend: RegistrationBackend,
        fixed: object,
        moving_linear: object,
        fixed_mask: object,
        moving_linear_mask: object,
        affine: object,
        paths: RegistrationPaths,
        resolved: ResolvedNativeThreads,
    ) -> tuple[Path, object, object]:
        del affine
        prepared_scalar: object | None = None
        temporary_warp: Path | None = None
        warp_commit_attempted = False
        warp_committed = False
        try:
            if self._prepare_nonlinear_before_reuse:
                try:
                    prepared_scalar, temporary_warp = self._estimate_nonlinear(
                        backend,
                        fixed,
                        moving_linear,
                        fixed_mask,
                        moving_linear_mask,
                    )
                except (
                    CacheValidationError,
                    ChunkRegistrationError,
                    RegistrationStageError,
                ):
                    raise
                except Exception as error:
                    raise RegistrationStageError(
                        f"nonlinear registration failed for round {self.imaging_round}"
                    ) from error

            try:
                affine_identity = FileIdentity.from_path(paths.affine)
                nonlinear_evidence = dict(self._nonlinear_request_evidence)
                preprocessing = self._preprocessing_parameters(derived=True)
                if preprocessing is not None:
                    nonlinear_evidence["preprocessing"] = preprocessing
                parameters = nonlinear_stage_parameters(
                    moving_geometry=backend.geometry(moving_linear),
                    registration_mode=self._nonlinear_mode,
                    nonlinear_accuracy=self.nonlinear_accuracy,
                    nonlinear_metric=self.nonlinear_metric,
                    nonlinear_radius=self.nonlinear_radius,
                    scalar_postprocessing_policy=self._scalar_postprocessing_policy,
                    evidence=nonlinear_evidence,
                )
                request = self._request(
                    backend,
                    fixed,
                    resolved,
                    stage="nonlinear",
                    method=self.nonlinear_type,
                    accuracy=self.nonlinear_accuracy,
                    upstream=affine_identity,
                    parameters=parameters,
                )
            except RegistrationStageError:
                raise
            except Exception as error:
                raise RegistrationStageError(
                    f"nonlinear registration failed for round {self.imaging_round}"
                ) from error

            if not self.force_override:
                warp = self._reuse_artifact(paths.warp, request, backend.read_warp)
                if warp is not None:
                    try:
                        moving_nonlinear, moving_nonlinear_mask = (
                            self._reconstruct_nonlinear_images(
                                backend,
                                fixed,
                                moving_linear,
                                moving_linear_mask,
                                paths.warp,
                                prepared_scalar=prepared_scalar,
                            )
                        )
                    except Exception as error:
                        raise RegistrationStageError(
                            "cached nonlinear reconstruction failed "
                            f"for round {self.imaging_round}"
                        ) from error
                    return paths.warp, moving_nonlinear, moving_nonlinear_mask

            if temporary_warp is None:
                prepared_scalar, temporary_warp = self._estimate_nonlinear(
                    backend,
                    fixed,
                    moving_linear,
                    fixed_mask,
                    moving_linear_mask,
                )

            def writer(path: Path) -> None:
                if temporary_warp is None:
                    raise RuntimeError("nonlinear warp candidate was not prepared")
                shutil.copyfile(temporary_warp, path)

            def validator(path: Path) -> bool:
                try:
                    backend.read_warp(path)
                except Exception:
                    return False
                return True

            warp_commit_attempted = True
            self._commit_artifact(paths.warp, request, writer, validator)
            warp_committed = True
            moving_nonlinear, moving_nonlinear_mask = (
                self._reconstruct_nonlinear_images(
                    backend,
                    fixed,
                    moving_linear,
                    moving_linear_mask,
                    paths.warp,
                    prepared_scalar=prepared_scalar,
                )
            )
        except ChunkRegistrationError:
            if warp_committed:
                self._store().mark_stale(paths.warp)
            raise
        except RegistrationStageError:
            if warp_committed:
                self._store().mark_stale(paths.warp)
            raise
        except CacheValidationError as error:
            if warp_committed:
                self._store().mark_stale(paths.warp)
            if warp_commit_attempted:
                raise RegistrationStageError(
                    f"nonlinear registration failed for round {self.imaging_round}"
                ) from error
            raise
        except Exception as error:
            if warp_committed:
                self._store().mark_stale(paths.warp)
            raise RegistrationStageError(
                f"nonlinear registration failed for round {self.imaging_round}"
            ) from error
        finally:
            discard_path(temporary_warp)
        return paths.warp, moving_nonlinear, moving_nonlinear_mask

    def _reconstruct_nonlinear_images(
        self,
        backend: RegistrationBackend,
        fixed: object,
        moving_linear: object,
        moving_linear_mask: object,
        warp_path: Path,
        *,
        prepared_scalar: object | None,
    ) -> tuple[object, object]:
        """Apply the mode's scalar policy and rebuild the committed-warp mask."""
        if self._scalar_postprocessing_policy == WHOLE_SCALAR_POLICY:
            raw_scalar = backend.apply_warp(
                fixed=fixed,
                moving=moving_linear,
                warp_path=warp_path,
                interpolation="linear",
            )
            scalar = backend.histogram_match(
                source=raw_scalar,
                reference=moving_linear,
            )
        elif self._scalar_postprocessing_policy == RAW_RECONSTRUCTED_SCALAR_POLICY:
            scalar = backend.apply_warp(
                fixed=fixed,
                moving=moving_linear,
                warp_path=warp_path,
                interpolation="linear",
            )
        elif self._scalar_postprocessing_policy == CHUNK_SCALAR_POLICY:
            if prepared_scalar is None:
                raise RuntimeError(
                    "serial chunk scalar policy requires a prepared local core stitch"
                )
            scalar = prepared_scalar
        else:
            raise RuntimeError(
                "unsupported nonlinear scalar postprocessing policy: "
                f"{self._scalar_postprocessing_policy}"
            )
        raw_mask = backend.apply_warp(
            fixed=fixed,
            moving=moving_linear_mask,
            warp_path=warp_path,
            interpolation="nearestNeighbor",
        )
        return scalar, backend.threshold_mask(raw_mask)

    def _estimate_nonlinear(
        self,
        backend: RegistrationBackend,
        fixed: object,
        moving_linear: object,
        fixed_mask: object,
        moving_linear_mask: object,
    ) -> tuple[object, Path]:
        """Return a unique disposable candidate owned by this method's caller."""
        return backend.nonlinear(
            moving_image=moving_linear,
            fixed_image=fixed,
            fixed_mask=fixed_mask,
            moving_mask=moving_linear_mask,
            registration_type=self.nonlinear_type,
            accuracy=self.nonlinear_accuracy,
            metric=self.nonlinear_metric,
            radius=self.nonlinear_radius,
            verbose=True,
        )

    def _commit_registered_image(
        self,
        backend: RegistrationBackend,
        moving_nonlinear: object,
        paths: RegistrationPaths,
        resolved: ResolvedNativeThreads,
        warp_path: Path,
    ) -> None:
        try:
            minimum, maximum = self._image_range(backend, moving_nonlinear)
            expected_geometry = backend.geometry(moving_nonlinear)
            parameters = registered_image_parameters(
                minimum=minimum,
                maximum=maximum,
                scalar_postprocessing_policy=self._scalar_postprocessing_policy,
            )
            preprocessing = self._preprocessing_parameters(derived=True)
            if preprocessing is not None:
                parameters["preprocessing"] = preprocessing
            request = self._request(
                backend,
                moving_nonlinear,
                resolved,
                stage="registered-image",
                method=self.nonlinear_type,
                accuracy=self.nonlinear_accuracy,
                upstream=FileIdentity.from_path(warp_path),
                parameters=parameters,
            )
        except RegistrationStageError:
            raise
        except Exception as error:
            raise RegistrationStageError(
                f"registered image persistence failed for round {self.imaging_round}"
            ) from error

        def reader(path: Path) -> object:
            image = backend.load_image(path, self.pixel_size)
            if not geometry_compatible(backend.geometry(image), expected_geometry):
                raise ValueError(
                    "registered TIFF geometry does not match the nonlinear image"
                )
            return image

        if not self.force_override:
            cached = self._reuse_artifact(paths.registered, request, reader)
            if cached is not None:
                return

        def validator(path: Path) -> bool:
            try:
                reader(path)
            except Exception:
                return False
            return True

        try:

            def writer(path: Path) -> None:
                if self.preprocess_filter_sigma_zyx is not None:
                    backend.write_clipped_tiff(moving_nonlinear, path)
                    return
                backend.write_tiff(
                    moving_nonlinear,
                    path,
                    min_value=minimum,
                    max_value=maximum,
                )

            self._commit_artifact(
                paths.registered,
                request,
                writer=writer,
                validator=validator,
            )
        except RegistrationStageError:
            raise
        except Exception as error:
            raise RegistrationStageError(
                f"registered image persistence failed for round {self.imaging_round}"
            ) from error


def register_round(
    fixed_image_path: str | os.PathLike[str],
    moving_image_path: str | os.PathLike[str],
    save_directory: str | os.PathLike[str],
    imaging_round: int = 0,
    crop: bool = False,
    enable_logging: bool = True,
    *,
    pixel_size: PixelSize = (0.15, 0.15, 0.15),
    force_override: bool = False,
    native_threads: NativeThreadConfig | None = None,
    linear_type: str = "TRSAA",
    linear_accuracy: str = "low",
    nonlinear_type: str = "SyNOnly",
    nonlinear_accuracy: str = "high",
    nonlinear_metric: object = "mattes",
    nonlinear_radius: object = None,
    preprocess_filter_sigma_zyx: tuple[float, float, float] | None = None,
    preprocess_feature_mask: FeatureMaskSpec | None = None,
    initial_alignment: InitialAlignmentPolicy | None = None,
) -> None:
    """Register one moving round and persist restart-safe artifacts.

    This convenience function constructs :class:`ImageRegistration` with the
    supplied scientific policy and immediately runs its lifecycle. Parameter units,
    validation, output ownership, and failure behavior are identical to that class.

    Parameters
    ----------
    fixed_image_path, moving_image_path, save_directory : str or path-like
        Fixed input, moving input, and dedicated output directory.
    imaging_round : int, optional
        Nonnegative round identifier.
    crop, enable_logging, force_override : bool, optional
        Crop, logging, and forced-recomputation controls.
    pixel_size : tuple of float, optional
        Physical spacing in ``(x, y, z)`` micrometers.
    native_threads : NativeThreadConfig, optional
        Native ITK and BLAS-family limits.
    linear_type, linear_accuracy, nonlinear_type, nonlinear_accuracy : str, optional
        Registration methods and schedules.
    nonlinear_metric, nonlinear_radius : object, optional
        Nonlinear similarity policy.
    preprocess_filter_sigma_zyx : tuple of float, optional
        Feature filter sigma in NumPy ``(z, y, x)`` voxels.
    preprocess_feature_mask : FeatureMaskSpec, optional
        Optional feature-derived mask policy.
    initial_alignment : InitialAlignmentPolicy, optional
        Physical XY rotation or uniform-similarity seed policy for this moving round.

    See Also
    --------
    ImageRegistration :
        Stateful owner of the registration and artifact lifecycle.
    """
    optional_arguments: dict[str, Any] = {}
    if initial_alignment is not None:
        optional_arguments["initial_alignment"] = initial_alignment
    ImageRegistration(
        fixed_image_path,
        moving_image_path,
        save_directory,
        imaging_round,
        pixel_size,
        crop,
        enable_logging,
        force_override,
        native_threads=native_threads,
        linear_type=linear_type,
        linear_accuracy=linear_accuracy,
        nonlinear_type=nonlinear_type,
        nonlinear_accuracy=nonlinear_accuracy,
        nonlinear_metric=nonlinear_metric,
        nonlinear_radius=nonlinear_radius,
        preprocess_filter_sigma_zyx=preprocess_filter_sigma_zyx,
        preprocess_feature_mask=preprocess_feature_mask,
        **optional_arguments,
    ).register()


__all__ = ["ImageRegistration", "RegistrationPaths", "register_round"]
