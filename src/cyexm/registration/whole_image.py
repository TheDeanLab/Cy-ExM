from __future__ import annotations

import json
import logging
import math
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal

from cyexm.errors import (
    CacheValidationError,
    ChunkRegistrationError,
    RegistrationStageError,
)
from cyexm.io.images import Bounds3D, robust_bounding_box
from cyexm.metrics import read_metrics, write_metrics
from cyexm.registration._native_paths import discard_path
from cyexm.registration._request_policy import (
    CHUNK_SCALAR_POLICY,
    WHOLE_SCALAR_POLICY,
    affine_parameters,
    common_parameters,
    crop_parameters,
    fixed_reference_parameters,
    linear_parameters,
    nonlinear_parameters,
    nonlinear_stage_parameters,
    registered_image_parameters,
)
from cyexm.registration.artifacts import ArtifactStore, FileIdentity, StageRequest
from cyexm.registration.backend import AntsRegistrationBackend, RegistrationBackend
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
        )


class ImageRegistration:
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
        native_threads: NativeThreadConfig | None = None,
        linear_type: str = "TRSAA",
        linear_accuracy: str = "low",
        nonlinear_type: str = "SyNOnly",
        nonlinear_accuracy: str = "high",
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
        self._backend = _backend
        self._artifact_store = _artifact_store
        self._log = logger
        self._nonlinear_mode: Literal["whole-image", "serial-chunked"] = "whole-image"
        self._scalar_postprocessing_policy = WHOLE_SCALAR_POLICY
        self._nonlinear_request_evidence: dict[str, object] = {}
        self._prepare_nonlinear_before_reuse = False
        self._source_paths: tuple[Path, Path] | None = None
        self._source_identities: tuple[FileIdentity, FileIdentity] | None = None

    def register(self) -> None:
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
        fixed, moving = self._crop_if_requested(backend, fixed, moving, paths, resolved)
        fixed_mask = backend.zero_mask(fixed)
        moving_mask = backend.zero_mask(moving)

        self._commit_fixed_reference(backend, fixed, paths, resolved)
        self._commit_metrics(
            "pre-registration",
            backend,
            fixed,
            moving,
            fixed_mask,
            moving_mask,
            paths,
            resolved,
        )

        affine, moving_linear, moving_linear_mask = self._load_or_estimate_affine(
            backend,
            fixed,
            moving,
            fixed_mask,
            moving_mask,
            paths,
            resolved,
        )
        self._commit_metrics(
            "linearly-registered",
            backend,
            fixed,
            moving_linear,
            fixed_mask,
            moving_linear_mask,
            paths,
            resolved,
        )

        warp_path, moving_nonlinear, moving_nonlinear_mask = (
            self._load_or_estimate_nonlinear(
                backend,
                fixed,
                moving_linear,
                fixed_mask,
                moving_linear_mask,
                affine,
                paths,
                resolved,
            )
        )
        self._commit_metrics(
            "nonlinear-registered",
            backend,
            fixed,
            moving_nonlinear,
            fixed_mask,
            moving_nonlinear_mask,
            paths,
            resolved,
        )
        self._commit_registered_image(
            backend, moving_nonlinear, paths, resolved, warp_path
        )

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

    @staticmethod
    def _validated_bounds(value: object, shape: tuple[int, int, int]) -> Bounds3D:
        if isinstance(value, str | bytes):
            raise TypeError("crop bounds must contain exactly six integers")
        try:
            values = tuple(value)  # type: ignore[arg-type]
        except TypeError as error:
            raise TypeError("crop bounds must contain exactly six integers") from error
        if len(values) != 6:
            raise ValueError("crop bounds must contain exactly six integers")
        if any(
            isinstance(item, bool) or not isinstance(item, Integral) for item in values
        ):
            raise TypeError("crop bounds must contain exactly six integers")
        z0, z1, y0, y1, x0, x1 = (int(item) for item in values)
        pairs = ((z0, z1), (y0, y1), (x0, x1))
        if any(
            start < 0 or stop > size or start >= stop
            for (start, stop), size in zip(pairs, shape, strict=True)
        ):
            raise ValueError("crop bounds must define a non-empty image region")
        return z0, z1, y0, y1, x0, x1

    @staticmethod
    def _bounds_text(bounds: Bounds3D) -> str:
        return json.dumps(list(bounds), separators=(",", ":")) + "\n"

    @classmethod
    def _read_bounds(
        cls,
        path: Path,
        shape: tuple[int, int, int],
    ) -> Bounds3D:
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise TypeError("crop bounds must be a JSON array")
        bounds = cls._validated_bounds(payload, shape)
        if cls._bounds_text(bounds) != text:
            raise ValueError("crop bounds artifact is not canonical")
        return bounds

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
                estimated = robust_bounding_box(
                    backend.to_array(image),
                    down_sampling=4,
                    lower_pct=0.1,
                    upper_pct=99.9,
                )
                bounds = self._validated_bounds(estimated, geometry.shape)
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
        return nonlinear_parameters(self.nonlinear_accuracy)

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
            upstream = None
            if stage == "linearly-registered":
                upstream = FileIdentity.from_path(paths.affine)
            elif stage == "nonlinear-registered":
                upstream = FileIdentity.from_path(paths.warp)
            parameters: dict[str, object] = {
                "moving_geometry": backend.geometry(moving).to_dict(),
                "sampling": "regular",
                "sampling_pct": 1.0,
            }
            if stage == "nonlinear-registered":
                parameters["scalar_postprocessing_policy"] = (
                    self._scalar_postprocessing_policy
                )
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
            request = self._request(
                backend,
                fixed,
                resolved,
                stage="affine",
                method=self.linear_type,
                accuracy=self.linear_accuracy,
                upstream=None,
                parameters=affine_parameters(
                    moving_geometry=backend.geometry(moving),
                    linear_type=self.linear_type,
                    accuracy=self.linear_accuracy,
                ),
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
                return transform, moving_linear, moving_linear_mask

        try:
            moving_linear, moving_linear_mask, transform = backend.linear(
                moving_image=moving,
                fixed_image=fixed,
                fixed_mask=fixed_mask,
                moving_mask=moving_mask,
                registration_type=self.linear_type,
                accuracy=self.linear_accuracy,
                verbose=True,
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
                parameters = nonlinear_stage_parameters(
                    moving_geometry=backend.geometry(moving_linear),
                    registration_mode=self._nonlinear_mode,
                    nonlinear_accuracy=self.nonlinear_accuracy,
                    scalar_postprocessing_policy=self._scalar_postprocessing_policy,
                    evidence=self._nonlinear_request_evidence,
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
            request = self._request(
                backend,
                moving_nonlinear,
                resolved,
                stage="registered-image",
                method=self.nonlinear_type,
                accuracy=self.nonlinear_accuracy,
                upstream=FileIdentity.from_path(warp_path),
                parameters=registered_image_parameters(
                    minimum=minimum,
                    maximum=maximum,
                    scalar_postprocessing_policy=self._scalar_postprocessing_policy,
                ),
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
            self._commit_artifact(
                paths.registered,
                request,
                writer=lambda path: backend.write_tiff(
                    moving_nonlinear,
                    path,
                    min_value=minimum,
                    max_value=maximum,
                ),
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
) -> None:
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
    ).register()


__all__ = ["ImageRegistration", "RegistrationPaths", "register_round"]
