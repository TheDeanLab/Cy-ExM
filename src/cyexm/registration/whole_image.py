from __future__ import annotations

import json
import logging
import math
import os
import shutil
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from cyexm.errors import CacheValidationError, RegistrationStageError
from cyexm.io.images import Bounds3D, robust_bounding_box
from cyexm.metrics import read_metrics, write_metrics
from cyexm.registration.artifacts import ArtifactStore, FileIdentity, StageRequest
from cyexm.registration.backend import AntsRegistrationBackend, RegistrationBackend
from cyexm.runtime import (
    NativeThreadConfig,
    ResolvedNativeThreads,
    configure_native_threads,
)
from cyexm.spatial import PixelSize

logger = logging.getLogger(__name__)

_LINEAR_TYPES = frozenset({"Translation", "Rigid", "Similarity", "Affine", "TRSAA"})
_NONLINEAR_TYPES = frozenset({"Elastic", "SyNOnly", "SyN"})
_ACCURACIES = frozenset({"high", "medium", "low", "dry run"})
_LINEAR_ITERATIONS = {
    "high": (1000, 500, 250, 100),
    "medium": (500, 250, 100, 50),
    "low": (100, 50, 25, 10),
}
_NONLINEAR_ITERATIONS = {
    "high": (100, 70, 50),
    "medium": (50, 35, 25),
    "low": (5, 5, 5),
    "dry run": (1, 1, 1),
}


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

        fixed = backend.load_image(Path(self.fixed_image_path), self.pixel_size)
        moving = backend.load_image(Path(self.moving_image_path), self.pixel_size)
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
        evidence: dict[str, object] = {
            "crop": self.crop,
            "imaging_round": self.imaging_round,
            "native_thread_sources": dict(resolved.sources),
            "pixel_size_xyz": self.pixel_size,
        }
        evidence.update(parameters)
        return StageRequest.for_files(
            stage=stage,
            inputs=(Path(self.fixed_image_path), Path(self.moving_image_path)),
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
        role: str,
        resolved: ResolvedNativeThreads,
    ) -> object:
        geometry = backend.geometry(image)
        request = self._request(
            backend,
            image,
            resolved,
            stage=f"crop:{role}",
            method="robust-bounding-box",
            accuracy="legacy-robust",
            upstream=None,
            parameters={
                "down_sampling": 4,
                "image_role": role,
                "lower_pct": 0.1,
                "upper_pct": 99.9,
            },
        )
        bounds = None
        if not self.force_override:
            bounds = self._store().reuse(
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

                self._store().commit(
                    path,
                    request,
                    writer=writer,
                    validator=lambda candidate: (
                        self._read_bounds(candidate, geometry.shape) == bounds
                    ),
                )
            except CacheValidationError:
                raise
            except Exception as error:
                raise RegistrationStageError(
                    f"{role} crop stage failed for round {self.imaging_round}"
                ) from error
        return backend.crop_image(image, bounds)

    @staticmethod
    def _image_range(
        backend: RegistrationBackend, image: object
    ) -> tuple[float, float]:
        array: Any = backend.to_array(image)
        return float(array.min()), float(array.max())

    def _linear_parameters(self) -> dict[str, object]:
        parameters: dict[str, object] = {
            "aff_metric": "mattes",
            "aff_random_sampling_rate": 0.5,
            "aff_sampling": 32,
            "mask_all_stages": True,
            "verbose": True,
        }
        if self.linear_type == "TRSAA" and self.linear_accuracy != "dry run":
            parameters["reg_iterations"] = _LINEAR_ITERATIONS[self.linear_accuracy]
        if self.linear_accuracy == "dry run":
            parameters.update(
                effective_registration_type="Rigid",
                aff_iterations=(1,),
                aff_smoothing_sigmas=(0,),
                aff_shrink_factors=(8,),
                aff_random_sampling_rate=0.1,
            )
        return parameters

    def _nonlinear_parameters(self) -> dict[str, object]:
        return {
            "aff_random_sampling_rate": 1.0,
            "flow_sigma": 3.0,
            "initial_transform": "Identity",
            "mask_all_stages": True,
            "reg_iterations": _NONLINEAR_ITERATIONS[self.nonlinear_accuracy],
            "singleprecision": False,
            "syn_metric": "mattes",
            "syn_sampling": 32,
            "total_sigma": 0.5,
            "verbose": True,
        }

    def _commit_fixed_reference(
        self,
        backend: RegistrationBackend,
        fixed: object,
        paths: RegistrationPaths,
        resolved: ResolvedNativeThreads,
    ) -> None:
        minimum, maximum = self._image_range(backend, fixed)
        request = self._request(
            backend,
            fixed,
            resolved,
            stage="fixed-reference",
            method="uint16-tiff",
            accuracy="exact",
            upstream=None,
            parameters={"minimum": minimum, "maximum": maximum},
        )

        def reader(path: Path) -> object:
            return backend.load_image(path, self.pixel_size)

        if not self.force_override:
            cached = self._store().reuse(paths.fixed_reference, request, reader)
            if cached is not None:
                return

        def validator(path: Path) -> bool:
            try:
                reader(path)
            except Exception:
                return False
            return True

        self._store().commit(
            paths.fixed_reference,
            request,
            writer=lambda path: backend.write_tiff(
                fixed, path, min_value=minimum, max_value=maximum
            ),
            validator=validator,
        )

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
        upstream = None
        if stage == "linearly-registered":
            upstream = FileIdentity.from_path(paths.affine)
        elif stage == "nonlinear-registered":
            upstream = FileIdentity.from_path(paths.warp)
        request = self._request(
            backend,
            fixed,
            resolved,
            stage=f"metrics:{stage}",
            method="Correlation+MattesMutualInformation",
            accuracy="full",
            upstream=upstream,
            parameters={
                "moving_geometry": backend.geometry(moving).to_dict(),
                "sampling": "regular",
                "sampling_pct": 1.0,
            },
        )
        destination = path_by_stage[stage]
        if not self.force_override:
            cached = self._store().reuse(destination, request, read_metrics)
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
            self._store().commit(
                destination,
                request,
                writer=lambda path: write_metrics(metrics, path),
                validator=lambda path: read_metrics(path) == metrics,
            )
        except CacheValidationError:
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
        request = self._request(
            backend,
            fixed,
            resolved,
            stage="affine",
            method=self.linear_type,
            accuracy=self.linear_accuracy,
            upstream=None,
            parameters={
                "image_interpolation": "linear",
                "mask_interpolation": "nearestNeighbor",
                "moving_geometry": backend.geometry(moving).to_dict(),
                "registration_parameters": self._linear_parameters(),
            },
        )
        if not self.force_override:
            transform = self._store().reuse(paths.affine, request, backend.read_affine)
            if transform is not None:
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

            self._store().commit(
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
        except CacheValidationError:
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
        affine_identity = FileIdentity.from_path(paths.affine)
        request = self._request(
            backend,
            fixed,
            resolved,
            stage="nonlinear",
            method=self.nonlinear_type,
            accuracy=self.nonlinear_accuracy,
            upstream=affine_identity,
            parameters={
                "image_interpolation": "linear",
                "mask_interpolation": "nearestNeighbor",
                "moving_geometry": backend.geometry(moving_linear).to_dict(),
                "registration_parameters": self._nonlinear_parameters(),
            },
        )
        if not self.force_override:
            warp = self._store().reuse(paths.warp, request, backend.read_warp)
            if warp is not None:
                moving_nonlinear = backend.apply_warp(
                    fixed=fixed,
                    moving=moving_linear,
                    warp_path=paths.warp,
                    interpolation="linear",
                )
                moving_nonlinear_mask = backend.threshold_mask(
                    backend.apply_warp(
                        fixed=fixed,
                        moving=moving_linear_mask,
                        warp_path=paths.warp,
                        interpolation="nearestNeighbor",
                    )
                )
                return paths.warp, moving_nonlinear, moving_nonlinear_mask

        try:
            moving_nonlinear, temporary_warp = self._estimate_nonlinear(
                backend,
                fixed,
                moving_linear,
                fixed_mask,
                moving_linear_mask,
            )

            def writer(path: Path) -> None:
                shutil.copyfile(temporary_warp, path)

            def validator(path: Path) -> bool:
                try:
                    backend.read_warp(path)
                except Exception:
                    return False
                return True

            self._store().commit(paths.warp, request, writer, validator)
            moving_nonlinear_mask = backend.threshold_mask(
                backend.apply_warp(
                    fixed=fixed,
                    moving=moving_linear_mask,
                    warp_path=paths.warp,
                    interpolation="nearestNeighbor",
                )
            )
        except CacheValidationError:
            raise
        except Exception as error:
            raise RegistrationStageError(
                f"nonlinear registration failed for round {self.imaging_round}"
            ) from error
        return paths.warp, moving_nonlinear, moving_nonlinear_mask

    def _estimate_nonlinear(
        self,
        backend: RegistrationBackend,
        fixed: object,
        moving_linear: object,
        fixed_mask: object,
        moving_linear_mask: object,
    ) -> tuple[object, Path]:
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
        minimum, maximum = self._image_range(backend, moving_nonlinear)
        request = self._request(
            backend,
            moving_nonlinear,
            resolved,
            stage="registered-image",
            method=self.nonlinear_type,
            accuracy=self.nonlinear_accuracy,
            upstream=FileIdentity.from_path(warp_path),
            parameters={
                "format": "uint16-tiff",
                "minimum": minimum,
                "maximum": maximum,
            },
        )

        def reader(path: Path) -> object:
            return backend.load_image(path, self.pixel_size)

        if not self.force_override:
            cached = self._store().reuse(paths.registered, request, reader)
            if cached is not None:
                return

        def validator(path: Path) -> bool:
            try:
                reader(path)
            except Exception:
                return False
            return True

        self._store().commit(
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
