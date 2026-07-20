"""Restart-safe transformation of one source channel."""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING

from cyexm.errors import CacheValidationError, RegistrationStageError
from cyexm.registration.artifacts import ArtifactStore, FileIdentity, StageRequest
from cyexm.runtime import NativeThreadConfig, configure_native_threads
from cyexm.spatial import (
    AXIS_CONVENTION_VERSION,
    ImageGeometry,
    PixelSize,
    geometry_compatible,
)

if TYPE_CHECKING:
    from cyexm.registration.backend import RegistrationBackend
    from cyexm.runtime import ResolvedNativeThreads

_INTERPOLATIONS = frozenset({"linear", "nearestNeighbor"})
_OUTPUT_FORMAT_POLICY = "ome-compatible-tiff-uint16-v1"
_POSTPROCESSING_POLICY = "clip-uint16-no-normalization-v1"

logger = logging.getLogger(__name__)


def _path(value: object, name: str) -> Path:
    if not isinstance(value, str | os.PathLike):
        raise TypeError(f"{name} must be path-like")
    try:
        return Path(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be path-like") from error


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


def _validated_interpolation(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("interpolation must be a string")
    if value not in _INTERPOLATIONS:
        supported = ", ".join(sorted(_INTERPOLATIONS))
        raise ValueError(
            f"Unsupported interpolation: {value}. Supported values: {supported}"
        )
    return value


def _production_backend(
    native_threads: NativeThreadConfig | None,
) -> RegistrationBackend:
    from cyexm.registration.backend import AntsRegistrationBackend

    return AntsRegistrationBackend(native_threads)


def _verify_input_identities(
    paths: tuple[Path, Path, Path, Path],
    expected: tuple[FileIdentity, FileIdentity, FileIdentity, FileIdentity],
) -> None:
    for path, identity in zip(paths, expected, strict=True):
        try:
            current = FileIdentity.from_path(path)
        except (OSError, RuntimeError) as error:
            validation_error = CacheValidationError(
                f"Channel-transform input is no longer readable: {path}"
            )
            validation_error.__cause__ = error
            raise RegistrationStageError(
                f"channel-transform input identity changed during execution: {path}"
            ) from validation_error
        if current != identity:
            raise RegistrationStageError(
                f"channel-transform input identity changed during execution: {path}"
            ) from CacheValidationError(
                f"Channel-transform input no longer matches its snapshot: {path}"
            )


def _artifact_snapshot(path: Path) -> tuple[int, int, int, int] | None:
    try:
        metadata = path.stat()
    except FileNotFoundError:
        return None
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _commit_channel(
    *,
    store: ArtifactStore,
    output: Path,
    request: StageRequest,
    backend: RegistrationBackend,
    transformed: object,
    fixed_geometry: ImageGeometry,
    pixel_size: PixelSize,
    input_paths: tuple[Path, Path, Path, Path],
    input_identities: tuple[FileIdentity, FileIdentity, FileIdentity, FileIdentity],
) -> None:
    artifact_before = _artifact_snapshot(output)

    def readable_tiff(path: Path) -> object:
        image = backend.load_image(path, pixel_size)
        if not geometry_compatible(backend.geometry(image), fixed_geometry):
            raise ValueError(
                "transformed channel TIFF geometry does not match the fixed image"
            )
        return image

    try:
        store.commit(
            output,
            request,
            writer=lambda path: backend.write_clipped_tiff(transformed, path),
            validator=lambda path: _is_readable(readable_tiff, path),
        )
    except Exception as error:
        artifact_after = _artifact_snapshot(output)
        if artifact_after != artifact_before:
            record = store.record_path(output)
            if record.exists():
                store.mark_stale(output)
            elif not store.stale_path(output).exists():
                output.unlink(missing_ok=True)
        try:
            _verify_input_identities(input_paths, input_identities)
        except RegistrationStageError:
            raise
        raise RegistrationStageError(
            f"channel-transform artifact commit failed: {output}"
        ) from error

    try:
        _verify_input_identities(input_paths, input_identities)
    except RegistrationStageError:
        store.mark_stale(output)
        raise


def _is_readable(reader: Callable[[Path], object], path: Path) -> bool:
    try:
        reader(path)
    except Exception:
        return False
    return True


def transform_channel(
    *,
    fixed_image_path: str | os.PathLike[str],
    moving_image_path: str | os.PathLike[str],
    affine_path: str | os.PathLike[str],
    warp_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    pixel_size: PixelSize,
    interpolation: str,
    native_threads: NativeThreadConfig | None = None,
    force_override: bool = False,
    _backend: RegistrationBackend | None = None,
    _artifact_store: ArtifactStore | None = None,
) -> Path:
    """Transform one channel through the accepted warp-plus-affine chain."""
    if not isinstance(force_override, bool):
        raise TypeError("force_override must be a boolean")
    spacing = _validated_pixel_size(pixel_size)
    interpolation_name = _validated_interpolation(interpolation)
    supplied_paths = (
        _path(fixed_image_path, "fixed_image_path"),
        _path(moving_image_path, "moving_image_path"),
        _path(affine_path, "affine_path"),
        _path(warp_path, "warp_path"),
    )
    initial_identities = tuple(FileIdentity.from_path(path) for path in supplied_paths)
    input_identities = (
        initial_identities[0],
        initial_identities[1],
        initial_identities[2],
        initial_identities[3],
    )
    input_paths = (
        Path(input_identities[0].path),
        Path(input_identities[1].path),
        Path(input_identities[2].path),
        Path(input_identities[3].path),
    )
    fixed_path, moving_path, affine, warp = input_paths
    for path in input_paths:
        if not path.is_file():
            raise ValueError(f"channel-transform input must be a regular file: {path}")
    output = _path(output_path, "output_path").resolve()
    if output.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("output_path must use a .tif or .tiff extension")
    if output in input_paths:
        raise ValueError("output_path must not alias a channel-transform input")

    resolved: ResolvedNativeThreads = configure_native_threads(native_threads)
    logger.info("Native thread settings: %s", resolved.describe())
    backend = _backend if _backend is not None else _production_backend(native_threads)
    store = _artifact_store if _artifact_store is not None else ArtifactStore()

    try:
        fixed = backend.load_image(fixed_path, spacing)
        moving = backend.load_image(moving_path, spacing)
        _verify_input_identities(input_paths, input_identities)
        fixed_geometry = backend.geometry(fixed)
        moving_geometry = backend.geometry(moving)
        if fixed_geometry.shape != moving_geometry.shape:
            raise RegistrationStageError(
                "channel-transform moving shape does not match the fixed image"
            )
        request = StageRequest.for_files(
            stage="channel-transform-v1",
            inputs=input_paths,
            geometry=fixed_geometry,
            method="warp-affine",
            accuracy="not-applicable",
            native_threads=resolved.values,
            upstream=FileIdentity.from_path(warp),
            parameters={
                "pixel_size_xyz_um": list(spacing),
                "interpolation": interpolation_name,
                "transform_order": ["warp", "affine"],
                "whichtoinvert": [False, False],
                "axis_convention": AXIS_CONVENTION_VERSION,
                "output_format": _OUTPUT_FORMAT_POLICY,
                "postprocessing": _POSTPROCESSING_POLICY,
            },
        )
        if request.inputs != input_identities:
            _verify_input_identities(input_paths, input_identities)

        def readable_tiff(path: Path) -> object:
            image = backend.load_image(path, spacing)
            if not geometry_compatible(backend.geometry(image), fixed_geometry):
                raise ValueError(
                    "transformed channel TIFF geometry does not match the fixed image"
                )
            return image

        if force_override:
            store.mark_stale(output)
        else:
            cached = store.reuse(output, request, readable_tiff)
            if cached is not None:
                _verify_input_identities(input_paths, input_identities)
                return output

        transformed = backend.apply_transform_chain(
            fixed=fixed,
            moving=moving,
            affine_path=affine,
            warp_path=warp,
            interpolation=interpolation_name,
        )
        if not geometry_compatible(backend.geometry(transformed), fixed_geometry):
            raise RegistrationStageError(
                "channel-transform result geometry does not match the fixed image"
            )
        _verify_input_identities(input_paths, input_identities)
        _commit_channel(
            store=store,
            output=output,
            request=request,
            backend=backend,
            transformed=transformed,
            fixed_geometry=fixed_geometry,
            pixel_size=spacing,
            input_paths=input_paths,
            input_identities=input_identities,
        )
    except (CacheValidationError, RegistrationStageError):
        raise
    except Exception as error:
        try:
            _verify_input_identities(input_paths, input_identities)
        except RegistrationStageError:
            raise
        raise RegistrationStageError(f"channel transform failed: {output}") from error
    return output


__all__ = ["transform_channel"]
