from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from cyexm.runtime import NativeThreadConfig, configure_native_threads
from cyexm.spatial import ImageGeometry

logger = logging.getLogger(__name__)


def _scientific_runtime() -> tuple[Any, Any]:
    import ants
    import numpy as np

    return ants, np


def _configured_runtime(
    native_threads: NativeThreadConfig | None,
) -> tuple[Any, Any]:
    resolved = configure_native_threads(native_threads)
    logger.info("Native thread settings: %s", resolved.describe())
    return _scientific_runtime()


def apply_affine(
    *,
    fixed: Any,
    moving: Any,
    transform: Any,
    interpolation: str,
    native_threads: NativeThreadConfig | None = None,
) -> object:
    """Apply an in-memory affine transform in the fixed image space."""
    _configured_runtime(native_threads)
    return transform.apply_to_image(
        image=moving,
        reference=fixed,
        interpolation=interpolation,
    )


def create_xy_rotation_transform(
    *,
    geometry: ImageGeometry,
    fixed_geometry: ImageGeometry | None = None,
    angle_deg: float,
    translation_yx_px: tuple[float, float] = (0.0, 0.0),
    native_threads: NativeThreadConfig | None = None,
) -> object:
    """Create the inverse-map affine for a visible moving-image correction."""
    ants, np = _configured_runtime(native_threads)
    if type(geometry) is not ImageGeometry:
        raise TypeError("geometry must be an ImageGeometry")
    if fixed_geometry is None:
        fixed_geometry = geometry
    if type(fixed_geometry) is not ImageGeometry:
        raise TypeError("fixed_geometry must be an ImageGeometry")
    angle = float(angle_deg)
    shift_y, shift_x = (float(value) for value in translation_yx_px)
    if not all(math.isfinite(value) for value in (angle, shift_y, shift_x)):
        raise ValueError("rotation and translation values must be finite")
    shape_xyz = np.asarray(geometry.shape[::-1], dtype=np.float64)
    spacing = np.asarray(geometry.spacing, dtype=np.float64)
    direction = np.asarray(geometry.direction, dtype=np.float64)
    origin = np.asarray(geometry.origin, dtype=np.float64)
    center = origin + direction @ (spacing * (shape_xyz - 1.0) / 2.0)
    fixed_shape_xyz = np.asarray(fixed_geometry.shape[::-1], dtype=np.float64)
    fixed_spacing = np.asarray(fixed_geometry.spacing, dtype=np.float64)
    fixed_direction = np.asarray(fixed_geometry.direction, dtype=np.float64)
    fixed_origin = np.asarray(fixed_geometry.origin, dtype=np.float64)
    fixed_center = fixed_origin + fixed_direction @ (
        fixed_spacing * (fixed_shape_xyz - 1.0) / 2.0
    )

    radians = math.radians(-angle)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    local_matrix = np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    matrix = direction @ local_matrix @ direction.T
    visible_shift = direction @ (
        spacing * np.asarray((shift_x, shift_y, 0.0), dtype=np.float64)
    )
    transform_translation = matrix @ (center - fixed_center - visible_shift)
    return ants.create_ants_transform(
        transform_type="AffineTransform",
        dimension=3,
        matrix=matrix,
        center=tuple(float(value) for value in center),
        translation=tuple(float(value) for value in transform_translation),
    )


def _affine_homogeneous(np: Any, transform: object) -> Any:
    runtime_transform: Any = transform
    if runtime_transform.dimension != 3:
        raise ValueError("affine transform must be three-dimensional")
    if runtime_transform.transform_type != "AffineTransform":
        raise ValueError("only AffineTransform values can be composed")
    parameters = np.asarray(runtime_transform.parameters, dtype=np.float64)
    fixed = np.asarray(runtime_transform.fixed_parameters, dtype=np.float64)
    if parameters.shape != (12,) or fixed.shape[0] < 3:
        raise ValueError("affine transform parameters are not three-dimensional")
    matrix = parameters[:9].reshape(3, 3)
    translation = parameters[9:12]
    center = fixed[:3]
    homogeneous = np.eye(4, dtype=np.float64)
    homogeneous[:3, :3] = matrix
    homogeneous[:3, 3] = translation + center - matrix @ center
    return homogeneous


def compose_affine_transforms(
    transforms: tuple[object, ...],
    *,
    native_threads: NativeThreadConfig | None = None,
) -> object:
    """Flatten affine transforms in ANTs forward-list application order."""
    if type(transforms) is not tuple or not transforms:
        raise TypeError("transforms must be a non-empty tuple")
    ants, np = _configured_runtime(native_threads)
    complete = np.eye(4, dtype=np.float64)
    for transform in transforms:
        complete = _affine_homogeneous(np, transform) @ complete
    return ants.create_ants_transform(
        transform_type="AffineTransform",
        dimension=3,
        matrix=complete[:3, :3],
        center=(0.0, 0.0, 0.0),
        translation=tuple(float(value) for value in complete[:3, 3]),
    )


def apply_warp(
    *,
    fixed: Any,
    moving: Any,
    warp_path: Path,
    interpolation: str,
    native_threads: NativeThreadConfig | None = None,
) -> object:
    """Apply a forward displacement field without inversion."""
    ants, _ = _configured_runtime(native_threads)
    return ants.apply_transforms(
        fixed=fixed,
        moving=moving,
        transformlist=[str(warp_path)],
        whichtoinvert=[False],
        interpolator=interpolation,
    )


def apply_transform_chain(
    *,
    fixed: Any,
    moving: Any,
    affine_path: Path,
    warp_path: Path,
    interpolation: str,
    native_threads: NativeThreadConfig | None = None,
) -> object:
    """Apply the forward warp and affine in their established list order."""
    ants, _ = _configured_runtime(native_threads)
    return ants.apply_transforms(
        fixed=fixed,
        moving=moving,
        transformlist=[str(warp_path), str(affine_path)],
        whichtoinvert=[False, False],
        interpolator=interpolation,
    )


def transform_image(
    *,
    fixed: Any,
    moving: Any,
    affine_path: Path,
    warp_path: Path,
    interpolation: str = "linear",
    native_threads: NativeThreadConfig | None = None,
) -> object:
    """Apply a persisted warp-plus-affine chain to an image."""
    return apply_transform_chain(
        fixed=fixed,
        moving=moving,
        affine_path=affine_path,
        warp_path=warp_path,
        interpolation=interpolation,
        native_threads=native_threads,
    )


def export_affine_transform(
    affine_transform: object,
    path: str | Path,
    *,
    native_threads: NativeThreadConfig | None = None,
) -> None:
    """Write an affine transform exactly to the supplied artifact path."""
    ants, _ = _configured_runtime(native_threads)
    ants.write_transform(affine_transform, str(path))


def import_affine_transform(
    path: str | Path,
    *,
    native_threads: NativeThreadConfig | None = None,
) -> object:
    """Read an affine transform from the supplied artifact path."""
    ants, _ = _configured_runtime(native_threads)
    return ants.read_transform(str(path))


def set_origin_and_spacing(
    image: Any,
    origin: tuple[float, float, float] | None = None,
    spacing: tuple[float, float, float] | None = None,
    *,
    native_threads: NativeThreadConfig | None = None,
) -> object:
    """Update optional XYZ origin and spacing values in place."""
    resolved = configure_native_threads(native_threads)
    logger.info("Native thread settings: %s", resolved.describe())
    if origin is not None:
        image.set_origin(new_origin=origin)
    if spacing is not None:
        image.set_spacing(new_spacing=spacing)
    logger.info("Image origin set to: %s", image.origin)
    logger.info("Image spacing (pixel size in XYZ) set to: %s", image.spacing)
    return image


__all__ = [
    "apply_affine",
    "apply_transform_chain",
    "apply_warp",
    "compose_affine_transforms",
    "create_xy_rotation_transform",
    "export_affine_transform",
    "import_affine_transform",
    "set_origin_and_spacing",
    "transform_image",
]
