"""Construct the sole supported registration feature from one ZYX scalar volume.

Use :func:`background_subtract_asinh` to derive transform-estimation features.
Workflow orchestration remains responsible for cropping, raw masks, metrics,
transform application, and artifact persistence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from cyexm.registration._request_policy import REGISTRATION_FEATURE_POLICY


def _scientific_runtime() -> tuple[Any, Any]:
    import numpy as np
    from scipy import ndimage as ndi

    return np, ndi


def _validated_sigma(
    value: tuple[float, float, float],
) -> tuple[float, float, float]:
    if type(value) is not tuple or len(value) != 3:
        raise TypeError("filter sigma must be an exact ZYX tuple of three values")
    normalized: list[float] = []
    for axis, item in zip("zyx", value, strict=True):
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise TypeError(f"filter sigma {axis} must be a positive finite number")
        number = float(item)
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(f"filter sigma {axis} must be a positive finite number")
        normalized.append(number)
    return normalized[0], normalized[1], normalized[2]


@dataclass(frozen=True, slots=True)
class RegistrationFeature:
    """Derived registration values and their deterministic robust scale."""

    array: object
    robust_scale: float


@dataclass(frozen=True, slots=True)
class FeatureMask:
    """Derived binary registration mask and realized support counts."""

    array: object
    foreground_voxels: int
    total_voxels: int


def _positive_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a positive finite number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


def _validated_spacing(
    value: tuple[float, float, float],
) -> tuple[float, float, float]:
    if type(value) is not tuple or len(value) != 3:
        raise TypeError("spacing must be an exact ZYX tuple of three values")
    return (
        _positive_finite(value[0], "spacing z"),
        _positive_finite(value[1], "spacing y"),
        _positive_finite(value[2], "spacing x"),
    )


def feature_mask(
    feature: object,
    spacing_zyx: tuple[float, float, float],
    *,
    min_component_volume_um3: float,
    dilation_radius_um: float,
) -> FeatureMask:
    """Return cleaned and physically dilated positive feature support."""
    spacing = _validated_spacing(spacing_zyx)
    minimum_volume = _positive_finite(
        min_component_volume_um3,
        "minimum component volume",
    )
    dilation = _positive_finite(dilation_radius_um, "dilation radius")
    np, ndi = _scientific_runtime()
    array = np.asarray(feature, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("feature mask construction requires a 3D image")
    if not bool(np.isfinite(array).all()):
        raise ValueError("feature mask construction requires finite feature values")

    support = array > 0.0
    labels, _ = ndi.label(
        support,
        structure=np.ones((3, 3, 3), dtype=bool),
    )
    minimum_voxels = max(
        1,
        math.ceil(minimum_volume / math.prod(spacing)),
    )
    counts = np.bincount(labels.ravel())
    keep = counts >= minimum_voxels
    keep[0] = False
    cleaned = keep[labels]
    if not bool(cleaned.any()):
        raise ValueError("feature mask has no foreground after component filtering")

    radii = tuple(math.ceil(dilation / axis_spacing) for axis_spacing in spacing)
    footprint_shape = tuple(2 * radius + 1 for radius in radii)
    coordinates = np.indices(footprint_shape, dtype=np.float64)
    distance_squared = np.zeros(footprint_shape, dtype=np.float64)
    for axis, (radius, axis_spacing) in enumerate(zip(radii, spacing, strict=True)):
        distance_squared += ((coordinates[axis] - radius) * axis_spacing) ** 2
    footprint = distance_squared <= dilation**2
    mask = ndi.binary_dilation(cleaned, structure=footprint).astype(
        np.float32,
        copy=False,
    )
    foreground = int(np.count_nonzero(mask))
    if foreground == 0:
        raise ValueError("feature mask has no foreground after dilation")
    return FeatureMask(
        array=mask,
        foreground_voxels=foreground,
        total_voxels=int(mask.size),
    )


def background_subtract_asinh(
    image: object,
    filter_sigma_zyx: tuple[float, float, float],
) -> RegistrationFeature:
    """Return a background-subtracted, robustly scaled asinh feature.

    Parameters
    ----------
    image
        Finite scalar volume with NumPy axis order ``(z, y, x)``. The input is
        not mutated.
    filter_sigma_zyx
        Positive finite Gaussian widths in voxel-domain ``(z, y, x)`` order.

    Returns
    -------
    RegistrationFeature
        Shape-preserving float32 feature values and the signed-residual MAD
        scale used to normalize them.

    Raises
    ------
    TypeError
        If the sigma is not an exact three-value tuple.
    ValueError
        If the image is not finite and three-dimensional or a sigma value is
        not positive and finite.
    """
    sigma = _validated_sigma(filter_sigma_zyx)
    np, ndi = _scientific_runtime()
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("registration preprocessing requires a 3D image")
    if not bool(np.isfinite(array).all()):
        raise ValueError("registration preprocessing requires finite image values")
    background = ndi.gaussian_filter(array, sigma=sigma, mode="reflect")
    residual = array - background
    sample_stride = max(1, array.size // 2_000_000)
    sample = residual.ravel()[::sample_stride]
    median = float(np.median(sample))
    mad = float(np.median(np.abs(sample - median)))
    scale = max(1.4826 * mad, float(np.finfo(np.float32).eps))
    feature = np.arcsinh(np.maximum(residual, 0.0) / scale).astype(
        np.float32,
        copy=False,
    )
    return RegistrationFeature(array=feature, robust_scale=scale)


__all__ = (
    "REGISTRATION_FEATURE_POLICY",
    "FeatureMask",
    "RegistrationFeature",
    "background_subtract_asinh",
    "feature_mask",
)
