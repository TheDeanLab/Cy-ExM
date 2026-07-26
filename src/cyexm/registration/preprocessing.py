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
    "RegistrationFeature",
    "background_subtract_asinh",
)
