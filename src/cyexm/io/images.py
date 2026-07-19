from __future__ import annotations

import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from cyexm.errors import InputValidationError
from cyexm.spatial import ImageGeometry

type Bounds3D = tuple[int, int, int, int, int, int]

_SUPPORTED_EXTENSIONS = frozenset({".npy", ".tif", ".tiff"})


def _resolved_input_file(path: str | Path) -> Path:
    try:
        resolved = Path(path).resolve(strict=True)
    except (OSError, RuntimeError, TypeError) as error:
        raise InputValidationError(
            f"Input path must resolve to a file: {path}"
        ) from error
    if not resolved.is_file():
        raise InputValidationError(f"Input path must resolve to a file: {path}")
    return resolved


def load_array(path: str | Path) -> Any:
    """Eagerly load a numeric three-dimensional NumPy or TIFF volume."""
    resolved = _resolved_input_file(path)
    extension = resolved.suffix.lower()
    if extension not in _SUPPORTED_EXTENSIONS:
        raise InputValidationError(
            "Unsupported image extension; expected .npy, .tif, or .tiff"
        )

    import numpy as np

    try:
        if extension == ".npy":
            loaded = np.load(resolved, allow_pickle=False)
        else:
            import tifffile

            loaded = tifffile.imread(resolved)
    except ValueError as error:
        raise InputValidationError(
            f"Image data must be a three-dimensional numeric array: {resolved}"
        ) from error
    except OSError as error:
        raise InputValidationError(
            f"Image file could not be loaded: {resolved}"
        ) from error

    array = np.array(loaded, copy=True)
    if array.ndim != 3:
        raise InputValidationError("Image data must be three-dimensional")
    if np.issubdtype(array.dtype, np.bool_) or not np.issubdtype(
        array.dtype, np.number
    ):
        raise InputValidationError(
            "Image data must contain numeric, non-boolean values"
        )
    return array


def _pixel_size(pixel_size_xyz: object) -> tuple[float, float, float]:
    if isinstance(pixel_size_xyz, str | bytes):
        raise InputValidationError(
            "pixel_size_xyz must contain exactly three positive finite values"
        )
    try:
        values = tuple(pixel_size_xyz)  # type: ignore[arg-type]
    except TypeError as error:
        raise InputValidationError(
            "pixel_size_xyz must contain exactly three positive finite values"
        ) from error
    if len(values) != 3:
        raise InputValidationError(
            "pixel_size_xyz must contain exactly three positive finite values"
        )

    spacing: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise InputValidationError(
                "pixel_size_xyz must contain exactly three positive finite values"
            )
        number = float(value)
        if not math.isfinite(number) or number <= 0.0:
            raise InputValidationError(
                "pixel_size_xyz must contain exactly three positive finite values"
            )
        spacing.append(number)
    return spacing[0], spacing[1], spacing[2]


def load_ants_image(path: str | Path, pixel_size_xyz: object) -> Any:
    """Load a ZYX volume as an ANTs image with public XYZ spacing."""
    spacing = _pixel_size(pixel_size_xyz)
    array = load_array(path)

    import ants

    return ants.from_numpy(
        array,
        origin=(0.0, 0.0, 0.0),
        spacing=spacing,
    )


def _triple(values: object, name: str) -> tuple[Any, Any, Any]:
    try:
        items = tuple(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise InputValidationError(
            f"{name} must contain exactly three values for a three-dimensional image"
        ) from error
    if len(items) != 3:
        raise InputValidationError(
            f"{name} must contain exactly three values for a three-dimensional image"
        )
    return items[0], items[1], items[2]


def ants_geometry(image: Any) -> ImageGeometry:
    """Copy an ANTs image's three-dimensional geometry into a neutral value."""
    shape_values = _triple(image.shape, "shape")
    origin_values = _triple(image.origin, "origin")
    spacing_values = _triple(image.spacing, "spacing")
    direction_rows = _triple(image.direction, "direction")
    direction_0 = _triple(direction_rows[0], "direction[0]")
    direction_1 = _triple(direction_rows[1], "direction[1]")
    direction_2 = _triple(direction_rows[2], "direction[2]")

    return ImageGeometry(
        shape=(
            int(shape_values[0]),
            int(shape_values[1]),
            int(shape_values[2]),
        ),
        origin=(
            float(origin_values[0]),
            float(origin_values[1]),
            float(origin_values[2]),
        ),
        spacing=(
            float(spacing_values[0]),
            float(spacing_values[1]),
            float(spacing_values[2]),
        ),
        direction=(
            (
                float(direction_0[0]),
                float(direction_0[1]),
                float(direction_0[2]),
            ),
            (
                float(direction_1[0]),
                float(direction_1[1]),
                float(direction_1[2]),
            ),
            (
                float(direction_2[0]),
                float(direction_2[1]),
                float(direction_2[2]),
            ),
        ),
    )


def zero_mask(image: Any) -> Any:
    """Return a positive-valued mask while preserving all ANTs geometry."""
    import ants
    import numpy as np

    data = (image.numpy() > 0).astype(np.float32)
    return ants.from_numpy(
        data,
        origin=image.origin,
        spacing=image.spacing,
        direction=image.direction,
    )


def _validated_down_sampling(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("down_sampling must be a positive integer")
    down_sampling = int(value)
    if down_sampling <= 0:
        raise ValueError("down_sampling must be a positive integer")
    return down_sampling


def _validated_percentiles(
    lower_pct: object,
    upper_pct: object,
) -> tuple[float, float]:
    if (
        isinstance(lower_pct, bool)
        or isinstance(upper_pct, bool)
        or not isinstance(lower_pct, Real)
        or not isinstance(upper_pct, Real)
    ):
        raise TypeError("percentiles must be finite numeric values")
    lower = float(lower_pct)
    upper = float(upper_pct)
    if (
        not math.isfinite(lower)
        or not math.isfinite(upper)
        or not 0.0 <= lower < upper <= 100.0
    ):
        raise ValueError("percentiles must satisfy 0 <= lower_pct < upper_pct <= 100")
    return lower, upper


def robust_bounding_box(
    array: Any,
    *,
    down_sampling: object = 8,
    lower_pct: object = 5.0,
    upper_pct: object = 95.0,
) -> Bounds3D:
    """Find robust exclusive ZYX bounds after downsampled Otsu thresholding."""
    factor = _validated_down_sampling(down_sampling)
    lower, upper = _validated_percentiles(lower_pct, upper_pct)

    import numpy as np

    if not isinstance(array, np.ndarray):
        raise TypeError("array must be a NumPy array")
    if array.ndim != 3:
        raise ValueError("array must be three-dimensional")
    if np.issubdtype(array.dtype, np.bool_):
        binary = array
    else:
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError("array must be a numeric NumPy array")
        from skimage.filters import threshold_otsu

        threshold = threshold_otsu(array[::factor, ::factor, ::factor])
        binary = array > threshold

    def find_bounds(distribution: Any) -> tuple[int, int]:
        cumulative = np.cumsum(distribution)
        total = int(cumulative[-1])
        if total == 0:
            raise ValueError("No foreground found")
        if lower == 0.0:
            lower_index = int(np.flatnonzero(distribution)[0])
        else:
            lower_index = int(np.searchsorted(cumulative, total * lower / 100.0))
        upper_index = int(np.searchsorted(cumulative, total * upper / 100.0)) + 1
        return lower_index, upper_index

    z0, z1 = find_bounds(np.count_nonzero(binary, axis=(1, 2)))
    y0, y1 = find_bounds(np.count_nonzero(binary, axis=(0, 2)))
    x0, x1 = find_bounds(np.count_nonzero(binary, axis=(0, 1)))
    return z0, z1, y0, y1, x0, x1


def crop_array(array: Any, bounds: object) -> Any:
    """Crop a NumPy ZYX volume with exclusive six-integer bounds."""
    import numpy as np

    if not isinstance(array, np.ndarray):
        raise TypeError("array must be a NumPy array")
    if array.ndim != 3:
        raise ValueError("array must be three-dimensional")
    if isinstance(bounds, str | bytes):
        raise TypeError("bounds must contain exactly six integers")
    try:
        values = tuple(bounds)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError("bounds must contain exactly six integers") from error
    if len(values) != 6:
        raise ValueError("bounds must contain exactly six integers")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in values
    ):
        raise TypeError("bounds must contain exactly six integers")

    z0, z1, y0, y1, x0, x1 = (int(value) for value in values)
    pairs = (
        (z0, z1, array.shape[0]),
        (y0, y1, array.shape[1]),
        (x0, x1, array.shape[2]),
    )
    if any(start < 0 or stop > size or start >= stop for start, stop, size in pairs):
        raise ValueError("bounds must define non-empty regions inside the array")
    return array[z0:z1, y0:y1, x0:x1]


__all__ = [
    "Bounds3D",
    "ants_geometry",
    "crop_array",
    "load_ants_image",
    "load_array",
    "robust_bounding_box",
    "zero_mask",
]
