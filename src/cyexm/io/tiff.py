from __future__ import annotations

import math
from numbers import Real
from pathlib import Path
from typing import Any

from cyexm.errors import InputValidationError
from cyexm.io.atomic import atomic_write


def _range_value(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InputValidationError(
            "TIFF output range must contain finite uint16-compatible values"
        )
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 65535.0:
        raise InputValidationError(
            "TIFF output range must contain finite uint16-compatible values"
        )
    return number


def write_uint16_tiff(
    image: Any,
    path: str | Path,
    min_value: object,
    max_value: object,
) -> None:
    """Scale a real three-dimensional image into a uint16 TIFF atomically."""
    minimum = _range_value(min_value, "min_value")
    maximum = _range_value(max_value, "max_value")
    if minimum > maximum:
        raise InputValidationError("TIFF output range minimum must not exceed maximum")

    destination = Path(path)
    if destination.suffix.lower() not in {".tif", ".tiff"}:
        raise InputValidationError(
            "TIFF output path must use a .tif or .tiff extension"
        )

    import numpy as np

    if isinstance(image, np.ndarray):
        source = image
    else:
        import ants

        if not isinstance(image, ants.core.ants_image.ANTsImage):
            raise TypeError("image must be an ANTs image or NumPy array")
        source = image.numpy()

    source = np.asarray(source)
    if source.ndim != 3:
        raise InputValidationError("TIFF image data must be three-dimensional")
    if (
        np.issubdtype(source.dtype, np.bool_)
        or np.issubdtype(source.dtype, np.complexfloating)
        or not np.issubdtype(source.dtype, np.number)
    ):
        raise InputValidationError("TIFF image data must contain real numeric values")
    if source.size == 0:
        raise InputValidationError("TIFF image data must not be empty")

    data = np.array(source, dtype=np.float64, copy=True)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    np.maximum(data, 0.0, out=data)
    data_minimum = float(data.min())
    data_maximum = float(data.max())
    if data_minimum == data_maximum:
        scaled = np.full(data.shape, minimum, dtype=np.float64)
    else:
        normalized = (data - data_minimum) / (data_maximum - data_minimum)
        scaled = minimum + normalized * (maximum - minimum)
    encoded = np.clip(scaled, minimum, maximum).astype(np.uint16)

    import tifffile

    def writer(temporary: Path) -> None:
        tifffile.imwrite(temporary, encoded)

    def validator(temporary: Path) -> bool:
        try:
            written = tifffile.imread(temporary)
        except Exception:
            return False
        return (
            written.dtype == np.dtype(np.uint16)
            and written.shape == encoded.shape
            and np.array_equal(written, encoded)
        )

    atomic_write(destination, writer, validator)


__all__ = ["write_uint16_tiff"]
