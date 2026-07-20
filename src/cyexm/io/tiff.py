from __future__ import annotations

import json
import math
import re
from numbers import Real
from pathlib import Path
from typing import Any

from cyexm.errors import InputValidationError
from cyexm.io.atomic import atomic_write
from cyexm.spatial import (
    AXIS_CONVENTION_VERSION,
    ImageGeometry,
)

_TIFF_GEOMETRY_SCHEMA_VERSION = 1
_CYEXM_TIFF_KEYS = frozenset({"schema_version", "axis_convention_version", "geometry"})
_JSON_UNICODE_ESCAPE = re.compile(r"(\\+)u([0-9A-Fa-f]{4})")
_CYEXM_ASCII_CODEPOINTS = {ord(letter): letter for letter in "cyexm"}


def _metadata_error(
    message: str, error: Exception | None = None
) -> InputValidationError:
    wrapped = InputValidationError(
        f"Malformed Cy-ExM TIFF geometry metadata: {message}"
    )
    if error is not None:
        wrapped.__cause__ = error
    return wrapped


def _geometry_from_shaped_metadata(
    metadata: object,
    data_shape: tuple[int, ...],
) -> ImageGeometry | None:
    if not isinstance(metadata, dict) or "cyexm" not in metadata:
        return None
    try:
        if metadata.get("axes") != "ZYX":
            raise ValueError("axes must be exactly ZYX")
        payload = metadata["cyexm"]
        if not isinstance(payload, dict):
            raise TypeError("cyexm must be a JSON object")
        if set(payload) != _CYEXM_TIFF_KEYS:
            raise ValueError(
                "cyexm keys must be exactly: " + ", ".join(sorted(_CYEXM_TIFF_KEYS))
            )
        schema_version = payload["schema_version"]
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != _TIFF_GEOMETRY_SCHEMA_VERSION
        ):
            raise ValueError(f"schema_version must be {_TIFF_GEOMETRY_SCHEMA_VERSION}")
        if payload["axis_convention_version"] != AXIS_CONVENTION_VERSION:
            raise ValueError(
                f"axis_convention_version must be {AXIS_CONVENTION_VERSION}"
            )
        geometry_payload = payload["geometry"]
        if not isinstance(geometry_payload, dict):
            raise TypeError("geometry must be a JSON object")
        geometry = ImageGeometry.from_dict(geometry_payload)
        if len(data_shape) != 3 or geometry.shape != data_shape:
            raise ValueError(
                f"geometry shape {geometry.shape} does not match "
                f"TIFF shape {data_shape}"
            )
    except (KeyError, TypeError, ValueError) as error:
        raise _metadata_error(str(error), error) from error
    return geometry


def _malformed_description_has_cyexm_marker(description: str) -> bool:
    """Detect semantic lowercase cyexm text without broadly decoding malformed data.

    Only unescaped JSON ``\\uXXXX`` sequences for the five ASCII marker letters are
    normalized. Other escapes and escaped backslashes remain untouched.
    """

    def replace_target_escape(match: re.Match[str]) -> str:
        backslashes, hexadecimal = match.groups()
        if len(backslashes) % 2 == 0:
            return match.group(0)
        letter = _CYEXM_ASCII_CODEPOINTS.get(int(hexadecimal, 16))
        if letter is None:
            return match.group(0)
        return backslashes[:-1] + letter

    normalized = _JSON_UNICODE_ESCAPE.sub(replace_target_escape, description)
    return "cyexm" in normalized


def _geometry_metadata_from_description(
    description: object,
) -> dict[str, object] | None:
    if not isinstance(description, str):
        return None
    try:
        decoded = json.loads(description)
    except json.JSONDecodeError as error:
        if _malformed_description_has_cyexm_marker(description):
            raise _metadata_error(
                "raw malformed ImageDescription contains a semantic cyexm marker",
                error,
            ) from error
        return None
    if isinstance(decoded, dict) and "cyexm" in decoded:
        return decoded
    return None


def read_tiff_geometry(
    path: str | Path,
    data_shape: tuple[int, ...],
) -> ImageGeometry | None:
    """Return strict Cy-ExM geometry, or None only when the custom key is absent."""
    import tifffile

    try:
        with tifffile.TiffFile(path) as handle:
            series = handle.series[0]
            series_axes = series.axes
            shaped = handle.shaped_metadata
            metadata = shaped[0] if shaped else None
            description = series.keyframe.description
            selected_pages = tuple(
                page.aspage() if isinstance(page, tifffile.TiffFrame) else page
                for page in series.pages
            )
            scalar_grayscale_pages = all(
                page is not None
                and "S" not in page.axes
                and "C" not in page.axes
                and page.samplesperpixel == 1
                and page.photometric
                in {
                    tifffile.PHOTOMETRIC.MINISBLACK,
                    tifffile.PHOTOMETRIC.MINISWHITE,
                }
                for page in selected_pages
            )
    except (OSError, ValueError) as error:
        raise InputValidationError(
            f"TIFF metadata could not be read: {path}"
        ) from error

    if not isinstance(metadata, dict) or "cyexm" not in metadata:
        raw_metadata = _geometry_metadata_from_description(description)
        if raw_metadata is not None:
            metadata = raw_metadata

    geometry = _geometry_from_shaped_metadata(metadata, data_shape)
    singleton_z = (
        series_axes == "YX"
        and len(data_shape) == 3
        and data_shape[0] == 1
        and isinstance(metadata, dict)
        and metadata.get("axes") == "ZYX"
    )
    if not scalar_grayscale_pages or (
        series_axes not in {"ZYX", "QYX"} and not singleton_z
    ):
        raise InputValidationError(
            "Unsupported TIFF spatial layout; expected ZYX, grayscale QYX, "
            "or reconstructed singleton-Z YX"
        )
    return geometry


def _metadata_for_geometry(geometry: ImageGeometry | None) -> dict[str, object]:
    metadata: dict[str, object] = {"axes": "ZYX"}
    if geometry is not None:
        metadata["cyexm"] = {
            "schema_version": _TIFF_GEOMETRY_SCHEMA_VERSION,
            "axis_convention_version": AXIS_CONVENTION_VERSION,
            "geometry": geometry.to_dict(),
        }
    return metadata


def _range_value(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InputValidationError(
            "TIFF output range must contain finite uint16-compatible values"
        )
    try:
        number = float(value)
    except OverflowError as error:
        raise InputValidationError(
            "TIFF output range must contain finite uint16-compatible values"
        ) from error
    if not math.isfinite(number) or not 0.0 <= number <= 65535.0:
        raise InputValidationError(
            "TIFF output range must contain finite uint16-compatible values"
        )
    return number


def _tiff_destination(path: str | Path) -> Path:
    destination = Path(path)
    if destination.suffix.lower() not in {".tif", ".tiff"}:
        raise InputValidationError(
            "TIFF output path must use a .tif or .tiff extension"
        )
    return destination


def _external_zyx_data(image: Any) -> tuple[Any, ImageGeometry | None]:
    import numpy as np

    geometry: ImageGeometry | None = None
    if isinstance(image, np.ndarray):
        source = image
    else:
        import ants

        if not isinstance(image, ants.core.ants_image.ANTsImage):
            raise TypeError("image must be an ANTs image or NumPy array")
        from cyexm.io.images import ants_geometry, ants_to_external_array

        source = ants_to_external_array(image.numpy())
        geometry = ants_geometry(image)

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
    return source, geometry


def _write_encoded_uint16_tiff(
    encoded: Any,
    destination: Path,
    geometry: ImageGeometry | None,
) -> None:
    import numpy as np
    import tifffile

    metadata = _metadata_for_geometry(geometry)

    def writer(temporary: Path) -> None:
        tifffile.imwrite(
            temporary,
            encoded,
            metadata=metadata,
            photometric="minisblack",
        )

    def validator(temporary: Path) -> bool:
        try:
            with tifffile.TiffFile(temporary) as handle:
                series = handle.series[0]
                shaped = handle.shaped_metadata
                declared_axes = (
                    shaped[0].get("axes")
                    if shaped and isinstance(shaped[0], dict)
                    else None
                )
                singleton_z_is_squeezed = encoded.shape[0] == 1 and series.axes == "YX"
                if declared_axes != "ZYX" or (
                    series.axes != "ZYX" and not singleton_z_is_squeezed
                ):
                    return False
                written = handle.asarray()
                written_geometry = _geometry_from_shaped_metadata(
                    shaped[0] if shaped else None,
                    tuple(int(value) for value in written.shape),
                )
        except Exception:
            return False
        return (
            written.dtype == np.dtype(np.uint16)
            and written.shape == encoded.shape
            and np.array_equal(written, encoded)
            and (
                written_geometry is None
                if geometry is None
                else written_geometry is not None and written_geometry == geometry
            )
        )

    atomic_write(destination, writer, validator)


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

    destination = _tiff_destination(path)

    import numpy as np

    source, geometry = _external_zyx_data(image)

    if np.issubdtype(source.dtype, np.integer):
        data = np.array(source, copy=True)
        if np.issubdtype(data.dtype, np.signedinteger):
            np.maximum(data, 0, out=data)
        data_minimum = data.min()
        data_maximum = data.max()
    else:
        data = np.array(source, dtype=np.longdouble, copy=True)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        np.maximum(data, np.longdouble(0), out=data)
        data_minimum = data.min()
        data_maximum = data.max()
    if data_minimum == data_maximum:
        scaled = np.full(data.shape, minimum, dtype=np.longdouble)
    else:
        offsets = data - data_minimum
        normalized = offsets.astype(np.longdouble) / np.longdouble(
            data_maximum - data_minimum
        )
        scaled = minimum + normalized * (maximum - minimum)
    encoded = np.clip(scaled, minimum, maximum).astype(np.uint16)
    _write_encoded_uint16_tiff(encoded, destination, geometry)


def write_clipped_uint16_tiff(image: Any, path: str | Path) -> None:
    """Clip a real three-dimensional image to uint16 without normalization."""
    destination = _tiff_destination(path)

    import numpy as np

    source, geometry = _external_zyx_data(image)
    if np.issubdtype(source.dtype, np.floating):
        data = np.nan_to_num(source, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        data = source
    encoded = np.clip(data, 0, 65535).astype(np.uint16)
    _write_encoded_uint16_tiff(encoded, destination, geometry)


__all__ = [
    "read_tiff_geometry",
    "write_clipped_uint16_tiff",
    "write_uint16_tiff",
]
