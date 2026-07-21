from __future__ import annotations

import json
import math
import os
import re
import stat
import struct
from dataclasses import dataclass
from io import BufferedReader
from itertools import pairwise
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
_TIFF_FIELD_SIZES = {
    1: 1,
    2: 1,
    3: 2,
    4: 4,
    5: 8,
    6: 1,
    7: 1,
    8: 2,
    9: 4,
    10: 8,
    11: 4,
    12: 8,
    13: 4,
    16: 8,
    17: 8,
    18: 8,
}
_MAX_TIFF_IFD_ENTRIES = 4096
_MAX_TIFF_FIELD_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class _TiffField:
    field_type: int
    count: int
    data: bytes


@dataclass(frozen=True, slots=True)
class _TiffFormat:
    byte_order: str
    header_size: int
    inline_size: int
    count_format: str
    count_size: int
    entry_size: int
    next_size: int


def _tiff_error(message: str) -> InputValidationError:
    return InputValidationError(f"Malformed Cy-ExM uint16 TIFF: {message}")


def _read_tiff_bytes(
    stream: BufferedReader,
    offset: int,
    size: int,
    file_size: int,
) -> bytes:
    _validate_tiff_byte_range(offset, size, file_size)
    stream.seek(offset)
    data = stream.read(size)
    if len(data) != size:
        raise _tiff_error(f"could not read {size} bytes at offset {offset}")
    return data


def _validate_tiff_byte_range(offset: int, size: int, file_size: int) -> None:
    if offset < 0 or size < 0 or offset > file_size - size:
        raise _tiff_error(
            f"byte range [{offset}, {offset + size}) is outside a {file_size}-byte file"
        )


def _tiff_format(header: bytes) -> tuple[_TiffFormat, int]:
    if len(header) < 8 or header[:2] not in {b"II", b"MM"}:
        raise _tiff_error("byte-order marker is invalid")
    byte_order = "<" if header[:2] == b"II" else ">"
    version = struct.unpack_from(f"{byte_order}H", header, 2)[0]
    if version == 42:
        return (
            _TiffFormat(byte_order, 8, 4, "H", 2, 12, 4),
            struct.unpack_from(f"{byte_order}I", header, 4)[0],
        )
    if version != 43:
        raise _tiff_error(f"unsupported TIFF version {version}")
    if len(header) < 16:
        raise _tiff_error("BigTIFF header is truncated")
    offset_size, reserved = struct.unpack_from(f"{byte_order}HH", header, 4)
    if offset_size != 8 or reserved != 0:
        raise _tiff_error("BigTIFF offset-size header is invalid")
    return (
        _TiffFormat(byte_order, 16, 8, "Q", 8, 20, 8),
        struct.unpack_from(f"{byte_order}Q", header, 8)[0],
    )


def _tiff_unsigned_values(
    field: _TiffField,
    tag: int,
    *,
    allowed_types: frozenset[int] = frozenset({1, 3, 4, 16}),
    byte_order: str,
) -> tuple[int, ...]:
    if field.field_type not in allowed_types:
        raise _tiff_error(f"tag {tag} has unsupported field type {field.field_type}")
    formats = {1: "B", 3: "H", 4: "I", 16: "Q"}
    code = formats[field.field_type]
    return tuple(
        item[0] for item in struct.iter_unpack(f"{byte_order}{code}", field.data)
    )


def _tiff_scalar(
    fields: dict[int, _TiffField],
    tag: int,
    *,
    byte_order: str,
    allowed_types: frozenset[int] = frozenset({1, 3, 4, 16}),
) -> int:
    try:
        field = fields[tag]
    except KeyError as error:
        raise _tiff_error(f"required tag {tag} is absent") from error
    values = _tiff_unsigned_values(
        field,
        tag,
        allowed_types=allowed_types,
        byte_order=byte_order,
    )
    if len(values) != 1:
        raise _tiff_error(f"tag {tag} must contain exactly one value")
    return values[0]


def _strict_tiff_json(text: str) -> object:
    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        decoded: dict[str, object] = {}
        for key, value in pairs:
            if key in decoded:
                raise ValueError(f"duplicate JSON key {key!r}")
            decoded[key] = value
        return decoded

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant {value}")

    return json.loads(
        text,
        object_pairs_hook=object_pairs,
        parse_constant=reject_constant,
    )


def _geometry_from_tiff_description(field: _TiffField) -> ImageGeometry:
    if field.field_type != 2 or field.count < 2:
        raise _tiff_error("ImageDescription must be a nonempty ASCII value")
    if not field.data.endswith(b"\0") or b"\0" in field.data[:-1]:
        raise _tiff_error("ImageDescription must have exactly one trailing NUL")
    try:
        description = field.data[:-1].decode("ascii")
        metadata = _strict_tiff_json(description)
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise _tiff_error(f"ImageDescription JSON is invalid: {error}") from error
    if not isinstance(metadata, dict) or set(metadata) != {"shape", "axes", "cyexm"}:
        raise _tiff_error(
            "ImageDescription keys must be exactly axes, cyexm, and shape"
        )
    raw_shape = metadata["shape"]
    if (
        not isinstance(raw_shape, list)
        or len(raw_shape) != 3
        or any(type(value) is not int or value <= 0 for value in raw_shape)
    ):
        raise _tiff_error("ImageDescription shape must contain three positive integers")
    data_shape = (raw_shape[0], raw_shape[1], raw_shape[2])
    try:
        geometry = _geometry_from_shaped_metadata(metadata, data_shape)
    except InputValidationError as error:
        raise _tiff_error(str(error)) from error
    if geometry is None:
        raise _tiff_error("ImageDescription lacks Cy-ExM geometry")
    return geometry


def _validate_nonoverlapping_ranges(
    structural: list[tuple[int, int]],
    pixels: list[tuple[int, int]],
) -> None:
    # TIFF permits multiple tags to reference the same immutable metadata value
    # (tifffile shares X/Y resolution rationals across later pages).
    ordered_structural = sorted(set(structural))
    for previous, current in pairwise(ordered_structural):
        if current[0] < previous[1]:
            raise _tiff_error("TIFF structural byte ranges overlap")
    ordered_pixels = sorted(pixels)
    for previous, current in pairwise(ordered_pixels):
        if current[0] < previous[1]:
            raise _tiff_error("TIFF pixel byte ranges overlap")
    structural_index = 0
    for pixel_start, pixel_end in ordered_pixels:
        while (
            structural_index < len(ordered_structural)
            and ordered_structural[structural_index][1] <= pixel_start
        ):
            structural_index += 1
        if (
            structural_index < len(ordered_structural)
            and ordered_structural[structural_index][0] < pixel_end
        ):
            raise _tiff_error("TIFF pixel data overlaps structural metadata")


def _read_uint16_tiff_geometry_without_runtime(path: str | Path) -> ImageGeometry:
    """Read exact uncompressed Cy-ExM TIFF structure without scientific runtimes."""
    source = Path(path)
    try:
        with source.open("rb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                raise _tiff_error("source is not a regular file")
            file_size = metadata.st_size
            header = _read_tiff_bytes(stream, 0, min(file_size, 16), file_size)
            tiff_format, next_ifd = _tiff_format(header)
            structural_ranges = [(0, tiff_format.header_size)]
            pixel_ranges: list[tuple[int, int]] = []
            seen_ifds: set[int] = set()
            geometry: ImageGeometry | None = None
            page_number = 0
            while next_ifd:
                if next_ifd in seen_ifds:
                    raise _tiff_error("IFD chain contains a cycle")
                seen_ifds.add(next_ifd)
                count_bytes = _read_tiff_bytes(
                    stream,
                    next_ifd,
                    tiff_format.count_size,
                    file_size,
                )
                entry_count = struct.unpack(
                    f"{tiff_format.byte_order}{tiff_format.count_format}",
                    count_bytes,
                )[0]
                if entry_count == 0 or entry_count > _MAX_TIFF_IFD_ENTRIES:
                    raise _tiff_error(f"IFD entry count {entry_count} is invalid")
                ifd_size = (
                    tiff_format.count_size
                    + entry_count * tiff_format.entry_size
                    + tiff_format.next_size
                )
                ifd_data = _read_tiff_bytes(stream, next_ifd, ifd_size, file_size)
                structural_ranges.append((next_ifd, next_ifd + ifd_size))
                fields: dict[int, _TiffField] = {}
                for index in range(entry_count):
                    offset = tiff_format.count_size + index * tiff_format.entry_size
                    tag, field_type = struct.unpack_from(
                        f"{tiff_format.byte_order}HH", ifd_data, offset
                    )
                    if tag in fields:
                        raise _tiff_error(f"IFD contains duplicate tag {tag}")
                    try:
                        field_size = _TIFF_FIELD_SIZES[field_type]
                    except KeyError as error:
                        raise _tiff_error(
                            f"tag {tag} has unknown field type {field_type}"
                        ) from error
                    if tiff_format.inline_size == 4:
                        count = struct.unpack_from(
                            f"{tiff_format.byte_order}I", ifd_data, offset + 4
                        )[0]
                        slot_offset = offset + 8
                    else:
                        count = struct.unpack_from(
                            f"{tiff_format.byte_order}Q", ifd_data, offset + 4
                        )[0]
                        slot_offset = offset + 12
                    total_size = count * field_size
                    if count == 0 or total_size > _MAX_TIFF_FIELD_BYTES:
                        raise _tiff_error(
                            f"tag {tag} field size {total_size} is invalid"
                        )
                    slot = ifd_data[slot_offset : slot_offset + tiff_format.inline_size]
                    if total_size <= tiff_format.inline_size:
                        field_data = slot[:total_size]
                    else:
                        value_offset = int.from_bytes(
                            slot,
                            byteorder=(
                                "little" if tiff_format.byte_order == "<" else "big"
                            ),
                        )
                        field_data = _read_tiff_bytes(
                            stream,
                            value_offset,
                            total_size,
                            file_size,
                        )
                        structural_ranges.append(
                            (value_offset, value_offset + total_size)
                        )
                    fields[tag] = _TiffField(field_type, count, field_data)

                next_offset = ifd_size - tiff_format.next_size
                next_ifd = int.from_bytes(
                    ifd_data[next_offset:],
                    byteorder=("little" if tiff_format.byte_order == "<" else "big"),
                )
                if page_number == 0:
                    try:
                        description = fields[270]
                    except KeyError as error:
                        raise _tiff_error("first IFD lacks ImageDescription") from error
                    geometry = _geometry_from_tiff_description(description)
                elif 270 in fields:
                    raise _tiff_error("only the first IFD may contain ImageDescription")
                assert geometry is not None
                if page_number >= geometry.shape[0]:
                    raise _tiff_error("TIFF contains more pages than declared")
                width = _tiff_scalar(fields, 256, byte_order=tiff_format.byte_order)
                height = _tiff_scalar(fields, 257, byte_order=tiff_format.byte_order)
                if (height, width) != geometry.shape[1:]:
                    raise _tiff_error(
                        f"page shape {(height, width)!r} does not match geometry"
                    )
                required_scalars = {
                    258: 16,
                    259: 1,
                    277: 1,
                }
                for tag, expected in required_scalars.items():
                    if (
                        _tiff_scalar(
                            fields,
                            tag,
                            byte_order=tiff_format.byte_order,
                        )
                        != expected
                    ):
                        raise _tiff_error(f"tag {tag} must be {expected}")
                if _tiff_scalar(fields, 262, byte_order=tiff_format.byte_order) not in {
                    0,
                    1,
                }:
                    raise _tiff_error("photometric interpretation is not grayscale")
                if (
                    284 in fields
                    and _tiff_scalar(fields, 284, byte_order=tiff_format.byte_order)
                    != 1
                ):
                    raise _tiff_error("planar configuration must be chunky")
                if (
                    339 in fields
                    and _tiff_scalar(fields, 339, byte_order=tiff_format.byte_order)
                    != 1
                ):
                    raise _tiff_error("sample format must be unsigned integer")
                if 324 in fields or 325 in fields:
                    raise _tiff_error("tiled TIFF data is not supported")
                rows_per_strip = _tiff_scalar(
                    fields, 278, byte_order=tiff_format.byte_order
                )
                if rows_per_strip <= 0:
                    raise _tiff_error("RowsPerStrip must be positive")
                offsets = _tiff_unsigned_values(
                    fields[273],
                    273,
                    allowed_types=frozenset({3, 4, 16}),
                    byte_order=tiff_format.byte_order,
                )
                byte_counts = _tiff_unsigned_values(
                    fields[279],
                    279,
                    allowed_types=frozenset({3, 4, 16}),
                    byte_order=tiff_format.byte_order,
                )
                strip_count = (height + rows_per_strip - 1) // rows_per_strip
                if len(offsets) != strip_count or len(byte_counts) != strip_count:
                    raise _tiff_error("strip offset/count cardinality is invalid")
                for strip, (data_offset, byte_count) in enumerate(
                    zip(offsets, byte_counts, strict=True)
                ):
                    rows = min(rows_per_strip, height - strip * rows_per_strip)
                    expected_bytes = rows * width * 2
                    if byte_count != expected_bytes:
                        raise _tiff_error(
                            f"strip byte count {byte_count} is not {expected_bytes}"
                        )
                    _validate_tiff_byte_range(data_offset, byte_count, file_size)
                    pixel_ranges.append((data_offset, data_offset + byte_count))
                page_number += 1
            if geometry is None or page_number != geometry.shape[0]:
                raise _tiff_error("TIFF page count does not match declared Z shape")
            _validate_nonoverlapping_ranges(structural_ranges, pixel_ranges)
            return geometry
    except OSError as error:
        raise _tiff_error(f"could not read {source}: {error}") from error


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
