from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self

type PixelSize = tuple[float, float, float]
type Shape3D = tuple[int, int, int]
type Direction3D = tuple[PixelSize, PixelSize, PixelSize]

AXIS_CONVENTION_VERSION = "external-zyx-ants-xyz-v1"
TIFF_GEOMETRY_ATOL = 1e-7
_GEOMETRY_KEYS = frozenset({"shape", "origin", "spacing", "direction"})


def _triple_array(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON array")
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    return value


def _json_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a JSON integer")
    return value


def _finite_json_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a JSON number")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _shape_triple(value: object) -> Shape3D:
    values = _triple_array(value, "shape")
    shape = (
        _json_integer(values[0], "shape[0]"),
        _json_integer(values[1], "shape[1]"),
        _json_integer(values[2], "shape[2]"),
    )
    if any(item <= 0 for item in shape):
        raise ValueError("shape values must be positive")
    return shape


def _float_triple(value: object, name: str) -> PixelSize:
    values = _triple_array(value, name)
    return (
        _finite_json_number(values[0], f"{name}[0]"),
        _finite_json_number(values[1], f"{name}[1]"),
        _finite_json_number(values[2], f"{name}[2]"),
    )


def _direction_matrix(value: object) -> Direction3D:
    rows = _triple_array(value, "direction")
    direction = (
        _float_triple(rows[0], "direction[0]"),
        _float_triple(rows[1], "direction[1]"),
        _float_triple(rows[2], "direction[2]"),
    )
    scale = max(abs(item) for row in direction for item in row)
    if scale == 0.0:
        raise ValueError("direction must be non-singular")
    (a, b, c), (d, e, f), (g, h, i) = (
        tuple(item / scale for item in row) for row in direction
    )
    determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(determinant) <= 1e-12:
        raise ValueError("direction must be non-singular")
    return direction


@dataclass(frozen=True, slots=True)
class ImageGeometry:
    """Three-dimensional external shape and physical ANTs geometry.

    ``shape`` is ordered ZYX for NumPy/TIFF/chunk consumers. ``origin``,
    ``spacing``, and the rows and columns of ``direction`` remain physical XYZ.
    """

    shape: Shape3D
    origin: PixelSize
    spacing: PixelSize
    direction: Direction3D

    def to_dict(self) -> dict[str, object]:
        return {
            "shape": list(self.shape),
            "origin": list(self.origin),
            "spacing": list(self.spacing),
            "direction": [list(row) for row in self.direction],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        if set(payload) != _GEOMETRY_KEYS:
            raise ValueError(
                "Image geometry keys must be exactly: "
                + ", ".join(sorted(_GEOMETRY_KEYS))
            )
        spacing = _float_triple(payload["spacing"], "spacing")
        if any(value <= 0.0 for value in spacing):
            raise ValueError("spacing values must be positive")
        return cls(
            shape=_shape_triple(payload["shape"]),
            origin=_float_triple(payload["origin"], "origin"),
            spacing=spacing,
            direction=_direction_matrix(payload["direction"]),
        )


def geometry_compatible(
    actual: ImageGeometry,
    expected: ImageGeometry,
    *,
    relative_tolerance: float = 0.0,
    absolute_tolerance: float = TIFF_GEOMETRY_ATOL,
) -> bool:
    """Return whether two values describe the same serialized physical grid."""
    if actual.shape != expected.shape:
        return False

    def close(left: float, right: float) -> bool:
        return math.isclose(
            left,
            right,
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        )

    return (
        all(
            close(left, right)
            for left, right in zip(actual.origin, expected.origin, strict=True)
        )
        and all(
            close(left, right)
            for left, right in zip(actual.spacing, expected.spacing, strict=True)
        )
        and all(
            close(left, right)
            for actual_row, expected_row in zip(
                actual.direction,
                expected.direction,
                strict=True,
            )
            for left, right in zip(actual_row, expected_row, strict=True)
        )
    )


__all__ = [
    "AXIS_CONVENTION_VERSION",
    "TIFF_GEOMETRY_ATOL",
    "ImageGeometry",
    "PixelSize",
    "geometry_compatible",
]
