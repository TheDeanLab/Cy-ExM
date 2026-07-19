from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self

type PixelSize = tuple[float, float, float]
type Shape3D = tuple[int, int, int]
type Direction3D = tuple[PixelSize, PixelSize, PixelSize]

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
    return (
        _json_integer(values[0], "shape[0]"),
        _json_integer(values[1], "shape[1]"),
        _json_integer(values[2], "shape[2]"),
    )


def _float_triple(value: object, name: str) -> PixelSize:
    values = _triple_array(value, name)
    return (
        _finite_json_number(values[0], f"{name}[0]"),
        _finite_json_number(values[1], f"{name}[1]"),
        _finite_json_number(values[2], f"{name}[2]"),
    )


@dataclass(frozen=True, slots=True)
class ImageGeometry:
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
        direction = _triple_array(payload["direction"], "direction")
        return cls(
            shape=_shape_triple(payload["shape"]),
            origin=_float_triple(payload["origin"], "origin"),
            spacing=_float_triple(payload["spacing"], "spacing"),
            direction=(
                _float_triple(direction[0], "direction[0]"),
                _float_triple(direction[1], "direction[1]"),
                _float_triple(direction[2], "direction[2]"),
            ),
        )


__all__ = ["ImageGeometry", "PixelSize"]
