from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

type PixelSize = tuple[float, float, float]
type Shape3D = tuple[int, int, int]
type Direction3D = tuple[PixelSize, PixelSize, PixelSize]


def _float_triple(values: Any) -> PixelSize:
    x, y, z = values
    return float(x), float(y), float(z)


def _int_triple(values: Any) -> Shape3D:
    z, y, x = values
    return int(z), int(y), int(x)


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
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        direction = payload["direction"]
        return cls(
            shape=_int_triple(payload["shape"]),
            origin=_float_triple(payload["origin"]),
            spacing=_float_triple(payload["spacing"]),
            direction=(
                _float_triple(direction[0]),
                _float_triple(direction[1]),
                _float_triple(direction[2]),
            ),
        )


__all__ = ["ImageGeometry", "PixelSize"]
