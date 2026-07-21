from __future__ import annotations

import json
from numbers import Integral
from pathlib import Path

from cyexm.io.images import Bounds3D, robust_bounding_box
from cyexm.registration._request_policy import canonical_compact_json
from cyexm.registration.backend import RegistrationBackend


def validate_bounds(value: object, shape: tuple[int, int, int]) -> Bounds3D:
    if isinstance(value, str | bytes):
        raise TypeError("crop bounds must contain exactly six integers")
    try:
        values = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError("crop bounds must contain exactly six integers") from error
    if len(values) != 6:
        raise ValueError("crop bounds must contain exactly six integers")
    if any(isinstance(item, bool) or not isinstance(item, Integral) for item in values):
        raise TypeError("crop bounds must contain exactly six integers")
    z0, z1, y0, y1, x0, x1 = (int(item) for item in values)
    pairs = ((z0, z1), (y0, y1), (x0, x1))
    if any(
        start < 0 or stop > size or start >= stop
        for (start, stop), size in zip(pairs, shape, strict=True)
    ):
        raise ValueError("crop bounds must define a non-empty image region")
    return z0, z1, y0, y1, x0, x1


def bounds_text(bounds: Bounds3D) -> str:
    return canonical_compact_json(list(bounds))


def read_bounds(path: Path, shape: tuple[int, int, int]) -> Bounds3D:
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)
    if not isinstance(payload, list):
        raise TypeError("crop bounds must be a JSON array")
    bounds = validate_bounds(payload, shape)
    if bounds_text(bounds) != text:
        raise ValueError("crop bounds artifact is not canonical")
    return bounds


def estimate_bounds(backend: RegistrationBackend, image: object) -> Bounds3D:
    estimated = robust_bounding_box(
        backend.to_array(image),
        down_sampling=4,
        lower_pct=0.1,
        upper_pct=99.9,
    )
    return validate_bounds(estimated, backend.geometry(image).shape)


__all__ = ("bounds_text", "estimate_bounds", "read_bounds", "validate_bounds")
