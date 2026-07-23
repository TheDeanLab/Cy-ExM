from __future__ import annotations

import math
from dataclasses import dataclass, fields
from numbers import Integral, Real
from typing import Any


def _positive_int_triple(value: object, name: str) -> tuple[int, int, int]:
    if isinstance(value, str | bytes):
        raise TypeError(f"{name} must contain exactly three positive integers")
    try:
        values = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(
            f"{name} must contain exactly three positive integers"
        ) from error
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three positive integers")
    if any(isinstance(item, bool) or not isinstance(item, Integral) for item in values):
        raise TypeError(f"{name} must contain exactly three positive integers")
    normalized = tuple(int(item) for item in values)
    if any(item <= 0 for item in normalized):
        raise ValueError(f"{name} must contain exactly three positive integers")
    return normalized  # type: ignore[return-value]


def _validated_overlap(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("overlap_fraction must be a finite number")
    overlap = float(value)
    if not math.isfinite(overlap):
        raise ValueError("overlap_fraction must be finite")
    if overlap < 0.0 or overlap >= 1.0:
        raise ValueError("overlap_fraction must satisfy 0 <= value < 1")
    return overlap


def _grid_positive_int_triple(value: object, name: str) -> tuple[int, int, int]:
    values = tuple(value)  # type: ignore[arg-type]
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three integers")
    if any(isinstance(item, bool) or not isinstance(item, Integral) for item in values):
        raise TypeError(f"{name} must contain exactly three integers")
    normalized = tuple(int(item) for item in values)
    if any(item <= 0 for item in normalized):
        raise ValueError(f"{name} values must be positive")
    return normalized  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class ChunkInfo:
    chunk_id: int
    z_start: int
    z_end: int
    y_start: int
    y_end: int
    x_start: int
    x_end: int
    z_start_ext: int
    z_end_ext: int
    y_start_ext: int
    y_end_ext: int
    x_start_ext: int
    x_end_ext: int

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{item.name} must be an integer")
        if self.chunk_id < 0:
            raise ValueError("chunk_id must be a non-negative integer")
        for axis in "zyx":
            start = getattr(self, f"{axis}_start")
            end = getattr(self, f"{axis}_end")
            start_ext = getattr(self, f"{axis}_start_ext")
            end_ext = getattr(self, f"{axis}_end_ext")
            if start_ext < 0 or start < 0 or start >= end:
                raise ValueError(f"{axis} core bounds must be non-empty")
            if start_ext > start or end > end_ext:
                raise ValueError(f"{axis} extended bounds must contain the core")

    @property
    def core_shape(self) -> tuple[int, int, int]:
        return (
            self.z_end - self.z_start,
            self.y_end - self.y_start,
            self.x_end - self.x_start,
        )

    @property
    def extended_shape(self) -> tuple[int, int, int]:
        return (
            self.z_end_ext - self.z_start_ext,
            self.y_end_ext - self.y_start_ext,
            self.x_end_ext - self.x_start_ext,
        )

    @property
    def core_within_extended(self) -> tuple[slice, slice, slice]:
        z0 = self.z_start - self.z_start_ext
        y0 = self.y_start - self.y_start_ext
        x0 = self.x_start - self.x_start_ext
        return (
            slice(z0, z0 + self.core_shape[0]),
            slice(y0, y0 + self.core_shape[1]),
            slice(x0, x0 + self.core_shape[2]),
        )

    @property
    def extended_slices(self) -> tuple[slice, slice, slice]:
        return (
            slice(self.z_start_ext, self.z_end_ext),
            slice(self.y_start_ext, self.y_end_ext),
            slice(self.x_start_ext, self.x_end_ext),
        )

    @property
    def core_slices(self) -> tuple[slice, slice, slice]:
        return (
            slice(self.z_start, self.z_end),
            slice(self.y_start, self.y_end),
            slice(self.x_start, self.x_end),
        )


def chunk_grid(
    image_shape: object,
    chunk_size: object,
    overlap_fraction: float,
) -> tuple[dict[str, int], ...]:
    shape_z, shape_y, shape_x = _grid_positive_int_triple(image_shape, "image_shape")
    chunk_z, chunk_y, chunk_x = _grid_positive_int_triple(chunk_size, "chunk_size")
    overlap = _validated_overlap(overlap_fraction)
    overlap_z, overlap_y, overlap_x = (
        int(chunk_z * overlap),
        int(chunk_y * overlap),
        int(chunk_x * overlap),
    )
    step_z, step_y, step_x = (
        chunk_z - overlap_z,
        chunk_y - overlap_y,
        chunk_x - overlap_x,
    )
    z_positions = (0,) if shape_z <= chunk_z else tuple(range(0, shape_z, step_z))
    y_positions = (0,) if shape_y <= chunk_y else tuple(range(0, shape_y, step_y))
    x_positions = (0,) if shape_x <= chunk_x else tuple(range(0, shape_x, step_x))

    chunks: list[dict[str, int]] = []
    for z_start in z_positions:
        for y_start in y_positions:
            for x_start in x_positions:
                z_end = min(z_start + chunk_z, shape_z)
                y_end = min(y_start + chunk_y, shape_y)
                x_end = min(x_start + chunk_x, shape_x)
                chunks.append(
                    {
                        "chunk_id": len(chunks),
                        "z_start": z_start,
                        "z_end": z_end,
                        "y_start": y_start,
                        "y_end": y_end,
                        "x_start": x_start,
                        "x_end": x_end,
                        "z_start_ext": max(0, z_start - overlap_z // 2),
                        "z_end_ext": min(shape_z, z_end + overlap_z // 2),
                        "y_start_ext": max(0, y_start - overlap_y // 2),
                        "y_end_ext": min(shape_y, y_end + overlap_y // 2),
                        "x_start_ext": max(0, x_start - overlap_x // 2),
                        "x_end_ext": min(shape_x, x_end + overlap_x // 2),
                    }
                )
    return tuple(chunks)


def compute_chunk_grid(
    image_shape: tuple[int, int, int] | list[int],
    chunk_size: tuple[int, int, int] | list[int] = (512, 512, 512),
    overlap_fraction: float = 0.15,
) -> tuple[ChunkInfo, ...]:
    return tuple(
        ChunkInfo(**item)
        for item in chunk_grid(image_shape, chunk_size, overlap_fraction)
    )


def _array_shape(array: object, name: str) -> tuple[int, ...]:
    shape = getattr(array, "shape", None)
    if not isinstance(shape, tuple | list):
        raise TypeError(f"{name} must expose an array shape")
    if len(shape) not in {3, 4}:
        raise ValueError(f"{name} must have three spatial dimensions")
    spatial = _positive_int_triple(shape[:3], f"{name} spatial shape")
    if len(shape) == 4:
        component_count = shape[3]
        if (
            isinstance(component_count, bool)
            or not isinstance(component_count, Integral)
            or int(component_count) != 3
        ):
            raise ValueError(f"{name} warp data must have exactly three components")
        return (*spatial, 3)
    return spatial


def extract_chunk(array: object, chunk: ChunkInfo) -> Any:
    shape = _array_shape(array, "array")
    if any(
        stop > size
        for stop, size in zip(
            (chunk.z_end_ext, chunk.y_end_ext, chunk.x_end_ext),
            shape[:3],
            strict=True,
        )
    ):
        raise ValueError("chunk extended bounds exceed the array shape")
    try:
        extracted: Any = array[chunk.extended_slices]  # type: ignore[index]
    except (IndexError, TypeError) as error:
        raise TypeError("array must support three-dimensional slicing") from error
    extracted_shape = _array_shape(extracted, "extracted chunk")
    expected = (*chunk.extended_shape, *shape[3:])
    if extracted_shape != expected:
        raise ValueError(
            f"extracted chunk shape {extracted_shape} does not match {expected}"
        )
    copy = getattr(extracted, "copy", None)
    return copy() if callable(copy) else extracted


def stitch_core(destination: object, extended: object, chunk: ChunkInfo) -> None:
    destination_shape = _array_shape(destination, "destination")
    extended_shape = _array_shape(extended, "extended chunk")
    expected_extended = (*chunk.extended_shape, *destination_shape[3:])
    if extended_shape != expected_extended:
        raise ValueError(
            f"extended chunk shape {extended_shape} does not match {expected_extended}"
        )
    if any(
        stop > size
        for stop, size in zip(
            (chunk.z_end, chunk.y_end, chunk.x_end),
            destination_shape[:3],
            strict=True,
        )
    ):
        raise ValueError("chunk core bounds exceed the destination shape")
    core: Any = extended[chunk.core_within_extended]  # type: ignore[index]
    core_shape = _array_shape(core, "chunk core")
    expected_core = (*chunk.core_shape, *destination_shape[3:])
    if core_shape != expected_core:
        raise ValueError(
            f"chunk core shape {core_shape} does not match {expected_core}"
        )
    destination[chunk.core_slices] = core  # type: ignore[index]


__all__ = ["ChunkInfo", "compute_chunk_grid", "extract_chunk", "stitch_core"]
