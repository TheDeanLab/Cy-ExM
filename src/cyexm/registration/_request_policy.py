"""Lightweight canonical request policy shared by producers and readers."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Literal

from cyexm._nonlinear_metric import resolve_nonlinear_metric
from cyexm.spatial import AXIS_CONVENTION_VERSION, ImageGeometry, PixelSize

WHOLE_SCALAR_POLICY = "whole-global-histogram-match-v1"
CHUNK_SCALAR_POLICY = "serial-chunk-local-histogram-match-core-stitch-v1"
OUTPUT_FORMAT_POLICY = "ome-compatible-tiff-uint16-v1"
CHANNEL_POSTPROCESSING_POLICY = "clip-uint16-no-normalization-v1"

_LINEAR_ITERATIONS = {
    "high": [1000, 500, 250, 100],
    "medium": [500, 250, 100, 50],
    "low": [100, 50, 25, 10],
}
_NONLINEAR_ITERATIONS = {
    "high": [100, 70, 50],
    "medium": [50, 35, 25],
    "low": [5, 5, 5],
    "dry run": [1, 1, 1],
}


def canonical_compact_json(value: object) -> str:
    """Serialize one canonical compact JSON artifact with a trailing newline."""
    return (
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
    )


def common_parameters(
    *, crop: bool, imaging_round: int, pixel_size_xyz: PixelSize
) -> dict[str, object]:
    """Return evidence common to every Phase 1 registration stage."""
    return {
        "axis_convention_version": AXIS_CONVENTION_VERSION,
        "crop": crop,
        "imaging_round": imaging_round,
        "pixel_size_xyz": list(pixel_size_xyz),
    }


def crop_parameters(role: Literal["fixed", "moving"]) -> dict[str, object]:
    """Return canonical robust-crop stage parameters."""
    return {
        "down_sampling": 4,
        "image_role": role,
        "lower_pct": 0.1,
        "upper_pct": 99.9,
    }


def reference_crop_parameters() -> dict[str, object]:
    return {
        "policy": "reference-crop-v1",
        "down_sampling": 4,
        "lower_pct": 0.1,
        "upper_pct": 99.9,
    }


def reference_channel_parameters(*, index: int, cropped: bool) -> dict[str, object]:
    return {
        "policy": "reference-channel-v1",
        "channel_index": index,
        "cropped": cropped,
        "axis_convention": AXIS_CONVENTION_VERSION,
        "output_format": OUTPUT_FORMAT_POLICY,
        "postprocessing": CHANNEL_POSTPROCESSING_POLICY,
    }


def fixed_reference_parameters(minimum: float, maximum: float) -> dict[str, object]:
    """Return canonical fixed-reference persistence parameters."""
    return {"minimum": minimum, "maximum": maximum}


def linear_parameters(linear_type: str, accuracy: str) -> dict[str, object]:
    """Return the canonical Phase 1 affine registration parameter evidence."""
    parameters: dict[str, object] = {
        "aff_metric": "mattes",
        "aff_random_sampling_rate": 0.5,
        "aff_sampling": 32,
        "mask_all_stages": True,
        "verbose": True,
    }
    if linear_type == "TRSAA" and accuracy != "dry run":
        parameters["reg_iterations"] = _LINEAR_ITERATIONS[accuracy]
    if accuracy == "dry run":
        parameters.update(
            effective_registration_type="Rigid",
            aff_iterations=[1],
            aff_smoothing_sigmas=[0],
            aff_shrink_factors=[8],
            aff_random_sampling_rate=0.1,
        )
    return parameters


def affine_parameters(
    *, moving_geometry: ImageGeometry, linear_type: str, accuracy: str
) -> dict[str, object]:
    """Return canonical affine-stage-specific parameters."""
    return {
        "image_interpolation": "linear",
        "mask_interpolation": "nearestNeighbor",
        "moving_geometry": moving_geometry.to_dict(),
        "registration_parameters": linear_parameters(linear_type, accuracy),
    }


def nonlinear_parameters(
    accuracy: str,
    metric: object = "mattes",
    radius: object = None,
) -> dict[str, object]:
    """Return the canonical Phase 1 nonlinear registration parameter evidence."""
    resolved = resolve_nonlinear_metric(metric, radius)
    return {
        "aff_random_sampling_rate": 1.0,
        "flow_sigma": 3.0,
        "initial_transform": "Identity",
        "mask_all_stages": True,
        "reg_iterations": _NONLINEAR_ITERATIONS[accuracy],
        "singleprecision": False,
        "syn_metric": resolved.metric,
        "syn_sampling": resolved.syn_sampling,
        "total_sigma": 0.5,
        "verbose": True,
    }


def nonlinear_stage_parameters(
    *,
    moving_geometry: ImageGeometry,
    registration_mode: Literal["whole-image", "serial-chunked"],
    nonlinear_accuracy: str,
    nonlinear_metric: object = "mattes",
    nonlinear_radius: object = None,
    scalar_postprocessing_policy: str,
    evidence: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return canonical full nonlinear-stage parameters."""
    parameters: dict[str, object] = {
        "image_interpolation": "linear",
        "mask_interpolation": "nearestNeighbor",
        "moving_geometry": moving_geometry.to_dict(),
        "registration_mode": registration_mode,
        "registration_parameters": nonlinear_parameters(
            nonlinear_accuracy, nonlinear_metric, nonlinear_radius
        ),
        "scalar_postprocessing_policy": scalar_postprocessing_policy,
    }
    if evidence is not None:
        parameters.update(evidence)
    return parameters


def registered_image_parameters(
    *, minimum: float, maximum: float, scalar_postprocessing_policy: str
) -> dict[str, object]:
    """Return canonical registered-image persistence parameters."""
    return {
        "format": "uint16-tiff",
        "minimum": minimum,
        "maximum": maximum,
        "scalar_postprocessing_policy": scalar_postprocessing_policy,
    }


def channel_transform_parameters(
    *, pixel_size_xyz: PixelSize, interpolation: str
) -> dict[str, object]:
    """Return canonical transformed-channel parameters."""
    return {
        "pixel_size_xyz_um": list(pixel_size_xyz),
        "interpolation": interpolation,
        "transform_order": ["warp", "affine"],
        "whichtoinvert": [False, False],
        "axis_convention": AXIS_CONVENTION_VERSION,
        "output_format": OUTPUT_FORMAT_POLICY,
        "postprocessing": CHANNEL_POSTPROCESSING_POLICY,
    }


def scalar_policy(strategy: Literal["whole_image", "serial_chunked"]) -> str:
    """Return the Phase 1 registered-scalar policy for a workflow strategy."""
    return WHOLE_SCALAR_POLICY if strategy == "whole_image" else CHUNK_SCALAR_POLICY


def _positive_int_triple(value: Sequence[int], name: str) -> tuple[int, int, int]:
    values = tuple(value)
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three integers")
    if any(isinstance(item, bool) or not isinstance(item, Integral) for item in values):
        raise TypeError(f"{name} must contain exactly three integers")
    normalized = tuple(int(item) for item in values)
    if any(item <= 0 for item in normalized):
        raise ValueError(f"{name} values must be positive")
    return normalized[0], normalized[1], normalized[2]


def _overlap_fraction(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("overlap_fraction must be a finite number")
    overlap = float(value)
    if not math.isfinite(overlap):
        raise ValueError("overlap_fraction must be finite")
    if overlap < 0.0 or overlap >= 1.0:
        raise ValueError("overlap_fraction must satisfy 0 <= value < 1")
    return overlap


def chunk_grid(
    image_shape_zyx: Sequence[int],
    chunk_size_zyx: Sequence[int],
    overlap_fraction: float,
) -> tuple[dict[str, int], ...]:
    """Derive the exact canonical serial chunk grid."""
    shape_z, shape_y, shape_x = _positive_int_triple(image_shape_zyx, "image_shape")
    chunk_z, chunk_y, chunk_x = _positive_int_triple(chunk_size_zyx, "chunk_size")
    overlap = _overlap_fraction(overlap_fraction)
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


def chunk_bounds(chunk: Mapping[str, int]) -> tuple[int, int, int, int, int, int]:
    """Return one chunk's extended ZYX bounds."""
    return (
        chunk["z_start_ext"],
        chunk["z_end_ext"],
        chunk["y_start_ext"],
        chunk["y_end_ext"],
        chunk["x_start_ext"],
        chunk["x_end_ext"],
    )


def crop_geometry(
    geometry: ImageGeometry,
    bounds_zyx: Sequence[int],
) -> ImageGeometry:
    """Derive the exact ANTs physical geometry for a ZYX crop."""
    bounds = tuple(bounds_zyx)
    if len(bounds) != 6 or any(
        isinstance(item, bool) or not isinstance(item, Integral) for item in bounds
    ):
        raise TypeError("crop bounds must contain exactly six integers")
    z0, z1, y0, y1, x0, x1 = (int(item) for item in bounds)
    for (start, stop), size in zip(
        ((z0, z1), (y0, y1), (x0, x1)), geometry.shape, strict=True
    ):
        if start < 0 or stop > size or start >= stop:
            raise ValueError("crop bounds must define a non-empty image region")
    offset_xyz = (
        x0 * geometry.spacing[0],
        y0 * geometry.spacing[1],
        z0 * geometry.spacing[2],
    )
    physical_offset = (
        sum(
            coefficient * offset
            for coefficient, offset in zip(
                geometry.direction[0], offset_xyz, strict=True
            )
        ),
        sum(
            coefficient * offset
            for coefficient, offset in zip(
                geometry.direction[1], offset_xyz, strict=True
            )
        ),
        sum(
            coefficient * offset
            for coefficient, offset in zip(
                geometry.direction[2], offset_xyz, strict=True
            )
        ),
    )
    return ImageGeometry(
        shape=(z1 - z0, y1 - y0, x1 - x0),
        origin=(
            geometry.origin[0] + physical_offset[0],
            geometry.origin[1] + physical_offset[1],
            geometry.origin[2] + physical_offset[2],
        ),
        spacing=geometry.spacing,
        direction=geometry.direction,
    )


def local_chunk_parameters(
    *,
    chunk: Mapping[str, int],
    chunks: Sequence[Mapping[str, int]],
    chunk_size_zyx: Sequence[int],
    fixed_mask_geometry: ImageGeometry | None,
    image_shape_zyx: Sequence[int],
    moving_geometry: ImageGeometry,
    moving_mask_geometry: ImageGeometry | None,
    overlap_fraction: float,
    nonlinear_accuracy: str,
    nonlinear_metric: object = "mattes",
    nonlinear_radius: object = None,
    scalar_postprocessing_policy: str,
) -> dict[str, object]:
    """Return canonical local-warp parameters."""
    parameters: dict[str, object] = {
        "axis_convention_version": AXIS_CONVENTION_VERSION,
        "chunk": dict(chunk),
        "chunk_grid": [dict(item) for item in chunks],
        "chunk_size_zyx": list(chunk_size_zyx),
        "fixed_mask_geometry": (
            None if fixed_mask_geometry is None else fixed_mask_geometry.to_dict()
        ),
        "image_interpolation": "linear",
        "image_shape_zyx": list(image_shape_zyx),
        "low_variance_policy": "identity-displacement-below-1e-6",
        "mask_interpolation": "nearestNeighbor+threshold-0.5",
        "moving_geometry": moving_geometry.to_dict(),
        "moving_mask_geometry": (
            None if moving_mask_geometry is None else moving_mask_geometry.to_dict()
        ),
        "overlap_fraction": overlap_fraction,
        "registration_parameters": nonlinear_parameters(
            nonlinear_accuracy, nonlinear_metric, nonlinear_radius
        ),
        "scalar_postprocessing_policy": scalar_postprocessing_policy,
    }
    return parameters


def chunk_manifest_payload(
    *,
    round_number: int,
    image_shape_zyx: Sequence[int],
    chunk_size_zyx: Sequence[int],
    overlap_fraction: float,
    chunks: Sequence[Mapping[str, int]],
    local_warps: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Return the canonical schema-v2 serial chunk manifest payload."""
    return {
        "axis_convention_version": AXIS_CONVENTION_VERSION,
        "chunk_size_zyx": list(chunk_size_zyx),
        "chunks": [
            {"chunk": dict(chunk), "warp": dict(warp)}
            for chunk, warp in zip(chunks, local_warps, strict=True)
        ],
        "image_shape_zyx": list(image_shape_zyx),
        "overlap_fraction": overlap_fraction,
        "round_number": round_number,
        "schema_version": 2,
    }


def chunk_manifest_parameters(
    *,
    chunk_size_zyx: Sequence[int],
    image_shape_zyx: Sequence[int],
    local_warps: Sequence[Mapping[str, object]],
    nonlinear_accuracy: str,
    nonlinear_type: str,
    nonlinear_metric: object = "mattes",
    nonlinear_radius: object = None,
    overlap_fraction: float,
) -> dict[str, object]:
    """Return canonical chunk-manifest completion parameters."""
    resolved = resolve_nonlinear_metric(nonlinear_metric, nonlinear_radius)
    parameters: dict[str, object] = {
        "axis_convention_version": AXIS_CONVENTION_VERSION,
        "chunk_size_zyx": list(chunk_size_zyx),
        "image_shape_zyx": list(image_shape_zyx),
        "local_warps": [dict(identity) for identity in local_warps],
        "nonlinear_accuracy": nonlinear_accuracy,
        "nonlinear_type": nonlinear_type,
        "overlap_fraction": overlap_fraction,
    }
    if resolved.metric == "CC":
        parameters.update(
            nonlinear_metric=resolved.metric,
            nonlinear_sampling=resolved.syn_sampling,
        )
    return parameters


__all__ = [
    "CHANNEL_POSTPROCESSING_POLICY",
    "CHUNK_SCALAR_POLICY",
    "OUTPUT_FORMAT_POLICY",
    "WHOLE_SCALAR_POLICY",
    "affine_parameters",
    "canonical_compact_json",
    "channel_transform_parameters",
    "chunk_bounds",
    "chunk_grid",
    "chunk_manifest_parameters",
    "chunk_manifest_payload",
    "common_parameters",
    "crop_geometry",
    "crop_parameters",
    "fixed_reference_parameters",
    "linear_parameters",
    "local_chunk_parameters",
    "nonlinear_parameters",
    "nonlinear_stage_parameters",
    "registered_image_parameters",
    "scalar_policy",
]
