"""Deterministic spatial quality analysis for fixed-affine chunk experiments."""

from __future__ import annotations

import base64
import json
import math
import zlib
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import tifffile
from numpy.typing import NDArray
from scipy.ndimage import sobel
from scipy.signal import correlate

from cyexm.chunking import ChunkInfo, compute_chunk_grid
from cyexm.registration.artifacts import ArtifactStore, CompletionRecord, FileIdentity
from cyexm.workflow._round_provenance import RegistrationArtifactPaths
from hpc.environment import write_evidence_json

if TYPE_CHECKING:
    from hpc.fixed_affine_chunk_characterization import (
        FixedAffineCharacterizationSpec,
        ValidatedFixedAffineSource,
    )


@dataclass(frozen=True, slots=True)
class RoiResidual:
    """One exhaustive integer-shift gradient-NCC result."""

    correction_dy_px: int | None
    correction_dx_px: int | None
    residual_px: float | None
    current_ncc: float | None
    best_ncc: float | None
    runner_up_ncc: float | None
    prominence: float | None
    boundary_optimum: bool | None
    defined: bool

    @classmethod
    def undefined(cls) -> RoiResidual:
        return cls(None, None, None, None, None, None, None, None, False)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BlockResidual:
    """One fixed-window residual annotated with signal and seam context."""

    center_y: int
    center_x: int
    fixed_mean_intensity: float
    fixed_gradient_energy: float
    intensity_quartile: int
    seam_class: Literal["seam", "interior"]
    residual: RoiResidual

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["residual"] = self.residual.to_dict()
        return value


def _float_image(value: object, name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 1:
        raise ValueError(f"{name} must be a nonempty two-dimensional image")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _gradient_magnitude(image: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.hypot(
        sobel(image, axis=0, mode="reflect"),
        sobel(image, axis=1, mode="reflect"),
    )


def _window_sums(
    context: NDArray[np.float64],
    height: int,
    width: int,
) -> NDArray[np.float64]:
    integral = np.pad(context, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return (
        integral[height:, width:]
        - integral[:-height, width:]
        - integral[height:, :-width]
        + integral[:-height, :-width]
    )


def _ncc_surface(
    fixed_patch: NDArray[np.float64],
    moving_context: NDArray[np.float64],
) -> NDArray[np.float64]:
    height, width = fixed_patch.shape
    centered = fixed_patch - fixed_patch.mean()
    fixed_norm = float(np.linalg.norm(centered))
    if fixed_norm == 0.0:
        return np.full(
            (
                moving_context.shape[0] - height + 1,
                moving_context.shape[1] - width + 1,
            ),
            np.nan,
        )
    numerator = correlate(moving_context, centered, mode="valid", method="fft")
    sums = _window_sums(moving_context, height, width)
    square_sums = _window_sums(moving_context * moving_context, height, width)
    variance_terms = np.maximum(square_sums - (sums * sums) / (height * width), 0.0)
    denominator = fixed_norm * np.sqrt(variance_terms)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0.0,
    )


def gradient_ncc_search(
    fixed: object,
    moving: object,
    *,
    roi_yx: tuple[int, int, int, int],
    search_radius_px: int,
) -> RoiResidual:
    """Exhaustively find the integer correction applied to moving gradient data."""
    fixed_image = _float_image(fixed, "fixed")
    moving_image = _float_image(moving, "moving")
    if fixed_image.shape != moving_image.shape:
        raise ValueError("fixed and moving images must have identical shapes")
    if type(search_radius_px) is not int or search_radius_px < 0:
        raise ValueError("search_radius_px must be a nonnegative integer")
    if (
        type(roi_yx) is not tuple
        or len(roi_yx) != 4
        or any(type(value) is not int for value in roi_yx)
    ):
        raise ValueError("ROI must contain four exact integer bounds")
    y0, y1, x0, x1 = roi_yx
    radius = search_radius_px
    if (
        y0 < radius
        or x0 < radius
        or y0 >= y1
        or x0 >= x1
        or y1 + radius > fixed_image.shape[0]
        or x1 + radius > fixed_image.shape[1]
    ):
        raise ValueError("ROI is invalid or lacks complete search context")

    return _gradient_ncc_from_gradients(
        _gradient_magnitude(fixed_image),
        _gradient_magnitude(moving_image),
        roi_yx=roi_yx,
        search_radius_px=search_radius_px,
    )


def _gradient_ncc_from_gradients(
    fixed_gradient: NDArray[np.float64],
    moving_gradient: NDArray[np.float64],
    *,
    roi_yx: tuple[int, int, int, int],
    search_radius_px: int,
) -> RoiResidual:
    y0, y1, x0, x1 = roi_yx
    radius = search_radius_px
    fixed_patch = fixed_gradient[y0:y1, x0:x1]
    moving_context = moving_gradient[
        y0 - radius : y1 + radius,
        x0 - radius : x1 + radius,
    ]
    surface = _ncc_surface(fixed_patch, moving_context)
    if not np.isfinite(surface).any():
        return RoiResidual.undefined()

    positions = np.argwhere(np.isfinite(surface))
    ranked = sorted(
        (
            (
                float(surface[row, column]),
                -(abs(radius - int(row)) + abs(radius - int(column))),
                radius - int(row),
                radius - int(column),
            )
            for row, column in positions
        ),
        reverse=True,
    )
    best_ncc, _tie, dy, dx = ranked[0]
    runner_scores = [
        score
        for score, _distance, candidate_dy, candidate_dx in ranked[1:]
        if max(abs(candidate_dy - dy), abs(candidate_dx - dx)) > 2
    ]
    runner_up = max(runner_scores) if runner_scores else math.nan
    current = float(surface[radius, radius])
    return RoiResidual(
        correction_dy_px=dy,
        correction_dx_px=dx,
        residual_px=math.hypot(dy, dx),
        current_ncc=current if math.isfinite(current) else None,
        best_ncc=best_ncc,
        runner_up_ncc=runner_up if math.isfinite(runner_up) else None,
        prominence=(best_ncc - runner_up if math.isfinite(runner_up) else None),
        boundary_optimum=abs(dy) == radius or abs(dx) == radius,
        defined=True,
    )


def classify_seam_proximity(
    center_y: int,
    center_x: int,
    chunks: tuple[ChunkInfo, ...],
    *,
    distance_px: int = 16,
) -> Literal["seam", "interior"]:
    """Classify a YX point against internal canonical chunk core boundaries."""
    if not chunks:
        raise ValueError("chunks must be nonempty")
    if type(distance_px) is not int or distance_px < 0:
        raise ValueError("distance_px must be a nonnegative integer")
    shape_y = max(chunk.y_end for chunk in chunks)
    shape_x = max(chunk.x_end for chunk in chunks)
    # Assembly writes canonical cores in chunk-ID order. Later overlapping cores
    # replace earlier values, so the visible transition is each later core start,
    # not the earlier core end.
    y_boundaries = {
        chunk.y_start for chunk in chunks if chunk.y_start not in {0, shape_y}
    }
    x_boundaries = {
        chunk.x_start for chunk in chunks if chunk.x_start not in {0, shape_x}
    }
    if any(abs(center_y - value) <= distance_px for value in y_boundaries) or any(
        abs(center_x - value) <= distance_px for value in x_boundaries
    ):
        return "seam"
    return "interior"


def blockwise_residuals(
    fixed: object,
    moving: object,
    chunks: tuple[ChunkInfo, ...],
    *,
    search_radius_px: int = 30,
    window_size_px: int = 128,
    stride_px: int = 64,
    seam_distance_px: int = 16,
) -> tuple[BlockResidual, ...]:
    """Measure fixed-grid residuals on deterministic overlapping QC windows."""
    fixed_image = _float_image(fixed, "fixed")
    moving_image = _float_image(moving, "moving")
    if fixed_image.shape != moving_image.shape:
        raise ValueError("fixed and moving images must have identical shapes")
    if (
        type(window_size_px) is not int
        or window_size_px <= 0
        or type(stride_px) is not int
        or stride_px <= 0
    ):
        raise ValueError("window and stride must be positive integers")
    gradient = _gradient_magnitude(fixed_image)
    moving_gradient = _gradient_magnitude(moving_image)
    candidates: list[tuple[int, int, float, float, RoiResidual]] = []
    for y0 in range(
        search_radius_px,
        fixed_image.shape[0] - search_radius_px - window_size_px + 1,
        stride_px,
    ):
        for x0 in range(
            search_radius_px,
            fixed_image.shape[1] - search_radius_px - window_size_px + 1,
            stride_px,
        ):
            y1 = y0 + window_size_px
            x1 = x0 + window_size_px
            fixed_patch = fixed_image[y0:y1, x0:x1]
            gradient_patch = gradient[y0:y1, x0:x1]
            candidates.append(
                (
                    y0 + window_size_px // 2,
                    x0 + window_size_px // 2,
                    float(fixed_patch.mean()),
                    float(np.mean(gradient_patch * gradient_patch)),
                    _gradient_ncc_from_gradients(
                        gradient,
                        moving_gradient,
                        roi_yx=(y0, y1, x0, x1),
                        search_radius_px=search_radius_px,
                    ),
                )
            )
    if not candidates:
        raise ValueError("image is too small for the requested blockwise analysis")
    means = np.asarray([item[2] for item in candidates], dtype=np.float64)
    boundaries = np.percentile(means, (25, 50, 75))
    return tuple(
        BlockResidual(
            center_y=center_y,
            center_x=center_x,
            fixed_mean_intensity=mean,
            fixed_gradient_energy=energy,
            intensity_quartile=int(np.searchsorted(boundaries, mean, side="right") + 1),
            seam_class=classify_seam_proximity(
                center_y,
                center_x,
                chunks,
                distance_px=seam_distance_px,
            ),
            residual=residual,
        )
        for center_y, center_x, mean, energy, residual in candidates
    )


def summarize_jacobian_array(values: object) -> dict[str, float | int]:
    """Summarize determinant validity without discarding invalid observations."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Jacobian determinant array must be nonempty")
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError("Jacobian determinant array has no finite values")
    percentiles = np.percentile(finite, (1, 5, 50, 95, 99))
    nonpositive = int(np.count_nonzero(finite <= 0.0))
    return {
        "voxel_count": int(array.size),
        "finite_count": int(finite.size),
        "nonfinite_count": int(array.size - finite.size),
        "nonpositive_count": nonpositive,
        "nonpositive_fraction": nonpositive / int(finite.size),
        "minimum": float(finite.min()),
        "percentile_01": float(percentiles[0]),
        "percentile_05": float(percentiles[1]),
        "percentile_50": float(percentiles[2]),
        "percentile_95": float(percentiles[3]),
        "percentile_99": float(percentiles[4]),
        "maximum": float(finite.max()),
    }


def jacobian_summary(
    fixed_image: Path,
    warp: Path,
    *,
    voxel_spacing_xyz: tuple[float, float, float],
) -> dict[str, float | int]:
    """Compute ANTs' geometric Jacobian determinant on the exact fixed domain."""
    import ants

    from cyexm.io.images import load_ants_image

    domain = load_ants_image(fixed_image, voxel_spacing_xyz)
    determinant = ants.create_jacobian_determinant_image(
        domain,
        str(warp),
        do_log=False,
        geom=True,
    )
    return summarize_jacobian_array(determinant.numpy())


def _validated_artifact(path: Path) -> FileIdentity:
    record = CompletionRecord.read(ArtifactStore().record_path(path))
    current = FileIdentity.from_path(path)
    if current != record.artifact:
        raise ValueError(f"artifact identity does not match completion record: {path}")
    return current


def _mip(path: Path) -> NDArray[np.float64]:
    _validated_artifact(path)
    array = np.asarray(tifffile.imread(path))
    if array.ndim != 3:
        raise ValueError(f"registered image must be a ZYX volume: {path}")
    return np.asarray(array.max(axis=0), dtype=np.float64)


def _encoded_gradient_crop(
    image: NDArray[np.float64],
    roi: tuple[int, int, int, int],
) -> dict[str, object]:
    y0, y1, x0, x1 = roi
    crop = _gradient_magnitude(image)[y0:y1, x0:x1]
    low, high = np.percentile(crop, (1, 99))
    if high <= low:
        normalized = np.zeros(crop.shape, dtype=np.uint8)
    else:
        normalized = np.asarray(
            np.clip((crop - low) / (high - low), 0.0, 1.0) * 255.0,
            dtype=np.uint8,
        )
    return {
        "shape_yx": list(normalized.shape),
        "encoding": "zlib+base64+uint8",
        "data": base64.b64encode(zlib.compress(normalized.tobytes())).decode("ascii"),
    }


def _runtime_summary(payload: dict[str, object]) -> dict[str, float | int]:
    raw_results = payload.get("fine_results")
    if not isinstance(raw_results, list):
        raise ValueError("condition summary lacks fine_results")
    results = [
        cast(dict[str, object], item)
        for item in raw_results
        if isinstance(item, dict) and item.get("stage") in {"chunk", "assembly"}
    ]
    chunks = [item for item in results if item.get("stage") == "chunk"]
    if not results or not chunks:
        raise ValueError("condition summary lacks nonlinear task results")
    started = min(
        datetime.fromisoformat(cast(str, item["started_at"])) for item in results
    )
    ended = max(datetime.fromisoformat(cast(str, item["ended_at"])) for item in results)
    wall_seconds = (ended - started).total_seconds()
    task_seconds = sum(float(item["duration_seconds"]) for item in chunks)
    return {
        "nonlinear_wall_seconds": wall_seconds,
        "chunk_task_seconds": task_seconds,
        "chunk_count": len(chunks),
        "chunks_per_hour": len(chunks) * 3600.0 / wall_seconds,
        "condition_elapsed_seconds": float(payload["elapsed_seconds"]),
    }


def analyze_characterization(
    *,
    output_root: Path,
    evidence_dir: Path,
    spec: FixedAffineCharacterizationSpec,
    source: ValidatedFixedAffineSource,
    analysis_path: Path,
) -> dict[str, object]:
    """Analyze all completed conditions into one atomic machine-readable result."""
    fixed_mip = _mip(source.fixed_reference)
    roi_values = {
        "dim": (
            spec.dim_roi_yx.y_start,
            spec.dim_roi_yx.y_end,
            spec.dim_roi_yx.x_start,
            spec.dim_roi_yx.x_end,
        ),
        "bright": (
            spec.bright_roi_yx.y_start,
            spec.bright_roi_yx.y_end,
            spec.bright_roi_yx.x_start,
            spec.bright_roi_yx.x_end,
        ),
    }
    source_rounds = {item.round: item for item in source.rounds}
    regional: list[dict[str, object]] = []
    blockwise: list[dict[str, object]] = []
    jacobians: list[dict[str, object]] = []
    runtimes: list[dict[str, object]] = []
    overlay_crops: list[dict[str, object]] = []
    for label in spec.condition_order:
        condition_root = output_root / f"chunk-{label}"
        dataset_root = condition_root / f"fixed-affine-chunk-{label}"
        summary_path = evidence_dir / f"chunk-{label}" / "condition-summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(summary, dict) or summary.get("execution_error") is not None:
            raise ValueError(f"condition {label} is not a successful exact run")
        runtimes.append({"condition": label, **_runtime_summary(summary)})
        chunk_size = spec.chunk_sizes_zyx[label]
        chunks = compute_chunk_grid(
            source.rounds[0].fixed_geometry.shape,
            chunk_size,
            spec.overlap_fraction,
        )
        for round_number in spec.rounds:
            registration = (
                dataset_root / f"rounds/round-{round_number:03d}" / "registration"
            )
            paths = RegistrationArtifactPaths.for_round(registration, round_number)
            registered = (
                dataset_root
                / f"rounds/round-{round_number:03d}"
                / "channels"
                / source_rounds[round_number].moving_image.name
            )
            moving_mip = _mip(registered)
            if moving_mip.shape != fixed_mip.shape:
                raise ValueError(
                    f"condition {label} round {round_number} changed fixed geometry"
                )
            for roi_name, roi in roi_values.items():
                residual = gradient_ncc_search(
                    fixed_mip,
                    moving_mip,
                    roi_yx=roi,
                    search_radius_px=spec.search_radius_px,
                )
                regional.append(
                    {
                        "condition": label,
                        "round": round_number,
                        "roi": roi_name,
                        **residual.to_dict(),
                    }
                )
                overlay_crops.append(
                    {
                        "condition": label,
                        "round": round_number,
                        "roi": roi_name,
                        "fixed": _encoded_gradient_crop(fixed_mip, roi),
                        "registered": _encoded_gradient_crop(moving_mip, roi),
                    }
                )
            for block in blockwise_residuals(
                fixed_mip,
                moving_mip,
                chunks,
                search_radius_px=spec.search_radius_px,
            ):
                blockwise.append(
                    {
                        "condition": label,
                        "round": round_number,
                        **block.to_dict(),
                    }
                )
            _validated_artifact(paths.warp)
            jacobians.append(
                {
                    "condition": label,
                    "round": round_number,
                    **jacobian_summary(
                        source.fixed_reference,
                        paths.warp,
                        voxel_spacing_xyz=source.rounds[0].fixed_geometry.spacing,
                    ),
                }
            )
    result: dict[str, object] = {
        "schema_version": 1,
        "condition_order": list(spec.condition_order),
        "chunk_sizes_zyx": {
            label: list(spec.chunk_sizes_zyx[label]) for label in spec.condition_order
        },
        "overlap_fraction": spec.overlap_fraction,
        "search_radius_px": spec.search_radius_px,
        "rois_yx": {name: list(value) for name, value in roi_values.items()},
        "regional_residuals": regional,
        "blockwise_residuals": blockwise,
        "jacobians": jacobians,
        "overlay_crops": overlay_crops,
        "runtimes": runtimes,
    }
    write_evidence_json(analysis_path, result)
    return result


__all__ = [
    "BlockResidual",
    "RoiResidual",
    "analyze_characterization",
    "blockwise_residuals",
    "classify_seam_proximity",
    "gradient_ncc_search",
    "jacobian_summary",
    "summarize_jacobian_array",
]
