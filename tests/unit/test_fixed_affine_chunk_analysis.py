from __future__ import annotations

import math

import numpy as np
import pytest
from hpc.fixed_affine_chunk_analysis import (
    RoiResidual,
    blockwise_residuals,
    classify_seam_proximity,
    gradient_ncc_search,
    summarize_jacobian_array,
)
from scipy.ndimage import gaussian_filter

from cyexm.chunking import compute_chunk_grid

pytestmark = pytest.mark.unit


def _translate(image: np.ndarray, dy: int, dx: int) -> np.ndarray:
    result = np.zeros_like(image)
    source_y = slice(max(0, -dy), min(image.shape[0], image.shape[0] - dy))
    source_x = slice(max(0, -dx), min(image.shape[1], image.shape[1] - dx))
    target_y = slice(max(0, dy), min(image.shape[0], image.shape[0] + dy))
    target_x = slice(max(0, dx), min(image.shape[1], image.shape[1] + dx))
    result[target_y, target_x] = image[source_y, source_x]
    return result


def test_gradient_ncc_recovers_shift_and_ignores_brightness_scaling() -> None:
    rng = np.random.default_rng(731)
    fixed = gaussian_filter(rng.normal(size=(160, 170)), sigma=2.0)
    moving = 7.5 * _translate(fixed, -4, 6) + 23.0

    result = gradient_ncc_search(
        fixed,
        moving,
        roi_yx=(40, 120, 45, 125),
        search_radius_px=12,
    )

    assert result.correction_dy_px == 4
    assert result.correction_dx_px == -6
    assert result.residual_px == pytest.approx(math.sqrt(52))
    assert result.best_ncc == pytest.approx(1.0)
    assert result.current_ncc < result.best_ncc
    assert result.prominence > 0
    assert result.boundary_optimum is False
    assert result.defined is True


def test_gradient_ncc_zero_shift_and_boundary_flag() -> None:
    rng = np.random.default_rng(911)
    fixed = gaussian_filter(rng.normal(size=(120, 130)), sigma=1.5)

    zero = gradient_ncc_search(
        fixed,
        fixed * 2.0 + 5.0,
        roi_yx=(30, 90, 35, 95),
        search_radius_px=8,
    )
    boundary = gradient_ncc_search(
        fixed,
        _translate(fixed, -8, 0),
        roi_yx=(30, 90, 35, 95),
        search_radius_px=8,
    )

    assert (zero.correction_dy_px, zero.correction_dx_px) == (0, 0)
    assert zero.residual_px == 0
    assert zero.boundary_optimum is False
    assert (boundary.correction_dy_px, boundary.correction_dx_px) == (8, 0)
    assert boundary.boundary_optimum is True


def test_gradient_ncc_marks_uniform_regions_undefined() -> None:
    result = gradient_ncc_search(
        np.ones((80, 90)),
        np.ones((80, 90)),
        roi_yx=(20, 60, 25, 65),
        search_radius_px=10,
    )

    assert result == RoiResidual.undefined()


@pytest.mark.parametrize(
    "roi",
    [
        (-1, 20, 10, 30),
        (20, 20, 10, 30),
        (20, 40, 30, 30),
        (20, 75, 10, 30),
    ],
)
def test_gradient_ncc_rejects_invalid_or_contextless_rois(
    roi: tuple[int, int, int, int],
) -> None:
    with pytest.raises(ValueError, match="ROI"):
        gradient_ncc_search(
            np.ones((80, 90)),
            np.ones((80, 90)),
            roi_yx=roi,
            search_radius_px=10,
        )


def test_seam_proximity_uses_internal_core_boundaries() -> None:
    chunks = compute_chunk_grid((8, 300, 300), (8, 128, 128), 0.15)

    assert classify_seam_proximity(109, 64, chunks, distance_px=16) == "seam"
    assert classify_seam_proximity(128, 64, chunks, distance_px=16) == "interior"
    assert classify_seam_proximity(64, 64, chunks, distance_px=16) == "interior"
    assert classify_seam_proximity(290, 290, chunks, distance_px=16) == "interior"


def test_blockwise_residuals_are_deterministic_and_quartiled() -> None:
    rng = np.random.default_rng(181)
    fixed = gaussian_filter(rng.normal(size=(300, 300)), sigma=1.5)
    moving = _translate(fixed, -2, 3)
    chunks = compute_chunk_grid((8, 300, 300), (8, 128, 128), 0.15)

    blocks = blockwise_residuals(
        fixed,
        moving,
        chunks,
        search_radius_px=10,
        window_size_px=64,
        stride_px=64,
    )

    assert len(blocks) == 16
    assert {block.intensity_quartile for block in blocks} == {1, 2, 3, 4}
    assert {block.seam_class for block in blocks} == {"seam", "interior"}
    assert all(
        (block.residual.correction_dy_px, block.residual.correction_dx_px) == (2, -3)
        for block in blocks
    )


def test_jacobian_array_summary_preserves_invalid_counts() -> None:
    values = np.array([1.0, 2.0, 0.0, -0.5, np.nan, np.inf])

    summary = summarize_jacobian_array(values)

    assert summary["voxel_count"] == 6
    assert summary["finite_count"] == 4
    assert summary["nonfinite_count"] == 2
    assert summary["nonpositive_count"] == 2
    assert summary["nonpositive_fraction"] == pytest.approx(0.5)
    assert summary["minimum"] == -0.5
    assert summary["maximum"] == 2.0
