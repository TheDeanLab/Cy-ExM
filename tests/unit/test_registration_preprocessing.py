from __future__ import annotations

import importlib

import numpy as np
import pytest
from scipy import ndimage as ndi

pytestmark = pytest.mark.unit


def _production():
    try:
        return importlib.import_module("cyexm.registration.preprocessing")
    except ModuleNotFoundError:
        pytest.fail("registration preprocessing module is missing")


def _reference_feature(
    image: np.ndarray,
    sigma: tuple[float, float, float],
) -> tuple[np.ndarray, float]:
    array = np.asarray(image, dtype=np.float32)
    residual = array - ndi.gaussian_filter(array, sigma=sigma, mode="reflect")
    stride = max(1, array.size // 2_000_000)
    sample = residual.ravel()[::stride]
    median = float(np.median(sample))
    mad = float(np.median(np.abs(sample - median)))
    scale = max(1.4826 * mad, float(np.finfo(np.float32).eps))
    feature = np.arcsinh(np.maximum(residual, 0.0) / scale).astype(np.float32)
    return feature, scale


def test_feature_matches_approved_reference_formula() -> None:
    production = _production()
    rng = np.random.default_rng(142)
    image = rng.normal(100, 20, size=(9, 17, 23)).astype(np.float32)
    expected, expected_scale = _reference_feature(image, (2.0, 3.0, 4.0))

    result = production.background_subtract_asinh(image, (2.0, 3.0, 4.0))

    np.testing.assert_array_equal(result.array, expected)
    assert result.robust_scale == expected_scale


def test_constant_feature_is_finite_float32_zero_without_mutating_input() -> None:
    production = _production()
    image = np.full((7, 9, 11), 50, dtype=np.uint16)
    original = image.copy()

    result = production.background_subtract_asinh(image, (2.0, 2.0, 2.0))

    assert result.array.shape == image.shape
    assert result.array.dtype == np.dtype("float32")
    assert np.isfinite(result.array).all()
    assert np.count_nonzero(result.array) == 0
    assert result.robust_scale == float(np.finfo(np.float32).eps)
    np.testing.assert_array_equal(image, original)


def test_anisotropic_filter_uses_declared_zyx_axis_order() -> None:
    production = _production()
    image = np.zeros((9, 17, 25), dtype=np.float32)
    image[4, 8, 4:21] = np.linspace(5, 100, 17, dtype=np.float32)

    z_smoothed = production.background_subtract_asinh(
        image,
        (5.0, 1.0, 1.0),
    )
    x_smoothed = production.background_subtract_asinh(
        image,
        (1.0, 1.0, 5.0),
    )
    expected_z, _ = _reference_feature(image, (5.0, 1.0, 1.0))
    expected_x, _ = _reference_feature(image, (1.0, 1.0, 5.0))

    np.testing.assert_array_equal(z_smoothed.array, expected_z)
    np.testing.assert_array_equal(x_smoothed.array, expected_x)
    assert not np.array_equal(z_smoothed.array, x_smoothed.array)


def test_scale_uses_signed_residual_before_negative_clipping() -> None:
    production = _production()
    image = np.zeros((5, 9, 13), dtype=np.float32)
    image[2, 4, 6] = 100
    expected, expected_scale = _reference_feature(image, (1.0, 2.0, 3.0))

    result = production.background_subtract_asinh(image, (1.0, 2.0, 3.0))

    clipped_residual_scale = float(
        np.median(
            np.abs(
                np.maximum(
                    image - ndi.gaussian_filter(image, (1.0, 2.0, 3.0)),
                    0,
                )
            )
        )
    )
    assert result.robust_scale == expected_scale
    assert result.robust_scale != clipped_residual_scale
    np.testing.assert_array_equal(result.array, expected)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((3, 4), dtype=np.float32),
        np.zeros((2, 3, 4, 1), dtype=np.float32),
    ],
)
def test_feature_rejects_non_3d_images(image: np.ndarray) -> None:
    production = _production()

    with pytest.raises(ValueError, match="3D"):
        production.background_subtract_asinh(image, (1.0, 1.0, 1.0))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_feature_rejects_nonfinite_images(value: float) -> None:
    production = _production()
    image = np.zeros((3, 4, 5), dtype=np.float32)
    image[1, 2, 3] = value

    with pytest.raises(ValueError, match="finite"):
        production.background_subtract_asinh(image, (1.0, 1.0, 1.0))


@pytest.mark.parametrize(
    "sigma",
    [
        (0.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
        (float("inf"), 1.0, 1.0),
        (True, 1.0, 1.0),
    ],
)
def test_feature_defensively_rejects_noncanonical_sigma(
    sigma: tuple[object, object, object],
) -> None:
    production = _production()

    with pytest.raises((TypeError, ValueError), match="sigma"):
        production.background_subtract_asinh(
            np.zeros((3, 4, 5), dtype=np.float32),
            sigma,
        )
