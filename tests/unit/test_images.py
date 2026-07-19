from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

from cyexm.errors import InputValidationError
from cyexm.io.images import (
    ants_geometry,
    crop_array,
    load_ants_image,
    load_array,
    robust_bounding_box,
    zero_mask,
)
from cyexm.spatial import ImageGeometry

pytestmark = pytest.mark.unit


def test_new_image_and_metric_modules_import_without_scientific_libraries() -> None:
    code = """
import os
import sys

before = dict(os.environ)
import cyexm
import cyexm.io
import cyexm.io.images
import cyexm.io.tiff
import cyexm.metrics
after = dict(os.environ)

assert before == after
for root in ("ants", "numpy", "scipy", "skimage", "tifffile"):
    assert not any(name == root or name.startswith(root + ".") for name in sys.modules)
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_load_array_eagerly_reads_npy_volume(tmp_path: Path) -> None:
    path = tmp_path / "volume.npy"
    expected = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    np.save(path, expected)

    loaded = load_array(path)

    path.unlink()
    assert type(loaded) is np.ndarray
    np.testing.assert_array_equal(loaded, expected)


def test_load_array_reads_tiff_in_zyx_order(tmp_path: Path) -> None:
    path = tmp_path / "volume.tiff"
    expected = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)
    tifffile.imwrite(path, expected, metadata={"axes": "ZYX"})

    loaded = load_array(path)

    np.testing.assert_array_equal(loaded, expected)
    assert loaded.shape == (2, 3, 4)


def test_load_array_rejects_non_3d_data(tmp_path: Path) -> None:
    path = tmp_path / "plane.npy"
    np.save(path, np.zeros((4, 4), dtype=np.uint16))

    with pytest.raises(InputValidationError, match="three-dimensional"):
        load_array(path)


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(np.zeros((1, 1, 1), dtype=np.bool_), id="boolean"),
        pytest.param(np.full((1, 1, 1), "signal", dtype=object), id="object"),
        pytest.param(np.full((1, 1, 1), "signal", dtype="U6"), id="string"),
    ],
)
def test_load_array_rejects_non_numeric_data(tmp_path: Path, data: np.ndarray) -> None:
    path = tmp_path / "volume.npy"
    np.save(path, data)

    with pytest.raises(InputValidationError, match="numeric"):
        load_array(path)


def test_load_array_rejects_missing_paths(tmp_path: Path) -> None:
    with pytest.raises(InputValidationError, match="file"):
        load_array(tmp_path / "missing.npy")


def test_load_array_rejects_directories(tmp_path: Path) -> None:
    directory = tmp_path / "volume.npy"
    directory.mkdir()

    with pytest.raises(InputValidationError, match="file"):
        load_array(directory)


def test_load_array_rejects_unsupported_extensions(tmp_path: Path) -> None:
    path = tmp_path / "volume.npz"
    path.write_bytes(b"not-an-array")

    with pytest.raises(InputValidationError, match="extension"):
        load_array(path)


def test_load_tiff_preserves_zyx_and_assigns_xyz_spacing(tmp_path: Path) -> None:
    path = tmp_path / "volume.tif"
    data = np.arange(4 * 6 * 8, dtype=np.uint16).reshape(4, 6, 8)
    tifffile.imwrite(
        path,
        data,
        metadata={"axes": "ZYX"},
        photometric="minisblack",
    )

    image = load_ants_image(path, (0.15, 0.15, 0.4))

    np.testing.assert_array_equal(image.numpy(), data)
    assert image.spacing == pytest.approx((0.15, 0.15, 0.4))
    assert image.origin == pytest.approx((0.0, 0.0, 0.0))


@pytest.mark.parametrize(
    "pixel_size_xyz",
    [
        pytest.param((0.1, 0.2), id="too-short"),
        pytest.param((0.1, 0.2, 0.3, 0.4), id="too-long"),
        pytest.param((0.0, 0.2, 0.3), id="zero"),
        pytest.param((-0.1, 0.2, 0.3), id="negative"),
        pytest.param((float("nan"), 0.2, 0.3), id="nan"),
        pytest.param((float("inf"), 0.2, 0.3), id="infinite"),
        pytest.param((True, 0.2, 0.3), id="boolean"),
        pytest.param(("0.1", 0.2, 0.3), id="string"),
        pytest.param((10**10000, 0.2, 0.3), id="overflowing-integral"),
    ],
)
def test_load_ants_image_rejects_invalid_xyz_spacing(
    tmp_path: Path,
    pixel_size_xyz: object,
) -> None:
    path = tmp_path / "volume.npy"
    np.save(path, np.ones((2, 3, 4), dtype=np.uint16))

    with pytest.raises(InputValidationError, match="pixel_size_xyz"):
        load_ants_image(path, pixel_size_xyz)


def test_ants_geometry_returns_dependency_neutral_three_dimensional_value() -> None:
    import ants

    direction = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    image = ants.from_numpy(
        np.ones((2, 3, 4), dtype=np.float32),
        origin=(1.0, 2.0, 3.0),
        spacing=(0.2, 0.3, 0.4),
        direction=direction,
    )

    geometry = ants_geometry(image)

    assert geometry == ImageGeometry(
        shape=(2, 3, 4),
        origin=(1.0, 2.0, 3.0),
        spacing=(0.2, 0.3, 0.4),
        direction=((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    )


def test_ants_geometry_rejects_non_3d_images() -> None:
    import ants

    image = ants.from_numpy(np.ones((3, 4), dtype=np.float32))

    with pytest.raises(InputValidationError, match="three-dimensional"):
        ants_geometry(image)


def test_zero_mask_preserves_geometry() -> None:
    import ants

    direction = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    image = ants.from_numpy(
        np.array([[[0, 2], [3, 0]]], dtype=np.float32),
        origin=(1.0, 2.0, 3.0),
        spacing=(0.2, 0.3, 0.4),
        direction=direction,
    )

    mask = zero_mask(image)

    np.testing.assert_array_equal(mask.numpy(), image.numpy() > 0)
    assert mask.origin == image.origin
    assert mask.spacing == image.spacing
    np.testing.assert_array_equal(mask.direction, image.direction)


def test_robust_bounding_box_finds_an_off_center_cuboid_despite_outliers() -> None:
    array = np.zeros((10, 13, 16), dtype=np.uint16)
    array[2:7, 3:9, 5:13] = 100
    array[0, 0, 0] = 100
    array[-1, -1, -1] = 100

    bounds = robust_bounding_box(
        array,
        down_sampling=1,
        lower_pct=5.0,
        upper_pct=95.0,
    )

    assert bounds == (2, 7, 3, 9, 5, 13)
    assert all(type(value) is int for value in bounds)


def test_robust_bounding_box_rejects_empty_foreground() -> None:
    array = np.zeros((4, 5, 6), dtype=np.uint16)

    with pytest.raises(ValueError, match="No foreground"):
        robust_bounding_box(array, down_sampling=1)


def test_robust_bounding_box_upper_bounds_are_exclusive() -> None:
    array = np.zeros((5, 6, 7), dtype=np.uint16)
    array[1:4, 2:5, 3:6] = 1

    bounds = robust_bounding_box(
        array,
        down_sampling=1,
        lower_pct=5.0,
        upper_pct=100.0,
    )

    assert bounds == (1, 4, 2, 5, 3, 6)
    np.testing.assert_array_equal(crop_array(array, bounds), np.ones((3, 3, 3)))


def test_robust_bounding_box_zero_lower_percentile_keeps_leading_planes() -> None:
    array = np.zeros((5, 6, 7), dtype=np.uint16)
    array[1:4, 2:5, 3:6] = 1

    bounds = robust_bounding_box(
        array,
        down_sampling=1,
        lower_pct=0.0,
        upper_pct=100.0,
    )

    assert bounds == (0, 4, 0, 5, 0, 6)
    assert all(type(value) is int for value in bounds)


@pytest.mark.parametrize(
    "down_sampling",
    [
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(1.5, id="non-integer"),
        pytest.param(True, id="boolean"),
    ],
)
def test_robust_bounding_box_validates_down_sampling(down_sampling: object) -> None:
    array = np.ones((2, 2, 2), dtype=np.uint16)

    with pytest.raises((TypeError, ValueError), match="down_sampling"):
        robust_bounding_box(array, down_sampling=down_sampling)


@pytest.mark.parametrize(
    ("lower_pct", "upper_pct"),
    [
        pytest.param(-1.0, 95.0, id="negative-lower"),
        pytest.param(5.0, 101.0, id="upper-over-100"),
        pytest.param(50.0, 50.0, id="equal"),
        pytest.param(80.0, 20.0, id="reversed"),
        pytest.param(float("nan"), 95.0, id="nan"),
        pytest.param(5.0, float("inf"), id="infinite"),
        pytest.param(True, 95.0, id="boolean"),
        pytest.param(5.0, 10**10000, id="overflowing-integral"),
    ],
)
def test_robust_bounding_box_validates_percentiles(
    lower_pct: object,
    upper_pct: object,
) -> None:
    array = np.ones((2, 2, 2), dtype=np.uint16)

    with pytest.raises((TypeError, ValueError), match="percent"):
        robust_bounding_box(
            array,
            down_sampling=1,
            lower_pct=lower_pct,
            upper_pct=upper_pct,
        )


def test_robust_bounding_box_requires_a_numpy_volume() -> None:
    with pytest.raises(TypeError, match="NumPy"):
        robust_bounding_box([[[1]]], down_sampling=1)


def test_crop_array_applies_zyx_bounds() -> None:
    array = np.arange(5 * 6 * 7).reshape(5, 6, 7)

    cropped = crop_array(array, (1, 4, 2, 5, 3, 7))

    np.testing.assert_array_equal(cropped, array[1:4, 2:5, 3:7])


@pytest.mark.parametrize(
    "bounds",
    [
        pytest.param((0, 1, 0, 1, 0), id="wrong-length"),
        pytest.param((-1, 1, 0, 1, 0, 1), id="negative-start"),
        pytest.param((0, 4, 0, 1, 0, 1), id="stop-outside-volume"),
        pytest.param((1, 1, 0, 1, 0, 1), id="empty-axis"),
        pytest.param((0, 1, 0, 1, 0, True), id="boolean"),
    ],
)
def test_crop_array_rejects_invalid_bounds(bounds: tuple[object, ...]) -> None:
    with pytest.raises((TypeError, ValueError), match="bounds"):
        crop_array(np.ones((3, 3, 3)), bounds)


def test_crop_array_requires_a_numpy_volume() -> None:
    with pytest.raises(TypeError, match="NumPy"):
        crop_array([[[1]]], (0, 1, 0, 1, 0, 1))
