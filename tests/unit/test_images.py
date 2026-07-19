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

    np.testing.assert_array_equal(image.numpy(), np.transpose(data, (2, 1, 0)))
    assert image.shape == (8, 6, 4)
    assert image.spacing == pytest.approx((0.15, 0.15, 0.4))
    assert image.origin == pytest.approx((0.0, 0.0, 0.0))
    np.testing.assert_array_equal(image.direction, np.eye(3))


@pytest.mark.parametrize(
    "description",
    [
        pytest.param('{"axes":"ZYX","cyexm":', id="truncated-value"),
        pytest.param('{"axes":"ZYX","cyexm"', id="truncated-key"),
        pytest.param(
            '{"cyexm":{"schema_version":1',
            id="truncated-nested-object",
        ),
    ],
)
def test_load_ants_image_rejects_truncated_raw_cyexm_descriptions(
    tmp_path: Path,
    description: str,
) -> None:
    path = tmp_path / "truncated-cyexm.tif"
    tifffile.imwrite(
        path,
        np.ones((2, 3, 4), dtype=np.uint16),
        description=description,
        metadata=None,
        photometric="minisblack",
    )
    with tifffile.TiffFile(path) as handle:
        assert '"cyexm"' in handle.pages[0].description
        assert handle.shaped_metadata is None

    with pytest.raises(InputValidationError, match="Cy-ExM TIFF geometry metadata"):
        load_ants_image(path, (0.2, 0.3, 0.5))


def test_load_ants_image_accepts_plain_qyx_grayscale_stack(tmp_path: Path) -> None:
    path = tmp_path / "plain-qyx.tif"
    data = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)
    tifffile.imwrite(
        path,
        data,
        metadata={"axes": "QYX"},
        photometric="minisblack",
    )
    with tifffile.TiffFile(path) as handle:
        assert handle.series[0].axes == "QYX"

    image = load_ants_image(path, (0.2, 0.3, 0.5))

    np.testing.assert_array_equal(image.numpy(), np.transpose(data, (2, 1, 0)))
    assert image.origin == (0.0, 0.0, 0.0)
    np.testing.assert_array_equal(image.direction, np.eye(3))


def test_load_ants_image_accepts_plain_reconstructed_singleton_z(
    tmp_path: Path,
) -> None:
    path = tmp_path / "plain-singleton-z.tif"
    data = np.arange(1 * 3 * 4, dtype=np.uint16).reshape(1, 3, 4)
    tifffile.imwrite(
        path,
        data,
        metadata={"axes": "ZYX"},
        photometric="minisblack",
    )
    with tifffile.TiffFile(path) as handle:
        assert handle.series[0].axes == "YX"

    image = load_ants_image(path, (0.2, 0.3, 0.5))

    assert image.shape == (4, 3, 1)
    np.testing.assert_array_equal(image.numpy(), np.transpose(data, (2, 1, 0)))


@pytest.mark.parametrize("layout", ["YXS", "SYX", "CYX"])
def test_load_ants_image_rejects_plain_sample_or_channel_layouts(
    tmp_path: Path,
    layout: str,
) -> None:
    path = tmp_path / f"plain-{layout.lower()}.tif"
    if layout == "YXS":
        tifffile.imwrite(
            path,
            np.ones((3, 4, 3), dtype=np.uint8),
            metadata=None,
            photometric="rgb",
        )
    elif layout == "SYX":
        tifffile.imwrite(
            path,
            np.ones((3, 3, 4), dtype=np.uint8),
            metadata=None,
            photometric="rgb",
            planarconfig="separate",
        )
    else:
        tifffile.imwrite(
            path,
            np.ones((3, 3, 4), dtype=np.uint8),
            metadata={"axes": "CYX"},
            photometric="minisblack",
        )
    with tifffile.TiffFile(path) as handle:
        assert handle.series[0].axes == layout

    with pytest.raises(InputValidationError, match=r"TIFF.*layout"):
        load_ants_image(path, (0.2, 0.3, 0.5))


def _cyexm_tiff_metadata(
    *,
    shape: list[int] | None = None,
    spacing: list[float] | None = None,
) -> dict[str, object]:
    return {
        "axes": "ZYX",
        "cyexm": {
            "schema_version": 1,
            "axis_convention_version": "external-zyx-ants-xyz-v1",
            "geometry": {
                "shape": [2, 3, 4] if shape is None else shape,
                "origin": [10.0, 20.0, 30.0],
                "spacing": [0.2, 0.3, 0.5] if spacing is None else spacing,
                "direction": [
                    [0.0, -1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            },
        },
    }


@pytest.mark.parametrize(
    "cyexm_payload",
    [
        pytest.param(None, id="null-present-payload"),
        pytest.param({}, id="empty-present-payload"),
        pytest.param("not-an-object", id="non-object"),
        pytest.param(
            {
                "schema_version": True,
                "axis_convention_version": "external-zyx-ants-xyz-v1",
                "geometry": {},
            },
            id="boolean-schema",
        ),
        pytest.param(
            {
                "schema_version": 2,
                "axis_convention_version": "external-zyx-ants-xyz-v1",
                "geometry": {},
            },
            id="unknown-schema",
        ),
        pytest.param(
            {
                "schema_version": 1,
                "axis_convention_version": "legacy-ambiguous-v0",
                "geometry": {},
            },
            id="wrong-axis-convention",
        ),
    ],
)
def test_load_ants_image_rejects_malformed_present_cyexm_metadata(
    tmp_path: Path,
    cyexm_payload: object,
) -> None:
    path = tmp_path / "malformed.tif"
    tifffile.imwrite(
        path,
        np.ones((2, 3, 4), dtype=np.uint16),
        metadata={"axes": "ZYX", "cyexm": cyexm_payload},
        photometric="minisblack",
    )

    with pytest.raises(InputValidationError, match="Cy-ExM TIFF geometry metadata"):
        load_ants_image(path, (0.2, 0.3, 0.5))


@pytest.mark.parametrize(
    ("level", "mutation"),
    [
        pytest.param("cyexm", "missing", id="cyexm-missing-key"),
        pytest.param("cyexm", "extra", id="cyexm-extra-key"),
        pytest.param("geometry", "missing", id="geometry-missing-key"),
        pytest.param("geometry", "extra", id="geometry-extra-key"),
    ],
)
def test_load_ants_image_rejects_noncanonical_metadata_key_sets(
    tmp_path: Path,
    level: str,
    mutation: str,
) -> None:
    metadata = _cyexm_tiff_metadata()
    cyexm = metadata["cyexm"]
    assert isinstance(cyexm, dict)
    target = cyexm if level == "cyexm" else cyexm["geometry"]
    assert isinstance(target, dict)
    if mutation == "missing":
        target.pop("schema_version" if level == "cyexm" else "origin")
    else:
        target["unexpected"] = "value"
    path = tmp_path / f"{level}-{mutation}.tif"
    tifffile.imwrite(
        path,
        np.ones((2, 3, 4), dtype=np.uint16),
        metadata=metadata,
        photometric="minisblack",
    )

    with pytest.raises(InputValidationError, match="Cy-ExM TIFF geometry metadata"):
        load_ants_image(path, (0.2, 0.3, 0.5))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("shape", [2, 3, 0], id="zero-shape"),
        pytest.param("shape", [2, 3, 4.5], id="noninteger-shape"),
        pytest.param("origin", ["10.0", 20.0, 30.0], id="nonnumeric-origin"),
        pytest.param("spacing", [0.2, 0.0, 0.5], id="zero-spacing"),
        pytest.param("spacing", [0.2, "0.3", 0.5], id="nonnumeric-spacing"),
        pytest.param(
            "direction",
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            id="singular-direction",
        ),
        pytest.param(
            "direction",
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            id="zero-direction-row",
        ),
        pytest.param(
            "direction",
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1e-14]],
            id="near-singular-direction",
        ),
    ],
)
def test_load_ants_image_rejects_malformed_geometry_values(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    metadata = _cyexm_tiff_metadata()
    cyexm = metadata["cyexm"]
    assert isinstance(cyexm, dict)
    geometry = cyexm["geometry"]
    assert isinstance(geometry, dict)
    geometry[field] = value
    path = tmp_path / f"malformed-{field}.tif"
    tifffile.imwrite(
        path,
        np.ones((2, 3, 4), dtype=np.uint16),
        metadata=metadata,
        photometric="minisblack",
    )

    with pytest.raises(InputValidationError, match="Cy-ExM TIFF geometry metadata"):
        load_ants_image(path, (0.2, 0.3, 0.5))


def test_load_ants_image_rejects_wrong_axes_when_cyexm_metadata_is_present(
    tmp_path: Path,
) -> None:
    metadata = _cyexm_tiff_metadata()
    metadata["axes"] = "XYZ"
    path = tmp_path / "wrong-axes.tif"
    tifffile.imwrite(
        path,
        np.ones((2, 3, 4), dtype=np.uint16),
        metadata=metadata,
        photometric="minisblack",
    )

    with pytest.raises(InputValidationError, match="axes"):
        load_ants_image(path, (0.2, 0.3, 0.5))


def test_load_ants_image_rejects_embedded_geometry_shape_mismatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "wrong-shape.tif"
    tifffile.imwrite(
        path,
        np.ones((2, 3, 4), dtype=np.uint16),
        metadata=_cyexm_tiff_metadata(shape=[9, 3, 4]),
        photometric="minisblack",
    )

    with pytest.raises(InputValidationError, match="shape"):
        load_ants_image(path, (0.2, 0.3, 0.5))


def test_load_ants_image_rejects_requested_spacing_incompatible_with_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "wrong-spacing.tif"
    tifffile.imwrite(
        path,
        np.ones((2, 3, 4), dtype=np.uint16),
        metadata=_cyexm_tiff_metadata(),
        photometric="minisblack",
    )

    with pytest.raises(InputValidationError, match="requested spacing"):
        load_ants_image(path, (0.2, 0.3, 0.6))


def test_load_ants_image_restores_present_singleton_z_geometry_exactly(
    tmp_path: Path,
) -> None:
    path = tmp_path / "singleton-z.tif"
    data = np.arange(1 * 3 * 4, dtype=np.uint16).reshape(1, 3, 4)
    tifffile.imwrite(
        path,
        data,
        metadata=_cyexm_tiff_metadata(shape=[1, 3, 4]),
        photometric="minisblack",
    )

    image = load_ants_image(path, (0.2, 0.3, 0.5))

    assert image.shape == (4, 3, 1)
    assert ants_geometry(image).shape == (1, 3, 4)
    np.testing.assert_array_equal(image.numpy(), np.transpose(data, (2, 1, 0)))


def test_load_ants_image_wraps_embedded_ants_geometry_construction_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "valid-geometry.tif"
    tifffile.imwrite(
        path,
        np.ones((2, 3, 4), dtype=np.uint16),
        metadata=_cyexm_tiff_metadata(),
        photometric="minisblack",
    )

    class FailingAnts:
        @staticmethod
        def from_numpy(*args: object, **kwargs: object) -> object:
            del args, kwargs
            raise RuntimeError("injected ITK geometry failure")

    monkeypatch.setitem(sys.modules, "ants", FailingAnts())

    with pytest.raises(
        InputValidationError,
        match="embedded Cy-ExM geometry",
    ) as error:
        load_ants_image(path, (0.2, 0.3, 0.5))

    assert isinstance(error.value.__cause__, RuntimeError)


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
        shape=(4, 3, 2),
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
        np.arange(5 * 3 * 2, dtype=np.float32).reshape(5, 3, 2),
        origin=(1.0, 2.0, 3.0),
        spacing=(0.2, 0.3, 0.4),
        direction=direction,
    )

    mask = zero_mask(image)

    assert mask.shape == (5, 3, 2)
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
