from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from cyexm.errors import CacheValidationError, InputValidationError
from cyexm.io.images import ants_geometry, load_ants_image
from cyexm.io.tiff import write_clipped_uint16_tiff, write_uint16_tiff
from cyexm.spatial import ImageGeometry

pytestmark = pytest.mark.unit


def test_write_clipped_uint16_tiff_does_not_normalize(tmp_path: Path) -> None:
    source = np.asarray([[[-4.0, 0.0, 12.9, 65535.0, 70000.0]]], dtype=np.float32)
    destination = tmp_path / "channel.tif"

    write_clipped_uint16_tiff(source, destination)

    written = tifffile.imread(destination)
    assert written.dtype == np.dtype(np.uint16)
    assert written.tolist() == [[[0, 0, 12, 65535, 65535]]]


def test_write_clipped_uint16_tiff_maps_every_nonfinite_value_to_zero(
    tmp_path: Path,
) -> None:
    source = np.asarray([[[np.nan, np.inf, -np.inf, 7.9]]], dtype=np.float64)
    destination = tmp_path / "nonfinite.tiff"

    write_clipped_uint16_tiff(source, destination)

    np.testing.assert_array_equal(
        tifffile.imread(destination),
        np.asarray([[[0, 0, 0, 7]]], dtype=np.uint16),
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        pytest.param(
            np.asarray([[[-2, 0, 42, 65535, 65536]]], dtype=np.int64),
            [[[0, 0, 42, 65535, 65535]]],
            id="signed",
        ),
        pytest.param(
            np.asarray([[[0, 42, 65535, 65536, 2**64 - 1]]], dtype=np.uint64),
            [[[0, 42, 65535, 65535, 65535]]],
            id="unsigned",
        ),
    ],
)
def test_write_clipped_uint16_tiff_clips_integer_inputs_without_rescaling(
    tmp_path: Path,
    source: np.ndarray,
    expected: list[list[list[int]]],
) -> None:
    destination = tmp_path / "integer.tif"

    write_clipped_uint16_tiff(source, destination)

    np.testing.assert_array_equal(
        tifffile.imread(destination), np.asarray(expected, dtype=np.uint16)
    )


@pytest.mark.parametrize(
    ("source", "message"),
    [
        pytest.param(np.ones((2, 2)), "three-dimensional", id="two-dimensional"),
        pytest.param(np.empty((0, 2, 2)), "must not be empty", id="empty"),
        pytest.param(np.ones((1, 1, 1), dtype=np.bool_), "real numeric", id="bool"),
        pytest.param(
            np.ones((1, 1, 1), dtype=np.complex64), "real numeric", id="complex"
        ),
        pytest.param(
            np.full((1, 1, 1), "signal", dtype="U6"), "real numeric", id="string"
        ),
    ],
)
def test_write_clipped_uint16_tiff_rejects_invalid_array_data(
    tmp_path: Path,
    source: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(InputValidationError, match=message):
        write_clipped_uint16_tiff(source, tmp_path / "channel.tif")


def test_write_clipped_uint16_tiff_rejects_non_tiff_extension(
    tmp_path: Path,
) -> None:
    with pytest.raises(InputValidationError, match=r"\.tif or \.tiff"):
        write_clipped_uint16_tiff(np.ones((1, 1, 1)), tmp_path / "channel.png")


def test_write_clipped_uint16_tiff_preserves_exact_ants_geometry(
    tmp_path: Path,
) -> None:
    import ants

    external_zyx = np.asarray([[[-4.0, 12.9], [65535.0, 70000.0]]], dtype=np.float32)
    direction = np.asarray(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    image = ants.from_numpy(
        np.transpose(external_zyx, (2, 1, 0)),
        origin=(10.0, 20.0, 30.0),
        spacing=(0.2, 0.3, 0.5),
        direction=direction,
    )
    destination = tmp_path / "channel-geometry.tif"

    write_clipped_uint16_tiff(image, destination)

    np.testing.assert_array_equal(
        tifffile.imread(destination),
        np.asarray([[[0, 12], [65535, 65535]]], dtype=np.uint16),
    )
    with tifffile.TiffFile(destination) as written:
        metadata = written.shaped_metadata
        assert metadata and metadata[0] == {
            "shape": [1, 2, 2],
            "axes": "ZYX",
            "cyexm": {
                "schema_version": 1,
                "axis_convention_version": "external-zyx-ants-xyz-v1",
                "geometry": {
                    "shape": [1, 2, 2],
                    "origin": [10.0, 20.0, 30.0],
                    "spacing": list(image.spacing),
                    "direction": [list(row) for row in image.direction],
                },
            },
        }

    restored = load_ants_image(destination, (0.2, 0.3, 0.5))
    assert ants_geometry(restored) == ImageGeometry(
        shape=(1, 2, 2),
        origin=(10.0, 20.0, 30.0),
        spacing=image.spacing,
        direction=tuple(tuple(float(value) for value in row) for row in direction),
    )


def test_write_clipped_uint16_tiff_failure_is_atomic_and_cleans_temporary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "channel.part.tiff"
    destination.write_bytes(b"previous")
    seen: list[Path] = []

    def failing_imwrite(
        path: Path,
        data: np.ndarray,
        *,
        metadata: dict[str, object],
        photometric: str,
    ) -> None:
        del data
        assert metadata == {"axes": "ZYX"}
        assert photometric == "minisblack"
        temporary = Path(path)
        seen.append(temporary)
        temporary.write_bytes(b"partial")
        raise RuntimeError("injected clipped TIFF failure")

    monkeypatch.setattr(tifffile, "imwrite", failing_imwrite)

    with pytest.raises(RuntimeError, match="injected clipped TIFF failure"):
        write_clipped_uint16_tiff(np.ones((1, 1, 1)), destination)

    assert len(seen) == 1
    assert seen[0].name.endswith(".part.tiff")
    assert destination.read_bytes() == b"previous"
    assert list(tmp_path.glob(".*.tmp*")) == []


def test_write_clipped_uint16_tiff_validates_exact_encoded_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "channel.tif"
    destination.write_bytes(b"previous")
    real_imwrite = tifffile.imwrite

    def altered_imwrite(
        path: Path,
        data: np.ndarray,
        *,
        metadata: dict[str, object],
        photometric: str,
    ) -> None:
        altered = np.array(data, copy=True)
        altered[0, 0, 0] += 1
        real_imwrite(path, altered, metadata=metadata, photometric=photometric)

    monkeypatch.setattr(tifffile, "imwrite", altered_imwrite)

    with pytest.raises(CacheValidationError, match="failed validation"):
        write_clipped_uint16_tiff(np.ones((1, 1, 1)), destination)

    assert destination.read_bytes() == b"previous"
    assert list(tmp_path.glob(".*.tmp*")) == []


def test_uint16_export_matches_legacy_range_mapping(tmp_path: Path) -> None:
    data = np.array([[[0.0, 5.0, 10.0, np.nan]]], dtype=np.float32)
    destination = tmp_path / "registered.tif"

    write_uint16_tiff(data, destination, min_value=100.0, max_value=1100.0)

    written = tifffile.imread(destination)
    assert written.dtype == np.uint16
    np.testing.assert_array_equal(written, [[[100, 600, 1100, 100]]])


def test_uint16_export_replaces_nonfinite_values_and_clips_negatives(
    tmp_path: Path,
) -> None:
    data = np.array(
        [[[-5.0, 0.0, 10.0, np.nan, np.inf, -np.inf]]],
        dtype=np.float32,
    )
    destination = tmp_path / "registered.tiff"

    write_uint16_tiff(data, destination, min_value=100.0, max_value=1100.0)

    np.testing.assert_array_equal(
        tifffile.imread(destination),
        [[[100, 100, 1100, 100, 100, 100]]],
    )


def test_uint16_export_maps_the_data_minimum_to_the_caller_minimum(
    tmp_path: Path,
) -> None:
    data = np.array([[[5.0, 10.0]]], dtype=np.float32)
    destination = tmp_path / "registered.tif"

    write_uint16_tiff(data, destination, min_value=100.0, max_value=1100.0)

    np.testing.assert_array_equal(tifffile.imread(destination), [[[100, 1100]]])


def test_uint16_export_maps_constant_images_to_the_minimum(tmp_path: Path) -> None:
    data = np.full((2, 3, 4), 7.0, dtype=np.float32)
    destination = tmp_path / "constant.tif"

    write_uint16_tiff(data, destination, min_value=123.0, max_value=456.0)

    np.testing.assert_array_equal(
        tifffile.imread(destination),
        np.full((2, 3, 4), 123, dtype=np.uint16),
    )


def test_uint16_export_preserves_adjacent_uint64_extrema(tmp_path: Path) -> None:
    data = np.array(
        [[[np.uint64(2**64 - 2), np.uint64(2**64 - 1)]]],
        dtype=np.uint64,
    )
    destination = tmp_path / "uint64.tif"

    write_uint16_tiff(data, destination, min_value=100.0, max_value=1100.0)

    np.testing.assert_array_equal(tifffile.imread(destination), [[[100, 1100]]])


def test_uint16_export_preserves_longdouble_values_outside_float64(
    tmp_path: Path,
) -> None:
    float64_max = np.longdouble(np.finfo(np.float64).max)
    data = np.array(
        [[[float64_max * np.longdouble(2), float64_max * np.longdouble(4)]]],
        dtype=np.longdouble,
    )
    assert np.all(np.isfinite(data))
    destination = tmp_path / "longdouble.tif"

    write_uint16_tiff(data, destination, min_value=100.0, max_value=1100.0)

    np.testing.assert_array_equal(tifffile.imread(destination), [[[100, 1100]]])


def test_uint16_export_accepts_ants_images(tmp_path: Path) -> None:
    import ants

    external_zyx = np.full((2, 3, 5), 2.0, dtype=np.float32)
    external_zyx[1, 2, 4] = 4.0
    image = ants.from_numpy(np.transpose(external_zyx, (2, 1, 0)))
    destination = tmp_path / "ants-image.tif"

    write_uint16_tiff(image, destination, min_value=10.0, max_value=20.0)

    expected = np.full(external_zyx.shape, 10, dtype=np.uint16)
    expected[1, 2, 4] = 20
    np.testing.assert_array_equal(tifffile.imread(destination), expected)


def test_ants_tiff_round_trip_restores_complete_geometry_and_external_axes(
    tmp_path: Path,
) -> None:
    import ants

    external_zyx = np.zeros((2, 3, 4), dtype=np.float32)
    external_zyx[1, 2, 3] = 1.0
    direction = np.asarray(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    image = ants.from_numpy(
        np.transpose(external_zyx, (2, 1, 0)),
        origin=(10.0, 20.0, 30.0),
        spacing=(0.2, 0.3, 0.5),
        direction=direction,
    )
    destination = tmp_path / "ants-geometry.tif"

    write_uint16_tiff(image, destination, min_value=0.0, max_value=100.0)

    np.testing.assert_array_equal(
        tifffile.imread(destination),
        external_zyx.astype(np.uint16) * 100,
    )
    with tifffile.TiffFile(destination) as written:
        metadata = written.shaped_metadata
        assert metadata and metadata[0]["cyexm"] == {
            "schema_version": 1,
            "axis_convention_version": "external-zyx-ants-xyz-v1",
            "geometry": {
                "shape": [2, 3, 4],
                "origin": [10.0, 20.0, 30.0],
                "spacing": list(image.spacing),
                "direction": [list(row) for row in image.direction],
            },
        }

    restored = load_ants_image(destination, (0.2, 0.3, 0.5))
    assert ants_geometry(restored) == ImageGeometry(
        shape=(2, 3, 4),
        origin=(10.0, 20.0, 30.0),
        spacing=image.spacing,
        direction=tuple(tuple(float(value) for value in row) for row in direction),
    )
    np.testing.assert_array_equal(
        restored.numpy(),
        np.transpose(external_zyx.astype(np.uint16) * 100, (2, 1, 0)),
    )


def test_singleton_z_ants_tiff_round_trip_keeps_three_dimensional_geometry(
    tmp_path: Path,
) -> None:
    import ants

    external_zyx = np.zeros((1, 3, 4), dtype=np.float32)
    external_zyx[0, 2, 3] = 1.0
    image = ants.from_numpy(
        np.transpose(external_zyx, (2, 1, 0)),
        origin=(1.0, 2.0, 3.0),
        spacing=(0.2, 0.3, 0.5),
    )
    destination = tmp_path / "singleton-z.tif"

    write_uint16_tiff(image, destination, min_value=0.0, max_value=1.0)
    restored = load_ants_image(destination, (0.2, 0.3, 0.5))

    assert ants_geometry(restored).shape == (1, 3, 4)
    assert restored.shape == (4, 3, 1)
    assert restored.origin == image.origin
    assert restored.spacing == image.spacing
    np.testing.assert_array_equal(restored.direction, image.direction)
    np.testing.assert_array_equal(
        restored.numpy(),
        np.transpose(external_zyx.astype(np.uint16), (2, 1, 0)),
    )


@pytest.mark.parametrize(
    ("min_value", "max_value"),
    [
        pytest.param(-1.0, 100.0, id="negative-minimum"),
        pytest.param(0.0, 65536.0, id="maximum-over-uint16"),
        pytest.param(200.0, 100.0, id="reversed"),
        pytest.param(float("nan"), 100.0, id="nan"),
        pytest.param(0.0, float("inf"), id="infinite"),
        pytest.param(True, 100.0, id="boolean"),
        pytest.param("0", 100.0, id="string"),
        pytest.param(0.0, 10**10000, id="overflowing-integral"),
    ],
)
def test_uint16_export_rejects_unusable_ranges(
    tmp_path: Path,
    min_value: object,
    max_value: object,
) -> None:
    with pytest.raises(InputValidationError, match="range"):
        write_uint16_tiff(
            np.ones((1, 1, 1)),
            tmp_path / "registered.tif",
            min_value=min_value,
            max_value=max_value,
        )


def test_uint16_export_rejects_non_numeric_arrays(tmp_path: Path) -> None:
    data = np.full((1, 1, 1), "signal", dtype="U6")

    with pytest.raises(InputValidationError, match="numeric"):
        write_uint16_tiff(
            data,
            tmp_path / "registered.tif",
            min_value=0.0,
            max_value=100.0,
        )


def test_uint16_export_rejects_non_3d_arrays(tmp_path: Path) -> None:
    with pytest.raises(InputValidationError, match="three-dimensional"):
        write_uint16_tiff(
            np.ones((2, 2)),
            tmp_path / "registered.tif",
            min_value=0.0,
            max_value=100.0,
        )


def test_uint16_export_preserves_compound_suffix_on_atomic_temporary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "registered.part.tiff"
    seen: list[Path] = []
    seen_metadata: list[dict[str, str]] = []
    real_imwrite = tifffile.imwrite

    def recording_imwrite(
        path: Path,
        data: np.ndarray,
        *,
        metadata: dict[str, str],
        photometric: str,
    ) -> None:
        assert photometric == "minisblack"
        seen.append(Path(path))
        seen_metadata.append(metadata)
        real_imwrite(path, data, metadata=metadata, photometric=photometric)

    monkeypatch.setattr(tifffile, "imwrite", recording_imwrite)

    write_uint16_tiff(
        np.arange(8, dtype=np.float32).reshape(2, 2, 2),
        destination,
        min_value=0.0,
        max_value=100.0,
    )

    assert seen[0] != destination
    assert seen[0].parent == destination.parent
    assert seen[0].name.endswith(".part.tiff")
    assert seen_metadata == [{"axes": "ZYX"}]
    assert destination.is_file()
    with tifffile.TiffFile(destination) as written:
        assert written.series[0].axes == "ZYX"
    assert list(tmp_path.glob(".*.tmp*")) == []


def test_uint16_export_failure_keeps_the_previous_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "registered.tif"
    destination.write_bytes(b"previous")

    def failing_imwrite(
        path: Path,
        data: np.ndarray,
        *,
        metadata: dict[str, str],
        photometric: str,
    ) -> None:
        del data
        assert photometric == "minisblack"
        assert metadata == {"axes": "ZYX"}
        Path(path).write_bytes(b"partial")
        raise RuntimeError("injected TIFF failure")

    monkeypatch.setattr(tifffile, "imwrite", failing_imwrite)

    with pytest.raises(RuntimeError, match="injected TIFF failure"):
        write_uint16_tiff(
            np.ones((1, 1, 1)),
            destination,
            min_value=0.0,
            max_value=100.0,
        )

    assert destination.read_bytes() == b"previous"
    assert list(tmp_path.glob(".*.tmp*")) == []


def test_uint16_export_rejects_atomic_temporary_with_wrong_axes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "registered.tif"
    destination.write_bytes(b"previous")
    real_imwrite = tifffile.imwrite

    def wrong_axes_imwrite(
        path: Path,
        data: np.ndarray,
        *,
        metadata: dict[str, str],
        photometric: str,
    ) -> None:
        assert metadata == {"axes": "ZYX"}
        assert photometric == "minisblack"
        real_imwrite(path, data, photometric=photometric)

    monkeypatch.setattr(tifffile, "imwrite", wrong_axes_imwrite)

    with pytest.raises(CacheValidationError, match="failed validation"):
        write_uint16_tiff(
            np.arange(8, dtype=np.float32).reshape(2, 2, 2),
            destination,
            min_value=0.0,
            max_value=100.0,
        )

    assert destination.read_bytes() == b"previous"
    assert list(tmp_path.glob(".*.tmp*")) == []


def test_ants_export_rejects_atomic_temporary_with_wrong_geometry_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ants

    destination = tmp_path / "registered.tif"
    destination.write_bytes(b"previous")
    image = ants.from_numpy(
        np.ones((4, 3, 2), dtype=np.float32),
        origin=(10.0, 20.0, 30.0),
        spacing=(0.2, 0.3, 0.5),
    )
    real_imwrite = tifffile.imwrite

    def wrong_geometry_imwrite(
        path: Path,
        data: np.ndarray,
        *,
        metadata: dict[str, object],
        photometric: str,
    ) -> None:
        assert photometric == "minisblack"
        wrong_metadata = dict(metadata)
        wrong_metadata["cyexm"] = {
            "schema_version": 1,
            "axis_convention_version": "external-zyx-ants-xyz-v1",
            "geometry": {
                "shape": list(data.shape),
                "origin": [999.0, 20.0, 30.0],
                "spacing": [0.2, 0.3, 0.5],
                "direction": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            },
        }
        real_imwrite(
            path,
            data,
            metadata=wrong_metadata,
            photometric=photometric,
        )

    monkeypatch.setattr(tifffile, "imwrite", wrong_geometry_imwrite)

    with pytest.raises(CacheValidationError, match="failed validation"):
        write_uint16_tiff(
            image,
            destination,
            min_value=0.0,
            max_value=100.0,
        )

    assert destination.read_bytes() == b"previous"
    assert list(tmp_path.glob(".*.tmp*")) == []


def test_ants_export_requires_exact_geometry_metadata_before_atomic_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ants

    destination = tmp_path / "registered.tif"
    destination.write_bytes(b"previous")
    image = ants.from_numpy(
        np.ones((4, 3, 2), dtype=np.float32),
        origin=(10.0, 20.0, 30.0),
        spacing=(0.2, 0.3, 0.5),
    )
    real_imwrite = tifffile.imwrite

    def subtly_wrong_geometry_imwrite(
        path: Path,
        data: np.ndarray,
        *,
        metadata: dict[str, object],
        photometric: str,
    ) -> None:
        wrong_metadata = dict(metadata)
        payload = metadata["cyexm"]
        assert isinstance(payload, dict)
        geometry = payload["geometry"]
        assert isinstance(geometry, dict)
        wrong_geometry = dict(geometry)
        wrong_geometry["origin"] = [10.0 + 5e-8, 20.0, 30.0]
        wrong_payload = dict(payload)
        wrong_payload["geometry"] = wrong_geometry
        wrong_metadata["cyexm"] = wrong_payload
        real_imwrite(
            path,
            data,
            metadata=wrong_metadata,
            photometric=photometric,
        )

    monkeypatch.setattr(tifffile, "imwrite", subtly_wrong_geometry_imwrite)

    with pytest.raises(CacheValidationError, match="failed validation"):
        write_uint16_tiff(
            image,
            destination,
            min_value=0.0,
            max_value=100.0,
        )

    assert destination.read_bytes() == b"previous"
    assert list(tmp_path.glob(".*.tmp*")) == []
