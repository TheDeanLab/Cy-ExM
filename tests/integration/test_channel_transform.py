from __future__ import annotations

from pathlib import Path

import pytest

from cyexm.io.tiff import read_tiff_geometry
from cyexm.registration import transform_channel
from cyexm.registration.artifacts import ArtifactStore, CompletionRecord, FileIdentity
from cyexm.registration.backend import AntsRegistrationBackend
from cyexm.registration.transforms import apply_transform_chain
from cyexm.runtime import NativeThreadConfig
from cyexm.spatial import AXIS_CONVENTION_VERSION, ImageGeometry

pytestmark = pytest.mark.integration

_PIXEL_SIZE = (1.0, 1.0, 1.0)


def _temporary_siblings(root: Path) -> tuple[Path, ...]:
    return tuple(
        path for path in root.rglob(".*.tmp*") if path.is_file() or path.is_symlink()
    )


def test_transform_channel_resamples_unequal_native_shape_to_fixed_geometry(
    tmp_path: Path,
    native_threads: NativeThreadConfig,
) -> None:
    import ants
    import numpy as np
    import tifffile

    fixed_shape = (4, 6, 8)
    moving_shape = (3, 5, 7)
    fixed_geometry = ImageGeometry(
        shape=fixed_shape,
        origin=(0.5, -1.0, 2.0),
        spacing=_PIXEL_SIZE,
        direction=((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    )
    fixed_path = tmp_path / "fixed.tif"
    moving_path = tmp_path / "moving-channel.tif"
    tifffile.imwrite(
        fixed_path,
        np.ones(fixed_shape, dtype=np.float32),
        metadata={
            "axes": "ZYX",
            "cyexm": {
                "schema_version": 1,
                "axis_convention_version": AXIS_CONVENTION_VERSION,
                "geometry": fixed_geometry.to_dict(),
            },
        },
        photometric="minisblack",
    )
    tifffile.imwrite(
        moving_path,
        np.arange(np.prod(moving_shape), dtype=np.float32).reshape(moving_shape),
        metadata={"axes": "ZYX"},
        photometric="minisblack",
    )

    backend = AntsRegistrationBackend(native_threads)
    fixed = backend.load_image(fixed_path, _PIXEL_SIZE)
    affine_path = tmp_path / "identity-affine.mat"
    warp_path = tmp_path / "zero-warp.nii.gz"
    backend.write_affine(
        ants.create_ants_transform(transform_type="AffineTransform", dimension=3),
        affine_path,
    )
    backend.write_identity_warp(fixed, warp_path)

    output_path = tmp_path / "registered-unequal-channel.tif"
    transform_channel(
        fixed_image_path=fixed_path,
        moving_image_path=moving_path,
        affine_path=affine_path,
        warp_path=warp_path,
        output_path=output_path,
        pixel_size=_PIXEL_SIZE,
        interpolation="linear",
        native_threads=native_threads,
    )

    written = tifffile.imread(output_path)
    assert written.shape == fixed_shape
    assert read_tiff_geometry(output_path, written.shape) == fixed_geometry
    assert (
        backend.geometry(backend.load_image(output_path, _PIXEL_SIZE))
        == fixed_geometry
    )


def test_transform_channel_applies_identity_chain_clips_and_reuses_exactly(
    tmp_path: Path,
    native_threads: NativeThreadConfig,
) -> None:
    import ants
    import numpy as np
    import tifffile

    shape = (4, 6, 6)
    fixed_data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1.0
    moving_data = np.broadcast_to(
        np.asarray((-10_000.0, 12_000.0, 70_000.0, 50_000.0), dtype=np.float32)[
            :, None, None
        ],
        shape,
    ).copy()
    fixed_geometry = ImageGeometry(
        shape=shape,
        origin=(5.0, 0.0, 0.0),
        spacing=_PIXEL_SIZE,
        direction=((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    )
    fixed_path = tmp_path / "fixed.tif"
    moving_path = tmp_path / "moving-channel.tif"
    tifffile.imwrite(
        fixed_path,
        fixed_data,
        metadata={
            "axes": "ZYX",
            "cyexm": {
                "schema_version": 1,
                "axis_convention_version": AXIS_CONVENTION_VERSION,
                "geometry": fixed_geometry.to_dict(),
            },
        },
        photometric="minisblack",
    )
    tifffile.imwrite(
        moving_path,
        moving_data,
        metadata={"axes": "ZYX"},
        photometric="minisblack",
    )

    backend = AntsRegistrationBackend(native_threads)
    fixed = backend.load_image(fixed_path, _PIXEL_SIZE)
    moving = backend.load_image(moving_path, _PIXEL_SIZE)
    assert read_tiff_geometry(fixed_path, shape) == fixed_geometry
    assert backend.geometry(fixed) == fixed_geometry
    assert backend.geometry(moving) != fixed_geometry
    affine_path = tmp_path / "identity-affine.mat"
    warp_path = tmp_path / "zero-warp.nii.gz"
    backend.write_affine(
        ants.create_ants_transform(transform_type="AffineTransform", dimension=3),
        affine_path,
    )
    backend.write_identity_warp(fixed, warp_path)

    direct = apply_transform_chain(
        fixed=fixed,
        moving=moving,
        affine_path=affine_path,
        warp_path=warp_path,
        interpolation="linear",
        native_threads=native_threads,
    )
    direct_clipped = np.clip(np.asarray(backend.to_array(direct)), 0, 65_535).astype(
        np.uint16
    )
    source_clipped = np.clip(moving_data, 0, 65_535).astype(np.uint16)
    np.testing.assert_array_equal(direct_clipped, source_clipped)

    output_path = tmp_path / "registered-channel.tif"
    assert (
        transform_channel(
            fixed_image_path=fixed_path,
            moving_image_path=moving_path,
            affine_path=affine_path,
            warp_path=warp_path,
            output_path=output_path,
            pixel_size=_PIXEL_SIZE,
            interpolation="linear",
            native_threads=native_threads,
        )
        == output_path.resolve()
    )

    written = tifffile.imread(output_path)
    assert written.dtype == np.dtype(np.uint16)
    assert written.shape == shape
    np.testing.assert_array_equal(written, source_clipped)
    np.testing.assert_array_equal(written, direct_clipped)

    embedded_output_geometry = read_tiff_geometry(output_path, written.shape)
    assert embedded_output_geometry == fixed_geometry
    transformed = backend.load_image(output_path, _PIXEL_SIZE)
    assert backend.geometry(transformed) == fixed_geometry
    store = ArtifactStore()
    record_path = store.record_path(output_path)
    record = CompletionRecord.read(record_path)
    assert record.request.stage == "channel-transform-v1"
    assert record.artifact == FileIdentity.from_path(output_path)

    first_identities = (
        FileIdentity.from_path(output_path),
        FileIdentity.from_path(record_path),
    )
    assert (
        transform_channel(
            fixed_image_path=fixed_path,
            moving_image_path=moving_path,
            affine_path=affine_path,
            warp_path=warp_path,
            output_path=output_path,
            pixel_size=_PIXEL_SIZE,
            interpolation="linear",
            native_threads=native_threads,
        )
        == output_path.resolve()
    )
    assert (
        FileIdentity.from_path(output_path),
        FileIdentity.from_path(record_path),
    ) == first_identities
    assert _temporary_siblings(tmp_path) == ()
