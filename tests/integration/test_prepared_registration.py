from __future__ import annotations

from pathlib import Path

import pytest

from cyexm.registration.artifacts import ArtifactStore
from cyexm.registration.backend import AntsRegistrationBackend
from cyexm.registration.chunked import compute_chunk_grid
from cyexm.registration.prepared import commit_prepared_round, read_prepared_region
from cyexm.registration.whole_image import ImageRegistration
from cyexm.runtime import NativeThreadConfig

pytestmark = pytest.mark.integration


def test_real_ants_prepared_bundle_reconstructs_one_fixed_grid_region(
    synthetic_tiffs: tuple[Path, Path],
    tmp_path: Path,
    native_threads: NativeThreadConfig,
) -> None:
    import numpy as np

    fixed_path, moving_path = synthetic_tiffs
    backend = AntsRegistrationBackend(native_threads)
    registration = ImageRegistration(
        fixed_path,
        moving_path,
        tmp_path / "output",
        imaging_round=4,
        pixel_size=(0.2, 0.3, 0.5),
        enable_logging=False,
        native_threads=native_threads,
        linear_accuracy="dry run",
        _backend=backend,
        _artifact_store=ArtifactStore(),
    )

    manifest = commit_prepared_round(registration)
    chunk = compute_chunk_grid(manifest.geometry.shape, (8, 16, 16), 0.25)[1]
    region = read_prepared_region(manifest, chunk)

    assert manifest.pixeltypes == {
        "fixed_mask": "float",
        "fixed_scalar": "float",
        "moving_linear_mask": "float",
        "moving_linear_scalar": "float",
    }
    assert region.pixeltypes == manifest.pixeltypes
    assert region.geometry.shape == chunk.extended_shape
    assert region.geometry.spacing == manifest.geometry.spacing
    for array in (
        region.fixed_scalar,
        region.moving_linear_scalar,
        region.fixed_mask,
        region.moving_linear_mask,
    ):
        assert array.shape == chunk.extended_shape
        assert array.dtype == np.dtype(np.float32)
        assert array.flags.c_contiguous
        assert array.flags.owndata
        assert np.isfinite(array).all()
    reconstructed = (
        backend.from_geometry(
            region.fixed_scalar,
            geometry=region.geometry,
            pixeltype=region.pixeltypes["fixed_scalar"],
        ),
        backend.from_geometry(
            region.moving_linear_scalar,
            geometry=region.geometry,
            pixeltype=region.pixeltypes["moving_linear_scalar"],
        ),
        backend.from_geometry(
            region.fixed_mask,
            geometry=region.geometry,
            pixeltype=region.pixeltypes["fixed_mask"],
        ),
        backend.from_geometry(
            region.moving_linear_mask,
            geometry=region.geometry,
            pixeltype=region.pixeltypes["moving_linear_mask"],
        ),
    )
    assert tuple(image.pixeltype for image in reconstructed) == (
        "float",
        "float",
        "float",
        "float",
    )
    assert any(
        np.any(values % np.float32(1.0))
        for values in (region.fixed_scalar, region.moving_linear_scalar)
    )
