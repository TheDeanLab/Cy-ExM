from __future__ import annotations

from pathlib import Path

import pytest

from cyexm.io.images import zero_mask
from cyexm.registration import linear, nonlinear
from cyexm.runtime import NativeThreadConfig

pytestmark = pytest.mark.integration


def test_dry_run_kernels_preserve_geometry_masks_and_improve_similarity(
    synthetic_pair: tuple[object, object],
    native_threads: NativeThreadConfig,
) -> None:
    import ants
    import numpy as np

    fixed_data, moving_data = synthetic_pair
    direction = np.asarray(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    geometry = {
        "origin": (1.0, 2.0, 3.0),
        "spacing": (0.2, 0.2, 0.5),
        "direction": direction,
    }
    fixed = ants.from_numpy(np.asarray(fixed_data), **geometry)
    moving = ants.from_numpy(np.asarray(moving_data), **geometry)
    fixed_mask = zero_mask(fixed)
    moving_mask = zero_mask(moving)

    affine_image, affine_mask, affine = linear.register_image(
        moving,
        fixed,
        moving_mask=moving_mask,
        fixed_mask=fixed_mask,
        accuracy="dry run",
        native_threads=native_threads,
    )
    assert affine_mask is not None
    nonlinear_image, warp_path = nonlinear.register_image(
        affine_image,
        fixed,
        moving_mask=affine_mask,
        fixed_mask=fixed_mask,
        accuracy="dry run",
        native_threads=native_threads,
    )
    try:
        for image in (affine_image, affine_mask, nonlinear_image):
            assert image.shape == fixed.shape
            assert image.origin == fixed.origin
            assert image.spacing == fixed.spacing
            np.testing.assert_array_equal(image.direction, fixed.direction)
            assert np.isfinite(image.numpy()).all()
        np.testing.assert_array_equal(np.unique(affine_mask.numpy()), (0.0, 1.0))
        assert affine.transform_type == "AffineTransform"

        assert isinstance(warp_path, Path)
        assert warp_path.is_file()
        warp = ants.image_read(str(warp_path))
        assert warp.shape == fixed.shape
        assert warp.has_components
        assert warp.origin == fixed.origin
        # NIfTI spacing is float32; 1e-7 covers serialization rounding only.
        np.testing.assert_allclose(warp.spacing, fixed.spacing, rtol=0.0, atol=1e-7)
        np.testing.assert_array_equal(warp.direction, fixed.direction)
        assert warp.numpy().shape == (*fixed.shape, 3)
        assert np.isfinite(warp.numpy()).all()

        pre_mse = float(np.mean((np.asarray(fixed_data) - moving.numpy()) ** 2))
        nonlinear_mse = float(
            np.mean((np.asarray(fixed_data) - nonlinear_image.numpy()) ** 2)
        )
        assert nonlinear_mse < pre_mse
    finally:
        warp_path.unlink(missing_ok=True)
