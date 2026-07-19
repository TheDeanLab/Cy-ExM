from __future__ import annotations

import pytest

from cyexm.registration.backend import AntsRegistrationBackend
from cyexm.runtime import NativeThreadConfig

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("role", ["scalar", "mask"])
def test_crop_image_maps_external_zyx_start_to_real_ants_xyz_geometry(
    role: str,
    native_threads: NativeThreadConfig,
) -> None:
    import ants
    import numpy as np

    scalar_zyx = np.arange(6 * 7 * 8, dtype=np.float32).reshape(6, 7, 8)
    array_zyx = (
        scalar_zyx if role == "scalar" else (scalar_zyx % 3 == 0).astype(np.float32)
    )
    origin = (10.0, 20.0, 30.0)
    spacing = (0.5, 1.0, 2.0)
    direction = np.asarray(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    image = ants.from_numpy(
        np.transpose(array_zyx, (2, 1, 0)),
        origin=origin,
        spacing=spacing,
        direction=direction,
    )
    bounds = (1, 5, 2, 6, 3, 6)

    backend = AntsRegistrationBackend(native_threads)
    cropped = backend.crop_image(image, bounds)

    expected_origin = (8.0, 21.5, 32.0)
    np.testing.assert_array_equal(cropped.origin, expected_origin)
    assert cropped.spacing == image.spacing
    np.testing.assert_array_equal(cropped.direction, image.direction)
    np.testing.assert_array_equal(
        backend.to_array(cropped),
        array_zyx[1:5, 2:6, 3:6],
    )


def test_vector_array_round_trip_transposes_only_spatial_axes(
    native_threads: NativeThreadConfig,
) -> None:
    import ants
    import numpy as np

    external_zyx = np.zeros((2, 3, 4, 3), dtype=np.float32)
    external_zyx[..., 0] = 11.0
    external_zyx[..., 1] = 22.0
    external_zyx[..., 2] = 33.0
    reference = ants.from_numpy(np.zeros((4, 3, 2), dtype=np.float32))
    backend = AntsRegistrationBackend(native_threads)

    vector = backend.from_array(
        external_zyx,
        reference=reference,
        has_components=True,
    )

    assert vector.shape == (4, 3, 2)
    assert vector.has_components
    np.testing.assert_array_equal(
        vector.numpy(),
        np.transpose(external_zyx, (2, 1, 0, 3)),
    )
    np.testing.assert_array_equal(backend.to_array(vector), external_zyx)
