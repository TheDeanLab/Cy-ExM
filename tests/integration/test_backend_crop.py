from __future__ import annotations

import pytest

from cyexm.registration.backend import AntsRegistrationBackend
from cyexm.runtime import NativeThreadConfig

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("role", ["scalar", "mask"])
def test_crop_image_uses_numpy_axis_start_for_real_ants_geometry(
    role: str,
    native_threads: NativeThreadConfig,
) -> None:
    import ants
    import numpy as np

    scalar = np.arange(6 * 7 * 8, dtype=np.float32).reshape(6, 7, 8)
    array = scalar if role == "scalar" else (scalar % 3 == 0).astype(np.float32)
    origin = (10.0, 20.0, 30.0)
    spacing = (0.5, 1.0, 2.0)
    direction = np.asarray(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    image = ants.from_numpy(
        array,
        origin=origin,
        spacing=spacing,
        direction=direction,
    )
    bounds = (1, 5, 2, 6, 3, 7)

    cropped = AntsRegistrationBackend(native_threads).crop_image(image, bounds)

    expected_origin = ants.transform_index_to_physical_point(image, (1, 2, 3))
    np.testing.assert_array_equal(cropped.origin, expected_origin)
    assert cropped.spacing == image.spacing
    np.testing.assert_array_equal(cropped.direction, image.direction)
    np.testing.assert_array_equal(cropped.numpy(), array[1:5, 2:6, 3:7])
