from __future__ import annotations

from pathlib import Path

import pytest

from cyexm.io.images import ants_geometry, load_ants_image
from cyexm.io.tiff import write_uint16_tiff
from cyexm.runtime import NativeThreadConfig

pytestmark = pytest.mark.integration


def test_anisotropic_noncubic_tiff_load_and_round_trip_have_physical_xyz_com(
    tmp_path: Path,
    native_threads: NativeThreadConfig,
) -> None:
    del native_threads
    import ants
    import numpy as np
    import tifffile

    data_zyx = np.zeros((4, 6, 7), dtype=np.float32)
    data_zyx[2, 4, 5] = 1.0
    spacing_xyz = (0.2, 0.3, 0.5)
    source = tmp_path / "source.tif"
    tifffile.imwrite(
        source,
        data_zyx,
        metadata={"axes": "ZYX"},
        photometric="minisblack",
    )

    loaded = load_ants_image(source, spacing_xyz)

    # External ZYX index (2, 4, 5) is physical XYZ index (5, 4, 2).
    expected_physical_xyz = (1.0, 1.2, 1.0)
    np.testing.assert_allclose(
        ants.get_center_of_mass(loaded),
        expected_physical_xyz,
        rtol=0.0,
        atol=1e-7,
    )
    assert ants_geometry(loaded).shape == data_zyx.shape

    destination = tmp_path / "round-trip.tif"
    write_uint16_tiff(loaded, destination, min_value=0.0, max_value=1.0)
    restored = load_ants_image(destination, spacing_xyz)

    np.testing.assert_allclose(
        ants.get_center_of_mass(restored),
        expected_physical_xyz,
        rtol=0.0,
        atol=1e-7,
    )
    assert ants_geometry(restored) == ants_geometry(loaded)
    np.testing.assert_array_equal(tifffile.imread(destination), data_zyx)
