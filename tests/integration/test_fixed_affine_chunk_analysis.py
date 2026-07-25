from __future__ import annotations

from pathlib import Path

import ants
import numpy as np
import pytest
import tifffile
from hpc.fixed_affine_chunk_analysis import jacobian_summary

pytestmark = pytest.mark.integration


def test_identity_warp_has_unit_geometric_jacobian(tmp_path: Path) -> None:
    fixed = tmp_path / "fixed.tif"
    warp = tmp_path / "identity-warp.nii.gz"
    tifffile.imwrite(fixed, np.zeros((12, 13, 14), dtype=np.uint16))
    vector = ants.from_numpy(
        np.zeros((14, 13, 12, 3), dtype=np.float32),
        spacing=(0.4, 0.5, 0.8),
        has_components=True,
    )
    ants.image_write(vector, str(warp))

    summary = jacobian_summary(
        fixed,
        warp,
        voxel_spacing_xyz=(0.4, 0.5, 0.8),
    )

    assert summary["nonfinite_count"] == 0
    assert summary["nonpositive_count"] == 0
    assert summary["minimum"] == pytest.approx(1.0)
    assert summary["maximum"] == pytest.approx(1.0)
