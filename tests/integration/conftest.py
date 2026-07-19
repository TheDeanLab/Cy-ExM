from __future__ import annotations

import os
from pathlib import Path

import pytest

from cyexm.runtime import NativeThreadConfig, configure_native_threads


@pytest.fixture(scope="session", autouse=True)
def native_threads() -> NativeThreadConfig:
    """Resolve one native thread before any integration test imports NumPy/ANTs."""
    os.environ["ANTS_RANDOM_SEED"] = "20260719"
    config = NativeThreadConfig(itk_threads=1, blas_threads=1)
    resolved = configure_native_threads(config)
    assert set(resolved.values.values()) == {1}
    return config


@pytest.fixture
def synthetic_pair(native_threads: NativeThreadConfig) -> tuple[object, object]:
    del native_threads
    import numpy as np

    z, y, x = np.mgrid[:16, :32, :32]
    fixed = np.exp(
        -(((z - 8.0) / 3.0) ** 2 + ((y - 16.0) / 6.0) ** 2 + ((x - 16.0) / 6.0) ** 2)
    ).astype(np.float32)
    moving = np.roll(fixed, shift=(0, 1, -1), axis=(0, 1, 2))
    fixed[fixed < 1e-4] = 0
    moving[moving < 1e-4] = 0
    return fixed, moving


@pytest.fixture
def synthetic_tiffs(
    tmp_path: Path,
    synthetic_pair: tuple[object, object],
    native_threads: NativeThreadConfig,
) -> tuple[Path, Path]:
    del native_threads
    import numpy as np
    import tifffile

    fixed, moving = synthetic_pair
    fixed_path = tmp_path / "fixed.tif"
    moving_path = tmp_path / "moving.tif"
    # Exercise the workflow's full uint16 range while retaining float input data.
    tifffile.imwrite(
        fixed_path,
        np.asarray(fixed, dtype=np.float32) * np.float32(60_000.0),
        metadata={"axes": "ZYX"},
    )
    tifffile.imwrite(
        moving_path,
        np.asarray(moving, dtype=np.float32) * np.float32(60_000.0),
        metadata={"axes": "ZYX"},
    )
    return fixed_path, moving_path
