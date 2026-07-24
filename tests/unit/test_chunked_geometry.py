from __future__ import annotations

import subprocess
import sys
from itertools import product

import numpy as np
import pytest

from cyexm import chunking
from cyexm.chunking import compute_chunk_grid as shared_compute_chunk_grid
from cyexm.registration import chunked as registration_chunked
from cyexm.registration.chunked import (
    ChunkInfo,
    compute_chunk_grid,
    extract_chunk,
    stitch_core,
)

pytestmark = pytest.mark.unit


def test_registration_chunk_geometry_is_shared_by_import_identity() -> None:
    assert registration_chunked.ChunkInfo is chunking.ChunkInfo
    assert registration_chunked.compute_chunk_grid is shared_compute_chunk_grid
    assert registration_chunked.extract_chunk is chunking.extract_chunk
    assert registration_chunked.stitch_core is chunking.stitch_core


def test_shared_chunk_geometry_import_is_standard_library_only() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import cyexm.chunking; "
                "forbidden={'numpy','scipy','ants','dask'}; "
                "assert not forbidden.intersection(sys.modules)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_nontrivial_chunk_grid_is_exact() -> None:
    chunks = shared_compute_chunk_grid((9, 17, 19), (4, 8, 8), 0.25)
    z_bounds = ((0, 4, 0, 4), (3, 7, 3, 7), (6, 9, 6, 9))
    y_bounds = ((0, 8, 0, 9), (6, 14, 5, 15), (12, 17, 11, 17))
    x_bounds = (
        (0, 8, 0, 9),
        (6, 14, 5, 15),
        (12, 19, 11, 19),
        (18, 19, 17, 19),
    )

    assert chunks == tuple(
        ChunkInfo(
            chunk_id=chunk_id,
            z_start=z[0],
            z_end=z[1],
            y_start=y[0],
            y_end=y[1],
            x_start=x[0],
            x_end=x[1],
            z_start_ext=z[2],
            z_end_ext=z[3],
            y_start_ext=y[2],
            y_end_ext=y[3],
            x_start_ext=x[2],
            x_end_ext=x[3],
        )
        for chunk_id, (z, y, x) in enumerate(product(z_bounds, y_bounds, x_bounds))
    )


def test_single_chunk_covers_smaller_volume() -> None:
    chunks = compute_chunk_grid(
        (32, 64, 96),
        chunk_size=(512, 512, 512),
        overlap_fraction=0.15,
    )
    assert chunks == (
        ChunkInfo(
            chunk_id=0,
            z_start=0,
            z_end=32,
            y_start=0,
            y_end=64,
            x_start=0,
            x_end=96,
            z_start_ext=0,
            z_end_ext=32,
            y_start_ext=0,
            y_end_ext=64,
            x_start_ext=0,
            x_end_ext=96,
        ),
    )


def test_legacy_grid_positions_and_overlap_are_preserved() -> None:
    chunks = compute_chunk_grid(
        (20, 20, 1000),
        chunk_size=(20, 20, 512),
        overlap_fraction=0.15,
    )
    assert [(chunk.x_start, chunk.x_end) for chunk in chunks] == [
        (0, 512),
        (436, 948),
        (872, 1000),
    ]
    assert [(chunk.x_start_ext, chunk.x_end_ext) for chunk in chunks] == [
        (0, 550),
        (398, 986),
        (834, 1000),
    ]


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [((0, 8, 8), 0.1), ((8, 8, 8), -0.1), ((8, 8, 8), 1.0)],
)
def test_invalid_chunk_configuration_is_rejected(
    chunk_size: tuple[int, int, int], overlap: float
) -> None:
    with pytest.raises(ValueError):
        compute_chunk_grid((16, 16, 16), chunk_size, overlap)


def test_grid_is_ordered_zyx_and_uses_exclusive_upper_bounds() -> None:
    chunks = compute_chunk_grid((5, 6, 7), (3, 4, 5), 0.0)

    starts = [(chunk.z_start, chunk.y_start, chunk.x_start) for chunk in chunks]
    assert starts == sorted(starts)
    assert all(
        0 <= chunk.z_start < chunk.z_end <= 5
        and 0 <= chunk.y_start < chunk.y_end <= 6
        and 0 <= chunk.x_start < chunk.x_end <= 7
        for chunk in chunks
    )
    assert sum(chunk.core_shape == (2, 2, 2) for chunk in chunks) == 1


def test_extended_shape_and_core_relative_slices_are_exact() -> None:
    chunk = compute_chunk_grid((20, 20, 1000), (20, 20, 512), 0.15)[1]

    assert chunk.extended_shape == (20, 20, 588)
    assert chunk.core_shape == (20, 20, 512)
    assert chunk.core_within_extended == (
        slice(0, 20),
        slice(0, 20),
        slice(38, 550),
    )


def test_extract_chunk_uses_extended_bounds_without_including_upper_bound() -> None:
    array = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    chunk = ChunkInfo(
        chunk_id=0,
        z_start=1,
        z_end=3,
        y_start=2,
        y_end=4,
        x_start=2,
        x_end=5,
        z_start_ext=0,
        z_end_ext=4,
        y_start_ext=1,
        y_end_ext=5,
        x_start_ext=1,
        x_end_ext=6,
    )

    extracted = extract_chunk(array, chunk)

    assert extracted.shape == chunk.extended_shape
    np.testing.assert_array_equal(extracted, array[0:4, 1:5, 1:6])


def test_stitch_core_places_only_the_nonoverlap_region() -> None:
    destination = np.zeros((4, 5, 6), dtype=np.float32)
    chunk = ChunkInfo(
        chunk_id=0,
        z_start=1,
        z_end=3,
        y_start=2,
        y_end=4,
        x_start=2,
        x_end=5,
        z_start_ext=0,
        z_end_ext=4,
        y_start_ext=1,
        y_end_ext=5,
        x_start_ext=1,
        x_end_ext=6,
    )
    extended = np.arange(4 * 4 * 5, dtype=np.float32).reshape(4, 4, 5)

    stitch_core(destination, extended, chunk)

    expected = np.zeros_like(destination)
    expected[1:3, 2:4, 2:5] = extended[1:3, 1:3, 1:4]
    np.testing.assert_array_equal(destination, expected)


def test_stitch_core_requires_three_warp_components() -> None:
    chunk = compute_chunk_grid((4, 4, 4), (4, 4, 4), 0.0)[0]

    with pytest.raises(ValueError, match="three components"):
        stitch_core(
            np.zeros((4, 4, 4, 3), dtype=np.float32),
            np.zeros((4, 4, 4, 2), dtype=np.float32),
            chunk,
        )
