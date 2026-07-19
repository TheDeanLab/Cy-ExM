from __future__ import annotations

from pathlib import Path

import pytest

from cyexm.io.images import zero_mask
from cyexm.registration import nonlinear
from cyexm.registration.artifacts import ArtifactStore
from cyexm.registration.backend import AntsRegistrationBackend
from cyexm.registration.chunked import ChunkedImageRegistration, compute_chunk_grid
from cyexm.registration.whole_image import ImageRegistration, RegistrationPaths
from cyexm.runtime import NativeThreadConfig

pytestmark = pytest.mark.integration

_PIXEL_SIZE = (0.2, 0.2, 0.5)
_SHAPE = (16, 32, 32)


class _SharedWarpBackend(AntsRegistrationBackend):
    """Use real ANTs application kernels with one characterized transform pair."""

    def __init__(
        self,
        native_threads: NativeThreadConfig,
        shared_warp: Path,
    ) -> None:
        super().__init__(native_threads)
        self.shared_warp = shared_warp

    def linear(self, **kwargs: object) -> tuple[object, object | None, object]:
        import ants

        fixed = kwargs["fixed_image"]
        moving = kwargs["moving_image"]
        moving_mask = kwargs["moving_mask"]
        transform = ants.create_ants_transform(
            transform_type="AffineTransform",
            dimension=3,
        )
        scalar = transform.apply_to_image(
            image=moving,
            reference=fixed,
            interpolation="linear",
        )
        mask = None
        if moving_mask is not None:
            raw_mask = transform.apply_to_image(
                image=moving_mask,
                reference=fixed,
                interpolation="nearestNeighbor",
            )
            mask = self.threshold_mask(raw_mask)
        return scalar, mask, transform

    def nonlinear(self, **kwargs: object) -> tuple[object, Path]:
        return kwargs["moving_image"], self.shared_warp


def _run_whole(
    fixed_path: Path,
    moving_path: Path,
    output: Path,
    native_threads: NativeThreadConfig,
    shared_warp: Path,
) -> RegistrationPaths:
    ImageRegistration(
        fixed_path,
        moving_path,
        output,
        imaging_round=2,
        pixel_size=_PIXEL_SIZE,
        enable_logging=False,
        native_threads=native_threads,
        linear_accuracy="dry run",
        nonlinear_accuracy="dry run",
        _backend=_SharedWarpBackend(native_threads, shared_warp),
    ).register()
    return RegistrationPaths.for_round(output, 2)


def _run_one_chunk(
    fixed_path: Path,
    moving_path: Path,
    output: Path,
    native_threads: NativeThreadConfig,
    shared_warp: Path,
) -> tuple[RegistrationPaths, Path]:
    registration = ChunkedImageRegistration(
        fixed_path,
        moving_path,
        output,
        imaging_round=2,
        pixel_size=_PIXEL_SIZE,
        enable_logging=False,
        chunk_size=_SHAPE,
        overlap_fraction=0.25,
        native_threads=native_threads,
        linear_accuracy="dry run",
        nonlinear_accuracy="dry run",
        _backend=_SharedWarpBackend(native_threads, shared_warp),
    )
    registration.register()
    return RegistrationPaths.for_round(output, 2), registration.chunk_paths(0).warp


def test_one_chunk_serial_output_matches_whole_image_with_same_real_warp(
    synthetic_tiffs: tuple[Path, Path],
    tmp_path: Path,
    native_threads: NativeThreadConfig,
) -> None:
    import ants
    import numpy as np
    import tifffile

    fixed_path, moving_path = synthetic_tiffs
    backend = AntsRegistrationBackend(native_threads)
    fixed = backend.load_image(fixed_path, _PIXEL_SIZE)
    moving = backend.load_image(moving_path, _PIXEL_SIZE)
    _registered, candidate_warp = nonlinear.register_image(
        moving,
        fixed,
        moving_mask=zero_mask(moving),
        fixed_mask=zero_mask(fixed),
        accuracy="dry run",
        native_threads=native_threads,
    )
    shared_warp = tmp_path / "shared-real-warp.nii.gz"
    backend.write_warp(ants.image_read(str(candidate_warp)), shared_warp)
    candidate_warp.unlink(missing_ok=True)

    whole = _run_whole(
        fixed_path,
        moving_path,
        tmp_path / "whole",
        native_threads,
        shared_warp,
    )
    chunked, local_warp = _run_one_chunk(
        fixed_path,
        moving_path,
        tmp_path / "chunked",
        native_threads,
        shared_warp,
    )

    assert len(compute_chunk_grid(_SHAPE, _SHAPE, 0.25)) == 1
    store = ArtifactStore()
    for artifact in (whole.warp, chunked.warp, local_warp):
        assert artifact.is_file()
        assert store.record_path(artifact).is_file()

    reference_warp = ants.image_read(str(shared_warp)).numpy()
    whole_warp = ants.image_read(str(whole.warp)).numpy()
    chunked_warp = ants.image_read(str(chunked.warp)).numpy()
    local_warp_array = ants.image_read(str(local_warp)).numpy()
    for warp in (whole_warp, chunked_warp, local_warp_array):
        assert warp.shape == (*_SHAPE, 3)
        # The approved tolerance covers NIfTI serialization and chunk float32 assembly.
        np.testing.assert_allclose(warp, reference_warp, rtol=1e-4, atol=1e-5)

    whole_output = tifffile.imread(whole.registered)
    chunked_output = tifffile.imread(chunked.registered)
    assert whole_output.dtype == np.dtype(np.uint16)
    assert chunked_output.dtype == np.dtype(np.uint16)
    np.testing.assert_allclose(
        chunked_output,
        whole_output,
        rtol=1e-4,
        atol=1e-5,
    )
