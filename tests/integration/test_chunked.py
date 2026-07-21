from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from cyexm.io.images import zero_mask
from cyexm.registration import linear, nonlinear
from cyexm.registration._native_paths import discard_path
from cyexm.registration.artifacts import ArtifactStore
from cyexm.registration.backend import AntsRegistrationBackend
from cyexm.registration.chunked import ChunkedImageRegistration, compute_chunk_grid
from cyexm.registration.whole_image import ImageRegistration, RegistrationPaths
from cyexm.runtime import NativeThreadConfig
from cyexm.spatial import geometry_compatible

pytestmark = pytest.mark.integration

_PIXEL_SIZE = (0.2, 0.2, 0.5)
_SHAPE = (16, 32, 32)


class _EstimatedTransformPairBackend(AntsRegistrationBackend):
    """Apply one externally estimated real affine and warp in both workflows."""

    def __init__(
        self,
        native_threads: NativeThreadConfig,
        shared_affine: object,
        shared_warp: Path,
        returned_candidates: list[Path],
    ) -> None:
        super().__init__(native_threads)
        self.shared_affine = shared_affine
        self.shared_warp = shared_warp
        self.returned_candidates = returned_candidates

    def linear(self, **kwargs: object) -> tuple[object, object | None, object]:
        fixed = kwargs["fixed_image"]
        moving = kwargs["moving_image"]
        moving_mask = kwargs["moving_mask"]
        scalar = self.apply_affine(
            fixed=fixed,
            moving=moving,
            transform=self.shared_affine,
            interpolation="linear",
        )
        mask = None
        if moving_mask is not None:
            raw_mask = self.apply_affine(
                fixed=fixed,
                moving=moving_mask,
                transform=self.shared_affine,
                interpolation="nearestNeighbor",
            )
            mask = self.threshold_mask(raw_mask)
        return scalar, mask, self.shared_affine

    def nonlinear(self, **kwargs: object) -> tuple[object, Path]:
        with tempfile.NamedTemporaryFile(
            dir=self.shared_warp.parent,
            prefix=".estimated-pair-candidate-",
            suffix=".nii.gz",
            delete=False,
        ) as temporary:
            candidate = Path(temporary.name)
        shutil.copyfile(self.shared_warp, candidate)
        self.returned_candidates.append(candidate)
        return kwargs["moving_image"], candidate


def _run_whole_with_estimated_pair(
    fixed_path: Path,
    moving_path: Path,
    output: Path,
    native_threads: NativeThreadConfig,
    shared_affine: object,
    shared_warp: Path,
    returned_candidates: list[Path],
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
        _backend=_EstimatedTransformPairBackend(
            native_threads, shared_affine, shared_warp, returned_candidates
        ),
    ).register()
    return RegistrationPaths.for_round(output, 2)


def _run_one_chunk_with_estimated_pair(
    fixed_path: Path,
    moving_path: Path,
    output: Path,
    native_threads: NativeThreadConfig,
    shared_affine: object,
    shared_warp: Path,
    returned_candidates: list[Path],
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
        _backend=_EstimatedTransformPairBackend(
            native_threads, shared_affine, shared_warp, returned_candidates
        ),
    )
    registration.register()
    return RegistrationPaths.for_round(output, 2), registration.chunk_paths(0).warp


def test_one_chunk_and_whole_apply_the_same_real_estimated_transform_pair(
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
    fixed_mask = zero_mask(fixed)
    moving_mask = zero_mask(moving)
    affine_image, affine_mask, estimated_affine = linear.register_image(
        moving_image=moving,
        fixed_image=fixed,
        moving_mask=moving_mask,
        fixed_mask=fixed_mask,
        accuracy="dry run",
        native_threads=native_threads,
    )
    assert affine_mask is not None
    _registered, candidate_warp = nonlinear.register_image(
        affine_image,
        fixed,
        moving_mask=affine_mask,
        fixed_mask=fixed_mask,
        accuracy="dry run",
        native_threads=native_threads,
    )
    shared_warp = tmp_path / "shared-real-warp.nii.gz"
    try:
        backend.write_warp(ants.image_read(str(candidate_warp)), shared_warp)
    finally:
        discard_path(candidate_warp)
    returned_candidates: list[Path] = []

    whole = _run_whole_with_estimated_pair(
        fixed_path,
        moving_path,
        tmp_path / "whole",
        native_threads,
        shared_affine=estimated_affine,
        shared_warp=shared_warp,
        returned_candidates=returned_candidates,
    )
    chunked, local_warp = _run_one_chunk_with_estimated_pair(
        fixed_path,
        moving_path,
        tmp_path / "chunked",
        native_threads,
        shared_affine=estimated_affine,
        shared_warp=shared_warp,
        returned_candidates=returned_candidates,
    )

    assert len(compute_chunk_grid(_SHAPE, _SHAPE, 0.25)) == 1
    store = ArtifactStore()
    for artifact in (whole.warp, chunked.warp, local_warp):
        assert artifact.is_file()
        assert store.record_path(artifact).is_file()
    assert returned_candidates
    assert all(not path.exists() for path in returned_candidates)

    for affine_path in (whole.affine, chunked.affine):
        persisted_affine = backend.read_affine(affine_path)
        np.testing.assert_allclose(
            persisted_affine.parameters,
            estimated_affine.parameters,
            rtol=0.0,
            atol=0.0,
        )

    reference_warp = ants.image_read(str(shared_warp)).numpy()
    whole_warp = ants.image_read(str(whole.warp)).numpy()
    chunked_warp = ants.image_read(str(chunked.warp)).numpy()
    local_warp_array = ants.image_read(str(local_warp)).numpy()
    for warp in (whole_warp, chunked_warp, local_warp_array):
        assert warp.shape == (*reversed(_SHAPE), 3)
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


def test_unequal_native_shapes_are_affine_resampled_before_chunking(
    tmp_path: Path,
    native_threads: NativeThreadConfig,
) -> None:
    import numpy as np
    import tifffile

    fixed_shape = (12, 16, 18)
    moving_shape = (10, 14, 15)
    fixed_path = tmp_path / "fixed.tif"
    moving_path = tmp_path / "moving.tif"

    def centered_blob(shape: tuple[int, int, int]) -> object:
        coordinates = np.indices(shape, dtype=np.float32)
        center = np.asarray([(size - 1) / 2.0 for size in shape], dtype=np.float32)
        squared_distance = sum(
            (coordinates[axis] - center[axis]) ** 2 for axis in range(3)
        )
        return np.exp(-squared_distance / 12.0).astype(np.float32) * 50_000.0

    tifffile.imwrite(
        fixed_path,
        centered_blob(fixed_shape),
        metadata={"axes": "ZYX"},
        photometric="minisblack",
    )
    tifffile.imwrite(
        moving_path,
        centered_blob(moving_shape),
        metadata={"axes": "ZYX"},
        photometric="minisblack",
    )

    class RecordingIdentityNonlinearBackend(AntsRegistrationBackend):
        def __init__(self) -> None:
            super().__init__(native_threads)
            self.nonlinear_geometries: list[tuple[object, object]] = []

        def nonlinear(self, **kwargs: object) -> tuple[object, Path]:
            fixed = kwargs["fixed_image"]
            moving = kwargs["moving_image"]
            self.nonlinear_geometries.append(
                (self.geometry(fixed), self.geometry(moving))
            )
            with tempfile.NamedTemporaryFile(
                dir=tmp_path,
                prefix=".identity-local-warp-",
                suffix=".nii.gz",
                delete=False,
            ) as temporary:
                candidate = Path(temporary.name)
            self.write_identity_warp(fixed, candidate)
            return moving, candidate

    backend = RecordingIdentityNonlinearBackend()
    output = tmp_path / "chunked-unequal"
    registration = ChunkedImageRegistration(
        fixed_path,
        moving_path,
        output,
        imaging_round=2,
        pixel_size=_PIXEL_SIZE,
        enable_logging=False,
        chunk_size=fixed_shape,
        overlap_fraction=0.25,
        native_threads=native_threads,
        linear_accuracy="dry run",
        nonlinear_accuracy="dry run",
        _backend=backend,
    )

    registration.register()

    fixed_geometry = backend.geometry(backend.load_image(fixed_path, _PIXEL_SIZE))
    assert backend.geometry(backend.load_image(moving_path, _PIXEL_SIZE)).shape == (
        moving_shape
    )
    assert backend.nonlinear_geometries
    assert all(
        fixed_chunk_geometry == moving_chunk_geometry
        for fixed_chunk_geometry, moving_chunk_geometry in backend.nonlinear_geometries
    )
    paths = RegistrationPaths.for_round(output, 2)
    assert backend.geometry(backend.load_image(paths.registered, _PIXEL_SIZE)) == (
        fixed_geometry
    )
    # NIfTI serializes spacing as float32; the scientific geometry contract uses
    # the repository's explicit serialization tolerance for this representation.
    assert geometry_compatible(
        backend.geometry(backend.read_warp(paths.warp)), fixed_geometry
    )
