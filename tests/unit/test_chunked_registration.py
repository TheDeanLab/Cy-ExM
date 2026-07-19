from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fakes import FakeChunkBackend, fake_image

from cyexm.errors import ChunkRegistrationError
from cyexm.registration.artifacts import ArtifactStore, CompletionRecord, FileIdentity
from cyexm.registration.chunked import ChunkedImageRegistration, ChunkPaths
from cyexm.registration.whole_image import RegistrationPaths

pytestmark = pytest.mark.unit


def make_chunked_registration(
    tmp_path: Path,
    *,
    backend: FakeChunkBackend,
    chunk_size: tuple[int, int, int] = (8, 8, 8),
    overlap_fraction: float = 0.0,
    **kwargs: Any,
) -> ChunkedImageRegistration:
    fixed = tmp_path / "fixed.tif"
    moving = tmp_path / "moving.tif"
    if not fixed.exists():
        fixed.write_bytes(b"fixed-input")
    if not moving.exists():
        moving.write_bytes(b"moving-input")
    return ChunkedImageRegistration(
        fixed,
        moving,
        tmp_path / "output",
        2,
        chunk_size=chunk_size,
        overlap_fraction=overlap_fraction,
        _backend=backend,
        _artifact_store=ArtifactStore(),
        **kwargs,
    )


def round_paths(tmp_path: Path) -> RegistrationPaths:
    return RegistrationPaths.for_round(tmp_path / "output", 2)


def manifest_path(tmp_path: Path) -> Path:
    return tmp_path / "output" / "round_2_chunks" / "manifest.json"


def test_chunk_paths_are_round_scoped_and_include_completion_record(
    tmp_path: Path,
) -> None:
    paths = ChunkPaths.for_chunk(tmp_path, 3, 7)

    assert paths.directory == tmp_path / "round_3_chunks"
    assert paths.warp == paths.directory / "chunk_0007_warp.nii.gz"
    assert paths.completion_record == ArtifactStore().record_path(paths.warp)


def test_chunk_failure_propagates_without_final_success(tmp_path: Path) -> None:
    backend = FakeChunkBackend(fail_chunk=1)
    registration = make_chunked_registration(tmp_path, backend=backend)

    with pytest.raises(ChunkRegistrationError, match="Chunk 1") as error:
        registration.register()

    assert isinstance(error.value.__cause__, RuntimeError)
    assert registration.chunk_paths(0).warp.is_file()
    assert registration.chunk_paths(0).completion_record.is_file()
    assert not manifest_path(tmp_path).exists()
    assert not round_paths(tmp_path).warp.exists()


def test_resume_reuses_completed_chunk(tmp_path: Path) -> None:
    first = FakeChunkBackend(fail_chunk=1)
    with pytest.raises(ChunkRegistrationError):
        make_chunked_registration(tmp_path, backend=first).register()

    resumed = FakeChunkBackend()
    make_chunked_registration(tmp_path, backend=resumed).register()

    assert 0 not in resumed.registered_chunk_ids
    assert 1 in resumed.registered_chunk_ids


def test_local_registration_uses_extended_physical_image_geometry(
    tmp_path: Path,
) -> None:
    direction = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    backend = FakeChunkBackend(
        fixed_image=fake_image(
            data=np.arange(16 * 8 * 8, dtype=np.float32).reshape(16, 8, 8),
            origin=(3.0, 4.0, 5.0),
            direction=direction,
        ),
        moving_image=fake_image(
            data=np.arange(16 * 8 * 8, dtype=np.float32).reshape(16, 8, 8),
            origin=(3.0, 4.0, 5.0),
            direction=direction,
        ),
    )

    make_chunked_registration(tmp_path, backend=backend).register()

    second_fixed = backend.nonlinear_kwargs[1]["fixed_image"]
    second_moving = backend.nonlinear_kwargs[1]["moving_image"]
    assert second_fixed.origin == (3.0, 4.0, 6.2)  # type: ignore[union-attr]
    assert second_moving.origin == second_fixed.origin  # type: ignore[union-attr]
    assert second_fixed.spacing == (0.15, 0.15, 0.15)  # type: ignore[union-attr]
    assert second_fixed.direction == direction  # type: ignore[union-attr]


def test_manifest_is_canonical_ordered_and_records_local_warp_identities(
    tmp_path: Path,
) -> None:
    registration = make_chunked_registration(tmp_path, backend=FakeChunkBackend())

    registration.register()

    path = manifest_path(tmp_path)
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)
    assert (
        text
        == json.dumps(
            payload,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    assert [item["chunk"]["chunk_id"] for item in payload["chunks"]] == [0, 1]
    assert [item["warp"] for item in payload["chunks"]] == [
        FileIdentity.from_path(registration.chunk_paths(chunk_id).warp).to_dict()
        for chunk_id in (0, 1)
    ]
    assert ArtifactStore().record_path(path).is_file()


def test_local_request_captures_sources_geometries_threads_affine_and_grid(
    tmp_path: Path,
) -> None:
    registration = make_chunked_registration(
        tmp_path,
        backend=FakeChunkBackend(),
        overlap_fraction=0.25,
    )

    registration.register()

    request = CompletionRecord.read(
        registration.chunk_paths(0).completion_record
    ).request
    assert request.inputs == (
        FileIdentity.from_path(tmp_path / "fixed.tif"),
        FileIdentity.from_path(tmp_path / "moving.tif"),
    )
    assert request.geometry.shape == (9, 8, 8)
    assert request.parameters["moving_geometry"] == request.geometry.to_dict()
    assert request.native_threads and set(request.native_threads.values()) == {1}
    assert request.upstream == FileIdentity.from_path(round_paths(tmp_path).affine)
    assert request.method == "SyNOnly"
    assert request.accuracy == "high"
    assert request.parameters["chunk"]["chunk_id"] == 0  # type: ignore[index]
    assert request.parameters["chunk_size_zyx"] == [8, 8, 8]
    assert request.parameters["image_shape_zyx"] == [16, 8, 8]
    assert request.parameters["overlap_fraction"] == 0.25
    assert "registration_parameters" in request.parameters


def test_committed_local_warp_reconstructs_scalar_and_mask_before_placement(
    tmp_path: Path,
) -> None:
    backend = FakeChunkBackend()

    make_chunked_registration(tmp_path, backend=backend).register()

    local_applications = [
        (path, interpolation)
        for path, interpolation in backend.warp_applications
        if "chunk_" in path.name
    ]
    assert local_applications == [
        (tmp_path / "output/round_2_chunks/chunk_0000_warp.nii.gz", "linear"),
        (
            tmp_path / "output/round_2_chunks/chunk_0000_warp.nii.gz",
            "nearestNeighbor",
        ),
        (tmp_path / "output/round_2_chunks/chunk_0001_warp.nii.gz", "linear"),
        (
            tmp_path / "output/round_2_chunks/chunk_0001_warp.nii.gz",
            "nearestNeighbor",
        ),
    ]
    assert backend.threshold_mask_calls >= 3


def test_nonfinite_chunk_fails_without_a_local_success_record(
    tmp_path: Path,
) -> None:
    backend = FakeChunkBackend()
    backend.moving_image.data[8, 0, 0] = np.nan
    registration = make_chunked_registration(tmp_path, backend=backend)

    with pytest.raises(ChunkRegistrationError, match="finite") as error:
        registration.register()

    assert isinstance(error.value.__cause__, ValueError)
    assert registration.chunk_paths(0).completion_record.is_file()
    assert not registration.chunk_paths(1).completion_record.exists()
    assert not round_paths(tmp_path).warp.exists()


def test_low_variance_chunk_commits_identity_warp(tmp_path: Path) -> None:
    backend = FakeChunkBackend()
    backend.fixed_image.data[:8] = 2.0
    backend.moving_image.data[:8] = 3.0

    registration = make_chunked_registration(tmp_path, backend=backend)
    registration.register()

    assert backend.identity_chunk_ids == [0]
    assert 0 not in backend.registered_chunk_ids
    assert registration.chunk_paths(0).completion_record.is_file()


def test_invalid_local_warp_shape_is_rejected_before_placement(
    tmp_path: Path,
) -> None:
    backend = FakeChunkBackend(invalid_warp_chunk=1)
    registration = make_chunked_registration(tmp_path, backend=backend)

    with pytest.raises(ChunkRegistrationError, match="three components"):
        registration.register()

    assert registration.chunk_paths(0).completion_record.is_file()
    assert not registration.chunk_paths(1).completion_record.exists()
    assert not manifest_path(tmp_path).exists()


def test_nifti_header_roundoff_preserves_compatible_local_geometry(
    tmp_path: Path,
) -> None:
    # NIfTI stores geometry headers at finite precision. This difference is far
    # below a nanometer and does not represent another physical sampling grid.
    backend = FakeChunkBackend(warp_origin_delta=1e-7)

    make_chunked_registration(tmp_path, backend=backend).register()

    assert round_paths(tmp_path).warp.is_file()


def test_material_local_warp_geometry_mismatch_is_rejected(tmp_path: Path) -> None:
    backend = FakeChunkBackend(warp_origin_delta=0.01)

    with pytest.raises(ChunkRegistrationError, match="geometry"):
        make_chunked_registration(tmp_path, backend=backend).register()


def test_scalar_core_shape_is_validated_after_local_commit(tmp_path: Path) -> None:
    backend = FakeChunkBackend(invalid_scalar_chunk=1)
    registration = make_chunked_registration(tmp_path, backend=backend)

    with pytest.raises(ChunkRegistrationError, match="scalar"):
        registration.register()

    assert registration.chunk_paths(1).completion_record.is_file()
    assert not manifest_path(tmp_path).exists()
    assert not round_paths(tmp_path).warp.exists()


def test_force_replacement_writer_failure_preserves_old_local_pair(
    tmp_path: Path,
) -> None:
    initial = make_chunked_registration(tmp_path, backend=FakeChunkBackend())
    initial.register()
    chunk = initial.chunk_paths(0)
    old_warp = chunk.warp.read_bytes()
    old_record = chunk.completion_record.read_bytes()

    replacement = FakeChunkBackend(missing_warp_chunk=0)
    with pytest.raises(ChunkRegistrationError):
        make_chunked_registration(
            tmp_path,
            backend=replacement,
            force_override=True,
        ).register()

    assert chunk.warp.read_bytes() == old_warp
    assert chunk.completion_record.read_bytes() == old_record


def test_full_candidate_warp_is_unique_validated_and_float32(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first_backend = FakeChunkBackend()
    make_chunked_registration(first_root, backend=first_backend).register()
    first_candidate = first_backend.written_warp_paths[0]
    first_image = first_backend.read_warp(first_candidate)

    second_backend = FakeChunkBackend()
    make_chunked_registration(second_root, backend=second_backend).register()
    second_candidate = second_backend.written_warp_paths[0]

    assert first_candidate != second_candidate
    assert first_candidate.parent == first_root / "output"
    assert first_image.data.dtype == np.float32  # type: ignore[union-attr]
    assert first_image.data.shape == (16, 8, 8, 3)  # type: ignore[union-attr]


def test_parallel_chunked_class_is_not_exported_in_a_fresh_process() -> None:
    code = """
import sys
import cyexm.registration as registration

assert not hasattr(registration, "ParallelChunkedImageRegistration")
assert registration.__all__ == [
    "ChunkedImageRegistration",
    "ImageRegistration",
    "register_round",
]
for root in ("ants", "numpy", "scipy", "tifffile"):
    assert not any(name == root or name.startswith(root + ".") for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
