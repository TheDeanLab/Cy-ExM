from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fakes import FakeChunkBackend

from cyexm.registration._request_policy import CHUNK_SCALAR_POLICY
from cyexm.registration.artifacts import ArtifactStore, CompletionRecord
from cyexm.registration.chunk_stages import (
    AssembledNonlinear,
    ChunkStageArtifacts,
    ChunkStageContext,
    assemble_nonlinear_chunks,
    run_nonlinear_chunk,
)
from cyexm.registration.chunked import ChunkedImageRegistration

pytestmark = pytest.mark.unit


def _registration(
    tmp_path: Path, backend: FakeChunkBackend
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
        imaging_round=2,
        chunk_size=(8, 8, 8),
        overlap_fraction=0.0,
        _backend=backend,
        _artifact_store=ArtifactStore(),
    )


def test_shared_chunk_stage_interfaces_are_internal_values() -> None:
    assert ChunkStageContext.__dataclass_params__.frozen
    assert ChunkStageArtifacts.__dataclass_params__.frozen
    assert AssembledNonlinear.__dataclass_params__.frozen
    assert callable(run_nonlinear_chunk)
    assert callable(assemble_nonlinear_chunks)


def test_serial_registration_commits_float32_scalar_cores(tmp_path: Path) -> None:
    registration = _registration(tmp_path, FakeChunkBackend())

    registration.register()

    for chunk_id in (0, 1):
        core = (
            tmp_path
            / "output"
            / "round_2_chunks"
            / f"chunk_{chunk_id:04d}_registered_core.npy"
        )
        assert core.is_file()
        assert ArtifactStore().record_path(core).is_file()
        array = np.load(core, allow_pickle=False)
        assert array.shape == (8, 8, 8)
        assert array.dtype == np.dtype(np.float32)
        request = CompletionRecord.read(ArtifactStore().record_path(core)).request
        assert request.stage == f"nonlinear-chunk-scalar-core:{chunk_id}"
        assert request.parameters == {
            "scalar_postprocessing_policy": CHUNK_SCALAR_POLICY
        }


def test_missing_scalar_core_is_reconstructed_from_valid_warp(
    tmp_path: Path,
) -> None:
    initial = _registration(tmp_path, FakeChunkBackend())
    initial.register()
    core = tmp_path / "output" / "round_2_chunks" / "chunk_0001_registered_core.npy"
    core.unlink()
    ArtifactStore().record_path(core).unlink()
    resumed_backend = FakeChunkBackend()

    _registration(tmp_path, resumed_backend).register()

    assert core.is_file()
    assert resumed_backend.nonlinear_calls == 0
    assert resumed_backend.registered_chunk_ids == []
