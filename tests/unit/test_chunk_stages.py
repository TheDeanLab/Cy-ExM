from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fakes import FakeChunkBackend, FakeImage, fake_image

from cyexm.chunking import ChunkInfo, compute_chunk_grid
from cyexm.errors import RegistrationStageError
from cyexm.registration import chunked as chunked_module
from cyexm.registration._request_policy import CHUNK_SCALAR_POLICY
from cyexm.registration.artifacts import ArtifactStore, CompletionRecord
from cyexm.registration.chunk_stages import (
    AssembledNonlinear,
    ChunkRegion,
    ChunkStageArtifacts,
    ChunkStageContext,
    _require_complete_coverage,
    assemble_nonlinear_chunks,
    run_nonlinear_chunk,
)
from cyexm.registration.chunked import ChunkedImageRegistration

pytestmark = pytest.mark.unit


class _CapturedAssembly(RuntimeError):
    pass


class OverlapValueBackend(FakeChunkBackend):
    assembled_scalars: list[np.ndarray]
    assembled_displacements: list[np.ndarray]

    @staticmethod
    def _artifact_chunk_id(path: Path) -> int | None:
        if path.name.startswith("chunk_"):
            return int(path.name.split("_", 2)[1])
        return None

    def read_warp(self, path: Path) -> object:
        warp = super().read_warp(path)
        chunk_id = self._artifact_chunk_id(path)
        if chunk_id is None or not isinstance(warp, FakeImage):
            return warp
        return fake_image(
            data=np.full((*warp.shape, 3), chunk_id, dtype=np.float32),
            origin=warp.origin,
            spacing=warp.spacing,
            direction=warp.direction,
            has_components=True,
        )

    def apply_warp(
        self,
        *,
        fixed: object,
        moving: object,
        warp_path: Path,
        interpolation: str,
    ) -> object:
        chunk_id = self._artifact_chunk_id(warp_path)
        if (
            chunk_id is not None
            and interpolation == "linear"
            and isinstance(moving, FakeImage)
        ):
            return fake_image(
                data=np.full(moving.shape, chunk_id, dtype=np.float32),
                origin=moving.origin,
                spacing=moving.spacing,
                direction=moving.direction,
            )
        return super().apply_warp(
            fixed=fixed,
            moving=moving,
            warp_path=warp_path,
            interpolation=interpolation,
        )

    def histogram_match(self, *, source: object, reference: object) -> object:
        del reference
        return source

    def from_array(
        self,
        array: object,
        *,
        reference: object,
        has_components: bool = False,
    ) -> object:
        converted = np.asarray(array)
        if converted.shape == self.fixed_image.shape:
            self.assembled_scalars.append(converted.copy())
        elif converted.shape == (*self.fixed_image.shape, 3):
            self.assembled_displacements.append(converted.copy())
        return super().from_array(
            converted,
            reference=reference,
            has_components=has_components,
        )


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


def _capture_chunk_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    overlap_fraction: float = 0.0,
) -> tuple[
    ChunkStageContext,
    tuple[ChunkStageArtifacts, ...],
    ChunkRegion,
    ChunkInfo,
]:
    captured: dict[str, Any] = {}
    backend = FakeChunkBackend(chunk_depth=6)
    registration = _registration(tmp_path, backend)
    registration.overlap_fraction = overlap_fraction
    canonical_run = chunked_module.run_nonlinear_chunk

    def capture_run(context: object, region: object, chunk: object) -> object:
        captured.setdefault("region", region)
        captured.setdefault("chunk", chunk)
        return canonical_run(context, region, chunk)  # type: ignore[arg-type]

    def stop_before_assembly(context: object, artifacts: object) -> object:
        captured["context"] = context
        captured["artifacts"] = tuple(artifacts)  # type: ignore[arg-type]
        raise _CapturedAssembly

    monkeypatch.setattr(chunked_module, "run_nonlinear_chunk", capture_run)
    monkeypatch.setattr(
        chunked_module,
        "assemble_nonlinear_chunks",
        stop_before_assembly,
    )
    with pytest.raises(RegistrationStageError) as error:
        registration.register()
    assert isinstance(error.value.__cause__, _CapturedAssembly)
    return (
        captured["context"],
        captured["artifacts"],
        captured["region"],
        captured["chunk"],
    )


def test_shared_chunk_stage_interfaces_are_internal_values() -> None:
    assert vars(ChunkStageContext)["__dataclass_params__"].frozen
    assert vars(ChunkStageArtifacts)["__dataclass_params__"].frozen
    assert vars(AssembledNonlinear)["__dataclass_params__"].frozen
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


def test_overlap_assembly_uses_later_chunk_for_scalar_and_displacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = OverlapValueBackend(chunk_depth=6)
    backend.assembled_scalars = []
    backend.assembled_displacements = []
    registration = _registration(tmp_path, backend)
    registration.overlap_fraction = 0.25
    canonical_assemble = chunked_module.assemble_nonlinear_chunks

    def assemble_permuted(context: object, artifacts: object) -> object:
        completed = tuple(artifacts)  # type: ignore[arg-type]
        return canonical_assemble(context, tuple(reversed(completed)))  # type: ignore[arg-type]

    monkeypatch.setattr(
        chunked_module,
        "assemble_nonlinear_chunks",
        assemble_permuted,
    )

    registration.register()

    scalar = backend.assembled_scalars[-1]
    displacement = backend.assembled_displacements[-1]
    expected = np.zeros(backend.fixed_image.shape, dtype=np.float32)
    expected[6:12] = 1.0
    expected[12:16] = 2.0
    np.testing.assert_array_equal(scalar, expected)
    np.testing.assert_array_equal(
        displacement,
        np.repeat(expected[..., np.newaxis], 3, axis=-1),
    )


def test_direct_assembly_commits_every_advertised_final_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, artifacts, _region, _chunk = _capture_chunk_inputs(
        tmp_path,
        monkeypatch,
    )
    monkeypatch.undo()

    assembled = assemble_nonlinear_chunks(context, tuple(reversed(artifacts)))

    for path in (
        assembled.warp_path,
        assembled.registered_path,
        assembled.nonlinear_metrics_path,
        assembled.chunk_manifest_path,
    ):
        assert path.is_file()
        assert context.store.record_path(path).is_file()


def test_context_rejects_a_noncanonical_chunk_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, _artifacts, _region, _chunk = _capture_chunk_inputs(
        tmp_path,
        monkeypatch,
        overlap_fraction=0.25,
    )

    with pytest.raises(ValueError, match="canonical"):
        replace(context, chunks=context.chunks[:-1])


def test_chunk_stage_rejects_a_chunk_outside_the_canonical_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, _artifacts, region, chunk = _capture_chunk_inputs(
        tmp_path,
        monkeypatch,
    )
    unexpected = replace(chunk, chunk_id=len(context.chunks))

    with pytest.raises(ValueError, match="canonical"):
        run_nonlinear_chunk(context, region, unexpected)


def test_coverage_validation_rejects_a_gap_but_accepts_overlap() -> None:
    _require_complete_coverage(np.asarray([1, 2, 1], dtype=np.uint32))

    with pytest.raises(ValueError, match="gap"):
        _require_complete_coverage(np.asarray([1, 0, 2], dtype=np.uint32))


def test_context_accepts_the_exact_legacy_overlapping_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, _artifacts, _region, _chunk = _capture_chunk_inputs(
        tmp_path,
        monkeypatch,
        overlap_fraction=0.25,
    )

    assert context.chunks == compute_chunk_grid(
        context.fixed_geometry.shape,
        context.chunk_size,
        context.overlap_fraction,
    )
    assert any(
        left.z_end > right.z_start
        for left, right in zip(context.chunks, context.chunks[1:], strict=False)
    )
