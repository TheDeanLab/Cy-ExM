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
    _chunk_support_evidence,
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


def test_chunk_support_rejects_a_nonzero_coarsest_axis_sliver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, _artifacts, region, _chunk = _capture_chunk_inputs(
        tmp_path,
        monkeypatch,
    )
    thin = np.zeros((8, 8, 8), dtype=np.float32)
    thin[:, :, :2] = 1.0
    thin_region = replace(
        region,
        fixed_mask=fake_image(data=thin),
        moving_mask=fake_image(data=thin),
    )

    evidence = _chunk_support_evidence(
        np,
        context.backend,
        thin_region,
        context.nonlinear_accuracy,
    )

    assert evidence is not None
    assert evidence.insufficient is True
    assert evidence.fine_common_voxels == 128
    assert evidence.fine_total_voxels == 512
    assert evidence.coarsest_common_voxels == 4
    assert evidence.coarsest_shrink_factor == 4
    assert evidence.coarsest_span_zyx == (2, 2, 1)


def test_chunk_support_accepts_the_minimum_coarsest_3d_lattice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, _artifacts, region, _chunk = _capture_chunk_inputs(
        tmp_path,
        monkeypatch,
    )
    full = np.ones((8, 8, 8), dtype=np.float32)
    full_region = replace(
        region,
        fixed_mask=fake_image(data=full),
        moving_mask=fake_image(data=full),
    )

    evidence = _chunk_support_evidence(
        np,
        context.backend,
        full_region,
        context.nonlinear_accuracy,
    )

    assert evidence is not None
    assert evidence.insufficient is False
    assert evidence.coarsest_common_voxels == 8
    assert evidence.coarsest_span_zyx == (2, 2, 2)


def test_chunk_support_accepts_a_nearly_full_minimum_3d_lattice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, _artifacts, region, _chunk = _capture_chunk_inputs(
        tmp_path,
        monkeypatch,
    )
    nearly_full = np.ones((8, 8, 8), dtype=np.float32)
    nearly_full[0, 0, 0] = 0.0
    nearly_full_region = replace(
        region,
        fixed_mask=fake_image(data=nearly_full),
        moving_mask=fake_image(data=nearly_full),
    )

    evidence = _chunk_support_evidence(
        np,
        context.backend,
        nearly_full_region,
        context.nonlinear_accuracy,
    )

    assert evidence is not None
    assert evidence.insufficient is False
    assert evidence.coarsest_common_voxels == 7
    assert evidence.coarsest_span_zyx == (2, 2, 2)


def test_chunk_support_is_unassessed_when_either_mask_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, _artifacts, region, _chunk = _capture_chunk_inputs(
        tmp_path,
        monkeypatch,
    )

    assert (
        _chunk_support_evidence(
            np,
            context.backend,
            replace(region, moving_mask=None),
            context.nonlinear_accuracy,
        )
        is None
    )


@pytest.mark.parametrize(
    "mask, message",
    [
        pytest.param(
            np.ones((7, 8, 8), dtype=np.float32),
            "shape",
            id="wrong-shape",
        ),
        pytest.param(
            np.full((8, 8, 8), np.nan, dtype=np.float32),
            "finite",
            id="nonfinite",
        ),
    ],
)
def test_chunk_support_rejects_malformed_masks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mask: np.ndarray,
    message: str,
) -> None:
    context, _artifacts, region, _chunk = _capture_chunk_inputs(
        tmp_path,
        monkeypatch,
    )
    malformed = replace(region, moving_mask=fake_image(data=mask))

    with pytest.raises(ValueError, match=message):
        _chunk_support_evidence(
            np,
            context.backend,
            malformed,
            context.nonlinear_accuracy,
        )


def test_nonzero_insufficient_support_commits_identity_without_calling_ants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, _artifacts, region, chunk = _capture_chunk_inputs(
        tmp_path,
        monkeypatch,
    )
    thin = np.zeros(chunk.extended_shape, dtype=np.float32)
    thin[:, :, :2] = 1.0
    thin_region = replace(
        region,
        fixed_mask=fake_image(data=thin),
        moving_mask=fake_image(data=thin),
    )
    isolated = replace(
        context,
        force=True,
        output_directory=tmp_path / "insufficient-support",
    )
    backend = context.backend
    calls_before = backend.nonlinear_calls  # type: ignore[attr-defined]
    identities_before = len(backend.identity_chunk_ids)  # type: ignore[attr-defined]

    artifact = run_nonlinear_chunk(isolated, thin_region, chunk)

    assert backend.nonlinear_calls == calls_before  # type: ignore[attr-defined]
    assert backend.identity_chunk_ids[identities_before:] == [  # type: ignore[attr-defined]
        chunk.chunk_id
    ]
    request = CompletionRecord.read(
        isolated.store.record_path(artifact.warp_path)
    ).request
    assert request.parameters["insufficient_support"] == {
        "coarsest_common_voxels": 4,
        "coarsest_shrink_factor": 4,
        "coarsest_span_zyx": [2, 2, 1],
        "decision": "identity-insufficient-support",
        "fine_common_voxels": 128,
        "fine_total_voxels": 512,
        "policy": "common-mask-coarsest-2x2x2-v1",
    }


def test_sufficient_support_preserves_the_historical_request_and_calls_ants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, _artifacts, region, chunk = _capture_chunk_inputs(
        tmp_path,
        monkeypatch,
    )
    full = np.ones(chunk.extended_shape, dtype=np.float32)
    full_region = replace(
        region,
        fixed_mask=fake_image(data=full),
        moving_mask=fake_image(data=full),
    )
    isolated = replace(
        context,
        force=True,
        output_directory=tmp_path / "sufficient-support",
    )
    backend = context.backend
    calls_before = backend.nonlinear_calls  # type: ignore[attr-defined]

    artifact = run_nonlinear_chunk(isolated, full_region, chunk)

    assert backend.nonlinear_calls == calls_before + 1  # type: ignore[attr-defined]
    request = CompletionRecord.read(
        isolated.store.record_path(artifact.warp_path)
    ).request
    assert "insufficient_support" not in request.parameters


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(
            RuntimeError("Registration failed with error code 1"),
            id="generic-ants-code-one",
        ),
        pytest.param(RuntimeError("different runtime failure"), id="other-runtime"),
        pytest.param(ValueError("invalid backend geometry"), id="other-type"),
    ],
)
def test_sufficient_support_never_turns_backend_failures_into_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    context, _artifacts, region, chunk = _capture_chunk_inputs(
        tmp_path,
        monkeypatch,
    )
    full = np.ones(chunk.extended_shape, dtype=np.float32)
    full_region = replace(
        region,
        fixed_mask=fake_image(data=full),
        moving_mask=fake_image(data=full),
    )
    isolated = replace(
        context,
        force=True,
        output_directory=tmp_path / f"propagate-{type(error).__name__}",
    )

    def fail_nonlinear(**_kwargs: object) -> tuple[object, Path]:
        raise error

    monkeypatch.setattr(context.backend, "nonlinear", fail_nonlinear)

    with pytest.raises(type(error), match=str(error)) as raised:
        run_nonlinear_chunk(isolated, full_region, chunk)

    assert raised.value is error


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
