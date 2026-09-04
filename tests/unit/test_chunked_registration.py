from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fakes import FakeChunkBackend, FakeImage, fake_image

from cyexm.errors import (
    CacheValidationError,
    ChunkRegistrationError,
    RegistrationStageError,
)
from cyexm.registration import chunked as chunked_module
from cyexm.registration._request_policy import (
    CHUNK_SCALAR_POLICY,
    RAW_RECONSTRUCTED_SCALAR_POLICY,
    WHOLE_SCALAR_POLICY,
)
from cyexm.registration.artifacts import ArtifactStore, CompletionRecord, FileIdentity
from cyexm.registration.chunked import ChunkedImageRegistration, ChunkPaths
from cyexm.registration.whole_image import ImageRegistration, RegistrationPaths

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


def make_whole_registration(
    tmp_path: Path,
    *,
    backend: FakeChunkBackend,
) -> ImageRegistration:
    fixed = tmp_path / "fixed.tif"
    moving = tmp_path / "moving.tif"
    if not fixed.exists():
        fixed.write_bytes(b"fixed-input")
    if not moving.exists():
        moving.write_bytes(b"moving-input")
    return ImageRegistration(
        fixed,
        moving,
        tmp_path / "output",
        2,
        _backend=backend,
        _artifact_store=ArtifactStore(),
    )


def round_paths(tmp_path: Path) -> RegistrationPaths:
    return RegistrationPaths.for_round(tmp_path / "output", 2)


def manifest_path(tmp_path: Path) -> Path:
    return tmp_path / "output" / "round_2_chunks" / "manifest.json"


def scalar_core_path(tmp_path: Path, chunk_id: int) -> Path:
    return (
        tmp_path
        / "output"
        / "round_2_chunks"
        / f"chunk_{chunk_id:04d}_registered_core.npy"
    )


def remove_scalar_core(tmp_path: Path, chunk_id: int) -> None:
    path = scalar_core_path(tmp_path, chunk_id)
    path.unlink()
    ArtifactStore().record_path(path).unlink()


def transitive_chunk_artifacts(
    registration: ChunkedImageRegistration,
    tmp_path: Path,
    chunk_id: int,
) -> tuple[Path, Path, Path, Path, Path]:
    paths = round_paths(tmp_path)
    return (
        registration.chunk_paths(chunk_id).warp,
        manifest_path(tmp_path),
        paths.warp,
        paths.nonlinear_metrics,
        paths.registered,
    )


def rewrite_completion_artifact_identity(artifact: Path) -> None:
    record_path = ArtifactStore().record_path(artifact)
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    payload["artifact"] = FileIdentity.from_path(artifact).to_dict()
    record_path.write_text(
        json.dumps(payload, allow_nan=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


class DistributionSensitiveHistogramBackend(FakeChunkBackend):
    def histogram_match(self, *, source: object, reference: object) -> object:
        super().histogram_match(source=source, reference=reference)
        if not isinstance(source, FakeImage) or not isinstance(reference, FakeImage):
            raise TypeError("source and reference must be FakeImages")
        reference_mean = np.float32(np.mean(reference.data, dtype=np.float64))
        return fake_image(
            data=np.full(source.shape, reference_mean, dtype=np.float32),
            origin=source.origin,
            spacing=source.spacing,
            direction=source.direction,
        )


class FullMaskStageErrorBackend(FakeChunkBackend):
    def threshold_mask(self, image: object) -> object:
        if (
            self.warp_applications
            and self.warp_applications[-1][0].name == "nonlinear_warp_2.nii.gz"
        ):
            raise RegistrationStageError(
                "injected full mask reconstruction stage error"
            )
        return super().threshold_mask(image)


def expected_local_histogram_stitch(moving: np.ndarray) -> np.ndarray:
    expected = np.empty(moving.shape, dtype=np.float32)
    for z_start, z_end in ((0, 8), (8, 16)):
        expected[z_start:z_end] = np.float32(
            np.mean(moving[z_start:z_end], dtype=np.float64)
        )
    return expected


def expected_global_histogram_result(moving: np.ndarray) -> np.ndarray:
    return np.full(
        moving.shape,
        np.float32(np.mean(moving, dtype=np.float64)),
        dtype=np.float32,
    )


def test_chunk_paths_are_round_scoped_and_include_completion_record(
    tmp_path: Path,
) -> None:
    paths = ChunkPaths.for_chunk(tmp_path, 3, 7)

    assert paths.directory == tmp_path / "round_3_chunks"
    assert paths.warp == paths.directory / "chunk_0007_warp.nii.gz"
    assert paths.completion_record == ArtifactStore().record_path(paths.warp)


def test_successful_chunked_registration_removes_kernel_warp_candidates(
    tmp_path: Path,
) -> None:
    backend = FakeChunkBackend()

    make_chunked_registration(tmp_path, backend=backend).register()

    assert backend.temporary_warps
    assert all(not path.exists() for path in backend.temporary_warps)


def test_preprocessed_chunks_estimate_features_and_reconstruct_raw_output(
    tmp_path: Path,
) -> None:
    backend = FakeChunkBackend()
    registration = make_chunked_registration(
        tmp_path,
        backend=backend,
        preprocess_filter_sigma_zyx=(1.0, 2.0, 3.0),
    )

    registration.register()

    raw_fixed, raw_moving = backend.loaded_images[:2]
    assert backend.zero_mask_inputs == [raw_fixed, raw_moving]
    assert backend.linear_kwargs[0]["fixed_image"] is not raw_fixed
    assert all(
        call["fixed_image"] is not raw_fixed for call in backend.nonlinear_kwargs
    )
    assert backend.metric_calls[0][0] is raw_fixed
    assert backend.metric_calls[1][0] is raw_fixed
    assert backend.metric_calls[2][0] is raw_fixed
    raw_nonlinear = backend.metric_calls[2][1]
    assert isinstance(raw_nonlinear, FakeImage)
    np.testing.assert_array_equal(raw_nonlinear.data, raw_moving.data + 10.0)
    np.testing.assert_array_equal(
        backend.clipped_tiff_values[-1],
        np.clip(raw_nonlinear.data, 0, 65535).astype(np.uint16),
    )

    paths = round_paths(tmp_path)
    warp_request = CompletionRecord.read(
        ArtifactStore().record_path(paths.warp)
    ).request
    registered_request = CompletionRecord.read(
        ArtifactStore().record_path(paths.registered)
    ).request
    assert (
        warp_request.parameters["scalar_postprocessing_policy"]
        == RAW_RECONSTRUCTED_SCALAR_POLICY
    )
    assert warp_request.parameters["preprocessing"]["filter_sigma_zyx"] == [
        1.0,
        2.0,
        3.0,
    ]
    assert (
        registered_request.parameters["preprocessing"]
        == (warp_request.parameters["preprocessing"])
    )
    local_request = CompletionRecord.read(
        ArtifactStore().record_path(registration.chunk_paths(0).warp)
    ).request
    assert (
        local_request.parameters["preprocessing"]
        == (warp_request.parameters["preprocessing"])
    )


def test_local_candidate_cleanup_oserror_does_not_fail_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeChunkBackend()
    original_unlink = Path.unlink

    def fail_local_candidate_unlink(
        path: Path,
        missing_ok: bool = False,
    ) -> None:
        if path.name.startswith("temporary-chunk-"):
            raise OSError("injected local candidate cleanup failure")
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_local_candidate_unlink)
    try:
        make_chunked_registration(tmp_path, backend=backend).register()

        assert round_paths(tmp_path).registered.is_file()
    finally:
        for candidate in backend.temporary_warps:
            original_unlink(candidate, missing_ok=True)


def test_full_candidate_cleanup_oserror_does_not_fail_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeChunkBackend()
    original_unlink = Path.unlink

    def fail_full_candidate_unlink(
        path: Path,
        missing_ok: bool = False,
    ) -> None:
        if path.name.startswith(".chunked-full-warp-"):
            raise OSError("injected full candidate cleanup failure")
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_full_candidate_unlink)
    try:
        make_chunked_registration(tmp_path, backend=backend).register()

        assert round_paths(tmp_path).registered.is_file()
    finally:
        for candidate in round_paths(tmp_path).output.glob(
            ".chunked-full-warp-*.nii.gz"
        ):
            original_unlink(candidate, missing_ok=True)


@pytest.mark.parametrize(
    "interruption",
    [
        pytest.param(
            KeyboardInterrupt("injected candidate interruption"),
            id="keyboard-interrupt",
        ),
        pytest.param(SystemExit("injected candidate exit"), id="system-exit"),
    ],
)
def test_full_candidate_interruption_removes_exact_materialized_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interruption: BaseException,
) -> None:
    backend = FakeChunkBackend()
    original_read_warp = backend.read_warp

    def interrupt_after_materialization(path: Path) -> object:
        if path.name.startswith(".chunked-full-warp-"):
            assert path.is_file()
            raise interruption
        return original_read_warp(path)

    monkeypatch.setattr(backend, "read_warp", interrupt_after_materialization)

    with pytest.raises(type(interruption)) as error:
        make_chunked_registration(tmp_path, backend=backend).register()

    assert error.value is interruption
    assert len(backend.written_warp_paths) == 1
    candidate = backend.written_warp_paths[0]
    assert candidate.name.startswith(".chunked-full-warp-")
    assert not candidate.exists()
    assert not list(candidate.parent.glob(".chunked-full-warp-*.nii.gz"))


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


def test_chunked_cc_policy_is_forwarded_to_every_local_registration(
    tmp_path: Path,
) -> None:
    backend = FakeChunkBackend()
    registration = make_chunked_registration(
        tmp_path,
        backend=backend,
        nonlinear_metric="CC",
        nonlinear_radius=2,
    )

    registration.register()

    assert backend.nonlinear_kwargs
    assert all(call["metric"] == "CC" for call in backend.nonlinear_kwargs)
    assert all(call["radius"] == 2 for call in backend.nonlinear_kwargs)
    request = CompletionRecord.read(
        registration.chunk_paths(0).completion_record
    ).request
    parameters = request.parameters["registration_parameters"]
    assert parameters["syn_metric"] == "CC"
    assert parameters["syn_sampling"] == 2


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
    expected_inputs = (
        FileIdentity.from_path(tmp_path / "fixed.tif"),
        FileIdentity.from_path(tmp_path / "moving.tif"),
    )

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
    assert payload["schema_version"] == 2
    assert payload["axis_convention_version"] == "external-zyx-ants-xyz-v1"
    assert [item["warp"] for item in payload["chunks"]] == [
        FileIdentity.from_path(registration.chunk_paths(chunk_id).warp).to_dict()
        for chunk_id in (0, 1)
    ]
    completion_path = ArtifactStore().record_path(path)
    assert completion_path.is_file()
    request = CompletionRecord.read(completion_path).request
    assert request.inputs == expected_inputs
    assert request.parameters["axis_convention_version"] == "external-zyx-ants-xyz-v1"


def test_assembly_is_independent_of_chunk_completion_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_assemble = chunked_module.assemble_nonlinear_chunks
    observed_input_order: list[int] = []

    def assemble_reversed(context: object, artifacts: object) -> object:
        completed = tuple(artifacts)  # type: ignore[arg-type]
        observed_input_order.extend(
            int(item.warp_path.name.split("_", 2)[1]) for item in completed
        )
        return canonical_assemble(context, tuple(reversed(completed)))  # type: ignore[arg-type]

    monkeypatch.setattr(
        chunked_module,
        "assemble_nonlinear_chunks",
        assemble_reversed,
    )

    make_chunked_registration(tmp_path, backend=FakeChunkBackend()).register()

    assert observed_input_order == [0, 1]
    assert round_paths(tmp_path).registered.is_file()


def test_chunked_full_request_records_transitive_chunk_evidence(
    tmp_path: Path,
) -> None:
    registration = make_chunked_registration(
        tmp_path,
        backend=FakeChunkBackend(),
        overlap_fraction=0.25,
    )
    expected_inputs = (
        FileIdentity.from_path(tmp_path / "fixed.tif"),
        FileIdentity.from_path(tmp_path / "moving.tif"),
    )

    registration.register()

    request = CompletionRecord.read(
        ArtifactStore().record_path(round_paths(tmp_path).warp)
    ).request
    assert request.inputs == expected_inputs
    chunks = request.parameters["chunk_grid"]
    assert request.parameters["registration_mode"] == "serial-chunked"
    assert request.parameters["axis_convention_version"] == "external-zyx-ants-xyz-v1"
    assert request.parameters["scalar_postprocessing_policy"] == CHUNK_SCALAR_POLICY
    assert request.parameters["chunk_size_zyx"] == [8, 8, 8]
    assert request.parameters["overlap_fraction"] == 0.25
    assert [item["chunk_id"] for item in chunks] == [0, 1, 2]  # type: ignore[index]
    assert (
        request.parameters["chunk_manifest"]
        == FileIdentity.from_path(manifest_path(tmp_path)).to_dict()
    )
    assert request.parameters["local_warps"] == [
        FileIdentity.from_path(registration.chunk_paths(chunk_id).warp).to_dict()
        for chunk_id in (0, 1, 2)
    ]


def test_every_chunked_stage_request_uses_one_invocation_snapshot(
    tmp_path: Path,
) -> None:
    registration = make_chunked_registration(tmp_path, backend=FakeChunkBackend())
    expected_inputs = (
        FileIdentity.from_path(tmp_path / "fixed.tif"),
        FileIdentity.from_path(tmp_path / "moving.tif"),
    )

    registration.register()

    paths = round_paths(tmp_path)
    artifacts = (
        paths.fixed_reference,
        paths.pre_metrics,
        paths.affine,
        paths.linear_metrics,
        registration.chunk_paths(0).warp,
        registration.chunk_paths(1).warp,
        manifest_path(tmp_path),
        paths.warp,
        paths.nonlinear_metrics,
        paths.registered,
    )
    requests = tuple(
        CompletionRecord.read(ArtifactStore().record_path(path)).request
        for path in artifacts
    )

    assert {request.stage for request in requests} == {
        "fixed-reference",
        "metrics:pre-registration",
        "affine",
        "metrics:linearly-registered",
        "nonlinear-chunk:0",
        "nonlinear-chunk:1",
        "nonlinear-chunk-manifest",
        "nonlinear",
        "metrics:nonlinear-registered",
        "registered-image",
    }
    assert all(request.inputs == expected_inputs for request in requests)


def test_chunked_orchestration_does_not_bypass_cache_guard_helpers() -> None:
    import inspect

    source = inspect.getsource(ChunkedImageRegistration)

    assert ".reuse(" not in source
    assert ".commit(" not in source


def test_whole_image_full_warp_cannot_be_reused_as_chunked(
    tmp_path: Path,
) -> None:
    make_whole_registration(tmp_path, backend=FakeChunkBackend()).register()
    full_request = CompletionRecord.read(
        ArtifactStore().record_path(round_paths(tmp_path).warp)
    ).request
    assert full_request.parameters["registration_mode"] == "whole-image"
    resumed = FakeChunkBackend()
    registration = make_chunked_registration(tmp_path, backend=resumed)

    with pytest.raises(ChunkRegistrationError) as error:
        registration.register()

    assert isinstance(error.value.__cause__, CacheValidationError)
    assert resumed.registered_chunk_ids == []
    assert not registration.chunk_paths(0).warp.exists()


@pytest.mark.parametrize(
    "changed_configuration",
    [
        pytest.param({"chunk_size": (16, 8, 8)}, id="chunk-size"),
        pytest.param({"overlap_fraction": 0.25}, id="overlap"),
    ],
)
def test_changed_chunk_configuration_rejects_full_warp_reuse(
    tmp_path: Path,
    changed_configuration: dict[str, object],
) -> None:
    make_chunked_registration(tmp_path, backend=FakeChunkBackend()).register()
    resumed = FakeChunkBackend()

    with pytest.raises(ChunkRegistrationError) as error:
        make_chunked_registration(
            tmp_path,
            backend=resumed,
            **changed_configuration,  # type: ignore[arg-type]
        ).register()

    assert isinstance(error.value.__cause__, CacheValidationError)
    assert resumed.registered_chunk_ids == []


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_existing_full_warp_requires_valid_local_warp_cache(
    tmp_path: Path,
    damage: str,
) -> None:
    initial = make_chunked_registration(tmp_path, backend=FakeChunkBackend())
    initial.register()
    local_warp = initial.chunk_paths(1).warp
    if damage == "missing":
        local_warp.unlink()
    else:
        local_warp.write_bytes(b"corrupt-local-warp")
    resumed = FakeChunkBackend()

    with pytest.raises(ChunkRegistrationError) as error:
        make_chunked_registration(tmp_path, backend=resumed).register()

    assert isinstance(error.value.__cause__, CacheValidationError)
    assert resumed.registered_chunk_ids == []


@pytest.mark.parametrize("damage", ["missing", "corrupt", "mismatch"])
def test_existing_full_warp_requires_valid_canonical_manifest(
    tmp_path: Path,
    damage: str,
) -> None:
    make_chunked_registration(tmp_path, backend=FakeChunkBackend()).register()
    manifest = manifest_path(tmp_path)
    if damage == "missing":
        manifest.unlink()
    elif damage == "corrupt":
        manifest.write_text("not-json\n", encoding="utf-8")
    else:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["round_number"] = 999
        manifest.write_text(
            json.dumps(
                payload,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        rewrite_completion_artifact_identity(manifest)
    resumed = FakeChunkBackend()

    with pytest.raises(CacheValidationError):
        make_chunked_registration(tmp_path, backend=resumed).register()

    assert resumed.registered_chunk_ids == []


def test_normal_rerun_reuses_local_full_and_downstream_artifact_identities(
    tmp_path: Path,
) -> None:
    initial = make_chunked_registration(tmp_path, backend=FakeChunkBackend())
    initial.register()
    paths = round_paths(tmp_path)
    artifacts = (
        initial.chunk_paths(0).warp,
        initial.chunk_paths(1).warp,
        manifest_path(tmp_path),
        paths.warp,
        paths.nonlinear_metrics,
        paths.registered,
    )
    identities_before = tuple(FileIdentity.from_path(path) for path in artifacts)
    records_before = tuple(
        ArtifactStore().record_path(path).read_bytes() for path in artifacts
    )
    resumed = FakeChunkBackend()

    make_chunked_registration(tmp_path, backend=resumed).register()

    assert resumed.registered_chunk_ids == []
    assert (
        tuple(FileIdentity.from_path(path) for path in artifacts) == identities_before
    )
    assert (
        tuple(ArtifactStore().record_path(path).read_bytes() for path in artifacts)
        == records_before
    )
    assert len(resumed.written_warp_paths) == 1
    assert not resumed.written_warp_paths[0].exists()


def test_serial_chunk_scalar_is_local_core_stitch_fresh_and_on_full_warp_reuse(
    tmp_path: Path,
) -> None:
    fresh = DistributionSensitiveHistogramBackend()
    registration = make_chunked_registration(tmp_path, backend=fresh)

    registration.register()

    paths = round_paths(tmp_path)
    expected_local = expected_local_histogram_stitch(fresh.moving_image.data)
    hypothetical_global = expected_global_histogram_result(fresh.moving_image.data)
    fresh_metric_scalar = fresh.metric_calls[-1][1]
    fresh_final_scalar = fresh.tiff_images[-1]
    assert isinstance(fresh_metric_scalar, FakeImage)
    assert isinstance(fresh_final_scalar, FakeImage)
    np.testing.assert_array_equal(fresh_metric_scalar.data, expected_local)
    np.testing.assert_array_equal(fresh_final_scalar.data, expected_local)
    assert not np.array_equal(fresh_metric_scalar.data, hypothetical_global)
    assert (
        float(np.max(np.abs(fresh_metric_scalar.data - hypothetical_global))) == 256.0
    )
    assert len(fresh.histogram_match_calls) == 2
    assert all(
        isinstance(reference, FakeImage) and reference.shape == (8, 8, 8)
        for _, reference in fresh.histogram_match_calls
    )
    assert not any(
        path.name == "nonlinear_warp_2.nii.gz" and interpolation == "linear"
        for path, interpolation in fresh.warp_applications
    )
    assert (
        paths.warp,
        "nearestNeighbor",
    ) in fresh.warp_applications

    preserved_artifacts = (
        registration.chunk_paths(0).warp,
        registration.chunk_paths(1).warp,
        paths.warp,
    )
    preserved_identities = tuple(
        FileIdentity.from_path(path) for path in preserved_artifacts
    )
    nonlinear_metrics_bytes = paths.nonlinear_metrics.read_bytes()
    registered_bytes = paths.registered.read_bytes()
    store = ArtifactStore()
    store.mark_stale(paths.nonlinear_metrics)
    store.mark_stale(paths.registered)
    resumed = DistributionSensitiveHistogramBackend()

    make_chunked_registration(tmp_path, backend=resumed).register()

    resumed_metric_scalar = resumed.metric_calls[-1][1]
    resumed_final_scalar = resumed.tiff_images[-1]
    assert isinstance(resumed_metric_scalar, FakeImage)
    assert isinstance(resumed_final_scalar, FakeImage)
    np.testing.assert_array_equal(resumed_metric_scalar.data, expected_local)
    np.testing.assert_array_equal(resumed_final_scalar.data, expected_local)
    np.testing.assert_array_equal(
        resumed_metric_scalar.data,
        fresh_metric_scalar.data,
    )
    assert resumed.histogram_match_calls == []
    assert resumed.registered_chunk_ids == []
    assert resumed.nonlinear_calls == 0
    assert tuple(FileIdentity.from_path(path) for path in preserved_artifacts) == (
        preserved_identities
    )
    assert paths.nonlinear_metrics.read_bytes() == nonlinear_metrics_bytes
    assert paths.registered.read_bytes() == registered_bytes
    assert not any(
        path.name == "nonlinear_warp_2.nii.gz" and interpolation == "linear"
        for path, interpolation in resumed.warp_applications
    )
    assert (paths.warp, "nearestNeighbor") in resumed.warp_applications


def test_scalar_policy_is_recorded_on_local_full_metrics_and_final_requests(
    tmp_path: Path,
) -> None:
    registration = make_chunked_registration(tmp_path, backend=FakeChunkBackend())

    registration.register()

    store = ArtifactStore()
    paths = round_paths(tmp_path)
    artifacts = (
        registration.chunk_paths(0).warp,
        registration.chunk_paths(1).warp,
        paths.warp,
        paths.nonlinear_metrics,
        paths.registered,
    )
    requests = tuple(
        CompletionRecord.read(store.record_path(path)).request for path in artifacts
    )
    assert all(
        request.parameters["scalar_postprocessing_policy"] == CHUNK_SCALAR_POLICY
        for request in requests
    )


def test_legacy_chunk_scalar_policy_record_fails_closed(tmp_path: Path) -> None:
    make_chunked_registration(tmp_path, backend=FakeChunkBackend()).register()
    store = ArtifactStore()
    full_record = store.record_path(round_paths(tmp_path).warp)
    payload = json.loads(full_record.read_text(encoding="utf-8"))
    parameters = payload["request"]["parameters"]
    parameters.pop("scalar_postprocessing_policy", None)
    parameters["scalar_postprocessing"] = "histogram-match-to-moving-linear"
    full_record.write_text(
        json.dumps(payload, allow_nan=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    resumed = FakeChunkBackend()

    with pytest.raises(CacheValidationError, match="request"):
        make_chunked_registration(tmp_path, backend=resumed).register()

    assert resumed.registered_chunk_ids == []


def test_whole_image_scalar_remains_global_and_canonical_on_warp_reuse(
    tmp_path: Path,
) -> None:
    fresh = DistributionSensitiveHistogramBackend()

    make_whole_registration(tmp_path, backend=fresh).register()

    paths = round_paths(tmp_path)
    expected_global = expected_global_histogram_result(fresh.moving_image.data)
    fresh_metric_scalar = fresh.metric_calls[-1][1]
    fresh_final_scalar = fresh.tiff_images[-1]
    assert isinstance(fresh_metric_scalar, FakeImage)
    assert isinstance(fresh_final_scalar, FakeImage)
    np.testing.assert_array_equal(fresh_metric_scalar.data, expected_global)
    np.testing.assert_array_equal(fresh_final_scalar.data, expected_global)
    assert len(fresh.histogram_match_calls) == 1
    assert isinstance(fresh.histogram_match_calls[0][1], FakeImage)
    assert fresh.histogram_match_calls[0][1].shape == (16, 8, 8)  # type: ignore[union-attr]

    store = ArtifactStore()
    requests = tuple(
        CompletionRecord.read(store.record_path(path)).request
        for path in (paths.warp, paths.nonlinear_metrics, paths.registered)
    )
    assert all(
        request.parameters["scalar_postprocessing_policy"] == WHOLE_SCALAR_POLICY
        for request in requests
    )
    store.mark_stale(paths.nonlinear_metrics)
    store.mark_stale(paths.registered)
    resumed = DistributionSensitiveHistogramBackend()

    make_whole_registration(tmp_path, backend=resumed).register()

    resumed_metric_scalar = resumed.metric_calls[-1][1]
    resumed_final_scalar = resumed.tiff_images[-1]
    assert isinstance(resumed_metric_scalar, FakeImage)
    assert isinstance(resumed_final_scalar, FakeImage)
    np.testing.assert_array_equal(resumed_metric_scalar.data, expected_global)
    np.testing.assert_array_equal(resumed_final_scalar.data, expected_global)
    assert len(resumed.histogram_match_calls) == 1
    assert resumed.nonlinear_calls == 0


def test_prepared_candidate_is_removed_when_full_request_reuse_raises(
    tmp_path: Path,
) -> None:
    make_chunked_registration(tmp_path, backend=FakeChunkBackend()).register()
    full_record = ArtifactStore().record_path(round_paths(tmp_path).warp)
    payload = json.loads(full_record.read_text(encoding="utf-8"))
    payload["request"]["parameters"]["registration_mode"] = "tampered"
    full_record.write_text(
        json.dumps(payload, allow_nan=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    record_before = full_record.read_bytes()
    resumed = FakeChunkBackend()

    with pytest.raises(CacheValidationError, match="request"):
        make_chunked_registration(tmp_path, backend=resumed).register()

    assert len(resumed.written_warp_paths) == 1
    assert not resumed.written_warp_paths[0].exists()
    assert full_record.read_bytes() == record_before


def test_local_request_captures_sources_geometries_threads_affine_and_grid(
    tmp_path: Path,
) -> None:
    registration = make_chunked_registration(
        tmp_path,
        backend=FakeChunkBackend(),
        overlap_fraction=0.25,
    )
    expected_inputs = (
        FileIdentity.from_path(tmp_path / "fixed.tif"),
        FileIdentity.from_path(tmp_path / "moving.tif"),
    )

    registration.register()

    request = CompletionRecord.read(
        registration.chunk_paths(0).completion_record
    ).request
    assert request.inputs == expected_inputs
    assert request.geometry.shape == (9, 8, 8)
    assert request.parameters["moving_geometry"] == request.geometry.to_dict()
    assert request.native_threads and set(request.native_threads.values()) == {1}
    assert request.upstream == FileIdentity.from_path(round_paths(tmp_path).affine)
    assert request.method == "SyNOnly"
    assert request.accuracy == "high"
    assert request.parameters["axis_convention_version"] == "external-zyx-ants-xyz-v1"
    assert request.parameters["chunk"]["chunk_id"] == 0  # type: ignore[index]
    assert request.parameters["chunk_size_zyx"] == [8, 8, 8]
    assert request.parameters["image_shape_zyx"] == [16, 8, 8]
    assert request.parameters["overlap_fraction"] == 0.25
    assert request.parameters["scalar_postprocessing_policy"] == CHUNK_SCALAR_POLICY
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


def test_failed_fresh_local_postcommit_record_is_stale_and_recomputed(
    tmp_path: Path,
) -> None:
    backend = FakeChunkBackend(invalid_scalar_chunk=1)
    registration = make_chunked_registration(tmp_path, backend=backend)

    with pytest.raises(ChunkRegistrationError, match="scalar"):
        registration.register()

    local = registration.chunk_paths(1)
    store = ArtifactStore()
    assert not local.completion_record.exists()
    assert store.stale_path(local.warp).is_file()
    assert not manifest_path(tmp_path).exists()
    assert not round_paths(tmp_path).warp.exists()

    resumed = FakeChunkBackend()
    make_chunked_registration(tmp_path, backend=resumed).register()

    assert resumed.registered_chunk_ids == [1]
    assert local.completion_record.is_file()
    assert not store.stale_path(local.warp).exists()
    assert round_paths(tmp_path).warp.is_file()


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(FakeChunkBackend(fail_histogram_match=True), id="histogram"),
        pytest.param(FakeChunkBackend(fail_threshold_mask=True), id="mask"),
    ],
)
def test_fresh_local_reconstruction_failure_marks_record_stale(
    tmp_path: Path,
    backend: FakeChunkBackend,
) -> None:
    registration = make_chunked_registration(tmp_path, backend=backend)

    with pytest.raises(ChunkRegistrationError):
        registration.register()

    local = registration.chunk_paths(0)
    store = ArtifactStore()
    assert local.warp.is_file()
    assert not local.completion_record.exists()
    assert store.stale_path(local.warp).is_file()


def test_cached_local_reconstruction_failure_preserves_valid_record(
    tmp_path: Path,
) -> None:
    initial = make_chunked_registration(tmp_path, backend=FakeChunkBackend())
    initial.register()
    local = initial.chunk_paths(0)
    record_before = local.completion_record.read_bytes()
    remove_scalar_core(tmp_path, 0)
    resumed = FakeChunkBackend(fail_histogram_match=True)

    with pytest.raises(ChunkRegistrationError):
        make_chunked_registration(tmp_path, backend=resumed).register()

    assert local.completion_record.read_bytes() == record_before
    assert not ArtifactStore().stale_path(local.warp).exists()


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param("scalar-shape", id="scalar-shape"),
        pytest.param("mask-shape", id="mask-shape"),
        pytest.param("warp-float32", id="warp-float32"),
    ],
)
def test_cached_deterministic_chunk_output_failure_stales_chain_and_recovers(
    tmp_path: Path,
    failure: str,
) -> None:
    initial = make_chunked_registration(tmp_path, backend=FakeChunkBackend())
    initial.register()
    artifacts = transitive_chunk_artifacts(initial, tmp_path, 1)
    if failure != "warp-float32":
        remove_scalar_core(tmp_path, 1)
    failing_backend = FakeChunkBackend(
        invalid_scalar_chunk=1 if failure == "scalar-shape" else None,
        invalid_mask_chunk=1 if failure == "mask-shape" else None,
        overflow_warp_chunk=1 if failure == "warp-float32" else None,
    )

    with pytest.raises(ChunkRegistrationError) as error:
        make_chunked_registration(tmp_path, backend=failing_backend).register()

    assert isinstance(error.value.__cause__, ValueError)
    assert failing_backend.registered_chunk_ids == []
    store = ArtifactStore()
    for artifact in artifacts:
        assert not store.record_path(artifact).exists()
        assert store.stale_path(artifact).is_file()
    resumed = FakeChunkBackend()

    make_chunked_registration(tmp_path, backend=resumed).register()

    assert resumed.registered_chunk_ids == [1]
    for artifact in artifacts:
        assert store.record_path(artifact).is_file()
        assert not store.stale_path(artifact).exists()


def test_cached_backend_value_error_preserves_transitive_records(
    tmp_path: Path,
) -> None:
    initial = make_chunked_registration(tmp_path, backend=FakeChunkBackend())
    initial.register()
    artifacts = transitive_chunk_artifacts(initial, tmp_path, 1)
    remove_scalar_core(tmp_path, 1)
    store = ArtifactStore()
    records_before = tuple(
        store.record_path(artifact).read_bytes() for artifact in artifacts
    )
    failing_backend = FakeChunkBackend(backend_value_error_chunk=1)

    with pytest.raises(ChunkRegistrationError) as error:
        make_chunked_registration(tmp_path, backend=failing_backend).register()

    assert isinstance(error.value.__cause__, ValueError)
    assert str(error.value.__cause__) == "injected backend chunk value error"
    assert failing_backend.registered_chunk_ids == []
    assert (
        tuple(store.record_path(artifact).read_bytes() for artifact in artifacts)
        == records_before
    )
    for artifact in artifacts:
        assert not store.stale_path(artifact).exists()


def test_stale_full_entry_allows_local_and_manifest_recovery(
    tmp_path: Path,
) -> None:
    initial = make_chunked_registration(tmp_path, backend=FakeChunkBackend())
    initial.register()
    failed = make_chunked_registration(
        tmp_path,
        backend=FakeChunkBackend(invalid_scalar_chunk=1),
        force_override=True,
    )

    with pytest.raises(ChunkRegistrationError):
        failed.register()

    store = ArtifactStore()
    full = round_paths(tmp_path).warp
    local = failed.chunk_paths(1).warp
    manifest = manifest_path(tmp_path)
    for artifact in (full, local, manifest):
        assert not store.record_path(artifact).exists()
        assert store.stale_path(artifact).is_file()
    resumed = FakeChunkBackend()

    make_chunked_registration(tmp_path, backend=resumed).register()

    assert resumed.registered_chunk_ids == [1]
    for artifact in (full, local, manifest):
        assert store.record_path(artifact).is_file()
        assert not store.stale_path(artifact).exists()


@pytest.mark.parametrize(
    "overflow",
    [
        pytest.param("scalar", id="scalar-core"),
        pytest.param("warp", id="warp-core"),
    ],
)
def test_float64_core_overflow_is_rejected_after_float32_cast(
    tmp_path: Path,
    overflow: str,
) -> None:
    backend = FakeChunkBackend(
        overflow_scalar_chunk=1 if overflow == "scalar" else None,
        overflow_warp_chunk=1 if overflow == "warp" else None,
    )
    registration = make_chunked_registration(tmp_path, backend=backend)

    with pytest.raises(ChunkRegistrationError, match="float32"):
        registration.register()

    local = registration.chunk_paths(1)
    assert not local.completion_record.exists()
    assert ArtifactStore().stale_path(local.warp).is_file()
    assert not round_paths(tmp_path).warp.exists()


def test_full_candidate_reread_rejects_non_float32_dtype(tmp_path: Path) -> None:
    backend = FakeChunkBackend(full_warp_float64=True)

    with pytest.raises(RegistrationStageError) as error:
        make_chunked_registration(tmp_path, backend=backend).register()

    assert isinstance(error.value.__cause__, ValueError)
    assert "float32" in str(error.value.__cause__)
    assert backend.written_warp_paths
    assert not backend.written_warp_paths[0].exists()
    assert not round_paths(tmp_path).warp.exists()


def test_local_backend_verbose_value_matches_recorded_parameters(
    tmp_path: Path,
) -> None:
    backend = FakeChunkBackend()
    registration = make_chunked_registration(tmp_path, backend=backend)

    registration.register()

    request = CompletionRecord.read(
        registration.chunk_paths(0).completion_record
    ).request
    recorded = request.parameters["registration_parameters"]["verbose"]  # type: ignore[index]
    assert backend.nonlinear_kwargs[0]["verbose"] == recorded


def test_scalar_conversion_failure_cannot_leak_full_candidate(tmp_path: Path) -> None:
    backend = FakeChunkBackend(fail_scalar_from_array=True)

    with pytest.raises(RegistrationStageError):
        make_chunked_registration(tmp_path, backend=backend).register()

    output = round_paths(tmp_path).output
    assert not list(output.glob(".chunked-full-warp-*.nii.gz"))
    assert not round_paths(tmp_path).warp.exists()


def test_fresh_full_stage_error_marks_committed_warp_record_stale(
    tmp_path: Path,
) -> None:
    backend = FullMaskStageErrorBackend()

    with pytest.raises(RegistrationStageError, match="full mask reconstruction"):
        make_chunked_registration(tmp_path, backend=backend).register()

    warp = round_paths(tmp_path).warp
    store = ArtifactStore()
    assert warp.is_file()
    assert not store.record_path(warp).exists()
    assert store.stale_path(warp).is_file()
    assert not list(warp.parent.glob(".chunked-full-warp-*.nii.gz"))


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
    first_image = first_backend.read_warp(round_paths(first_root).warp)

    second_backend = FakeChunkBackend()
    make_chunked_registration(second_root, backend=second_backend).register()
    second_candidate = second_backend.written_warp_paths[0]

    assert first_candidate != second_candidate
    assert first_candidate.parent == first_root / "output"
    assert not first_candidate.exists()
    assert not second_candidate.exists()
    assert first_image.data.dtype == np.float32  # type: ignore[union-attr]
    assert first_image.data.shape == (16, 8, 8, 3)  # type: ignore[union-attr]


def test_parallel_chunked_class_is_not_exported_in_a_fresh_process() -> None:
    code = """
import sys
import cyexm.registration as registration

assert not hasattr(registration, "ParallelChunkedImageRegistration")
assert registration.__all__ == [
    "ChunkedImageRegistration",
    "EstimatedXYRotationPolicy",
    "ExplicitXYRotationPolicy",
    "ImageRegistration",
    "InitialAlignmentPolicy",
    "register_round",
    "transform_channel",
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
