from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from cyexm.chunking import compute_chunk_grid
from cyexm.config.models import (
    ChannelSpec,
    DatasetManifest,
    ExecutionPaths,
    ExecutionProfile,
    NativeThreadSpec,
    NonlinearRegistrationStageSpec,
    ParallelismSpec,
    ReferenceChannelSpec,
    ReferenceSpec,
    RegistrationSpec,
    RegistrationStageSpec,
    RoundSpec,
)
from cyexm.execution.sequential import SequentialExecutor
from cyexm.io.tiff import read_tiff_geometry
from cyexm.metrics import read_metrics
from cyexm.registration.artifacts import ArtifactStore, CompletionRecord, FileIdentity
from cyexm.registration.backend import AntsRegistrationBackend
from cyexm.registration.prepared import (
    PreparedManifest,
    PreparedRoundPaths,
    read_prepared_region,
)
from cyexm.runtime import NativeThreadConfig
from cyexm.spatial import ImageGeometry, geometry_compatible
from cyexm.workflow._round_provenance import RegistrationArtifactPaths
from cyexm.workflow.api import run_workflow
from cyexm.workflow.fine_models import (
    ChannelTransformTask,
    FineTaskResult,
    FineWorkflowTask,
    NonlinearChunkTask,
)
from cyexm.workflow.fine_task import run_fine_task
from cyexm.workflow.models import WorkflowPlan
from cyexm.workflow.planning import build_workflow_plan
from cyexm.workflow.task import run_workflow_task

pytestmark = pytest.mark.integration

_SEED = "20260719"
_PIXEL_SIZE = (0.2, 0.2, 0.5)
_FIXED_SHAPE = (16, 32, 32)
_REGISTRATION_SHAPE = (14, 29, 28)
_SIGNAL_SHAPE = (15, 30, 29)
_LABEL_SHAPE = (13, 31, 30)
_CHUNK_SIZE = (10, 24, 32)
_OVERLAP = 0.25
_ROUND = 2


@dataclass(frozen=True, slots=True)
class _ScientificCase:
    dataset_root: Path
    manifest: DatasetManifest


def _blob(
    np: Any,
    shape: tuple[int, int, int],
    *,
    center_offset: tuple[float, float, float],
) -> object:
    z, y, x = np.indices(shape, dtype=np.float32)
    center = tuple(
        (size - 1.0) / 2.0 + offset
        for size, offset in zip(shape, center_offset, strict=True)
    )
    primary = np.exp(
        -(
            ((z - center[0]) / 3.1) ** 2
            + ((y - center[1]) / 5.8) ** 2
            + ((x - center[2]) / 5.3) ** 2
        )
    )
    secondary = np.exp(
        -(
            ((z - (center[0] - 2.0)) / 2.0) ** 2
            + ((y - (center[1] + 5.0)) / 3.0) ** 2
            + ((x - (center[2] - 4.0)) / 2.5) ** 2
        )
    )
    values = (primary + np.float32(0.55) * secondary) * np.float32(55_000.0)
    values[values < np.float32(1.0)] = np.float32(0.0)
    return values.astype(np.float32)


@pytest.fixture
def fine_scientific_case(
    tmp_path: Path,
    native_threads: NativeThreadConfig,
) -> _ScientificCase:
    del native_threads
    import numpy as np
    import tifffile

    assert os.environ["ANTS_RANDOM_SEED"] == _SEED
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    fixed = _blob(np, _FIXED_SHAPE, center_offset=(0.0, 0.0, 0.0))
    registration = _blob(
        np,
        _REGISTRATION_SHAPE,
        center_offset=(0.8, -1.2, 1.0),
    )
    signal = _blob(np, _SIGNAL_SHAPE, center_offset=(-0.5, 1.0, -1.4))
    label_source = _blob(np, _LABEL_SHAPE, center_offset=(0.0, -0.8, 0.6))
    label = np.where(label_source > np.float32(8_000.0), 41_000, 0).astype(np.uint16)
    sources = {
        "round-1-reference.tif": fixed,
        "round-2-registration.tif": registration,
        "round-2-signal.tif": signal,
        "round-2-label.tif": label,
    }
    for name, values in sources.items():
        tifffile.imwrite(
            dataset_root / name,
            values,
            metadata={"axes": "ZYX"},
            photometric="minisblack",
        )

    manifest = DatasetManifest(
        schema_version=1,
        dataset_id="fine-real-ants-parity",
        voxel_spacing_um=_PIXEL_SIZE,
        reference=ReferenceSpec(
            round=1,
            registration_channel=PurePosixPath("round-1-reference.tif"),
            channels=(ReferenceChannelSpec(PurePosixPath("round-1-reference.tif")),),
        ),
        registration=RegistrationSpec(
            crop=False,
            linear=RegistrationStageSpec(type="TRSAA", accuracy="dry run"),
            nonlinear=NonlinearRegistrationStageSpec(
                type="SyNOnly",
                accuracy="dry run",
                metric="mattes",
                radius=None,
            ),
        ),
        rounds=(
            RoundSpec(
                round=_ROUND,
                strategy="serial_chunked",
                registration_channel=PurePosixPath("round-2-registration.tif"),
                channels=(
                    ChannelSpec(
                        PurePosixPath("round-2-registration.tif"),
                        "linear",
                    ),
                    ChannelSpec(PurePosixPath("round-2-signal.tif"), "linear"),
                    ChannelSpec(
                        PurePosixPath("round-2-label.tif"),
                        "nearestNeighbor",
                    ),
                ),
                chunk_size_zyx=_CHUNK_SIZE,
                overlap_fraction=_OVERLAP,
            ),
        ),
    )
    return _ScientificCase(dataset_root=dataset_root, manifest=manifest)


def _profile(
    root: Path,
    case: _ScientificCase,
    *,
    name: str,
    mode: str,
) -> ExecutionProfile:
    return ExecutionProfile(
        schema_version=1,
        executor="sequential",
        paths=ExecutionPaths(
            dataset_root=case.dataset_root,
            output_root=(root / name).resolve(),
            worker_local_directory=None,
        ),
        native_threads=NativeThreadSpec(itk=1, blas=1),
        local_dask=None,
        slurm=None,
        source_path=(root / f"{name}.yaml").resolve(),
        parallelism=ParallelismSpec(mode=mode),  # type: ignore[arg-type]
    )


def _registration_paths(plan: WorkflowPlan) -> RegistrationArtifactPaths:
    task = plan.tasks[0]
    return RegistrationArtifactPaths.for_round(
        task.registration_directory,
        task.round,
    )


def _assert_completion(path: Path) -> CompletionRecord:
    store = ArtifactStore()
    record_path = store.record_path(path)
    assert path.is_file()
    assert record_path.is_file()
    record = CompletionRecord.read(record_path)
    assert record.artifact == FileIdentity.from_path(path)
    assert record.to_json() == record_path.read_text(encoding="utf-8")
    assert set(record.request.native_threads.values()) == {1}
    return record


def _artifact_identities(
    paths: tuple[Path, ...],
) -> tuple[tuple[FileIdentity, FileIdentity], ...]:
    store = ArtifactStore()
    return tuple(
        (
            FileIdentity.from_path(path),
            FileIdentity.from_path(store.record_path(path)),
        )
        for path in paths
    )


def _assert_warp_equal(
    backend: AntsRegistrationBackend,
    first: Path,
    second: Path,
    *,
    expected_geometry: ImageGeometry,
) -> None:
    import numpy as np

    first_warp = backend.read_warp(first)
    second_warp = backend.read_warp(second)
    assert geometry_compatible(backend.geometry(first_warp), expected_geometry)
    assert geometry_compatible(backend.geometry(second_warp), expected_geometry)
    first_values = np.asarray(backend.to_array(first_warp))
    second_values = np.asarray(backend.to_array(second_warp))
    assert first_values.shape == second_values.shape == (*expected_geometry.shape, 3)
    assert np.isfinite(first_values).all()
    assert np.isfinite(second_values).all()
    # Established migration tolerance covers deterministic ANTSpyX/NIfTI float32 IO.
    np.testing.assert_allclose(first_values, second_values, rtol=1e-4, atol=1e-5)


def _assert_tiff_equal(
    first: Path,
    second: Path,
    *,
    expected_geometry: ImageGeometry,
) -> None:
    import numpy as np
    import tifffile

    first_values = tifffile.imread(first)
    second_values = tifffile.imread(second)
    assert first_values.shape == second_values.shape == expected_geometry.shape
    assert first_values.dtype == second_values.dtype == np.dtype(np.uint16)
    assert read_tiff_geometry(first, first_values.shape) == expected_geometry
    assert read_tiff_geometry(second, second_values.shape) == expected_geometry
    # Established migration tolerance covers independent deterministic ANTs runs.
    np.testing.assert_allclose(first_values, second_values, rtol=1e-4, atol=1e-5)


def test_seeded_real_ants_coarse_and_fine_outputs_match_on_fixed_grid(
    tmp_path: Path,
    fine_scientific_case: _ScientificCase,
    native_threads: NativeThreadConfig,
) -> None:
    import ants
    import numpy as np
    import tifffile

    chunks = compute_chunk_grid(_FIXED_SHAPE, _CHUNK_SIZE, _OVERLAP)
    assert len(chunks) == 4
    source_shapes = tuple(
        tifffile.imread(fine_scientific_case.dataset_root / name).shape
        for name in (
            "round-1-reference.tif",
            "round-2-registration.tif",
            "round-2-signal.tif",
            "round-2-label.tif",
        )
    )
    assert source_shapes == (
        _FIXED_SHAPE,
        _REGISTRATION_SHAPE,
        _SIGNAL_SHAPE,
        _LABEL_SHAPE,
    )
    assert all(
        all(
            fixed_size != moving_size
            for fixed_size, moving_size in zip(
                _FIXED_SHAPE,
                moving_shape,
                strict=True,
            )
        )
        for moving_shape in source_shapes[1:]
    )
    coarse = build_workflow_plan(
        fine_scientific_case.manifest,
        _profile(tmp_path, fine_scientific_case, name="coarse", mode="coarse"),
    )
    fine = build_workflow_plan(
        fine_scientific_case.manifest,
        _profile(
            tmp_path,
            fine_scientific_case,
            name="fine",
            mode="fine_grained",
        ),
    )
    assert coarse.scientific_json == fine.scientific_json
    assert coarse.scientific_fingerprint == fine.scientific_fingerprint

    coarse_result = run_workflow(coarse)
    fine_result = run_workflow(fine)
    assert coarse_result.succeeded
    assert fine_result.succeeded

    backend = AntsRegistrationBackend(native_threads)
    fixed_geometry = backend.geometry(
        backend.load_image(coarse.fixed_image, _PIXEL_SIZE)
    )
    assert fixed_geometry.shape == _FIXED_SHAPE
    coarse_paths = _registration_paths(coarse)
    fine_paths = _registration_paths(fine)
    coarse_affine = backend.read_affine(coarse_paths.affine)
    fine_affine = backend.read_affine(fine_paths.affine)
    np.testing.assert_allclose(
        coarse_affine.parameters,
        fine_affine.parameters,
        rtol=1e-4,
        atol=1e-5,
    )
    identity = np.asarray(
        ants.create_ants_transform(
            transform_type="AffineTransform",
            dimension=3,
        ).parameters
    )
    assert not np.allclose(coarse_affine.parameters, identity, rtol=0.0, atol=1e-7)

    nonzero_local_warp = False
    for chunk in chunks:
        coarse_warp = (
            coarse_paths.output
            / f"round_{_ROUND}_chunks"
            / f"chunk_{chunk.chunk_id:04d}_warp.nii.gz"
        )
        fine_warp = (
            fine_paths.output
            / f"round_{_ROUND}_chunks"
            / f"chunk_{chunk.chunk_id:04d}_warp.nii.gz"
        )
        region_geometry = read_prepared_region(
            PreparedManifest.from_dict(
                json.loads(
                    PreparedRoundPaths.for_round(
                        fine_paths.output,
                        _ROUND,
                    ).manifest.read_text(encoding="utf-8")
                )
            ),
            chunk,
        ).geometry
        _assert_warp_equal(
            backend,
            coarse_warp,
            fine_warp,
            expected_geometry=region_geometry,
        )
        nonzero_local_warp |= bool(
            np.any(np.abs(backend.to_array(backend.read_warp(fine_warp))) > 1e-7)
        )
        _assert_completion(coarse_warp)
        _assert_completion(fine_warp)
    assert nonzero_local_warp

    _assert_warp_equal(
        backend,
        coarse_paths.warp,
        fine_paths.warp,
        expected_geometry=fixed_geometry,
    )
    _assert_tiff_equal(
        coarse_paths.registered,
        fine_paths.registered,
        expected_geometry=fixed_geometry,
    )
    for coarse_metric, fine_metric in (
        (coarse_paths.pre_metrics, fine_paths.pre_metrics),
        (coarse_paths.linear_metrics, fine_paths.linear_metrics),
        (coarse_paths.nonlinear_metrics, fine_paths.nonlinear_metrics),
    ):
        coarse_values = read_metrics(coarse_metric)
        fine_values = read_metrics(fine_metric)
        assert set(coarse_values) == set(fine_values)
        np.testing.assert_allclose(
            tuple(coarse_values.values()),
            tuple(fine_values.values()),
            rtol=1e-4,
            atol=1e-5,
        )

    assert len(coarse.tasks[0].channels) == len(fine.tasks[0].channels) == 3
    for coarse_channel, fine_channel in zip(
        coarse.tasks[0].channels,
        fine.tasks[0].channels,
        strict=True,
    ):
        _assert_tiff_equal(
            coarse_channel.output,
            fine_channel.output,
            expected_geometry=fixed_geometry,
        )
        _assert_completion(coarse_channel.output)
        _assert_completion(fine_channel.output)

    prepared_paths = PreparedRoundPaths.for_round(fine_paths.output, _ROUND)
    prepared_manifest = PreparedManifest.from_dict(
        json.loads(prepared_paths.manifest.read_text(encoding="utf-8"))
    )
    assert prepared_manifest.geometry == fixed_geometry
    assert prepared_manifest.pixeltypes == {
        "fixed_mask": "float",
        "fixed_scalar": "unsigned int",
        "moving_linear_mask": "float",
        "moving_linear_scalar": "float",
    }
    for path in prepared_paths.array_paths:
        values = np.load(path, allow_pickle=False)
        assert values.shape == _FIXED_SHAPE
        assert values.dtype == np.dtype(np.float32)
        assert np.isfinite(values).all()
        _assert_completion(path)
    _assert_completion(prepared_paths.manifest)
    for chunk in chunks:
        region = read_prepared_region(prepared_manifest, chunk)
        assert region.geometry.shape == chunk.extended_shape
        local_warp = (
            fine_paths.output
            / f"round_{_ROUND}_chunks"
            / f"chunk_{chunk.chunk_id:04d}_warp.nii.gz"
        )
        assert geometry_compatible(
            backend.geometry(backend.read_warp(local_warp)),
            region.geometry,
        )
        core = np.load(
            local_warp.with_name(f"chunk_{chunk.chunk_id:04d}_registered_core.npy"),
            allow_pickle=False,
        )
        assert core.shape == chunk.core_shape
        assert core.dtype == np.dtype(np.float32)
        assert np.isfinite(core).all()


def test_fine_restart_preserves_two_chunks_then_one_channel_exactly(
    tmp_path: Path,
    fine_scientific_case: _ScientificCase,
    native_threads: NativeThreadConfig,
) -> None:
    del native_threads
    profile = _profile(
        tmp_path,
        fine_scientific_case,
        name="fine-resume",
        mode="fine_grained",
    )
    plan = build_workflow_plan(fine_scientific_case.manifest, profile)
    task = plan.tasks[0]
    paths = _registration_paths(plan)
    chunks = compute_chunk_grid(_FIXED_SHAPE, _CHUNK_SIZE, _OVERLAP)
    assert len(chunks) == 4
    chunk_outputs = tuple(
        path
        for chunk in chunks[:2]
        for path in (
            paths.output
            / f"round_{_ROUND}_chunks"
            / f"chunk_{chunk.chunk_id:04d}_warp.nii.gz",
            paths.output
            / f"round_{_ROUND}_chunks"
            / f"chunk_{chunk.chunk_id:04d}_registered_core.npy",
        )
    )

    first_results: list[FineTaskResult] = []
    committed_chunks = 0

    def interrupt_after_two_chunks(fine_task: FineWorkflowTask) -> FineTaskResult:
        nonlocal committed_chunks
        result = run_fine_task(fine_task)
        first_results.append(result)
        if type(fine_task) is NonlinearChunkTask:
            committed_chunks += 1
            if committed_chunks == 2:
                raise KeyboardInterrupt("after two committed chunks")
        return result

    first = SequentialExecutor(
        profile,
        runner=run_workflow_task,
        fine_runner=interrupt_after_two_chunks,
    )
    with pytest.raises(KeyboardInterrupt, match="after two committed chunks"):
        first.execute(plan.reference_task, plan.tasks)
    assert committed_chunks == 2
    assert all(path.is_file() for path in chunk_outputs)
    chunk_identities = _artifact_identities(chunk_outputs)

    second_results: list[FineTaskResult] = []
    committed_channels = 0

    def interrupt_after_one_channel(fine_task: FineWorkflowTask) -> FineTaskResult:
        nonlocal committed_channels
        result = run_fine_task(fine_task)
        second_results.append(result)
        if type(fine_task) is ChannelTransformTask:
            committed_channels += 1
            if committed_channels == 1:
                raise KeyboardInterrupt("after one committed channel")
        return result

    second = SequentialExecutor(
        profile,
        runner=run_workflow_task,
        fine_runner=interrupt_after_one_channel,
    )
    with pytest.raises(KeyboardInterrupt, match="after one committed channel"):
        second.execute(plan.reference_task, plan.tasks)
    assert _artifact_identities(chunk_outputs) == chunk_identities
    first_two_resumed = tuple(
        result
        for result in second_results
        if type(result.task) is NonlinearChunkTask
        and result.task.chunk.chunk_id in {0, 1}
    )
    assert len(first_two_resumed) == 2
    assert {result.status for result in first_two_resumed} == {"reused"}

    first_channel = task.channels[0].output
    first_channel_identities = _artifact_identities((first_channel,))
    final_results: list[FineTaskResult] = []

    def recording_runner(fine_task: FineWorkflowTask) -> FineTaskResult:
        result = run_fine_task(fine_task)
        final_results.append(result)
        return result

    final = SequentialExecutor(
        profile,
        runner=run_workflow_task,
        fine_runner=recording_runner,
    )
    round_results = final.execute(plan.reference_task, plan.tasks)
    assert tuple(result.status for result in round_results) == (
        "completed",
        "completed",
    )
    assert _artifact_identities(chunk_outputs) == chunk_identities
    assert _artifact_identities((first_channel,)) == first_channel_identities
    channel_results = tuple(
        result for result in final_results if type(result.task) is ChannelTransformTask
    )
    assert len(channel_results) == 3
    assert tuple(result.status for result in channel_results) == (
        "reused",
        "completed",
        "completed",
    )
    assert all(path.is_file() for path in plan.expected_artifacts)
