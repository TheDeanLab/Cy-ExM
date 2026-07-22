from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path, PurePosixPath

import pytest

from cyexm.config.models import (
    ChannelSpec,
    DatasetManifest,
    ExecutionPaths,
    ExecutionProfile,
    ExecutorName,
    LocalDaskSpec,
    NativeThreadSpec,
    NonlinearRegistrationStageSpec,
    ReferenceChannelSpec,
    ReferenceSpec,
    RegistrationSpec,
    RegistrationStageSpec,
    RoundSpec,
)
from cyexm.io.tiff import read_tiff_geometry
from cyexm.registration.artifacts import ArtifactStore, CompletionRecord, FileIdentity
from cyexm.runtime import NATIVE_THREAD_ENV_VARS, NativeThreadConfig
from cyexm.workflow._round_provenance import RegistrationArtifactPaths
from cyexm.workflow.api import run_workflow
from cyexm.workflow.models import RoundResult, RoundTask, WorkflowPlan
from cyexm.workflow.planning import build_workflow_plan

pytestmark = pytest.mark.integration

_PIXEL_SIZE = (0.2, 0.2, 0.5)
_SHAPE = (16, 32, 32)
_DATASET_ID = "local-dask-scientific-equivalence"


def _write_dataset(
    dataset_root: Path,
    synthetic_tiffs: tuple[Path, Path],
) -> None:
    import numpy as np
    import tifffile

    fixed_path, moving_path = synthetic_tiffs
    fixed = np.asarray(tifffile.imread(fixed_path), dtype=np.float32)
    moving = np.asarray(tifffile.imread(moving_path), dtype=np.float32)
    assert fixed.shape == moving.shape == _SHAPE

    dataset_root.mkdir()
    sources = {
        "round-1-reference.tif": fixed,
        "round-2-registration.tif": moving,
        "round-2-signal.tif": np.flip(moving, axis=2) * np.float32(0.5)
        + np.float32(127.0),
        "round-3-registration.tif": np.roll(moving, shift=1, axis=1),
        "round-3-mask.tif": np.where(moving > 1_000.0, 60_000, 0).astype(np.uint16),
    }
    for name, values in sources.items():
        tifffile.imwrite(dataset_root / name, values, metadata={"axes": "ZYX"})


def _manifest() -> DatasetManifest:
    return DatasetManifest(
        schema_version=1,
        dataset_id=_DATASET_ID,
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
                type="SyNOnly", accuracy="dry run", metric="mattes", radius=None
            ),
        ),
        rounds=(
            RoundSpec(
                round=2,
                strategy="whole_image",
                registration_channel=PurePosixPath("round-2-registration.tif"),
                channels=(
                    ChannelSpec(
                        PurePosixPath("round-2-registration.tif"),
                        "linear",
                    ),
                    ChannelSpec(PurePosixPath("round-2-signal.tif"), "linear"),
                ),
            ),
            RoundSpec(
                round=3,
                strategy="serial_chunked",
                registration_channel=PurePosixPath("round-3-registration.tif"),
                channels=(
                    ChannelSpec(
                        PurePosixPath("round-3-registration.tif"),
                        "linear",
                    ),
                    ChannelSpec(
                        PurePosixPath("round-3-mask.tif"),
                        "nearestNeighbor",
                    ),
                ),
                # One full-volume chunk exercises the real serial chunked strategy
                # without expanding the bounded scientific fixture.
                chunk_size_zyx=_SHAPE,
                overlap_fraction=0.25,
            ),
        ),
    )


def _profile(
    tmp_path: Path,
    *,
    dataset_root: Path,
    executor: ExecutorName,
) -> ExecutionProfile:
    output_root = (tmp_path / f"{executor}-output").resolve()
    local_dask = None
    worker_local_directory = None
    if executor == "local_dask":
        worker_local_directory = (tmp_path / "local-dask-scratch").resolve()
        local_dask = LocalDaskSpec(
            workers=2,
            threads_per_worker=1,
            processes=True,
            memory_limit="2GB",
            startup_timeout="30s",
            task_retries=0,
        )
    return ExecutionProfile(
        schema_version=1,
        executor=executor,
        paths=ExecutionPaths(
            dataset_root=dataset_root.resolve(),
            output_root=output_root,
            worker_local_directory=worker_local_directory,
        ),
        native_threads=NativeThreadSpec(itk=1, blas=1),
        local_dask=local_dask,
        slurm=None,
        source_path=(tmp_path / f"execution-{executor}.yaml").resolve(),
    )


def _record(artifact: Path, *, stage: str) -> CompletionRecord:
    store = ArtifactStore()
    assert artifact.is_file()
    record_path = store.record_path(artifact)
    assert record_path.is_file()
    record = CompletionRecord.read(record_path)
    assert record.to_json() == record_path.read_text(encoding="utf-8")
    assert record.request.stage == stage
    assert record.artifact == FileIdentity.from_path(artifact)
    assert set(record.request.native_threads) == set(NATIVE_THREAD_ENV_VARS)
    assert set(record.request.native_threads.values()) == {1}
    return record


def _authoritative_records(task: RoundTask) -> dict[str, CompletionRecord]:
    paths = RegistrationArtifactPaths.for_round(
        task.registration_directory,
        task.round,
    )
    records = {
        "affine": _record(paths.affine, stage="affine"),
        "warp": _record(paths.warp, stage="nonlinear"),
        "diagnostic": _record(paths.registered, stage="registered-image"),
    }
    records.update(
        {
            f"channel:{channel.source.name}": _record(
                channel.output,
                stage="channel-transform-v1",
            )
            for channel in task.channels
        }
    )
    return records


def _assert_record_semantics_match(
    sequential: dict[str, CompletionRecord],
    local: dict[str, CompletionRecord],
) -> None:
    assert set(sequential) == set(local)
    for name in sequential:
        sequential_request = sequential[name].request
        local_request = local[name].request
        assert sequential_request.stage == local_request.stage
        assert sequential_request.geometry == local_request.geometry
        assert sequential_request.method == local_request.method
        assert sequential_request.accuracy == local_request.accuracy
        assert sequential_request.native_threads == local_request.native_threads
        assert set(sequential_request.parameters) == set(local_request.parameters)
        if name.startswith("channel:"):
            assert sequential_request.parameters == local_request.parameters


def _assert_tiff_equivalent(sequential: Path, local: Path) -> None:
    import numpy as np
    import tifffile

    sequential_values = tifffile.imread(sequential)
    local_values = tifffile.imread(local)
    assert sequential_values.dtype == local_values.dtype == np.dtype(np.uint16)
    assert sequential_values.shape == local_values.shape == _SHAPE

    sequential_geometry = read_tiff_geometry(sequential, sequential_values.shape)
    local_geometry = read_tiff_geometry(local, local_values.shape)
    assert sequential_geometry is not None
    assert local_geometry is not None
    assert sequential_geometry == local_geometry
    assert sequential_geometry.shape == local_geometry.shape == _SHAPE
    assert sequential_geometry.origin == local_geometry.origin
    assert sequential_geometry.spacing == local_geometry.spacing == _PIXEL_SIZE
    assert sequential_geometry.direction == local_geometry.direction

    # This established migration tolerance covers ANTSpyX floating-point
    # variability between independently estimated affine-plus-warp chains.
    np.testing.assert_allclose(
        sequential_values,
        local_values,
        rtol=1e-4,
        atol=1e-5,
    )


def _assert_round_outputs_equivalent(
    sequential: RoundTask,
    local: RoundTask,
) -> None:
    assert sequential.round == local.round
    assert sequential.strategy == local.strategy
    sequential_records = _authoritative_records(sequential)
    local_records = _authoritative_records(local)
    _assert_record_semantics_match(sequential_records, local_records)

    sequential_paths = RegistrationArtifactPaths.for_round(
        sequential.registration_directory,
        sequential.round,
    )
    local_paths = RegistrationArtifactPaths.for_round(
        local.registration_directory,
        local.round,
    )
    _assert_tiff_equivalent(sequential_paths.registered, local_paths.registered)
    assert len(sequential.channels) == len(local.channels)
    for sequential_channel, local_channel in zip(
        sequential.channels,
        local.channels,
        strict=True,
    ):
        assert sequential_channel.source == local_channel.source
        assert sequential_channel.interpolation == local_channel.interpolation
        _assert_tiff_equivalent(sequential_channel.output, local_channel.output)


def _artifact_and_record_identities(
    plan: WorkflowPlan,
) -> tuple[tuple[FileIdentity, FileIdentity], ...]:
    store = ArtifactStore()
    return tuple(
        (
            FileIdentity.from_path(artifact),
            FileIdentity.from_path(store.record_path(artifact)),
        )
        for artifact in plan.expected_artifacts
    )


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _assert_distinct_overlapping_workers(results: tuple[RoundResult, ...]) -> None:
    assert len(results) == 2
    assert {result.status for result in results} == {"completed"}
    assert len({result.pid for result in results}) == 2
    assert all(result.pid != os.getpid() for result in results)
    assert all(result.dask_task_key is not None for result in results)
    assert len({result.dask_task_key for result in results}) == 2
    assert all(
        result.dask_task_key == f"cyexm-{_DATASET_ID}-round-{result.round:03d}"
        for result in results
    )
    assert all(result.slurm_job_id is None for result in results)
    assert all(
        result.native_threads == tuple((name, 1) for name in NATIVE_THREAD_ENV_VARS)
        for result in results
    )

    first, second = results
    assert _parse_timestamp(first.started_at) < _parse_timestamp(second.ended_at)
    assert _parse_timestamp(second.started_at) < _parse_timestamp(first.ended_at)


def test_sequential_and_local_dask_workflows_are_scientifically_equivalent_and_reuse(
    tmp_path: Path,
    synthetic_tiffs: tuple[Path, Path],
    native_threads: NativeThreadConfig,
) -> None:
    del native_threads
    dataset_root = tmp_path / "dataset"
    _write_dataset(dataset_root, synthetic_tiffs)
    manifest = _manifest()
    sequential_plan = build_workflow_plan(
        manifest,
        _profile(tmp_path, dataset_root=dataset_root, executor="sequential"),
    )
    local_plan = build_workflow_plan(
        manifest,
        _profile(tmp_path, dataset_root=dataset_root, executor="local_dask"),
    )
    assert sequential_plan.execution.paths.output_root != (
        local_plan.execution.paths.output_root
    )
    assert sequential_plan.scientific_fingerprint == local_plan.scientific_fingerprint

    sequential_result = run_workflow(sequential_plan)
    local_result = run_workflow(local_plan)

    assert sequential_result.succeeded
    assert local_result.succeeded
    assert tuple(result.status for result in sequential_result.rounds) == (
        "completed",
        "completed",
        "completed",
    )
    assert local_result.rounds[0].strategy == "reference-preparation"
    _assert_distinct_overlapping_workers(local_result.rounds[1:])
    for sequential_task, local_task in zip(
        sequential_plan.tasks,
        local_plan.tasks,
        strict=True,
    ):
        _assert_round_outputs_equivalent(sequential_task, local_task)

    identities_before_reuse = _artifact_and_record_identities(local_plan)
    reused = run_workflow(local_plan)

    assert reused.succeeded
    assert tuple(result.status for result in reused.rounds) == (
        "completed",
        "reused",
        "reused",
    )
    expected_by_round = {
        local_plan.reference_task.round: tuple(
            str(channel.output) for channel in local_plan.reference_task.channels
        ),
        **{
            task.round: tuple(
                str(path)
                for path in local_plan.expected_artifacts
                if path.is_relative_to(task.registration_directory.parent)
            )
            for task in local_plan.tasks
        },
    }
    assert tuple(result.artifact_paths for result in reused.rounds) == tuple(
        expected_by_round[result.round] for result in reused.rounds
    )
    assert tuple(result.reused_artifacts for result in reused.rounds) == tuple(
        result.artifact_paths for result in reused.rounds
    )
    assert _artifact_and_record_identities(local_plan) == identities_before_reuse
