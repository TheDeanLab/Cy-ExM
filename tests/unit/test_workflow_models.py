from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import FrozenInstanceError
from pathlib import Path, PurePosixPath

import pytest

from cyexm.config.models import (
    ChannelSpec,
    DatasetManifest,
    ExecutionPaths,
    ExecutionProfile,
    LocalDaskSpec,
    NativeThreadSpec,
    ReferenceSpec,
    RegistrationSpec,
    RegistrationStageSpec,
    RoundSpec,
    SlurmSpec,
)
from cyexm.spatial import ImageGeometry
from cyexm.workflow.layout import WorkflowLayout
from cyexm.workflow.models import (
    ChannelTask,
    ImageHeader,
    RoundFailure,
    RoundResult,
    RoundStatus,
    RoundTask,
    WorkflowPlan,
    WorkflowResult,
    WorkflowStatus,
    canonical_json,
    sha256_fingerprint,
)

pytestmark = pytest.mark.unit


def _registration() -> RegistrationSpec:
    return RegistrationSpec(
        crop=True,
        linear=RegistrationStageSpec(type="TRSAA", accuracy="high"),
        nonlinear=RegistrationStageSpec(type="SyNOnly", accuracy="high"),
    )


def _manifest() -> DatasetManifest:
    return DatasetManifest(
        schema_version=1,
        dataset_id="dataset-1",
        voxel_spacing_um=(0.324, 0.324, 0.9),
        reference=ReferenceSpec(
            round=1,
            channel=PurePosixPath("round_1_488.tif"),
        ),
        registration=_registration(),
        rounds=(
            RoundSpec(
                round=3,
                strategy="serial_chunked",
                registration_channel=PurePosixPath("round_3_488.tif"),
                channels=(
                    ChannelSpec(PurePosixPath("round_3_mask.tif"), "nearestNeighbor"),
                    ChannelSpec(PurePosixPath("round_3_488.tif"), "linear"),
                ),
                chunk_size_zyx=(4, 8, 8),
                overlap_fraction=0.25,
            ),
            RoundSpec(
                round=2,
                strategy="whole_image",
                registration_channel=PurePosixPath("round_2_488.tif"),
                channels=(ChannelSpec(PurePosixPath("round_2_488.tif"), "linear"),),
            ),
        ),
    )


def _task(tmp_path: Path, round_number: int) -> RoundTask:
    round_root = tmp_path / f"round-{round_number:03d}"
    channel_sources = (
        tmp_path / f"round_{round_number}_mask.tif",
        tmp_path / f"round_{round_number}_488.tif",
    )
    return RoundTask(
        dataset_id="dataset-1",
        round=round_number,
        strategy="serial_chunked" if round_number == 3 else "whole_image",
        fixed_image=tmp_path / "round_1_488.tif",
        moving_image=tmp_path / f"round_{round_number}_488.tif",
        registration_directory=round_root / "registration",
        log_path=round_root / "logs" / f"round-{round_number:03d}.log",
        voxel_spacing_xyz=(0.324, 0.324, 0.9),
        registration=_registration(),
        native_threads=NativeThreadSpec(itk=1, blas=1),
        channels=tuple(
            ChannelTask(
                source=source,
                output=round_root / "channels" / source.name,
                interpolation=(
                    "nearestNeighbor" if "mask" in source.name else "linear"
                ),
            )
            for source in channel_sources
        ),
        chunk_size_zyx=(4, 8, 8) if round_number == 3 else None,
        overlap_fraction=0.25 if round_number == 3 else None,
    )


def _execution_profile(tmp_path: Path, executor: str) -> ExecutionProfile:
    paths = ExecutionPaths(
        dataset_root=tmp_path / "dataset",
        output_root=tmp_path / "output",
        worker_local_directory=(
            None if executor == "sequential" else tmp_path / "scratch"
        ),
    )
    local_dask = (
        LocalDaskSpec(
            workers=2,
            threads_per_worker=1,
            processes=True,
            memory_limit="32GB",
            startup_timeout="30s",
            task_retries=1,
        )
        if executor == "local_dask"
        else None
    )
    slurm = (
        SlurmSpec(
            queue="512GB",
            workers=4,
            cores_per_worker=72,
            processes_per_worker=1,
            memory="500GB",
            walltime="72:00:00",
            interface="ib0",
            scheduler_interface="ib0",
            startup_timeout="15m",
            task_retries=0,
            job_name="cyexm-registration",
            job_extra_directives=("--exclusive",),
        )
        if executor == "slurm"
        else None
    )
    return ExecutionProfile(
        schema_version=1,
        executor=executor,  # type: ignore[arg-type]
        paths=paths,
        native_threads=NativeThreadSpec(
            itk={"sequential": 1, "local_dask": 2, "slurm": 4}[executor],
            blas={"sequential": 1, "local_dask": 3, "slurm": 5}[executor],
        ),
        local_dask=local_dask,
        slurm=slurm,
        source_path=tmp_path / f"{executor}.yaml",
    )


def _scientific_mapping(tmp_path: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset_id": "dataset-1",
        "voxel_spacing_xyz_um": [0.324, 0.324, 0.9],
        "registration": {
            "crop": True,
            "linear": {"type": "TRSAA", "accuracy": "high"},
            "nonlinear": {"type": "SyNOnly", "accuracy": "high"},
        },
        "sources": [
            {
                "path": str(tmp_path / "round_1_488.tif"),
                "shape_zyx": [4, 8, 8],
                "dtype": "uint16",
                "axes": "ZYX",
                "st_size": 1024,
                "st_mtime_ns": 123,
            }
        ],
        "rounds": [
            {
                "round": 2,
                "strategy": "whole_image",
                "channels": [
                    {
                        "file": str(tmp_path / "round_2_488.tif"),
                        "interpolation": "linear",
                    }
                ],
            },
            {
                "round": 3,
                "strategy": "serial_chunked",
                "chunk_size_zyx": [4, 8, 8],
                "overlap_fraction": 0.25,
                "channels": [
                    {
                        "file": str(tmp_path / "round_3_mask.tif"),
                        "interpolation": "nearestNeighbor",
                    },
                    {
                        "file": str(tmp_path / "round_3_488.tif"),
                        "interpolation": "linear",
                    },
                ],
            },
        ],
        "policies": {
            "axis_convention": "external-zyx-ants-xyz-v1",
            "output_format": "ome-compatible-tiff-uint16-v1",
            "postprocessing": "clip-uint16-no-normalization-v1",
        },
    }


def _round_result() -> RoundResult:
    return RoundResult(
        dataset_id="dataset-1",
        round=2,
        status="failed",
        started_at="2026-07-20T10:00:00Z",
        ended_at="2026-07-20T10:00:02Z",
        duration_seconds=2.0,
        hostname="worker-01",
        pid=1234,
        dask_task_key="cyexm-round-2",
        slurm_job_id="98765",
        strategy="whole_image",
        native_threads=(("itk", 1), ("blas", 1)),
        artifact_paths=("/archive/output/affine.mat",),
        reused_artifacts=("/archive/output/affine.mat",),
        log_path="/archive/output/round-002.log",
        failure=RoundFailure(
            stage="nonlinear",
            exception_type="RuntimeError",
            message="injected failure",
        ),
    )


def test_layout_isolates_each_round_and_channel_output(tmp_path: Path) -> None:
    layout = WorkflowLayout.for_dataset(tmp_path, "dataset-1")
    round_layout = layout.round(2)

    assert layout.plan_path == tmp_path / "dataset-1" / "scientific-plan.json"
    assert layout.workflow_runs == tmp_path / "dataset-1" / "workflow-runs"
    assert round_layout.registration == (
        tmp_path / "dataset-1/rounds/round-002/registration"
    )
    assert round_layout.channels == tmp_path / "dataset-1/rounds/round-002/channels"
    assert round_layout.channel("nested/round_2_488.tif") == (
        tmp_path / "dataset-1/rounds/round-002/channels/round_2_488.tif"
    )
    assert round_layout.log == (
        tmp_path / "dataset-1/rounds/round-002/logs/round-002.log"
    )


def test_layout_construction_creates_no_files_or_directories(tmp_path: Path) -> None:
    output_root = tmp_path / "missing-output"

    layout = WorkflowLayout.for_dataset(output_root, "dataset-1")
    _ = layout.round(3)

    assert not output_root.exists()


def test_all_workflow_values_are_frozen_and_slotted(tmp_path: Path) -> None:
    task = _task(tmp_path, 2)
    scientific_json = canonical_json(_scientific_mapping(tmp_path))
    fingerprint = sha256_fingerprint(scientific_json)
    round_result = _round_result()
    values = (
        task.channels[0],
        task,
        ImageHeader(
            path=tmp_path / "round_1_488.tif",
            shape_zyx=(4, 8, 8),
            dtype="uint16",
            axes="ZYX",
            geometry=ImageGeometry(
                shape=(4, 8, 8),
                origin=(0.0, 0.0, 0.0),
                spacing=(0.324, 0.324, 0.9),
                direction=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            ),
            st_size=1024,
            st_mtime_ns=123,
        ),
        round_result.failure,
        round_result,
        WorkflowPlan(
            schema_version=1,
            dataset_id="dataset-1",
            manifest=_manifest(),
            execution=_execution_profile(tmp_path, "sequential"),
            fixed_image=tmp_path / "round_1_488.tif",
            tasks=(task,),
            expected_artifacts=(task.registration_directory,),
            scientific_json=scientific_json,
            scientific_fingerprint=fingerprint,
        ),
        WorkflowResult(
            dataset_id="dataset-1",
            scientific_fingerprint=fingerprint,
            status="failed",
            started_at="2026-07-20T10:00:00Z",
            ended_at="2026-07-20T10:00:02Z",
            rounds=(round_result,),
            run_record="/archive/output/run.json",
        ),
        RoundStatus(
            round=2,
            status="partial",
            complete_artifacts=("affine.mat",),
            incomplete_artifacts=("warp.nii.gz",),
            invalid_artifacts=(),
        ),
        WorkflowStatus(
            dataset_id="dataset-1",
            scientific_fingerprint=fingerprint,
            rounds=(),
            complete=False,
        ),
        WorkflowLayout.for_dataset(tmp_path, "dataset-1"),
        WorkflowLayout.for_dataset(tmp_path, "dataset-1").round(2),
    )

    for value in values:
        assert not hasattr(value, "__dict__")
        field_name = next(iter(value.__dataclass_fields__))  # type: ignore[attr-defined]
        with pytest.raises(FrozenInstanceError):
            setattr(value, field_name, getattr(value, field_name))


def test_round_task_is_pickle_safe_for_dask_transport(tmp_path: Path) -> None:
    task = _task(tmp_path, 3)

    restored = pickle.loads(pickle.dumps(task))

    assert restored == task


def test_workflow_plan_sorts_rounds_stably_and_preserves_channel_order(
    tmp_path: Path,
) -> None:
    round_3_first = _task(tmp_path, 3)
    round_3_second = _task(tmp_path / "duplicate", 3)
    round_2 = _task(tmp_path, 2)
    scientific_json = canonical_json(_scientific_mapping(tmp_path))

    plan = WorkflowPlan(
        schema_version=1,
        dataset_id="dataset-1",
        manifest=_manifest(),
        execution=_execution_profile(tmp_path, "sequential"),
        fixed_image=tmp_path / "round_1_488.tif",
        tasks=(round_3_first, round_2, round_3_second),
        expected_artifacts=(),
        scientific_json=scientific_json,
        scientific_fingerprint=sha256_fingerprint(scientific_json),
    )

    assert plan.tasks == (round_2, round_3_first, round_3_second)
    assert tuple(channel.source.name for channel in plan.tasks[1].channels) == (
        "round_3_mask.tif",
        "round_3_488.tif",
    )


def test_result_and_status_dictionaries_are_deeply_json_compatible() -> None:
    round_result = _round_result()
    workflow_result = WorkflowResult(
        dataset_id="dataset-1",
        scientific_fingerprint="a" * 64,
        status="failed",
        started_at="2026-07-20T10:00:00Z",
        ended_at="2026-07-20T10:00:02Z",
        rounds=(round_result,),
        run_record="/archive/output/run.json",
        execution_evidence_json='{"executor":"slurm","job_ids":["98765"]}',
    )
    round_status = RoundStatus(
        round=2,
        status="invalid",
        complete_artifacts=("affine.mat",),
        incomplete_artifacts=("warp.nii.gz",),
        invalid_artifacts=("channel.tif",),
    )
    workflow_status = WorkflowStatus(
        dataset_id="dataset-1",
        scientific_fingerprint="a" * 64,
        rounds=(round_status,),
        complete=False,
    )

    round_payload = round_result.to_dict()
    workflow_payload = workflow_result.to_dict()
    round_status_payload = round_status.to_dict()
    workflow_status_payload = workflow_status.to_dict()

    assert round_payload["native_threads"] == [["itk", 1], ["blas", 1]]
    assert round_payload["failure"] == {
        "stage": "nonlinear",
        "exception_type": "RuntimeError",
        "message": "injected failure",
    }
    assert workflow_payload["rounds"] == [round_payload]
    assert workflow_payload["execution_evidence_json"] == (
        '{"executor":"slurm","job_ids":["98765"]}'
    )
    assert round_status_payload["complete_artifacts"] == ["affine.mat"]
    assert workflow_status_payload["rounds"] == [round_status_payload]
    for payload in (
        round_payload,
        workflow_payload,
        round_status_payload,
        workflow_status_payload,
    ):
        json.dumps(payload, allow_nan=False)


def test_workflow_result_succeeded_only_for_completed_status() -> None:
    common = {
        "dataset_id": "dataset-1",
        "scientific_fingerprint": "a" * 64,
        "started_at": "2026-07-20T10:00:00Z",
        "ended_at": "2026-07-20T10:00:02Z",
        "rounds": (),
        "run_record": "/archive/output/run.json",
    }

    assert WorkflowResult(status="completed", **common).succeeded  # type: ignore[arg-type]
    assert not WorkflowResult(status="failed", **common).succeeded  # type: ignore[arg-type]
    assert not WorkflowResult(status="cancelled", **common).succeeded  # type: ignore[arg-type]


def test_canonical_json_is_compact_sorted_and_rejects_nan() -> None:
    encoded = canonical_json({"z": [3, 2, 1], "a": {"truth": True}})

    assert encoded == '{"a":{"truth":true},"z":[3,2,1]}'
    with pytest.raises(ValueError, match="Out of range float values"):
        canonical_json({"invalid": float("nan")})


def test_sha256_fingerprint_hashes_the_canonical_utf8_bytes() -> None:
    encoded = '{"dataset_id":"dataset-1","schema_version":1}'

    assert (
        sha256_fingerprint(encoded)
        == hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    )


def test_scientific_identity_excludes_all_execution_profile_choices(
    tmp_path: Path,
) -> None:
    scientific_values = _scientific_mapping(tmp_path)
    scientific_json = canonical_json(scientific_values)
    fingerprint = sha256_fingerprint(scientific_json)
    tasks = (_task(tmp_path, 2), _task(tmp_path, 3))

    plans = tuple(
        WorkflowPlan(
            schema_version=1,
            dataset_id="dataset-1",
            manifest=_manifest(),
            execution=_execution_profile(tmp_path, executor),
            fixed_image=tmp_path / "round_1_488.tif",
            tasks=tasks,
            expected_artifacts=tuple(task.registration_directory for task in tasks),
            scientific_json=scientific_json,
            scientific_fingerprint=fingerprint,
        )
        for executor in ("sequential", "local_dask", "slurm")
    )

    assert {plan.execution.executor for plan in plans} == {
        "sequential",
        "local_dask",
        "slurm",
    }
    assert {plan.execution.native_threads for plan in plans} == {
        NativeThreadSpec(itk=1, blas=1),
        NativeThreadSpec(itk=2, blas=3),
        NativeThreadSpec(itk=4, blas=5),
    }
    assert {plan.scientific_json for plan in plans} == {scientific_json}
    assert {plan.scientific_fingerprint for plan in plans} == {fingerprint}
