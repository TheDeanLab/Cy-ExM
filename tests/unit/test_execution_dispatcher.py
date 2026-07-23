from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from cyexm.config.models import (
    NativeThreadSpec,
    NonlinearRegistrationStageSpec,
    RegistrationSpec,
    RegistrationStageSpec,
)
from cyexm.execution.dispatcher import FineGraphController
from cyexm.workflow.fine_models import (
    FineTaskResult,
    FineWorkflowTask,
    RoundPreparationTask,
    RoundValidationTask,
)
from cyexm.workflow.models import ChannelTask, RoundFailure, RoundTask

pytestmark = pytest.mark.unit


def _round_task(tmp_path: Path, round_number: int) -> RoundTask:
    root = tmp_path / f"round-{round_number:03d}"
    return RoundTask(
        dataset_id="dataset",
        round=round_number,
        strategy="serial_chunked",
        fixed_image=tmp_path / "fixed.tif",
        moving_image=tmp_path / f"moving-{round_number}.tif",
        registration_directory=root / "registration",
        log_path=root / "logs" / f"round-{round_number:03d}.log",
        voxel_spacing_xyz=(1.0, 1.0, 1.0),
        registration=RegistrationSpec(
            crop=False,
            linear=RegistrationStageSpec("TRSAA", "high"),
            nonlinear=NonlinearRegistrationStageSpec("SyNOnly", "high", "CC", 4),
        ),
        native_threads=NativeThreadSpec(2, 3),
        channels=(
            ChannelTask(
                source=tmp_path / f"channel-{round_number}-0.tif",
                output=root / "channels" / "0.tif",
                interpolation="linear",
            ),
            ChannelTask(
                source=tmp_path / f"channel-{round_number}-1.tif",
                output=root / "channels" / "1.tif",
                interpolation="nearestNeighbor",
            ),
        ),
        chunk_size_zyx=(4, 8, 8),
        overlap_fraction=0.0,
    )


def _descriptor(z_size: int) -> dict[str, object]:
    identity = {"path": "/tmp/artifact", "st_size": 1, "st_mtime_ns": 2}
    return {
        "format_version": 1,
        "fixed_scalar": identity,
        "moving_linear_scalar": identity,
        "fixed_mask": identity,
        "moving_linear_mask": identity,
        "native_fixed": identity,
        "native_moving": identity,
        "moving_crop": None,
        "affine": identity,
        "geometry": {
            "shape": [z_size, 8, 8],
            "origin": [0.0, 0.0, 0.0],
            "spacing": [1.0, 1.0, 1.0],
            "direction": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        },
        "scalar_policy": "float32-preserve-registration-values-v1",
        "mask_policy": (
            "greater-than-zero-affine-nearest-neighbor-threshold-0.5-float32-v1"
        ),
        "native_threads": {"ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": 2},
    }


def _result(
    task: FineWorkflowTask,
    *,
    status: str = "completed",
    second: int = 0,
    reused: bool = False,
    failure: RoundFailure | None = None,
    prepared_z: int | None = None,
) -> FineTaskResult:
    artifacts = tuple(str(path) for path in task.output_paths)
    return FineTaskResult(
        task=task,
        status=status,  # type: ignore[arg-type]
        started_at=f"2026-07-23T00:00:{second:02d}Z",
        ended_at=f"2026-07-23T00:00:{second + 1:02d}Z",
        duration_seconds=1.0,
        hostname=f"host-{task.key.stage}",
        pid=100 + second,
        dask_task_key=f"dask-{task.key.encoded}",
        slurm_job_id=str(1000 + second),
        native_threads=(("itk", 2), ("blas", 3)),
        artifact_paths=artifacts if status in {"completed", "reused"} else (),
        reused_artifacts=artifacts if reused else (),
        log_path=str(task.log_path),
        failure=failure,
        prepared_descriptor=(
            _descriptor(prepared_z)
            if type(task) is RoundPreparationTask
            and status in {"completed", "reused"}
            and prepared_z is not None
            else None
        ),
    )


def _record_successes(
    controller: FineGraphController,
    tasks: tuple[FineWorkflowTask, ...],
    *,
    start: int,
) -> None:
    for offset, task in enumerate(tasks):
        controller.record(task, _result(task, second=start + offset))


def test_dispatcher_is_round_robin_and_work_conserving(tmp_path: Path) -> None:
    controller = FineGraphController(
        (_round_task(tmp_path, 3), _round_task(tmp_path, 2))
    )
    preparations = controller.pop_ready(limit=2)
    assert [task.round for task in preparations] == [2, 3]

    controller.record(preparations[1], _result(preparations[1], prepared_z=12))
    controller.record(preparations[0], _result(preparations[0], prepared_z=8))

    chunks = controller.pop_ready(limit=4)
    assert [task.round for task in chunks] == [2, 3, 2, 3]
    assert [task.key.item for task in chunks] == ["0000", "0000", "0001", "0001"]

    _record_successes(controller, chunks, start=2)
    remaining = controller.pop_ready(limit=4)
    assert len(remaining) == 2
    assert {task.round for task in remaining} == {2, 3}
    assert {task.key.stage for task in remaining} == {"chunk", "assembly"}

    round_three_chunk = next(task for task in remaining if task.key.stage == "chunk")
    controller.record(round_three_chunk, _result(round_three_chunk, second=8))
    round_three_assembly = controller.pop_ready(limit=4)
    assert len(round_three_assembly) == 1
    assert round_three_assembly[0].round == 3
    assert round_three_assembly[0].key.stage == "assembly"


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_dispatcher_requires_a_positive_non_boolean_limit(
    tmp_path: Path, limit: object
) -> None:
    controller = FineGraphController((_round_task(tmp_path, 2),))

    with pytest.raises((TypeError, ValueError), match="positive"):
        controller.pop_ready(limit=limit)  # type: ignore[arg-type]


def test_failed_chunk_blocks_assembly_after_running_siblings_settle(
    tmp_path: Path,
) -> None:
    controller = FineGraphController((_round_task(tmp_path, 2),))
    preparation = controller.pop_ready(limit=1)[0]
    controller.record(preparation, _result(preparation, prepared_z=12))
    chunks = controller.pop_ready(limit=3)
    assert len(chunks) == 3

    failure = RoundFailure("chunk", "RuntimeError", "injected chunk failure")
    controller.record(
        chunks[1],
        _result(chunks[1], status="failed", second=3, failure=failure),
    )
    assert not controller.terminal
    controller.record(chunks[2], _result(chunks[2], second=4))
    assert not controller.terminal
    controller.record(chunks[0], _result(chunks[0], second=2))

    assert controller.pop_ready(limit=4) == ()
    assert controller.terminal
    assert tuple(result.key for result in controller.task_results) == (
        preparation.key,
        chunks[0].key,
        chunks[1].key,
        chunks[2].key,
    )
    summary = controller.round_results()[0]
    assert summary.status == "failed"
    assert summary.failure == failure
    assert summary.hostname == "host-chunk"
    assert summary.pid == 103
    assert summary.artifact_paths == (str(preparation.output_paths[0]),)
    assert summary.started_at == "2026-07-23T00:00:00Z"
    assert summary.ended_at == "2026-07-23T00:00:05Z"
    assert summary.duration_seconds == 5.0


def test_round_results_use_validation_summary_and_producer_reuse(
    tmp_path: Path,
) -> None:
    controller = FineGraphController((_round_task(tmp_path, 2),))
    second = 0
    while not controller.terminal:
        ready = controller.pop_ready(limit=8)
        assert ready
        for task in reversed(ready):
            result = _result(
                task,
                second=second,
                reused=True,
                prepared_z=8 if type(task) is RoundPreparationTask else None,
            )
            controller.record(task, result)
            second += 1

    summary = controller.round_results()[0]
    validation = next(
        result
        for result in controller.task_results
        if type(result.task) is RoundValidationTask
    )
    assert summary.status == "reused"
    assert summary.hostname == validation.hostname
    assert summary.pid == validation.pid
    assert summary.dask_task_key == validation.dask_task_key
    assert summary.slurm_job_id == validation.slurm_job_id
    assert summary.log_path == validation.log_path
    assert summary.artifact_paths == validation.artifact_paths
    assert summary.reused_artifacts == validation.artifact_paths
    assert tuple(result.key for result in controller.task_results) != tuple(
        result.key for result in reversed(controller.task_results)
    )


def test_round_results_reject_premature_access_and_inconsistent_threads(
    tmp_path: Path,
) -> None:
    controller = FineGraphController((_round_task(tmp_path, 2),))
    with pytest.raises(RuntimeError, match="terminal"):
        controller.round_results()

    preparation = controller.pop_ready(limit=1)[0]
    controller.record(preparation, _result(preparation, prepared_z=8))
    chunks = controller.pop_ready(limit=2)
    first = _result(chunks[0], second=1)
    inconsistent = replace(first, native_threads=(("itk", 99),))
    controller.record(chunks[0], inconsistent)
    controller.record(chunks[1], _result(chunks[1], second=2))
    assembly = controller.pop_ready(limit=1)[0]
    controller.record(assembly, _result(assembly, second=3))
    channels = controller.pop_ready(limit=2)
    _record_successes(controller, channels, start=4)
    validation = controller.pop_ready(limit=1)[0]
    controller.record(validation, _result(validation, second=6))

    with pytest.raises(ValueError, match="native-thread"):
        controller.round_results()


def test_round_results_reject_an_individual_negative_task_interval(
    tmp_path: Path,
) -> None:
    controller = FineGraphController((_round_task(tmp_path, 2),))
    changed = False
    second = 0
    while not controller.terminal:
        ready = controller.pop_ready(limit=8)
        for task in ready:
            result = _result(
                task,
                second=second,
                prepared_z=8 if type(task) is RoundPreparationTask else None,
            )
            if task.key.stage == "chunk" and not changed:
                result = replace(
                    result,
                    started_at="2026-07-23T00:00:05Z",
                    ended_at="2026-07-23T00:00:04Z",
                )
                changed = True
            controller.record(task, result)
            second += 1

    with pytest.raises(ValueError, match="end timestamp precedes"):
        controller.round_results()


def test_record_rejects_unsubmitted_mismatched_and_duplicate_results(
    tmp_path: Path,
) -> None:
    controller = FineGraphController(
        (_round_task(tmp_path, 2), _round_task(tmp_path, 3))
    )
    first, second = controller.pop_ready(limit=2)

    with pytest.raises(ValueError, match="match"):
        controller.record(first, _result(second, prepared_z=8))
    controller.record(first, _result(first, prepared_z=8))
    with pytest.raises(ValueError, match="running"):
        controller.record(first, _result(first, prepared_z=8))
