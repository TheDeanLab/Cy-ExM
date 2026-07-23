from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import get_type_hints

import pytest

from cyexm.config.models import (
    NativeThreadSpec,
    NonlinearRegistrationStageSpec,
    RegistrationSpec,
    RegistrationStageSpec,
)
from cyexm.execution import SequentialExecutor, WorkflowExecutor, WorkflowTaskRunner
from cyexm.workflow.models import (
    ChannelTask,
    ReferenceChannelTask,
    ReferenceTask,
    RoundResult,
    RoundTask,
    WorkflowTask,
)
from cyexm.workflow.task import run_workflow_task

pytestmark = pytest.mark.unit


def test_execution_namespace_imports_in_a_fresh_process() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import cyexm.execution; assert cyexm.execution.SequentialExecutor",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_workflow_task_runner_alias_resolves_at_runtime() -> None:
    assert "Callable" in str(WorkflowTaskRunner.__value__)


def test_executor_annotations_resolve_at_runtime() -> None:
    protocol_hints = get_type_hints(WorkflowExecutor.execute)
    sequential_hints = get_type_hints(SequentialExecutor.execute)

    assert protocol_hints["reference_task"] is ReferenceTask
    assert protocol_hints["round_tasks"] == tuple[RoundTask, ...]
    assert protocol_hints["return"] == tuple[RoundResult, ...]
    assert sequential_hints == protocol_hints


def _task(tmp_path: Path, round_number: int) -> RoundTask:
    round_root = tmp_path / f"round-{round_number:03d}"
    return RoundTask(
        dataset_id="dataset-1",
        round=round_number,
        strategy="whole_image",
        fixed_image=tmp_path / "fixed.tif",
        moving_image=tmp_path / f"moving-{round_number}.tif",
        registration_directory=round_root / "registration",
        log_path=round_root / "logs" / f"round-{round_number:03d}.log",
        voxel_spacing_xyz=(1.0, 1.0, 1.0),
        registration=RegistrationSpec(
            crop=False,
            linear=RegistrationStageSpec(type="TRSAA", accuracy="high"),
            nonlinear=NonlinearRegistrationStageSpec(
                type="SyNOnly", accuracy="high", metric="mattes", radius=None
            ),
        ),
        native_threads=NativeThreadSpec(itk=2, blas=3),
        channels=(
            ChannelTask(
                source=tmp_path / f"moving-{round_number}.tif",
                output=round_root / "channels" / f"moving-{round_number}.tif",
                interpolation="linear",
            ),
        ),
        chunk_size_zyx=None,
        overlap_fraction=None,
    )


def _reference_task(tmp_path: Path) -> ReferenceTask:
    source = tmp_path / "fixed-source.tif"
    output = tmp_path / "round-001/channels/fixed.tif"
    return ReferenceTask(
        dataset_id="dataset-1",
        round=1,
        registration_channel_source=source,
        registration_channel_output=output,
        registration_directory=tmp_path / "round-001/registration",
        crop_bounds_path=None,
        log_path=tmp_path / "round-001/logs/round-001.log",
        voxel_spacing_xyz=(1.0, 1.0, 1.0),
        crop=False,
        native_threads=NativeThreadSpec(itk=2, blas=3),
        channels=(ReferenceChannelTask(source, output),),
    )


def _successful_result(task: WorkflowTask) -> RoundResult:
    return RoundResult(
        dataset_id=task.dataset_id,
        round=task.round,
        status="completed",
        started_at="2026-07-20T10:00:00Z",
        ended_at="2026-07-20T10:00:01Z",
        duration_seconds=1.0,
        hostname="worker-01",
        pid=123,
        dask_task_key=None,
        slurm_job_id=None,
        strategy=(
            "reference-preparation" if type(task) is ReferenceTask else task.strategy
        ),
        native_threads=(("itk", 2), ("blas", 3)),
        artifact_paths=tuple(str(channel.output) for channel in task.channels),
        reused_artifacts=(),
        log_path=str(task.log_path),
        failure=None,
    )


def test_sequential_executor_runs_every_task_in_order(tmp_path: Path) -> None:
    calls: list[int] = []

    def runner(task: WorkflowTask) -> RoundResult:
        calls.append(task.round)
        return _successful_result(task)

    tasks = (_task(tmp_path, 2), _task(tmp_path, 3))
    results = SequentialExecutor(runner=runner).execute(
        _reference_task(tmp_path), tasks
    )

    assert calls == [1, 2, 3]
    assert tuple(result.round for result in results) == (1, 2, 3)


def test_sequential_executor_rejects_a_noncoarse_result(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="exact RoundResult"):
        SequentialExecutor(runner=lambda _task: object()).execute(  # type: ignore[arg-type,return-value]
            _reference_task(tmp_path), ()
        )


def test_sequential_executor_records_failure_and_continues(tmp_path: Path) -> None:
    calls: list[int] = []

    def runner(task: WorkflowTask) -> RoundResult:
        calls.append(task.round)
        if task.round == 2:
            raise RuntimeError("injected round failure")
        return _successful_result(task)

    tasks = (_task(tmp_path, 2), _task(tmp_path, 3))
    reference, failed, completed = SequentialExecutor(runner=runner).execute(
        _reference_task(tmp_path), tasks
    )

    assert calls == [1, 2, 3]
    assert reference.status == "completed"
    assert failed.status == "failed"
    assert failed.round == 2
    assert failed.failure is not None
    assert failed.failure.stage == "round"
    assert failed.failure.exception_type == "RuntimeError"
    assert failed.failure.message == "injected round failure"
    assert failed.native_threads == (("itk", 2), ("blas", 3))
    assert failed.artifact_paths == ()
    assert completed.status == "completed"


def test_sequential_executor_propagates_keyboard_interrupt(tmp_path: Path) -> None:
    calls: list[int] = []

    def runner(task: WorkflowTask) -> RoundResult:
        calls.append(task.round)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        SequentialExecutor(runner=runner).execute(
            _reference_task(tmp_path), (_task(tmp_path, 2), _task(tmp_path, 3))
        )

    assert calls == [1]


def test_sequential_executor_attaches_results_completed_before_interrupt(
    tmp_path: Path,
) -> None:
    tasks = (_task(tmp_path, 2), _task(tmp_path, 3))

    def runner(task: WorkflowTask) -> RoundResult:
        if task.round == 3:
            raise KeyboardInterrupt("injected")
        return _successful_result(task)

    with pytest.raises(KeyboardInterrupt) as captured:
        SequentialExecutor(runner=runner).execute(_reference_task(tmp_path), tasks)

    assert tuple(result.round for result in captured.value.completed_results) == (1, 2)  # type: ignore[attr-defined]
    assert captured.value.original.args == ("injected",)  # type: ignore[attr-defined]


def test_sequential_executor_does_not_mutate_tasks(tmp_path: Path) -> None:
    original = _task(tmp_path, 2)
    forced = replace(original, force=True)

    result = SequentialExecutor(runner=_successful_result).execute(
        _reference_task(tmp_path), (forced,)
    )

    assert original.force is False
    assert forced.force is True
    assert result[1].round == forced.round


def test_sequential_executor_stops_after_failed_reference(tmp_path: Path) -> None:
    calls: list[int] = []

    def runner(task: WorkflowTask) -> RoundResult:
        calls.append(task.round)
        result = _successful_result(task)
        if type(task) is ReferenceTask:
            return replace(result, status="failed")
        return result

    results = SequentialExecutor(runner=runner).execute(
        _reference_task(tmp_path), (_task(tmp_path, 2), _task(tmp_path, 3))
    )

    assert calls == [1]
    assert tuple(result.round for result in results) == (1,)


def test_workflow_dispatcher_rejects_task_subclasses() -> None:
    with pytest.raises(TypeError, match="exact ReferenceTask or RoundTask"):
        run_workflow_task(object())  # type: ignore[arg-type]
