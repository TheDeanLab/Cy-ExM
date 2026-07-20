from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from cyexm.config.models import (
    NativeThreadSpec,
    RegistrationSpec,
    RegistrationStageSpec,
)
from cyexm.execution import RoundTaskRunner, SequentialExecutor
from cyexm.workflow.models import ChannelTask, RoundResult, RoundTask

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


def test_round_task_runner_alias_resolves_at_runtime() -> None:
    assert "Callable" in str(RoundTaskRunner.__value__)


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
            nonlinear=RegistrationStageSpec(type="SyNOnly", accuracy="high"),
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


def _successful_result(task: RoundTask) -> RoundResult:
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
        strategy=task.strategy,
        native_threads=(("itk", 2), ("blas", 3)),
        artifact_paths=tuple(str(channel.output) for channel in task.channels),
        reused_artifacts=(),
        log_path=str(task.log_path),
        failure=None,
    )


def test_sequential_executor_runs_every_task_in_order(tmp_path: Path) -> None:
    calls: list[int] = []

    def runner(task: RoundTask) -> RoundResult:
        calls.append(task.round)
        return _successful_result(task)

    tasks = (_task(tmp_path, 2), _task(tmp_path, 3))
    results = SequentialExecutor(runner=runner).execute(tasks)

    assert calls == [2, 3]
    assert tuple(result.round for result in results) == (2, 3)


def test_sequential_executor_records_failure_and_continues(tmp_path: Path) -> None:
    calls: list[int] = []

    def runner(task: RoundTask) -> RoundResult:
        calls.append(task.round)
        if task.round == 2:
            raise RuntimeError("injected round failure")
        return _successful_result(task)

    tasks = (_task(tmp_path, 2), _task(tmp_path, 3))
    failed, completed = SequentialExecutor(runner=runner).execute(tasks)

    assert calls == [2, 3]
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

    def runner(task: RoundTask) -> RoundResult:
        calls.append(task.round)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        SequentialExecutor(runner=runner).execute(
            (_task(tmp_path, 2), _task(tmp_path, 3))
        )

    assert calls == [2]


def test_sequential_executor_does_not_mutate_tasks(tmp_path: Path) -> None:
    original = _task(tmp_path, 2)
    forced = replace(original, force=True)

    result = SequentialExecutor(runner=_successful_result).execute((forced,))

    assert original.force is False
    assert forced.force is True
    assert result[0].round == forced.round
