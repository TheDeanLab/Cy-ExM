from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import get_type_hints

import pytest

from cyexm.config.models import (
    ExecutionPaths,
    ExecutionProfile,
    NativeThreadSpec,
    NonlinearRegistrationStageSpec,
    ParallelismSpec,
    RegistrationSpec,
    RegistrationStageSpec,
)
from cyexm.errors import ConfigurationValidationError
from cyexm.execution import SequentialExecutor, WorkflowExecutor, WorkflowTaskRunner
from cyexm.workflow._execution_validation import _FineExecutionEvidence
from cyexm.workflow.fine_models import (
    FineTaskResult,
    FineWorkflowTask,
    RoundPreparationTask,
)
from cyexm.workflow.models import (
    ChannelTask,
    ReferenceChannelTask,
    ReferenceTask,
    RoundFailure,
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


def _profile(tmp_path: Path, mode: str) -> ExecutionProfile:
    return ExecutionProfile(
        schema_version=1,
        executor="sequential",
        paths=ExecutionPaths(tmp_path / "dataset", tmp_path / "output", None),
        native_threads=NativeThreadSpec(2, 3),
        local_dask=None,
        slurm=None,
        source_path=tmp_path / "execution.yaml",
        parallelism=ParallelismSpec(mode=mode),  # type: ignore[arg-type]
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


def _fine_result(
    task: FineWorkflowTask,
    *,
    status: str = "completed",
    failure: RoundFailure | None = None,
) -> FineTaskResult:
    descriptor = None
    if type(task) is RoundPreparationTask and status in {"completed", "reused"}:
        identity = {"path": "/tmp/item", "st_size": 1, "st_mtime_ns": 2}
        descriptor = {
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
                "shape": [4, 8, 8],
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
    return FineTaskResult(
        task=task,
        status=status,  # type: ignore[arg-type]
        started_at="2026-07-23T00:00:00Z",
        ended_at="2026-07-23T00:00:01Z",
        duration_seconds=1.0,
        hostname="host",
        pid=1,
        dask_task_key=None,
        slurm_job_id=None,
        native_threads=(("itk", 2), ("blas", 3)),
        artifact_paths=(
            tuple(str(path) for path in task.output_paths)
            if status in {"completed", "reused"}
            else ()
        ),
        reused_artifacts=(),
        log_path=str(task.log_path),
        failure=failure,
        prepared_descriptor=descriptor,
    )


def test_sequential_executor_runs_every_task_in_order(tmp_path: Path) -> None:
    calls: list[int] = []

    def runner(task: WorkflowTask) -> RoundResult:
        calls.append(task.round)
        return _successful_result(task)

    tasks = (_task(tmp_path, 2), _task(tmp_path, 3))
    results = SequentialExecutor(_profile(tmp_path, "coarse"), runner=runner).execute(
        _reference_task(tmp_path), tasks
    )

    assert calls == [1, 2, 3]
    assert tuple(result.round for result in results) == (1, 2, 3)


def test_sequential_executor_rejects_a_noncoarse_result(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="exact RoundResult"):
        SequentialExecutor(
            _profile(tmp_path, "coarse"),
            runner=lambda _task: object(),  # type: ignore[arg-type,return-value]
        ).execute(_reference_task(tmp_path), ())


def test_sequential_executor_records_failure_and_continues(tmp_path: Path) -> None:
    calls: list[int] = []

    def runner(task: WorkflowTask) -> RoundResult:
        calls.append(task.round)
        if task.round == 2:
            raise RuntimeError("injected round failure")
        return _successful_result(task)

    tasks = (_task(tmp_path, 2), _task(tmp_path, 3))
    reference, failed, completed = SequentialExecutor(
        _profile(tmp_path, "coarse"), runner=runner
    ).execute(_reference_task(tmp_path), tasks)

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
        SequentialExecutor(_profile(tmp_path, "coarse"), runner=runner).execute(
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
        SequentialExecutor(_profile(tmp_path, "coarse"), runner=runner).execute(
            _reference_task(tmp_path), tasks
        )

    assert tuple(result.round for result in captured.value.completed_results) == (1, 2)  # type: ignore[attr-defined]
    assert captured.value.original.args == ("injected",)  # type: ignore[attr-defined]


def test_sequential_executor_does_not_mutate_tasks(tmp_path: Path) -> None:
    original = _task(tmp_path, 2)
    forced = replace(original, force=True)

    result = SequentialExecutor(
        _profile(tmp_path, "coarse"), runner=_successful_result
    ).execute(_reference_task(tmp_path), (forced,))

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

    results = SequentialExecutor(_profile(tmp_path, "coarse"), runner=runner).execute(
        _reference_task(tmp_path), (_task(tmp_path, 2), _task(tmp_path, 3))
    )

    assert calls == [1]
    assert tuple(result.round for result in results) == (1,)


def test_workflow_dispatcher_rejects_task_subclasses() -> None:
    with pytest.raises(TypeError, match="exact ReferenceTask or RoundTask"):
        run_workflow_task(object())  # type: ignore[arg-type]


def test_sequential_fine_mode_executes_the_exact_graph_one_at_a_time(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def coarse_runner(task: WorkflowTask) -> RoundResult:
        calls.append(f"coarse-{task.round}")
        return _successful_result(task)

    def fine_runner(task: FineWorkflowTask) -> FineTaskResult:
        calls.append(task.key.encoded)
        return _fine_result(task)

    chunked = replace(
        _task(tmp_path, 2),
        strategy="serial_chunked",
        chunk_size_zyx=(4, 8, 8),
        overlap_fraction=0.0,
    )
    results = SequentialExecutor(
        _profile(tmp_path, "fine_grained"),
        runner=coarse_runner,
        fine_runner=fine_runner,
    ).execute(_reference_task(tmp_path), (chunked,))

    assert calls == [
        "coarse-1",
        "round-002:preparation:main",
        "round-002:chunk:0000",
        "round-002:assembly:main",
        "round-002:channel:0000",
        "round-002:validation:main",
    ]
    assert tuple(result.round for result in results) == (1, 2)


def test_sequential_fine_mode_retains_detached_granular_evidence(
    tmp_path: Path,
) -> None:
    chunked = replace(
        _task(tmp_path, 2),
        strategy="serial_chunked",
        chunk_size_zyx=(4, 8, 8),
        overlap_fraction=0.0,
    )
    executor = SequentialExecutor(
        _profile(tmp_path, "fine_grained"),
        runner=_successful_result,
        fine_runner=_fine_result,
    )

    executor.execute(_reference_task(tmp_path), (chunked,))
    evidence = executor.execution_evidence

    assert type(evidence) is _FineExecutionEvidence
    assert evidence.executor == "sequential"
    assert evidence.worker_cap == 1
    assert tuple(key for key, _dependencies in evidence.task_graph) == (
        "round-002:preparation:main",
        "round-002:chunk:0000",
        "round-002:assembly:main",
        "round-002:channel:0000",
        "round-002:validation:main",
    )
    assert tuple(key for key, _attempts in evidence.attempt_counts) == tuple(
        key for key, _dependencies in evidence.task_graph
    )
    assert evidence.blocked_descendants == ()
    assert not evidence.cancelled
    assert evidence.cleanup == ()


def test_sequential_coarse_mode_never_calls_fine_runner(tmp_path: Path) -> None:
    calls: list[int] = []

    def coarse_runner(task: WorkflowTask) -> RoundResult:
        calls.append(task.round)
        return _successful_result(task)

    executor = SequentialExecutor(
        _profile(tmp_path, "coarse"),
        runner=coarse_runner,
        fine_runner=lambda _task: pytest.fail("fine runner called in coarse mode"),
    )

    executor.execute(_reference_task(tmp_path), (_task(tmp_path, 2),))
    assert calls == [1, 2]


def test_sequential_executor_revalidates_direct_execution_profiles(
    tmp_path: Path,
) -> None:
    invalid = replace(
        _profile(tmp_path, "coarse"),
        parallelism=ParallelismSpec(mode="fine"),  # type: ignore[arg-type]
    )

    with pytest.raises(ConfigurationValidationError, match=r"parallelism\.mode"):
        SequentialExecutor(invalid, runner=_successful_result)


def test_sequential_fine_mode_settles_ready_siblings_after_failure(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def fine_runner(task: FineWorkflowTask) -> FineTaskResult:
        calls.append(task.key.encoded)
        if task.key.stage == "chunk" and task.key.item == "0000":
            failure = RoundFailure("chunk", "RuntimeError", "injected")
            return _fine_result(task, status="failed", failure=failure)
        return _fine_result(task)

    chunked = replace(
        _task(tmp_path, 2),
        strategy="serial_chunked",
        chunk_size_zyx=(2, 8, 8),
        overlap_fraction=0.0,
    )
    reference, failed = SequentialExecutor(
        _profile(tmp_path, "fine_grained"),
        runner=_successful_result,
        fine_runner=fine_runner,
    ).execute(_reference_task(tmp_path), (chunked,))

    assert reference.status == "completed"
    assert failed.status == "failed"
    assert failed.failure == RoundFailure("chunk", "RuntimeError", "injected")
    assert calls == [
        "round-002:preparation:main",
        "round-002:chunk:0000",
        "round-002:chunk:0001",
    ]


def test_sequential_fine_interruption_retains_actual_task_evidence(
    tmp_path: Path,
) -> None:
    calls = 0

    def fine_runner(task: FineWorkflowTask) -> FineTaskResult:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt("injected fine interruption")
        return _fine_result(task)

    chunked = replace(
        _task(tmp_path, 2),
        strategy="serial_chunked",
        chunk_size_zyx=(4, 8, 8),
        overlap_fraction=0.0,
    )
    with pytest.raises(KeyboardInterrupt) as captured:
        SequentialExecutor(
            _profile(tmp_path, "fine_grained"),
            runner=_successful_result,
            fine_runner=fine_runner,
        ).execute(_reference_task(tmp_path), (chunked,))

    assert captured.value.original.args == ("injected fine interruption",)  # type: ignore[attr-defined]
    assert tuple(result.round for result in captured.value.completed_results) == (1,)  # type: ignore[attr-defined]
    assert tuple(  # type: ignore[attr-defined]
        result.key.encoded for result in captured.value.fine_task_results
    ) == ("round-002:preparation:main",)
