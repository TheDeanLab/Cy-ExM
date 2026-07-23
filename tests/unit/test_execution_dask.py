from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from cyexm.config.models import (
    ExecutionPaths,
    ExecutionProfile,
    LocalDaskSpec,
    NativeThreadSpec,
    NonlinearRegistrationStageSpec,
    ParallelismSpec,
    RegistrationSpec,
    RegistrationStageSpec,
)
from cyexm.errors import ExecutorStartupError, RoundExecutionError
from cyexm.execution import dask as dask_module
from cyexm.execution._dask_common import collect_distributed_fine_results
from cyexm.execution.dask import LocalDaskExecutor
from cyexm.execution.dispatcher import FineGraphController
from cyexm.runtime import NATIVE_THREAD_ENV_VARS
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

pytestmark = pytest.mark.unit


def test_real_module_loader_gets_timeout_parser_from_dask_utils(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    distributed = ModuleType("distributed")
    distributed.Client = object  # type: ignore[attr-defined]
    distributed.LocalCluster = object  # type: ignore[attr-defined]
    distributed.as_completed = object()  # type: ignore[attr-defined]
    distributed_utils = ModuleType("distributed.utils")
    dask = ModuleType("dask")
    dask.__path__ = []  # type: ignore[attr-defined]
    dask_config = ModuleType("dask.config")
    config_set = object()
    dask_config.set = config_set  # type: ignore[attr-defined]
    dask.config = dask_config  # type: ignore[attr-defined]
    dask_utils = ModuleType("dask.utils")
    timeout_parser = object()
    dask_utils.parse_timedelta = timeout_parser  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "distributed", distributed)
    monkeypatch.setitem(sys.modules, "distributed.utils", distributed_utils)
    monkeypatch.setitem(sys.modules, "dask", dask)
    monkeypatch.setitem(sys.modules, "dask.utils", dask_utils)

    modules = dask_module._distributed_modules()

    assert modules.parse_timedelta is timeout_parser
    assert modules.config_set is config_set


def _profile(
    tmp_path: Path,
    *,
    workers: int = 4,
    retries: int = 0,
    mode: str = "coarse",
) -> ExecutionProfile:
    return ExecutionProfile(
        schema_version=1,
        executor="local_dask",
        paths=ExecutionPaths(
            dataset_root=tmp_path / "dataset",
            output_root=tmp_path / "output",
            worker_local_directory=tmp_path / "scratch",
        ),
        native_threads=NativeThreadSpec(itk=1, blas=1),
        local_dask=LocalDaskSpec(
            workers=workers,
            threads_per_worker=1,
            processes=True,
            memory_limit="8GB",
            startup_timeout="30s",
            task_retries=retries,
        ),
        slurm=None,
        source_path=tmp_path / "execution.yaml",
        parallelism=ParallelismSpec(mode=mode),  # type: ignore[arg-type]
    )


def _task(
    tmp_path: Path,
    round_number: int,
    *,
    dataset_id: str = "dataset-1",
) -> RoundTask:
    round_root = tmp_path / f"round-{round_number:03d}"
    return RoundTask(
        dataset_id=dataset_id,
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
    round_root = tmp_path / "round-001"
    source = tmp_path / "reference-source.tif"
    output = round_root / "channels" / "reference.tif"
    return ReferenceTask(
        dataset_id="dataset-1",
        round=1,
        registration_channel_source=source,
        registration_channel_output=output,
        registration_directory=round_root / "registration",
        crop_bounds_path=None,
        log_path=round_root / "logs" / "round-001.log",
        voxel_spacing_xyz=(1.0, 1.0, 1.0),
        crop=False,
        native_threads=NativeThreadSpec(itk=2, blas=3),
        channels=(ReferenceChannelTask(source=source, output=output),),
    )


def _chunked_task(tmp_path: Path, round_number: int) -> RoundTask:
    return replace(
        _task(tmp_path, round_number),
        strategy="serial_chunked",
        chunk_size_zyx=(2, 8, 8),
        overlap_fraction=0.25,
    )


def _prepared_descriptor() -> dict[str, object]:
    identity = {"path": "/tmp/item", "st_size": 1, "st_mtime_ns": 2}
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


def _result(task: RoundTask, *, status: str = "completed") -> RoundResult:
    failure = (
        RoundFailure("round", "RuntimeError", "injected failed result")
        if status == "failed"
        else None
    )
    return RoundResult(
        dataset_id=task.dataset_id,
        round=task.round,
        status=status,  # type: ignore[arg-type]
        started_at="2026-07-20T10:00:00Z",
        ended_at="2026-07-20T10:00:01Z",
        duration_seconds=1.0,
        hostname="worker-01",
        pid=123,
        dask_task_key=None,
        slurm_job_id=None,
        strategy=task.strategy,
        native_threads=(("itk", 2), ("blas", 3)),
        artifact_paths=(
            str(task.registration_directory / f"linear_transform_{task.round}.mat"),
            str(task.registration_directory / f"nonlinear_warp_{task.round}.nii.gz"),
            str(task.registration_directory / f"nonlinear_registered_{task.round}.tif"),
            *(str(channel.output) for channel in task.channels),
        ),
        reused_artifacts=(),
        log_path=str(task.log_path),
        failure=failure,
    )


def _reference_result(task: ReferenceTask, *, status: str = "completed") -> RoundResult:
    failure = (
        RoundFailure("reference", "RuntimeError", "injected failed result")
        if status == "failed"
        else None
    )
    return RoundResult(
        dataset_id=task.dataset_id,
        round=task.round,
        status=status,  # type: ignore[arg-type]
        started_at="2026-07-20T10:00:00Z",
        ended_at="2026-07-20T10:00:01Z",
        duration_seconds=1.0,
        hostname="worker-01",
        pid=123,
        dask_task_key=None,
        slurm_job_id=None,
        strategy="reference-preparation",
        native_threads=(("itk", 2), ("blas", 3)),
        artifact_paths=(str(task.registration_channel_output),),
        reused_artifacts=(),
        log_path=str(task.log_path),
        failure=failure,
    )


def _workflow_result(task: WorkflowTask) -> RoundResult:
    return _reference_result(task) if isinstance(task, ReferenceTask) else _result(task)


def _fine_result(
    task: FineWorkflowTask,
    *,
    status: str = "completed",
    dask_task_key: str | None = "AUTO",
) -> FineTaskResult:
    failure = (
        RoundFailure(task.key.stage, "RuntimeError", "injected failed result")
        if status == "failed"
        else None
    )
    return FineTaskResult(
        task=task,
        status=status,  # type: ignore[arg-type]
        started_at="2026-07-20T10:00:00Z",
        ended_at="2026-07-20T10:00:01Z",
        duration_seconds=1.0,
        hostname="worker-01",
        pid=123,
        dask_task_key=dask_task_key,
        slurm_job_id=None,
        native_threads=tuple(
            (
                name,
                (
                    task.native_threads.itk
                    if name == NATIVE_THREAD_ENV_VARS[0]
                    else task.native_threads.blas
                ),
            )
            for name in NATIVE_THREAD_ENV_VARS
        ),
        artifact_paths=(
            tuple(str(path) for path in task.output_paths)
            if status in {"completed", "reused"}
            else ()
        ),
        reused_artifacts=(),
        log_path=str(task.log_path),
        failure=failure,
        prepared_descriptor=(
            _prepared_descriptor()
            if type(task) is RoundPreparationTask and status in {"completed", "reused"}
            else None
        ),
    )


@dataclass(frozen=True)
class _SubmitCall:
    task: WorkflowTask | FineWorkflowTask
    key: str
    pure: bool
    retries: int
    resources: dict[str, int]


class _FakeFuture:
    def __init__(
        self,
        task: WorkflowTask | FineWorkflowTask,
        key: str,
        outcome: object,
        *,
        done: bool,
    ) -> None:
        self.task = task
        self.key = key
        self.outcome = outcome
        self._done = done

    def done(self) -> bool:
        return self._done


class _FakeAsCompleted(Iterator[tuple[_FakeFuture, object]]):
    def __init__(
        self,
        modules: _FakeDistributedModules,
        futures: list[_FakeFuture],
        *,
        interrupt_after: int | None,
    ) -> None:
        self._modules = modules
        order = {
            round_number: index for index, round_number in enumerate(modules.order)
        }
        self._queue = sorted(
            futures,
            key=lambda future: order.get(future.task.round, len(order)),
        )
        self._yielded = 0
        self._interrupt_after = interrupt_after

    def __iter__(self) -> _FakeAsCompleted:
        return self

    def __next__(self) -> tuple[_FakeFuture, object]:
        if self._interrupt_after is not None and self._yielded == self._interrupt_after:
            self._interrupt_after = None
            raise KeyboardInterrupt("injected collection interruption")
        if not self._queue:
            raise StopIteration
        self._yielded += 1
        future = self._queue.pop(0)
        future._done = True
        return future, future.outcome

    def add(self, future: _FakeFuture) -> None:
        self._queue.append(future)


class _FakeCluster:
    def __init__(self, modules: _FakeDistributedModules) -> None:
        self._modules = modules

    def close(self) -> None:
        self._modules.cluster_close_config_active = self._modules.config_context_active
        self._modules.events.append("cluster.close")
        if self._modules.cluster_close_error is not None:
            raise self._modules.cluster_close_error


class _FakeClient:
    def __init__(self, modules: _FakeDistributedModules) -> None:
        self._modules = modules

    def wait_for_workers(self, workers: int, *, timeout: float) -> None:
        self._modules.wait_call = (workers, timeout)
        if self._modules.wait_error is not None:
            raise self._modules.wait_error

    def submit(
        self,
        runner: Callable[..., object],
        task: WorkflowTask | FineWorkflowTask,
        *,
        key: str,
        pure: bool,
        retries: int,
        resources: dict[str, int],
    ) -> _FakeFuture:
        del runner
        self._modules.submission_config_active.append(
            self._modules.config_context_active
        )
        self._modules.submit_calls.append(
            _SubmitCall(task, key, pure, retries, resources)
        )
        identity = task.key.encoded if hasattr(task, "key") else task.round
        configured = self._modules.outcomes.setdefault(identity, [])
        outcome = (
            configured.pop(0)
            if configured
            else (
                _fine_result(task) if hasattr(task, "key") else _workflow_result(task)
            )
        )
        if type(outcome) is FineTaskResult and outcome.dask_task_key == "AUTO":
            outcome = replace(outcome, dask_task_key=key)
        return _FakeFuture(
            task,
            key,
            outcome,
            done=task.round not in self._modules.pending_rounds,
        )

    def cancel(self, futures: tuple[_FakeFuture, ...], *, force: bool) -> None:
        self._modules.cancel_call = (futures, force)
        if self._modules.cancel_error is not None:
            raise self._modules.cancel_error

    def close(self) -> None:
        self._modules.client_close_config_active = self._modules.config_context_active
        self._modules.events.append("client.close")
        if self._modules.client_close_error is not None:
            raise self._modules.client_close_error


class _FakeDistributedModules:
    def __init__(self) -> None:
        self.cluster_kwargs: dict[str, object] | None = None
        self.client_timeout: str | None = None
        self.wait_call: tuple[int, float] | None = None
        self.as_completed_call: tuple[bool, bool] | None = None
        self.as_completed_initial_counts: list[int] = []
        self.submit_calls: list[_SubmitCall] = []
        self.events: list[str] = []
        self.cancel_call: tuple[tuple[_FakeFuture, ...], bool] | None = None
        self.outcomes: dict[int | str, list[object]] = {}
        self.order: tuple[int, ...] = ()
        self.interrupt_after: int | None = None
        self.interrupt_as_completed_call: int | None = None
        self.cluster_error: Exception | None = None
        self.client_error: Exception | None = None
        self.wait_error: Exception | None = None
        self.pending_rounds: set[int] = set()
        self.cancel_error: BaseException | None = None
        self.client_close_error: BaseException | None = None
        self.cluster_close_error: BaseException | None = None
        self.config_exit_error: BaseException | None = None
        self.config_overlays: list[dict[str, object]] = []
        self.config_context_active = False
        self.config_exit_info: list[tuple[object, object, object]] = []
        self.cluster_config_active: bool | None = None
        self.submission_config_active: list[bool] = []
        self.client_close_config_active: bool | None = None
        self.cluster_close_config_active: bool | None = None

    class _ConfigContext(AbstractContextManager[None]):
        def __init__(
            self,
            modules: _FakeDistributedModules,
            values: dict[str, object],
        ) -> None:
            self._modules = modules
            self._values = values

        def __enter__(self) -> None:
            self._modules.config_overlays.append(self._values)
            self._modules.config_context_active = True

        def __exit__(
            self,
            exception_type: object,
            exception: object,
            traceback: object,
        ) -> None:
            self._modules.config_exit_info.append(
                (exception_type, exception, traceback)
            )
            self._modules.config_context_active = False
            if self._modules.config_exit_error is not None:
                raise self._modules.config_exit_error

    def config_set(self, values: dict[str, object]) -> AbstractContextManager[None]:
        return self._ConfigContext(self, values)

    def LocalCluster(self, **kwargs: object) -> _FakeCluster:
        self.cluster_config_active = self.config_context_active
        if self.cluster_error is not None:
            raise self.cluster_error
        self.cluster_kwargs = kwargs
        return _FakeCluster(self)

    def Client(self, cluster: _FakeCluster, *, timeout: str) -> _FakeClient:
        del cluster
        self.client_timeout = timeout
        if self.client_error is not None:
            raise self.client_error
        return _FakeClient(self)

    def as_completed(
        self,
        futures: list[_FakeFuture],
        *,
        with_results: bool,
        raise_errors: bool,
    ) -> _FakeAsCompleted:
        self.as_completed_call = (with_results, raise_errors)
        self.as_completed_initial_counts.append(len(futures))
        call_number = len(self.as_completed_initial_counts)
        interrupt_after = (
            self.interrupt_after
            if self.interrupt_as_completed_call in {None, call_number}
            else None
        )
        return _FakeAsCompleted(
            self,
            futures,
            interrupt_after=interrupt_after,
        )

    def parse_timedelta(self, value: str) -> float:
        assert value == "30s"
        return 30.0


def test_local_executor_uses_one_round_slot_per_bounded_worker(tmp_path: Path) -> None:
    modules = _FakeDistributedModules()
    executor = LocalDaskExecutor(_profile(tmp_path), _modules=lambda: modules)

    results = executor.execute(
        _reference_task(tmp_path), (_task(tmp_path, 2), _task(tmp_path, 3))
    )

    assert modules.cluster_kwargs == {
        "n_workers": 2,
        "threads_per_worker": 1,
        "processes": True,
        "memory_limit": "8GB",
        "local_directory": str(executor.profile.paths.worker_local_directory),
        "resources": {"round_slot": 1},
    }
    assert modules.client_timeout == "30s"
    assert modules.wait_call == (2, 30.0)
    assert modules.as_completed_call == (True, False)
    assert [call.resources for call in modules.submit_calls] == [
        {"round_slot": 1},
        {"round_slot": 1},
        {"round_slot": 1},
    ]
    assert tuple(result.round for result in results) == (1, 2, 3)


def test_fine_local_executor_starts_exact_configured_workers_and_expands_on_coordinator(
    tmp_path: Path,
) -> None:
    modules = _FakeDistributedModules()
    task = _task(tmp_path, 2)

    results = LocalDaskExecutor(
        _profile(tmp_path, workers=3, mode="fine_grained"),
        _modules=lambda: modules,
    ).execute(_reference_task(tmp_path), (task,))

    assert modules.cluster_kwargs is not None
    assert modules.cluster_kwargs["n_workers"] == 3
    assert modules.wait_call == (3, 30.0)
    assert [call.task.round for call in modules.submit_calls] == [1, 2, 2, 2]
    fine_calls = modules.submit_calls[1:]
    assert [call.task.key.stage for call in fine_calls] == [  # type: ignore[union-attr]
        "whole-image",
        "channel",
        "validation",
    ]
    assert [call.key for call in fine_calls] == [
        "cyexm-round-002-whole-image-main",
        "cyexm-round-002-channel-0000",
        "cyexm-round-002-validation-main",
    ]
    assert all(call.resources == {"round_slot": 1} for call in fine_calls)
    assert all(call.pure is False and call.retries == 0 for call in fine_calls)
    assert tuple(result.round for result in results) == (1, 2)
    assert results[1].status == "completed"


def test_fine_collector_queries_scheduler_neutral_capacity_after_each_transition(
    tmp_path: Path,
) -> None:
    modules = _FakeDistributedModules()
    controller = FineGraphController((_task(tmp_path, 2), _task(tmp_path, 3)))
    client = _FakeClient(modules)
    observations: list[tuple[int, int]] = []

    def capacity_provider(graph: FineGraphController, running: int) -> int:
        observations.append((running, graph.ready_count))
        return 2

    results, task_results, _attempts = collect_distributed_fine_results(
        controller,
        runner=_fine_result,
        client=client,
        as_completed=modules.as_completed,
        task_retries=0,
        capacity=4,
        active=set(),
        capacity_provider=capacity_provider,
    )

    assert tuple(result.round for result in results) == (2, 3)
    assert task_results
    assert observations[0] == (0, 2)
    assert len(observations) == len(task_results) + 1
    assert all(running <= 2 and ready >= 0 for running, ready in observations)


def test_fine_local_executor_retries_exact_task_without_recording_intermediate_failure(
    tmp_path: Path,
) -> None:
    modules = _FakeDistributedModules()
    task = _task(tmp_path, 2)
    key = "round-002:whole-image:main"
    from cyexm.workflow.fine_planning import initial_fine_tasks

    fine_task = initial_fine_tasks((task,))[0]
    modules.outcomes[key] = [
        _fine_result(fine_task, status="failed"),
        _fine_result(fine_task),
    ]

    results = LocalDaskExecutor(
        _profile(tmp_path, workers=2, retries=1, mode="fine_grained"),
        _modules=lambda: modules,
    ).execute(_reference_task(tmp_path), (task,))

    assert results[1].status == "completed"
    assert [call.key for call in modules.submit_calls[1:3]] == [
        "cyexm-round-002-whole-image-main",
        "cyexm-round-002-whole-image-main-attempt-002",
    ]


def test_fine_local_executor_bounds_initial_work_and_settles_independent_sibling(
    tmp_path: Path,
) -> None:
    modules = _FakeDistributedModules()
    first = _task(tmp_path, 2)
    second = _task(tmp_path, 3)
    from cyexm.workflow.fine_planning import initial_fine_tasks

    failed = initial_fine_tasks((first,))[0]
    modules.outcomes[failed.key.encoded] = [_fine_result(failed, status="failed")]

    reference, first_result, second_result = LocalDaskExecutor(
        _profile(tmp_path, workers=2, mode="fine_grained"),
        _modules=lambda: modules,
    ).execute(_reference_task(tmp_path), (first, second))

    assert modules.as_completed_initial_counts == [1, 2]
    assert [call.task.round for call in modules.submit_calls[1:3]] == [2, 3]
    assert first_result.status == "failed"
    assert second_result.status == "completed"
    assert reference.status == "completed"
    assert not any(
        hasattr(call.task, "key")
        and call.task.round == 2
        and call.task.key.stage == "channel"
        for call in modules.submit_calls
    )


@pytest.mark.parametrize(
    "outcome",
    [
        RuntimeError("raised by worker"),
        (RuntimeError, "not an exception", None),
        object(),
    ],
)
def test_fine_local_executor_structures_worker_and_malformed_failures(
    tmp_path: Path,
    outcome: object,
) -> None:
    modules = _FakeDistributedModules()
    task = _task(tmp_path, 2)
    from cyexm.workflow.fine_planning import initial_fine_tasks

    initial = initial_fine_tasks((task,))[0]
    modules.outcomes[initial.key.encoded] = [outcome]

    result = LocalDaskExecutor(
        _profile(tmp_path, workers=2, mode="fine_grained"),
        _modules=lambda: modules,
    ).execute(_reference_task(tmp_path), (task,))[1]

    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.stage == "whole-image"


def test_fine_local_executor_converts_mismatched_exact_result_to_submitted_task(
    tmp_path: Path,
) -> None:
    modules = _FakeDistributedModules()
    submitted_round = _task(tmp_path, 2)
    other_round = _task(tmp_path, 3)
    from cyexm.workflow.fine_planning import initial_fine_tasks

    submitted = initial_fine_tasks((submitted_round,))[0]
    mismatched = initial_fine_tasks((other_round,))[0]
    modules.outcomes[submitted.key.encoded] = [_fine_result(mismatched)]

    result = LocalDaskExecutor(
        _profile(tmp_path, mode="fine_grained"),
        _modules=lambda: modules,
    ).execute(_reference_task(tmp_path), (submitted_round,))[1]

    assert result.round == 2
    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.stage == "whole-image"
    assert result.failure.exception_type == "RoundExecutionError"
    assert "does not match the submitted task" in result.failure.message


@pytest.mark.parametrize(
    "malformed",
    [
        lambda value: replace(value, dask_task_key=None),
        lambda value: replace(value, dask_task_key="cyexm-wrong-attempt-999"),
        lambda value: replace(
            value,
            native_threads=(("itk", 2), ("blas", 3)),
        ),
        lambda value: replace(
            value,
            native_threads=tuple(reversed(value.native_threads)),
        ),
        lambda value: replace(
            value,
            native_threads=(
                *value.native_threads[:-1],
                (value.native_threads[-1][0], 999),
            ),
        ),
        lambda value: replace(value, started_at="not-a-timestamp"),
        lambda value: replace(value, started_at="2026-07-20T10:00:00"),
        lambda value: replace(
            value,
            started_at="2026-07-20T10:00:02Z",
            ended_at="2026-07-20T10:00:01Z",
        ),
        lambda value: replace(
            value,
            dask_task_key="cyexm-wrong-attempt-999",
            native_threads=(("itk", 999), ("blas", 999)),
            started_at="not-a-timestamp",
        ),
    ],
)
def test_fine_local_executor_rejects_matching_task_with_malformed_evidence(
    tmp_path: Path,
    malformed: Callable[[FineTaskResult], FineTaskResult],
) -> None:
    modules = _FakeDistributedModules()
    task = _task(tmp_path, 2)
    from cyexm.workflow.fine_planning import initial_fine_tasks

    initial = initial_fine_tasks((task,))[0]
    expanded = tuple(
        (
            name,
            2 if index == 0 else 3,
        )
        for index, name in enumerate(NATIVE_THREAD_ENV_VARS)
    )
    outcome = replace(_fine_result(initial), native_threads=expanded)
    modules.outcomes[initial.key.encoded] = [malformed(outcome)]

    executor = LocalDaskExecutor(
        _profile(tmp_path, mode="fine_grained"),
        _modules=lambda: modules,
    )
    result = executor.execute(_reference_task(tmp_path), (task,))[1]

    assert result.round == 2
    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.stage == "whole-image"
    assert not any(
        hasattr(call.task, "key") and call.task.key.stage == "channel"
        for call in modules.submit_calls
    )
    assert len(executor._fine_task_results) == 1
    assert executor._fine_task_results[0].status == "failed"


def test_fine_local_executor_retries_malformed_evidence_before_descendants(
    tmp_path: Path,
) -> None:
    modules = _FakeDistributedModules()
    task = _task(tmp_path, 2)
    from cyexm.workflow.fine_planning import initial_fine_tasks

    initial = initial_fine_tasks((task,))[0]
    first = replace(
        _fine_result(initial),
        dask_task_key="cyexm-wrong-attempt-999",
    )
    modules.outcomes[initial.key.encoded] = [first, _fine_result(initial)]

    executor = LocalDaskExecutor(
        _profile(tmp_path, mode="fine_grained", retries=1),
        _modules=lambda: modules,
    )
    result = executor.execute(_reference_task(tmp_path), (task,))[1]

    assert result.status == "completed"
    assert [call.task.key.stage for call in modules.submit_calls[1:]] == [  # type: ignore[union-attr]
        "whole-image",
        "whole-image",
        "channel",
        "validation",
    ]
    assert executor._fine_attempt_counts[0] == (
        initial.key.encoded,
        2,
    )
    evidence = executor.execution_evidence
    assert type(evidence) is _FineExecutionEvidence
    assert evidence.executor == "local_dask"
    assert evidence.worker_cap == 4
    assert evidence.attempt_counts[0] == (initial.key.encoded, 2)
    assert tuple(result.key.encoded for result in evidence.task_results) == tuple(
        key for key, _dependencies in evidence.task_graph
    )
    assert evidence.cleanup == (
        ("client", True, None, None),
        ("cluster", True, None, None),
        ("configuration", True, None, None),
    )


def test_fine_local_executor_rejects_bad_prep_descriptor_and_settles_sibling(
    tmp_path: Path,
) -> None:
    modules = _FakeDistributedModules()
    chunked = _chunked_task(tmp_path, 2)
    sibling = _task(tmp_path, 3)
    from cyexm.workflow.fine_planning import initial_fine_tasks

    preparation = initial_fine_tasks((chunked,))[0]
    assert type(preparation) is RoundPreparationTask
    modules.outcomes[preparation.key.encoded] = [
        replace(
            _fine_result(preparation),
            prepared_descriptor={"bad": "descriptor"},
        )
    ]

    executor = LocalDaskExecutor(
        _profile(tmp_path, workers=2, mode="fine_grained"),
        _modules=lambda: modules,
    )
    reference, chunked_result, sibling_result = executor.execute(
        _reference_task(tmp_path),
        (chunked, sibling),
    )

    assert reference.status == "completed"
    assert chunked_result.status == "failed"
    assert sibling_result.status == "completed"
    assert chunked_result.failure is not None
    assert chunked_result.failure.stage == "preparation"
    assert len(executor._fine_task_results) == 4
    preparation_result = next(
        result
        for result in executor._fine_task_results
        if result.task.key == preparation.key
    )
    assert preparation_result.status == "failed"
    assert preparation_result.dask_task_key == "cyexm-round-002-preparation-main"
    assert not any(
        hasattr(call.task, "key")
        and call.task.round == 2
        and call.task.key.stage == "chunk"
        for call in modules.submit_calls
    )


def test_fine_local_executor_retries_malformed_preparation_descriptor_before_chunks(
    tmp_path: Path,
) -> None:
    modules = _FakeDistributedModules()
    chunked = _chunked_task(tmp_path, 2)
    from cyexm.workflow.fine_planning import initial_fine_tasks

    preparation = initial_fine_tasks((chunked,))[0]
    assert type(preparation) is RoundPreparationTask
    modules.outcomes[preparation.key.encoded] = [
        replace(
            _fine_result(preparation),
            prepared_descriptor={"bad": "descriptor"},
        ),
        _fine_result(preparation),
    ]

    executor = LocalDaskExecutor(
        _profile(tmp_path, workers=2, retries=1, mode="fine_grained"),
        _modules=lambda: modules,
    )
    result = executor.execute(_reference_task(tmp_path), (chunked,))[1]

    assert result.status == "completed"
    round_calls = [
        call for call in modules.submit_calls if call.task.round == chunked.round
    ]
    assert [call.task.key.stage for call in round_calls[:2]] == [  # type: ignore[union-attr]
        "preparation",
        "preparation",
    ]
    assert all(  # type: ignore[union-attr]
        call.task.key.stage != "chunk" for call in round_calls[:2]
    )
    assert any(  # type: ignore[union-attr]
        call.task.key.stage == "chunk" for call in round_calls[2:]
    )
    assert executor._fine_attempt_counts[0] == (
        preparation.key.encoded,
        2,
    )


def test_fine_local_executor_exhausts_retries_for_malformed_evidence(
    tmp_path: Path,
) -> None:
    modules = _FakeDistributedModules()
    task = _task(tmp_path, 2)
    from cyexm.workflow.fine_planning import initial_fine_tasks

    initial = initial_fine_tasks((task,))[0]
    wrong = replace(_fine_result(initial), dask_task_key="wrong-attempt")
    modules.outcomes[initial.key.encoded] = [wrong, wrong]

    executor = LocalDaskExecutor(
        _profile(tmp_path, mode="fine_grained", retries=1),
        _modules=lambda: modules,
    )
    result = executor.execute(_reference_task(tmp_path), (task,))[1]

    assert result.status == "failed"
    assert executor._fine_attempt_counts == ((initial.key.encoded, 2),)
    assert len(executor._fine_task_results) == 1
    assert executor._fine_task_results[0].dask_task_key == (
        "cyexm-round-002-whole-image-main-attempt-002"
    )


def test_fine_local_executor_cancelled_result_blocks_only_descendants(
    tmp_path: Path,
) -> None:
    modules = _FakeDistributedModules()
    cancelled_round = _task(tmp_path, 2)
    sibling_round = _task(tmp_path, 3)
    from cyexm.workflow.fine_planning import initial_fine_tasks

    cancelled = initial_fine_tasks((cancelled_round,))[0]
    modules.outcomes[cancelled.key.encoded] = [
        _fine_result(cancelled, status="cancelled")
    ]

    reference, cancelled_result, sibling_result = LocalDaskExecutor(
        _profile(tmp_path, workers=2, mode="fine_grained"),
        _modules=lambda: modules,
    ).execute(_reference_task(tmp_path), (cancelled_round, sibling_round))

    assert reference.status == "completed"
    assert cancelled_result.status == "cancelled"
    assert sibling_result.status == "completed"
    assert not any(
        hasattr(call.task, "key")
        and call.task.round == 2
        and call.task.key.stage == "channel"
        for call in modules.submit_calls
    )


def test_fine_local_executor_settles_reverse_order_multiple_failures(
    tmp_path: Path,
) -> None:
    modules = _FakeDistributedModules()
    modules.order = (3, 2)
    first = _task(tmp_path, 2)
    second = _task(tmp_path, 3)
    from cyexm.workflow.fine_planning import initial_fine_tasks

    first_task, second_task = initial_fine_tasks((first, second))
    modules.outcomes[first_task.key.encoded] = [
        _fine_result(first_task, status="failed")
    ]
    modules.outcomes[second_task.key.encoded] = [
        _fine_result(second_task, status="failed")
    ]

    results = LocalDaskExecutor(
        _profile(tmp_path, workers=2, mode="fine_grained"),
        _modules=lambda: modules,
    ).execute(_reference_task(tmp_path), (first, second))

    assert tuple(result.round for result in results) == (1, 2, 3)
    assert tuple(result.status for result in results) == (
        "completed",
        "failed",
        "failed",
    )


def test_fine_local_executor_preserves_dynamic_cross_round_fairness(
    tmp_path: Path,
) -> None:
    modules = _FakeDistributedModules()
    first = _task(tmp_path, 2)
    second = _task(tmp_path, 3)

    LocalDaskExecutor(
        _profile(tmp_path, workers=2, mode="fine_grained"),
        _modules=lambda: modules,
    ).execute(_reference_task(tmp_path), (first, second))

    moving_calls = modules.submit_calls[1:]
    assert [(call.task.round, call.task.key.stage) for call in moving_calls] == [  # type: ignore[union-attr]
        (2, "whole-image"),
        (3, "whole-image"),
        (2, "channel"),
        (3, "channel"),
        (2, "validation"),
        (3, "validation"),
    ]


def test_fine_local_executor_cancels_unfinished_stage_on_interrupt(
    tmp_path: Path,
) -> None:
    modules = _FakeDistributedModules()
    modules.interrupt_after = 0
    modules.interrupt_as_completed_call = 2
    modules.pending_rounds = {2}
    executor = LocalDaskExecutor(
        _profile(tmp_path, workers=2, mode="fine_grained"),
        _modules=lambda: modules,
    )

    with pytest.raises(KeyboardInterrupt, match="collection interruption"):
        executor.execute(_reference_task(tmp_path), (_task(tmp_path, 2),))

    assert modules.cancel_call is not None
    cancelled, force = modules.cancel_call
    assert len(cancelled) == 1
    assert cancelled[0].task.key.stage == "whole-image"  # type: ignore[union-attr]
    assert force is True
    assert modules.events == ["client.close", "cluster.close"]
    evidence = executor.execution_evidence
    assert evidence.cancelled
    assert evidence.attempt_counts == (("round-002:whole-image:main", 1),)
    assert evidence.task_results == ()
    assert evidence.cleanup == (
        ("client", True, None, None),
        ("cluster", True, None, None),
        ("configuration", True, None, None),
    )


def test_local_executor_owns_worker_ttl_config_for_cluster_and_execution(
    tmp_path: Path,
) -> None:
    modules = _FakeDistributedModules()

    LocalDaskExecutor(_profile(tmp_path), _modules=lambda: modules).execute(
        _reference_task(tmp_path), (_task(tmp_path, 2),)
    )

    assert modules.config_overlays == [{"distributed.scheduler.worker-ttl": None}]
    assert modules.cluster_config_active is True
    assert modules.submission_config_active == [True, True]
    assert modules.client_close_config_active is True
    assert modules.cluster_close_config_active is True
    assert modules.config_context_active is False
    assert len(modules.config_exit_info) == 1


def test_local_executor_submits_deterministic_non_pure_zero_retry_keys(
    tmp_path: Path,
) -> None:
    modules = _FakeDistributedModules()
    task = _task(tmp_path, 7, dataset_id="sample_A.1")

    LocalDaskExecutor(_profile(tmp_path), _modules=lambda: modules).execute(
        _reference_task(tmp_path), (task,)
    )

    assert modules.submit_calls[0].key == "cyexm-dataset-1-reference-round-001"
    call = modules.submit_calls[1]
    assert call.key == "cyexm-sample_A.1-round-007"
    assert call.pure is False
    assert call.retries == 0


def test_local_executor_runs_reference_when_no_moving_rounds_are_pending(
    tmp_path: Path,
) -> None:
    loaded = False

    def modules() -> _FakeDistributedModules:
        nonlocal loaded
        loaded = True
        return _FakeDistributedModules()

    results = LocalDaskExecutor(_profile(tmp_path), _modules=modules).execute(
        _reference_task(tmp_path), ()
    )

    assert tuple(result.round for result in results) == (1,)
    assert loaded is True


def test_local_executor_collects_independently_but_returns_plan_order(
    tmp_path: Path,
) -> None:
    modules = _FakeDistributedModules()
    modules.order = (3, 2)

    results = LocalDaskExecutor(_profile(tmp_path), _modules=lambda: modules).execute(
        _reference_task(tmp_path), (_task(tmp_path, 2), _task(tmp_path, 3))
    )

    assert tuple(result.round for result in results) == (1, 2, 3)


def test_local_executor_does_not_submit_moving_rounds_after_reference_failure(
    tmp_path: Path,
) -> None:
    modules = _FakeDistributedModules()
    reference = _reference_task(tmp_path)
    modules.outcomes[1] = [_reference_result(reference, status="failed")]

    results = LocalDaskExecutor(_profile(tmp_path), _modules=lambda: modules).execute(
        reference, (_task(tmp_path, 2), _task(tmp_path, 3))
    )

    assert tuple(result.round for result in results) == (1,)
    assert results[0].status == "failed"
    assert [call.task.round for call in modules.submit_calls] == [1]


def test_local_executor_resubmits_failed_results_with_a_unique_bounded_key(
    tmp_path: Path,
) -> None:
    modules = _FakeDistributedModules()
    task = _task(tmp_path, 2)
    modules.outcomes[2] = [_result(task, status="failed"), _result(task)]

    result = LocalDaskExecutor(
        _profile(tmp_path, retries=1), _modules=lambda: modules
    ).execute(_reference_task(tmp_path), (task,))

    assert result[1].status == "completed"
    assert [call.key for call in modules.submit_calls] == [
        "cyexm-dataset-1-reference-round-001",
        "cyexm-dataset-1-round-002",
        "cyexm-dataset-1-round-002-attempt-002",
    ]
    assert all(call.retries == 0 for call in modules.submit_calls)


def test_local_executor_resubmits_future_exceptions_and_structures_exhaustion(
    tmp_path: Path,
) -> None:
    modules = _FakeDistributedModules()
    task = _task(tmp_path, 2)
    modules.outcomes[2] = [
        (ValueError, ValueError("first"), None),
        (RuntimeError, RuntimeError("second"), None),
    ]

    result = LocalDaskExecutor(
        _profile(tmp_path, retries=1), _modules=lambda: modules
    ).execute(_reference_task(tmp_path), (task,))[1]

    assert len(modules.submit_calls) == 3
    assert result.status == "failed"
    assert result.round == 2
    assert result.dask_task_key == "cyexm-dataset-1-round-002-attempt-002"
    assert result.hostname == "unknown"
    assert result.pid > 0
    assert result.failure == RoundFailure(
        stage="round",
        exception_type="RuntimeError",
        message="second",
    )


@pytest.mark.parametrize(
    "outcome",
    [
        (RuntimeError, "not an exception", None),
        ("RuntimeError", RuntimeError("bad type"), None),
        (RuntimeError, RuntimeError("missing traceback slot")),
    ],
)
def test_local_executor_defensively_structures_malformed_error_payloads(
    tmp_path: Path,
    outcome: object,
) -> None:
    modules = _FakeDistributedModules()
    modules.outcomes[2] = [outcome]

    result = LocalDaskExecutor(_profile(tmp_path), _modules=lambda: modules).execute(
        _reference_task(tmp_path), (_task(tmp_path, 2),)
    )[1]

    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.exception_type == "TypeError"
    assert "malformed distributed error payload" in result.failure.message
    assert result.pid > 0


class _RoundResultSubclass(RoundResult):
    pass


@pytest.mark.parametrize(
    "malformed",
    [
        lambda value: replace(value, dataset_id="wrong-dataset"),
        lambda value: replace(value, round=99),
        lambda value: replace(value, native_threads=(("itk", 999), ("blas", 3))),
        lambda value: replace(value, pid=0),
        lambda value: replace(
            value,
            started_at="2026-07-20T10:00:02Z",
            ended_at="2026-07-20T10:00:01Z",
        ),
        lambda value: _RoundResultSubclass(
            **{
                field: getattr(value, field)
                for field in RoundResult.__dataclass_fields__
            }
        ),
    ],
)
def test_local_executor_rejects_malformed_or_wrong_task_round_results(
    tmp_path: Path,
    malformed: Callable[[RoundResult], RoundResult],
) -> None:
    modules = _FakeDistributedModules()
    task = _task(tmp_path, 2)
    modules.outcomes[2] = [malformed(_result(task))]

    result = LocalDaskExecutor(_profile(tmp_path), _modules=lambda: modules).execute(
        _reference_task(tmp_path), (task,)
    )[1]

    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.exception_type == "RoundExecutionError"


def test_local_executor_validates_tasks_before_loading_scheduler_modules(
    tmp_path: Path,
) -> None:
    loaded = False

    def modules() -> _FakeDistributedModules:
        nonlocal loaded
        loaded = True
        return _FakeDistributedModules()

    malformed = replace(_task(tmp_path, 2), round=True)
    with pytest.raises(RoundExecutionError, match=r"tasks\[1\].*round"):
        LocalDaskExecutor(_profile(tmp_path), _modules=modules).execute(
            _reference_task(tmp_path), (malformed,)
        )

    assert loaded is False


def test_local_executor_rejects_duplicate_task_identities_before_cluster_mutation(
    tmp_path: Path,
) -> None:
    loaded = False

    def modules() -> _FakeDistributedModules:
        nonlocal loaded
        loaded = True
        return _FakeDistributedModules()

    task = _task(tmp_path, 2)
    with pytest.raises(RoundExecutionError, match="duplicate task identity"):
        LocalDaskExecutor(_profile(tmp_path), _modules=modules).execute(
            _reference_task(tmp_path), (task, task)
        )

    assert loaded is False


def test_local_executor_bookkeeping_uses_task_identity_not_round_only(
    tmp_path: Path,
) -> None:
    modules = _FakeDistributedModules()
    first = _task(tmp_path, 2, dataset_id="dataset-1")
    second = _task(tmp_path, 2, dataset_id="dataset-2")

    results = LocalDaskExecutor(_profile(tmp_path), _modules=lambda: modules).execute(
        _reference_task(tmp_path), (first, second)
    )

    assert tuple(result.dataset_id for result in results) == (
        "dataset-1",
        "dataset-1",
        "dataset-2",
    )


def test_local_executor_does_not_cancel_other_rounds_after_terminal_failure(
    tmp_path: Path,
) -> None:
    modules = _FakeDistributedModules()
    first = _task(tmp_path, 2)
    modules.outcomes[2] = [_result(first, status="failed")]

    reference, failed, completed = LocalDaskExecutor(
        _profile(tmp_path), _modules=lambda: modules
    ).execute(_reference_task(tmp_path), (first, _task(tmp_path, 3)))

    assert reference.status == "completed"
    assert failed.status == "failed"
    assert completed.status == "completed"
    assert modules.cancel_call is None


@pytest.mark.parametrize("failure_point", ["cluster", "client", "wait"])
def test_local_executor_wraps_startup_failures_and_cleans_owned_resources(
    tmp_path: Path,
    failure_point: str,
) -> None:
    modules = _FakeDistributedModules()
    setattr(modules, f"{failure_point}_error", RuntimeError(f"{failure_point} failed"))

    with pytest.raises(
        ExecutorStartupError,
        match=r"2 local Dask workers.*30s",
    ) as captured:
        LocalDaskExecutor(_profile(tmp_path), _modules=lambda: modules).execute(
            _reference_task(tmp_path), (_task(tmp_path, 2), _task(tmp_path, 3))
        )

    assert isinstance(captured.value.__cause__, RuntimeError)
    expected = {
        "cluster": [],
        "client": ["cluster.close"],
        "wait": ["client.close", "cluster.close"],
    }
    assert modules.events == expected[failure_point]
    assert modules.config_overlays == [{"distributed.scheduler.worker-ttl": None}]
    assert modules.config_context_active is False
    assert len(modules.config_exit_info) == 1


def test_local_executor_closes_client_before_cluster_on_success(tmp_path: Path) -> None:
    modules = _FakeDistributedModules()

    LocalDaskExecutor(_profile(tmp_path), _modules=lambda: modules).execute(
        _reference_task(tmp_path), (_task(tmp_path, 2),)
    )

    assert modules.events == ["client.close", "cluster.close"]


def test_local_executor_logs_cleanup_failures_without_losing_results(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    modules = _FakeDistributedModules()
    modules.client_close_error = RuntimeError("client cleanup failed")
    modules.cluster_close_error = RuntimeError("cluster cleanup failed")
    executor = LocalDaskExecutor(
        _profile(tmp_path, mode="fine_grained"),
        _modules=lambda: modules,
    )

    with caplog.at_level(logging.WARNING, logger="cyexm.execution.dask"):
        results = executor.execute(_reference_task(tmp_path), (_task(tmp_path, 2),))

    assert results[0].status == "completed"
    assert modules.events == ["client.close", "cluster.close"]
    assert "client cleanup failed" in caplog.text
    assert "cluster cleanup failed" in caplog.text
    assert executor.execution_evidence.cleanup == (
        ("client", False, "RuntimeError", "client cleanup failed"),
        ("cluster", False, "RuntimeError", "cluster cleanup failed"),
        ("configuration", True, None, None),
    )


def test_local_executor_cancels_only_unfinished_futures_on_keyboard_interrupt(
    tmp_path: Path,
) -> None:
    modules = _FakeDistributedModules()
    modules.order = (2, 3, 4)
    modules.interrupt_after = 2
    modules.pending_rounds = {4}

    with pytest.raises(KeyboardInterrupt, match="collection interruption"):
        LocalDaskExecutor(_profile(tmp_path), _modules=lambda: modules).execute(
            _reference_task(tmp_path),
            (_task(tmp_path, 2), _task(tmp_path, 3), _task(tmp_path, 4)),
        )

    assert modules.cancel_call is not None
    cancelled, force = modules.cancel_call
    assert {future.task.round for future in cancelled} == {4}
    assert force is True
    assert modules.events == ["client.close", "cluster.close"]
    assert modules.client_close_config_active is True
    assert modules.cluster_close_config_active is True
    assert modules.config_context_active is False
    assert len(modules.config_exit_info) == 1
    assert modules.config_exit_info[0][0] is KeyboardInterrupt


def test_local_executor_preserves_interrupt_when_cancel_and_cleanup_fail(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    modules = _FakeDistributedModules()
    modules.interrupt_after = 0
    modules.pending_rounds = {1}
    modules.cancel_error = SystemExit("cancel failed")
    modules.client_close_error = KeyboardInterrupt("client close failed")
    modules.cluster_close_error = SystemExit("cluster close failed")

    with (
        caplog.at_level(logging.WARNING, logger="cyexm.execution.dask"),
        pytest.raises(KeyboardInterrupt, match="collection interruption"),
    ):
        LocalDaskExecutor(_profile(tmp_path), _modules=lambda: modules).execute(
            _reference_task(tmp_path), (_task(tmp_path, 2),)
        )

    assert "cancel failed" in caplog.text
    assert "client close failed" in caplog.text
    assert "cluster close failed" in caplog.text
    assert modules.events == ["client.close", "cluster.close"]


def test_local_executor_exports_from_execution_namespace() -> None:
    import cyexm.execution

    assert cyexm.execution.LocalDaskExecutor is LocalDaskExecutor
    assert "LocalDaskExecutor" in cyexm.execution.__all__


def test_local_executor_rejects_non_local_profiles_before_module_loading(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    invalid = ExecutionProfile(
        schema_version=profile.schema_version,
        executor="sequential",
        paths=profile.paths,
        native_threads=profile.native_threads,
        local_dask=None,
        slurm=None,
        source_path=profile.source_path,
    )

    with pytest.raises(ExecutorStartupError, match="requires a local_dask profile"):
        LocalDaskExecutor(invalid).execute(
            _reference_task(tmp_path), (_task(tmp_path, 2),)
        )


def test_local_executor_rejects_missing_local_spec_before_module_loading(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    invalid = ExecutionProfile(
        schema_version=profile.schema_version,
        executor=profile.executor,
        paths=profile.paths,
        native_threads=profile.native_threads,
        local_dask=None,
        slurm=None,
        source_path=profile.source_path,
    )

    with pytest.raises(ExecutorStartupError, match="requires LocalDaskSpec"):
        LocalDaskExecutor(invalid).execute(
            _reference_task(tmp_path), (_task(tmp_path, 2),)
        )


def test_local_executor_wraps_lazy_module_loading_failure(tmp_path: Path) -> None:
    def unavailable() -> Any:
        raise ModuleNotFoundError("distributed")

    with pytest.raises(
        ExecutorStartupError,
        match=r"1 local Dask workers.*30s",
    ) as captured:
        LocalDaskExecutor(_profile(tmp_path), _modules=unavailable).execute(
            _reference_task(tmp_path), (_task(tmp_path, 2),)
        )

    assert isinstance(captured.value.__cause__, ModuleNotFoundError)
