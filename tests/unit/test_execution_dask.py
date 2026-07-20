from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from cyexm.config.models import (
    ExecutionPaths,
    ExecutionProfile,
    LocalDaskSpec,
    NativeThreadSpec,
    RegistrationSpec,
    RegistrationStageSpec,
)
from cyexm.errors import ExecutorStartupError
from cyexm.execution.dask import LocalDaskExecutor
from cyexm.workflow.models import (
    ChannelTask,
    RoundFailure,
    RoundResult,
    RoundTask,
)

pytestmark = pytest.mark.unit


def _profile(
    tmp_path: Path,
    *,
    workers: int = 4,
    retries: int = 0,
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
        artifact_paths=tuple(str(channel.output) for channel in task.channels),
        reused_artifacts=(),
        log_path=str(task.log_path),
        failure=failure,
    )


@dataclass(frozen=True)
class _SubmitCall:
    task: RoundTask
    key: str
    pure: bool
    retries: int
    resources: dict[str, int]


class _FakeFuture:
    def __init__(self, task: RoundTask, key: str, outcome: object) -> None:
        self.task = task
        self.key = key
        self.outcome = outcome


class _FakeAsCompleted(Iterator[tuple[_FakeFuture, object]]):
    def __init__(
        self,
        modules: _FakeDistributedModules,
        futures: list[_FakeFuture],
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

    def __iter__(self) -> _FakeAsCompleted:
        return self

    def __next__(self) -> tuple[_FakeFuture, object]:
        if (
            self._modules.interrupt_after is not None
            and self._yielded == self._modules.interrupt_after
        ):
            self._modules.interrupt_after = None
            raise KeyboardInterrupt("injected collection interruption")
        if not self._queue:
            raise StopIteration
        self._yielded += 1
        future = self._queue.pop(0)
        return future, future.outcome

    def add(self, future: _FakeFuture) -> None:
        self._queue.append(future)


class _FakeCluster:
    def __init__(self, modules: _FakeDistributedModules) -> None:
        self._modules = modules

    def close(self) -> None:
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
        runner: Callable[[RoundTask], RoundResult],
        task: RoundTask,
        *,
        key: str,
        pure: bool,
        retries: int,
        resources: dict[str, int],
    ) -> _FakeFuture:
        del runner
        self._modules.submit_calls.append(
            _SubmitCall(task, key, pure, retries, resources)
        )
        configured = self._modules.outcomes.setdefault(task.round, [])
        outcome = configured.pop(0) if configured else _result(task)
        return _FakeFuture(task, key, outcome)

    def cancel(self, futures: tuple[_FakeFuture, ...], *, force: bool) -> None:
        self._modules.cancel_call = (futures, force)
        if self._modules.cancel_error is not None:
            raise self._modules.cancel_error

    def close(self) -> None:
        self._modules.events.append("client.close")
        if self._modules.client_close_error is not None:
            raise self._modules.client_close_error


class _FakeDistributedModules:
    def __init__(self) -> None:
        self.cluster_kwargs: dict[str, object] | None = None
        self.client_timeout: str | None = None
        self.wait_call: tuple[int, float] | None = None
        self.as_completed_call: tuple[bool, bool] | None = None
        self.submit_calls: list[_SubmitCall] = []
        self.events: list[str] = []
        self.cancel_call: tuple[tuple[_FakeFuture, ...], bool] | None = None
        self.outcomes: dict[int, list[object]] = {}
        self.order: tuple[int, ...] = ()
        self.interrupt_after: int | None = None
        self.cluster_error: Exception | None = None
        self.client_error: Exception | None = None
        self.wait_error: Exception | None = None
        self.cancel_error: Exception | None = None
        self.client_close_error: Exception | None = None
        self.cluster_close_error: Exception | None = None

    def LocalCluster(self, **kwargs: object) -> _FakeCluster:
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
        return _FakeAsCompleted(self, futures)

    def parse_timedelta(self, value: str) -> float:
        assert value == "30s"
        return 30.0


def test_local_executor_uses_one_round_slot_per_bounded_worker(tmp_path: Path) -> None:
    modules = _FakeDistributedModules()
    executor = LocalDaskExecutor(_profile(tmp_path), _modules=lambda: modules)

    results = executor.execute((_task(tmp_path, 2), _task(tmp_path, 3)))

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
    ]
    assert tuple(result.round for result in results) == (2, 3)


def test_local_executor_submits_deterministic_non_pure_zero_retry_keys(
    tmp_path: Path,
) -> None:
    modules = _FakeDistributedModules()
    task = _task(tmp_path, 7, dataset_id="sample_A.1")

    LocalDaskExecutor(_profile(tmp_path), _modules=lambda: modules).execute((task,))

    call = modules.submit_calls[0]
    assert call.key == "cyexm-sample_A.1-round-007"
    assert call.pure is False
    assert call.retries == 0


def test_local_executor_returns_immediate_empty_result_without_loading_modules(
    tmp_path: Path,
) -> None:
    loaded = False

    def modules() -> _FakeDistributedModules:
        nonlocal loaded
        loaded = True
        return _FakeDistributedModules()

    assert LocalDaskExecutor(_profile(tmp_path), _modules=modules).execute(()) == ()
    assert loaded is False


def test_local_executor_collects_independently_but_returns_plan_order(
    tmp_path: Path,
) -> None:
    modules = _FakeDistributedModules()
    modules.order = (3, 2)

    results = LocalDaskExecutor(_profile(tmp_path), _modules=lambda: modules).execute(
        (_task(tmp_path, 2), _task(tmp_path, 3))
    )

    assert tuple(result.round for result in results) == (2, 3)


def test_local_executor_resubmits_failed_results_with_a_unique_bounded_key(
    tmp_path: Path,
) -> None:
    modules = _FakeDistributedModules()
    task = _task(tmp_path, 2)
    modules.outcomes[2] = [_result(task, status="failed"), _result(task)]

    result = LocalDaskExecutor(
        _profile(tmp_path, retries=1), _modules=lambda: modules
    ).execute((task,))

    assert result[0].status == "completed"
    assert [call.key for call in modules.submit_calls] == [
        "cyexm-dataset-1-round-002",
        "cyexm-dataset-1-round-002-attempt-002",
    ]
    assert all(call.retries == 0 for call in modules.submit_calls)


def test_local_executor_resubmits_future_exceptions_and_structures_exhaustion(
    tmp_path: Path,
) -> None:
    modules = _FakeDistributedModules()
    task = _task(tmp_path, 2)
    modules.outcomes[2] = [ValueError("first"), RuntimeError("second")]

    result = LocalDaskExecutor(
        _profile(tmp_path, retries=1), _modules=lambda: modules
    ).execute((task,))[0]

    assert len(modules.submit_calls) == 2
    assert result.status == "failed"
    assert result.round == 2
    assert result.dask_task_key == "cyexm-dataset-1-round-002-attempt-002"
    assert result.hostname == "unknown"
    assert result.pid == 0
    assert result.failure == RoundFailure(
        stage="round",
        exception_type="RuntimeError",
        message="second",
    )


def test_local_executor_does_not_cancel_other_rounds_after_terminal_failure(
    tmp_path: Path,
) -> None:
    modules = _FakeDistributedModules()
    first = _task(tmp_path, 2)
    modules.outcomes[2] = [_result(first, status="failed")]

    failed, completed = LocalDaskExecutor(
        _profile(tmp_path), _modules=lambda: modules
    ).execute((first, _task(tmp_path, 3)))

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
            (_task(tmp_path, 2), _task(tmp_path, 3))
        )

    assert isinstance(captured.value.__cause__, RuntimeError)
    expected = {
        "cluster": [],
        "client": ["cluster.close"],
        "wait": ["client.close", "cluster.close"],
    }
    assert modules.events == expected[failure_point]


def test_local_executor_closes_client_before_cluster_on_success(tmp_path: Path) -> None:
    modules = _FakeDistributedModules()

    LocalDaskExecutor(_profile(tmp_path), _modules=lambda: modules).execute(
        (_task(tmp_path, 2),)
    )

    assert modules.events == ["client.close", "cluster.close"]


def test_local_executor_logs_cleanup_failures_without_losing_results(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    modules = _FakeDistributedModules()
    modules.client_close_error = RuntimeError("client cleanup failed")
    modules.cluster_close_error = RuntimeError("cluster cleanup failed")

    with caplog.at_level(logging.WARNING, logger="cyexm.execution.dask"):
        results = LocalDaskExecutor(
            _profile(tmp_path), _modules=lambda: modules
        ).execute((_task(tmp_path, 2),))

    assert results[0].status == "completed"
    assert modules.events == ["client.close", "cluster.close"]
    assert "client cleanup failed" in caplog.text
    assert "cluster cleanup failed" in caplog.text


def test_local_executor_cancels_only_unfinished_futures_on_keyboard_interrupt(
    tmp_path: Path,
) -> None:
    modules = _FakeDistributedModules()
    modules.order = (2, 3, 4)
    modules.interrupt_after = 1

    with pytest.raises(KeyboardInterrupt, match="collection interruption"):
        LocalDaskExecutor(_profile(tmp_path), _modules=lambda: modules).execute(
            (_task(tmp_path, 2), _task(tmp_path, 3), _task(tmp_path, 4))
        )

    assert modules.cancel_call is not None
    cancelled, force = modules.cancel_call
    assert {future.task.round for future in cancelled} == {3, 4}
    assert force is True
    assert modules.events == ["client.close", "cluster.close"]


def test_local_executor_preserves_interrupt_when_cancel_and_cleanup_fail(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    modules = _FakeDistributedModules()
    modules.interrupt_after = 0
    modules.cancel_error = RuntimeError("cancel failed")
    modules.client_close_error = RuntimeError("close failed")

    with (
        caplog.at_level(logging.WARNING, logger="cyexm.execution.dask"),
        pytest.raises(KeyboardInterrupt, match="collection interruption"),
    ):
        LocalDaskExecutor(_profile(tmp_path), _modules=lambda: modules).execute(
            (_task(tmp_path, 2),)
        )

    assert "cancel failed" in caplog.text
    assert "close failed" in caplog.text
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
        LocalDaskExecutor(invalid).execute((_task(tmp_path, 2),))


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
        LocalDaskExecutor(invalid).execute((_task(tmp_path, 2),))


def test_local_executor_wraps_lazy_module_loading_failure(tmp_path: Path) -> None:
    def unavailable() -> Any:
        raise ModuleNotFoundError("distributed")

    with pytest.raises(
        ExecutorStartupError,
        match=r"1 local Dask workers.*30s",
    ) as captured:
        LocalDaskExecutor(_profile(tmp_path), _modules=unavailable).execute(
            (_task(tmp_path, 2),)
        )

    assert isinstance(captured.value.__cause__, ModuleNotFoundError)
