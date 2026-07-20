"""Bounded, process-based local Dask execution for round tasks."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from types import TracebackType
from typing import Any, cast

from cyexm.config.models import ExecutionProfile, LocalDaskSpec
from cyexm.errors import ExecutorStartupError, RoundExecutionError
from cyexm.execution.base import RoundTaskRunner
from cyexm.workflow._execution_validation import (
    _validate_executor_round_task,
    _validate_round_result,
)
from cyexm.workflow.models import RoundFailure, RoundResult, RoundTask
from cyexm.workflow.round_task import run_round_task

_LOGGER = logging.getLogger(__name__)
_ROUND_RESOURCE = {"round_slot": 1}


@dataclass(frozen=True, slots=True)
class _DistributedModules:
    Client: Any
    LocalCluster: Any
    as_completed: Any
    parse_timedelta: Any


@dataclass(frozen=True, slots=True)
class _Submission:
    task_index: int
    task: RoundTask
    key: str
    started_at: str
    started: float


def _distributed_modules() -> _DistributedModules:
    """Load scheduler dependencies only when a local executor is opened."""
    from distributed import Client, LocalCluster, as_completed
    from distributed.utils import parse_timedelta

    return _DistributedModules(Client, LocalCluster, as_completed, parse_timedelta)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _task_key(task: RoundTask, attempt: int) -> str:
    base = f"cyexm-{task.dataset_id}-round-{task.round:03d}"
    if attempt == 1:
        return base
    return f"{base}-attempt-{attempt:03d}"


def _exception_result(
    task: RoundTask,
    error: BaseException,
    *,
    key: str,
    started_at: str,
    started: float,
) -> RoundResult:
    return RoundResult(
        dataset_id=task.dataset_id,
        round=task.round,
        status="failed",
        started_at=started_at,
        ended_at=_utc_now(),
        duration_seconds=max(0.0, time.monotonic() - started),
        hostname="unknown",
        pid=os.getpid(),
        dask_task_key=key,
        slurm_job_id=None,
        strategy=task.strategy,
        native_threads=(
            ("itk", task.native_threads.itk),
            ("blas", task.native_threads.blas),
        ),
        artifact_paths=(),
        reused_artifacts=(),
        log_path=str(task.log_path),
        failure=RoundFailure(
            stage="round",
            exception_type=type(error).__name__,
            message=str(error),
        ),
    )


def _close_owned(client: Any | None, cluster: Any | None) -> None:
    if client is not None:
        try:
            client.close()
        except BaseException as error:
            _LOGGER.warning("Failed to close owned Dask client: %s", error)
    if cluster is not None:
        try:
            cluster.close()
        except BaseException as error:
            _LOGGER.warning("Failed to close owned local Dask cluster: %s", error)


def _distributed_error(outcome: object) -> BaseException:
    if isinstance(outcome, BaseException):
        return outcome
    if type(outcome) is not tuple:
        return TypeError(f"Dask round task returned an unsupported result: {outcome!r}")
    if len(outcome) != 3:
        return TypeError(
            f"malformed distributed error payload: expected 3 entries; got {outcome!r}"
        )
    exception_type, exception, traceback = outcome
    if (
        not isinstance(exception_type, type)
        or not issubclass(exception_type, BaseException)
        or not isinstance(exception, BaseException)
        or type(exception) is not exception_type
        or (traceback is not None and type(traceback) is not TracebackType)
    ):
        return TypeError(f"malformed distributed error payload: {outcome!r}")
    return exception


def _future_is_unfinished(future: Any) -> bool:
    try:
        return not bool(future.done())
    except BaseException as error:
        status = getattr(future, "status", None)
        if status in {"finished", "error", "cancelled"}:
            return False
        _LOGGER.warning("Could not inspect Dask future completion state: %s", error)
        return True


class LocalDaskExecutor:
    """Execute independent rounds on an owned, bounded ``LocalCluster``."""

    def __init__(
        self,
        profile: ExecutionProfile,
        *,
        _runner: RoundTaskRunner = run_round_task,
        _modules: Callable[[], Any] = _distributed_modules,
    ) -> None:
        self.profile = profile
        self._runner = _runner
        self._modules = _modules

    def _spec(self) -> LocalDaskSpec:
        if self.profile.executor != "local_dask":
            raise ExecutorStartupError(
                "LocalDaskExecutor requires a local_dask profile"
            )
        if not isinstance(self.profile.local_dask, LocalDaskSpec):
            raise ExecutorStartupError(
                "LocalDaskExecutor requires LocalDaskSpec settings"
            )
        return self.profile.local_dask

    def execute(self, tasks: tuple[RoundTask, ...]) -> tuple[RoundResult, ...]:
        if not tasks:
            return ()

        spec = self._spec()
        identities: set[tuple[str, int]] = set()
        for index, task in enumerate(tasks):
            _validate_executor_round_task(task, index)
            identity = (task.dataset_id, task.round)
            if identity in identities:
                raise RoundExecutionError(
                    "LocalDaskExecutor received a duplicate task identity: "
                    f"{identity!r}"
                )
            identities.add(identity)
        worker_count = min(len(tasks), spec.workers)
        cluster: Any | None = None
        client: Any | None = None
        active: set[Any] = set()

        try:
            try:
                modules = self._modules()
                cluster = modules.LocalCluster(
                    n_workers=worker_count,
                    threads_per_worker=spec.threads_per_worker,
                    processes=True,
                    memory_limit=spec.memory_limit,
                    local_directory=str(self.profile.paths.worker_local_directory),
                    resources=dict(_ROUND_RESOURCE),
                )
                opened_client = modules.Client(cluster, timeout=spec.startup_timeout)
                client = opened_client
                opened_client.wait_for_workers(
                    worker_count,
                    timeout=modules.parse_timedelta(spec.startup_timeout),
                )
            except Exception as error:
                raise ExecutorStartupError(
                    f"Failed to start {worker_count} local Dask workers within "
                    f"{spec.startup_timeout}: {error}"
                ) from error

            attempts = [0 for _task in tasks]
            submissions: dict[Any, _Submission] = {}
            execution_client = client
            assert execution_client is not None

            def submit(task_index: int, task: RoundTask) -> Any:
                attempts[task_index] += 1
                attempt = attempts[task_index]
                started_at = _utc_now()
                started = time.monotonic()
                key = _task_key(task, attempt)
                future = execution_client.submit(
                    self._runner,
                    task,
                    key=key,
                    pure=False,
                    retries=0,
                    resources=dict(_ROUND_RESOURCE),
                )
                submissions[future] = _Submission(
                    task_index,
                    task,
                    key,
                    started_at,
                    started,
                )
                active.add(future)
                return future

            initial = [submit(index, task) for index, task in enumerate(tasks)]
            completed = modules.as_completed(
                initial,
                with_results=True,
                raise_errors=False,
            )
            terminal: list[RoundResult | None] = [None for _task in tasks]
            for future, outcome in completed:
                active.discard(future)
                submission = submissions[future]
                task = submission.task
                if isinstance(outcome, RoundResult):
                    result = outcome
                    try:
                        _validate_round_result(task, result)
                    except RoundExecutionError as error:
                        result = _exception_result(
                            task,
                            error,
                            key=submission.key,
                            started_at=submission.started_at,
                            started=submission.started,
                        )
                else:
                    result = _exception_result(
                        task,
                        _distributed_error(outcome),
                        key=submission.key,
                        started_at=submission.started_at,
                        started=submission.started,
                    )
                _validate_round_result(task, result)

                if (
                    result.status == "failed"
                    and attempts[submission.task_index] <= spec.task_retries
                ):
                    completed.add(submit(submission.task_index, task))
                    continue
                terminal[submission.task_index] = result

            if any(result is None for result in terminal):
                raise RoundExecutionError(
                    "Local Dask completion ended before every task settled"
                )
            return cast(tuple[RoundResult, ...], tuple(terminal))
        except KeyboardInterrupt:
            if client is not None and active:
                unfinished = tuple(
                    future for future in active if _future_is_unfinished(future)
                )
                try:
                    if unfinished:
                        client.cancel(unfinished, force=True)
                except BaseException as error:
                    _LOGGER.warning(
                        "Failed to cancel unfinished local Dask futures: %s", error
                    )
            raise
        finally:
            _close_owned(client, cluster)


__all__ = ["LocalDaskExecutor"]
