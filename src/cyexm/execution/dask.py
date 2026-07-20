"""Bounded, process-based local Dask execution for round tasks."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from cyexm.config.models import ExecutionProfile, LocalDaskSpec
from cyexm.errors import ExecutorStartupError
from cyexm.execution.base import RoundTaskRunner
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
        pid=0,
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
        except Exception as error:
            _LOGGER.warning("Failed to close owned Dask client: %s", error)
    if cluster is not None:
        try:
            cluster.close()
        except Exception as error:
            _LOGGER.warning("Failed to close owned local Dask cluster: %s", error)


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

            attempts: dict[int, int] = {}
            attempt_started: dict[int, tuple[str, float]] = {}
            tasks_by_future: dict[Any, RoundTask] = {}
            keys_by_future: dict[Any, str] = {}
            execution_client = client
            assert execution_client is not None

            def submit(task: RoundTask) -> Any:
                attempt = attempts.get(task.round, 0) + 1
                attempts[task.round] = attempt
                attempt_started[task.round] = (_utc_now(), time.monotonic())
                key = _task_key(task, attempt)
                future = execution_client.submit(
                    self._runner,
                    task,
                    key=key,
                    pure=False,
                    retries=0,
                    resources=dict(_ROUND_RESOURCE),
                )
                tasks_by_future[future] = task
                keys_by_future[future] = key
                active.add(future)
                return future

            initial = [submit(task) for task in tasks]
            completed = modules.as_completed(
                initial,
                with_results=True,
                raise_errors=False,
            )
            terminal: dict[int, RoundResult] = {}
            for future, outcome in completed:
                active.discard(future)
                task = tasks_by_future[future]
                key = keys_by_future[future]
                if isinstance(outcome, RoundResult):
                    result = outcome
                elif isinstance(outcome, BaseException):
                    started_at, started = attempt_started[task.round]
                    result = _exception_result(
                        task,
                        outcome,
                        key=key,
                        started_at=started_at,
                        started=started,
                    )
                else:
                    started_at, started = attempt_started[task.round]
                    result = _exception_result(
                        task,
                        TypeError(
                            "Dask round task returned an unsupported result: "
                            f"{outcome!r}"
                        ),
                        key=key,
                        started_at=started_at,
                        started=started,
                    )

                if (
                    result.status == "failed"
                    and attempts[task.round] <= spec.task_retries
                ):
                    completed.add(submit(task))
                    continue
                terminal[task.round] = result

            return tuple(terminal[task.round] for task in tasks)
        except KeyboardInterrupt:
            if client is not None and active:
                try:
                    client.cancel(tuple(active), force=True)
                except Exception as error:
                    _LOGGER.warning(
                        "Failed to cancel unfinished local Dask futures: %s", error
                    )
            raise
        finally:
            _close_owned(client, cluster)


__all__ = ["LocalDaskExecutor"]
