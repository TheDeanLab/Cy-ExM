"""Bounded, process-based local Dask execution for round tasks."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from cyexm.config.models import ExecutionProfile, LocalDaskSpec
from cyexm.errors import ExecutorStartupError
from cyexm.execution._dask_common import (
    ROUND_RESOURCE,
    collect_distributed_results,
    future_is_unfinished,
    validate_distributed_tasks,
)
from cyexm.execution.base import RoundTaskRunner
from cyexm.workflow.models import RoundResult, RoundTask
from cyexm.workflow.round_task import run_round_task

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _DistributedModules:
    Client: Any
    LocalCluster: Any
    as_completed: Any
    parse_timedelta: Any


def _distributed_modules() -> _DistributedModules:
    """Load scheduler dependencies only when a local executor is opened."""
    from dask.utils import parse_timedelta
    from distributed import Client, LocalCluster, as_completed

    return _DistributedModules(Client, LocalCluster, as_completed, parse_timedelta)


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
        validate_distributed_tasks(tasks, executor_name="LocalDaskExecutor")
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
                    resources=dict(ROUND_RESOURCE),
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

            execution_client = client
            assert execution_client is not None
            results, _attempts = collect_distributed_results(
                tasks,
                runner=self._runner,
                client=execution_client,
                as_completed=modules.as_completed,
                task_retries=spec.task_retries,
                active=active,
            )
            return results
        except KeyboardInterrupt:
            if client is not None and active:
                unfinished = tuple(
                    future
                    for future in active
                    if future_is_unfinished(future, logger=_LOGGER)
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
