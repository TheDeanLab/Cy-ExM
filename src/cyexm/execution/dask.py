"""Bounded, process-based local Dask execution for round tasks."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

from cyexm.config.models import ExecutionProfile, LocalDaskSpec
from cyexm.errors import ExecutorStartupError
from cyexm.execution._dask_common import (
    ROUND_RESOURCE,
    collect_distributed_fine_results,
    collect_distributed_results,
    dask_scheduler_config_overlay,
    future_is_unfinished,
    validate_distributed_tasks,
)
from cyexm.execution.base import FineWorkflowTaskRunner, WorkflowTaskRunner
from cyexm.execution.dispatcher import FineGraphController
from cyexm.workflow._execution_validation import _FineExecutionEvidence
from cyexm.workflow.fine_models import FineTaskResult
from cyexm.workflow.models import ReferenceTask, RoundResult, RoundTask
from cyexm.workflow.task import run_workflow_task

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _DistributedModules:
    Client: Any
    LocalCluster: Any
    as_completed: Any
    parse_timedelta: Any
    config_set: Any


def _distributed_modules() -> _DistributedModules:
    """Load scheduler dependencies only when a local executor is opened."""
    from dask import config as dask_config
    from dask.utils import parse_timedelta
    from distributed import Client, LocalCluster, as_completed

    return _DistributedModules(
        Client,
        LocalCluster,
        as_completed,
        parse_timedelta,
        dask_config.set,
    )


def _close_owned(
    client: Any | None,
    cluster: Any | None,
    configuration: AbstractContextManager[Any] | None,
) -> tuple[tuple[str, bool, str | None, str | None], ...]:
    exit_info = sys.exc_info()
    statuses: list[tuple[str, bool, str | None, str | None]] = []

    def close(name: str, operation: Callable[[], object]) -> None:
        try:
            operation()
        except BaseException as error:
            statuses.append((name, False, type(error).__name__, str(error)))
            _LOGGER.warning("Failed to close owned local Dask %s: %s", name, error)
        else:
            statuses.append((name, True, None, None))

    if client is not None:
        close("client", client.close)
    if cluster is not None:
        close("cluster", cluster.close)
    if configuration is not None:
        close("configuration", lambda: configuration.__exit__(*exit_info))
    return tuple(statuses)


class LocalDaskExecutor:
    """Execute coarse rounds or fine stages on an owned ``LocalCluster``.

    The normalized profile selects the existing round-level runner or the shared
    scheduler-neutral fine graph. This class owns the local cluster, client,
    bounded worker pool, cancellation, and cleanup; workers execute one immutable
    task and never create clients or submit descendants.

    Parameters
    ----------
    profile : ExecutionProfile
        Exact normalized ``local_dask`` profile. ``local_dask.workers`` is the hard
        global task cap, and ``parallelism.mode`` selects coarse or fine admission.

    Raises
    ------
    ExecutorStartupError
        If the profile is incompatible or the owned scheduler lifecycle cannot
        start within the configured timeout.

    See Also
    --------
    SequentialExecutor :
        Execute the same normalized coarse or fine work without a worker pool.
    SlurmExecutor :
        Execute the same work through preflight-gated SLURM workers.

    Notes
    -----
    Reference preparation is a hard barrier in the same client lifecycle. Fine
    siblings settle independently after a failure, and the coordinator retains the
    exact graph, attempts, task results, cancellation, and cleanup evidence for
    strict public run-record validation. Scientific reuse remains authorized only
    by Phase 1 completion records.
    """

    def __init__(
        self,
        profile: ExecutionProfile,
        *,
        _runner: WorkflowTaskRunner = run_workflow_task,
        _fine_runner: FineWorkflowTaskRunner | None = None,
        _modules: Callable[[], Any] = _distributed_modules,
    ) -> None:
        self.profile = profile
        self._runner = _runner
        self._fine_runner = _fine_runner
        self._modules = _modules
        self._fine_task_results: tuple[FineTaskResult, ...] = ()
        self._fine_attempt_counts: tuple[tuple[str, int], ...] = ()
        self._fine_live_attempts: dict[str, int] = {}
        self._fine_controller: FineGraphController | None = None
        self._fine_reference_result: RoundResult | None = None
        self._fine_pre_controller_terminal: tuple[str, str, str, str] | None = None
        self._fine_cancelled = False
        self._fine_cleanup: tuple[tuple[str, bool, str | None, str | None], ...] = ()

    @property
    def execution_evidence(self) -> _FineExecutionEvidence:
        """Return detached scheduler-neutral fine graph evidence."""
        controller = self._fine_controller
        attempts = self._fine_attempt_counts
        if self._fine_cancelled and not attempts:
            attempts = tuple(
                (key, self._fine_live_attempts[key])
                for key, _dependencies in (
                    () if controller is None else controller.graph_description
                )
                if key in self._fine_live_attempts
            )
        return _FineExecutionEvidence(
            executor="local_dask",
            worker_cap=self._spec().workers,
            reference_result=self._fine_reference_result,
            round_results=(
                () if controller is None else controller._settled_round_results()
            ),
            pre_controller_terminal=self._fine_pre_controller_terminal,
            tasks=() if controller is None else controller.tasks,
            task_graph=() if controller is None else controller.graph_description,
            task_results=() if controller is None else controller.task_results,
            attempt_counts=attempts,
            blocked_descendants=(
                ()
                if controller is None
                else tuple(key.encoded for key in controller.blocked_keys)
            ),
            cancelled=self._fine_cancelled,
            cleanup=self._fine_cleanup,
        )

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

    def execute(
        self,
        reference_task: ReferenceTask,
        round_tasks: tuple[RoundTask, ...],
    ) -> tuple[RoundResult, ...]:
        spec = self._spec()
        validate_distributed_tasks(
            (reference_task, *round_tasks), executor_name="LocalDaskExecutor"
        )
        fine_mode = self.profile.parallelism.mode == "fine_grained"
        worker_count = (
            spec.workers if fine_mode else min(max(1, len(round_tasks)), spec.workers)
        )
        cluster: Any | None = None
        client: Any | None = None
        configuration: AbstractContextManager[Any] | None = None
        active: set[Any] = set()
        self._fine_task_results = ()
        self._fine_attempt_counts = ()
        self._fine_live_attempts = {}
        self._fine_controller = None
        self._fine_reference_result = None
        self._fine_pre_controller_terminal = None
        self._fine_cancelled = False
        self._fine_cleanup = ()

        try:
            try:
                modules = self._modules()
                opened_configuration = modules.config_set(
                    dask_scheduler_config_overlay()
                )
                opened_configuration.__enter__()
                configuration = opened_configuration
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
            reference_results, _attempts = collect_distributed_results(
                (reference_task,),
                runner=self._runner,
                client=execution_client,
                as_completed=modules.as_completed,
                task_retries=spec.task_retries,
                active=active,
            )
            if fine_mode:
                self._fine_reference_result = reference_results[0]
            if reference_results[0].status not in {"completed", "reused"}:
                if fine_mode:
                    failure = reference_results[0].failure
                    self._fine_pre_controller_terminal = (
                        "reference",
                        reference_results[0].status,
                        "" if failure is None else failure.exception_type,
                        "" if failure is None else failure.message,
                    )
                return reference_results
            if not round_tasks:
                if fine_mode:
                    self._fine_pre_controller_terminal = (
                        "moving",
                        "reused",
                        "",
                        "all submitted moving rounds were already complete",
                    )
                return reference_results
            if fine_mode:
                fine_runner = self._fine_runner
                if fine_runner is None:
                    from cyexm.workflow.fine_task import run_fine_task

                    fine_runner = run_fine_task
                controller = FineGraphController(round_tasks)
                self._fine_controller = controller
                moving_results, task_results, attempts = (
                    collect_distributed_fine_results(
                        controller,
                        runner=fine_runner,
                        client=execution_client,
                        as_completed=modules.as_completed,
                        task_retries=spec.task_retries,
                        capacity=worker_count,
                        active=active,
                        submission_observer=lambda task, attempt: (
                            self._fine_live_attempts.__setitem__(
                                task.key.encoded,
                                attempt,
                            )
                        ),
                    )
                )
                self._fine_task_results = task_results
                self._fine_attempt_counts = attempts
            else:
                moving_results, _attempts = collect_distributed_results(
                    round_tasks,
                    runner=self._runner,
                    client=execution_client,
                    as_completed=modules.as_completed,
                    task_retries=spec.task_retries,
                    active=active,
                )
            return (*reference_results, *moving_results)
        except KeyboardInterrupt as error:
            self._fine_cancelled = fine_mode
            if fine_mode and self._fine_controller is None:
                phase = (
                    "moving"
                    if self._fine_reference_result is not None
                    and self._fine_reference_result.status in {"completed", "reused"}
                    else "reference"
                )
                self._fine_pre_controller_terminal = (
                    phase,
                    "cancelled",
                    type(error).__name__,
                    str(error),
                )
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
            self._fine_cleanup = _close_owned(client, cluster, configuration)


__all__ = ["LocalDaskExecutor"]
