from __future__ import annotations

import os
import socket
import time
from datetime import UTC, datetime

from cyexm.config.execution import _validate_execution_profile
from cyexm.config.models import ExecutionProfile
from cyexm.execution.base import FineWorkflowTaskRunner, WorkflowTaskRunner
from cyexm.execution.dispatcher import FineGraphController
from cyexm.workflow._execution_validation import _FineExecutionEvidence
from cyexm.workflow.fine_models import FineTaskResult, FineWorkflowTask
from cyexm.workflow.models import ReferenceTask, RoundResult, RoundTask, WorkflowTask


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class _SequentialExecutionInterrupted(KeyboardInterrupt):
    """Carry durable results completed before the original interruption."""

    def __init__(
        self,
        original: KeyboardInterrupt,
        completed_results: tuple[RoundResult, ...],
        fine_task_results: tuple[FineTaskResult, ...] = (),
    ) -> None:
        super().__init__(*original.args)
        self.original = original
        self.completed_results = completed_results
        self.fine_task_results = fine_task_results


class SequentialExecutor:
    """Execute a normalized coarse or fine workflow in the current process.

    This executor is the scheduling reference for distributed adapters. It owns no
    worker pool or scheduler: reference preparation and all admitted moving work run
    one task at a time in the calling process.

    Parameters
    ----------
    profile : ExecutionProfile
        Exact normalized sequential profile. ``parallelism.mode`` selects the
        existing coarse round runner or the scheduler-neutral fine graph.
    runner : WorkflowTaskRunner
        Injectable coarse reference/round runner.
    fine_runner : FineWorkflowTaskRunner, optional
        Injectable fine-stage runner. Omission lazily selects ``run_fine_task``.

    Raises
    ------
    ConfigurationValidationError
        If ``profile`` does not satisfy the strict execution-profile contract.
    ValueError
        If ``profile`` is not an exact sequential execution profile.

    Notes
    -----
    Reference preparation is always a hard coarse prerequisite. Fine mode never
    falls back to coarse moving-round execution. Independent ready siblings settle
    after a failure, and interruption retains the actual partial graph, task
    results, admissions, and blocked descendants for the public run record.
    Scientific reuse remains authorized only by Phase 1 completion records.

    See Also
    --------
    FineGraphController :
        Owns dependency transitions and deterministic fine-task admission.
    """

    def __init__(
        self,
        profile: ExecutionProfile,
        *,
        runner: WorkflowTaskRunner,
        fine_runner: FineWorkflowTaskRunner | None = None,
    ) -> None:
        if type(profile) is not ExecutionProfile or profile.executor != "sequential":
            raise ValueError(
                "SequentialExecutor requires an exact sequential ExecutionProfile"
            )
        _validate_execution_profile(profile)
        self._profile = profile
        self._runner = runner
        self._fine_runner = fine_runner
        self._execution_evidence: _FineExecutionEvidence | None = None

    @property
    def execution_evidence(self) -> _FineExecutionEvidence:
        """Return detached scheduler-neutral fine graph evidence."""
        if self._execution_evidence is None:
            raise RuntimeError("fine execution evidence is not available")
        return self._execution_evidence

    def _record_fine_evidence(
        self,
        controller: FineGraphController,
        attempts: list[tuple[str, int]],
        *,
        cancelled: bool,
    ) -> None:
        self._execution_evidence = _FineExecutionEvidence(
            executor="sequential",
            worker_cap=1,
            tasks=controller.tasks,
            task_graph=controller.graph_description,
            task_results=controller.task_results,
            attempt_counts=tuple(attempts),
            blocked_descendants=tuple(key.encoded for key in controller.blocked_keys),
            cancelled=cancelled,
            cleanup=(),
        )

    def _execute_tasks(
        self, tasks: tuple[WorkflowTask, ...]
    ) -> tuple[RoundResult, ...]:
        results: list[RoundResult] = []
        for task in tasks:
            if type(task) not in {ReferenceTask, RoundTask}:
                raise TypeError("coarse sequential tasks must use exact task types")
            started_at = _utc_now()
            started = time.monotonic()
            try:
                result = self._runner(task)
            except KeyboardInterrupt as error:
                raise _SequentialExecutionInterrupted(error, tuple(results)) from error
            except Exception as error:
                from cyexm.workflow.models import RoundFailure

                results.append(
                    RoundResult(
                        dataset_id=task.dataset_id,
                        round=task.round,
                        status="failed",
                        started_at=started_at,
                        ended_at=_utc_now(),
                        duration_seconds=max(0.0, time.monotonic() - started),
                        hostname=socket.gethostname(),
                        pid=os.getpid(),
                        dask_task_key=None,
                        slurm_job_id=None,
                        strategy=(
                            "reference-preparation"
                            if isinstance(task, ReferenceTask)
                            else task.strategy
                        ),
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
                )
            else:
                if type(result) is not RoundResult:
                    raise TypeError(
                        "coarse sequential runner must return an exact RoundResult"
                    )
                results.append(result)
        return tuple(results)

    def _run_fine(self, task: FineWorkflowTask) -> FineTaskResult:
        runner = self._fine_runner
        if runner is None:
            from cyexm.workflow.fine_task import run_fine_task

            runner = run_fine_task
        result = runner(task)
        if type(result) is not FineTaskResult:
            raise TypeError(
                "fine sequential runner must return an exact FineTaskResult"
            )
        return result

    def _execute_fine_tasks(
        self, round_tasks: tuple[RoundTask, ...]
    ) -> tuple[RoundResult, ...]:
        controller = FineGraphController(round_tasks)
        attempts: list[tuple[str, int]] = []
        while not controller.terminal:
            ready = controller.pop_ready(limit=1)
            if len(ready) != 1:
                raise RuntimeError(
                    "non-terminal fine graph has no scheduler-ready task"
                )
            task = ready[0]
            attempts.append((task.key.encoded, 1))
            try:
                result = self._run_fine(task)
            except KeyboardInterrupt as error:
                self._record_fine_evidence(
                    controller,
                    attempts,
                    cancelled=True,
                )
                raise _SequentialExecutionInterrupted(
                    error,
                    controller._settled_round_results(),
                    controller.task_results,
                ) from error
            controller.record(task, result)
        self._record_fine_evidence(controller, attempts, cancelled=False)
        return controller.round_results()

    def execute(
        self,
        reference_task: ReferenceTask,
        round_tasks: tuple[RoundTask, ...],
    ) -> tuple[RoundResult, ...]:
        self._execution_evidence = None
        reference_results = self._execute_tasks((reference_task,))
        if reference_results[0].status not in {"completed", "reused"}:
            return reference_results
        try:
            moving_results = (
                self._execute_tasks(round_tasks)
                if self._profile.parallelism.mode == "coarse"
                else self._execute_fine_tasks(round_tasks)
            )
        except _SequentialExecutionInterrupted as error:
            raise _SequentialExecutionInterrupted(
                error.original,
                (*reference_results, *error.completed_results),
                error.fine_task_results,
            ) from error
        return (*reference_results, *moving_results)


__all__ = ["SequentialExecutor"]
