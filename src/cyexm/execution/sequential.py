from __future__ import annotations

import os
import socket
import time
from datetime import UTC, datetime
from typing import cast

from cyexm.execution.base import WorkflowTaskRunner
from cyexm.workflow.models import ReferenceTask, RoundResult, RoundTask, WorkflowTask


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class _SequentialExecutionInterrupted(KeyboardInterrupt):
    """Carry durable results completed before the original interruption."""

    def __init__(
        self,
        original: KeyboardInterrupt,
        completed_results: tuple[RoundResult, ...],
    ) -> None:
        super().__init__(*original.args)
        self.original = original
        self.completed_results = completed_results


class SequentialExecutor:
    """Run independent round tasks in canonical order in the current process."""

    def __init__(self, *, runner: WorkflowTaskRunner) -> None:
        self._runner = runner

    def _execute_tasks(
        self, tasks: tuple[WorkflowTask, ...]
    ) -> tuple[RoundResult, ...]:
        results: list[RoundResult] = []
        for task in tasks:
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
                results.append(cast(RoundResult, result))
        return tuple(results)

    def execute(
        self,
        reference_task: ReferenceTask,
        round_tasks: tuple[RoundTask, ...],
    ) -> tuple[RoundResult, ...]:
        reference_results = self._execute_tasks((reference_task,))
        if reference_results[0].status not in {"completed", "reused"}:
            return reference_results
        try:
            moving_results = self._execute_tasks(round_tasks)
        except _SequentialExecutionInterrupted as error:
            raise _SequentialExecutionInterrupted(
                error.original,
                (*reference_results, *error.completed_results),
            ) from error
        return (*reference_results, *moving_results)


__all__ = ["SequentialExecutor"]
