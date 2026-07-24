"""Scheduler-neutral construction of failed fine-task evidence."""

from __future__ import annotations

from cyexm.runtime import NATIVE_THREAD_ENV_VARS
from cyexm.workflow.fine_models import FineTaskResult, FineWorkflowTask
from cyexm.workflow.models import RoundFailure


def failed_fine_task_result(
    task: FineWorkflowTask,
    error: BaseException,
    *,
    started_at: str,
    ended_at: str,
    duration_seconds: float,
    hostname: str,
    pid: int,
    dask_task_key: str | None,
    slurm_job_id: str | None,
) -> FineTaskResult:
    """Build exact terminal failure evidence for one submitted immutable task."""
    return FineTaskResult(
        task=task,
        status="failed",
        started_at=started_at,
        ended_at=ended_at,
        duration_seconds=duration_seconds,
        hostname=hostname,
        pid=pid,
        dask_task_key=dask_task_key,
        slurm_job_id=slurm_job_id,
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
        artifact_paths=(),
        reused_artifacts=(),
        log_path=str(task.log_path),
        failure=RoundFailure(
            stage=task.key.stage,
            exception_type=type(error).__name__,
            message=str(error),
        ),
    )


__all__ = ["failed_fine_task_result"]
