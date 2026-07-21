"""Scheduler-neutral policy shared by distributed round executors."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from types import TracebackType
from typing import Any, cast

from cyexm.errors import RoundExecutionError
from cyexm.execution.base import WorkflowTaskRunner
from cyexm.workflow._execution_validation import (
    _validate_executor_reference_task,
    _validate_executor_round_task,
    _validate_round_result,
)
from cyexm.workflow.models import (
    ReferenceTask,
    RoundFailure,
    RoundResult,
    WorkflowTask,
)

ROUND_RESOURCE = {"round_slot": 1}


def dask_scheduler_config_overlay() -> dict[str, object]:
    """Return owned scheduler settings required by coarse native round tasks."""
    return {"distributed.scheduler.worker-ttl": None}


@dataclass(frozen=True, slots=True)
class Submission:
    task_index: int
    task: WorkflowTask
    key: str
    started_at: str
    started: float


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def task_key(task: WorkflowTask, attempt: int) -> str:
    role = "reference-round" if isinstance(task, ReferenceTask) else "round"
    base = f"cyexm-{task.dataset_id}-{role}-{task.round:03d}"
    if attempt == 1:
        return base
    return f"{base}-attempt-{attempt:03d}"


def exception_result(
    task: WorkflowTask,
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
        ended_at=utc_now(),
        duration_seconds=max(0.0, time.monotonic() - started),
        hostname="unknown",
        pid=os.getpid(),
        dask_task_key=key,
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


def distributed_error(outcome: object) -> BaseException:
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


def future_is_unfinished(future: Any, *, logger: logging.Logger) -> bool:
    try:
        return not bool(future.done())
    except BaseException as error:
        status = getattr(future, "status", None)
        if status in {"finished", "error", "cancelled"}:
            return False
        logger.warning("Could not inspect Dask future completion state: %s", error)
        return True


def validate_distributed_tasks(
    tasks: tuple[WorkflowTask, ...],
    *,
    executor_name: str,
) -> None:
    identities: set[tuple[str, int]] = set()
    for index, task in enumerate(tasks):
        if isinstance(task, ReferenceTask):
            _validate_executor_reference_task(task)
        else:
            _validate_executor_round_task(task, index)
        identity = (task.dataset_id, task.round)
        if identity in identities:
            raise RoundExecutionError(
                f"{executor_name} received a duplicate task identity: {identity!r}"
            )
        identities.add(identity)


def collect_distributed_results(
    tasks: tuple[WorkflowTask, ...],
    *,
    runner: WorkflowTaskRunner,
    client: Any,
    as_completed: Callable[..., Any],
    task_retries: int,
    active: set[Any],
    attempts: list[int] | None = None,
) -> tuple[tuple[RoundResult, ...], tuple[int, ...]]:
    """Submit coarse round tasks and return terminal results in input order."""
    attempt_counts = [0 for _task in tasks] if attempts is None else attempts
    if len(attempt_counts) != len(tasks):
        raise ValueError("attempt count storage must match the submitted task count")
    submissions: dict[Any, Submission] = {}

    def submit(task_index: int, task: WorkflowTask) -> Any:
        attempt = attempt_counts[task_index] + 1
        started_at = utc_now()
        started = time.monotonic()
        key = task_key(task, attempt)
        future = client.submit(
            runner,
            task,
            key=key,
            pure=False,
            retries=0,
            resources=dict(ROUND_RESOURCE),
        )
        attempt_counts[task_index] = attempt
        submissions[future] = Submission(
            task_index,
            task,
            key,
            started_at,
            started,
        )
        active.add(future)
        return future

    initial = [submit(index, task) for index, task in enumerate(tasks)]
    completed = as_completed(
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
                result = exception_result(
                    task,
                    error,
                    key=submission.key,
                    started_at=submission.started_at,
                    started=submission.started,
                )
        else:
            result = exception_result(
                task,
                distributed_error(outcome),
                key=submission.key,
                started_at=submission.started_at,
                started=submission.started,
            )
        _validate_round_result(task, result)

        if (
            result.status == "failed"
            and attempt_counts[submission.task_index] <= task_retries
        ):
            completed.add(submit(submission.task_index, task))
            continue
        terminal[submission.task_index] = result

    if any(result is None for result in terminal):
        raise RoundExecutionError(
            "Dask completion ended before every submitted round task settled"
        )
    return cast(tuple[RoundResult, ...], tuple(terminal)), tuple(attempt_counts)


__all__ = [
    "ROUND_RESOURCE",
    "collect_distributed_results",
    "dask_scheduler_config_overlay",
    "future_is_unfinished",
    "utc_now",
    "validate_distributed_tasks",
]
