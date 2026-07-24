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
from cyexm.execution._fine_common import failed_fine_task_result
from cyexm.execution.base import FineWorkflowTaskRunner, WorkflowTaskRunner
from cyexm.execution.dispatcher import FineGraphController
from cyexm.workflow._execution_validation import (
    _validate_executor_reference_task,
    _validate_executor_round_task,
    _validate_fine_task_result,
    _validate_round_result,
)
from cyexm.workflow.fine_models import (
    FineTaskResult,
    FineWorkflowTask,
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


@dataclass(frozen=True, slots=True)
class FineAttemptObservation:
    """One scheduler-settled attempt before retry or dependency release."""

    task: FineWorkflowTask
    attempt: int
    dask_task_key: str
    coordinator_submitted_at: str
    coordinator_ready_at: str
    raw_result: FineTaskResult | None
    settled_result: FineTaskResult
    rejection_reason: str | None


@dataclass(frozen=True, slots=True)
class FineSubmission:
    task: FineWorkflowTask
    key: str
    started_at: str
    started: float
    attempt: int


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def task_key(task: WorkflowTask, attempt: int) -> str:
    role = "reference-round" if isinstance(task, ReferenceTask) else "round"
    base = f"cyexm-{task.dataset_id}-{role}-{task.round:03d}"
    if attempt == 1:
        return base
    return f"{base}-attempt-{attempt:03d}"


def fine_task_key(task: FineWorkflowTask, attempt: int) -> str:
    """Return one stable scheduler-safe key for an exact fine-task attempt."""
    base = f"cyexm-{task.key.encoded.replace(':', '-')}"
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


def fine_exception_result(
    task: FineWorkflowTask,
    error: BaseException,
    *,
    key: str,
    started_at: str,
    started: float,
    placement: FineTaskResult | None = None,
) -> FineTaskResult:
    return failed_fine_task_result(
        task,
        error,
        started_at=started_at,
        ended_at=utc_now(),
        duration_seconds=max(0.0, time.monotonic() - started),
        hostname="unknown" if placement is None else placement.hostname,
        pid=os.getpid() if placement is None else placement.pid,
        dask_task_key=key,
        slurm_job_id=None if placement is None else placement.slurm_job_id,
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


def collect_distributed_fine_results(
    controller: FineGraphController,
    *,
    runner: FineWorkflowTaskRunner,
    client: Any,
    as_completed: Callable[..., Any],
    task_retries: int,
    capacity: int,
    active: set[Any],
    capacity_provider: Callable[[FineGraphController, int], int] | None = None,
    active_wait: Callable[[tuple[Any, ...]], None] | None = None,
    before_settle: Callable[[tuple[Any, ...]], None] | None = None,
    result_validator: (
        Callable[[FineWorkflowTask, FineTaskResult, str, str], FineTaskResult] | None
    ) = None,
    attempt_observer: Callable[[FineAttemptObservation], None] | None = None,
    submission_observer: Callable[[FineWorkflowTask, int], None] | None = None,
) -> tuple[
    tuple[RoundResult, ...],
    tuple[FineTaskResult, ...],
    tuple[tuple[str, int], ...],
]:
    """Run a bounded coordinator-owned fine graph on an existing Dask client."""
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
        raise ValueError("fine distributed capacity must be a positive integer")
    submissions: dict[Any, FineSubmission] = {}
    attempts: dict[str, int] = {}
    completion: Any | None = None

    def submit(task: FineWorkflowTask) -> Any:
        encoded = task.key.encoded
        attempt = attempts.get(encoded, 0) + 1
        key = fine_task_key(task, attempt)
        started_at = utc_now()
        started = time.monotonic()
        future = client.submit(
            runner,
            task,
            key=key,
            pure=False,
            retries=0,
            resources=dict(ROUND_RESOURCE),
        )
        attempts[encoded] = attempt
        if submission_observer is not None:
            submission_observer(task, attempt)
        submissions[future] = FineSubmission(
            task=task,
            key=key,
            started_at=started_at,
            started=started,
            attempt=attempt,
        )
        active.add(future)
        return future

    def admit() -> list[Any]:
        eligible_capacity = (
            capacity_provider(controller, len(active))
            if capacity_provider is not None
            else capacity
        )
        if (
            isinstance(eligible_capacity, bool)
            or not isinstance(eligible_capacity, int)
            or eligible_capacity < 1
            or eligible_capacity > capacity
        ):
            raise ValueError(
                "fine distributed eligible capacity must be between one and "
                "the configured capacity"
            )
        available = eligible_capacity - len(active)
        if available <= 0:
            return []
        return [submit(task) for task in controller.pop_ready(limit=available)]

    initial = admit()
    if initial:
        completion = as_completed(
            initial,
            with_results=True,
            raise_errors=False,
        )
    elif not controller.terminal:
        raise RuntimeError("non-terminal fine graph has neither active nor ready work")

    def settle(future: Any, outcome: object, coordinator_ready_at: str) -> None:
        nonlocal completion
        assert completion is not None
        active.discard(future)
        submission = submissions[future]
        task = submission.task
        raw_result = outcome if type(outcome) is FineTaskResult else None
        rejection_reason: str | None = None
        if type(outcome) is FineTaskResult:
            try:
                result = _validate_fine_task_result(task, outcome)
                if result.dask_task_key != submission.key:
                    raise RoundExecutionError(
                        "Dask fine task result key does not match the current "
                        "submitted attempt"
                    )
                if result_validator is not None:
                    result = result_validator(
                        task,
                        result,
                        submission.started_at,
                        coordinator_ready_at,
                    )
            except RoundExecutionError as error:
                rejection_reason = f"{type(error).__name__}: {error}"
                result = fine_exception_result(
                    task,
                    error,
                    key=submission.key,
                    started_at=submission.started_at,
                    started=submission.started,
                    placement=outcome,
                )
        else:
            error = (
                distributed_error(outcome)
                if isinstance(outcome, BaseException) or type(outcome) is tuple
                else TypeError(
                    f"Dask fine task returned an unsupported result: {outcome!r}"
                )
            )
            rejection_reason = f"{type(error).__name__}: {error}"
            result = fine_exception_result(
                task,
                error,
                key=submission.key,
                started_at=submission.started_at,
                started=submission.started,
            )

        if attempt_observer is not None:
            attempt_observer(
                FineAttemptObservation(
                    task=task,
                    attempt=submission.attempt,
                    dask_task_key=submission.key,
                    coordinator_submitted_at=submission.started_at,
                    coordinator_ready_at=coordinator_ready_at,
                    raw_result=raw_result,
                    settled_result=result,
                    rejection_reason=rejection_reason,
                )
            )

        if result.status == "failed" and submission.attempt <= task_retries:
            completion.add(submit(task))
            return

        controller.record(task, result)
        for admitted in admit():
            completion.add(admitted)

    if completion is not None and active_wait is None:
        for future, outcome in completion:
            settle(future, outcome, utc_now())
    elif completion is not None:
        wait_for_active = active_wait
        assert wait_for_active is not None
        while active:
            batch = tuple(completion.next_batch(block=False))
            if not batch:
                wait_for_active(tuple(active))
                continue
            coordinator_ready_at = utc_now()
            if before_settle is not None:
                before_settle(tuple(active))
            for future, outcome in batch:
                settle(future, outcome, coordinator_ready_at)

    if active:
        raise RuntimeError("Dask completion ended with active fine work")
    if not controller.terminal:
        raise RuntimeError("non-terminal fine graph has neither active nor ready work")
    return (
        controller.round_results(),
        controller.task_results,
        tuple(
            (result.key.encoded, attempts[result.key.encoded])
            for result in controller.task_results
        ),
    )


__all__ = [
    "ROUND_RESOURCE",
    "FineAttemptObservation",
    "collect_distributed_fine_results",
    "collect_distributed_results",
    "dask_scheduler_config_overlay",
    "fine_task_key",
    "future_is_unfinished",
    "utc_now",
    "validate_distributed_tasks",
]
