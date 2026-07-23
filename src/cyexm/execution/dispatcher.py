"""Scheduler-neutral state transitions for fine-grained workflow execution."""

from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Literal

from cyexm.workflow.fine_models import (
    ChannelTransformTask,
    ChunkAssemblyTask,
    FineTaskKey,
    FineTaskResult,
    FineWorkflowTask,
    NonlinearChunkTask,
    RoundPreparationTask,
    RoundValidationTask,
    WholeImageRegistrationTask,
)
from cyexm.workflow.fine_planning import (
    channel_tasks,
    chunk_tasks,
    initial_fine_tasks,
)
from cyexm.workflow.models import RoundResult, RoundTask

type _NodeState = Literal[
    "ready",
    "running",
    "succeeded",
    "failed",
    "blocked",
    "cancelled",
    "never-submitted",
]

_EXACT_TASK_TYPES = (
    RoundPreparationTask,
    NonlinearChunkTask,
    ChunkAssemblyTask,
    WholeImageRegistrationTask,
    ChannelTransformTask,
    RoundValidationTask,
)
_STAGE_ORDER = {
    "preparation": 0,
    "whole-image": 0,
    "chunk": 1,
    "assembly": 2,
    "channel": 3,
    "validation": 4,
}
_SUCCESS = frozenset({"completed", "reused"})


def _canonical_key(key: FineTaskKey) -> tuple[int, int, int]:
    item = int(key.item) if key.stage in {"chunk", "channel"} else -1
    return (key.round, _STAGE_ORDER[key.stage], item)


def _parsed_timestamp(value: str, role: str) -> datetime:
    if not isinstance(value, str):
        raise TypeError(f"{role} must be an ISO-8601 string")
    encoded = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        timestamp = datetime.fromisoformat(encoded)
    except ValueError as error:
        raise ValueError(f"{role} must be a valid ISO-8601 timestamp") from error
    if timestamp.tzinfo is None:
        raise ValueError(f"{role} must include a timezone")
    return timestamp


class FineGraphController:
    """Admit immutable fine tasks fairly without owning a scheduler.

    The controller owns only dependency and settlement state. Executors remain
    responsible for running tasks, retries, cancellation, and resource lifecycle.

    Parameters
    ----------
    round_tasks : tuple of RoundTask
        Exact moving-round tasks. Round numbers must be unique.

    Notes
    -----
    ``task_results`` retains every actual terminal worker result in canonical task
    order. Blocked and never-submitted descendants never receive synthetic results.
    """

    def __init__(self, round_tasks: tuple[RoundTask, ...]) -> None:
        if not isinstance(round_tasks, tuple) or any(
            type(task) is not RoundTask for task in round_tasks
        ):
            raise TypeError("round_tasks must be a tuple of exact RoundTask values")
        if len({task.round for task in round_tasks}) != len(round_tasks):
            raise ValueError("round_tasks must use unique round numbers")
        self._round_tasks = {
            task.round: task
            for task in sorted(round_tasks, key=lambda item: item.round)
        }
        self._tasks: dict[FineTaskKey, FineWorkflowTask] = {}
        self._states: dict[FineTaskKey, _NodeState] = {}
        self._results: dict[FineTaskKey, FineTaskResult] = {}
        self._ready: dict[int, deque[FineWorkflowTask]] = {
            round_number: deque() for round_number in self._round_tasks
        }
        self._rotation: deque[int] = deque()
        self._last_admitted_round: int | None = None
        self._register(tuple(initial_fine_tasks(round_tasks)))

    def _register(self, tasks: tuple[FineWorkflowTask, ...]) -> None:
        for task in tasks:
            if type(task) not in _EXACT_TASK_TYPES:
                raise TypeError("fine graph nodes must use exact fine task types")
            if task.round not in self._round_tasks:
                raise ValueError("fine graph task belongs to an unknown round")
            if task.key in self._tasks:
                raise ValueError(f"duplicate fine task key: {task.key.encoded}")
            self._tasks[task.key] = task
            self._states[task.key] = "never-submitted"
        for task in tasks:
            self._refresh(task.key)

    def _activate_round(self, round_number: int) -> None:
        active = set(self._rotation)
        active.add(round_number)
        ordered = sorted(active)
        if self._last_admitted_round is not None:
            ordered = [
                *(item for item in ordered if item > self._last_admitted_round),
                *(item for item in ordered if item <= self._last_admitted_round),
            ]
        self._rotation = deque(ordered)

    def _refresh(self, key: FineTaskKey) -> None:
        if self._states[key] != "never-submitted":
            return
        task = self._tasks[key]
        dependency_states = tuple(self._states[item] for item in task.dependencies)
        if any(
            state in {"failed", "blocked", "cancelled"} for state in dependency_states
        ):
            self._states[key] = "blocked"
            self._block_descendants(key)
        elif all(state == "succeeded" for state in dependency_states):
            self._states[key] = "ready"
            self._ready[task.round].append(task)
            self._activate_round(task.round)

    def _block_descendants(self, failed_key: FineTaskKey) -> None:
        for key, task in self._tasks.items():
            if failed_key in task.dependencies and self._states[key] in {
                "never-submitted",
                "ready",
            }:
                if self._states[key] == "ready":
                    self._ready[task.round] = deque(
                        item for item in self._ready[task.round] if item.key != key
                    )
                self._states[key] = "blocked"
                self._block_descendants(key)

    def pop_ready(self, limit: int) -> tuple[FineWorkflowTask, ...]:
        """Admit at most ``limit`` ready tasks in round-robin order."""
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be a positive integer")
        if limit < 1:
            raise ValueError("limit must be a positive integer")
        admitted: list[FineWorkflowTask] = []
        while self._rotation and len(admitted) < limit:
            round_number = self._rotation.popleft()
            queue = self._ready[round_number]
            if not queue:
                continue
            task = queue.popleft()
            if self._states[task.key] != "ready":
                raise RuntimeError("fine graph ready queue contains a non-ready task")
            self._states[task.key] = "running"
            admitted.append(task)
            self._last_admitted_round = round_number
            if queue:
                self._activate_round(round_number)
        return tuple(admitted)

    def record(self, task: FineWorkflowTask, result: FineTaskResult) -> None:
        """Settle one currently running exact task result."""
        if type(task) not in _EXACT_TASK_TYPES:
            raise TypeError("task must be an exact fine workflow task")
        if type(result) is not FineTaskResult:
            raise TypeError("result must be an exact FineTaskResult")
        if task.key not in self._tasks or self._tasks[task.key] != task:
            raise ValueError("task does not match a node in this fine graph")
        if result.task != task:
            raise ValueError("result must match the exact recorded task")
        if self._states[task.key] != "running":
            raise ValueError("record requires a currently running task")
        if task.key in self._results:
            raise ValueError("fine task result was already recorded")

        self._results[task.key] = result
        if result.status in _SUCCESS:
            self._states[task.key] = "succeeded"
            if type(task) is RoundPreparationTask:
                self._register(tuple(chunk_tasks(result)))
            elif type(task) in {ChunkAssemblyTask, WholeImageRegistrationTask}:
                self._register(tuple(channel_tasks(result)))
        elif result.status == "failed":
            self._states[task.key] = "failed"
            self._block_descendants(task.key)
        else:
            self._states[task.key] = "cancelled"
            self._block_descendants(task.key)

        for key, candidate in tuple(self._tasks.items()):
            if task.key in candidate.dependencies:
                self._refresh(key)

    @property
    def task_results(self) -> tuple[FineTaskResult, ...]:
        """Return actual worker results in deterministic canonical task order."""
        return tuple(
            self._results[key] for key in sorted(self._results, key=_canonical_key)
        )

    def _round_terminal(self, round_number: int) -> bool:
        states = tuple(
            state for key, state in self._states.items() if key.round == round_number
        )
        return bool(states) and all(
            state in {"succeeded", "failed", "blocked", "cancelled"} for state in states
        )

    @property
    def terminal(self) -> bool:
        """Whether all reachable work has settled or become blocked."""
        return all(
            self._round_terminal(round_number) for round_number in self._round_tasks
        )

    def _round_result(self, round_number: int) -> RoundResult:
        if not self._round_terminal(round_number):
            raise RuntimeError("round result requires a terminal fine graph round")
        round_task = self._round_tasks[round_number]
        actual = tuple(
            result for result in self.task_results if result.round == round_number
        )
        if not actual:
            raise RuntimeError("terminal fine graph round has no actual task results")
        thread_evidence = actual[0].native_threads
        if any(result.native_threads != thread_evidence for result in actual[1:]):
            raise ValueError(
                f"round {round_number} has inconsistent native-thread evidence"
            )
        intervals = tuple(
            (
                _parsed_timestamp(result.started_at, "started_at"),
                _parsed_timestamp(result.ended_at, "ended_at"),
                result,
            )
            for result in actual
        )
        if any(ended < started for started, ended, _result in intervals):
            raise ValueError("fine task end timestamp precedes its start timestamp")
        started, started_result = min(
            ((started, result) for started, _ended, result in intervals),
            key=lambda item: (item[0], _canonical_key(item[1].key)),
        )
        ended, ended_result = max(
            ((ended, result) for _started, ended, result in intervals),
            key=lambda item: (item[0], _canonical_key(item[1].key)),
        )
        duration = (ended - started).total_seconds()
        if duration < 0:
            raise ValueError("fine round end timestamp precedes its start timestamp")

        validation = next(
            (
                result
                for result in actual
                if type(result.task) is RoundValidationTask
                and result.status in _SUCCESS
            ),
            None,
        )
        failed = tuple(result for result in actual if result.status == "failed")
        cancelled = tuple(result for result in actual if result.status == "cancelled")
        if failed:
            representative = min(failed, key=lambda item: _canonical_key(item.key))
            status: Literal["reused", "completed", "failed", "cancelled"] = "failed"
            failure = representative.failure
        elif cancelled:
            representative = min(cancelled, key=lambda item: _canonical_key(item.key))
            status = "cancelled"
            failure = None
        elif validation is not None:
            representative = validation
            failure = None
            producer_reuse = {
                path
                for result in actual
                if type(result.task) is not RoundValidationTask
                for path in result.reused_artifacts
            }
            status = (
                "reused"
                if set(validation.artifact_paths).issubset(producer_reuse)
                else "completed"
            )
        else:
            raise RuntimeError(
                "terminal successful fine round lacks validation evidence"
            )

        validation_task = RoundValidationTask.create(
            round_task,
            dependencies=tuple(
                FineTaskKey(round_number, "channel", f"{index:04d}")
                for index in range(len(round_task.channels))
            ),
        )
        public_order = tuple(str(path) for path in validation_task.output_paths)
        reported = {path for result in actual for path in result.artifact_paths}
        reused_reported = {
            path
            for result in actual
            if type(result.task) is not RoundValidationTask
            for path in result.reused_artifacts
        }
        artifact_paths = (
            validation.artifact_paths
            if validation is not None and status in _SUCCESS
            else tuple(path for path in public_order if path in reported)
        )
        reused_artifacts = tuple(
            path for path in artifact_paths if path in reused_reported
        )
        return RoundResult(
            dataset_id=round_task.dataset_id,
            round=round_number,
            status=status,
            started_at=started_result.started_at,
            ended_at=ended_result.ended_at,
            duration_seconds=duration,
            hostname=representative.hostname,
            pid=representative.pid,
            dask_task_key=representative.dask_task_key,
            slurm_job_id=representative.slurm_job_id,
            strategy=round_task.strategy,
            native_threads=thread_evidence,
            artifact_paths=artifact_paths,
            reused_artifacts=reused_artifacts,
            log_path=representative.log_path,
            failure=failure,
        )

    def _settled_round_results(self) -> tuple[RoundResult, ...]:
        return tuple(
            self._round_result(round_number)
            for round_number in self._round_tasks
            if self._round_terminal(round_number)
        )

    def round_results(self) -> tuple[RoundResult, ...]:
        """Summarize every round after all reachable graph work settles."""
        if not self.terminal:
            raise RuntimeError("round_results requires a terminal fine graph")
        return self._settled_round_results()


__all__ = ("FineGraphController",)
