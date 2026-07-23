"""Scheduler-neutral state transitions for fine-grained workflow execution."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
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
type FineExpansionKind = Literal["chunks", "channels"]

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
_STOPPED = frozenset({"failed", "cancelled"})
_EXPANSION_BY_STAGE: dict[str, FineExpansionKind] = {
    "preparation": "chunks",
    "assembly": "channels",
    "whole-image": "channels",
}


def _canonical_key(key: FineTaskKey) -> tuple[int, int, int]:
    item = int(key.item) if key.stage in {"chunk", "channel"} else -1
    return (key.round, _STAGE_ORDER[key.stage], item)


def canonical_fine_task_key(value: str) -> tuple[int, int, int]:
    """Return scheduler-neutral canonical ordering for an encoded fine key."""
    return _canonical_key(FineTaskKey.from_encoded(value))


def fine_expansion_kind(key: FineTaskKey) -> FineExpansionKind | None:
    """Return the one controller-owned successor expansion for a task key."""
    return _EXPANSION_BY_STAGE.get(key.stage)


def _expand_successors(
    task: FineWorkflowTask,
    result: FineTaskResult,
) -> tuple[FineWorkflowTask, ...]:
    kind = fine_expansion_kind(task.key)
    if kind == "chunks":
        return tuple(chunk_tasks(result))
    if kind == "channels":
        return tuple(channel_tasks(result))
    return ()


def _successors_may_be_absent(
    key: FineTaskKey,
    status: str | None,
    *,
    external_stop: bool,
) -> bool:
    if fine_expansion_kind(key) is None:
        raise ValueError("fine task does not own a successor expansion")
    return status in _STOPPED or (status is None and external_stop)


def validate_fine_graph_description(
    description: Sequence[tuple[str, tuple[str, ...]]],
    round_tasks: tuple[RoundTask, ...],
    *,
    allow_partial: bool,
    terminal_statuses: Mapping[str, str],
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    """Validate scheduler-neutral materialized graph shape and canonical order."""
    task_keys = tuple(key for key, _dependencies in description)
    dependencies = dict(description)
    if len(dependencies) != len(task_keys):
        raise ValueError("fine graph task keys must be unique")
    decoded = {key: FineTaskKey.from_encoded(key) for key in task_keys}
    if task_keys != tuple(sorted(task_keys, key=canonical_fine_task_key)):
        raise ValueError("fine graph tasks must use canonical ordering")
    if any(
        dependency not in dependencies
        for values in dependencies.values()
        for dependency in values
    ):
        raise ValueError("fine graph dependencies must name graph task keys")
    for key in terminal_statuses:
        if key not in dependencies:
            continue
        if any(
            terminal_statuses.get(dependency) not in _SUCCESS
            for dependency in dependencies[key]
        ):
            raise ValueError(
                "attempted fine graph tasks require successful terminal dependencies"
            )
    if tuple(sorted({key.round for key in decoded.values()})) != tuple(
        sorted(task.round for task in round_tasks)
    ):
        raise ValueError("fine graph rounds must match submitted moving rounds")

    for round_task in round_tasks:
        round_keys = tuple(
            key for key in task_keys if decoded[key].round == round_task.round
        )
        stages = {
            stage: tuple(key for key in round_keys if decoded[key].stage == stage)
            for stage in _STAGE_ORDER
        }
        prefix = f"round-{round_task.round:03d}:"
        channels = tuple(
            f"{prefix}channel:{index:04d}" for index in range(len(round_task.channels))
        )
        validation = f"{prefix}validation:main"
        if round_task.strategy == "whole_image":
            producer = f"{prefix}whole-image:main"
            if stages["whole-image"] != (producer,) or any(
                stages[stage] for stage in ("preparation", "chunk", "assembly")
            ):
                raise ValueError("whole-image fine graph has incompatible stages")
            if dependencies[producer]:
                raise ValueError("whole-image task must have no dependencies")
            producer_status = terminal_statuses.get(producer)
            producer_blocked = False
        else:
            preparation = f"{prefix}preparation:main"
            producer = f"{prefix}assembly:main"
            chunks = stages["chunk"]
            if stages["preparation"] != (preparation,) or stages["whole-image"]:
                raise ValueError("chunked fine graph has incompatible stages")
            preparation_status = terminal_statuses.get(preparation)
            if not chunks and _successors_may_be_absent(
                decoded[preparation],
                preparation_status,
                external_stop=allow_partial,
            ):
                if stages["assembly"] or stages["channel"] or stages["validation"]:
                    raise ValueError("partial chunk graph contains downstream stages")
                continue
            if not chunks or stages["assembly"] != (producer,):
                raise ValueError("chunked fine graph has incompatible stages")
            if preparation_status not in _SUCCESS:
                raise ValueError(
                    "materialized chunk expansion requires successful preparation"
                )
            if dependencies[preparation] or any(
                dependencies[key] != (preparation,) for key in chunks
            ):
                raise ValueError("fine chunks must depend only on preparation")
            if dependencies[producer] != chunks:
                raise ValueError("fine assembly must depend on every chunk")
            producer_status = terminal_statuses.get(producer)
            producer_blocked = any(
                terminal_statuses.get(key) in _STOPPED for key in chunks
            )
        if (
            _successors_may_be_absent(
                decoded[producer],
                producer_status,
                external_stop=allow_partial or producer_blocked,
            )
            and not stages["channel"]
            and not stages["validation"]
        ):
            continue
        if stages["channel"] != channels or stages["validation"] != (validation,):
            raise ValueError("fine graph channels/validation must match the round")
        if producer_status not in _SUCCESS:
            raise ValueError(
                "materialized channel expansion requires successful producer"
            )
        if dependencies[validation] != channels:
            raise ValueError("fine validation must depend on every channel task")
        if any(dependencies[key] != (producer,) for key in channels):
            raise ValueError("fine channels must depend only on registration producer")
    return task_keys, dependencies


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


def _summarize_fine_round(
    round_task: RoundTask,
    actual: tuple[FineTaskResult, ...],
) -> RoundResult:
    """Aggregate exact granular results into the canonical public round summary."""
    if type(round_task) is not RoundTask:
        raise TypeError("fine round summary requires an exact RoundTask")
    if type(actual) is not tuple or not actual:
        raise RuntimeError("fine round summary requires actual task results")
    if any(
        type(result) is not FineTaskResult
        or result.round != round_task.round
        or result.task.round_task != round_task
        for result in actual
    ):
        raise ValueError(
            "fine round summary results must bind to the exact authoritative round"
        )
    thread_evidence = actual[0].native_threads
    if any(result.native_threads != thread_evidence for result in actual[1:]):
        raise ValueError(
            f"round {round_task.round} has inconsistent native-thread evidence"
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
            if type(result.task) is RoundValidationTask and result.status in _SUCCESS
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
        raise RuntimeError("terminal successful fine round lacks validation evidence")

    validation_task = RoundValidationTask.create(
        round_task,
        dependencies=tuple(
            FineTaskKey(round_task.round, "channel", f"{index:04d}")
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
    reused_artifacts = tuple(path for path in artifact_paths if path in reused_reported)
    return RoundResult(
        dataset_id=round_task.dataset_id,
        round=round_task.round,
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
            self._register(_expand_successors(task, result))
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

    @property
    def ready_count(self) -> int:
        """Return the exact number of dependency-ready, unsubmitted tasks."""
        return sum(len(queue) for queue in self._ready.values())

    @property
    def tasks(self) -> tuple[FineWorkflowTask, ...]:
        """Return every currently materialized graph node canonically."""
        return tuple(
            self._tasks[key] for key in sorted(self._tasks, key=_canonical_key)
        )

    @property
    def graph_description(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Return immutable canonical task/dependency evidence."""
        return tuple(
            (
                task.key.encoded,
                tuple(dependency.encoded for dependency in task.dependencies),
            )
            for task in self.tasks
        )

    @property
    def blocked_keys(self) -> tuple[FineTaskKey, ...]:
        """Return dependency-blocked descendants in canonical order."""
        return tuple(
            key
            for key in sorted(self._states, key=_canonical_key)
            if self._states[key] == "blocked"
        )

    def _round_result(self, round_number: int) -> RoundResult:
        if not self._round_terminal(round_number):
            raise RuntimeError("round result requires a terminal fine graph round")
        round_task = self._round_tasks[round_number]
        actual = tuple(
            result for result in self.task_results if result.round == round_number
        )
        return _summarize_fine_round(round_task, actual)

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
