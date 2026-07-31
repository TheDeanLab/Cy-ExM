"""Typed fine-grained SLURM lifecycle and strict evidence policy."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import cast

from cyexm.errors import WorkerPreflightError
from cyexm.execution.dispatcher import (
    FineGraphController,
    canonical_fine_task_key,
    validate_fine_graph_description,
)
from cyexm.execution.preflight import (
    WorkerProbe,
    validate_slurm_worker_identities,
)
from cyexm.workflow.fine_models import (
    _FINE_TASK_RESULT_KEYS,
    FineTaskKey,
    FineTaskResult,
    FineWorkflowTask,
)
from cyexm.workflow.models import RoundFailure, RoundTask

_GRAPH_KEYS = frozenset(
    {
        "mode",
        "worker_cap",
        "candidate_workers",
        "validated_workers",
        "scale_events",
        "task_graph",
        "attempt_counts",
        "placements",
        "task_results",
        "blocked_descendants",
        "cancelled",
        "interruption",
        "pre_controller_terminal",
    }
)


def _object(
    value: object,
    location: str,
    keys: frozenset[str],
) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{location} must be an exact JSON object")
    result = cast(dict[object, object], value)
    if any(type(key) is not str for key in result):
        raise TypeError(f"{location} keys must be exact built-in strings")
    decoded = cast(dict[str, object], result)
    if frozenset(decoded) != keys:
        raise ValueError(f"{location} keys must be exactly {sorted(keys)!r}")
    return decoded


def _array(value: object, location: str) -> list[object]:
    if type(value) is not list:
        raise TypeError(f"{location} must be an exact JSON array")
    return cast(list[object], value)


def _string(value: object, location: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{location} must be a nonempty exact built-in string")
    return value


def _integer(value: object, location: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise TypeError(f"{location} must be an exact integer of at least {minimum}")
    return value


def _optional_string(value: object, location: str) -> str | None:
    if value is None:
        return None
    return _string(value, location)


def _optional_integer(value: object, location: str) -> int | None:
    if value is None:
        return None
    return _integer(value, location, 1)


def _number(value: object, location: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value < 0
    ):
        raise TypeError(f"{location} must be a finite non-negative number")
    return float(value)


def _timestamp(value: object, location: str) -> datetime:
    encoded = _string(value, location)
    if not encoded.endswith("Z"):
        raise ValueError(f"{location} must use canonical UTC Z notation")
    try:
        result = datetime.fromisoformat(encoded[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{location} must be an ISO 8601 timestamp") from error
    if result.utcoffset() != UTC.utcoffset(result):
        raise ValueError(f"{location} must be UTC")
    return result


def _strings(value: object, location: str) -> tuple[str, ...]:
    result = tuple(
        _string(item, f"{location}[{index}]")
        for index, item in enumerate(_array(value, location))
    )
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{location} must be unique and canonically sorted")
    return result


def _fine_keys(value: object, location: str) -> tuple[str, ...]:
    result = tuple(
        _string(item, f"{location}[{index}]")
        for index, item in enumerate(_array(value, location))
    )
    if len(result) != len(set(result)):
        raise ValueError(f"{location} must contain unique task keys")
    if result != tuple(sorted(result, key=canonical_fine_task_key)):
        raise ValueError(f"{location} must use canonical fine-task ordering")
    return result


@dataclass(slots=True)
class FineScaleEvent:
    """One deterministic upward-scale transition."""

    requested_jobs: int
    expected_processes: int
    started_at: str
    connected: list[str] = field(default_factory=list)
    departed: list[str] = field(default_factory=list)
    probed: list[str] = field(default_factory=list)
    activated: list[str] = field(default_factory=list)
    activated_probes: list[WorkerProbe] = field(default_factory=list)
    failed: list[dict[str, str]] = field(default_factory=list)
    ended_at: str | None = None

    def fail(self, error: BaseException, ended_at: str) -> None:
        self.failed = [{"type": type(error).__name__, "message": str(error)}]
        self.ended_at = ended_at

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "requested_jobs": self.requested_jobs,
            "expected_processes": self.expected_processes,
            "connected": sorted(self.connected),
            "departed": sorted(self.departed),
            "probed": sorted(self.probed),
            "activated": sorted(self.activated),
            "activated_probes": [
                probe.to_dict()
                for probe in sorted(
                    self.activated_probes,
                    key=lambda item: (
                        item.worker_address,
                        item.slurm_job_id,
                        item.hostname,
                        item.pid,
                    ),
                )
            ],
            "failed": [dict(item) for item in self.failed],
            "started_at": self.started_at,
        }
        if self.ended_at is not None:
            result["ended_at"] = self.ended_at
        return result


@dataclass(frozen=True, slots=True)
class PreControllerTerminal:
    """Exact cause proving why no moving fine controller could run."""

    phase: str
    status: str
    exception_type: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "phase": self.phase,
            "status": self.status,
            "exception_type": self.exception_type,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class FinePlacementAttempt:
    """Detached claim and admission decision for one actual task attempt."""

    task_key: str
    attempt: int
    status: str
    dask_task_key: str
    coordinator_submitted_at: str
    coordinator_ready_at: str
    started_at: str
    ended_at: str
    duration_seconds: float
    worker_address: str | None
    hostname: str | None
    pid: int | None
    slurm_job_id: str | None
    native_threads: tuple[tuple[str, int], ...]
    artifact_paths: tuple[str, ...]
    reused_artifacts: tuple[str, ...]
    log_path: str
    placement_accepted: bool
    rejection_reason: str | None
    failure: RoundFailure | None

    def to_dict(self) -> dict[str, object]:
        return {
            "task_key": self.task_key,
            "attempt": self.attempt,
            "status": self.status,
            "dask_task_key": self.dask_task_key,
            "coordinator_submitted_at": self.coordinator_submitted_at,
            "coordinator_ready_at": self.coordinator_ready_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": self.duration_seconds,
            "worker_address": self.worker_address,
            "hostname": self.hostname,
            "pid": self.pid,
            "slurm_job_id": self.slurm_job_id,
            "native_threads": [list(item) for item in self.native_threads],
            "artifact_paths": list(self.artifact_paths),
            "reused_artifacts": list(self.reused_artifacts),
            "log_path": self.log_path,
            "placement_accepted": self.placement_accepted,
            "rejection_reason": self.rejection_reason,
            "failure": None if self.failure is None else self.failure.to_dict(),
        }


@dataclass(slots=True)
class FineSlurmEvidence:
    """Mutable coordinator state with detached JSON snapshots."""

    job_cap: int
    processes_per_job: int
    worker_cap: int
    candidates: set[str] = field(default_factory=set)
    validated: set[str] = field(default_factory=set)
    events: list[FineScaleEvent] = field(default_factory=list)
    controller: FineGraphController | None = None
    attempts: tuple[tuple[str, int], ...] = ()
    results: tuple[FineTaskResult, ...] = ()
    probes: dict[str, WorkerProbe] = field(default_factory=dict)
    placement_attempts: list[FinePlacementAttempt] = field(default_factory=list)
    cancelled: bool = False
    interruption: tuple[str, str] | None = None
    pre_controller_terminal: PreControllerTerminal | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("job_cap", self.job_cap),
            ("processes_per_job", self.processes_per_job),
            ("worker_cap", self.worker_cap),
        ):
            if type(value) is not int or value < 1:
                raise TypeError(f"{name} must be a positive built-in integer")
        if self.worker_cap != self.job_cap * self.processes_per_job:
            raise ValueError(
                "worker_cap must equal job_cap * processes_per_job"
            )

    def begin_scale(self, requested_jobs: int, started_at: str) -> FineScaleEvent:
        if type(requested_jobs) is not int or not 1 <= requested_jobs <= self.job_cap:
            raise ValueError("requested_jobs must be within the configured job cap")
        event = FineScaleEvent(
            requested_jobs=requested_jobs,
            expected_processes=requested_jobs * self.processes_per_job,
            started_at=started_at,
        )
        self.events.append(event)
        return event

    def finalize_graph(
        self,
        controller: FineGraphController,
        attempts: tuple[tuple[str, int], ...],
        results: tuple[FineTaskResult, ...],
        probes: Mapping[str, WorkerProbe],
    ) -> None:
        self.controller = controller
        self.attempts = attempts
        self.results = results
        self.probes = dict(probes)

    def record_attempt(self, attempt: FinePlacementAttempt) -> None:
        self.placement_attempts.append(attempt)

    def to_dict(self) -> dict[str, object]:
        graph = () if self.controller is None else self.controller.graph_description
        attempt_counts: dict[str, int] = {}
        for attempt in self.placement_attempts:
            attempt_counts[attempt.task_key] = max(
                attempt_counts.get(attempt.task_key, 0),
                attempt.attempt,
            )
        blocked = (
            ()
            if self.controller is None
            else tuple(key.encoded for key in self.controller.blocked_keys)
        )
        return {
            "mode": "fine_grained",
            "worker_cap": self.worker_cap,
            "candidate_workers": sorted(self.candidates),
            "validated_workers": sorted(self.validated),
            "scale_events": [event.to_dict() for event in self.events],
            "task_graph": [
                {"key": key, "dependencies": list(dependencies)}
                for key, dependencies in graph
            ],
            "attempt_counts": [
                {"task_key": key, "attempts": attempt_counts[key]}
                for key in sorted(attempt_counts, key=canonical_fine_task_key)
            ],
            "placements": [
                attempt.to_dict()
                for attempt in sorted(
                    self.placement_attempts,
                    key=lambda item: (
                        canonical_fine_task_key(item.task_key),
                        item.attempt,
                    ),
                )
            ],
            "task_results": [result.to_dict() for result in self.results],
            "blocked_descendants": sorted(blocked, key=canonical_fine_task_key),
            "cancelled": self.cancelled,
            "interruption": (
                None
                if self.interruption is None
                else {
                    "exception_type": self.interruption[0],
                    "message": self.interruption[1],
                }
            ),
            "pre_controller_terminal": (
                None
                if self.pre_controller_terminal is None
                else self.pre_controller_terminal.to_dict()
            ),
        }


def _decode_task_graph(raw_graph: object) -> tuple[tuple[str, tuple[str, ...]], ...]:
    description: list[tuple[str, tuple[str, ...]]] = []
    seen: set[str] = set()
    for index, raw in enumerate(_array(raw_graph, "fine_graph.task_graph")):
        task = _object(
            raw,
            f"fine_graph.task_graph[{index}]",
            frozenset({"key", "dependencies"}),
        )
        key = _string(task["key"], f"fine_graph.task_graph[{index}].key")
        if key in seen:
            raise ValueError(f"duplicate fine graph task key: {key!r}")
        values = tuple(
            _string(item, f"fine_graph.task_graph[{index}].dependencies[{item_index}]")
            for item_index, item in enumerate(
                _array(
                    task["dependencies"],
                    f"fine_graph.task_graph[{index}].dependencies",
                )
            )
        )
        if len(values) != len(set(values)):
            raise ValueError("fine graph task dependencies must be unique")
        seen.add(key)
        description.append((key, values))
    return tuple(description)


def _decode_task_results(
    value: object,
    tasks: tuple[FineWorkflowTask, ...],
) -> tuple[dict[str, object], ...]:
    results: list[dict[str, object]] = []
    keys: list[str] = []
    task_by_key = {task.key.encoded: task for task in tasks}
    for index, raw in enumerate(_array(value, "fine_graph.task_results")):
        location = f"fine_graph.task_results[{index}]"
        result = _object(raw, location, _FINE_TASK_RESULT_KEYS)
        key = FineTaskKey.from_encoded(_string(result["key"], f"{location}.key"))
        task = task_by_key.get(key.encoded)
        if tasks and task is None:
            raise ValueError(f"{location} is not bound to an exact controller task")
        if task is not None:
            from cyexm.workflow._execution_validation import (
                _validate_fine_task_result,
            )

            decoded = FineTaskResult.from_dict(task, result)
            _validate_fine_task_result(task, decoded)
        keys.append(key.encoded)
        results.append(result)
    if len(keys) != len(set(keys)):
        raise ValueError("fine task result keys must be unique")
    if tuple(keys) != tuple(sorted(keys, key=canonical_fine_task_key)):
        raise ValueError("fine task results must use canonical task ordering")
    return tuple(results)


def validate_fine_slurm_evidence(
    value: object,
    *,
    job_cap: int,
    processes_per_job: int,
    worker_cap: int,
    expected_process_memory_bytes: int,
    task_retries: int,
    cancellation_present: bool,
    startup_outcome: str,
    reference_attempted: bool,
    round_tasks: tuple[RoundTask, ...],
    observed_workers: Sequence[str],
    observed_probes: Sequence[WorkerProbe],
    fine_tasks: tuple[FineWorkflowTask, ...] = (),
) -> None:
    """Strictly validate one version-3 fine SLURM evidence object."""
    graph = _object(value, "execution_evidence.fine_graph", _GRAPH_KEYS)
    if _string(graph["mode"], "fine_graph.mode") != "fine_grained":
        raise ValueError("fine_graph.mode must be 'fine_grained'")
    if worker_cap != job_cap * processes_per_job:
        raise ValueError("worker_cap must equal job_cap * processes_per_job")
    if _integer(graph["worker_cap"], "fine_graph.worker_cap", 1) != worker_cap:
        raise ValueError("fine_graph.worker_cap must match max_task_slots")
    candidates = _strings(graph["candidate_workers"], "fine_graph.candidates")
    validated = _strings(graph["validated_workers"], "fine_graph.validated")
    if not set(validated) <= set(candidates):
        raise ValueError("validated workers must be candidate workers")

    activated: set[str] = set()
    current_activated_probes: dict[str, WorkerProbe] = {}
    historical_activated_probes: list[WorkerProbe] = []
    active_intervals: dict[
        tuple[str, str, int, str | None],
        tuple[datetime, datetime | None],
    ] = {}
    probed_workers: set[str] = set()
    connected: set[str] = set()
    scale_probe_outcomes: list[tuple[set[str], set[str], bool]] = []
    previous_request = 0
    scale_failed = False
    for index, raw in enumerate(_array(graph["scale_events"], "fine_graph.events")):
        location = f"fine_graph.scale_events[{index}]"
        event = _object(
            raw,
            location,
            frozenset(
                {
                    "requested_jobs",
                    "expected_processes",
                    "connected",
                    "departed",
                    "probed",
                    "activated",
                    "activated_probes",
                    "failed",
                    "started_at",
                    "ended_at",
                }
            ),
        )
        requested = _integer(
            event["requested_jobs"],
            f"{location}.requested_jobs",
            1,
        )
        expected_processes = _integer(
            event["expected_processes"],
            f"{location}.expected_processes",
            1,
        )
        if expected_processes != requested * processes_per_job:
            raise ValueError(
                "fine scale expected_processes must equal "
                "requested_jobs * processes_per_job"
            )
        event_started = _timestamp(event["started_at"], f"{location}.started_at")
        event_ended = _timestamp(event["ended_at"], f"{location}.ended_at")
        if event_ended < event_started:
            raise ValueError("fine scale event ended before it started")
        departed = _strings(event["departed"], f"{location}.departed")
        replacement = bool(departed)
        event_failures = _array(event["failed"], f"{location}.failed")
        health_failure = (
            requested == previous_request and not replacement and bool(event_failures)
        )
        if (
            requested > job_cap
            or expected_processes > worker_cap
            or requested < previous_request
            or (
                requested == previous_request and not replacement and not health_failure
            )
        ):
            raise ValueError(
                "fine scale requests must increase, except explicit replacement"
            )
        previous_request = requested
        event_connected = _strings(event["connected"], f"{location}.connected")
        probed = _strings(event["probed"], f"{location}.probed")
        event_activated = _strings(event["activated"], f"{location}.activated")
        raw_activated_probes = _array(
            event["activated_probes"],
            f"{location}.activated_probes",
        )
        parsed_activated_probes: list[WorkerProbe] = []
        for probe_index, raw_probe in enumerate(raw_activated_probes):
            if not isinstance(raw_probe, Mapping):
                raise TypeError(
                    f"{location}.activated_probes[{probe_index}] must be an object"
                )
            parsed_activated_probes.append(
                WorkerProbe.from_dict(cast(Mapping[str, object], raw_probe))
            )
        event_activated_probes = tuple(parsed_activated_probes)
        if tuple(
            probe.worker_address for probe in event_activated_probes
        ) != event_activated or any(not probe.ok for probe in event_activated_probes):
            raise ValueError(
                "fine activated probes must exactly authenticate activated workers"
            )
        if not set(event_activated) <= set(probed) <= set(event_connected):
            raise ValueError("fine scale activation must be a subset of probed workers")
        if health_failure and (
            event_connected or probed or event_activated or event_activated_probes
        ):
            raise ValueError("fine health failure cannot claim a worker transition")
        if not set(departed) <= connected:
            raise ValueError("fine replacement can only remove connected workers")
        connected.difference_update(departed)
        activated.difference_update(departed)
        for address in departed:
            departed_probe = current_activated_probes.pop(address, None)
            if departed_probe is not None:
                identity = (
                    departed_probe.worker_address,
                    departed_probe.hostname,
                    departed_probe.pid,
                    departed_probe.slurm_job_id,
                )
                activated_at, previous_departure = active_intervals[identity]
                if previous_departure is not None or event_started < activated_at:
                    raise ValueError(
                        "fine worker departure contradicts activation interval"
                    )
                active_intervals[identity] = (activated_at, event_started)
        connected.update(event_connected)
        probed_workers.update(probed)
        event_failed = bool(event_failures)
        candidate_activated_probes = {
            **current_activated_probes,
            **{
                probe.worker_address: probe
                for probe in event_activated_probes
            },
        }
        activation_complete = not candidate_activated_probes
        if candidate_activated_probes:
            try:
                validate_slurm_worker_identities(
                    tuple(candidate_activated_probes.values()),
                    processes_per_job=processes_per_job,
                    expected_process_memory_bytes=(
                        expected_process_memory_bytes
                    ),
                )
            except WorkerPreflightError:
                activation_complete = False
            else:
                activation_complete = True
        if not event_failed and not activation_complete:
            raise ValueError(
                "fine activated workers do not form complete SLURM job groups"
            )
        if activation_complete:
            activated.update(event_activated)
        for probe in event_activated_probes:
            identity = (
                probe.worker_address,
                probe.hostname,
                probe.pid,
                probe.slurm_job_id,
            )
            if identity in active_intervals:
                raise ValueError("fine activated worker identity cannot be reused")
            historical_activated_probes.append(probe)
            active_intervals[identity] = (
                event_ended,
                event_ended if event_failed else None,
            )
            if activation_complete:
                current_activated_probes[probe.worker_address] = probe
        for failure_index, failure_value in enumerate(event_failures):
            scale_failed = True
            error = _object(
                failure_value,
                f"{location}.failed[{failure_index}]",
                frozenset({"type", "message"}),
            )
            _string(error["type"], f"{location}.failed[{failure_index}].type")
            if type(error["message"]) is not str:
                raise TypeError("fine scale failure message must be a string")
        scale_probe_outcomes.append(
            (set(probed), set(event_activated), bool(event_failures))
        )
    if tuple(sorted(connected)) != candidates:
        raise ValueError("scale-event candidates must match candidate_workers")
    if tuple(sorted(connected)) != tuple(observed_workers):
        raise ValueError(
            "scale-event candidates must exactly match observed connected workers"
        )
    if tuple(sorted(activated)) != validated:
        raise ValueError("scale-event activations must match validated_workers")

    terminal_raw = graph["pre_controller_terminal"]
    terminal: dict[str, object] | None = None
    if terminal_raw is not None:
        terminal = _object(
            terminal_raw,
            "fine_graph.pre_controller_terminal",
            frozenset({"phase", "status", "exception_type", "message"}),
        )
        phase = _string(terminal["phase"], "pre_controller_terminal.phase")
        status = _string(terminal["status"], "pre_controller_terminal.status")
        if phase == "moving":
            if type(terminal["exception_type"]) is not str:
                raise TypeError(
                    "pre_controller_terminal.exception_type must be a string"
                )
        else:
            _string(
                terminal["exception_type"],
                "pre_controller_terminal.exception_type",
            )
        if type(terminal["message"]) is not str:
            raise TypeError("pre_controller_terminal.message must be a string")
        if phase not in {"startup", "reference", "moving"}:
            raise ValueError("pre_controller_terminal phase is unsupported")
        if status not in {"failed", "cancelled", "reused"}:
            raise ValueError("pre_controller_terminal status is unsupported")
        if phase == "startup" and startup_outcome not in {"failed", "interrupted"}:
            raise ValueError("startup terminal cause contradicts startup outcome")
        if phase == "reference" and (
            startup_outcome != "succeeded" or not reference_attempted
        ):
            raise ValueError(
                "reference terminal cause lacks successful startup/attempt"
            )
        if phase == "moving":
            reused_terminal = (
                status == "reused"
                and not cancellation_present
                and not round_tasks
                and terminal["exception_type"] == ""
                and bool(terminal["message"])
            )
            cancelled_terminal = (
                status == "cancelled"
                and cancellation_present
                and bool(round_tasks)
                and bool(terminal["exception_type"])
            )
            if (
                startup_outcome != "succeeded"
                or not reference_attempted
                or (not reused_terminal and not cancelled_terminal)
            ):
                raise ValueError(
                    "moving terminal cause contradicts reference, pending work, "
                    "or cancellation"
                )
    raw_task_graph = _array(graph["task_graph"], "fine_graph.task_graph")
    if not raw_task_graph:
        if terminal is None:
            raise ValueError("empty fine graph requires an exact pre-controller cause")
        task_keys: tuple[str, ...] = ()
        dependencies: dict[str, tuple[str, ...]] = {}
    else:
        if terminal is not None:
            raise ValueError(
                "materialized fine graph cannot claim pre-controller cause"
            )
        task_keys, dependencies = validate_fine_graph_description(
            _decode_task_graph(raw_task_graph),
            round_tasks,
            allow_partial=cancellation_present or scale_failed,
            terminal_statuses={
                cast(str, item["task_key"]): cast(str, item["status"])
                for raw in _array(graph["placements"], "fine_graph.placements")
                if isinstance(raw, Mapping)
                for item in [raw]
                if type(item.get("task_key")) is str and type(item.get("status")) is str
            },
        )
    attempts: dict[str, int] = {}
    for index, raw in enumerate(
        _array(graph["attempt_counts"], "fine_graph.attempt_counts")
    ):
        item = _object(
            raw,
            f"fine_graph.attempt_counts[{index}]",
            frozenset({"task_key", "attempts"}),
        )
        key = _string(item["task_key"], "fine attempt task_key")
        if key in attempts or key not in dependencies:
            raise ValueError("fine attempt identity must be unique and in the graph")
        count = _integer(item["attempts"], "fine attempt count", 1)
        if count > task_retries + 1:
            raise ValueError("fine attempt count exceeds retry policy")
        attempts[key] = count
    if tuple(attempts) != tuple(key for key in task_keys if key in attempts):
        raise ValueError("fine attempts must follow canonical graph ordering")

    probes = tuple(observed_probes)
    probe_identities = {
        (
            probe.worker_address,
            probe.hostname,
            probe.pid,
            probe.slurm_job_id,
        ): probe
        for probe in probes
    }
    if len(probe_identities) != len(probes):
        raise ValueError("raw worker probes must use unique full identities")
    if {probe.worker_address for probe in probes} != probed_workers:
        raise ValueError("worker probes must exactly match attempted probe workers")
    for probed, event_activated, event_failed in scale_probe_outcomes:
        if probed - event_activated and not event_failed:
            raise ValueError(
                "non-activated probed candidates require a matching failed scale event"
            )
    if set(current_activated_probes) != set(validated):
        raise ValueError(
            "current activated probe identities must match validated workers"
        )
    activated_identity_keys = {
        (
            probe.worker_address,
            probe.hostname,
            probe.pid,
            probe.slurm_job_id,
        )
        for probe in historical_activated_probes
    }
    if not activated_identity_keys <= set(probe_identities):
        raise ValueError("activated probes must be present in raw probe history")
    validated_probes = tuple(current_activated_probes[address] for address in validated)
    try:
        validate_slurm_worker_identities(
            validated_probes,
            processes_per_job=processes_per_job,
            expected_process_memory_bytes=expected_process_memory_bytes,
        )
    except WorkerPreflightError as error:
        raise ValueError(
            "validated workers do not form complete SLURM job groups"
        ) from error
    if any(not probe.ok for probe in probes) and not scale_failed:
        raise ValueError("failed worker probe requires a failed scale event")
    if any(
        not current_activated_probes[address].ok
        for address in validated
        if address in current_activated_probes
    ):
        raise ValueError("activated workers must have successful probe evidence")
    statuses: dict[str, str] = {}
    placement_identities: list[tuple[str, int]] = []
    for index, raw in enumerate(_array(graph["placements"], "fine_graph.placements")):
        location = f"fine_graph.placements[{index}]"
        placement = _object(
            raw,
            location,
            frozenset(
                {
                    "task_key",
                    "attempt",
                    "status",
                    "dask_task_key",
                    "coordinator_submitted_at",
                    "coordinator_ready_at",
                    "started_at",
                    "ended_at",
                    "duration_seconds",
                    "worker_address",
                    "hostname",
                    "pid",
                    "slurm_job_id",
                    "native_threads",
                    "artifact_paths",
                    "reused_artifacts",
                    "log_path",
                    "placement_accepted",
                    "rejection_reason",
                    "failure",
                }
            ),
        )
        key = _string(placement["task_key"], f"{location}.task_key")
        if key not in attempts:
            raise ValueError("fine placement must name an attempted task")
        attempt = _integer(placement["attempt"], f"{location}.attempt", 1)
        placement_identities.append((key, attempt))
        status = _string(placement["status"], f"{location}.status")
        if status not in {"completed", "reused", "failed", "cancelled"}:
            raise ValueError("fine placement status is unsupported")
        statuses[key] = status
        expected_dask = f"cyexm-{key.replace(':', '-')}"
        if attempt > 1:
            expected_dask += f"-attempt-{attempt:03d}"
        observed_dask = _string(placement["dask_task_key"], f"{location}.dask_task_key")
        if observed_dask != expected_dask:
            raise ValueError("fine placement Dask key must match terminal attempt")
        coordinator_submitted_at = _timestamp(
            placement["coordinator_submitted_at"],
            f"{location}.coordinator_submitted_at",
        )
        coordinator_ready_at = _timestamp(
            placement["coordinator_ready_at"],
            f"{location}.coordinator_ready_at",
        )
        if coordinator_ready_at < coordinator_submitted_at:
            raise ValueError(
                "fine coordinator observed an attempt before its submission"
            )
        started_at = _timestamp(placement["started_at"], f"{location}.started_at")
        ended_at = _timestamp(placement["ended_at"], f"{location}.ended_at")
        if ended_at < started_at:
            raise ValueError("fine placement attempt ended before it started")
        _number(placement["duration_seconds"], f"{location}.duration_seconds")
        worker = _optional_string(
            placement["worker_address"],
            f"{location}.worker_address",
        )
        hostname = _optional_string(placement["hostname"], f"{location}.hostname")
        pid = _optional_integer(placement["pid"], f"{location}.pid")
        slurm_job_id = _optional_string(
            placement["slurm_job_id"],
            f"{location}.slurm_job_id",
        )
        native_threads = _array(
            placement["native_threads"],
            f"{location}.native_threads",
        )
        for thread_index, thread in enumerate(native_threads):
            pair = _array(
                thread,
                f"{location}.native_threads[{thread_index}]",
            )
            if len(pair) != 2 or type(pair[0]) is not str or type(pair[1]) is not int:
                raise TypeError("fine placement native_threads is invalid")
        for result_field in ("artifact_paths", "reused_artifacts"):
            values = tuple(
                _string(item, f"{location}.{result_field}[{item_index}]")
                for item_index, item in enumerate(
                    _array(placement[result_field], f"{location}.{result_field}")
                )
            )
            if len(values) != len(set(values)):
                raise ValueError(f"{location}.{result_field} must be unique")
        _string(placement["log_path"], f"{location}.log_path")
        accepted = placement["placement_accepted"]
        if type(accepted) is not bool:
            raise TypeError(f"{location}.placement_accepted must be an exact boolean")
        placement_identity = (worker, hostname, pid, slurm_job_id)
        interval = active_intervals.get(
            cast(tuple[str, str, int, str | None], placement_identity)
        )
        exact_identity = (
            worker is not None
            and placement_identity in activated_identity_keys
            and interval is not None
            and interval[0] <= coordinator_submitted_at <= coordinator_ready_at
            and (interval[1] is None or coordinator_ready_at <= interval[1])
        )
        if accepted and not exact_identity:
            raise ValueError("fine placement does not match worker probe identity")
        reason = _optional_string(
            placement["rejection_reason"],
            f"{location}.rejection_reason",
        )
        if not accepted and (status != "failed" or exact_identity or reason is None):
            raise ValueError(
                "rejected fine placement must be failed nonmatching evidence with "
                "a reason"
            )
        if reason is not None and status != "failed":
            raise ValueError("fine attempt rejection reason requires failed status")
        raw_failure = placement["failure"]
        failure: dict[str, object] | None = None
        if raw_failure is not None:
            failure = _object(
                raw_failure,
                f"{location}.failure",
                frozenset({"stage", "exception_type", "message"}),
            )
            _string(failure["stage"], f"{location}.failure.stage")
            exception_type = _string(
                failure["exception_type"],
                f"{location}.failure.exception_type",
            )
            if type(failure["message"]) is not str:
                raise TypeError(f"{location}.failure.message must be a string")
            if reason is not None and reason != (
                f"{exception_type}: {failure['message']}"
            ):
                raise ValueError(
                    "fine attempt rejection reason must match failure evidence"
                )
        if (status == "failed") != (failure is not None):
            raise ValueError("fine failed status and failure evidence must agree")
    expected_placements = tuple(
        (key, attempt)
        for key in task_keys
        if key in attempts
        for attempt in range(1, attempts[key] + 1)
    )
    if tuple(placement_identities) != expected_placements:
        raise ValueError(
            "fine placements must exactly and canonically match every attempt"
        )
    task_results = _decode_task_results(graph["task_results"], fine_tasks)
    final_placements = {
        cast(str, placement["task_key"]): placement
        for placement in (
            cast(dict[str, object], raw)
            for raw in _array(graph["placements"], "fine_graph.placements")
        )
    }
    if tuple(cast(str, result["key"]) for result in task_results) != tuple(
        final_placements
    ):
        raise ValueError(
            "fine task results must exactly and canonically cover final placements"
        )
    for result in task_results:
        placement = final_placements[cast(str, result["key"])]
        result_fields = (
            "status",
            "dask_task_key",
            "started_at",
            "ended_at",
            "duration_seconds",
            "native_threads",
            "artifact_paths",
            "reused_artifacts",
            "log_path",
            "failure",
        )
        if placement["placement_accepted"] is True:
            result_fields = (
                *result_fields,
                "hostname",
                "pid",
                "slurm_job_id",
            )
        for field_name in result_fields:
            if result[field_name] != placement[field_name]:
                raise ValueError(
                    "fine task result contradicts terminal placement field "
                    f"{field_name}"
                )

    blocked = _fine_keys(graph["blocked_descendants"], "fine_graph.blocked")
    if not set(blocked) <= set(task_keys) or set(blocked) & set(attempts):
        raise ValueError("blocked descendants must be unattempted graph tasks")
    failed = {
        key for key, status in statuses.items() if status in {"failed", "cancelled"}
    }
    for blocked_key in blocked:
        frontier = list(dependencies[blocked_key])
        ancestors: set[str] = set()
        while frontier:
            ancestor = frontier.pop()
            if ancestor in ancestors:
                continue
            ancestors.add(ancestor)
            frontier.extend(dependencies[ancestor])
        if not ancestors & failed:
            raise ValueError("blocked descendant must trace to failed/cancelled task")
    if type(graph["cancelled"]) is not bool:
        raise TypeError("fine_graph.cancelled must be an exact boolean")
    if graph["cancelled"] != cancellation_present:
        raise ValueError("fine cancellation contradicts observed cancellation")
    interruption = graph["interruption"]
    if interruption is None:
        if cancellation_present:
            raise ValueError("fine cancellation requires an exact interruption cause")
    else:
        cause = _object(
            interruption,
            "fine_graph.interruption",
            frozenset({"exception_type", "message"}),
        )
        _string(cause["exception_type"], "fine_graph.interruption.exception_type")
        if type(cause["message"]) is not str:
            raise TypeError("fine_graph.interruption.message must be a string")
        if not cancellation_present:
            raise ValueError("fine interruption cause requires cancellation")


__all__ = [
    "FineScaleEvent",
    "FineSlurmEvidence",
    "PreControllerTerminal",
    "validate_fine_slurm_evidence",
]
