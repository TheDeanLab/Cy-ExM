"""Typed fine-grained SLURM lifecycle and strict evidence policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import cast

from cyexm.execution.dispatcher import (
    FineGraphController,
    canonical_fine_task_key,
    validate_fine_graph_description,
)
from cyexm.execution.preflight import (
    WorkerProbe,
    validate_slurm_worker_identities,
)
from cyexm.workflow.fine_models import FineTaskResult
from cyexm.workflow.models import RoundTask

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
        "blocked_descendants",
        "cancelled",
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


@dataclass(slots=True)
class FineScaleEvent:
    """One deterministic upward-scale transition."""

    requested: int
    started_at: str
    connected: list[str] = field(default_factory=list)
    probed: list[str] = field(default_factory=list)
    activated: list[str] = field(default_factory=list)
    failed: list[dict[str, str]] = field(default_factory=list)
    ended_at: str | None = None

    def fail(self, error: BaseException, ended_at: str) -> None:
        self.failed = [{"type": type(error).__name__, "message": str(error)}]
        self.ended_at = ended_at

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "requested": self.requested,
            "connected": sorted(self.connected),
            "probed": sorted(self.probed),
            "activated": sorted(self.activated),
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


@dataclass(slots=True)
class FineSlurmEvidence:
    """Mutable coordinator state with detached JSON snapshots."""

    worker_cap: int
    candidates: set[str] = field(default_factory=set)
    validated: set[str] = field(default_factory=set)
    events: list[FineScaleEvent] = field(default_factory=list)
    controller: FineGraphController | None = None
    attempts: tuple[tuple[str, int], ...] = ()
    results: tuple[FineTaskResult, ...] = ()
    probes: dict[str, WorkerProbe] = field(default_factory=dict)
    cancelled: bool = False
    pre_controller_terminal: PreControllerTerminal | None = None

    def begin_scale(self, requested: int, started_at: str) -> FineScaleEvent:
        event = FineScaleEvent(requested=requested, started_at=started_at)
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

    def to_dict(self) -> dict[str, object]:
        graph = () if self.controller is None else self.controller.graph_description
        blocked = (
            ()
            if self.controller is None
            else tuple(key.encoded for key in self.controller.blocked_keys)
        )
        worker_by_job: dict[str, str] = {}
        for address, probe in self.probes.items():
            if probe.slurm_job_id is not None:
                worker_by_job[probe.slurm_job_id] = address

        def worker_address(result: FineTaskResult) -> str:
            if result.slurm_job_id is None:
                return ""
            return worker_by_job.get(result.slurm_job_id, "")

        def placement_accepted(result: FineTaskResult) -> bool:
            address = worker_address(result)
            probe = self.probes.get(address)
            return (
                address in self.validated
                and probe is not None
                and result.hostname == probe.hostname
                and result.pid == probe.pid
                and result.slurm_job_id == probe.slurm_job_id
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
                {"task_key": key, "attempts": count} for key, count in self.attempts
            ],
            "placements": [
                {
                    "task_key": result.key.encoded,
                    "status": result.status,
                    "dask_task_key": result.dask_task_key,
                    "worker_address": worker_address(result),
                    "hostname": result.hostname,
                    "pid": result.pid,
                    "slurm_job_id": result.slurm_job_id,
                    "placement_accepted": placement_accepted(result),
                }
                for result in self.results
            ],
            "blocked_descendants": sorted(blocked, key=canonical_fine_task_key),
            "cancelled": self.cancelled,
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


def validate_fine_slurm_evidence(
    value: object,
    *,
    configured_workers: int,
    task_retries: int,
    cancellation_present: bool,
    startup_outcome: str,
    reference_attempted: bool,
    round_tasks: tuple[RoundTask, ...],
    observed_probes: Sequence[WorkerProbe],
) -> None:
    """Strictly validate one version-2 fine SLURM evidence object."""
    graph = _object(value, "execution_evidence.fine_graph", _GRAPH_KEYS)
    if _string(graph["mode"], "fine_graph.mode") != "fine_grained":
        raise ValueError("fine_graph.mode must be 'fine_grained'")
    if _integer(graph["worker_cap"], "fine_graph.worker_cap", 1) != configured_workers:
        raise ValueError("fine_graph.worker_cap must match configured_workers")
    candidates = _strings(graph["candidate_workers"], "fine_graph.candidates")
    validated = _strings(graph["validated_workers"], "fine_graph.validated")
    if not set(validated) <= set(candidates):
        raise ValueError("validated workers must be candidate workers")

    activated: list[str] = []
    probed_workers: set[str] = set()
    connected: set[str] = set()
    previous_request = 0
    scale_failed = False
    for index, raw in enumerate(_array(graph["scale_events"], "fine_graph.events")):
        location = f"fine_graph.scale_events[{index}]"
        event = _object(
            raw,
            location,
            frozenset(
                {
                    "requested",
                    "connected",
                    "probed",
                    "activated",
                    "failed",
                    "started_at",
                    "ended_at",
                }
            ),
        )
        requested = _integer(event["requested"], f"{location}.requested", 1)
        if requested <= previous_request or requested > configured_workers:
            raise ValueError("fine scale requests must increase monotonically to cap")
        previous_request = requested
        event_connected = _strings(event["connected"], f"{location}.connected")
        probed = _strings(event["probed"], f"{location}.probed")
        event_activated = _strings(event["activated"], f"{location}.activated")
        if not set(event_activated) <= set(probed) <= set(event_connected):
            raise ValueError("fine scale activation must be a subset of probed workers")
        connected.update(event_connected)
        probed_workers.update(probed)
        activated.extend(event_activated)
        for failure_index, failure in enumerate(
            _array(event["failed"], f"{location}.failed")
        ):
            scale_failed = True
            error = _object(
                failure,
                f"{location}.failed[{failure_index}]",
                frozenset({"type", "message"}),
            )
            _string(error["type"], f"{location}.failed[{failure_index}].type")
            if type(error["message"]) is not str:
                raise TypeError("fine scale failure message must be a string")
        if _timestamp(event["ended_at"], f"{location}.ended_at") < _timestamp(
            event["started_at"], f"{location}.started_at"
        ):
            raise ValueError("fine scale event ended before it started")
    if tuple(sorted(connected)) != candidates:
        raise ValueError("scale-event candidates must match candidate_workers")
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
        _string(
            terminal["exception_type"],
            "pre_controller_terminal.exception_type",
        )
        if type(terminal["message"]) is not str:
            raise TypeError("pre_controller_terminal.message must be a string")
        if phase not in {"startup", "reference"}:
            raise ValueError("pre_controller_terminal phase is unsupported")
        if status not in {"failed", "cancelled"}:
            raise ValueError("pre_controller_terminal status is unsupported")
        if phase == "startup" and startup_outcome not in {"failed", "interrupted"}:
            raise ValueError("startup terminal cause contradicts startup outcome")
        if phase == "reference" and (
            startup_outcome != "succeeded" or not reference_attempted
        ):
            raise ValueError(
                "reference terminal cause lacks successful startup/attempt"
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
    validate_slurm_worker_identities(probes)
    probe_by_address = {probe.worker_address: probe for probe in probes}
    if tuple(sorted(probe_by_address)) != tuple(sorted(probed_workers)):
        raise ValueError("worker probes must exactly match successfully probed workers")
    statuses: dict[str, str] = {}
    placement_keys: list[str] = []
    for index, raw in enumerate(_array(graph["placements"], "fine_graph.placements")):
        location = f"fine_graph.placements[{index}]"
        placement = _object(
            raw,
            location,
            frozenset(
                {
                    "task_key",
                    "status",
                    "dask_task_key",
                    "worker_address",
                    "hostname",
                    "pid",
                    "slurm_job_id",
                    "placement_accepted",
                }
            ),
        )
        key = _string(placement["task_key"], f"{location}.task_key")
        if key in placement_keys or key not in attempts:
            raise ValueError("fine placement must uniquely name an attempted task")
        placement_keys.append(key)
        status = _string(placement["status"], f"{location}.status")
        if status not in {"completed", "reused", "failed", "cancelled"}:
            raise ValueError("fine placement status is unsupported")
        statuses[key] = status
        attempt = attempts[key]
        expected_dask = f"cyexm-{key.replace(':', '-')}"
        if attempt > 1:
            expected_dask += f"-attempt-{attempt:03d}"
        observed_dask = _string(placement["dask_task_key"], f"{location}.dask_task_key")
        if observed_dask != expected_dask:
            raise ValueError("fine placement Dask key must match terminal attempt")
        worker = placement["worker_address"]
        if type(worker) is not str:
            raise TypeError(f"{location}.worker_address must be an exact string")
        _integer(placement["pid"], f"{location}.pid", 1)
        accepted = placement["placement_accepted"]
        if type(accepted) is not bool:
            raise TypeError(f"{location}.placement_accepted must be an exact boolean")
        probe = probe_by_address.get(worker)
        exact_identity = (
            worker in validated
            and probe is not None
            and placement["hostname"] == probe.hostname
            and placement["pid"] == probe.pid
            and placement["slurm_job_id"] == probe.slurm_job_id
        )
        if accepted and not exact_identity:
            raise ValueError("fine placement does not match worker probe identity")
        if not accepted and (status != "failed" or exact_identity):
            raise ValueError(
                "rejected fine placement must be a failed nonmatching identity"
            )
    if set(placement_keys) != set(attempts):
        raise ValueError("fine placements must match attempted terminal tasks")

    blocked = _strings(graph["blocked_descendants"], "fine_graph.blocked")
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


__all__ = [
    "FineScaleEvent",
    "FineSlurmEvidence",
    "PreControllerTerminal",
    "validate_fine_slurm_evidence",
]
