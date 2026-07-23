"""Typed fine-grained SLURM lifecycle and strict evidence policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import cast

from cyexm.execution.dispatcher import FineGraphController
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
    }
)
_STAGE_ORDER = {
    "preparation": 0,
    "whole-image": 0,
    "chunk": 1,
    "assembly": 2,
    "channel": 3,
    "validation": 4,
}


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


def _key_order(key: str) -> tuple[int, int, str]:
    try:
        round_label, stage, item = key.split(":", 2)
        return (
            int(round_label.removeprefix("round-")),
            _STAGE_ORDER[stage],
            item,
        )
    except (ValueError, KeyError) as error:
        raise ValueError(f"malformed fine task key: {key!r}") from error


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
                }
                for result in self.results
            ],
            "blocked_descendants": sorted(blocked, key=_key_order),
            "cancelled": self.cancelled,
        }


def _validate_task_graph(
    raw_graph: object,
    round_tasks: tuple[RoundTask, ...],
    *,
    allow_partial: bool,
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    task_keys: list[str] = []
    dependencies: dict[str, tuple[str, ...]] = {}
    for index, raw in enumerate(_array(raw_graph, "fine_graph.task_graph")):
        task = _object(
            raw,
            f"fine_graph.task_graph[{index}]",
            frozenset({"key", "dependencies"}),
        )
        key = _string(task["key"], f"fine_graph.task_graph[{index}].key")
        if key in dependencies:
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
        task_keys.append(key)
        dependencies[key] = values
    if task_keys != sorted(task_keys, key=_key_order):
        raise ValueError("fine graph tasks must use canonical ordering")
    if any(
        dependency not in dependencies
        for values in dependencies.values()
        for dependency in values
    ):
        raise ValueError("fine graph dependencies must name graph task keys")
    if tuple(sorted({_key_order(key)[0] for key in task_keys})) != tuple(
        sorted(task.round for task in round_tasks)
    ):
        raise ValueError("fine graph rounds must match submitted moving rounds")

    for round_task in round_tasks:
        prefix = f"round-{round_task.round:03d}:"
        stages = {
            stage: tuple(
                key
                for key in task_keys
                if key.startswith(prefix) and key.split(":", 2)[1] == stage
            )
            for stage in _STAGE_ORDER
        }
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
        else:
            preparation = f"{prefix}preparation:main"
            producer = f"{prefix}assembly:main"
            chunks = stages["chunk"]
            if stages["preparation"] != (preparation,) or stages["whole-image"]:
                raise ValueError("chunked fine graph has incompatible stages")
            if not chunks and allow_partial:
                if stages["assembly"] or stages["channel"] or stages["validation"]:
                    raise ValueError("partial chunk graph contains downstream stages")
                continue
            if not chunks or stages["assembly"] != (producer,):
                raise ValueError("chunked fine graph has incompatible stages")
            if dependencies[preparation] or any(
                dependencies[key] != (preparation,) for key in chunks
            ):
                raise ValueError("fine chunks must depend only on preparation")
            if dependencies[producer] != chunks:
                raise ValueError("fine assembly must depend on every chunk")
        if allow_partial and not stages["channel"] and not stages["validation"]:
            continue
        if stages["channel"] != channels or stages["validation"] != (validation,):
            raise ValueError("fine graph channels/validation must match the round")
        if dependencies[validation] != channels:
            raise ValueError("fine validation must depend on every channel task")
        if any(dependencies[key] != (producer,) for key in channels):
            raise ValueError("fine channels must depend only on registration producer")
    return tuple(task_keys), dependencies


def validate_fine_slurm_evidence(
    value: object,
    *,
    configured_workers: int,
    task_retries: int,
    cancellation_present: bool,
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

    task_keys, dependencies = _validate_task_graph(
        graph["task_graph"],
        round_tasks,
        allow_partial=cancellation_present or scale_failed,
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
    if tuple(sorted(probe_by_address)) != validated:
        raise ValueError("worker probes must exactly match validated_workers")
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
        worker = _string(placement["worker_address"], f"{location}.worker_address")
        if worker not in probe_by_address:
            raise ValueError("fine placement must use a validated worker")
        probe = probe_by_address[worker]
        if (
            placement["hostname"] != probe.hostname
            or placement["slurm_job_id"] != probe.slurm_job_id
        ):
            raise ValueError("fine placement does not match worker probe identity")
        _integer(placement["pid"], f"{location}.pid", 1)
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
    "validate_fine_slurm_evidence",
]
