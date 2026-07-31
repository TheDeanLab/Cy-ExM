"""Focused one-versus-two-process SLURM acceptance policy and report builder."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import cast

from cyexm.config._slurm_resources import (
    SlurmJobResources,
    derive_slurm_job_resources,
)
from cyexm.config.models import ExecutionProfile
from cyexm.workflow.models import WorkflowResult
from hpc.environment import HpcAcceptanceContext


class HpcMultiProcessAcceptanceError(RuntimeError):
    """Focused multi-process HPC evidence is missing, stale, or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HpcMultiProcessAcceptanceError(message)


def _resolved(path: Path, role: str, *, directory: bool = False) -> Path:
    _require(not path.is_symlink(), f"{role} must not be a symlink: {path}")
    try:
        result = path.resolve(strict=True)
    except OSError as error:
        raise HpcMultiProcessAcceptanceError(f"{role} must exist: {path}") from error
    _require(
        result.is_dir() if directory else result.is_file(),
        f"{role} has the wrong filesystem type: {result}",
    )
    return result


def _disjoint(first: Path, second: Path) -> bool:
    return not first.is_relative_to(second) and not second.is_relative_to(first)


def multi_process_acceptance_profiles(
    single: ExecutionProfile,
    multi: ExecutionProfile,
) -> tuple[ExecutionProfile, ExecutionProfile]:
    """Validate the exact fair one-process/two-process acceptance pair."""
    for label, profile, process_count in (
        ("single", single, 1),
        ("multi", multi, 2),
    ):
        spec = profile.slurm
        _require(
            profile.executor == "slurm"
            and profile.parallelism.mode == "fine_grained"
            and spec is not None,
            f"{label} profile must use fine-grained SLURM execution",
        )
        assert spec is not None
        _require(
            spec.processes_per_worker == process_count,
            (
                "single profile must use one process"
                if process_count == 1
                else "multi profile must use exactly two processes"
            ),
        )
        _require(
            profile.native_threads.itk == 15 and profile.native_threads.blas == 15,
            f"{label} profile must use 15 ITK and 15 BLAS threads",
        )
        _require(
            spec.workers == 3
            and spec.cores_per_worker == 30
            and spec.memory == "500GB"
            and spec.queue == "512GB"
            and spec.job_extra_directives == ("--exclusive",)
            and spec.task_retries >= 1,
            f"{label} profile does not match the exact acceptance allocation",
        )
        derive_slurm_job_resources(
            spec,
            profile.native_threads,
            profile.parallelism,
        )
    assert single.slurm is not None and multi.slurm is not None
    _require(
        single.source_path.resolve() != multi.source_path.resolve(),
        "single and multi profile paths must be distinct",
    )
    _require(
        single.paths.dataset_root.resolve() == multi.paths.dataset_root.resolve(),
        "single and multi profiles must use the same dataset root",
    )
    single_scratch = single.paths.worker_local_directory
    multi_scratch = multi.paths.worker_local_directory
    _require(
        single_scratch is not None
        and multi_scratch is not None
        and _disjoint(
            single.paths.output_root.resolve(),
            multi.paths.output_root.resolve(),
        )
        and _disjoint(single_scratch.resolve(), multi_scratch.resolve())
        and single.slurm.job_name != multi.slurm.job_name,
        "single and multi output/scratch roots and job names must be distinct",
    )
    normalized = replace(
        multi,
        source_path=single.source_path,
        paths=single.paths,
        slurm=replace(
            multi.slurm,
            processes_per_worker=1,
            job_name=single.slurm.job_name,
        ),
    )
    _require(
        normalized == single,
        "single and multi profiles must be otherwise identical",
    )
    return single, multi


def _json_file(
    path: Path,
    role: str,
    *,
    owned_root: Path | None = None,
) -> dict[str, object]:
    resolved = _resolved(path, role)
    if owned_root is not None:
        _require(
            resolved.is_relative_to(owned_root),
            f"{role} must remain inside the approved evidence root",
        )
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HpcMultiProcessAcceptanceError(
            f"{role} must contain readable JSON"
        ) from error
    _require(type(value) is dict, f"{role} must contain one JSON object")
    return cast(dict[str, object], value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp(value: object, role: str) -> datetime:
    _require(type(value) is str, f"{role} must be an ISO-8601 string")
    assert isinstance(value, str)
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise HpcMultiProcessAcceptanceError(
            f"{role} must be an ISO-8601 timestamp"
        ) from error
    _require(result.tzinfo is not None, f"{role} must include a timezone")
    return result


def _fine_evidence(
    result: WorkflowResult,
    raw: Mapping[str, object],
    *,
    processes: int,
    role: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    try:
        embedded = json.loads(result.execution_evidence_json)
    except json.JSONDecodeError as error:
        raise HpcMultiProcessAcceptanceError(
            f"{role} result evidence is not JSON"
        ) from error
    _require(embedded == raw, f"{role} result evidence differs from raw evidence")
    _require(
        raw.get("schema_version") == 3,
        f"{role} evidence must use fine SLURM schema version 3",
    )
    requested = raw.get("requested")
    graph = raw.get("fine_graph")
    _require(
        type(requested) is dict and type(graph) is dict,
        f"{role} evidence lacks requested resources or a fine graph",
    )
    requested = cast(dict[str, object], requested)
    graph = cast(dict[str, object], graph)
    _require(
        requested.get("processes_per_job") == processes,
        f"{role} evidence has the wrong process count",
    )
    placements = graph.get("placements")
    _require(type(placements) is list, f"{role} placements must be an array")
    placement_values = cast(list[object], placements)
    _require(
        all(type(item) is dict for item in placement_values),
        f"{role} placements must be JSON objects",
    )
    return requested, cast(list[dict[str, object]], placement_values)


def _worker_identity(value: object, role: str) -> dict[str, object]:
    _require(type(value) is dict, f"{role} must be a worker identity object")
    assert isinstance(value, dict)
    identity = {
        name: value.get(name)
        for name in ("worker_address", "hostname", "pid", "slurm_job_id")
    }
    _require(
        all(
            type(identity[name]) is expected and bool(identity[name])
            for name, expected in (
                ("worker_address", str),
                ("hostname", str),
                ("pid", int),
                ("slurm_job_id", str),
            )
        ),
        f"{role} has a malformed worker identity",
    )
    return cast(dict[str, object], identity)


def _identity_key(value: Mapping[str, object]) -> tuple[str, str, int, str]:
    return (
        cast(str, value["worker_address"]),
        cast(str, value["hostname"]),
        cast(int, value["pid"]),
        cast(str, value["slurm_job_id"]),
    )


def _complete_groups(
    identities: Mapping[str, dict[str, object]],
    *,
    processes: int,
    process_memory_bytes: int,
    role: str,
) -> None:
    groups: dict[str, list[dict[str, object]]] = {}
    for identity in identities.values():
        groups.setdefault(cast(str, identity["slurm_job_id"]), []).append(identity)
    _require(bool(groups), f"{role} contains no activated SLURM job groups")
    for job_id, group in groups.items():
        _require(
            len(group) == processes
            and len({cast(str, item["hostname"]) for item in group}) == 1
            and len({cast(str, item["worker_address"]) for item in group}) == processes
            and len({cast(int, item["pid"]) for item in group}) == processes
            and all(
                item.get("ok") is True
                and item.get("memory_limit_bytes") == process_memory_bytes
                for item in group
            ),
            f"{role} SLURM job {job_id!r} is not one complete process group",
        )


def _realized_capacity(
    raw: Mapping[str, object],
    *,
    processes: int,
    configured_jobs: int,
    process_memory_bytes: int,
    role: str,
    require_full_capacity: bool,
) -> dict[str, object]:
    graph = raw.get("fine_graph")
    _require(type(graph) is dict, f"{role} fine graph is malformed")
    assert isinstance(graph, dict)
    events = graph.get("scale_events")
    _require(type(events) is list and bool(events), f"{role} lacks scale events")
    active: dict[str, dict[str, object]] = {}
    all_identities: dict[tuple[str, str, int, str], dict[str, object]] = {}
    peak_processes = 0
    peak_jobs = 0
    peak_requested_jobs = 0
    for index, raw_event in enumerate(cast(list[object], events)):
        _require(type(raw_event) is dict, f"{role} scale event {index} is malformed")
        event = cast(dict[str, object], raw_event)
        requested_jobs = event.get("requested_jobs")
        expected_processes = event.get("expected_processes")
        _require(
            type(requested_jobs) is int
            and 1 <= requested_jobs <= configured_jobs
            and expected_processes == requested_jobs * processes,
            f"{role} scale event {index} has inconsistent requested capacity",
        )
        requested_jobs_value = cast(int, requested_jobs)
        peak_requested_jobs = max(peak_requested_jobs, requested_jobs_value)
        departed = event.get("departed")
        activated = event.get("activated")
        activated_probes = event.get("activated_probes")
        _require(
            type(departed) is list
            and type(activated) is list
            and type(activated_probes) is list,
            f"{role} scale event {index} lacks transition evidence",
        )
        for address in cast(list[object], departed):
            _require(
                type(address) is str and address in active,
                f"{role} scale event {index} departs an unknown process",
            )
            active.pop(cast(str, address))
        probes = [
            {
                **_worker_identity(
                    item,
                    f"{role} scale event {index} activated probe",
                ),
                **(cast(dict[str, object], item) if type(item) is dict else {}),
            }
            for item in cast(list[object], activated_probes)
        ]
        _require(
            [item["worker_address"] for item in probes] == activated,
            f"{role} scale event {index} activation identities differ",
        )
        for probe in probes:
            key = _identity_key(probe)
            _require(
                key not in all_identities
                and cast(str, probe["worker_address"]) not in active,
                f"{role} reuses an activated process identity",
            )
            all_identities[key] = probe
            active[cast(str, probe["worker_address"])] = probe
        if not event.get("failed"):
            _complete_groups(
                active,
                processes=processes,
                process_memory_bytes=process_memory_bytes,
                role=f"{role} scale event {index}",
            )
        peak_processes = max(peak_processes, len(active))
        peak_jobs = max(
            peak_jobs,
            len({cast(str, item["slurm_job_id"]) for item in active.values()}),
        )
    if require_full_capacity:
        _require(
            peak_requested_jobs == configured_jobs
            and peak_jobs == configured_jobs
            and peak_processes == configured_jobs * processes,
            f"{role} did not realize the full approved job/process capacity",
        )
    placements = cast(list[dict[str, object]], graph.get("placements", []))
    placed_job_ids = {
        cast(str, item["slurm_job_id"])
        for item in placements
        if item.get("placement_accepted") is True
        and type(item.get("slurm_job_id")) is str
    }
    _require(
        placed_job_ids
        <= {cast(str, item["slurm_job_id"]) for item in all_identities.values()},
        f"{role} placements cite processes absent from activation evidence",
    )
    return {
        "peak_jobs": peak_jobs,
        "peak_processes": peak_processes,
        "peak_requested_jobs": peak_requested_jobs,
        "activated_job_ids": sorted(
            {cast(str, item["slurm_job_id"]) for item in all_identities.values()}
        ),
    }


def _overlap_evidence(
    placements: list[dict[str, object]],
) -> dict[str, object]:
    accepted = [
        item
        for item in placements
        if item.get("status") in {"completed", "reused"}
        and item.get("placement_accepted") is True
    ]
    for index, first in enumerate(accepted):
        for second in accepted[index + 1 :]:
            first_start = _timestamp(first.get("started_at"), "placement.started_at")
            first_end = _timestamp(first.get("ended_at"), "placement.ended_at")
            second_start = _timestamp(second.get("started_at"), "placement.started_at")
            second_end = _timestamp(second.get("ended_at"), "placement.ended_at")
            overlap = max(first_start, second_start) < min(first_end, second_end)
            same_job = first.get("slurm_job_id") == second.get("slurm_job_id")
            same_host = first.get("hostname") == second.get("hostname")
            distinct_process = first.get("worker_address") != second.get(
                "worker_address"
            ) and first.get("pid") != second.get("pid")
            valid_identities = all(
                type(item.get(name)) is expected
                for item in (first, second)
                for name, expected in (
                    ("worker_address", str),
                    ("hostname", str),
                    ("pid", int),
                    ("slurm_job_id", str),
                )
            )
            if (
                overlap
                and same_job
                and same_host
                and distinct_process
                and valid_identities
            ):
                return {
                    "same_job_distinct_process_overlap": True,
                    "first_task_key": first.get("task_key"),
                    "second_task_key": second.get("task_key"),
                    "slurm_job_id": first.get("slurm_job_id"),
                    "hostname": first.get("hostname"),
                    "worker_addresses": [
                        first.get("worker_address"),
                        second.get("worker_address"),
                    ],
                    "pids": [first.get("pid"), second.get("pid")],
                }
    raise HpcMultiProcessAcceptanceError(
        "multi-process evidence contains no validated overlapping tasks in "
        "distinct processes of one job"
    )


def _cleanup(
    evidence_dir: Path,
) -> tuple[dict[str, object], dict[str, set[str]]]:
    operations: list[dict[str, object]] = []
    owned_by_operation: dict[str, set[str]] = {}
    for name in ("single", "multi", "replacement"):
        operation = f"multi-process-{name}"
        outcome = _json_file(
            evidence_dir / f"{operation}-outcome.json",
            f"{operation} outcome",
            owned_root=evidence_dir,
        )
        queue = _json_file(
            evidence_dir / f"{operation}-queue-after.json",
            f"{operation} queue",
            owned_root=evidence_dir,
        )
        _require(
            outcome.get("operation") == operation
            and outcome.get("status") == "completed"
            and outcome.get("cleanup_outcome") == "empty"
            and type(outcome.get("owned_job_ids")) is list
            and bool(outcome.get("owned_job_ids")),
            f"{operation} outcome does not prove owned completion",
        )
        _require(
            queue.get("outcome") == "empty" and queue.get("jobs") == [],
            f"{operation} queue is not empty",
        )
        owned_ids = cast(list[object], outcome["owned_job_ids"])
        _require(
            all(type(job_id) is str and job_id for job_id in owned_ids)
            and len(owned_ids) == len(set(owned_ids)),
            f"{operation} outcome has malformed owned job identities",
        )
        owned_by_operation[name] = set(cast(list[str], owned_ids))
        operations.append(
            {
                "operation": operation,
                "owned_job_ids": sorted(owned_by_operation[name]),
                "queue_empty": True,
            }
        )
    return {"accepted": True, "operations": operations}, owned_by_operation


def _performance(
    raw: Mapping[str, object],
    placements: list[dict[str, object]],
) -> dict[str, object]:
    single = raw.get("single_duration_seconds")
    multi = raw.get("multi_duration_seconds")
    peaks = raw.get("peak_rss_bytes_by_process")
    _require(
        not isinstance(single, bool)
        and isinstance(single, int | float)
        and math.isfinite(float(single))
        and float(single) > 0
        and not isinstance(multi, bool)
        and isinstance(multi, int | float)
        and math.isfinite(float(multi))
        and float(multi) > 0,
        "performance evidence requires positive raw durations",
    )
    assert isinstance(single, int | float) and not isinstance(single, bool)
    assert isinstance(multi, int | float) and not isinstance(multi, bool)
    _require(
        type(peaks) is dict
        and bool(peaks)
        and all(
            type(address) is str and type(value) is int and value > 0
            for address, value in cast(dict[object, object], peaks).items()
        ),
        "performance evidence requires per-process peak RSS",
    )
    placed_addresses = {
        item.get("worker_address")
        for item in placements
        if item.get("placement_accepted") is True
    }
    _require(
        set(cast(dict[str, int], peaks)) <= placed_addresses,
        "peak RSS identities are absent from validated placement evidence",
    )
    single_seconds = float(single)
    multi_seconds = float(multi)
    return {
        "single_duration_seconds": single_seconds,
        "multi_duration_seconds": multi_seconds,
        "speedup": single_seconds / multi_seconds,
        "peak_rss_bytes_by_process": dict(cast(dict[str, int], peaks)),
        "speedup_is_characterization_only": True,
    }


def _replacement_evidence(
    raw: Mapping[str, object],
    marker: Mapping[str, object],
    *,
    owned_job_ids: set[str],
) -> dict[str, object]:
    expected_marker_keys = {
        "schema_version",
        "task_key",
        "worker_address",
        "hostname",
        "pid",
        "slurm_job_id",
        "status",
        "injected_at",
    }
    _require(
        set(marker) == expected_marker_keys
        and marker.get("schema_version") == 1
        and marker.get("status") == "worker-loss-injected"
        and type(marker.get("task_key")) is str
        and bool(marker.get("task_key")),
        "worker-loss marker has an invalid exact schema",
    )
    loss = _worker_identity(marker, "worker-loss marker")
    injected_at = _timestamp(marker.get("injected_at"), "worker-loss injected_at")
    graph = raw.get("fine_graph")
    _require(type(graph) is dict, "replacement evidence lacks a fine graph")
    assert isinstance(graph, dict)
    events = graph.get("scale_events")
    placements = graph.get("placements")
    _require(
        type(events) is list and type(placements) is list,
        "replacement evidence lacks raw events or placements",
    )
    activation_index: int | None = None
    original_group: list[dict[str, object]] = []
    for index, raw_event in enumerate(cast(list[object], events)):
        if type(raw_event) is not dict:
            continue
        probes = cast(dict[str, object], raw_event).get("activated_probes")
        if type(probes) is not list:
            continue
        identities = [
            {
                **_worker_identity(
                    item,
                    f"replacement scale event {index} activated probe",
                ),
                **cast(dict[str, object], item),
            }
            for item in cast(list[object], probes)
        ]
        if any(_identity_key(item) == _identity_key(loss) for item in identities):
            activation_index = index
            original_group = [
                item
                for item in identities
                if item["slurm_job_id"] == loss["slurm_job_id"]
                and item["hostname"] == loss["hostname"]
            ]
            break
    _require(
        activation_index is not None and len(original_group) == 2,
        "worker-loss identity lacks an original complete two-process group",
    )
    sibling = next(
        (item for item in original_group if _identity_key(item) != _identity_key(loss)),
        None,
    )
    _require(sibling is not None, "worker-loss group lacks a healthy sibling")
    departure_event: dict[str, object] | None = None
    replacement_identities: list[dict[str, object]] = []
    assert activation_index is not None
    for index, raw_event in enumerate(
        cast(list[object], events)[activation_index + 1 :],
        start=activation_index + 1,
    ):
        if type(raw_event) is not dict:
            continue
        event = cast(dict[str, object], raw_event)
        if loss["worker_address"] not in cast(list[object], event.get("departed", [])):
            continue
        raw_probes = event.get("activated_probes")
        _require(
            type(raw_probes) is list and bool(raw_probes),
            "worker departure lacks replacement activation evidence",
        )
        replacement_identities = [
            {
                **_worker_identity(
                    item,
                    f"replacement scale event {index} activated probe",
                ),
                **cast(dict[str, object], item),
            }
            for item in cast(list[object], raw_probes)
        ]
        departure_event = event
        break
    _require(
        departure_event is not None,
        "worker-loss identity is not correlated to a later departure event",
    )
    departure_ended = _timestamp(
        cast(dict[str, object], departure_event).get("ended_at"),
        "replacement departure ended_at",
    )
    _require(
        injected_at <= departure_ended,
        "worker-loss marker occurs after the correlated departure event",
    )
    target_key = cast(str, marker["task_key"])
    target = [
        cast(dict[str, object], item)
        for item in cast(list[object], placements)
        if type(item) is dict and item.get("task_key") == target_key
    ]
    _require(
        len(target) in {1, 2}
        and [item.get("attempt") for item in target] == list(range(1, len(target) + 1)),
        "worker-loss target lacks canonical attempt lineage",
    )
    recovered = target[-1]
    scheduler_rescheduled = len(target) == 1
    if scheduler_rescheduled:
        submitted = _timestamp(
            recovered.get("coordinator_submitted_at"),
            "replacement target submitted_at",
        )
        ready = _timestamp(
            recovered.get("coordinator_ready_at"),
            "replacement target ready_at",
        )
        _require(
            submitted <= injected_at <= ready,
            "scheduler-rescheduled target interval does not span worker loss",
        )
    else:
        failed = target[0]
        _require(
            failed.get("status") == "failed"
            and failed.get("placement_accepted") is False,
            "worker-loss target first attempt is not a recorded failure",
        )
        failed_started = _timestamp(
            failed.get("started_at"),
            "replacement failed attempt started_at",
        )
        failed_ended = _timestamp(
            failed.get("ended_at"),
            "replacement failed attempt ended_at",
        )
        _require(
            failed_started <= injected_at <= failed_ended,
            "worker-loss marker does not fall within the failed target attempt",
        )
    recovered_identity = _worker_identity(
        recovered,
        "replacement recovered target placement",
    )
    _require(
        recovered.get("status") in {"completed", "reused"}
        and recovered.get("placement_accepted") is True
        and type(recovered.get("artifact_paths")) is list
        and bool(recovered.get("artifact_paths")),
        "worker-loss target did not recover with validated artifacts",
    )
    replacement_identity_keys = {_identity_key(item) for item in replacement_identities}
    recovered_ready = _timestamp(
        recovered.get("coordinator_ready_at"),
        "replacement recovered target ready_at",
    )
    replacement_liveness = (
        recovered
        if _identity_key(recovered_identity) in replacement_identity_keys
        else next(
            (
                cast(dict[str, object], item)
                for item in cast(list[object], placements)
                if type(item) is dict
                and item.get("task_key") != target_key
                and item.get("placement_accepted") is True
                and item.get("status") in {"completed", "reused"}
                and type(item.get("artifact_paths")) is list
                and bool(item.get("artifact_paths"))
                and _identity_key(
                    _worker_identity(item, "replacement liveness placement")
                )
                in replacement_identity_keys
                and _timestamp(
                    item.get("coordinator_ready_at"),
                    "replacement liveness coordinator_ready_at",
                )
                > recovered_ready
            ),
            None,
        )
    )
    _require(
        replacement_liveness is not None,
        "new replacement process did not commit the target or a later task",
    )
    replacement_identity = _worker_identity(
        replacement_liveness,
        "replacement liveness",
    )
    assert sibling is not None
    sibling_liveness = next(
        (
            cast(dict[str, object], item)
            for item in cast(list[object], placements)
            if type(item) is dict
            and item.get("placement_accepted") is True
            and item.get("status") in {"completed", "reused"}
            and _identity_key(_worker_identity(item, "healthy-sibling placement"))
            == _identity_key(sibling)
            and _timestamp(
                item.get("coordinator_ready_at"),
                "healthy-sibling coordinator_ready_at",
            )
            > departure_ended
        ),
        None,
    )
    _require(
        sibling_liveness is not None,
        "replacement evidence does not prove healthy sibling liveness after loss",
    )
    lineage_job_ids = {
        cast(str, loss["slurm_job_id"]),
        cast(str, sibling["slurm_job_id"]),
        cast(str, recovered_identity["slurm_job_id"]),
        cast(str, replacement_identity["slurm_job_id"]),
    }
    _require(
        lineage_job_ids <= owned_job_ids,
        "replacement lineage cites a job absent from owned outcome evidence",
    )
    return {
        "accepted": True,
        "target_task_key": target_key,
        "scheduler_rescheduled": scheduler_rescheduled,
        "departed_process": loss,
        "recovered_target_process": recovered_identity,
        "replacement_process": replacement_identity,
        "healthy_sibling": _worker_identity(
            sibling_liveness,
            "healthy-sibling liveness",
        ),
    }


def build_multi_process_acceptance_report(
    *,
    context: HpcAcceptanceContext,
    single_profile: ExecutionProfile,
    multi_profile: ExecutionProfile,
    single_result: WorkflowResult,
    multi_result: WorkflowResult,
    replacement_result: WorkflowResult,
    evidence_dir: Path,
) -> dict[str, object]:
    """Build a checksum-bound report from raw owned-job acceptance evidence."""
    multi_process_acceptance_profiles(single_profile, multi_profile)
    root = _resolved(evidence_dir, "multi-process evidence root", directory=True)
    _require(
        context.evidence_dir.resolve(strict=True) == root,
        "report evidence root differs from the approved context",
    )
    _require(
        context.execution_path.resolve(strict=True)
        == single_profile.source_path.resolve(strict=True),
        "single profile differs from the approved context profile",
    )
    _require(
        all(
            result.status == "completed"
            for result in (single_result, multi_result, replacement_result)
        ),
        "all focused workflows must complete",
    )
    _require(
        len(
            {
                result.dataset_id
                for result in (single_result, multi_result, replacement_result)
            }
        )
        == 1,
        "focused workflow dataset identities differ",
    )
    fingerprints = {
        result.scientific_fingerprint
        for result in (single_result, multi_result, replacement_result)
    }
    _require(len(fingerprints) == 1, "focused scientific fingerprints differ")

    evidence_paths = {
        "single": root / "multi-process-single-slurm-execution.json",
        "multi": root / "multi-process-multi-slurm-execution.json",
        "replacement": root / "multi-process-replacement-slurm-execution.json",
    }
    raw_single = _json_file(
        evidence_paths["single"],
        "single execution evidence",
        owned_root=root,
    )
    raw_multi = _json_file(
        evidence_paths["multi"],
        "multi execution evidence",
        owned_root=root,
    )
    raw_replacement = _json_file(
        evidence_paths["replacement"],
        "replacement execution evidence",
        owned_root=root,
    )
    single_requested, _ = _fine_evidence(
        single_result,
        raw_single,
        processes=1,
        role="single",
    )
    multi_requested, placements = _fine_evidence(
        multi_result,
        raw_multi,
        processes=2,
        role="multi",
    )
    _fine_evidence(
        replacement_result,
        raw_replacement,
        processes=2,
        role="replacement",
    )

    single_spec = single_profile.slurm
    multi_spec = multi_profile.slurm
    assert single_spec is not None and multi_spec is not None
    single_resources = derive_slurm_job_resources(
        single_spec,
        single_profile.native_threads,
        single_profile.parallelism,
    )
    resources = derive_slurm_job_resources(
        multi_spec,
        multi_profile.native_threads,
        multi_profile.parallelism,
    )

    def expected_request(
        profile: ExecutionProfile,
        process_resources: SlurmJobResources,
    ) -> dict[str, object]:
        spec = profile.slurm
        assert spec is not None
        memory = process_resources.job_memory
        return {
            "jobs": process_resources.job_cap,
            "configured_jobs": process_resources.job_cap,
            "processes_per_job": process_resources.processes_per_job,
            "task_slots_per_job": process_resources.task_slots_per_job,
            "max_task_slots": process_resources.max_task_slots,
            "cores_per_process": process_resources.cores_per_process,
            "process_memory_bytes": process_resources.process_memory_bytes,
            "startup_timeout": spec.startup_timeout,
            "task_retries": spec.task_retries,
            "memory": {
                "profile": memory.profile,
                "semantic_bytes": memory.semantic_bytes,
                "scheduler": memory.scheduler,
                "scheduler_bytes": memory.scheduler_bytes,
                "dask": memory.dask,
            },
        }

    _require(
        single_requested == expected_request(single_profile, single_resources)
        and multi_requested == expected_request(multi_profile, resources),
        "raw requested resources differ from the complete approved profiles",
    )
    single_capacity = _realized_capacity(
        raw_single,
        processes=1,
        configured_jobs=single_resources.job_cap,
        process_memory_bytes=single_resources.process_memory_bytes,
        role="single execution",
        require_full_capacity=True,
    )
    multi_capacity = _realized_capacity(
        raw_multi,
        processes=2,
        configured_jobs=resources.job_cap,
        process_memory_bytes=resources.process_memory_bytes,
        role="multi execution",
        require_full_capacity=True,
    )
    replacement_capacity = _realized_capacity(
        raw_replacement,
        processes=2,
        configured_jobs=resources.job_cap,
        process_memory_bytes=resources.process_memory_bytes,
        role="replacement execution",
        require_full_capacity=False,
    )
    expected_resources = {
        "configured_jobs": resources.job_cap,
        "processes_per_job": resources.processes_per_job,
        "task_slots_per_job": resources.task_slots_per_job,
        "max_task_slots": resources.max_task_slots,
        "cores_per_job": resources.cores_per_job,
        "cores_per_process": resources.cores_per_process,
        "job_memory_bytes": resources.job_memory.scheduler_bytes,
        "process_memory_bytes": resources.process_memory_bytes,
        "observed_peak_jobs": multi_capacity["peak_jobs"],
        "observed_peak_processes": multi_capacity["peak_processes"],
    }

    comparison_path = root / "multi-process-scientific-comparison.json"
    comparison = _json_file(
        comparison_path,
        "scientific comparison",
        owned_root=root,
    )
    fingerprint = next(iter(fingerprints))
    _require(
        comparison.get("accepted") is True and bool(set(comparison) - {"accepted"}),
        "scientific comparison is missing or rejected",
    )
    from hpc.workflow_acceptance import (
        HpcWorkflowAcceptanceError,
        _validated_scientific_comparison_evidence,
    )

    try:
        validated_comparison = _validated_scientific_comparison_evidence(
            {key: value for key, value in comparison.items() if key != "accepted"},
            require_close=False,
            metric_absolute_tolerance=0.01,
        )
    except HpcWorkflowAcceptanceError as error:
        raise HpcMultiProcessAcceptanceError(
            f"scientific comparison failed strict decoding: {error}"
        ) from error
    _require(
        validated_comparison["scientific_fingerprint"] == fingerprint,
        "scientific comparison fingerprint differs from workflow results",
    )
    selected_rounds = tuple(
        sorted(
            (
                item
                for item in context.manifest.rounds
                if item.strategy == "serial_chunked"
            ),
            key=lambda item: item.round,
        )[:2]
    )
    expected_scientific_structure = [
        (
            round_spec.round,
            tuple(
                (
                    str(
                        single_profile.paths.dataset_root.joinpath(
                            *channel.file.parts
                        ).resolve(strict=False)
                    ),
                    channel.interpolation,
                )
                for channel in round_spec.channels
            ),
        )
        for round_spec in selected_rounds
    ]
    observed_scientific_structure = [
        (
            cast(int, round_evidence["round"]),
            tuple(
                (
                    cast(str, channel_evidence["source"]),
                    cast(str, channel_evidence["interpolation"]),
                )
                for channel_evidence in cast(
                    list[dict[str, object]],
                    round_evidence["channels"],
                )
            ),
        )
        for round_evidence in cast(
            list[dict[str, object]],
            validated_comparison["rounds"],
        )
    ]
    _require(
        len(selected_rounds) == 2
        and observed_scientific_structure == expected_scientific_structure,
        "scientific comparison rounds/channel contracts differ from the focused "
        "manifest",
    )
    marker_path = root / "multi-process-worker-loss-marker.json"
    marker = _json_file(
        marker_path,
        "worker-loss marker",
        owned_root=root,
    )
    cleanup, owned_by_operation = _cleanup(root)
    for name, capacity in (
        ("single", single_capacity),
        ("multi", multi_capacity),
        ("replacement", replacement_capacity),
    ):
        activated_job_ids = set(cast(list[str], capacity["activated_job_ids"]))
        _require(
            activated_job_ids <= owned_by_operation[name],
            f"{name} activation evidence cites a job absent from its owned outcome",
        )
    replacement = _replacement_evidence(
        raw_replacement,
        marker,
        owned_job_ids=owned_by_operation["replacement"],
    )
    multi_placement_job_ids = {
        cast(str, item["slurm_job_id"])
        for item in placements
        if item.get("placement_accepted") is True
        and type(item.get("slurm_job_id")) is str
    }
    _require(
        multi_placement_job_ids <= owned_by_operation["multi"],
        "multi-process placements cite a job absent from owned outcome evidence",
    )

    performance_path = root / "multi-process-performance.json"
    performance = _performance(
        _json_file(
            performance_path,
            "performance characterization",
            owned_root=root,
        ),
        placements,
    )
    placement = _overlap_evidence(placements)

    bound_paths = (
        single_profile.source_path,
        multi_profile.source_path,
        *evidence_paths.values(),
        comparison_path,
        performance_path,
        marker_path,
        *(
            root / f"multi-process-{operation}-{suffix}.json"
            for operation in ("single", "multi", "replacement")
            for suffix in ("outcome", "queue-after")
        ),
    )
    evidence_files = [
        {
            "relative_path": (
                path.resolve(strict=True).relative_to(root).as_posix()
                if path.resolve(strict=True).is_relative_to(root)
                else str(path.resolve(strict=True))
            ),
            "sha256": _sha256(path.resolve(strict=True)),
        }
        for path in bound_paths
    ]
    _require(
        len({item["relative_path"] for item in evidence_files}) == len(evidence_files),
        "checksum-bound evidence paths must be unique",
    )
    return {
        "schema_version": 1,
        "status": "completed",
        "git_commit": context.git_commit,
        "package_versions": dict(context.package_versions),
        "scientific_parity": {
            "accepted": True,
            "scientific_fingerprint": fingerprint,
        },
        "resources": expected_resources,
        "placement": placement,
        "replacement": replacement,
        "cleanup": cleanup,
        "performance_characterization": performance,
        "evidence_files": sorted(
            evidence_files, key=lambda item: item["relative_path"]
        ),
    }


__all__ = (
    "HpcMultiProcessAcceptanceError",
    "build_multi_process_acceptance_report",
    "multi_process_acceptance_profiles",
)
