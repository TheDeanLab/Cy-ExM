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

from cyexm.config._slurm_resources import derive_slurm_job_resources
from cyexm.config.models import ExecutionProfile
from cyexm.workflow.models import WorkflowResult
from hpc.environment import HpcAcceptanceContext


class HpcMultiProcessAcceptanceError(RuntimeError):
    """Focused multi-process HPC evidence is missing, stale, or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HpcMultiProcessAcceptanceError(message)


def _resolved(path: Path, role: str, *, directory: bool = False) -> Path:
    try:
        result = path.resolve(strict=True)
    except OSError as error:
        raise HpcMultiProcessAcceptanceError(
            f"{role} must exist: {path}"
        ) from error
    _require(
        result.is_dir() if directory else result.is_file(),
        f"{role} has the wrong filesystem type: {result}",
    )
    _require(not result.is_symlink(), f"{role} must not be a symlink: {result}")
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
            profile.native_threads.itk == 15
            and profile.native_threads.blas == 15,
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
        single.paths.dataset_root.resolve()
        == multi.paths.dataset_root.resolve(),
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


def _json_file(path: Path, role: str) -> dict[str, object]:
    resolved = _resolved(path, role)
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
            second_start = _timestamp(
                second.get("started_at"), "placement.started_at"
            )
            second_end = _timestamp(second.get("ended_at"), "placement.ended_at")
            overlap = max(first_start, second_start) < min(first_end, second_end)
            same_job = first.get("slurm_job_id") == second.get("slurm_job_id")
            same_host = first.get("hostname") == second.get("hostname")
            distinct_process = (
                first.get("worker_address") != second.get("worker_address")
                and first.get("pid") != second.get("pid")
            )
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


def _cleanup(evidence_dir: Path) -> dict[str, object]:
    operations: list[dict[str, object]] = []
    for name in ("single", "multi", "replacement"):
        operation = f"multi-process-{name}"
        outcome = _json_file(
            evidence_dir / f"{operation}-outcome.json",
            f"{operation} outcome",
        )
        queue = _json_file(
            evidence_dir / f"{operation}-queue-after.json",
            f"{operation} queue",
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
        operations.append(
            {
                "operation": operation,
                "owned_job_ids": outcome["owned_job_ids"],
                "queue_empty": True,
            }
        )
    return {"accepted": True, "operations": operations}


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
            type(address) is str
            and type(value) is int
            and value > 0
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
    raw_single = _json_file(evidence_paths["single"], "single execution evidence")
    raw_multi = _json_file(evidence_paths["multi"], "multi execution evidence")
    raw_replacement = _json_file(
        evidence_paths["replacement"], "replacement execution evidence"
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

    multi_spec = multi_profile.slurm
    assert multi_spec is not None
    resources = derive_slurm_job_resources(
        multi_spec,
        multi_profile.native_threads,
        multi_profile.parallelism,
    )
    expected_resources = {
        "jobs": resources.job_cap,
        "processes_per_job": resources.processes_per_job,
        "task_slots_per_job": resources.task_slots_per_job,
        "max_task_slots": resources.max_task_slots,
        "cores_per_job": resources.cores_per_job,
        "cores_per_process": resources.cores_per_process,
        "job_memory_bytes": resources.job_memory.scheduler_bytes,
        "process_memory_bytes": resources.process_memory_bytes,
    }
    for requested, process_count, process_memory in (
        (single_requested, 1, 500_363_689_984),
        (multi_requested, 2, resources.process_memory_bytes),
    ):
        _require(
            requested.get("processes_per_job") == process_count
            and requested.get("cores_per_process") == 30 // process_count
            and requested.get("process_memory_bytes") == process_memory,
            "raw requested resources differ from the approved profiles",
        )

    comparison_path = root / "multi-process-scientific-comparison.json"
    comparison = _json_file(comparison_path, "scientific comparison")
    fingerprint = next(iter(fingerprints))
    _require(
        comparison.get("accepted") is True
        and comparison.get("scientific_fingerprint") == fingerprint,
        "scientific comparison is missing or rejected",
    )
    _require(
        comparison.get("comparison_policy")
        == "stochastic-scientific-equivalence"
        and comparison.get("metric_absolute_tolerance") == 0.01
        and type(comparison.get("rounds")) is list
        and bool(comparison.get("rounds")),
        "scientific comparison lacks raw round-level equivalence evidence",
    )
    marker_path = root / "multi-process-worker-loss-marker.json"
    marker = _json_file(marker_path, "worker-loss marker")
    _require(
        marker.get("schema_version") == 1
        and marker.get("status") == "worker-loss-injected"
        and all(
            type(marker.get(name)) is expected
            for name, expected in (
                ("worker_address", str),
                ("hostname", str),
                ("pid", int),
                ("slurm_job_id", str),
            )
        ),
        "worker-loss marker is malformed",
    )
    replacement_placements = cast(
        dict[str, object], raw_replacement["fine_graph"]
    ).get("placements")
    _require(
        isinstance(replacement_placements, list)
        and any(
            type(item) is dict
            and item.get("worker_address") != marker.get("worker_address")
            and item.get("placement_accepted") is True
            for item in replacement_placements
        ),
        "replacement evidence does not prove a distinct validated live process",
    )

    performance_path = root / "multi-process-performance.json"
    performance = _performance(
        _json_file(performance_path, "performance characterization"),
        placements,
    )
    cleanup = _cleanup(root)
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
        len({item["relative_path"] for item in evidence_files})
        == len(evidence_files),
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
        "replacement": {
            "accepted": True,
            "departed_process": {
                name: marker[name]
                for name in (
                    "worker_address",
                    "hostname",
                    "pid",
                    "slurm_job_id",
                )
            },
        },
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
