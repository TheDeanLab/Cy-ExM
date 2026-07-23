from __future__ import annotations

import getpass
import json
from dataclasses import dataclass, replace
from functools import partial
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import pytest

from cyexm.errors import WorkflowExecutionError
from cyexm.registration.artifacts import FileIdentity
from cyexm.workflow.api import _execution_profile_mapping
from cyexm.workflow.models import WorkflowPlan, WorkflowResult, canonical_json
from cyexm.workflow.planning import build_workflow_plan
from cyexm.workflow.task import run_workflow_task
from hpc.environment import HpcAcceptanceContext, write_evidence_json
from hpc.fine_workflow_acceptance import build_fine_acceptance_report
from hpc.workflow_acceptance import (
    HpcWorkflowAcceptanceError,
    coordination_timeout_seconds,
    execute_cancellation_acceptance,
    execute_deterministic_parity_subprocess,
    execute_owned_slurm_workflow,
    run_fine_cancellation_task,
    run_fine_worker_loss_task,
    run_injected_fine_failure_task,
    run_profiled_fine_task,
    run_profiled_workflow_task,
    scientific_equivalence_evidence,
)

pytestmark = pytest.mark.hpc


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _ScenarioRoots:
    coarse: Path
    fine: Path
    chunk_failure: Path
    channel_failure: Path
    cancellation: Path
    replacement: Path
    deterministic: Path


def _prepare_roots(coarse_output: Path, fine_output: Path) -> _ScenarioRoots:
    roots = _ScenarioRoots(
        coarse=coarse_output / "cyexm-fine-acceptance",
        fine=fine_output / "cyexm-fine-acceptance" / "primary",
        chunk_failure=fine_output / "cyexm-fine-acceptance" / "chunk-failure",
        channel_failure=fine_output / "cyexm-fine-acceptance" / "channel-failure",
        cancellation=fine_output / "cyexm-fine-acceptance" / "cancellation",
        replacement=fine_output / "cyexm-fine-acceptance" / "replacement",
        deterministic=fine_output / "cyexm-fine-acceptance" / "deterministic",
    )
    bases = (coarse_output, fine_output)
    if any(base.is_symlink() or not base.is_dir() for base in bases):
        raise HpcWorkflowAcceptanceError(
            "fine acceptance profile output roots must be existing real directories"
        )
    if roots.coarse.exists() or roots.fine.parent.exists():
        raise HpcWorkflowAcceptanceError(
            "fine acceptance named roots must not exist before the exact run"
        )
    roots.coarse.mkdir()
    roots.fine.parent.mkdir()
    for path in (
        roots.fine,
        roots.chunk_failure,
        roots.channel_failure,
        roots.cancellation,
        roots.replacement,
        roots.deterministic,
    ):
        path.mkdir()
    for name in (
        "deterministic-inputs",
        "deterministic-parallel",
        "deterministic-sequential",
    ):
        (roots.deterministic / name).mkdir()
    return roots


def _bounded_manifest(context: HpcAcceptanceContext) -> Any:
    rounds = tuple(
        sorted(
            (
                item
                for item in context.manifest.rounds
                if item.strategy == "serial_chunked" and item.channels
            ),
            key=lambda item: item.round,
        )[:2]
    )
    if len(rounds) != 2:
        raise HpcWorkflowAcceptanceError(
            "fine acceptance requires two multi-channel serial_chunked rounds"
        )
    return replace(context.manifest, rounds=rounds)


def _profile_fingerprint(profile: Any) -> str:
    return sha256(
        canonical_json(_execution_profile_mapping(profile)).encode("utf-8")
    ).hexdigest()


def _artifact_entries(plan: WorkflowPlan) -> list[dict[str, str]]:
    root = plan.execution.paths.output_root.resolve(strict=True)
    entries = []
    for path in plan.expected_artifacts:
        resolved = path.resolve(strict=True)
        entries.append(
            {
                "relative_path": resolved.relative_to(root).as_posix(),
                "sha256": _file_sha256(resolved),
            }
        )
    return sorted(entries, key=lambda item: item["relative_path"])


def _artifact_path_entries(
    paths: list[str],
    *,
    owned_root: Path,
) -> list[dict[str, str]]:
    entries = {
        Path(path)
        .resolve(strict=True)
        .relative_to(owned_root.resolve(strict=True))
        .as_posix(): _file_sha256(Path(path))
        for path in paths
        if Path(path).is_file()
    }
    return [
        {"relative_path": relative, "sha256": entries[relative]}
        for relative in sorted(entries)
    ]


def _placement(evidence: dict[str, object]) -> dict[str, object]:
    fine = cast(dict[str, object], evidence["fine_graph"])
    raw_placements = cast(list[dict[str, object]], fine["placements"])
    tasks: list[dict[str, object]] = []
    for item in raw_placements:
        key = cast(str, item["task_key"])
        kind = key.split(":")[1]
        if kind not in {"chunk", "channel"} or not item["placement_accepted"]:
            continue
        tasks.append(
            {
                "task_key": key,
                "kind": kind,
                "round": int(key.split(":")[0].removeprefix("round-")),
                "job_id": item["slurm_job_id"],
                "hostname": item["hostname"],
                "worker": item["worker_address"],
                "started_at": item["started_at"],
                "ended_at": item["ended_at"],
                "duration_seconds": item["duration_seconds"],
                "_submitted": item["coordinator_submitted_at"],
            }
        )
    preparations = [
        item
        for item in raw_placements
        if ":preparation:" in cast(str, item["task_key"])
        and item["placement_accepted"]
        and item["status"] in {"completed", "reused"}
    ]
    if len({cast(str, item["task_key"]).split(":")[0] for item in preparations}) < 2:
        raise HpcWorkflowAcceptanceError(
            "fine acceptance lacks two completed round preparations for fairness"
        )
    both_rounds_ready = max(cast(str, item["ended_at"]) for item in preparations)
    fair_candidates = sorted(
        (
            item
            for item in tasks
            if item["kind"] == "chunk"
            and cast(str, item["_submitted"]) >= both_rounds_ready
        ),
        key=lambda item: cast(str, item["_submitted"]),
    )
    if len(fair_candidates) < 2 or (
        fair_candidates[0]["round"] == fair_candidates[1]["round"]
    ):
        raise HpcWorkflowAcceptanceError(
            "fine acceptance lacks a round-robin opportunity after two rounds "
            "were ready"
        )
    fair_rounds = [
        cast(int, fair_candidates[0]["round"]),
        cast(int, fair_candidates[1]["round"]),
    ]
    fair_order = [
        cast(str, fair_candidates[0]["task_key"]),
        cast(str, fair_candidates[1]["task_key"]),
    ]
    for item in tasks:
        item.pop("_submitted")
    scale_events = cast(list[dict[str, object]], fine["scale_events"])
    requests = sorted({cast(int, item["requested"]) for item in scale_events})
    return {
        "worker_cap": fine["worker_cap"],
        "maximum_connected_workers": max(requests),
        "scale_events": requests,
        "fair_ready_rounds": fair_rounds,
        "fair_dispatch_order": fair_order,
        "tasks": tasks,
    }


def _duration_summary(
    placement: dict[str, object], kind: str
) -> dict[str, int | float]:
    values = sorted(
        float(cast(int | float, item["duration_seconds"]))
        for item in cast(list[dict[str, object]], placement["tasks"])
        if item["kind"] == kind
    )
    middle = len(values) // 2
    median = (
        values[middle]
        if len(values) % 2
        else (values[middle - 1] + values[middle]) / 2.0
    )
    return {
        "count": len(values),
        "minimum": values[0],
        "median": median,
        "maximum": values[-1],
    }


def _fine_wall_seconds(placement: dict[str, object]) -> float:
    from datetime import datetime

    tasks = cast(list[dict[str, object]], placement["tasks"])
    nonlinear = [item for item in tasks if item["kind"] == "chunk"]
    starts = [
        datetime.fromisoformat(cast(str, item["started_at"]).replace("Z", "+00:00"))
        for item in nonlinear
    ]
    ends = [
        datetime.fromisoformat(cast(str, item["ended_at"]).replace("Z", "+00:00"))
        for item in nonlinear
    ]
    return (max(ends) - min(starts)).total_seconds()


def _profiled_performance(
    *,
    coarse_timing: Path,
    fine_timing: Path,
    fine_evidence: dict[str, object],
    fine_result: WorkflowResult,
    placement: dict[str, object],
) -> dict[str, object]:
    from datetime import datetime

    def records(root: Path) -> list[dict[str, object]]:
        return [
            cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(root.glob("*.json"))
        ]

    coarse_records = records(coarse_timing)
    fine_records = records(fine_timing)
    coarse_calls = [
        cast(dict[str, object], call)
        for record in coarse_records
        for call in cast(list[dict[str, object]], record["nonlinear_calls"])
    ]
    fine_calls = [
        cast(dict[str, object], call)
        for record in fine_records
        for call in cast(list[dict[str, object]], record["nonlinear_calls"])
    ]
    assemblies = [
        record
        for record in fine_records
        if ":assembly:" in cast(str, record["task_key"])
    ]
    if not coarse_calls or not fine_calls or not assemblies:
        raise HpcWorkflowAcceptanceError(
            "profiled acceptance lacks coarse/fine nonlinear timing boundaries"
        )

    def timestamp(value: object) -> datetime:
        return datetime.fromisoformat(cast(str, value).replace("Z", "+00:00"))

    coarse_wall = (
        max(timestamp(item["ended_at"]) for item in coarse_calls)
        - min(timestamp(item["started_at"]) for item in coarse_calls)
    ).total_seconds()
    fine_wall = (
        max(timestamp(item["ended_at"]) for item in assemblies)
        - min(timestamp(item["started_at"]) for item in fine_calls)
    ).total_seconds()
    result_wall = (
        timestamp(fine_result.ended_at) - timestamp(fine_result.started_at)
    ).total_seconds()
    task_seconds = sum(
        float(cast(int | float, item["duration_seconds"])) for item in fine_records
    )
    fine_graph = cast(dict[str, object], fine_evidence["fine_graph"])
    scale_events = cast(list[dict[str, object]], fine_graph["scale_events"])
    first_scale = scale_events[0]
    execution_started = cast(dict[str, object], fine_evidence["timestamps"])[
        "started_at"
    ]
    startup_overhead = (
        timestamp(first_scale["ended_at"]) - timestamp(execution_started)
    ).total_seconds()
    execution_ended = timestamp(
        cast(dict[str, object], fine_evidence["timestamps"])["ended_at"]
    )
    activated_at: dict[str, datetime] = {}
    worker_seconds = 0.0
    for event in scale_events:
        event_started = timestamp(event["started_at"])
        event_ended = timestamp(event["ended_at"])
        for worker in cast(list[str], event["departed"]):
            activated = activated_at.pop(worker, None)
            if activated is not None:
                worker_seconds += max(0.0, (event_started - activated).total_seconds())
        for worker in cast(list[str], event["activated"]):
            activated_at[worker] = event_ended
    worker_seconds += sum(
        max(0.0, (execution_ended - activated).total_seconds())
        for activated in activated_at.values()
    )
    if worker_seconds <= 0.0:
        raise HpcWorkflowAcceptanceError(
            "fine acceptance cannot derive positive active worker time"
        )
    return {
        "coarse_nonlinear_wall_seconds": coarse_wall,
        "fine_nonlinear_wall_seconds": fine_wall,
        "end_to_end_wall_seconds": result_wall,
        "task_seconds": task_seconds,
        "node_hours": worker_seconds / 3600.0,
        "peak_memory_bytes": max(
            cast(int, item["peak_memory_bytes"])
            for item in (*coarse_records, *fine_records)
        ),
        "startup_overhead_seconds": max(0.0, startup_overhead),
        "worker_utilization": min(
            1.0,
            task_seconds / worker_seconds,
        ),
        "chunk_duration_seconds": _duration_summary(placement, "chunk"),
        "channel_duration_seconds": _duration_summary(placement, "channel"),
    }


def _prepared_grid(
    evidence: dict[str, object],
    *,
    owned_root: Path,
) -> dict[str, object]:
    fine = cast(dict[str, object], evidence["fine_graph"])
    results = cast(list[dict[str, object]], fine["task_results"])
    descriptors = [
        cast(dict[str, object], item["prepared_descriptor"])
        for item in results
        if item["stage"] == "preparation" and item["status"] in {"completed", "reused"}
    ]
    if not descriptors:
        raise HpcWorkflowAcceptanceError(
            "fine acceptance lacks a successful prepared descriptor"
        )
    geometries = [cast(dict[str, object], item["geometry"]) for item in descriptors]
    if any(item != geometries[0] for item in geometries[1:]):
        raise HpcWorkflowAcceptanceError(
            "fine acceptance prepared fixed-grid geometries differ"
        )
    geometry = geometries[0]
    prepared_paths = sorted(
        {
            cast(str, cast(dict[str, object], descriptor[name])["path"])
            for descriptor in descriptors
            for name in (
                "fixed_scalar",
                "moving_linear_scalar",
                "fixed_mask",
                "moving_linear_mask",
            )
        }
    )
    for path in prepared_paths:
        Path(path).resolve(strict=True).relative_to(owned_root.resolve(strict=True))
    return {
        "shape_zyx": geometry["shape"],
        "origin_xyz": geometry["origin"],
        "spacing_xyz": geometry["spacing"],
        "direction": geometry["direction"],
        "prepared_paths": prepared_paths,
        "owned_roots": [str(owned_root.resolve(strict=True))],
    }


def _failure_resume(
    failed_evidence: dict[str, object],
    resumed: WorkflowResult,
    stage: str,
    owned_root: Path,
) -> dict[str, object]:
    fine = cast(dict[str, object], failed_evidence["fine_graph"])
    results = cast(list[dict[str, object]], fine["task_results"])
    failed = next(
        item
        for item in results
        if item["stage"] == stage and item["status"] == "failed"
    )
    settled = sorted(
        cast(str, item["key"])
        for item in results
        if item["stage"] == stage and item["status"] in {"completed", "reused"}
    )
    resume_evidence = json.loads(resumed.execution_evidence_json)
    resume_results = cast(
        list[dict[str, object]],
        cast(dict[str, object], resume_evidence["fine_graph"])["task_results"],
    )
    reused = sorted(
        cast(str, item["key"])
        for item in resume_results
        if item["stage"] == stage and item["status"] == "reused"
    )
    recomputed = sorted(
        cast(str, item["key"])
        for item in resume_results
        if item["stage"] == stage and item["status"] == "completed"
    )
    preserved_paths = sorted(
        {
            cast(str, path)
            for item in results
            if item["status"] in {"completed", "reused"}
            for path in cast(list[str], item["artifact_paths"])
            if Path(path).is_file()
        }
    )
    return {
        "injected_task_key": failed["key"],
        "failure_status": "failed",
        "settled_sibling_keys": settled,
        "preserved_artifacts": _artifact_path_entries(
            preserved_paths,
            owned_root=owned_root,
        ),
        "resume_reused_keys": reused,
        "resume_recomputed_keys": recomputed,
        "ordinary_artifact_store_reuse": bool(reused),
    }


def _tier1_summary(equivalence: dict[str, object]) -> dict[str, object]:
    differences: list[float] = []

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "absolute_differences" and isinstance(child, dict):
                    differences.extend(float(item) for item in child.values())
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(equivalence)
    return {
        "metric_absolute_tolerance": 0.01,
        "maximum_metric_absolute_difference": max(differences, default=0.0),
        "structure_equal": True,
    }


def _run_fine_acceptance(
    context: HpcAcceptanceContext, profiles: Any
) -> dict[str, object]:
    roots = _prepare_roots(
        profiles.coarse.paths.output_root,
        profiles.fine.paths.output_root,
    )
    manifest = _bounded_manifest(context)
    coarse_profile = replace(
        profiles.coarse,
        paths=replace(profiles.coarse.paths, output_root=roots.coarse),
    )
    fine_profile = replace(
        profiles.fine,
        paths=replace(profiles.fine.paths, output_root=roots.fine),
    )
    fine_scenario_profiles = {
        "primary": fine_profile,
        "chunk_failure": replace(
            profiles.fine,
            paths=replace(
                profiles.fine.paths,
                output_root=roots.chunk_failure,
            ),
        ),
        "channel_failure": replace(
            profiles.fine,
            paths=replace(
                profiles.fine.paths,
                output_root=roots.channel_failure,
            ),
        ),
        "cancellation": replace(
            profiles.fine,
            paths=replace(
                profiles.fine.paths,
                output_root=roots.cancellation,
            ),
        ),
        "replacement": replace(
            profiles.fine,
            paths=replace(
                profiles.fine.paths,
                output_root=roots.replacement,
            ),
        ),
    }
    coarse_plan = build_workflow_plan(manifest, coarse_profile)
    fine_plan = build_workflow_plan(manifest, fine_profile)
    if coarse_plan.scientific_fingerprint != fine_plan.scientific_fingerprint:
        raise HpcWorkflowAcceptanceError(
            "approved coarse and fine plans changed scientific identity"
        )
    scenario_roots = {
        "coarse": roots.coarse,
        "fine": roots.fine,
        "chunk_failure": roots.chunk_failure,
        "channel_failure": roots.channel_failure,
        "cancellation": roots.cancellation,
        "replacement": roots.replacement,
        "deterministic": roots.deterministic,
    }
    for root in scenario_roots.values():
        resolved = root.resolve(strict=True)
        if root is roots.coarse:
            resolved.relative_to(profiles.coarse.paths.output_root.resolve(strict=True))
        else:
            resolved.relative_to(profiles.fine.paths.output_root.resolve(strict=True))
    write_evidence_json(
        context.evidence_dir / "fine-acceptance-context.json",
        {
            "schema_version": 1,
            "git_commit": context.git_commit,
            "scientific_fingerprint": fine_plan.scientific_fingerprint,
            "approved_profiles": {
                "coarse": {
                    "identity": FileIdentity.from_path(profiles.coarse_path).to_dict(),
                    "sha256": _file_sha256(profiles.coarse_path),
                    "normalized": _execution_profile_mapping(profiles.coarse),
                },
                "fine": {
                    "identity": FileIdentity.from_path(profiles.fine_path).to_dict(),
                    "sha256": _file_sha256(profiles.fine_path),
                    "normalized": _execution_profile_mapping(profiles.fine),
                },
            },
            "derived_profiles": {
                "coarse": _execution_profile_mapping(coarse_profile),
                **{
                    name: _execution_profile_mapping(profile)
                    for name, profile in fine_scenario_profiles.items()
                },
            },
            "scenario_roots": {
                name: str(path.resolve(strict=True))
                for name, path in scenario_roots.items()
            },
        },
    )
    timing_root = context.evidence_dir / "fine-task-timings"
    timing_root.mkdir()
    coarse_timing = timing_root / "coarse"
    fine_timing = timing_root / "fine"
    coarse_timing.mkdir()
    fine_timing.mkdir()
    coarse_runner = partial(
        run_profiled_workflow_task,
        timing_root=coarse_timing,
    )
    fine_coarse_runner = partial(
        run_profiled_workflow_task,
        timing_root=fine_timing,
    )
    fine_runner = partial(run_profiled_fine_task, timing_root=fine_timing)
    user = getpass.getuser()
    cast(
        WorkflowResult,
        execute_owned_slurm_workflow(
            coarse_plan,
            runner=coarse_runner,
            evidence_dir=context.evidence_dir,
            operation="fine-acceptance-coarse",
            user=user,
        ),
    )
    fine = cast(
        WorkflowResult,
        execute_owned_slurm_workflow(
            fine_plan,
            runner=fine_coarse_runner,
            fine_runner=fine_runner,
            evidence_dir=context.evidence_dir,
            operation="fine-acceptance-primary",
            user=user,
        ),
    )
    equivalence = scientific_equivalence_evidence(coarse_plan, fine_plan)
    fine_evidence = cast(dict[str, object], json.loads(fine.execution_evidence_json))
    placement = _placement(fine_evidence)

    failure_records: dict[str, tuple[dict[str, object], WorkflowResult]] = {}
    for stage, root in (
        ("chunk", roots.chunk_failure),
        ("channel", roots.channel_failure),
    ):
        profile = fine_scenario_profiles[f"{stage}_failure"]
        plan = build_workflow_plan(manifest, profile)
        marker = root / f"injected-{stage}.json"
        runner = partial(
            run_injected_fine_failure_task,
            target_round=plan.tasks[0].round,
            target_stage=stage,
            failure_marker=marker,
            settled_marker=root / f"settled-{stage}-sibling.json",
            timeout_seconds=coordination_timeout_seconds(profile),
        )
        try:
            execute_owned_slurm_workflow(
                plan,
                runner=run_workflow_task,
                fine_runner=runner,
                evidence_dir=context.evidence_dir,
                operation=f"fine-{stage}-failure",
                user=user,
            )
        except WorkflowExecutionError:
            failed_evidence = cast(
                dict[str, object],
                json.loads(
                    (
                        context.evidence_dir
                        / f"fine-{stage}-failure-slurm-execution.json"
                    ).read_text(encoding="utf-8")
                ),
            )
        else:
            raise HpcWorkflowAcceptanceError(
                f"injected fine {stage} failure unexpectedly succeeded"
            )
        resumed = cast(
            WorkflowResult,
            execute_owned_slurm_workflow(
                plan,
                runner=run_workflow_task,
                evidence_dir=context.evidence_dir,
                operation=f"fine-{stage}-resume",
                user=user,
            ),
        )
        failure_records[stage] = (failed_evidence, resumed)

    cancellation_profile = fine_scenario_profiles["cancellation"]
    cancellation_plan = build_workflow_plan(manifest, cancellation_profile)
    ready = roots.cancellation / "cancellation-ready.json"
    cancellation_runner = partial(
        run_fine_cancellation_task,
        target_round=cancellation_plan.tasks[0].round,
        target_stage="chunk",
        ready_marker=ready,
        timeout_seconds=coordination_timeout_seconds(cancellation_profile),
    )
    execute_cancellation_acceptance(
        cancellation_plan,
        runner=run_workflow_task,
        fine_runner=cancellation_runner,
        evidence_dir=context.evidence_dir,
        operation="fine-cancellation",
        user=user,
        ready_marker=ready,
        successful_round=cancellation_plan.tasks[0].round,
        blocked_round=cancellation_plan.tasks[0].round,
        timeout_seconds=coordination_timeout_seconds(cancellation_profile),
    )
    cancelled_evidence = json.loads(
        (context.evidence_dir / "fine-cancellation-slurm-execution.json").read_text(
            encoding="utf-8"
        )
    )
    cancellation_resume = cast(
        WorkflowResult,
        execute_owned_slurm_workflow(
            cancellation_plan,
            runner=run_workflow_task,
            evidence_dir=context.evidence_dir,
            operation="fine-cancellation-resume",
            user=user,
        ),
    )

    replacement_profile = fine_scenario_profiles["replacement"]
    replacement_plan = build_workflow_plan(manifest, replacement_profile)
    replacement_runner = partial(
        run_fine_worker_loss_task,
        target_round=replacement_plan.tasks[0].round,
        target_stage="chunk",
        loss_marker=roots.replacement / "worker-loss.json",
    )
    replacement_result = cast(
        WorkflowResult,
        execute_owned_slurm_workflow(
            replacement_plan,
            runner=run_workflow_task,
            fine_runner=replacement_runner,
            evidence_dir=context.evidence_dir,
            operation="fine-worker-replacement",
            user=user,
        ),
    )
    replacement_evidence = cast(
        dict[str, object], json.loads(replacement_result.execution_evidence_json)
    )

    deterministic = execute_deterministic_parity_subprocess(
        execution_path=profiles.coarse_path,
        fine_execution_path=profiles.fine_path,
        owned_root=roots.deterministic,
        evidence_dir=context.evidence_dir,
        timeout_seconds=coordination_timeout_seconds(profiles.fine) + 600.0,
    )
    internal_parity = cast(dict[str, object], deterministic["strict_internal_parity"])
    tier2_channels = [
        {
            "relative_path": channel.output.relative_to(
                fine_plan.execution.paths.output_root
            ).as_posix(),
            "exact": True,
        }
        for task in fine_plan.tasks
        for channel in task.channels
    ]
    prepared = _prepared_grid(fine_evidence, owned_root=roots.fine)
    performance = _profiled_performance(
        coarse_timing=coarse_timing,
        fine_timing=fine_timing,
        fine_evidence=fine_evidence,
        fine_result=fine,
        placement=placement,
    )
    cancellation_graph = cast(
        dict[str, object], cast(dict[str, object], cancelled_evidence)["fine_graph"]
    )
    replacement_graph = cast(dict[str, object], replacement_evidence["fine_graph"])
    cancellation_results = cast(
        list[dict[str, object]], cancellation_graph["task_results"]
    )
    cancellation_reused = sorted(
        cast(str, item["key"])
        for item in json.loads(cancellation_resume.execution_evidence_json)[
            "fine_graph"
        ]["task_results"]
        if item["status"] == "reused"
    )
    cancellation_recomputed = sorted(
        cast(str, item["key"])
        for item in json.loads(cancellation_resume.execution_evidence_json)[
            "fine_graph"
        ]["task_results"]
        if item["status"] == "completed"
    )
    raw_files = tuple(
        sorted(path for path in context.evidence_dir.rglob("*") if path.is_file())
    )
    input_paths = tuple(
        sorted(
            {
                *(channel.source for channel in coarse_plan.reference_task.channels),
                *(task.moving_image for task in coarse_plan.tasks),
                *(
                    channel.source
                    for task in coarse_plan.tasks
                    for channel in task.channels
                ),
            }
        )
    )
    report = build_fine_acceptance_report(
        identities={
            "git_commit": context.git_commit,
            "package_versions": context.package_versions,
            "scientific_fingerprint": fine_plan.scientific_fingerprint,
            "coarse_execution_fingerprint": _profile_fingerprint(coarse_profile),
            "fine_execution_fingerprint": _profile_fingerprint(fine_profile),
            "coarse_profile": FileIdentity.from_path(profiles.coarse_path).to_dict(),
            "fine_profile": FileIdentity.from_path(profiles.fine_path).to_dict(),
            "coarse_profile_sha256": _file_sha256(profiles.coarse_path),
            "fine_profile_sha256": _file_sha256(profiles.fine_path),
            "inputs": [FileIdentity.from_path(path).to_dict() for path in input_paths],
            "input_checksums": [
                {
                    "path": FileIdentity.from_path(path).path,
                    "sha256": _file_sha256(path),
                }
                for path in input_paths
            ],
        },
        scientific={
            "coarse_fingerprint": coarse_plan.scientific_fingerprint,
            "fine_fingerprint": fine_plan.scientific_fingerprint,
            "public_artifacts_coarse": _artifact_entries(coarse_plan),
            "public_artifacts_fine": _artifact_entries(fine_plan),
            "tier1": _tier1_summary(equivalence),
            "tier2": {
                "seed": deterministic["seed"],
                "native_threads": 1,
                "affine_exact": True,
                "local_warp_exact": internal_parity["local_warp_exact"],
                "full_warp_exact": True,
                "diagnostic_exact": internal_parity["diagnostic_exact"],
                "metrics_exact": True,
                "channels": tier2_channels,
            },
        },
        fixed_grid=prepared,
        placement=placement,
        durability={
            "chunk_failure_resume": _failure_resume(
                failure_records["chunk"][0],
                failure_records["chunk"][1],
                "chunk",
                roots.chunk_failure,
            ),
            "channel_failure_resume": _failure_resume(
                failure_records["channel"][0],
                failure_records["channel"][1],
                "channel",
                roots.channel_failure,
            ),
            "cancellation_resume": {
                "status": "cancelled",
                "structured_cancellation": cancellation_graph["cancelled"],
                "preserved_artifacts": _artifact_path_entries(
                    [
                        cast(str, path)
                        for item in cancellation_results
                        for path in cast(list[str], item["artifact_paths"])
                        if Path(path).is_file()
                    ],
                    owned_root=roots.cancellation,
                ),
                "resume_reused_keys": cancellation_reused,
                "resume_recomputed_keys": cancellation_recomputed,
                "replacement_events": sum(
                    bool(item["departed"])
                    for item in cast(
                        list[dict[str, object]], replacement_graph["scale_events"]
                    )
                ),
                "liveness_proven": (
                    replacement_result.succeeded and bool(cancellation_recomputed)
                ),
                "owned_job_ids": sorted(
                    {
                        cast(str, item["slurm_job_id"])
                        for item in cast(
                            list[dict[str, object]],
                            cancellation_graph["placements"],
                        )
                        if item["slurm_job_id"]
                    }
                ),
                "remaining_owned_job_ids": [],
            },
        },
        performance=performance,
        evidence_root=context.evidence_dir,
        evidence_paths=raw_files,
    )
    write_evidence_json(
        context.evidence_dir / "fine-grained-acceptance.json",
        report,
    )
    return report


def test_fine_grained_slurm_acceptance(
    hpc_context: HpcAcceptanceContext,
    fine_hpc_profiles: Any,
) -> None:
    report = _run_fine_acceptance(hpc_context, fine_hpc_profiles)
    assert report["status"] == "completed"
