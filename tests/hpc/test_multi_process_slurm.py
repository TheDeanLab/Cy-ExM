from __future__ import annotations

import getpass
import json
from dataclasses import replace
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, cast

import pytest

from cyexm.workflow.models import WorkflowResult
from cyexm.workflow.planning import build_workflow_plan
from cyexm.workflow.task import run_workflow_task
from hpc.environment import HpcAcceptanceContext, write_evidence_json
from hpc.fine_workflow_acceptance import fine_acceptance_manifest
from hpc.multi_process_slurm_acceptance import (
    HpcMultiProcessAcceptanceError,
    build_multi_process_acceptance_report,
)
from hpc.workflow_acceptance import (
    execute_owned_slurm_workflow,
    run_fine_worker_loss_task,
    run_profiled_fine_task,
    scientific_equivalence_evidence,
)

pytestmark = pytest.mark.hpc


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _duration(result: WorkflowResult) -> float:
    seconds = (
        _timestamp(result.ended_at) - _timestamp(result.started_at)
    ).total_seconds()
    if seconds <= 0:
        raise HpcMultiProcessAcceptanceError(
            "workflow timing must have positive coordinator duration"
        )
    return seconds


def _process_peaks(
    timing_root: Path,
    result: WorkflowResult,
) -> dict[str, int]:
    execution = cast(
        dict[str, object],
        json.loads(result.execution_evidence_json),
    )
    graph = cast(dict[str, object], execution["fine_graph"])
    placements = cast(list[dict[str, object]], graph["placements"])
    accepted = {
        cast(str, item["task_key"]): cast(str, item["worker_address"])
        for item in placements
        if item.get("placement_accepted") is True
        and item.get("status") in {"completed", "reused"}
    }
    peaks: dict[str, int] = {}
    for path in sorted(timing_root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if type(payload) is not dict:
            raise HpcMultiProcessAcceptanceError(
                f"timing record must be a JSON object: {path}"
            )
        task_key = payload.get("task_key")
        peak = payload.get("peak_memory_bytes")
        if task_key not in accepted:
            continue
        if type(peak) is not int or peak < 1:
            raise HpcMultiProcessAcceptanceError(
                f"timing record lacks positive peak RSS: {path}"
            )
        address = accepted[cast(str, task_key)]
        peaks[address] = max(peaks.get(address, 0), peak)
    if len(peaks) < 2:
        raise HpcMultiProcessAcceptanceError(
            "multi-process timing does not cover two validated worker processes"
        )
    return peaks


def _fresh_root(base: Path, relative: str) -> Path:
    root = base / relative
    if root.exists():
        raise HpcMultiProcessAcceptanceError(
            f"focused acceptance root must be fresh: {root}"
        )
    root.mkdir(parents=True)
    return root


def test_multi_process_slurm_acceptance(
    hpc_context: HpcAcceptanceContext,
    multi_process_hpc_profiles: Any,
) -> None:
    profiles = multi_process_hpc_profiles
    manifest, targets = fine_acceptance_manifest(hpc_context.manifest)
    single_root = _fresh_root(
        profiles.single.paths.output_root,
        "cyexm-multi-process-acceptance",
    )
    multi_parent = _fresh_root(
        profiles.multi.paths.output_root,
        "cyexm-multi-process-acceptance",
    )
    multi_root = _fresh_root(multi_parent, "primary")
    replacement_root = _fresh_root(multi_parent, "replacement")
    timing_root = hpc_context.evidence_dir / "multi-process-task-timings"
    timing_root.mkdir()
    single_timing = timing_root / "single"
    multi_timing = timing_root / "multi"
    single_timing.mkdir()
    multi_timing.mkdir()

    single_profile = replace(
        profiles.single,
        paths=replace(profiles.single.paths, output_root=single_root),
    )
    multi_profile = replace(
        profiles.multi,
        paths=replace(profiles.multi.paths, output_root=multi_root),
    )
    replacement_profile = replace(
        profiles.multi,
        paths=replace(profiles.multi.paths, output_root=replacement_root),
    )
    single_plan = build_workflow_plan(manifest, single_profile)
    multi_plan = build_workflow_plan(manifest, multi_profile)
    replacement_plan = build_workflow_plan(manifest, replacement_profile)
    if len(
        {
            single_plan.scientific_fingerprint,
            multi_plan.scientific_fingerprint,
            replacement_plan.scientific_fingerprint,
        }
    ) != 1:
        raise HpcMultiProcessAcceptanceError(
            "focused process-count plans changed scientific identity"
        )

    user = getpass.getuser()
    single_result = cast(
        WorkflowResult,
        execute_owned_slurm_workflow(
            single_plan,
            runner=run_workflow_task,
            fine_runner=partial(
                run_profiled_fine_task,
                timing_root=single_timing,
            ),
            evidence_dir=hpc_context.evidence_dir,
            operation="multi-process-single",
            user=user,
        ),
    )
    multi_result = cast(
        WorkflowResult,
        execute_owned_slurm_workflow(
            multi_plan,
            runner=run_workflow_task,
            fine_runner=partial(
                run_profiled_fine_task,
                timing_root=multi_timing,
            ),
            evidence_dir=hpc_context.evidence_dir,
            operation="multi-process-multi",
            user=user,
        ),
    )
    comparison = scientific_equivalence_evidence(single_plan, multi_plan)
    write_evidence_json(
        hpc_context.evidence_dir / "multi-process-scientific-comparison.json",
        {"accepted": True, **comparison},
    )

    marker = hpc_context.evidence_dir / "multi-process-worker-loss-marker.json"
    replacement_result = cast(
        WorkflowResult,
        execute_owned_slurm_workflow(
            replacement_plan,
            runner=run_workflow_task,
            fine_runner=partial(
                run_fine_worker_loss_task,
                target_key=targets["replacement_target"],
                loss_marker=marker,
            ),
            evidence_dir=hpc_context.evidence_dir,
            operation="multi-process-replacement",
            user=user,
        ),
    )
    write_evidence_json(
        hpc_context.evidence_dir / "multi-process-performance.json",
        {
            "single_duration_seconds": _duration(single_result),
            "multi_duration_seconds": _duration(multi_result),
            "peak_rss_bytes_by_process": _process_peaks(
                multi_timing,
                multi_result,
            ),
        },
    )
    report = build_multi_process_acceptance_report(
        context=hpc_context,
        single_profile=profiles.single,
        multi_profile=profiles.multi,
        single_result=single_result,
        multi_result=multi_result,
        replacement_result=replacement_result,
        evidence_dir=hpc_context.evidence_dir,
    )
    write_evidence_json(
        hpc_context.evidence_dir / "multi-process-slurm-acceptance.json",
        report,
    )
    assert report["status"] == "completed"
