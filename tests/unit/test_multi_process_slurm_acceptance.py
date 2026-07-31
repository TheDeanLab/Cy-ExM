from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from hpc.multi_process_slurm_acceptance import (
    HpcMultiProcessAcceptanceError,
    build_multi_process_acceptance_report,
    multi_process_acceptance_profiles,
)

from cyexm.config.models import (
    ExecutionPaths,
    ExecutionProfile,
    NativeThreadSpec,
    ParallelismSpec,
    SlurmSpec,
)
from cyexm.workflow.models import WorkflowResult

pytestmark = pytest.mark.unit


def _profile(tmp_path: Path, *, processes: int, name: str) -> ExecutionProfile:
    source = tmp_path / f"{name}.yaml"
    source.write_text(f"profile: {name}\n", encoding="utf-8")
    dataset = tmp_path / "dataset"
    dataset.mkdir(exist_ok=True)
    output = tmp_path / f"{name}-output"
    output.mkdir()
    scratch = tmp_path / f"{name}-scratch"
    scratch.mkdir()
    return ExecutionProfile(
        schema_version=1,
        executor="slurm",
        paths=ExecutionPaths(
            dataset_root=dataset,
            output_root=output,
            worker_local_directory=scratch,
        ),
        native_threads=NativeThreadSpec(itk=15, blas=15),
        local_dask=None,
        slurm=SlurmSpec(
            queue="512GB",
            workers=3,
            cores_per_worker=30,
            processes_per_worker=processes,
            memory="500GB",
            walltime="72:00:00",
            interface="ib0",
            scheduler_interface="ib0",
            startup_timeout="15m",
            task_retries=1,
            job_name=f"cyexm-{name}",
            job_extra_directives=("--exclusive",),
        ),
        source_path=source,
        parallelism=ParallelismSpec(mode="fine_grained"),
    )


def _placement(
    *,
    task: str,
    address: str,
    pid: int,
    started: str,
    ended: str,
) -> dict[str, object]:
    return {
        "task_key": task,
        "attempt": 1,
        "status": "completed",
        "dask_task_key": f"cyexm-{task}",
        "coordinator_submitted_at": "2026-07-30T10:00:00Z",
        "coordinator_ready_at": "2026-07-30T10:00:00Z",
        "started_at": started,
        "ended_at": ended,
        "duration_seconds": 10.0,
        "worker_address": address,
        "hostname": "node-a",
        "pid": pid,
        "slurm_job_id": "12345",
        "native_threads": [],
        "artifact_paths": [],
        "reused_artifacts": [],
        "log_path": f"/archive/{task}.log",
        "placement_accepted": True,
        "rejection_reason": None,
        "failure": None,
    }


def _execution(processes: int) -> dict[str, object]:
    placements = (
        []
        if processes == 1
        else [
            _placement(
                task="round-002:chunk:0000",
                address="tcp://node-a:1001",
                pid=101,
                started="2026-07-30T10:00:01Z",
                ended="2026-07-30T10:00:11Z",
            ),
            _placement(
                task="round-003:chunk:0000",
                address="tcp://node-a:1002",
                pid=102,
                started="2026-07-30T10:00:02Z",
                ended="2026-07-30T10:00:12Z",
            ),
        ]
    )
    return {
        "schema_version": 3,
        "requested": {
            "processes_per_job": processes,
            "cores_per_process": 30 // processes,
            "process_memory_bytes": (
                500_363_689_984 if processes == 1 else 233 * 1024**3
            ),
        },
        "fine_graph": {"placements": placements},
    }


def _result(tmp_path: Path, name: str, processes: int) -> WorkflowResult:
    run_record = tmp_path / f"{name}-run.json"
    run_record.write_text("{}\n", encoding="utf-8")
    return WorkflowResult(
        dataset_id="dataset",
        scientific_fingerprint="f" * 64,
        status="completed",
        started_at="2026-07-30T10:00:00Z",
        ended_at="2026-07-30T10:01:00Z",
        rounds=(),
        run_record=str(run_record),
        execution_evidence_json=json.dumps(_execution(processes)),
    )


def _write_raw_evidence(
    root: Path,
    *,
    single: WorkflowResult,
    multi: WorkflowResult,
    replacement: WorkflowResult,
) -> None:
    payloads: dict[str, object] = {
        "multi-process-single-slurm-execution.json": json.loads(
            single.execution_evidence_json
        ),
        "multi-process-multi-slurm-execution.json": json.loads(
            multi.execution_evidence_json
        ),
        "multi-process-replacement-slurm-execution.json": json.loads(
            replacement.execution_evidence_json
        ),
        "multi-process-scientific-comparison.json": {
            "accepted": True,
            "scientific_fingerprint": "f" * 64,
            "comparison_policy": "stochastic-scientific-equivalence",
            "metric_absolute_tolerance": 0.01,
            "rounds": [{"round": 2}],
        },
        "multi-process-performance.json": {
            "single_duration_seconds": 120.0,
            "multi_duration_seconds": 80.0,
            "peak_rss_bytes_by_process": {
                "tcp://node-a:1001": 1000,
                "tcp://node-a:1002": 2000,
            },
        },
        "multi-process-worker-loss-marker.json": {
            "schema_version": 1,
            "status": "worker-loss-injected",
            "worker_address": "tcp://node-a:1001",
            "hostname": "node-a",
            "pid": 101,
            "slurm_job_id": "12345",
        },
    }
    for operation in ("single", "multi", "replacement"):
        payloads[f"multi-process-{operation}-outcome.json"] = {
            "schema_version": 1,
            "operation": f"multi-process-{operation}",
            "status": "completed",
            "owned_job_ids": ["12345"],
            "cleanup_outcome": "empty",
            "workflow_result": {},
        }
        payloads[f"multi-process-{operation}-queue-after.json"] = {
            "outcome": "empty",
            "jobs": [],
        }
    for name, payload in payloads.items():
        (root / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _accepted_inputs(
    tmp_path: Path,
) -> tuple[
    object,
    ExecutionProfile,
    ExecutionProfile,
    WorkflowResult,
    WorkflowResult,
    WorkflowResult,
]:
    single = _profile(tmp_path, processes=1, name="single")
    multi = _profile(tmp_path, processes=2, name="multi")
    single_result = _result(tmp_path, "single", 1)
    multi_result = _result(tmp_path, "multi", 2)
    replacement_result = _result(tmp_path, "replacement", 2)
    _write_raw_evidence(
        tmp_path,
        single=single_result,
        multi=multi_result,
        replacement=replacement_result,
    )
    context = SimpleNamespace(
        evidence_dir=tmp_path,
        execution_path=single.source_path,
        git_commit="a" * 40,
        package_versions={"cy-exm": "0.1.0"},
    )
    return (
        context,
        single,
        multi,
        single_result,
        multi_result,
        replacement_result,
    )


def test_profiles_require_exact_one_and_two_process_pair(tmp_path: Path) -> None:
    single = _profile(tmp_path, processes=1, name="single")
    multi = _profile(tmp_path, processes=2, name="multi")

    assert multi_process_acceptance_profiles(single, multi) == (single, multi)

    with pytest.raises(HpcMultiProcessAcceptanceError, match="two processes"):
        multi_process_acceptance_profiles(
            single,
            replace(
                multi,
                slurm=replace(cast(SlurmSpec, multi.slurm), processes_per_worker=3),
            ),
        )
    with pytest.raises(HpcMultiProcessAcceptanceError, match="otherwise identical"):
        multi_process_acceptance_profiles(
            single,
            replace(
                multi,
                slurm=replace(cast(SlurmSpec, multi.slurm), walltime="71:00:00"),
            ),
        )


def test_report_derives_overlap_resources_cleanup_and_performance(
    tmp_path: Path,
) -> None:
    context, single, multi, single_result, multi_result, replacement_result = (
        _accepted_inputs(tmp_path)
    )

    report = build_multi_process_acceptance_report(
        context=cast(object, context),  # type: ignore[arg-type]
        single_profile=single,
        multi_profile=multi,
        single_result=single_result,
        multi_result=multi_result,
        replacement_result=replacement_result,
        evidence_dir=tmp_path,
    )

    assert report["schema_version"] == 1
    assert report["scientific_parity"]["accepted"] is True  # type: ignore[index]
    assert report["placement"]["same_job_distinct_process_overlap"] is True  # type: ignore[index]
    assert report["resources"]["processes_per_job"] == 2  # type: ignore[index]
    assert report["resources"]["cores_per_process"] == 15  # type: ignore[index]
    assert report["resources"]["process_memory_bytes"] == 233 * 1024**3  # type: ignore[index]
    assert report["cleanup"]["accepted"] is True  # type: ignore[index]
    assert report["performance_characterization"]["speedup"] == 1.5  # type: ignore[index]
    assert report["replacement"]["accepted"] is True  # type: ignore[index]
    assert report["evidence_files"]


@pytest.mark.parametrize(
    "mutation",
    (
        "same_pid",
        "non_overlap",
        "different_job",
        "missing_comparison",
        "unvalidated_placement",
        "stale_cleanup",
        "wrong_resource",
        "missing_marker",
        "missing_timing",
    ),
)
def test_report_rejects_corrupt_raw_authority(
    tmp_path: Path,
    mutation: str,
) -> None:
    context, single, multi, single_result, multi_result, replacement_result = (
        _accepted_inputs(tmp_path)
    )
    evidence = json.loads(multi_result.execution_evidence_json)
    placements = evidence["fine_graph"]["placements"]
    if mutation == "same_pid":
        placements[1]["pid"] = placements[0]["pid"]
    elif mutation == "non_overlap":
        placements[1]["started_at"] = "2026-07-30T10:00:12Z"
    elif mutation == "different_job":
        placements[1]["slurm_job_id"] = "67890"
    elif mutation == "unvalidated_placement":
        placements[1]["placement_accepted"] = False
    elif mutation == "wrong_resource":
        evidence["requested"]["process_memory_bytes"] = 1
    if mutation in {
        "same_pid",
        "non_overlap",
        "different_job",
        "unvalidated_placement",
        "wrong_resource",
    }:
        multi_result = replace(
            multi_result,
            execution_evidence_json=json.dumps(evidence),
        )
        (tmp_path / "multi-process-multi-slurm-execution.json").write_text(
            json.dumps(evidence),
            encoding="utf-8",
        )
    elif mutation == "missing_comparison":
        (tmp_path / "multi-process-scientific-comparison.json").unlink()
    elif mutation == "stale_cleanup":
        queue = tmp_path / "multi-process-multi-queue-after.json"
        queue.write_text(
            json.dumps({"outcome": "pending", "jobs": [{"job_id": "12345"}]}),
            encoding="utf-8",
        )
    elif mutation == "missing_marker":
        (tmp_path / "multi-process-worker-loss-marker.json").unlink()
    elif mutation == "missing_timing":
        (tmp_path / "multi-process-performance.json").unlink()

    with pytest.raises(HpcMultiProcessAcceptanceError):
        build_multi_process_acceptance_report(
            context=cast(object, context),  # type: ignore[arg-type]
            single_profile=single,
            multi_profile=multi,
            single_result=single_result,
            multi_result=multi_result,
            replacement_result=replacement_result,
            evidence_dir=tmp_path,
        )
