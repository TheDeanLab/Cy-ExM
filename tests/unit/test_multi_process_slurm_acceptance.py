from __future__ import annotations

import json
from copy import deepcopy
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
        "hostname": "node-0",
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


def _probe(address: str, *, job: str, host: str, pid: int) -> dict[str, object]:
    return {
        "worker_address": address,
        "hostname": host,
        "pid": pid,
        "slurm_job_id": job,
        "ok": True,
        "memory_limit_bytes": 1,
    }


def _execution(processes: int, *, replacement: bool = False) -> dict[str, object]:
    process_memory = 500_363_689_984 if processes == 1 else 233 * 1024**3
    probes = [
        _probe(
            f"tcp://node-{job_index}:{1001 + process_index}",
            job=str(12345 + job_index),
            host=f"node-{job_index}",
            pid=100 * (job_index + 1) + process_index + 1,
        )
        for job_index in range(3)
        for process_index in range(processes)
    ]
    for probe in probes:
        probe["memory_limit_bytes"] = process_memory
    placements = (
        [
            _placement(
                task="round-002:chunk:0000",
                address="tcp://node-0:1001",
                pid=101,
                started="2026-07-30T10:00:01Z",
                ended="2026-07-30T10:00:11Z",
            ),
            _placement(
                task="round-003:chunk:0000",
                address="tcp://node-0:1002",
                pid=102,
                started="2026-07-30T10:00:02Z",
                ended="2026-07-30T10:00:12Z",
            ),
        ]
        if processes == 2
        else []
    )
    events: list[dict[str, object]] = [
        {
            "requested_jobs": 3,
            "expected_processes": 3 * processes,
            "connected": [probe["worker_address"] for probe in probes],
            "departed": [],
            "probed": [probe["worker_address"] for probe in probes],
            "activated": [probe["worker_address"] for probe in probes],
            "activated_probes": probes,
            "failed": [],
            "started_at": "2026-07-30T09:59:00Z",
            "ended_at": "2026-07-30T09:59:01Z",
        }
    ]
    if replacement:
        lost = probes[0]
        replacement_probe = _probe(
            "tcp://node-0:2001",
            job="12345",
            host="node-0",
            pid=999,
        )
        replacement_probe["memory_limit_bytes"] = process_memory
        events.append(
            {
                "requested_jobs": 3,
                "expected_processes": 6,
                "connected": [replacement_probe["worker_address"]],
                "departed": [lost["worker_address"]],
                "probed": [replacement_probe["worker_address"]],
                "activated": [replacement_probe["worker_address"]],
                "activated_probes": [replacement_probe],
                "failed": [],
                "started_at": "2026-07-30T10:00:20Z",
                "ended_at": "2026-07-30T10:00:21Z",
            }
        )
        failed = _placement(
            task="round-002:chunk:0000",
            address="tcp://node-0:1001",
            pid=101,
            started="2026-07-30T10:00:10Z",
            ended="2026-07-30T10:00:20Z",
        )
        failed.update(
            {
                "status": "failed",
                "placement_accepted": False,
                "rejection_reason": "worker departed",
                "failure": {"type": "KilledWorker", "message": "worker departed"},
            }
        )
        recovered = _placement(
            task="round-002:chunk:0000",
            address="tcp://node-0:2001",
            pid=999,
            started="2026-07-30T10:00:21Z",
            ended="2026-07-30T10:00:31Z",
        )
        recovered["attempt"] = 2
        recovered["coordinator_ready_at"] = "2026-07-30T10:00:22Z"
        recovered["artifact_paths"] = ["/archive/recovered.tif"]
        sibling = _placement(
            task="round-003:chunk:0000",
            address="tcp://node-0:1002",
            pid=102,
            started="2026-07-30T10:00:22Z",
            ended="2026-07-30T10:00:32Z",
        )
        sibling["coordinator_ready_at"] = "2026-07-30T10:00:23Z"
        sibling["artifact_paths"] = ["/archive/sibling.tif"]
        placements = [failed, recovered, sibling]
    return {
        "schema_version": 3,
        "requested": {
            "jobs": 3,
            "configured_jobs": 3,
            "processes_per_job": processes,
            "task_slots_per_job": processes,
            "max_task_slots": 3 * processes,
            "cores_per_process": 30 // processes,
            "process_memory_bytes": process_memory,
            "startup_timeout": "15m",
            "task_retries": 1,
            "memory": {
                "profile": "500GB",
                "semantic_bytes": 500_000_000_000,
                "scheduler": "466G",
                "scheduler_bytes": 500_363_689_984,
                "dask": "466GiB",
            },
        },
        "fine_graph": {
            "scale_events": events,
            "placements": placements,
        },
    }


def _result(
    tmp_path: Path,
    name: str,
    processes: int,
    *,
    replacement: bool = False,
) -> WorkflowResult:
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
        execution_evidence_json=json.dumps(
            _execution(processes, replacement=replacement)
        ),
    )


def _scientific_comparison(dataset_root: Path) -> dict[str, object]:
    numeric = {
        "shape": [2, 3, 4],
        "reference_dtype": "float32",
        "candidate_dtype": "float32",
        "array_equal": False,
        "close_required": False,
        "maximum_absolute_difference": 0.001,
        "mean_absolute_difference": 0.0001,
        "p99_9_absolute_difference": 0.0009,
        "rtol": 1e-4,
        "atol": 1e-5,
    }
    geometry = {
        "shape": [2, 3, 4],
        "origin": [0.0, 0.0, 0.0],
        "spacing": [0.2, 0.2, 0.5],
        "direction": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    }

    def image(name: str) -> dict[str, object]:
        return {
            "reference": f"/single/{name}.tif",
            "candidate": f"/multi/{name}.tif",
            "dtype": "float32",
            "shape": [2, 3, 4],
            "geometry": geometry,
            "numeric": numeric,
        }

    def metric(value: float) -> dict[str, object]:
        return {
            "reference": {
                "Correlation": value,
                "MattesMutualInformation": value,
            },
            "candidate": {
                "Correlation": value,
                "MattesMutualInformation": value,
            },
            "absolute_differences": {
                "Correlation": 0.0,
                "MattesMutualInformation": 0.0,
            },
            "absolute_tolerance": 0.01,
        }

    payload: dict[str, object] = {
        "accepted": True,
        "schema_version": 1,
        "scientific_fingerprint": "f" * 64,
        "rtol": 1e-4,
        "atol": 1e-5,
        "comparison_policy": "stochastic-scientific-equivalence",
        "metric_absolute_tolerance": 0.01,
        "voxel_differences_are_characterization": True,
        "rounds": [
            {
                "round": 2,
                "strategy": "serial_chunked",
                "affine": {
                    "transform_type": "AffineTransform",
                    "dimension": 3,
                    "fixed_parameters_exact": True,
                    "parameters": {**numeric, "shape": [12]},
                },
                "warp": {
                    "geometry": {
                        **geometry,
                        "has_components": True,
                    },
                    "numeric": {**numeric, "shape": [2, 3, 4, 3]},
                },
                "registered": image("registered"),
                "metrics": {
                    "pre-registration": metric(0.5),
                    "linearly-registered": metric(0.6),
                    "nonlinear-registered": metric(0.7),
                },
                "channels": [
                    {
                        "source": str((dataset_root / "channel.tif").resolve()),
                        "interpolation": "linear",
                        "comparison": image("channel"),
                    }
                ],
            }
        ],
    }
    rounds = cast(list[dict[str, object]], payload["rounds"])
    second_round = deepcopy(rounds[0])
    second_round["round"] = 3
    cast(list[dict[str, object]], second_round["channels"])[0]["source"] = str(
        (dataset_root / "channel-3.tif").resolve()
    )
    rounds.append(second_round)
    return payload


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
        "multi-process-scientific-comparison.json": _scientific_comparison(
            root / "dataset"
        ),
        "multi-process-performance.json": {
            "single_duration_seconds": 120.0,
            "multi_duration_seconds": 80.0,
            "peak_rss_bytes_by_process": {
                "tcp://node-0:1001": 1000,
                "tcp://node-0:1002": 2000,
            },
        },
        "multi-process-worker-loss-marker.json": {
            "schema_version": 1,
            "status": "worker-loss-injected",
            "task_key": "round-002:chunk:0000",
            "worker_address": "tcp://node-0:1001",
            "hostname": "node-0",
            "pid": 101,
            "slurm_job_id": "12345",
            "injected_at": "2026-07-30T10:00:20Z",
        },
    }
    for operation in ("single", "multi", "replacement"):
        payloads[f"multi-process-{operation}-outcome.json"] = {
            "schema_version": 1,
            "operation": f"multi-process-{operation}",
            "status": "completed",
            "owned_job_ids": ["12345", "12346", "12347"],
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
    replacement_result = _result(
        tmp_path,
        "replacement",
        2,
        replacement=True,
    )
    _write_raw_evidence(
        tmp_path,
        single=single_result,
        multi=multi_result,
        replacement=replacement_result,
    )
    context = SimpleNamespace(
        evidence_dir=tmp_path,
        execution_path=single.source_path,
        manifest=SimpleNamespace(
            rounds=(
                SimpleNamespace(
                    round=2,
                    strategy="serial_chunked",
                    channels=(
                        SimpleNamespace(
                            file=Path("channel.tif"),
                            interpolation="linear",
                        ),
                    ),
                ),
                SimpleNamespace(
                    round=3,
                    strategy="serial_chunked",
                    channels=(
                        SimpleNamespace(
                            file=Path("channel-3.tif"),
                            interpolation="linear",
                        ),
                    ),
                ),
            )
        ),
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


def test_report_accepts_manifest_nearest_neighbor_interpolation(
    tmp_path: Path,
) -> None:
    context, single, multi, single_result, multi_result, replacement_result = (
        _accepted_inputs(tmp_path)
    )
    context.manifest.rounds[0].channels[0].interpolation = "nearestNeighbor"
    comparison = tmp_path / "multi-process-scientific-comparison.json"
    payload = json.loads(comparison.read_text(encoding="utf-8"))
    payload["rounds"][0]["channels"][0]["interpolation"] = "nearestNeighbor"
    comparison.write_text(json.dumps(payload), encoding="utf-8")

    report = build_multi_process_acceptance_report(
        context=cast(object, context),  # type: ignore[arg-type]
        single_profile=single,
        multi_profile=multi,
        single_result=single_result,
        multi_result=multi_result,
        replacement_result=replacement_result,
        evidence_dir=tmp_path,
    )

    assert report["scientific_parity"]["accepted"] is True  # type: ignore[index]


def test_report_accepts_target_recovery_on_sibling_with_later_replacement_liveness(
    tmp_path: Path,
) -> None:
    context, single, multi, single_result, multi_result, replacement_result = (
        _accepted_inputs(tmp_path)
    )
    evidence = json.loads(replacement_result.execution_evidence_json)
    placements = evidence["fine_graph"]["placements"]
    recovered = placements[1]
    recovered.update(
        {
            "worker_address": "tcp://node-0:1002",
            "hostname": "node-0",
            "pid": 102,
            "slurm_job_id": "12345",
        }
    )
    later = _placement(
        task="round-004:chunk:0000",
        address="tcp://node-0:2001",
        pid=999,
        started="2026-07-30T10:00:24Z",
        ended="2026-07-30T10:00:34Z",
    )
    later["coordinator_ready_at"] = "2026-07-30T10:00:24Z"
    later["artifact_paths"] = ["/archive/later.tif"]
    placements.append(later)
    replacement_result = replace(
        replacement_result,
        execution_evidence_json=json.dumps(evidence),
    )
    (tmp_path / "multi-process-replacement-slurm-execution.json").write_text(
        json.dumps(evidence),
        encoding="utf-8",
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

    replacement = cast(dict[str, object], report["replacement"])
    assert replacement["accepted"] is True
    assert (
        cast(dict[str, object], replacement["recovered_target_process"])[
            "worker_address"
        ]
        == "tcp://node-0:1002"
    )
    assert (
        cast(dict[str, object], replacement["replacement_process"])["worker_address"]
        == "tcp://node-0:2001"
    )


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
        "incomplete_comparison",
        "under_realized_capacity",
        "wrong_marker_task",
        "missing_healthy_sibling_liveness",
        "unowned_replacement_lineage",
        "symlinked_evidence",
        "duplicate_group_pid",
        "unexpected_scientific_channel",
        "wrong_scientific_channel_source",
        "wrong_scientific_channel_interpolation",
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
    elif mutation == "incomplete_comparison":
        comparison = tmp_path / "multi-process-scientific-comparison.json"
        comparison.write_text(
            json.dumps(
                {
                    "accepted": True,
                    "scientific_fingerprint": "f" * 64,
                    "comparison_policy": "stochastic-scientific-equivalence",
                    "metric_absolute_tolerance": 0.01,
                    "rounds": [{"round": 2}],
                }
            ),
            encoding="utf-8",
        )
    elif mutation == "under_realized_capacity":
        event = evidence["fine_graph"]["scale_events"][0]
        for name in ("connected", "probed", "activated", "activated_probes"):
            event[name] = event[name][:-2]
        multi_result = replace(
            multi_result,
            execution_evidence_json=json.dumps(evidence),
        )
        (tmp_path / "multi-process-multi-slurm-execution.json").write_text(
            json.dumps(evidence),
            encoding="utf-8",
        )
    elif mutation == "wrong_marker_task":
        marker = tmp_path / "multi-process-worker-loss-marker.json"
        payload = json.loads(marker.read_text(encoding="utf-8"))
        payload["task_key"] = "round-999:chunk:0000"
        marker.write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "missing_healthy_sibling_liveness":
        replacement_evidence = json.loads(replacement_result.execution_evidence_json)
        replacement_evidence["fine_graph"]["placements"] = replacement_evidence[
            "fine_graph"
        ]["placements"][:2]
        replacement_result = replace(
            replacement_result,
            execution_evidence_json=json.dumps(replacement_evidence),
        )
        (tmp_path / "multi-process-replacement-slurm-execution.json").write_text(
            json.dumps(replacement_evidence), encoding="utf-8"
        )
    elif mutation == "unowned_replacement_lineage":
        outcome = tmp_path / "multi-process-replacement-outcome.json"
        payload = json.loads(outcome.read_text(encoding="utf-8"))
        payload["owned_job_ids"] = ["12346", "12347"]
        outcome.write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "symlinked_evidence":
        evidence_path = tmp_path / "multi-process-performance.json"
        target = tmp_path / "performance-target.json"
        evidence_path.rename(target)
        evidence_path.symlink_to(target)
    elif mutation == "duplicate_group_pid":
        evidence["fine_graph"]["scale_events"][0]["activated_probes"][1]["pid"] = (
            evidence["fine_graph"]["scale_events"][0]["activated_probes"][0]["pid"]
        )
        multi_result = replace(
            multi_result,
            execution_evidence_json=json.dumps(evidence),
        )
        (tmp_path / "multi-process-multi-slurm-execution.json").write_text(
            json.dumps(evidence),
            encoding="utf-8",
        )
    elif mutation == "unexpected_scientific_channel":
        comparison = tmp_path / "multi-process-scientific-comparison.json"
        payload = json.loads(comparison.read_text(encoding="utf-8"))
        payload["rounds"][0]["channels"].append(payload["rounds"][0]["channels"][0])
        comparison.write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "wrong_scientific_channel_source":
        comparison = tmp_path / "multi-process-scientific-comparison.json"
        payload = json.loads(comparison.read_text(encoding="utf-8"))
        payload["rounds"][0]["channels"][0]["source"] = str(
            (tmp_path / "dataset" / "other-channel.tif").resolve()
        )
        comparison.write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "wrong_scientific_channel_interpolation":
        comparison = tmp_path / "multi-process-scientific-comparison.json"
        payload = json.loads(comparison.read_text(encoding="utf-8"))
        payload["rounds"][0]["channels"][0]["interpolation"] = "nearestNeighbor"
        comparison.write_text(json.dumps(payload), encoding="utf-8")

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
