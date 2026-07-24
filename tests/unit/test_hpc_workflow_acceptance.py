from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import cast

import pytest
from hpc.deterministic_parity import (
    _require_one_thread_results,
    _require_owned_root,
)
from hpc.deterministic_parity import (
    main as deterministic_parity_main,
)
from hpc.fine_workflow_acceptance import (
    build_fine_acceptance_report,
    fine_acceptance_manifest,
    fine_acceptance_profiles,
    validate_fine_acceptance_report,
)
from hpc.test_fine_grained_workflow import _owned_cleanup_sources
from hpc.test_slurm_workflow import _assert_resume
from hpc.workflow_acceptance import (
    HpcWorkflowAcceptanceError,
    _validated_deterministic_parity_evidence,
    bounded_acceptance_manifest,
    bounded_slurm_profile,
    coordination_timeout_seconds,
    deterministic_acceptance_manifest,
    deterministic_sequential_profile,
    deterministic_slurm_profile,
    execute_cancellation_acceptance,
    execute_deterministic_parity_subprocess,
    execute_owned_slurm_workflow,
    metric_comparison_evidence,
    numeric_comparison_evidence,
    parallel_placement_evidence,
    prepare_task5_output_roots,
    run_cancellation_task,
    run_failure_task,
    run_fine_cancellation_task,
    run_fine_worker_loss_task,
    run_injected_fine_failure_task,
    run_profiled_fine_task,
    run_seeded_workflow_task,
    sequential_reference_profile,
    task5_output_roots,
    validate_injected_failure_record,
    validate_task5_output_ownership,
    workflow_module_by_value,
    write_deterministic_inputs,
)

from cyexm.config.models import (
    ChannelSpec,
    DatasetManifest,
    ExecutionPaths,
    ExecutionProfile,
    NativeThreadSpec,
    NonlinearRegistrationStageSpec,
    ParallelismSpec,
    ReferenceChannelSpec,
    ReferenceSpec,
    RegistrationSpec,
    RegistrationStageSpec,
    RoundSpec,
    SlurmSpec,
)
from cyexm.registration.artifacts import FileIdentity
from cyexm.workflow.models import (
    ReferenceChannelTask,
    ReferenceTask,
    RoundResult,
)

pytestmark = pytest.mark.unit


def _manifest(*rounds: RoundSpec) -> DatasetManifest:
    return DatasetManifest(
        schema_version=1,
        dataset_id="bounded",
        voxel_spacing_um=(0.2, 0.2, 0.5),
        reference=ReferenceSpec(
            round=1,
            registration_channel=PurePosixPath("reference.tif"),
            channels=(ReferenceChannelSpec(PurePosixPath("reference.tif")),),
        ),
        registration=RegistrationSpec(
            crop=False,
            linear=RegistrationStageSpec(type="TRSAA", accuracy="low"),
            nonlinear=NonlinearRegistrationStageSpec(
                type="SyNOnly", accuracy="high", metric="mattes", radius=None
            ),
        ),
        rounds=rounds,
    )


def _round(number: int, strategy: str) -> RoundSpec:
    chunk_size = (8, 16, 16) if strategy == "serial_chunked" else None
    overlap = 0.25 if strategy == "serial_chunked" else None
    return RoundSpec(
        round=number,
        strategy=strategy,  # type: ignore[arg-type]
        registration_channel=PurePosixPath(f"round-{number}.tif"),
        channels=(ChannelSpec(PurePosixPath(f"round-{number}.tif"), "linear"),),
        chunk_size_zyx=chunk_size,
        overlap_fraction=overlap,
    )


def _profile(tmp_path: Path, *, workers: int = 4) -> ExecutionProfile:
    return ExecutionProfile(
        schema_version=1,
        executor="slurm",
        paths=ExecutionPaths(
            dataset_root=tmp_path / "dataset",
            output_root=tmp_path / "general-results",
            worker_local_directory=tmp_path / "scratch",
        ),
        native_threads=NativeThreadSpec(itk=1, blas=1),
        local_dask=None,
        slurm=SlurmSpec(
            queue="approved",
            workers=workers,
            cores_per_worker=8,
            processes_per_worker=1,
            memory="64GiB",
            walltime="00:30:00",
            interface="ib0",
            scheduler_interface="ib0",
            startup_timeout="60s",
            task_retries=3,
            job_name="cyexm-task5",
            job_extra_directives=("--exclusive",),
        ),
        source_path=tmp_path / "execution.yaml",
    )


def _fine_acceptance_report(root: Path) -> dict[str, object]:
    executor_cleanup = {
        "client": {"attempted": True, "succeeded": True},
        "cluster": {"attempted": True, "succeeded": True},
        "configuration": {"attempted": True, "succeeded": True},
        "order": ["client", "cluster", "configuration"],
    }
    evidence_path = root / "fine-execution.json"
    tier1_path = root / "tier1-equivalence.json"
    tier1_path.write_text(
        json.dumps(_scientific_equivalence_payload(), sort_keys=True),
        encoding="utf-8",
    )
    tier2_path = root / "deterministic-executor-parity.json"
    tier2_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "completed",
                "seed": "20260722",
                "native_threads": {
                    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": 1,
                    "OMP_NUM_THREADS": 1,
                    "OPENBLAS_NUM_THREADS": 1,
                    "MKL_NUM_THREADS": 1,
                    "NUMEXPR_NUM_THREADS": 1,
                    "VECLIB_MAXIMUM_THREADS": 1,
                },
                "inputs": {},
                "parallel": {"status": "completed"},
                "sequential": {"status": "completed"},
                "strict_parity": _strict_parity_payload(),
                "strict_internal_parity": _strict_internal_parity_payload(),
                "final_queue": {
                    "jobs": [],
                    "coarse": {"jobs": []},
                    "fine": {"jobs": []},
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    replacement_path = root / "fine-worker-replacement-slurm-execution.json"
    replacement_path.write_text(
        json.dumps(
            {
                "observed": {
                    "worker_probes": [
                        {
                            "worker_address": "tcp://node-a:1",
                            "hostname": "node-a",
                            "pid": 111,
                            "slurm_job_id": "101",
                        },
                        {
                            "worker_address": "tcp://node-c:1",
                            "hostname": "node-c",
                            "pid": 333,
                            "slurm_job_id": "103",
                        },
                    ]
                },
                "fine_graph": {
                    "scale_events": [
                        {
                            "departed": [],
                            "activated": ["tcp://node-a:1"],
                            "activated_probes": [
                                {
                                    "worker_address": "tcp://node-a:1",
                                    "hostname": "node-a",
                                    "pid": 111,
                                    "slurm_job_id": "101",
                                }
                            ],
                        },
                        {
                            "departed": ["tcp://node-a:1"],
                            "activated": ["tcp://node-c:1"],
                            "activated_probes": [
                                {
                                    "worker_address": "tcp://node-c:1",
                                    "hostname": "node-c",
                                    "pid": 333,
                                    "slurm_job_id": "103",
                                }
                            ],
                        },
                    ],
                    "attempt_counts": [
                        {"task_key": "round-002:chunk:0000", "attempts": 2},
                        {"task_key": "round-003:chunk:0000", "attempts": 1},
                    ],
                    "placements": [
                        {
                            "task_key": "round-002:chunk:0000",
                            "attempt": 1,
                            "status": "failed",
                            "worker_address": None,
                            "hostname": None,
                            "pid": None,
                            "slurm_job_id": None,
                            "placement_accepted": False,
                            "coordinator_ready_at": "2026-07-22T10:00:01Z",
                        },
                        {
                            "task_key": "round-002:chunk:0000",
                            "attempt": 2,
                            "status": "completed",
                            "worker_address": "tcp://node-c:1",
                            "hostname": "node-c",
                            "pid": 333,
                            "slurm_job_id": "103",
                            "placement_accepted": True,
                            "coordinator_ready_at": "2026-07-22T10:00:02Z",
                        },
                        {
                            "task_key": "round-003:chunk:0000",
                            "attempt": 1,
                            "status": "completed",
                            "worker_address": "tcp://node-c:1",
                            "hostname": "node-c",
                            "pid": 333,
                            "slurm_job_id": "103",
                            "placement_accepted": True,
                            "coordinator_ready_at": "2026-07-22T10:00:03Z",
                        },
                    ],
                },
                "cleanup": executor_cleanup,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    replacement_queue_path = root / "fine-worker-replacement-queue-after.json"
    replacement_queue_path.write_text(
        json.dumps({"jobs": []}, sort_keys=True),
        encoding="utf-8",
    )
    replacement_outcome_path = root / "fine-worker-replacement-outcome.json"
    replacement_outcome_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "operation": "fine-worker-replacement",
                "status": "completed",
                "owned_job_ids": ["100", "101", "103"],
                "cleanup_outcome": "empty",
                "workflow_result": {"status": "completed"},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    loss_marker_path = root / "worker-loss.json"
    loss_marker_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_key": "round-002:chunk:0000",
                "worker_address": "tcp://node-a:1",
                "hostname": "node-a",
                "pid": 111,
                "slurm_job_id": "101",
                "status": "worker-loss-injected",
                "injected_at": "2026-07-22T10:00:00Z",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    committed_peer_path = root / "committed-peer.json"
    committed_peer_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_key": "round-003:chunk:0000",
                "status": "completed",
                "artifacts": [
                    {
                        "identity": {
                            "path": "/owned/peer",
                            "st_size": 1,
                            "st_mtime_ns": 1,
                        },
                        "sha256": "a" * 64,
                    }
                ],
                "committed_at": "2026-07-22T10:00:00Z",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    cancellation_ready_path = root / "cancellation-ready.json"
    cancellation_ready_path.write_text(
        json.dumps(
            {
                "blocked_task_key": "round-002:chunk:0000",
                "committed_peer_key": "round-003:chunk:0000",
                "status": "waiting-for-coordinator-interruption",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    cancellation_trigger_path = root / "fine-cancellation-trigger.json"
    cancellation_trigger_path.write_text(
        json.dumps(
            {
                "blocked_task_key": "round-002:chunk:0000",
                "committed_peer_key": "round-003:chunk:0000",
                "signal": "SIGINT",
                "status": "interrupted",
                "timeout_seconds": 1800.0,
                "exit_code": 130,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    cancellation_queue_path = root / "fine-cancellation-queue-after.json"
    cancellation_queue_path.write_text(
        json.dumps({"jobs": []}, sort_keys=True),
        encoding="utf-8",
    )
    cancellation_execution_path = root / "fine-cancellation-slurm-execution.json"
    cancellation_execution_path.write_text(
        json.dumps(
            {
                "fine_graph": {
                    "cancelled": True,
                    "placements": [
                        {"slurm_job_id": "101"},
                        {"slurm_job_id": "102"},
                    ],
                },
                "cleanup": executor_cleanup,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    cancellation_outcome_path = root / "fine-cancellation-outcome.json"
    cancellation_outcome_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "operation": "fine-cancellation",
                "status": "cancelled",
                "owned_job_ids": ["100", "101", "102"],
                "cleanup_outcome": "empty",
                "workflow_result": None,
                "error": {
                    "exception_type": "KeyboardInterrupt",
                    "message": "Task 5 coordinator interruption",
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    prepared_path = root / "fine" / "schema-v1" / "sample" / "prepared" / "fixed.npy"
    prepared_path.parent.mkdir(parents=True)
    prepared_path.write_bytes(b"prepared")
    input_path = root / "inputs" / "fixed.tif"
    input_path.parent.mkdir()
    input_path.write_bytes(b"fixed")
    coarse_profile_path = root / "coarse.yaml"
    fine_profile_path = root / "fine.yaml"
    coarse_profile_path.write_bytes(b"parallelism: coarse\n")
    fine_profile_path.write_bytes(b"parallelism: fine_grained\n")
    task_times = [
        {
            "task_key": "round-002:chunk:0000",
            "kind": "chunk",
            "round": 2,
            "job_id": "101",
            "hostname": "node-a",
            "worker": "tcp://node-a:1",
            "coordinator_submitted_at": "2026-07-22T10:00:00Z",
            "coordinator_ready_at": "2026-07-22T10:00:12Z",
            "duration_seconds": 10.0,
        },
        {
            "task_key": "round-003:chunk:0000",
            "kind": "chunk",
            "round": 3,
            "job_id": "102",
            "hostname": "node-b",
            "worker": "tcp://node-b:1",
            "coordinator_submitted_at": "2026-07-22T10:00:01Z",
            "coordinator_ready_at": "2026-07-22T10:00:13Z",
            "duration_seconds": 10.0,
        },
        {
            "task_key": "round-002:channel:0000",
            "kind": "channel",
            "round": 2,
            "job_id": "101",
            "hostname": "node-a",
            "worker": "tcp://node-a:1",
            "coordinator_submitted_at": "2026-07-22T10:00:12Z",
            "coordinator_ready_at": "2026-07-22T10:00:20Z",
            "duration_seconds": 6.0,
        },
        {
            "task_key": "round-003:channel:0000",
            "kind": "channel",
            "round": 3,
            "job_id": "102",
            "hostname": "node-b",
            "worker": "tcp://node-b:1",
            "coordinator_submitted_at": "2026-07-22T10:00:13Z",
            "coordinator_ready_at": "2026-07-22T10:00:21Z",
            "duration_seconds": 6.0,
        },
    ]
    raw_placements = [
        {
            "task_key": f"round-{round_number:03d}:preparation:main",
            "attempt": 1,
            "status": "completed",
            "placement_accepted": True,
            "coordinator_submitted_at": f"2026-07-22T09:59:5{round_number}Z",
            "coordinator_ready_at": (
                "2026-07-22T09:59:59Z" if round_number == 2 else "2026-07-22T10:00:00Z"
            ),
            "worker_address": f"tcp://node-{'a' if round_number == 2 else 'b'}:1",
            "hostname": f"node-{'a' if round_number == 2 else 'b'}",
            "pid": 100 + round_number,
            "slurm_job_id": str(99 + round_number),
            "duration_seconds": 1.0,
        }
        for round_number in (2, 3)
    ]
    raw_placements.extend(
        {
            "task_key": item["task_key"],
            "attempt": 1,
            "status": "completed",
            "placement_accepted": True,
            "coordinator_submitted_at": item["coordinator_submitted_at"],
            "coordinator_ready_at": item["coordinator_ready_at"],
            "worker_address": item["worker"],
            "hostname": item["hostname"],
            "pid": 111 if item["hostname"] == "node-a" else 222,
            "slurm_job_id": item["job_id"],
            "duration_seconds": item["duration_seconds"],
        }
        for item in task_times
    )
    raw_placements.extend(
        [
            {
                "task_key": f"round-{round_number:03d}:assembly:main",
                "attempt": 1,
                "status": "completed",
                "placement_accepted": True,
                "coordinator_submitted_at": "2026-07-22T10:01:00Z",
                "coordinator_ready_at": "2026-07-22T10:01:20Z",
                "worker_address": f"tcp://node-{'a' if round_number == 2 else 'b'}:1",
                "hostname": f"node-{'a' if round_number == 2 else 'b'}",
                "pid": 100 + round_number,
                "slurm_job_id": str(99 + round_number),
                "duration_seconds": 20.0,
            }
            for round_number in (2, 3)
        ]
    )
    evidence_path.write_text(
        json.dumps(
            {
                "fine_graph": {
                    "worker_cap": 4,
                    "scale_events": [
                        {
                            "requested": 1,
                            "activated": ["tcp://node-a:1"],
                            "departed": [],
                        },
                        {
                            "requested": 2,
                            "activated": ["tcp://node-b:1"],
                            "departed": [],
                        },
                        {
                            "requested": 4,
                            "activated": ["tcp://node-c:1", "tcp://node-d:1"],
                            "departed": [],
                        },
                    ],
                    "placements": raw_placements,
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    coarse_timing_paths = []
    for round_number, calls in (
        (
            2,
            [
                {
                    "duration_seconds": 60.0,
                    "monotonic_started_seconds": 10.0,
                    "monotonic_ended_seconds": 70.0,
                },
                {
                    "duration_seconds": 40.0,
                    "monotonic_started_seconds": 70.0,
                    "monotonic_ended_seconds": 110.0,
                },
            ],
        ),
        (
            3,
            [
                {
                    "duration_seconds": 70.0,
                    "monotonic_started_seconds": 20.0,
                    "monotonic_ended_seconds": 90.0,
                }
            ],
        ),
    ):
        timing_path = root / f"round-{round_number:03d}-coarse-main.json"
        timing_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "task_key": f"round-{round_number:03d}:coarse:main",
                    "status": "completed",
                    "nonlinear_calls": calls,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        coarse_timing_paths.append(timing_path)
    identity = FileIdentity.from_path(input_path).to_dict()
    artifact = {
        "relative_path": "schema-v1/sample/rounds/round-002/channels/result.tif",
        "sha256": "b" * 64,
    }

    def failure_sources(kind: str) -> dict[str, object]:
        target = f"round-002:{kind}:0000"
        peer = f"round-003:{kind}:0000"
        failed_path = root / f"{kind}-failure-execution.json"
        resume_path = root / f"{kind}-resume-execution.json"
        marker_path = root / f"{kind}-failure-marker.json"
        settled_path = root / f"{kind}-settled-peer.json"
        failed_path.write_text(
            json.dumps(
                {
                    "fine_graph": {
                        "placements": [
                            {
                                "task_key": target,
                                "attempt": 1,
                                "status": "failed",
                            }
                        ],
                        "task_results": [
                            {
                                "key": peer,
                                "status": "completed",
                                "artifact_paths": ["/owned/peer"],
                            }
                        ],
                    }
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        resume_path.write_text(
            json.dumps(
                {
                    "fine_graph": {
                        "task_results": [
                            {"key": target, "status": "completed"},
                            {"key": peer, "status": "reused"},
                        ]
                    }
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        marker_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "task_key": target,
                    "settled_peer_key": peer,
                    "status": "injected",
                    "injected_at": "2026-07-22T10:00:00Z",
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        settled_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "task_key": peer,
                    "status": "completed",
                    "artifacts": [{"path": "/owned/peer", "sha256": "a" * 64}],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return {
            "failure_evidence_file": {
                "relative_path": failed_path.name,
                "sha256": sha256(failed_path.read_bytes()).hexdigest(),
            },
            "resume_evidence_file": {
                "relative_path": resume_path.name,
                "sha256": sha256(resume_path.read_bytes()).hexdigest(),
            },
            "failure_marker_file": {
                "relative_path": marker_path.name,
                "sha256": sha256(marker_path.read_bytes()).hexdigest(),
            },
            "settled_peer_file": {
                "relative_path": settled_path.name,
                "sha256": sha256(settled_path.read_bytes()).hexdigest(),
            },
        }

    resume = {
        **failure_sources("chunk"),
        "injected_task_key": "round-002:chunk:0000",
        "failure_status": "failed",
        "failed_attempts": 1,
        "settled_peer_key": "round-003:chunk:0000",
        "preserved_artifacts": [artifact],
        "resume_reused_keys": ["round-003:chunk:0000"],
        "resume_recomputed_keys": ["round-002:chunk:0000"],
        "ordinary_artifact_store_reuse": True,
    }
    return {
        "schema_version": 1,
        "status": "completed",
        "identities": {
            "git_commit": "1" * 40,
            "package_versions": {"cy-exm": "0.1.0", "distributed": "2026.1.0"},
            "scientific_fingerprint": "2" * 64,
            "coarse_execution_fingerprint": "3" * 64,
            "fine_execution_fingerprint": "4" * 64,
            "coarse_profile": FileIdentity.from_path(coarse_profile_path).to_dict(),
            "fine_profile": FileIdentity.from_path(fine_profile_path).to_dict(),
            "coarse_profile_sha256": sha256(
                coarse_profile_path.read_bytes()
            ).hexdigest(),
            "fine_profile_sha256": sha256(fine_profile_path.read_bytes()).hexdigest(),
            "inputs": [identity],
            "input_checksums": [
                {
                    "path": identity["path"],
                    "sha256": sha256(input_path.read_bytes()).hexdigest(),
                }
            ],
        },
        "scientific": {
            "coarse_fingerprint": "2" * 64,
            "fine_fingerprint": "2" * 64,
            "public_artifacts_coarse": [artifact],
            "public_artifacts_fine": [artifact],
            "tier1": {
                "evidence_file": {
                    "relative_path": tier1_path.name,
                    "sha256": sha256(tier1_path.read_bytes()).hexdigest(),
                },
                "metric_absolute_tolerance": 0.01,
                "maximum_metric_absolute_difference": 0.009,
                "structure_equal": True,
            },
            "tier2": {
                "evidence_file": {
                    "relative_path": tier2_path.name,
                    "sha256": sha256(tier2_path.read_bytes()).hexdigest(),
                },
                "seed": "20260722",
                "native_threads": 1,
                "affine_exact": True,
                "local_warp_exact": True,
                "full_warp_exact": True,
                "diagnostic_exact": True,
                "metrics_exact": True,
                "channels": [
                    {
                        "relative_path": (
                            "round-002/channel-0000/round-002-channel.tif"
                        ),
                        "exact": True,
                    }
                ],
            },
        },
        "fixed_grid": {
            "shape_zyx": [16, 32, 48],
            "origin_xyz": [1.0, 2.0, 3.0],
            "spacing_xyz": [0.2, 0.2, 0.5],
            "direction": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "prepared_paths": [str(prepared_path)],
            "owned_roots": [str(root / "fine")],
        },
        "placement": {
            "evidence_file": {
                "relative_path": evidence_path.name,
                "sha256": sha256(evidence_path.read_bytes()).hexdigest(),
            },
            "worker_cap": 4,
            "maximum_connected_workers": 4,
            "scale_events": [1, 2, 4],
            "fair_ready_rounds": [2, 3],
            "fair_dispatch_order": [
                "round-002:chunk:0000",
                "round-003:chunk:0000",
            ],
            "tasks": task_times,
        },
        "durability": {
            "chunk_failure_resume": resume,
            "channel_failure_resume": {
                **resume,
                **failure_sources("channel"),
                "injected_task_key": "round-002:channel:0000",
                "settled_peer_key": "round-003:channel:0000",
                "resume_reused_keys": ["round-003:channel:0000"],
                "resume_recomputed_keys": ["round-002:channel:0000"],
            },
            "cancellation_resume": {
                "status": "cancelled",
                "structured_cancellation": True,
                "blocked_task_key": "round-002:chunk:0000",
                "committed_peer_key": "round-003:chunk:0000",
                "committed_peer_file": {
                    "relative_path": committed_peer_path.name,
                    "sha256": sha256(committed_peer_path.read_bytes()).hexdigest(),
                },
                "ready_marker_file": {
                    "relative_path": cancellation_ready_path.name,
                    "sha256": sha256(cancellation_ready_path.read_bytes()).hexdigest(),
                },
                "trigger_file": {
                    "relative_path": cancellation_trigger_path.name,
                    "sha256": sha256(
                        cancellation_trigger_path.read_bytes()
                    ).hexdigest(),
                },
                "execution_evidence_file": {
                    "relative_path": cancellation_execution_path.name,
                    "sha256": sha256(
                        cancellation_execution_path.read_bytes()
                    ).hexdigest(),
                },
                "outcome_file": {
                    "relative_path": cancellation_outcome_path.name,
                    "sha256": sha256(
                        cancellation_outcome_path.read_bytes()
                    ).hexdigest(),
                },
                "queue_after_file": {
                    "relative_path": cancellation_queue_path.name,
                    "sha256": sha256(cancellation_queue_path.read_bytes()).hexdigest(),
                },
                "preserved_artifacts": [artifact],
                "resume_reused_keys": ["round-003:chunk:0000"],
                "resume_recomputed_keys": ["round-002:chunk:0000"],
                "owned_job_ids": ["100", "101", "102"],
                "cleanup_outcome": "empty",
            },
            "worker_replacement": {
                "evidence_file": {
                    "relative_path": replacement_path.name,
                    "sha256": sha256(replacement_path.read_bytes()).hexdigest(),
                },
                "queue_after_file": {
                    "relative_path": replacement_queue_path.name,
                    "sha256": sha256(replacement_queue_path.read_bytes()).hexdigest(),
                },
                "outcome_file": {
                    "relative_path": replacement_outcome_path.name,
                    "sha256": sha256(replacement_outcome_path.read_bytes()).hexdigest(),
                },
                "loss_marker_file": {
                    "relative_path": loss_marker_path.name,
                    "sha256": sha256(loss_marker_path.read_bytes()).hexdigest(),
                },
                "loss_marker": {
                    "task_key": "round-002:chunk:0000",
                    "worker_address": "tcp://node-a:1",
                    "hostname": "node-a",
                    "pid": 111,
                    "slurm_job_id": "101",
                },
                "failed_attempt": {
                    "task_key": "round-002:chunk:0000",
                    "attempt": 1,
                    "status": "failed",
                },
                "recovery_mode": "coordinator-retry",
                "departed_worker": {
                    "worker_address": "tcp://node-a:1",
                    "hostname": "node-a",
                    "pid": 111,
                    "slurm_job_id": "101",
                },
                "replacement_worker": {
                    "worker_address": "tcp://node-c:1",
                    "hostname": "node-c",
                    "pid": 333,
                    "slurm_job_id": "103",
                },
                "replacement_liveness_path": "retry",
                "retry": {
                    "task_key": "round-002:chunk:0000",
                    "attempt": 2,
                    "status": "completed",
                    "worker_address": "tcp://node-c:1",
                    "hostname": "node-c",
                    "pid": 333,
                    "slurm_job_id": "103",
                },
                "liveness": None,
                "owned_job_ids": ["100", "101", "103"],
                "cleanup_outcome": "empty",
            },
        },
        "performance": {
            "clock_policy": (
                "coordinator-fine-upper-bound-vs-worker-monotonic-coarse-lower-bound"
            ),
            "coarse_nonlinear_wall_seconds": 100.0,
            "coarse_timing_policy": "per-round-monotonic-kernel-span",
            "coarse_timing_files": [
                {
                    "relative_path": path.name,
                    "sha256": sha256(path.read_bytes()).hexdigest(),
                }
                for path in coarse_timing_paths
            ],
            "fine_nonlinear_wall_seconds": 80.0,
            "end_to_end_wall_seconds": 140.0,
            "task_seconds": 32.0,
            "node_hours": 0.02,
            "peak_memory_bytes": 1_000_000,
            "startup_overhead_seconds": 4.0,
            "worker_utilization": 0.75,
            "chunk_duration_seconds": {
                "count": 2,
                "minimum": 10.0,
                "median": 10.0,
                "maximum": 10.0,
            },
            "channel_duration_seconds": {
                "count": 2,
                "minimum": 6.0,
                "median": 6.0,
                "maximum": 6.0,
            },
        },
        "owned_root": str(root),
        "evidence_files": [
            *[
                {
                    "relative_path": path.name,
                    "sha256": sha256(path.read_bytes()).hexdigest(),
                }
                for path in sorted(
                    (
                        tier2_path,
                        evidence_path,
                        replacement_path,
                        replacement_outcome_path,
                        replacement_queue_path,
                        loss_marker_path,
                        committed_peer_path,
                        cancellation_ready_path,
                        cancellation_trigger_path,
                        cancellation_execution_path,
                        cancellation_outcome_path,
                        cancellation_queue_path,
                        tier1_path,
                        *coarse_timing_paths,
                        *(
                            root / f"{kind}-{suffix}.json"
                            for kind in ("chunk", "channel")
                            for suffix in (
                                "failure-execution",
                                "resume-execution",
                                "failure-marker",
                                "settled-peer",
                            )
                        ),
                    )
                )
            ],
        ],
    }


def test_fine_acceptance_profiles_preserve_science_and_select_only_admission_mode(
    tmp_path: Path,
) -> None:
    approved = _profile(tmp_path)
    coarse_approved = replace(
        approved,
        paths=replace(approved.paths, output_root=tmp_path / "coarse"),
        parallelism=ParallelismSpec(mode="coarse"),
        source_path=tmp_path / "coarse.yaml",
    )
    fine_approved = replace(
        approved,
        paths=replace(
            approved.paths,
            output_root=tmp_path / "fine",
            worker_local_directory=tmp_path / "fine-scratch",
        ),
        slurm=replace(cast(SlurmSpec, approved.slurm), job_name="cyexm-task5-fine"),
        parallelism=ParallelismSpec(mode="fine_grained"),
        source_path=tmp_path / "fine.yaml",
    )

    coarse, fine = fine_acceptance_profiles(coarse_approved, fine_approved)

    assert coarse.parallelism.mode == "coarse"
    assert fine.parallelism.mode == "fine_grained"
    assert coarse.paths.output_root == tmp_path / "coarse"
    assert fine.paths.output_root == tmp_path / "fine"
    assert coarse is coarse_approved
    assert fine is fine_approved


@pytest.mark.parametrize(
    "mutation",
    (
        lambda profile: replace(
            profile, native_threads=NativeThreadSpec(itk=2, blas=1)
        ),
        lambda profile: replace(
            profile,
            slurm=replace(cast(SlurmSpec, profile.slurm), workers=5),
        ),
        lambda profile: replace(
            profile,
            slurm=replace(cast(SlurmSpec, profile.slurm), cores_per_worker=16),
        ),
        lambda profile: replace(
            profile,
            slurm=replace(cast(SlurmSpec, profile.slurm), memory="128GiB"),
        ),
        lambda profile: replace(
            profile,
            slurm=replace(cast(SlurmSpec, profile.slurm), interface="eth0"),
        ),
        lambda profile: replace(
            profile,
            slurm=replace(cast(SlurmSpec, profile.slurm), task_retries=2),
        ),
    ),
)
def test_fine_acceptance_rejects_unfair_profile_differences(
    tmp_path: Path,
    mutation: object,
) -> None:
    approved = _profile(tmp_path)
    coarse = replace(
        approved,
        paths=replace(approved.paths, output_root=tmp_path / "coarse"),
        parallelism=ParallelismSpec(mode="coarse"),
        source_path=tmp_path / "coarse.yaml",
    )
    fine = replace(
        approved,
        paths=replace(
            approved.paths,
            output_root=tmp_path / "fine",
            worker_local_directory=tmp_path / "fine-scratch",
        ),
        slurm=replace(cast(SlurmSpec, approved.slurm), job_name="cyexm-task5-fine"),
        parallelism=ParallelismSpec(mode="fine_grained"),
        source_path=tmp_path / "fine.yaml",
    )
    assert callable(mutation)

    with pytest.raises(HpcWorkflowAcceptanceError, match="otherwise identical"):
        fine_acceptance_profiles(coarse, cast(ExecutionProfile, mutation(fine)))


@pytest.mark.parametrize("shared", ("scratch", "job_name"))
def test_fine_acceptance_requires_distinct_operational_ownership(
    tmp_path: Path,
    shared: str,
) -> None:
    approved = _profile(tmp_path)
    coarse = replace(
        approved,
        paths=replace(approved.paths, output_root=tmp_path / "coarse"),
        parallelism=ParallelismSpec(mode="coarse"),
        source_path=tmp_path / "coarse.yaml",
    )
    fine = replace(
        approved,
        paths=replace(
            approved.paths,
            output_root=tmp_path / "fine",
            worker_local_directory=(
                approved.paths.worker_local_directory
                if shared == "scratch"
                else tmp_path / "fine-scratch"
            ),
        ),
        slurm=replace(
            cast(SlurmSpec, approved.slurm),
            job_name=(
                cast(SlurmSpec, approved.slurm).job_name
                if shared == "job_name"
                else "cyexm-task5-fine"
            ),
        ),
        parallelism=ParallelismSpec(mode="fine_grained"),
        source_path=tmp_path / "fine.yaml",
    )

    with pytest.raises(HpcWorkflowAcceptanceError, match="distinct"):
        fine_acceptance_profiles(coarse, fine)


@pytest.mark.parametrize("role", ("output", "scratch"))
def test_fine_acceptance_requires_canonically_disjoint_owned_roots(
    tmp_path: Path,
    role: str,
) -> None:
    approved = _profile(tmp_path)
    coarse = replace(
        approved,
        paths=replace(
            approved.paths,
            output_root=tmp_path / "coarse",
            worker_local_directory=tmp_path / "coarse-scratch",
        ),
        parallelism=ParallelismSpec(mode="coarse"),
        source_path=tmp_path / "coarse.yaml",
    )
    fine = replace(
        approved,
        paths=replace(
            approved.paths,
            output_root=(
                coarse.paths.output_root / "fine"
                if role == "output"
                else tmp_path / "fine"
            ),
            worker_local_directory=(
                cast(Path, coarse.paths.worker_local_directory) / "fine"
                if role == "scratch"
                else tmp_path / "fine-scratch"
            ),
        ),
        slurm=replace(cast(SlurmSpec, approved.slurm), job_name="cyexm-task5-fine"),
        parallelism=ParallelismSpec(mode="fine_grained"),
        source_path=tmp_path / "fine.yaml",
    )

    with pytest.raises(HpcWorkflowAcceptanceError, match="disjoint"):
        fine_acceptance_profiles(coarse, fine)


def test_fine_acceptance_manifest_requires_two_rounds_with_dependent_channels() -> None:
    insufficient = _manifest(
        RoundSpec(
            round=2,
            strategy="serial_chunked",
            registration_channel=PurePosixPath("round-2.tif"),
            channels=(ChannelSpec(PurePosixPath("round-2.tif"), "linear"),),
            chunk_size_zyx=(8, 16, 16),
            overlap_fraction=0.25,
        ),
        _round(3, "serial_chunked"),
    )

    with pytest.raises(
        HpcWorkflowAcceptanceError, match=r"dependent channel.*before submission"
    ):
        fine_acceptance_manifest(insufficient)


def test_fine_acceptance_manifest_preflights_exact_cross_round_task_keys() -> None:
    manifest = _manifest(
        RoundSpec(
            round=4,
            strategy="serial_chunked",
            registration_channel=PurePosixPath("round-4.tif"),
            channels=(
                ChannelSpec(PurePosixPath("round-4.tif"), "linear"),
                ChannelSpec(PurePosixPath("round-4-dependent.tif"), "linear"),
            ),
            chunk_size_zyx=(8, 16, 16),
            overlap_fraction=0.25,
        ),
        RoundSpec(
            round=7,
            strategy="serial_chunked",
            registration_channel=PurePosixPath("round-7.tif"),
            channels=(
                ChannelSpec(PurePosixPath("round-7-dependent.tif"), "linear"),
                ChannelSpec(PurePosixPath("round-7.tif"), "linear"),
            ),
            chunk_size_zyx=(8, 16, 16),
            overlap_fraction=0.25,
        ),
    )

    bounded, targets = fine_acceptance_manifest(manifest)

    assert tuple(item.round for item in bounded.rounds) == (4, 7)
    assert targets == {
        "chunk_target": "round-004:chunk:0000",
        "chunk_peer": "round-007:chunk:0000",
        "channel_target": "round-004:channel:0001",
        "channel_peer": "round-007:channel:0000",
        "replacement_target": "round-004:chunk:0000",
        "liveness_peer": "round-007:chunk:0000",
    }


def test_fine_acceptance_report_validates_every_portable_gate(tmp_path: Path) -> None:
    report = _fine_acceptance_report(tmp_path)

    validated = validate_fine_acceptance_report(report, evidence_root=tmp_path)

    assert validated == report
    assert validated is not report


def test_fine_acceptance_builder_hashes_only_owned_evidence(tmp_path: Path) -> None:
    payload = _fine_acceptance_report(tmp_path)

    built = build_fine_acceptance_report(
        identities=cast(dict[str, object], payload["identities"]),
        scientific=cast(dict[str, object], payload["scientific"]),
        fixed_grid=cast(dict[str, object], payload["fixed_grid"]),
        placement=cast(dict[str, object], payload["placement"]),
        durability=cast(dict[str, object], payload["durability"]),
        performance=cast(dict[str, object], payload["performance"]),
        evidence_root=tmp_path,
        evidence_paths=tuple(
            tmp_path / name
            for name in (
                "deterministic-executor-parity.json",
                "fine-execution.json",
                "fine-worker-replacement-slurm-execution.json",
                "fine-worker-replacement-outcome.json",
                "fine-worker-replacement-queue-after.json",
                "worker-loss.json",
                "committed-peer.json",
                "cancellation-ready.json",
                "fine-cancellation-trigger.json",
                "fine-cancellation-slurm-execution.json",
                "fine-cancellation-outcome.json",
                "fine-cancellation-queue-after.json",
                "tier1-equivalence.json",
                "round-002-coarse-main.json",
                "round-003-coarse-main.json",
                "chunk-failure-execution.json",
                "chunk-resume-execution.json",
                "chunk-failure-marker.json",
                "chunk-settled-peer.json",
                "channel-failure-execution.json",
                "channel-resume-execution.json",
                "channel-failure-marker.json",
                "channel-settled-peer.json",
            )
        ),
    )

    assert built == payload
    outside = tmp_path.parent / "outside-evidence.json"
    outside.write_text("{}\n", encoding="utf-8")
    with pytest.raises(HpcWorkflowAcceptanceError, match="escapes"):
        build_fine_acceptance_report(
            identities=cast(dict[str, object], payload["identities"]),
            scientific=cast(dict[str, object], payload["scientific"]),
            fixed_grid=cast(dict[str, object], payload["fixed_grid"]),
            placement=cast(dict[str, object], payload["placement"]),
            durability=cast(dict[str, object], payload["durability"]),
            performance=cast(dict[str, object], payload["performance"]),
            evidence_root=tmp_path,
            evidence_paths=(outside,),
        )


def test_fine_failure_and_cancellation_runners_bind_exact_target_and_peer_keys(
    tmp_path: Path,
) -> None:
    target = SimpleNamespace(
        round=2,
        key=SimpleNamespace(stage="chunk", encoded="round-002:chunk:0000"),
    )
    sibling = SimpleNamespace(
        round=3,
        key=SimpleNamespace(stage="chunk", encoded="round-003:chunk:0000"),
    )
    unrelated = SimpleNamespace(
        round=2,
        key=SimpleNamespace(stage="chunk", encoded="round-002:chunk:0001"),
    )
    observed: list[object] = []
    settled = tmp_path / "settled.json"
    settled.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_key": sibling.key.encoded,
                "status": "completed",
                "artifacts": [{"path": "/owned/peer", "sha256": "a" * 64}],
            }
        ),
        encoding="utf-8",
    )

    assert (
        run_injected_fine_failure_task(
            unrelated,
            target_key=target.key.encoded,
            peer_key=sibling.key.encoded,
            failure_marker=tmp_path / "failure.json",
            settled_marker=settled,
            _run_fine_task=lambda task: observed.append(task) or "completed",
        )
        == "completed"
    )
    with pytest.raises(RuntimeError, match="round-002:chunk:0000"):
        run_injected_fine_failure_task(
            target,
            target_key=target.key.encoded,
            peer_key=sibling.key.encoded,
            failure_marker=tmp_path / "failure.json",
            settled_marker=settled,
        )
    marker = json.loads((tmp_path / "failure.json").read_text(encoding="utf-8"))
    assert marker["task_key"] == "round-002:chunk:0000"
    assert marker["settled_peer_key"] == "round-003:chunk:0000"
    assert observed == [unrelated]
    assert (
        run_injected_fine_failure_task(
            target,
            target_key=target.key.encoded,
            peer_key=sibling.key.encoded,
            failure_marker=tmp_path / "failure.json",
            settled_marker=settled,
            _run_fine_task=lambda task: observed.append(task) or "retry-completed",
        )
        == "retry-completed"
    )

    peer_artifact = tmp_path / "peer-artifact.bin"
    peer_artifact.write_bytes(b"committed")
    committed = tmp_path / "committed-peer.json"
    peer_result = SimpleNamespace(
        status="completed",
        artifact_paths=(str(peer_artifact),),
    )
    assert (
        run_fine_cancellation_task(
            sibling,
            target_key=target.key.encoded,
            peer_key=sibling.key.encoded,
            committed_marker=committed,
            ready_marker=tmp_path / "ready.json",
            timeout_seconds=1.0,
            _run_fine_task=lambda _task: peer_result,
        )
        is peer_result
    )
    times = iter((0.0, 0.5, 1.0, 1.0))
    with pytest.raises(HpcWorkflowAcceptanceError, match="bounded wait"):
        run_fine_cancellation_task(
            target,
            target_key=target.key.encoded,
            peer_key=sibling.key.encoded,
            committed_marker=committed,
            ready_marker=tmp_path / "ready.json",
            timeout_seconds=1.0,
            _monotonic=lambda: next(times),
            _sleep=lambda _seconds: None,
        )
    assert json.loads((tmp_path / "ready.json").read_text(encoding="utf-8")) == {
        "blocked_task_key": "round-002:chunk:0000",
        "committed_peer_key": "round-003:chunk:0000",
        "status": "waiting-for-coordinator-interruption",
    }


def test_fine_worker_loss_is_injected_once_before_replacement_retry(
    tmp_path: Path,
) -> None:
    target = SimpleNamespace(
        round=2,
        key=SimpleNamespace(stage="chunk", encoded="round-002:chunk:0000"),
    )
    marker = tmp_path / "worker-loss.json"

    def injected_exit(code: int) -> None:
        raise SystemExit(code)

    with pytest.raises(SystemExit, match="75"):
        run_fine_worker_loss_task(
            target,
            target_key=target.key.encoded,
            loss_marker=marker,
            _worker_identity=lambda: {
                "worker_address": "tcp://node-a:1",
                "hostname": "node-a",
                "pid": 123,
                "slurm_job_id": "101",
            },
            _utc_now=lambda: "2026-07-23T00:00:00Z",
            _exit=injected_exit,
        )

    assert (
        run_fine_worker_loss_task(
            target,
            target_key=target.key.encoded,
            loss_marker=marker,
            _run_fine_task=lambda _task: "replacement-completed",
        )
        == "replacement-completed"
    )
    assert json.loads(marker.read_text(encoding="utf-8")) == {
        "hostname": "node-a",
        "injected_at": "2026-07-23T00:00:00Z",
        "pid": 123,
        "schema_version": 1,
        "slurm_job_id": "101",
        "status": "worker-loss-injected",
        "task_key": "round-002:chunk:0000",
        "worker_address": "tcp://node-a:1",
    }


def test_fine_profiler_writes_external_peak_memory_and_restores_backend(
    tmp_path: Path,
) -> None:
    from cyexm.registration.backend import AntsRegistrationBackend

    task = SimpleNamespace(key=SimpleNamespace(encoded="round-002:chunk:0000"))
    original = AntsRegistrationBackend.nonlinear

    assert (
        run_profiled_fine_task(
            task,
            timing_root=tmp_path,
            _run_fine_task=lambda _task: "completed",
        )
        == "completed"
    )

    assert AntsRegistrationBackend.nonlinear is original
    timing = json.loads(
        (tmp_path / "round-002-chunk-0000.json").read_text(encoding="utf-8")
    )
    assert timing["task_key"] == "round-002:chunk:0000"
    assert timing["status"] == "completed"
    assert timing["peak_memory_bytes"] > 0
    assert timing["nonlinear_calls"] == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.pop("performance"), "schema"),
        (
            lambda value: value["scientific"].update({"fine_fingerprint": "f" * 64}),
            "scientific fingerprint",
        ),
        (
            lambda value: value["scientific"]["tier1"].update(
                {"maximum_metric_absolute_difference": 0.0101}
            ),
            "Tier 1",
        ),
        (
            lambda value: value["scientific"]["tier2"].update(
                {"full_warp_exact": False}
            ),
            "Tier 2",
        ),
        (
            lambda value: value["scientific"].update({"public_artifacts_fine": []}),
            "public_artifacts",
        ),
        (
            lambda value: value["fixed_grid"].update({"shape_zyx": [16, 32]}),
            "fixed-grid",
        ),
        (
            lambda value: value["fixed_grid"]["prepared_paths"].append(
                "/tmp/forged.npy"
            ),
            "prepared path",
        ),
        (
            lambda value: value["placement"]["tasks"][1].update({"job_id": "101"}),
            "primary",
        ),
        (
            lambda value: value["placement"]["tasks"][3].update({"hostname": "node-a"}),
            "primary",
        ),
        (
            lambda value: value["placement"].update({"maximum_connected_workers": 5}),
            "primary",
        ),
        (
            lambda value: value["placement"].update(
                {
                    "fair_dispatch_order": [
                        "round-002:chunk:0000",
                        "round-002:chunk:0001",
                    ]
                }
            ),
            "primary",
        ),
        (
            lambda value: value["durability"]["chunk_failure_resume"].update(
                {"ordinary_artifact_store_reuse": False}
            ),
            "ArtifactStore",
        ),
        (
            lambda value: value["durability"]["cancellation_resume"].update(
                {"cleanup_outcome": "failed"}
            ),
            "cleanup",
        ),
        (
            lambda value: value["performance"].update(
                {"fine_nonlinear_wall_seconds": 80.0001}
            ),
            "fine timing",
        ),
        (
            lambda value: value["evidence_files"].append(
                deepcopy(value["evidence_files"][0])
            ),
            "duplicate",
        ),
        (
            lambda value: value["evidence_files"][0].update(
                {"relative_path": "../forged.json"}
            ),
            "canonical",
        ),
        (
            lambda value: value["identities"].update({"git_commit": "bad"}),
            "commit",
        ),
        (
            lambda value: value["identities"].update(
                {"coarse_profile_sha256": "f" * 64}
            ),
            "stale or forged",
        ),
        (
            lambda value: value["identities"]["input_checksums"][0].update(
                {"sha256": "f" * 64}
            ),
            "stale or forged",
        ),
    ],
)
def test_fine_acceptance_report_rejects_incomplete_or_forged_evidence(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    report = _fine_acceptance_report(tmp_path)
    assert callable(mutation)
    mutation(report)

    with pytest.raises(HpcWorkflowAcceptanceError, match=message):
        validate_fine_acceptance_report(report, evidence_root=tmp_path)


def _refresh_report_evidence_digest(
    report: dict[str, object],
    path: Path,
    *references: dict[str, object],
) -> None:
    digest = sha256(path.read_bytes()).hexdigest()
    for reference in references:
        reference["sha256"] = digest
    evidence_files = cast(list[dict[str, object]], report["evidence_files"])
    entry = next(item for item in evidence_files if item["relative_path"] == path.name)
    entry["sha256"] = digest


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value["fine_graph"]["attempt_counts"][0].update({"attempts": 3}),
        lambda value: value["fine_graph"]["scale_events"].reverse(),
        lambda value: value["fine_graph"]["scale_events"][1].update({"departed": []}),
        lambda value: value["fine_graph"]["scale_events"][1]["activated_probes"][
            0
        ].update({"slurm_job_id": "999"}),
        lambda value: value["fine_graph"]["placements"][0].update(
            {"status": "completed"}
        ),
        lambda value: value["fine_graph"]["placements"][1].update(
            {"worker_address": "tcp://node-a:1"}
        ),
        lambda value: value["cleanup"].update(
            {
                "recorded_job_ids": ["101", "103"],
                "remaining_owned_job_ids": [],
            }
        ),
    ),
)
def test_fine_acceptance_rejects_mutated_raw_replacement_lineage(
    tmp_path: Path,
    mutation: object,
) -> None:
    report = _fine_acceptance_report(tmp_path)
    path = tmp_path / "fine-worker-replacement-slurm-execution.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert callable(mutation)
    mutation(payload)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    replacement = cast(
        dict[str, object],
        cast(dict[str, object], report["durability"])["worker_replacement"],
    )
    _refresh_report_evidence_digest(
        report,
        path,
        cast(dict[str, object], replacement["evidence_file"]),
    )

    with pytest.raises(HpcWorkflowAcceptanceError, match="replacement"):
        validate_fine_acceptance_report(report, evidence_root=tmp_path)


def test_fine_acceptance_accepts_later_replacement_liveness_without_retry_affinity(
    tmp_path: Path,
) -> None:
    report = _fine_acceptance_report(tmp_path)
    path = tmp_path / "fine-worker-replacement-slurm-execution.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    retry = payload["fine_graph"]["placements"][1]
    retry.update(
        {
            "worker_address": "tcp://node-b:1",
            "hostname": "node-b",
            "pid": 222,
            "slurm_job_id": "102",
        }
    )
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    replacement = cast(
        dict[str, object],
        cast(dict[str, object], report["durability"])["worker_replacement"],
    )
    cast(dict[str, object], replacement["retry"]).update(
        {
            "worker_address": "tcp://node-b:1",
            "hostname": "node-b",
            "pid": 222,
            "slurm_job_id": "102",
        }
    )
    replacement["replacement_liveness_path"] = "later-task"
    replacement["liveness"] = {
        "task_key": "round-003:chunk:0000",
        "status": "completed",
        "worker_address": "tcp://node-c:1",
        "hostname": "node-c",
        "pid": 333,
        "slurm_job_id": "103",
    }
    replacement["owned_job_ids"] = ["100", "101", "102", "103"]
    outcome_path = tmp_path / "fine-worker-replacement-outcome.json"
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    outcome["owned_job_ids"] = ["100", "101", "102", "103"]
    outcome_path.write_text(json.dumps(outcome, sort_keys=True), encoding="utf-8")
    _refresh_report_evidence_digest(
        report,
        path,
        cast(dict[str, object], replacement["evidence_file"]),
    )
    _refresh_report_evidence_digest(
        report,
        outcome_path,
        cast(dict[str, object], replacement["outcome_file"]),
    )

    assert validate_fine_acceptance_report(report, evidence_root=tmp_path) == report


def test_fine_acceptance_accepts_scheduler_reschedule_before_replacement_liveness(
    tmp_path: Path,
) -> None:
    report = _fine_acceptance_report(tmp_path)
    path = tmp_path / "fine-worker-replacement-slurm-execution.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    graph = payload["fine_graph"]
    target = graph["placements"][1]
    target.update(
        {
            "attempt": 1,
            "worker_address": "tcp://node-b:1",
            "hostname": "node-b",
            "pid": 222,
            "slurm_job_id": "102",
            "coordinator_submitted_at": "2026-07-22T09:59:59Z",
            "coordinator_ready_at": "2026-07-22T10:00:02Z",
        }
    )
    graph["placements"] = [target, graph["placements"][2]]
    graph["attempt_counts"][0]["attempts"] = 1
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    replacement = cast(
        dict[str, object],
        cast(dict[str, object], report["durability"])["worker_replacement"],
    )
    replacement["failed_attempt"] = None
    replacement["recovery_mode"] = "scheduler-reschedule"
    cast(dict[str, object], replacement["retry"]).update(
        {
            "attempt": 1,
            "worker_address": "tcp://node-b:1",
            "hostname": "node-b",
            "pid": 222,
            "slurm_job_id": "102",
        }
    )
    replacement["replacement_liveness_path"] = "later-task"
    replacement["liveness"] = {
        "task_key": "round-003:chunk:0000",
        "status": "completed",
        "worker_address": "tcp://node-c:1",
        "hostname": "node-c",
        "pid": 333,
        "slurm_job_id": "103",
    }
    replacement["owned_job_ids"] = ["100", "101", "102", "103"]
    outcome_path = tmp_path / "fine-worker-replacement-outcome.json"
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    outcome["owned_job_ids"] = ["100", "101", "102", "103"]
    outcome_path.write_text(json.dumps(outcome, sort_keys=True), encoding="utf-8")
    _refresh_report_evidence_digest(
        report,
        path,
        cast(dict[str, object], replacement["evidence_file"]),
    )
    _refresh_report_evidence_digest(
        report,
        outcome_path,
        cast(dict[str, object], replacement["outcome_file"]),
    )

    assert validate_fine_acceptance_report(report, evidence_root=tmp_path) == report


def test_owned_cleanup_sources_preserve_complete_extra_owned_job_ids(
    tmp_path: Path,
) -> None:
    outcome_path = tmp_path / "fine-worker-replacement-outcome.json"
    queue_path = tmp_path / "fine-worker-replacement-queue-after.json"
    outcome_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "operation": "fine-worker-replacement",
                "status": "completed",
                "owned_job_ids": ["100", "101", "102", "103"],
                "cleanup_outcome": "empty",
                "workflow_result": {"status": "completed"},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    queue_path.write_text(json.dumps({"jobs": []}), encoding="utf-8")

    summary = _owned_cleanup_sources(
        outcome_path=outcome_path,
        queue_after_path=queue_path,
        operation="fine-worker-replacement",
        status="completed",
    )

    assert summary["owned_job_ids"] == ["100", "101", "102", "103"]
    assert summary["cleanup_outcome"] == "empty"


@pytest.mark.parametrize(
    ("outcome_mutation", "queue_jobs", "message"),
    (
        (
            lambda value: value.update(
                {
                    "cleanup": {
                        "recorded_job_ids": value.pop("owned_job_ids"),
                    }
                }
            ),
            [],
            "schema",
        ),
        (
            lambda value: None,
            [{"job_id": "103"}],
            "queue",
        ),
        (
            lambda value: value.update({"cleanup_outcome": "timeout"}),
            [],
            "cleanup outcome",
        ),
    ),
)
def test_owned_cleanup_sources_reject_invalid_authority(
    tmp_path: Path,
    outcome_mutation: object,
    queue_jobs: list[dict[str, str]],
    message: str,
) -> None:
    outcome_path = tmp_path / "fine-worker-replacement-outcome.json"
    queue_path = tmp_path / "fine-worker-replacement-queue-after.json"
    outcome = {
        "schema_version": 1,
        "operation": "fine-worker-replacement",
        "status": "completed",
        "owned_job_ids": ["100", "101", "103"],
        "cleanup_outcome": "empty",
        "workflow_result": {"status": "completed"},
    }
    assert callable(outcome_mutation)
    outcome_mutation(outcome)
    outcome_path.write_text(json.dumps(outcome, sort_keys=True), encoding="utf-8")
    queue_path.write_text(json.dumps({"jobs": queue_jobs}), encoding="utf-8")

    with pytest.raises(HpcWorkflowAcceptanceError, match=message):
        _owned_cleanup_sources(
            outcome_path=outcome_path,
            queue_after_path=queue_path,
            operation="fine-worker-replacement",
            status="completed",
        )


def test_fine_acceptance_rejects_outcome_missing_replacement_lineage_job(
    tmp_path: Path,
) -> None:
    report = _fine_acceptance_report(tmp_path)
    replacement = cast(
        dict[str, object],
        cast(dict[str, object], report["durability"])["worker_replacement"],
    )
    path = tmp_path / "fine-worker-replacement-outcome.json"
    outcome = json.loads(path.read_text(encoding="utf-8"))
    outcome["owned_job_ids"] = ["100", "101"]
    path.write_text(json.dumps(outcome, sort_keys=True), encoding="utf-8")
    replacement["owned_job_ids"] = ["100", "101"]
    _refresh_report_evidence_digest(
        report,
        path,
        cast(dict[str, object], replacement["outcome_file"]),
    )

    with pytest.raises(HpcWorkflowAcceptanceError, match="lineage"):
        validate_fine_acceptance_report(report, evidence_root=tmp_path)


def test_cancellation_outcome_preserves_owned_job_absent_from_placements(
    tmp_path: Path,
) -> None:
    report = _fine_acceptance_report(tmp_path)
    cancellation = cast(
        dict[str, object],
        cast(dict[str, object], report["durability"])["cancellation_resume"],
    )

    assert cancellation["owned_job_ids"] == ["100", "101", "102"]
    assert validate_fine_acceptance_report(report, evidence_root=tmp_path) == report


def test_fine_acceptance_rejects_replacement_without_either_liveness_path(
    tmp_path: Path,
) -> None:
    report = _fine_acceptance_report(tmp_path)
    replacement = cast(
        dict[str, object],
        cast(dict[str, object], report["durability"])["worker_replacement"],
    )
    replacement["replacement_liveness_path"] = "later-task"
    replacement["liveness"] = {
        "task_key": "round-003:chunk:0000",
        "status": "completed",
        "worker_address": "tcp://node-b:1",
        "hostname": "node-b",
        "pid": 222,
        "slurm_job_id": "102",
    }
    with pytest.raises(HpcWorkflowAcceptanceError, match="replacement"):
        validate_fine_acceptance_report(report, evidence_root=tmp_path)


def test_fine_acceptance_rejects_nonempty_replacement_queue(
    tmp_path: Path,
) -> None:
    report = _fine_acceptance_report(tmp_path)
    path = tmp_path / "fine-worker-replacement-queue-after.json"
    path.write_text(json.dumps({"jobs": [{"job_id": "103"}]}), encoding="utf-8")
    replacement = cast(
        dict[str, object],
        cast(dict[str, object], report["durability"])["worker_replacement"],
    )
    _refresh_report_evidence_digest(
        report,
        path,
        cast(dict[str, object], replacement["queue_after_file"]),
    )
    with pytest.raises(HpcWorkflowAcceptanceError, match="replacement"):
        validate_fine_acceptance_report(report, evidence_root=tmp_path)


def test_fine_acceptance_rejects_loss_marker_identity_drift(tmp_path: Path) -> None:
    report = _fine_acceptance_report(tmp_path)
    path = tmp_path / "worker-loss.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["hostname"] = "forged-node"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    replacement = cast(
        dict[str, object],
        cast(dict[str, object], report["durability"])["worker_replacement"],
    )
    _refresh_report_evidence_digest(
        report,
        path,
        cast(dict[str, object], replacement["loss_marker_file"]),
    )
    with pytest.raises(HpcWorkflowAcceptanceError, match="loss-marker"):
        validate_fine_acceptance_report(report, evidence_root=tmp_path)


@pytest.mark.parametrize(
    ("filename", "reference_name", "mutation"),
    (
        (
            "chunk-failure-execution.json",
            "failure_evidence_file",
            lambda value: value["fine_graph"]["placements"][0].update({"attempt": 2}),
        ),
        (
            "chunk-resume-execution.json",
            "resume_evidence_file",
            lambda value: value["fine_graph"]["task_results"][0].update(
                {"status": "reused"}
            ),
        ),
        (
            "chunk-failure-marker.json",
            "failure_marker_file",
            lambda value: value.update({"task_key": "round-003:chunk:0000"}),
        ),
        (
            "chunk-settled-peer.json",
            "settled_peer_file",
            lambda value: value.update({"artifacts": []}),
        ),
    ),
)
def test_fine_acceptance_rejects_mutated_failure_resume_sources(
    tmp_path: Path,
    filename: str,
    reference_name: str,
    mutation: object,
) -> None:
    report = _fine_acceptance_report(tmp_path)
    path = tmp_path / filename
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert callable(mutation)
    mutation(payload)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    resume = cast(
        dict[str, object],
        cast(dict[str, object], report["durability"])["chunk_failure_resume"],
    )
    _refresh_report_evidence_digest(
        report,
        path,
        cast(dict[str, object], resume[reference_name]),
    )
    with pytest.raises(HpcWorkflowAcceptanceError, match="failure"):
        validate_fine_acceptance_report(report, evidence_root=tmp_path)


@pytest.mark.parametrize(
    ("filename", "reference_name", "mutation"),
    (
        (
            "committed-peer.json",
            "committed_peer_file",
            lambda value: value.update({"task_key": "round-002:chunk:0000"}),
        ),
        (
            "cancellation-ready.json",
            "ready_marker_file",
            lambda value: value.update({"committed_peer_key": "round-002:chunk:0000"}),
        ),
        (
            "fine-cancellation-trigger.json",
            "trigger_file",
            lambda value: value.update({"status": "waiting"}),
        ),
        (
            "fine-cancellation-outcome.json",
            "outcome_file",
            lambda value: value.update({"cleanup_outcome": "timeout"}),
        ),
        (
            "fine-cancellation-slurm-execution.json",
            "execution_evidence_file",
            lambda value: value["cleanup"]["cluster"].update({"succeeded": False}),
        ),
        (
            "fine-cancellation-queue-after.json",
            "queue_after_file",
            lambda value: value.update({"jobs": [{"job_id": "102"}]}),
        ),
    ),
)
def test_fine_acceptance_rejects_mutated_cancellation_sources(
    tmp_path: Path,
    filename: str,
    reference_name: str,
    mutation: object,
) -> None:
    report = _fine_acceptance_report(tmp_path)
    path = tmp_path / filename
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert callable(mutation)
    mutation(payload)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    cancellation = cast(
        dict[str, object],
        cast(dict[str, object], report["durability"])["cancellation_resume"],
    )
    _refresh_report_evidence_digest(
        report,
        path,
        cast(dict[str, object], cancellation[reference_name]),
    )
    with pytest.raises(HpcWorkflowAcceptanceError, match="cancellation"):
        validate_fine_acceptance_report(report, evidence_root=tmp_path)


def test_fine_acceptance_rejects_cancellation_resume_key_drift(tmp_path: Path) -> None:
    report = _fine_acceptance_report(tmp_path)
    cancellation = cast(
        dict[str, object],
        cast(dict[str, object], report["durability"])["cancellation_resume"],
    )
    cancellation["resume_reused_keys"] = ["round-002:chunk:0000"]
    with pytest.raises(HpcWorkflowAcceptanceError, match="cancellation"):
        validate_fine_acceptance_report(report, evidence_root=tmp_path)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value["rounds"][0]["channels"].clear(),
        lambda value: value["rounds"][0]["registered"]["geometry"].update(
            {"spacing": [0.2, 0.2, 0.3]}
        ),
        lambda value: value["rounds"][0]["registered"].update({"dtype": "uint16"}),
        lambda value: (
            value["rounds"][0]["metrics"]["nonlinear-registered"]["reference"].update(
                {"Correlation": 0.4}
            ),
            value["rounds"][0]["metrics"]["nonlinear-registered"]["candidate"].update(
                {"Correlation": 0.409}
            ),
            value["rounds"][0]["metrics"]["nonlinear-registered"][
                "absolute_differences"
            ].update({"Correlation": 0.009}),
        ),
    ),
)
def test_fine_acceptance_rejects_tier1_contract_drift_with_refreshed_checksums(
    tmp_path: Path,
    mutation: object,
) -> None:
    report = _fine_acceptance_report(tmp_path)
    path = tmp_path / "tier1-equivalence.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert callable(mutation)
    mutation(payload)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    tier1 = cast(
        dict[str, object],
        cast(dict[str, object], report["scientific"])["tier1"],
    )
    _refresh_report_evidence_digest(
        report,
        path,
        cast(dict[str, object], tier1["evidence_file"]),
    )
    with pytest.raises(HpcWorkflowAcceptanceError, match="scientific comparison"):
        validate_fine_acceptance_report(report, evidence_root=tmp_path)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value["placement"]["tasks"][0].update({"job_id": "forged"}),
        lambda value: value["placement"].update({"maximum_connected_workers": 2}),
        lambda value: value["placement"]["fair_dispatch_order"].reverse(),
    ),
)
def test_fine_acceptance_rejects_primary_placement_summary_forgery(
    tmp_path: Path,
    mutation: object,
) -> None:
    report = _fine_acceptance_report(tmp_path)
    assert callable(mutation)
    mutation(report)
    with pytest.raises(HpcWorkflowAcceptanceError, match="primary"):
        validate_fine_acceptance_report(report, evidence_root=tmp_path)


def test_fine_acceptance_rejects_refreshed_primary_preparation_clock_forgery(
    tmp_path: Path,
) -> None:
    report = _fine_acceptance_report(tmp_path)
    path = tmp_path / "fine-execution.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    preparation = payload["fine_graph"]["placements"][1]
    preparation["coordinator_ready_at"] = "2026-07-22T10:00:02Z"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    placement = cast(dict[str, object], report["placement"])
    _refresh_report_evidence_digest(
        report,
        path,
        cast(dict[str, object], placement["evidence_file"]),
    )
    with pytest.raises(HpcWorkflowAcceptanceError, match="primary"):
        validate_fine_acceptance_report(report, evidence_root=tmp_path)


def test_fine_acceptance_coarse_bound_aggregates_all_calls_per_round(
    tmp_path: Path,
) -> None:
    report = _fine_acceptance_report(tmp_path)
    cast(dict[str, object], report["performance"])["coarse_nonlinear_wall_seconds"] = (
        60.0
    )
    with pytest.raises(HpcWorkflowAcceptanceError, match="coarse"):
        validate_fine_acceptance_report(report, evidence_root=tmp_path)


def test_fine_acceptance_records_summed_duration_fallback_for_all_calls(
    tmp_path: Path,
) -> None:
    report = _fine_acceptance_report(tmp_path)
    performance = cast(dict[str, object], report["performance"])
    references = cast(list[dict[str, object]], performance["coarse_timing_files"])
    for reference in references:
        path = tmp_path / cast(str, reference["relative_path"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        for call in payload["nonlinear_calls"]:
            call.pop("monotonic_started_seconds")
            call.pop("monotonic_ended_seconds")
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        _refresh_report_evidence_digest(report, path, reference)
    performance["coarse_timing_policy"] = "per-round-summed-kernel-durations"

    assert validate_fine_acceptance_report(report, evidence_root=tmp_path) == report


def test_fine_acceptance_requires_tier_source_in_evidence_checksum_set(
    tmp_path: Path,
) -> None:
    report = _fine_acceptance_report(tmp_path)
    evidence = cast(list[dict[str, object]], report["evidence_files"])
    evidence[:] = [
        item
        for item in evidence
        if item["relative_path"] != "deterministic-executor-parity.json"
    ]
    with pytest.raises(HpcWorkflowAcceptanceError, match="bound"):
        validate_fine_acceptance_report(report, evidence_root=tmp_path)


def test_fine_acceptance_rejects_nonexact_tier2_source_with_refreshed_checksums(
    tmp_path: Path,
) -> None:
    report = _fine_acceptance_report(tmp_path)
    path = tmp_path / "deterministic-executor-parity.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["strict_parity"]["rounds"][0]["affine"]["parameters"]["array_equal"] = False
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    tier2 = cast(
        dict[str, object],
        cast(dict[str, object], report["scientific"])["tier2"],
    )
    _refresh_report_evidence_digest(
        report,
        path,
        cast(dict[str, object], tier2["evidence_file"]),
    )
    with pytest.raises(HpcWorkflowAcceptanceError, match="affine"):
        validate_fine_acceptance_report(report, evidence_root=tmp_path)


def test_bounded_manifest_selects_one_round_per_strategy_in_round_order() -> None:
    manifest = _manifest(
        _round(7, "serial_chunked"),
        _round(4, "whole_image"),
        _round(9, "whole_image"),
    )

    selected = bounded_acceptance_manifest(manifest)

    assert tuple(item.round for item in selected.rounds) == (4, 7)
    assert tuple(item.strategy for item in selected.rounds) == (
        "whole_image",
        "serial_chunked",
    )
    assert selected == replace(
        manifest, rounds=(manifest.rounds[1], manifest.rounds[0])
    )


def test_resume_assertion_counts_reference_dispatch_but_proves_artifact_reuse() -> None:
    reference = SimpleNamespace(
        round=1,
        status="completed",
        artifact_paths=("reference.tif",),
        reused_artifacts=("reference.tif",),
    )
    reused = SimpleNamespace(
        round=2,
        status="reused",
        artifact_paths=("round-2.tif",),
        reused_artifacts=("round-2.tif",),
    )
    recomputed = SimpleNamespace(
        round=3,
        status="completed",
        artifact_paths=("round-3.tif",),
        reused_artifacts=(),
    )
    result = SimpleNamespace(
        succeeded=True,
        dataset_id="bounded",
        rounds=(reference, reused, recomputed),
        execution_evidence_json=json.dumps(
            {
                "attempt_counts": [
                    {"dataset_id": "bounded", "round": 1, "attempts": 1},
                    {"dataset_id": "bounded", "round": 3, "attempts": 1},
                ]
            }
        ),
    )

    _assert_resume(
        result,  # type: ignore[arg-type]
        reused_round=2,
        recomputed_round=3,
    )


@pytest.mark.parametrize(
    ("rounds", "message"),
    [
        ((_round(2, "whole_image"),), "serial_chunked"),
        ((_round(2, "serial_chunked"),), "whole_image"),
    ],
)
def test_bounded_manifest_requires_both_registration_strategies(
    rounds: tuple[RoundSpec, ...],
    message: str,
) -> None:
    with pytest.raises(HpcWorkflowAcceptanceError, match=message):
        bounded_acceptance_manifest(_manifest(*rounds))


def test_task5_profiles_are_bounded_and_confined_to_named_owned_roots(
    tmp_path: Path,
) -> None:
    approved = _profile(tmp_path)

    roots = task5_output_roots(approved)
    slurm = bounded_slurm_profile(approved, roots.parallel)
    sequential = sequential_reference_profile(approved, roots.sequential)

    assert roots.base == approved.paths.output_root / "cyexm-phase-2e-task-5"
    assert roots.scenario_roots == (
        roots.parallel,
        roots.sequential,
        roots.failure,
        roots.cancellation,
        roots.deterministic_inputs,
        roots.deterministic_parallel,
        roots.deterministic_sequential,
    )
    assert all(path.is_relative_to(roots.base) for path in roots.scenario_roots)
    assert not roots.base.exists()
    assert slurm.paths.output_root == roots.parallel
    assert approved.slurm is not None
    assert slurm.slurm == replace(approved.slurm, workers=2, task_retries=0)
    assert slurm.executor == "slurm"
    assert sequential.paths.output_root == roots.sequential
    assert sequential.paths.worker_local_directory is None
    assert sequential.executor == "sequential"
    assert sequential.slurm is None
    assert sequential.local_dask is None

    deterministic_slurm = deterministic_slurm_profile(
        approved, roots.deterministic_parallel
    )
    deterministic_sequential = deterministic_sequential_profile(
        approved, roots.deterministic_sequential
    )
    assert deterministic_slurm.native_threads == NativeThreadSpec(itk=1, blas=1)
    assert deterministic_slurm.slurm is not None
    assert deterministic_slurm.slurm.workers == 2
    assert deterministic_slurm.slurm.task_retries == 0
    assert deterministic_slurm.slurm.job_name == "cyexm-task5-det"
    assert deterministic_sequential.native_threads == NativeThreadSpec(itk=1, blas=1)
    assert deterministic_sequential.executor == "sequential"
    assert deterministic_sequential.slurm is None
    assert deterministic_sequential.paths.worker_local_directory is None


def _exact_numeric_payload() -> dict[str, object]:
    return {
        "array_equal": True,
        "close_required": True,
        "maximum_absolute_difference": 0.0,
        "rtol": 0.0,
        "atol": 0.0,
    }


def _scientific_equivalence_payload() -> dict[str, object]:
    numeric = {
        "shape": [16, 32, 48],
        "reference_dtype": "float32",
        "candidate_dtype": "float32",
        "array_equal": False,
        "close_required": False,
        "maximum_absolute_difference": 0.01,
        "mean_absolute_difference": 0.001,
        "p99_9_absolute_difference": 0.009,
        "rtol": 1e-4,
        "atol": 1e-5,
    }
    geometry = {
        "shape": [16, 32, 48],
        "origin": [1.0, 2.0, 3.0],
        "spacing": [0.2, 0.2, 0.5],
        "direction": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    }

    def image(name: str) -> dict[str, object]:
        return {
            "reference": f"/coarse/{name}.tif",
            "candidate": f"/fine/{name}.tif",
            "dtype": "float32",
            "shape": [16, 32, 48],
            "geometry": geometry,
            "numeric": numeric,
        }

    def metric(reference: float, candidate: float) -> dict[str, object]:
        return {
            "reference": {
                "Correlation": reference,
                "MattesMutualInformation": reference,
            },
            "candidate": {
                "Correlation": candidate,
                "MattesMutualInformation": candidate,
            },
            "absolute_differences": {
                "Correlation": abs(candidate - reference),
                "MattesMutualInformation": abs(candidate - reference),
            },
            "absolute_tolerance": 0.01,
        }

    return {
        "schema_version": 1,
        "scientific_fingerprint": "2" * 64,
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
                    "parameters": {
                        **numeric,
                        "shape": [12],
                    },
                },
                "warp": {
                    "geometry": {
                        "shape": [2, 3, 4],
                        "origin": [0.0, 0.0, 0.0],
                        "spacing": [0.2, 0.2, 0.5],
                        "direction": [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ],
                        "has_components": True,
                    },
                    "numeric": {
                        **numeric,
                        "shape": [2, 3, 4, 3],
                    },
                },
                "registered": image("registered"),
                "metrics": {
                    "pre-registration": metric(0.50, 0.509),
                    "linearly-registered": metric(0.60, 0.609),
                    "nonlinear-registered": metric(0.70, 0.709),
                },
                "channels": [
                    {
                        "source": "/inputs/round-002-channel.tif",
                        "interpolation": "linear",
                        "comparison": image("channel"),
                    }
                ],
            }
        ],
    }


def _strict_parity_payload() -> dict[str, object]:
    metric = {
        "absolute_tolerance": 0.0,
        "absolute_differences": {
            "Correlation": 0.0,
            "MattesMutualInformation": 0.0,
        },
    }
    return {
        "schema_version": 1,
        "scientific_fingerprint": "abc",
        "rtol": 0.0,
        "atol": 0.0,
        "comparison_policy": "strict-deterministic-numeric-parity",
        "metric_absolute_tolerance": 0.0,
        "voxel_differences_are_characterization": False,
        "rounds": [
            {
                "round": 2,
                "strategy": "serial_chunked",
                "affine": {"parameters": _exact_numeric_payload()},
                "warp": {"numeric": _exact_numeric_payload()},
                "registered": {"numeric": _exact_numeric_payload()},
                "metrics": {
                    "pre-registration": metric,
                    "linearly-registered": metric,
                    "nonlinear-registered": metric,
                },
                "channels": [
                    {
                        "source": "/inputs/round-002-channel.tif",
                        "comparison": {"numeric": _exact_numeric_payload()},
                    }
                ],
            }
        ],
    }


def _strict_internal_parity_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "local_warp_exact": True,
        "diagnostic_exact": True,
        "rounds": [
            {
                "round": 3,
                "local_warps": [
                    {
                        "name": "chunk_0000_warp.nii.gz",
                        "shape": [16, 16, 16, 3],
                        "dtype": "float32",
                        "exact": True,
                    }
                ],
                "diagnostics": [
                    {
                        "name": "chunk_0000_registered_core.npy",
                        "shape": [8, 16, 16],
                        "dtype": "float32",
                        "exact": True,
                    }
                ],
            }
        ],
    }


def test_deterministic_parity_evidence_rejects_nonexact_artifact_claim() -> None:
    strict = _strict_parity_payload()
    rounds = strict["rounds"]
    assert isinstance(rounds, list)
    round_evidence = rounds[0]
    assert isinstance(round_evidence, dict)
    affine = round_evidence["affine"]
    assert isinstance(affine, dict)
    parameters = affine["parameters"]
    assert isinstance(parameters, dict)
    parameters["array_equal"] = False
    payload = {
        "schema_version": 1,
        "status": "completed",
        "seed": "20260722",
        "native_threads": {
            "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": 1,
            "OMP_NUM_THREADS": 1,
            "OPENBLAS_NUM_THREADS": 1,
            "MKL_NUM_THREADS": 1,
            "NUMEXPR_NUM_THREADS": 1,
            "VECLIB_MAXIMUM_THREADS": 1,
        },
        "inputs": {},
        "parallel": {"status": "completed"},
        "sequential": {"status": "completed"},
        "strict_parity": strict,
        "final_queue": {"jobs": []},
    }

    with pytest.raises(HpcWorkflowAcceptanceError, match="affine is not exactly"):
        _validated_deterministic_parity_evidence(payload)


def test_deterministic_parity_subprocess_is_fresh_one_thread_and_fail_closed(
    tmp_path: Path,
) -> None:
    execution = tmp_path / "execution.yaml"
    execution.write_text("schema_version: 1\n", encoding="utf-8")
    owned_root = tmp_path / "owned"
    owned_root.mkdir()
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    observed: dict[str, object] = {}

    def complete(
        command: tuple[str, ...], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        observed["command"] = command
        observed.update(kwargs)
        evidence_path = evidence_dir / "deterministic-executor-parity.json"
        evidence_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "completed",
                    "seed": "20260722",
                    "native_threads": {
                        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": 1,
                        "OMP_NUM_THREADS": 1,
                        "OPENBLAS_NUM_THREADS": 1,
                        "MKL_NUM_THREADS": 1,
                        "NUMEXPR_NUM_THREADS": 1,
                        "VECLIB_MAXIMUM_THREADS": 1,
                    },
                    "inputs": {},
                    "parallel": {"status": "completed"},
                    "sequential": {"status": "completed"},
                    "strict_parity": _strict_parity_payload(),
                    "final_queue": {"jobs": []},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "child-out", "child-err")

    result = execute_deterministic_parity_subprocess(
        execution_path=execution,
        owned_root=owned_root,
        evidence_dir=evidence_dir,
        timeout_seconds=90.0,
        _run=complete,
        _environ={"PYTHONPATH": "/existing"},
    )

    command = observed["command"]
    assert isinstance(command, tuple)
    assert command[:3] == (sys.executable, "-m", "hpc.deterministic_parity")
    environment = observed["env"]
    assert isinstance(environment, dict)
    assert environment["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] == "1"
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        assert environment[name] == "1"
    assert environment["ANTS_RANDOM_SEED"] == "20260722"
    assert Path(environment["PYTHONPATH"].split(os.pathsep)[0]).name == "tests"
    assert observed["check"] is False
    assert observed["capture_output"] is True
    assert observed["text"] is True
    assert observed["timeout"] == 90.0
    assert result["status"] == "completed"
    assert (evidence_dir / "deterministic-coordinator.stdout.txt").read_text(
        encoding="utf-8"
    ) == "child-out"
    assert (evidence_dir / "deterministic-coordinator.stderr.txt").read_text(
        encoding="utf-8"
    ) == "child-err"


def test_fine_deterministic_parity_passes_both_exact_profiles_to_child(
    tmp_path: Path,
) -> None:
    coarse = tmp_path / "coarse.yaml"
    fine = tmp_path / "fine.yaml"
    coarse.write_text("schema_version: 1\n", encoding="utf-8")
    fine.write_text("schema_version: 1\n", encoding="utf-8")
    owned = tmp_path / "owned"
    owned.mkdir()
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    observed: dict[str, object] = {}

    def complete(
        command: tuple[str, ...], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        observed["command"] = command
        (evidence / "deterministic-executor-parity.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "completed",
                    "seed": "20260722",
                    "native_threads": {
                        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": 1,
                        "OMP_NUM_THREADS": 1,
                        "OPENBLAS_NUM_THREADS": 1,
                        "MKL_NUM_THREADS": 1,
                        "NUMEXPR_NUM_THREADS": 1,
                        "VECLIB_MAXIMUM_THREADS": 1,
                    },
                    "inputs": {},
                    "parallel": {"status": "completed"},
                    "sequential": {"status": "completed"},
                    "strict_parity": _strict_parity_payload(),
                    "strict_internal_parity": _strict_internal_parity_payload(),
                    "final_queue": {"jobs": []},
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    execute_deterministic_parity_subprocess(
        execution_path=coarse,
        fine_execution_path=fine,
        owned_root=owned,
        evidence_dir=evidence,
        timeout_seconds=10.0,
        _run=complete,
    )

    command = cast(tuple[str, ...], observed["command"])
    assert command[command.index("--execution") + 1] == str(coarse)
    assert command[command.index("--fine-execution") + 1] == str(fine)


def test_fine_deterministic_parity_rejects_forged_local_chunk_claim() -> None:
    payload = {
        "schema_version": 1,
        "status": "completed",
        "seed": "20260722",
        "native_threads": {
            "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": 1,
            "OMP_NUM_THREADS": 1,
            "OPENBLAS_NUM_THREADS": 1,
            "MKL_NUM_THREADS": 1,
            "NUMEXPR_NUM_THREADS": 1,
            "VECLIB_MAXIMUM_THREADS": 1,
        },
        "inputs": {},
        "parallel": {"status": "completed"},
        "sequential": {"status": "completed"},
        "strict_parity": _strict_parity_payload(),
        "strict_internal_parity": _strict_internal_parity_payload(),
        "final_queue": {"jobs": []},
    }
    internal = cast(dict[str, object], payload["strict_internal_parity"])
    internal["local_warp_exact"] = False

    with pytest.raises(HpcWorkflowAcceptanceError, match="internal parity"):
        _validated_deterministic_parity_evidence(
            payload,
            require_fine_internal=True,
        )


@pytest.mark.parametrize("failure", ["exit", "missing", "malformed"])
def test_deterministic_parity_subprocess_rejects_child_failure(
    tmp_path: Path,
    failure: str,
) -> None:
    execution = tmp_path / "execution.yaml"
    execution.write_text("schema_version: 1\n", encoding="utf-8")
    owned_root = tmp_path / "owned"
    owned_root.mkdir()
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()

    def fail(
        command: tuple[str, ...], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        evidence = evidence_dir / "deterministic-executor-parity.json"
        if failure == "malformed":
            evidence.write_text("{}\n", encoding="utf-8")
        return subprocess.CompletedProcess(
            command,
            7 if failure == "exit" else 0,
            "stdout",
            "stderr",
        )

    with pytest.raises(HpcWorkflowAcceptanceError, match="deterministic"):
        execute_deterministic_parity_subprocess(
            execution_path=execution,
            owned_root=owned_root,
            evidence_dir=evidence_dir,
            timeout_seconds=90.0,
            _run=fail,
        )
    assert (evidence_dir / "deterministic-coordinator.stdout.txt").is_file()
    assert (evidence_dir / "deterministic-coordinator.stderr.txt").is_file()


def test_deterministic_parity_child_rejects_thread_environment_before_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = tmp_path / "child.json"
    monkeypatch.setenv("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "8")
    monkeypatch.setenv("ANTS_RANDOM_SEED", "20260722")
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        monkeypatch.setenv(name, "1")

    exit_code = deterministic_parity_main(
        [
            "--execution",
            str(tmp_path / "execution.yaml"),
            "--owned-root",
            str(tmp_path / "owned"),
            "--evidence",
            str(evidence),
        ]
    )

    assert exit_code == 1
    failure = json.loads(evidence.read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    assert "environment mismatch" in failure["error"]["message"]


def test_deterministic_parity_child_requires_exact_empty_owned_roots(
    tmp_path: Path,
) -> None:
    owned = tmp_path / "owned"
    owned.mkdir()
    for name in (
        "deterministic-inputs",
        "deterministic-parallel",
        "deterministic-sequential",
    ):
        (owned / name).mkdir()

    roots = _require_owned_root(owned)

    assert roots == {
        "inputs": owned / "deterministic-inputs",
        "parallel": owned / "deterministic-parallel",
        "sequential": owned / "deterministic-sequential",
    }
    (roots["inputs"] / "unexpected").write_text("no", encoding="utf-8")
    with pytest.raises(RuntimeError, match="existing empty real directory"):
        _require_owned_root(owned)


def test_deterministic_parity_child_rejects_non_one_thread_result() -> None:
    result = SimpleNamespace(
        rounds=(SimpleNamespace(native_threads=(("OMP_NUM_THREADS", 2),)),)
    )

    with pytest.raises(RuntimeError, match="did not use one native thread"):
        _require_one_thread_results(result, "sequential")


def test_bounded_slurm_profile_requires_two_single_process_workers(
    tmp_path: Path,
) -> None:
    root = tmp_path / "owned"
    with pytest.raises(HpcWorkflowAcceptanceError, match="at least two workers"):
        bounded_slurm_profile(_profile(tmp_path, workers=1), root)

    approved = _profile(tmp_path)
    assert approved.slurm is not None
    invalid = replace(
        approved,
        slurm=replace(approved.slurm, processes_per_worker=2),
    )
    with pytest.raises(HpcWorkflowAcceptanceError, match="one process"):
        bounded_slurm_profile(invalid, root)


def test_owned_slurm_failure_atomically_preserves_evidence_and_proves_cleanup(
    tmp_path: Path,
) -> None:
    profile = bounded_slurm_profile(_profile(tmp_path), tmp_path / "owned")
    plan = SimpleNamespace(execution=profile)
    primary = RuntimeError("injected workflow failure")
    waited: list[tuple[str, ...]] = []

    class FakeExecutor:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.execution_evidence = {
                "observed": {
                    "worker_probes": [
                        {"slurm_job_id": "22"},
                        {"slurm_job_id": "11"},
                        {"slurm_job_id": "22"},
                    ]
                }
            }

    def fail_workflow(*_args: object, **_kwargs: object) -> None:
        raise primary

    def queue_snapshot(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {"jobs": [], "query": {"job_name": "cyexm-task5"}}

    def wait(
        _job_name: str,
        *,
        user: str,
        job_ids: tuple[str, ...],
    ) -> dict[str, object]:
        assert user == "scientist"
        waited.append(job_ids)
        return {"outcome": "empty", "jobs": [], "attempts": [{"outcome": "empty"}]}

    with pytest.raises(RuntimeError, match="injected workflow failure") as captured:
        execute_owned_slurm_workflow(
            plan,  # type: ignore[arg-type]
            runner=lambda task: task,
            evidence_dir=tmp_path,
            operation="failure",
            user="scientist",
            _executor_ctor=FakeExecutor,
            _run_workflow=fail_workflow,
            _collect_queue=queue_snapshot,
            _wait_for_queue=wait,
            _pickle_context=nullcontext,
        )

    assert captured.value is primary
    assert waited == [("11", "22")]
    assert not tuple(tmp_path.glob(".*.tmp"))
    assert (
        json.loads(
            (tmp_path / "failure-slurm-execution.json").read_text(encoding="utf-8")
        )
        == FakeExecutor().execution_evidence
    )
    assert (
        json.loads((tmp_path / "failure-queue-after.json").read_text(encoding="utf-8"))[
            "outcome"
        ]
        == "empty"
    )
    outcome = json.loads(
        (tmp_path / "failure-outcome.json").read_text(encoding="utf-8")
    )
    assert outcome["status"] == "failed"
    assert outcome["error"] == {
        "exception_type": "RuntimeError",
        "message": "injected workflow failure",
    }
    assert outcome["cleanup_outcome"] == "empty"


def _reference_task(tmp_path: Path) -> ReferenceTask:
    source = tmp_path / "reference.tif"
    output = tmp_path / "channels" / "reference.tif"
    return ReferenceTask(
        dataset_id="bounded",
        round=1,
        registration_channel_source=source,
        registration_channel_output=output,
        registration_directory=tmp_path / "registration",
        crop_bounds_path=None,
        log_path=tmp_path / "reference.log",
        voxel_spacing_xyz=(1.0, 1.0, 1.0),
        crop=False,
        native_threads=NativeThreadSpec(itk=8, blas=8),
        channels=(ReferenceChannelTask(source=source, output=output),),
    )


def _reference_result(task: ReferenceTask) -> RoundResult:
    return RoundResult(
        dataset_id=task.dataset_id,
        round=task.round,
        status="completed",
        started_at="2026-07-21T10:00:00Z",
        ended_at="2026-07-21T10:00:01Z",
        duration_seconds=1.0,
        hostname="node-reference",
        pid=101,
        dask_task_key="cyexm-bounded-reference-round-001",
        slurm_job_id="10",
        strategy="reference-preparation",
        native_threads=(
            ("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", 8),
            ("OMP_NUM_THREADS", 8),
        ),
        artifact_paths=(str(task.registration_channel_output),),
        reused_artifacts=(),
        log_path=str(task.log_path),
        failure=None,
    )


def test_failure_runner_dispatches_reference_without_moving_coordination(
    tmp_path: Path,
) -> None:
    task = _reference_task(tmp_path)
    expected = _reference_result(task)
    marker = tmp_path / "failure-marker.json"

    result = run_failure_task(
        task,
        successful_round=4,
        failing_round=7,
        committed_marker=marker,
        timeout_seconds=1_800.0,
        _run_workflow_task=lambda observed: (
            expected if observed is task else pytest.fail("unexpected task")
        ),
    )

    assert result is expected
    assert not marker.exists()


def test_cancellation_runner_dispatches_reference_without_moving_coordination(
    tmp_path: Path,
) -> None:
    task = _reference_task(tmp_path)
    expected = _reference_result(task)
    committed = tmp_path / "committed.json"
    ready = tmp_path / "ready.json"

    result = run_cancellation_task(
        task,
        successful_round=4,
        blocked_round=7,
        committed_marker=committed,
        ready_marker=ready,
        timeout_seconds=1_800.0,
        _run_workflow_task=lambda observed: (
            expected if observed is task else pytest.fail("unexpected task")
        ),
    )

    assert result is expected
    assert not committed.exists()
    assert not ready.exists()


def test_failure_runner_waits_for_committed_peer_before_failing(tmp_path: Path) -> None:
    marker = tmp_path / "peer-committed.json"
    successful = SimpleNamespace(round=4)
    failing = SimpleNamespace(round=7)
    completed = SimpleNamespace(
        status="completed",
        round=4,
        ended_at="2026-07-21T10:00:00Z",
        to_dict=lambda: {"round": 4},
    )

    result = run_failure_task(
        successful,  # type: ignore[arg-type]
        successful_round=4,
        failing_round=7,
        committed_marker=marker,
        timeout_seconds=1_800.0,
        _run_round_task=lambda _task: completed,  # type: ignore[arg-type]
        _utc_now=lambda: "2026-07-21T10:00:01Z",
    )

    assert result is completed
    assert json.loads(marker.read_text(encoding="utf-8")) == {
        "marker_written_at": "2026-07-21T10:00:01Z",
        "result_ended_at": "2026-07-21T10:00:00Z",
        "round": 4,
        "schema_version": 1,
        "status": "completed",
    }
    with pytest.raises(
        RuntimeError,
        match=(
            "injected Task 5 failure for round 7 after committed round 4 "
            "at 2026-07-21T10:00:01Z"
        ),
    ):
        run_failure_task(
            failing,  # type: ignore[arg-type]
            successful_round=4,
            failing_round=7,
            committed_marker=marker,
            timeout_seconds=1_800.0,
        )
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_cancellation_runner_announces_blocked_future_after_peer_commit(
    tmp_path: Path,
) -> None:
    committed = tmp_path / "peer-committed.json"
    ready = tmp_path / "cancellation-ready.json"
    committed.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "round": 4,
                "status": "completed",
                "result_ended_at": "2026-07-21T10:00:00Z",
                "marker_written_at": "2026-07-21T10:00:01Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    task = SimpleNamespace(round=7)

    def interrupt_sleep(_seconds: float) -> None:
        raise KeyboardInterrupt("unit stop")

    with pytest.raises(KeyboardInterrupt, match="unit stop"):
        run_cancellation_task(
            task,  # type: ignore[arg-type]
            successful_round=4,
            blocked_round=7,
            committed_marker=committed,
            ready_marker=ready,
            timeout_seconds=1_800.0,
            _sleep=interrupt_sleep,
        )

    assert json.loads(ready.read_text(encoding="utf-8")) == {
        "blocked_round": 7,
        "committed_round": 4,
        "status": "waiting-for-coordinator-interruption",
    }


def test_cancellation_harness_maps_coordinator_interrupt_to_130_and_restores_signal(
    tmp_path: Path,
) -> None:
    profile = bounded_slurm_profile(_profile(tmp_path), tmp_path / "owned")
    plan = SimpleNamespace(execution=profile)
    ready = tmp_path / "ready.json"
    ready.write_text(
        json.dumps(
            {
                "blocked_round": 7,
                "committed_round": 4,
                "status": "waiting-for-coordinator-interruption",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    signal_sent = threading.Event()
    previous = signal.getsignal(signal.SIGINT)

    def send_signal(pid: int, signum: int) -> None:
        assert pid > 0
        assert signum == signal.SIGINT
        signal_sent.set()

    def interrupted_execute(*_args: object, **_kwargs: object) -> None:
        assert signal_sent.wait(timeout=2.0)
        raise KeyboardInterrupt("injected coordinator interrupt")

    exit_code = execute_cancellation_acceptance(
        plan,  # type: ignore[arg-type]
        runner=lambda task: task,
        evidence_dir=tmp_path,
        operation="cancellation",
        user="scientist",
        ready_marker=ready,
        successful_round=4,
        blocked_round=7,
        timeout_seconds=2.0,
        _execute_owned=interrupted_execute,
        _send_signal=send_signal,
    )

    assert exit_code == 130
    assert signal.getsignal(signal.SIGINT) is previous
    assert json.loads(
        (tmp_path / "cancellation-trigger.json").read_text(encoding="utf-8")
    ) == {
        "blocked_round": 7,
        "committed_round": 4,
        "exit_code": 130,
        "signal": "SIGINT",
        "status": "interrupted",
        "timeout_seconds": 2.0,
    }


def test_task5_root_ownership_preserves_general_results_siblings(
    tmp_path: Path,
) -> None:
    general = tmp_path / "general-results"
    general.mkdir()
    unrelated = general / "caller-owned.txt"
    unrelated.write_text("preserve", encoding="utf-8")
    profile = replace(
        _profile(tmp_path),
        paths=replace(_profile(tmp_path).paths, output_root=general),
    )
    roots = task5_output_roots(profile)

    ownership = prepare_task5_output_roots(roots)
    validate_task5_output_ownership(roots, ownership)

    assert unrelated.read_text(encoding="utf-8") == "preserve"
    assert {path.name for path in general.iterdir()} == {
        "caller-owned.txt",
        "cyexm-phase-2e-task-5",
    }
    assert set(ownership) == {
        "schema_version",
        "general_results_root",
        "entries_before",
        "owned_base",
        "scenario_roots",
    }
    with pytest.raises(HpcWorkflowAcceptanceError, match="must not exist"):
        prepare_task5_output_roots(roots)
    assert unrelated.read_text(encoding="utf-8") == "preserve"


def test_deterministic_acceptance_inputs_and_manifest_are_bounded(
    tmp_path: Path,
) -> None:
    import numpy as np
    import tifffile

    input_root = tmp_path / "inputs"
    input_root.mkdir()
    evidence = write_deterministic_inputs(input_root)
    manifest = deterministic_acceptance_manifest(input_root)

    assert manifest.dataset_id == "phase2e-deterministic-parity"
    assert manifest.registration.linear == RegistrationStageSpec(
        type="TRSAA", accuracy="dry run"
    )
    assert manifest.registration.nonlinear == NonlinearRegistrationStageSpec(
        type="SyNOnly", accuracy="dry run", metric="mattes", radius=None
    )
    assert tuple(item.strategy for item in manifest.rounds) == (
        "whole_image",
        "serial_chunked",
    )
    assert manifest.rounds[1].chunk_size_zyx == (8, 16, 16)
    assert manifest.rounds[1].overlap_fraction == 0.25
    expected = {
        input_root / "fixed.tif",
        input_root / "moving-whole.tif",
        input_root / "moving-chunked.tif",
    }
    assert set(input_root.glob("*.tif")) == expected
    arrays = [tifffile.imread(path) for path in sorted(expected)]
    assert all(array.shape == (16, 32, 32) for array in arrays)
    assert all(array.dtype == np.dtype(np.float32) for array in arrays)
    assert all(np.all(np.isfinite(array)) for array in arrays)
    assert not np.array_equal(arrays[0], arrays[1])
    assert not np.array_equal(arrays[1], arrays[2])
    assert evidence["shape_zyx"] == [16, 32, 32]

    with pytest.raises(HpcWorkflowAcceptanceError, match="empty"):
        write_deterministic_inputs(input_root)


def test_seeded_runner_sets_and_restores_acceptance_seed() -> None:
    task = SimpleNamespace(round=2)
    environment: dict[str, str] = {}

    def observe(observed: object) -> str:
        assert observed is task
        assert environment["ANTS_RANDOM_SEED"] == "20260722"
        return "completed"

    assert (
        run_seeded_workflow_task(
            task,  # type: ignore[arg-type]
            _run_workflow_task=observe,  # type: ignore[arg-type]
            _environ=environment,
        )
        == "completed"
    )
    assert "ANTS_RANDOM_SEED" not in environment

    environment["ANTS_RANDOM_SEED"] = "caller-seed"

    def fail(_observed: object) -> RoundResult:
        assert environment["ANTS_RANDOM_SEED"] == "20260722"
        raise RuntimeError("injected seeded failure")

    with pytest.raises(RuntimeError, match="injected seeded failure"):
        run_seeded_workflow_task(
            task,  # type: ignore[arg-type]
            _run_workflow_task=fail,  # type: ignore[arg-type]
            _environ=environment,
        )
    assert environment["ANTS_RANDOM_SEED"] == "caller-seed"


def _round_result(
    round_number: int,
    *,
    job_id: str,
    hostname: str,
    started_at: str,
    ended_at: str,
) -> RoundResult:
    return RoundResult(
        dataset_id="bounded",
        round=round_number,
        status="completed",
        started_at=started_at,
        ended_at=ended_at,
        duration_seconds=2.0,
        hostname=hostname,
        pid=100 + round_number,
        dask_task_key=f"cyexm-bounded-round-{round_number:03d}",
        slurm_job_id=job_id,
        strategy="whole_image" if round_number == 4 else "serial_chunked",
        native_threads=(("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", 1),),
        artifact_paths=(f"/output/{round_number}.tif",),
        reused_artifacts=(),
        log_path=f"/output/{round_number}.log",
        failure=None,
    )


def test_parallel_placement_requires_two_overlapping_unique_jobs_and_hosts() -> None:
    moving = (
        _round_result(
            4,
            job_id="11",
            hostname="node-a",
            started_at="2026-07-21T10:00:00Z",
            ended_at="2026-07-21T10:00:04Z",
        ),
        _round_result(
            7,
            job_id="22",
            hostname="node-b",
            started_at="2026-07-21T10:00:01Z",
            ended_at="2026-07-21T10:00:05Z",
        ),
    )
    reference = replace(
        _round_result(
            1,
            job_id="10",
            hostname="node-a",
            started_at="2026-07-21T09:59:55Z",
            ended_at="2026-07-21T09:59:59Z",
        ),
        strategy="reference-preparation",
    )
    results = (reference, *moving)

    evidence = parallel_placement_evidence(results)

    assert evidence == {
        "rounds": [4, 7],
        "job_ids": ["11", "22"],
        "hostnames": ["node-a", "node-b"],
        "overlap_proven": True,
        "strategies": ["whole_image", "serial_chunked"],
    }
    with pytest.raises(HpcWorkflowAcceptanceError, match="unique hostname"):
        parallel_placement_evidence(
            (reference, moving[0], replace(moving[1], hostname="node-a"))
        )


def test_stochastic_numeric_evidence_records_nonrepeatable_difference() -> None:
    import numpy as np

    evidence = numeric_comparison_evidence(
        np.array([0, 100, 200], dtype=np.uint16),
        np.array([0, 102, 197], dtype=np.uint16),
        location="registered image",
        require_close=False,
    )

    assert evidence["array_equal"] is False
    assert evidence["maximum_absolute_difference"] == 3.0
    assert evidence["mean_absolute_difference"] == pytest.approx(5.0 / 3.0)
    assert evidence["close_required"] is False


def test_strict_numeric_evidence_rejects_the_same_difference() -> None:
    import numpy as np

    with pytest.raises(AssertionError):
        numeric_comparison_evidence(
            np.array([0, 100, 200], dtype=np.uint16),
            np.array([0, 102, 197], dtype=np.uint16),
            location="registered image",
            require_close=True,
        )


def test_strict_numeric_evidence_requires_bitwise_equal_values() -> None:
    import numpy as np

    evidence = numeric_comparison_evidence(
        np.array([1.0, 2.0], dtype=np.float32),
        np.array([1.0, 2.0], dtype=np.float32),
        location="strict artifact",
        require_close=True,
        rtol=0.0,
        atol=0.0,
    )

    assert evidence["array_equal"] is True
    assert evidence["maximum_absolute_difference"] == 0.0
    assert evidence["rtol"] == 0.0
    assert evidence["atol"] == 0.0


def test_metric_equivalence_accepts_observed_repeat_variation_and_rejects_excess() -> (
    None
):
    accepted = metric_comparison_evidence(
        {"Correlation": 0.8622, "MattesMutualInformation": 0.3864},
        {"Correlation": 0.8661, "MattesMutualInformation": 0.3882},
        location="round 2 nonlinear metrics",
    )
    assert accepted["absolute_tolerance"] == 0.01
    assert accepted["absolute_differences"] == pytest.approx(
        {"Correlation": 0.0039, "MattesMutualInformation": 0.0018}
    )

    with pytest.raises(HpcWorkflowAcceptanceError, match="Correlation"):
        metric_comparison_evidence(
            {"Correlation": 0.86, "MattesMutualInformation": 0.38},
            {"Correlation": 0.84, "MattesMutualInformation": 0.38},
            location="round 2 nonlinear metrics",
        )


def test_task5_runner_serializes_by_value_without_worker_hpc_import(
    tmp_path: Path,
) -> None:
    import cloudpickle
    from distributed.protocol import pickle as distributed_pickle

    module_name = run_failure_task.__module__
    registry = cloudpickle.cloudpickle._PICKLE_BY_VALUE_MODULES
    was_registered = module_name in registry
    with workflow_module_by_value():
        assert module_name in registry
        payload = distributed_pickle.dumps(run_failure_task)
    assert (module_name in registry) is was_registered

    payload_path = tmp_path / "runner.pickle"
    payload_path.write_bytes(payload)
    source_root = Path(__file__).resolve().parents[2] / "src"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(source_root)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import pathlib, sys\n"
            "from distributed.protocol import pickle as distributed_pickle\n"
            "runner = distributed_pickle.loads(\n"
            "    pathlib.Path(sys.argv[1]).read_bytes()\n"
            ")\n"
            "assert callable(runner)\n"
            "assert 'hpc' not in sys.modules\n"
            "assert 'hpc.workflow_acceptance' not in sys.modules",
            str(payload_path),
        ],
        cwd="/tmp",
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_injected_failure_rejects_timeout_with_same_final_status_shape(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "committed-before-failure.json"
    marker.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "round": 4,
                "status": "completed",
                "result_ended_at": "2026-07-21T10:00:00Z",
                "marker_written_at": "2026-07-21T10:00:01Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    same_shape_timeout = {
        "status": "failed",
        "rounds": [
            {
                "round": 4,
                "status": "completed",
                "ended_at": "2026-07-21T10:00:00Z",
                "failure": None,
            },
            {
                "round": 7,
                "status": "failed",
                "ended_at": "2026-07-21T10:15:00Z",
                "failure": {
                    "stage": "round",
                    "exception_type": "HpcWorkflowAcceptanceError",
                    "message": "timed out waiting for committed round 4",
                },
            },
        ],
    }

    with pytest.raises(
        HpcWorkflowAcceptanceError,
        match="intended injected RuntimeError",
    ):
        validate_injected_failure_record(
            same_shape_timeout,
            successful_round=4,
            failing_round=7,
            committed_marker=marker,
        )


def test_injected_failure_accepts_exact_failure_tied_to_durable_commit_marker(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "committed-before-failure.json"
    marker_payload = {
        "schema_version": 1,
        "round": 4,
        "status": "completed",
        "result_ended_at": "2026-07-21T10:00:00Z",
        "marker_written_at": "2026-07-21T10:00:01Z",
    }
    marker.write_text(json.dumps(marker_payload) + "\n", encoding="utf-8")
    record = {
        "status": "failed",
        "rounds": [
            {
                "round": 4,
                "status": "completed",
                "ended_at": "2026-07-21T10:00:00Z",
                "failure": None,
            },
            {
                "round": 7,
                "status": "failed",
                "failure": {
                    "stage": "round",
                    "exception_type": "RuntimeError",
                    "message": (
                        "injected Task 5 failure for round 7 after committed round 4 "
                        "at 2026-07-21T10:00:01Z"
                    ),
                },
            },
        ],
    }

    evidence = validate_injected_failure_record(
        record,
        successful_round=4,
        failing_round=7,
        committed_marker=marker,
    )

    assert evidence["ordering_proven"] is True
    assert evidence["committed_marker"] == marker_payload


@pytest.mark.parametrize(
    ("walltime", "seconds"),
    [
        ("72:00:00", 72 * 60 * 60),
        ("3-00:00:00", 3 * 24 * 60 * 60),
        ("90", 90 * 60),
        ("90:00", 90 * 60),
    ],
)
def test_coordination_timeout_uses_the_approved_slurm_walltime(
    tmp_path: Path,
    walltime: str,
    seconds: int,
) -> None:
    profile = _profile(tmp_path)
    assert profile.slurm is not None
    profile = replace(profile, slurm=replace(profile.slurm, walltime=walltime))

    assert coordination_timeout_seconds(profile) == float(seconds)
