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
    fine_acceptance_profiles,
    validate_fine_acceptance_report,
)
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
    evidence_path = root / "fine-execution.json"
    evidence_path.write_bytes(b'{"fine":"evidence"}\n')
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
            "started_at": "2026-07-22T10:00:00Z",
            "ended_at": "2026-07-22T10:00:10Z",
            "duration_seconds": 10.0,
        },
        {
            "task_key": "round-003:chunk:0000",
            "kind": "chunk",
            "round": 3,
            "job_id": "102",
            "hostname": "node-b",
            "worker": "tcp://node-b:1",
            "started_at": "2026-07-22T10:00:01Z",
            "ended_at": "2026-07-22T10:00:11Z",
            "duration_seconds": 10.0,
        },
        {
            "task_key": "round-002:channel:0000",
            "kind": "channel",
            "round": 2,
            "job_id": "101",
            "hostname": "node-a",
            "worker": "tcp://node-a:1",
            "started_at": "2026-07-22T10:00:12Z",
            "ended_at": "2026-07-22T10:00:18Z",
            "duration_seconds": 6.0,
        },
        {
            "task_key": "round-003:channel:0000",
            "kind": "channel",
            "round": 3,
            "job_id": "102",
            "hostname": "node-b",
            "worker": "tcp://node-b:1",
            "started_at": "2026-07-22T10:00:13Z",
            "ended_at": "2026-07-22T10:00:19Z",
            "duration_seconds": 6.0,
        },
    ]
    identity = FileIdentity.from_path(input_path).to_dict()
    artifact = {
        "relative_path": "schema-v1/sample/rounds/round-002/channels/result.tif",
        "sha256": "b" * 64,
    }
    resume = {
        "injected_task_key": "round-002:chunk:0000",
        "failure_status": "failed",
        "settled_sibling_keys": ["round-003:chunk:0000"],
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
                "metric_absolute_tolerance": 0.01,
                "maximum_metric_absolute_difference": 0.009,
                "structure_equal": True,
            },
            "tier2": {
                "seed": "20260722",
                "native_threads": 1,
                "affine_exact": True,
                "local_warp_exact": True,
                "full_warp_exact": True,
                "diagnostic_exact": True,
                "metrics_exact": True,
                "channels": [
                    {
                        "relative_path": artifact["relative_path"],
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
                "injected_task_key": "round-002:channel:0000",
                "settled_sibling_keys": ["round-003:channel:0000"],
                "resume_reused_keys": ["round-003:channel:0000"],
                "resume_recomputed_keys": ["round-002:channel:0000"],
            },
            "cancellation_resume": {
                "status": "cancelled",
                "structured_cancellation": True,
                "preserved_artifacts": [artifact],
                "resume_reused_keys": ["round-003:chunk:0000"],
                "resume_recomputed_keys": ["round-002:chunk:0000"],
                "replacement_events": 1,
                "liveness_proven": True,
                "owned_job_ids": ["101", "102"],
                "remaining_owned_job_ids": [],
            },
        },
        "performance": {
            "coarse_nonlinear_wall_seconds": 100.0,
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
            {
                "relative_path": "fine-execution.json",
                "sha256": sha256(evidence_path.read_bytes()).hexdigest(),
            }
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
    )
    fine_approved = replace(
        approved,
        paths=replace(approved.paths, output_root=tmp_path / "fine"),
        parallelism=ParallelismSpec(mode="fine_grained"),
    )

    coarse, fine = fine_acceptance_profiles(coarse_approved, fine_approved)

    assert coarse.parallelism.mode == "coarse"
    assert fine.parallelism.mode == "fine_grained"
    assert coarse.paths.output_root == tmp_path / "coarse"
    assert fine.paths.output_root == tmp_path / "fine"
    assert coarse is coarse_approved
    assert fine is fine_approved


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
        evidence_paths=(tmp_path / "fine-execution.json",),
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


def test_fine_failure_and_cancellation_runners_target_only_exact_stage(
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
    observed: list[object] = []

    assert (
        run_injected_fine_failure_task(
            sibling,
            target_round=2,
            target_stage="chunk",
            failure_marker=tmp_path / "failure.json",
            _run_fine_task=lambda task: observed.append(task) or "completed",
        )
        == "completed"
    )
    with pytest.raises(RuntimeError, match="round-002:chunk:0000"):
        run_injected_fine_failure_task(
            target,
            target_round=2,
            target_stage="chunk",
            failure_marker=tmp_path / "failure.json",
        )
    marker = json.loads((tmp_path / "failure.json").read_text(encoding="utf-8"))
    assert marker["task_key"] == "round-002:chunk:0000"
    assert observed == [sibling]

    times = iter((0.0, 0.5, 1.0, 1.0))
    with pytest.raises(HpcWorkflowAcceptanceError, match="bounded wait"):
        run_fine_cancellation_task(
            target,
            target_round=2,
            target_stage="chunk",
            ready_marker=tmp_path / "ready.json",
            timeout_seconds=1.0,
            _monotonic=lambda: next(times),
            _sleep=lambda _seconds: None,
        )
    assert json.loads((tmp_path / "ready.json").read_text(encoding="utf-8")) == {
        "blocked_round": 2,
        "committed_round": 2,
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
            target_round=2,
            target_stage="chunk",
            loss_marker=marker,
            _exit=injected_exit,
        )

    assert (
        run_fine_worker_loss_task(
            target,
            target_round=2,
            target_stage="chunk",
            loss_marker=marker,
            _run_fine_task=lambda _task: "replacement-completed",
        )
        == "replacement-completed"
    )
    assert json.loads(marker.read_text(encoding="utf-8"))["task_key"] == (
        "round-002:chunk:0000"
    )


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
            "chunk",
        ),
        (
            lambda value: value["placement"]["tasks"][3].update({"hostname": "node-a"}),
            "channel",
        ),
        (
            lambda value: value["placement"].update({"maximum_connected_workers": 5}),
            "worker cap",
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
            "fair",
        ),
        (
            lambda value: value["durability"]["chunk_failure_resume"].update(
                {"ordinary_artifact_store_reuse": False}
            ),
            "ArtifactStore",
        ),
        (
            lambda value: value["durability"]["cancellation_resume"].update(
                {"remaining_owned_job_ids": ["101"]}
            ),
            "cleanup",
        ),
        (
            lambda value: value["performance"].update(
                {"fine_nonlinear_wall_seconds": 80.0001}
            ),
            "20%",
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
                "affine": {"parameters": _exact_numeric_payload()},
                "warp": {"numeric": _exact_numeric_payload()},
                "registered": {"numeric": _exact_numeric_payload()},
                "metrics": {
                    "pre-registration": metric,
                    "linearly-registered": metric,
                    "nonlinear-registered": metric,
                },
                "channels": [{"comparison": {"numeric": _exact_numeric_payload()}}],
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
