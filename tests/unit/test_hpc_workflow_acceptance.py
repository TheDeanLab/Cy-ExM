from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest
from hpc.workflow_acceptance import (
    HpcWorkflowAcceptanceError,
    bounded_acceptance_manifest,
    bounded_slurm_profile,
    coordination_timeout_seconds,
    execute_cancellation_acceptance,
    execute_owned_slurm_workflow,
    parallel_placement_evidence,
    prepare_task5_output_roots,
    run_cancellation_task,
    run_failure_task,
    sequential_reference_profile,
    task5_output_roots,
    validate_injected_failure_record,
    validate_task5_output_ownership,
    workflow_module_by_value,
)

from cyexm.config.models import (
    ChannelSpec,
    DatasetManifest,
    ExecutionPaths,
    ExecutionProfile,
    NativeThreadSpec,
    ReferenceChannelSpec,
    ReferenceSpec,
    RegistrationSpec,
    RegistrationStageSpec,
    RoundSpec,
    SlurmSpec,
)
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
            nonlinear=RegistrationStageSpec(type="SyNOnly", accuracy="high"),
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
