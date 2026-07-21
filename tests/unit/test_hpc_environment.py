from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from hpc.environment import (
    ACCEPTANCE_PACKAGES,
    HpcAcceptanceError,
    build_smoke_tasks,
    collect_partition_snapshot,
    collect_queue_snapshot,
    parse_scontrol_record,
    parse_slurm_tres_memory_bytes,
    partition_is_exclusive,
    prepare_acceptance_context,
    require_cli_paths,
    run_environment_smoke,
    smoke_module_by_value,
    wait_for_queue_empty,
)

from cyexm.config.models import (
    NativeThreadSpec,
    RegistrationSpec,
    RegistrationStageSpec,
)
from cyexm.workflow.models import ChannelTask, RoundResult, RoundTask

pytestmark = pytest.mark.unit


def _write_approved_inputs(archive_root: Path, project_root: Path) -> tuple[Path, Path]:
    approved = archive_root / "approved"
    dataset_root = approved / "dataset"
    output_root = approved / "output"
    scratch_root = project_root / "cyexm-smoke"
    dataset_root.mkdir(parents=True)
    output_root.mkdir()
    scratch_root.mkdir(parents=True)
    (dataset_root / "fixed.tif").write_bytes(b"fixed")
    (dataset_root / "moving.tif").write_bytes(b"moving")
    dataset = approved / "dataset.yaml"
    dataset.write_text(
        """\
schema_version: 1
dataset_id: zenodo-smoke
voxel_spacing_um: {x: 1.0, y: 1.0, z: 1.0}
reference: {round: 1, channel: fixed.tif}
registration:
  crop: false
  linear: {type: TRSAA, accuracy: high}
  nonlinear: {type: SyNOnly, accuracy: high}
rounds:
  - round: 2
    strategy: whole_image
    registration_channel: moving.tif
    channels:
      - {file: moving.tif, interpolation: linear}
""",
        encoding="utf-8",
    )
    execution = approved / "execution.yaml"
    execution.write_text(
        f"""\
schema_version: 1
executor: slurm
paths:
  dataset_root: {dataset_root}
  output_root: {output_root}
  worker_local_directory: {scratch_root}
native_threads: {{itk: 1, blas: 1}}
slurm:
  queue: 512GB
  workers: 2
  cores_per_worker: 72
  processes_per_worker: 1
  memory: 500GB
  walltime: "01:00:00"
  interface: ib0
  scheduler_interface: ib0
  startup_timeout: 15m
  task_retries: 0
  job_name: cyexm-smoke
  job_extra_directives: []
""",
        encoding="utf-8",
    )
    return dataset, execution


@pytest.mark.parametrize("missing", ("dataset", "execution", "evidence"))
def test_hpc_cli_paths_require_all_three_explicit_options(missing: str) -> None:
    values = {
        "dataset": "/archive/dataset.yaml",
        "execution": "/archive/execution.yaml",
        "evidence": "/archive/evidence",
    }
    values[missing] = None

    with pytest.raises(HpcAcceptanceError, match=rf"--cyexm-{missing}"):
        require_cli_paths(values["dataset"], values["execution"], values["evidence"])


def test_prepare_context_loads_public_yaml_and_writes_atomic_provenance(
    tmp_path: Path,
) -> None:
    archive_root = tmp_path / "archive"
    project_root = tmp_path / "project"
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    dataset, execution = _write_approved_inputs(archive_root, project_root)
    evidence = archive_root / "approved" / "evidence"
    evidence.mkdir()
    versions = {name: f"{name}-version" for name in ACCEPTANCE_PACKAGES}

    context = prepare_acceptance_context(
        str(dataset),
        str(execution),
        str(evidence),
        repository_root=repository_root,
        archive_root=archive_root,
        project_root=project_root,
        live_roots=(tmp_path / "live-spheroid",),
        argv=("pytest", "-m", "hpc"),
        _now=lambda: datetime(2026, 7, 21, 12, 30, tzinfo=UTC),
        _git_commit=lambda _root: "a" * 40,
        _version=versions.__getitem__,
    )

    assert context.manifest.dataset_id == "zenodo-smoke"
    assert context.profile.executor == "slurm"
    assert context.evidence_dir == evidence.resolve()
    record = json.loads((evidence / "acceptance-context.json").read_text())
    assert record == {
        "schema_version": 1,
        "timestamp_utc": "2026-07-21T12:30:00Z",
        "git_commit": "a" * 40,
        "package_versions": versions,
        "paths": {
            "dataset_manifest": str(dataset.resolve()),
            "execution_profile": str(execution.resolve()),
            "evidence_dir": str(evidence.resolve()),
            "dataset_root": str(context.profile.paths.dataset_root),
            "output_root": str(context.profile.paths.output_root),
            "worker_local_directory": str(context.profile.paths.worker_local_directory),
        },
        "argv": ["pytest", "-m", "hpc"],
    }
    assert list(evidence.glob(".*acceptance-context*.tmp*")) == []


@pytest.mark.parametrize("unsafe_location", ("repository", "live"))
def test_prepare_context_rejects_evidence_near_protected_workflows(
    tmp_path: Path,
    unsafe_location: str,
) -> None:
    archive_root = tmp_path / "archive"
    project_root = tmp_path / "project"
    repository_root = archive_root / "repository"
    repository_root.mkdir(parents=True)
    live_root = archive_root / "live-spheroid"
    live_root.mkdir()
    dataset, execution = _write_approved_inputs(archive_root, project_root)
    parent = repository_root if unsafe_location == "repository" else live_root
    evidence = parent / "evidence"
    evidence.mkdir()

    with pytest.raises(HpcAcceptanceError, match="protected"):
        prepare_acceptance_context(
            dataset,
            execution,
            evidence,
            repository_root=repository_root,
            archive_root=archive_root,
            project_root=project_root,
            live_roots=(live_root,),
            argv=("pytest",),
            _git_commit=lambda _root: "a" * 40,
            _version=lambda name: name,
        )


def test_prepare_context_rejects_a_nonempty_evidence_directory(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    project_root = tmp_path / "project"
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    dataset, execution = _write_approved_inputs(archive_root, project_root)
    evidence = archive_root / "approved" / "evidence"
    evidence.mkdir()
    (evidence / "prior-run.json").write_text("{}", encoding="utf-8")

    with pytest.raises(HpcAcceptanceError, match="empty"):
        prepare_acceptance_context(
            dataset,
            execution,
            evidence,
            repository_root=repository_root,
            archive_root=archive_root,
            project_root=project_root,
            live_roots=(),
            argv=("pytest",),
            _git_commit=lambda _root: "a" * 40,
            _version=lambda name: name,
        )


def test_parse_scontrol_record_preserves_embedded_equals() -> None:
    assert parse_scontrol_record(
        "JobId=913 State=RUNNING TRES=cpu=72,mem=500G,node=1 NumNodes=1"
    ) == {
        "JobId": "913",
        "State": "RUNNING",
        "TRES": "cpu=72,mem=500G,node=1",
        "NumNodes": "1",
    }


def test_parse_slurm_tres_memory_uses_slurm_binary_units() -> None:
    assert parse_slurm_tres_memory_bytes("cpu=72,mem=500G,node=1") == 500 * 1024**3


class _Completed:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout
        self.stderr = ""
        self.returncode = 0


def test_partition_snapshot_uses_only_read_only_sinfo_and_scontrol() -> None:
    calls: list[tuple[str, ...]] = []

    def run(arguments: list[str], **_kwargs: Any) -> _Completed:
        calls.append(tuple(arguments))
        if arguments == ["sinfo", "--version"]:
            return _Completed("slurm 24.11.5\n")
        if arguments[0] == "sinfo":
            return _Completed(
                "512GB|up|idle|3-00:00:00|2|72|515000|exclusive|ib|n[1-2]\n"
            )
        if arguments[-2:] == ["partition", "512GB"]:
            return _Completed(
                "PartitionName=512GB State=UP TotalCPUs=144 TotalNodes=2 "
                "OverSubscribe=EXCLUSIVE MaxMemPerNode=515000\n"
            )
        if arguments[-2:] == ["show", "config"]:
            return _Completed(
                "SlurmctldHost = controller\nSelectType = select/cons_tres\n"
            )
        raise AssertionError(arguments)

    snapshot = collect_partition_snapshot("512GB", _run=run)

    assert [call[0] for call in calls] == ["sinfo", "scontrol", "scontrol", "sinfo"]
    assert snapshot["partition"]["OverSubscribe"] == "EXCLUSIVE"
    assert snapshot["rows"][0]["memory_mb"] == 515000
    assert snapshot["configuration"]["SlurmctldHost"] == "controller"
    assert snapshot["slurm_version"] == "slurm 24.11.5"


def test_queue_snapshot_is_scoped_to_user_and_exact_job_name() -> None:
    calls: list[tuple[str, ...]] = []

    def run(arguments: list[str], **_kwargs: Any) -> _Completed:
        calls.append(tuple(arguments))
        return _Completed("913|cyexm-smoke|RUNNING|node-a\n")

    snapshot = collect_queue_snapshot("cyexm-smoke", user="dean", _run=run)

    assert calls == [
        (
            "squeue",
            "--noheader",
            "--user",
            "dean",
            "--name",
            "cyexm-smoke",
            "--format=%i|%j|%T|%N",
        )
    ]
    assert snapshot["jobs"] == [
        {
            "job_id": "913",
            "job_name": "cyexm-smoke",
            "state": "RUNNING",
            "nodes": "node-a",
        }
    ]


def test_cleanup_wait_is_bounded_and_records_the_empty_observation() -> None:
    outputs = iter(("913|cyexm-smoke|COMPLETING|node-a\n", ""))
    clock = iter((10.0, 10.5, 11.0))

    def run(_arguments: list[str], **_kwargs: Any) -> _Completed:
        return _Completed(next(outputs))

    observed = wait_for_queue_empty(
        "cyexm-smoke",
        user="dean",
        timeout_seconds=5.0,
        poll_seconds=0.0,
        _run=run,
        _monotonic=lambda: next(clock),
        _sleep=lambda _seconds: None,
    )

    assert len(observed["attempts"]) == 2
    assert observed["jobs"] == []
    assert observed["elapsed_seconds"] == 1.0


def _smoke_task(tmp_path: Path) -> RoundTask:
    return RoundTask(
        dataset_id="zenodo-smoke",
        round=2,
        strategy="whole_image",
        fixed_image=tmp_path / "dataset" / "fixed.tif",
        moving_image=tmp_path / "dataset" / "moving.tif",
        registration_directory=tmp_path / "evidence" / "no-registration" / "round-002",
        log_path=tmp_path / "evidence" / "worker-round-002.json",
        voxel_spacing_xyz=(1.0, 1.0, 1.0),
        registration=RegistrationSpec(
            crop=False,
            linear=RegistrationStageSpec(type="TRSAA", accuracy="high"),
            nonlinear=RegistrationStageSpec(type="SyNOnly", accuracy="high"),
        ),
        native_threads=NativeThreadSpec(itk=1, blas=1),
        channels=(
            ChannelTask(
                source=tmp_path / "dataset" / "moving.tif",
                output=tmp_path / "evidence" / "no-registration" / "channel.tif",
                interpolation="linear",
            ),
        ),
        chunk_size_zyx=None,
        overlap_fraction=None,
    )


def test_environment_smoke_records_real_worker_facts_without_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _smoke_task(tmp_path)
    task.fixed_image.parent.mkdir()
    task.fixed_image.write_bytes(b"fixed")
    task.moving_image.write_bytes(b"moving")
    output_root = tmp_path / "output"
    scratch_root = tmp_path / "project" / "scratch"
    evidence = tmp_path / "evidence"
    output_root.mkdir()
    scratch_root.mkdir(parents=True)
    evidence.mkdir()
    monkeypatch.setenv("SLURM_JOB_ID", "913")
    monkeypatch.setenv("SLURMD_NODENAME", "slurm-node-a")
    worker = SimpleNamespace(
        address="tcp://10.0.0.2:43100",
        state=SimpleNamespace(total_resources={"round_slot": 1}),
        memory_manager=SimpleNamespace(memory_limit=500_000_000_000),
        get_current_task=lambda: "cyexm-zenodo-smoke-round-002",
    )

    def run(arguments: list[str], **_kwargs: Any) -> _Completed:
        if "job" in arguments:
            return _Completed(
                "JobId=913 JobState=RUNNING NumNodes=1 NumCPUs=72 "
                "NumTasks=1 NodeList=slurm-node-a "
                "ReqTRES=cpu=72,mem=500G,node=1 "
                "AllocTRES=cpu=72,mem=500G,node=1\n"
            )
        return _Completed(
            "NodeName=slurm-node-a State=ALLOCATED CPUTot=72 CPUAlloc=72 "
            "RealMemory=515000 AllocMem=512000\n"
        )

    result = run_environment_smoke(
        task,
        evidence_dir=str(evidence),
        dataset_root=str(task.fixed_image.parent),
        output_root=str(output_root),
        worker_local_directory=str(scratch_root),
        interface="ib0",
        _run=run,
        _get_worker=lambda: worker,
        _hostname=lambda: "node-a",
        _interface_facts=lambda _name: {
            "name": "ib0",
            "exists": True,
            "operstate": "up",
            "address": "00:11:22:33:44:55",
            "ipv4": "10.0.0.2",
        },
    )

    assert isinstance(result, RoundResult)
    assert result.status == "completed"
    assert result.slurm_job_id == "913"
    record = json.loads(task.log_path.read_text(encoding="utf-8"))
    assert record["round_slot"] == 1
    assert record["worker_memory_limit_bytes"] == 500_000_000_000
    assert record["job"]["JobId"] == "913"
    assert record["slurm_node_name"] == "slurm-node-a"
    assert record["node"]["State"] == "ALLOCATED"
    assert record["scratch_probe"] == {"created": True, "deleted": True}
    assert tuple(evidence.rglob("linear_transform_2.mat")) == ()
    assert tuple(evidence.rglob("nonlinear_warp_2.nii.gz")) == ()
    assert tuple(evidence.rglob("nonlinear_registered_2.tif")) == ()


def test_build_smoke_tasks_bounds_workers_and_uses_manifest_inputs(
    tmp_path: Path,
) -> None:
    archive_root = tmp_path / "archive"
    project_root = tmp_path / "project"
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    dataset, execution = _write_approved_inputs(archive_root, project_root)
    evidence = archive_root / "approved" / "evidence"
    evidence.mkdir()
    context = prepare_acceptance_context(
        dataset,
        execution,
        evidence,
        repository_root=repository_root,
        archive_root=archive_root,
        project_root=project_root,
        live_roots=(),
        argv=("pytest",),
        _git_commit=lambda _root: "a" * 40,
        _version=lambda name: name,
    )

    tasks = build_smoke_tasks(context, worker_count=1)

    assert len(tasks) == 1
    assert tasks[0].round == 2
    assert tasks[0].fixed_image == context.profile.paths.dataset_root / "fixed.tif"
    assert tasks[0].moving_image == context.profile.paths.dataset_root / "moving.tif"
    assert tasks[0].log_path == evidence / "worker-round-002.json"
    assert not tasks[0].registration_directory.exists()


def test_exclusive_policy_requires_partition_policy_or_explicit_directive() -> None:
    assert partition_is_exclusive(
        {"partition": {"OverSubscribe": "EXCLUSIVE"}}, directives=()
    )
    assert partition_is_exclusive(
        {"partition": {"OverSubscribe": "NO"}}, directives=("--exclusive",)
    )
    assert not partition_is_exclusive(
        {"partition": {"OverSubscribe": "NO"}}, directives=()
    )


def test_smoke_runner_serializes_without_worker_importing_test_package() -> None:
    import cloudpickle

    with smoke_module_by_value():
        encoded = base64.b64encode(cloudpickle.dumps(run_environment_smoke)).decode(
            "ascii"
        )
    source_root = Path(__file__).resolve().parents[2] / "src"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(source_root)

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import base64, cloudpickle, sys; "
            "runner = cloudpickle.loads(base64.b64decode(sys.argv[1])); "
            "assert runner.__name__ == 'run_environment_smoke'",
            encoded,
        ],
        cwd="/tmp",
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
