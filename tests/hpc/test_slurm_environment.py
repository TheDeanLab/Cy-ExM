from __future__ import annotations

import getpass
import hashlib
import json
from functools import partial
from typing import cast
from urllib.parse import urlsplit

import pytest

from cyexm.execution.slurm import SlurmExecutor
from cyexm.runtime import NATIVE_THREAD_ENV_VARS
from cyexm.workflow.models import RoundResult
from hpc.environment import (
    HpcAcceptanceContext,
    HpcAcceptanceError,
    build_smoke_tasks,
    collect_interface_facts,
    collect_partition_snapshot,
    collect_path_fact,
    collect_queue_snapshot,
    parse_slurm_tres_memory_bytes,
    partition_is_exclusive,
    run_environment_smoke,
    shared_path_identity,
    smoke_module_by_value,
    validate_rendered_job_script,
    validate_scheduler_interface,
    wait_for_queue_empty,
    write_evidence_json,
    write_evidence_text,
)

pytestmark = pytest.mark.hpc


def _memory_bytes(value: str) -> int:
    units = {
        "KB": 1_000,
        "MB": 1_000_000,
        "GB": 1_000_000_000,
        "TB": 1_000_000_000_000,
    }
    normalized = value.strip().upper()
    for unit, multiplier in units.items():
        if normalized.endswith(unit):
            return int(normalized.removesuffix(unit)) * multiplier
    raise AssertionError(f"unsupported acceptance memory value: {value!r}")


def _assert_partition(
    context: HpcAcceptanceContext,
    snapshot: dict[str, object],
) -> None:
    spec = context.profile.slurm
    assert spec is not None
    partition = cast(dict[str, str], snapshot["partition"])
    assert partition["PartitionName"] == spec.queue
    assert partition["State"].upper() == "UP"
    assert partition_is_exclusive(snapshot, directives=spec.job_extra_directives), (
        "The selected partition is not exclusive. Add an approved --exclusive "
        "directive to the strict profile, rerun portable profile tests, and retry."
    )
    requested_mib = _memory_bytes(spec.memory) // (1024 * 1024)
    rows = cast(list[dict[str, object]], snapshot["rows"])
    assert rows
    for row in rows:
        assert cast(str, row["partition"]).removesuffix("*") == spec.queue
        assert cast(str, row["availability"]).lower() == "up"
        assert cast(int, row["cpus"]) >= spec.cores_per_worker
        assert cast(int, row["memory_mb"]) >= requested_mib


def _assert_worker_records(
    context: HpcAcceptanceContext,
    results: tuple[RoundResult, ...],
    records: list[dict[str, object]],
    coordinator_paths: dict[str, dict[str, object]],
) -> None:
    spec = context.profile.slurm
    assert spec is not None
    expected_memory = _memory_bytes(spec.memory)
    expected_ids = {result.slurm_job_id for result in results}
    expected_hosts = {result.hostname for result in results}
    assert None not in expected_ids
    assert len(expected_ids) == len(results)
    assert len(expected_hosts) == len(results)
    assert len(records) == len(results)

    for record in records:
        job_id = cast(str, record["slurm_job_id"])
        hostname = cast(str, record["hostname"])
        node_name = cast(str, record["slurm_node_name"])
        assert job_id in expected_ids
        assert hostname in expected_hosts
        assert record["round_slot"] == 1
        assert record["worker_memory_limit_bytes"] == expected_memory

        job = cast(dict[str, str], record["job"])
        assert job["JobId"] == job_id
        assert job["JobState"].upper() == "RUNNING"
        assert job["Partition"] == spec.queue
        assert int(job["NumNodes"]) == 1
        assert int(job["NumCPUs"]) == spec.cores_per_worker
        assert int(job["NumTasks"]) == 1
        assert node_name in job["NodeList"]
        assert "node=1" in job["AllocTRES"]
        assert parse_slurm_tres_memory_bytes(job["ReqTRES"]) >= expected_memory
        assert parse_slurm_tres_memory_bytes(job["AllocTRES"]) >= expected_memory

        node = cast(dict[str, str], record["node"])
        assert node["NodeName"] == node_name
        assert "ALLOCATED" in node["State"].upper()
        assert "MIXED" not in node["State"].upper()
        assert int(node["CPUAlloc"]) == int(node["CPUTot"])
        assert int(node["RealMemory"]) >= expected_memory // (1024 * 1024)
        assert int(node["AllocMem"]) >= expected_memory // (1024 * 1024)

        interface = cast(dict[str, object], record["interface"])
        assert interface == {
            "name": spec.interface,
            "exists": True,
            "operstate": "up",
            "address": cast(str, interface["address"]),
            "ipv4": cast(str, interface["ipv4"]),
        }
        assert interface["address"]
        assert interface["ipv4"]
        worker_host = urlsplit(cast(str, record["worker_address"])).hostname
        assert worker_host == interface["ipv4"]

        paths = cast(dict[str, dict[str, object]], record["paths"])
        for path_fact in paths.values():
            assert path_fact["exists"] is True
        assert paths["dataset_root"]["resolved"] == str(
            context.profile.paths.dataset_root
        )
        assert paths["output_root"]["resolved"] == str(
            context.profile.paths.output_root
        )
        assert paths["evidence_dir"]["resolved"] == str(context.evidence_dir)
        for role in (
            "dataset_root",
            "output_root",
            "worker_local_directory",
            "evidence_dir",
        ):
            assert shared_path_identity(paths[role]) == shared_path_identity(
                coordinator_paths[role]
            )
        dataset_identity = shared_path_identity(coordinator_paths["dataset_root"])
        assert shared_path_identity(paths["fixed_image"]) == dataset_identity
        assert shared_path_identity(paths["moving_image"]) == dataset_identity
        assert record["scratch_probe"] == {"created": True, "deleted": True}


def _assert_executor_evidence(
    context: HpcAcceptanceContext,
    evidence: dict[str, object],
    worker_count: int,
    scheduler_interface: dict[str, object],
) -> None:
    spec = context.profile.slurm
    assert spec is not None
    requested = cast(dict[str, object], evidence["requested"])
    assert requested["workers"] == worker_count
    assert requested["configured_workers"] == spec.workers
    rendered = cast(dict[str, str], evidence["rendered"])
    script = rendered["rendered_job_script_template"]
    validate_rendered_job_script(script, context.profile)
    assert "round_slot=1" in script
    for index, name in enumerate(NATIVE_THREAD_ENV_VARS):
        count = (
            context.profile.native_threads.itk
            if index == 0
            else context.profile.native_threads.blas
        )
        assert f"export {name}={count}" in script

    observed = cast(dict[str, object], evidence["observed"])
    validate_scheduler_interface(
        cast(str, observed["scheduler_address"]),
        spec.scheduler_interface,
        scheduler_interface,
    )
    assert observed["coordinator_versions"] == context.package_versions
    assert len(cast(list[str], observed["connected_workers"])) == worker_count
    probes = cast(list[dict[str, object]], observed["worker_probes"])
    assert len(probes) == worker_count
    expected_threads = sorted(
        [
            name,
            str(
                context.profile.native_threads.itk
                if index == 0
                else context.profile.native_threads.blas
            ),
        ]
        for index, name in enumerate(NATIVE_THREAD_ENV_VARS)
    )
    expected_versions = sorted(context.package_versions.items())
    for probe in probes:
        assert probe["ok"] is True
        assert probe["dataset_visible"] is True
        assert probe["output_writable"] is True
        assert probe["scratch_usable"] is True
        assert probe["native_threads"] == expected_threads
        assert probe["package_versions"] == [list(item) for item in expected_versions]
        assert probe["memory_limit_bytes"] == _memory_bytes(spec.memory)
    assert len({cast(str, probe["slurm_job_id"]) for probe in probes}) == worker_count
    assert len({cast(str, probe["hostname"]) for probe in probes}) == worker_count

    assert evidence["startup"] == {
        "phase": "complete",
        "outcome": "succeeded",
        "scale_attempted": True,
        "scale_succeeded": True,
    }
    cleanup = cast(dict[str, object], evidence["cleanup"])
    assert cleanup["order"] == ["client", "cluster", "configuration"]
    for resource in ("client", "cluster", "configuration"):
        assert cleanup[resource] == {"attempted": True, "succeeded": True}


def test_real_slurm_environment_acceptance(
    hpc_context: HpcAcceptanceContext,
) -> None:
    context = hpc_context
    spec = context.profile.slurm
    assert spec is not None
    assert spec.processes_per_worker == 1
    worker_count = min(2, spec.workers)
    scratch = context.profile.paths.worker_local_directory
    assert scratch is not None
    scheduler_interface = collect_interface_facts(spec.scheduler_interface)
    coordinator_paths = {
        "dataset_root": collect_path_fact(context.profile.paths.dataset_root),
        "output_root": collect_path_fact(context.profile.paths.output_root),
        "worker_local_directory": collect_path_fact(scratch),
        "evidence_dir": collect_path_fact(context.evidence_dir),
    }

    partition = collect_partition_snapshot(spec.queue)
    write_evidence_json(context.evidence_dir / "partition-snapshot.json", partition)
    _assert_partition(context, partition)

    user = getpass.getuser()
    before = collect_queue_snapshot(spec.job_name, user=user)
    write_evidence_json(context.evidence_dir / "queue-before.json", before)
    assert before["jobs"] == [], (
        "Refusing to submit while the current user already has jobs with the "
        f"acceptance name {spec.job_name!r}"
    )

    tasks = build_smoke_tasks(context, worker_count=worker_count)
    runner = partial(
        run_environment_smoke,
        evidence_dir=str(context.evidence_dir),
        dataset_root=str(context.profile.paths.dataset_root),
        output_root=str(context.profile.paths.output_root),
        worker_local_directory=str(context.profile.paths.worker_local_directory),
        interface=spec.interface,
    )
    executor = SlurmExecutor(context.profile, _runner=runner)
    results: tuple[RoundResult, ...] = ()
    after: dict[str, object] = {}
    execution_error: BaseException | None = None
    with smoke_module_by_value():
        try:
            results = executor.execute(tasks)
        except BaseException as error:
            execution_error = error
        finally:
            execution_evidence = dict(executor.execution_evidence)
            observed = execution_evidence.get("observed", {})
            probes = (
                observed.get("worker_probes", []) if isinstance(observed, dict) else []
            )
            owned_job_ids = {
                job_id
                for job_id in (
                    *(result.slurm_job_id for result in results),
                    *(
                        probe.get("slurm_job_id")
                        for probe in probes
                        if isinstance(probe, dict)
                    ),
                )
                if isinstance(job_id, str) and job_id
            }
            after = wait_for_queue_empty(
                spec.job_name,
                user=user,
                job_ids=tuple(sorted(owned_job_ids)),
            )
            write_evidence_json(context.evidence_dir / "queue-after.json", after)
            write_evidence_json(
                context.evidence_dir / "slurm-execution-evidence.json",
                execution_evidence,
            )
    if after["outcome"] != "empty":
        raise HpcAcceptanceError(
            "owned SLURM cleanup could not be proven; inspect queue-after.json"
        ) from execution_error
    if execution_error is not None:
        raise execution_error

    evidence = dict(executor.execution_evidence)
    records = [json.loads(task.log_path.read_text(encoding="utf-8")) for task in tasks]
    script = cast(dict[str, str], evidence["rendered"])["rendered_job_script_template"]
    script_hash = hashlib.sha256(script.encode("utf-8")).hexdigest()
    write_evidence_text(context.evidence_dir / "submitted-job-script.sh", script)
    write_evidence_json(
        context.evidence_dir / "environment-smoke.json",
        {
            "schema_version": 1,
            "worker_count": worker_count,
            "job_ids": sorted(cast(str, result.slurm_job_id) for result in results),
            "hostnames": sorted(result.hostname for result in results),
            "submitted_job_script_sha256": script_hash,
            "results": [result.to_dict() for result in results],
            "worker_records": records,
            "coordinator_scheduler_interface": scheduler_interface,
            "coordinator_paths": coordinator_paths,
            "cleanup_queue": after,
        },
    )

    assert len(results) == worker_count
    assert all(result.status == "completed" for result in results)
    cleanup_query = cast(dict[str, object], after["query"])
    assert cleanup_query["job_ids"] == sorted(
        cast(str, result.slurm_job_id) for result in results
    )
    _assert_executor_evidence(context, evidence, worker_count, scheduler_interface)
    _assert_worker_records(context, results, records, coordinator_paths)
    for task in tasks:
        assert not task.registration_directory.exists(), (
            "environment smoke unexpectedly created registration output"
        )
