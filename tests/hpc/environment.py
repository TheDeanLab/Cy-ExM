"""Fail-closed helpers for explicit real-SLURM acceptance evidence."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

from cyexm.config import (
    DatasetManifest,
    ExecutionProfile,
    load_dataset_manifest,
    load_execution_profile,
)
from cyexm.execution.slurm import validate_slurm_job_script
from cyexm.io.atomic import atomic_write
from cyexm.runtime.threads import BLAS_THREAD_ENV_VARS, ITK_THREAD_ENV_VAR
from cyexm.workflow.models import (
    ChannelTask,
    ReferenceChannelTask,
    ReferenceTask,
    RoundResult,
    RoundTask,
    WorkflowTask,
)

ACCEPTANCE_PACKAGES = (
    "cy-exm",
    "dask",
    "distributed",
    "dask-jobqueue",
    "antspyx",
    "numpy",
    "scipy",
    "tifffile",
)

_SMOKE_PICKLE_LOCK = threading.RLock()
_SMOKE_PICKLE_DEPTH = 0
_SMOKE_PICKLE_WAS_REGISTERED = False


class HpcAcceptanceError(RuntimeError):
    """An explicit HPC acceptance prerequisite was not satisfied."""


@dataclass(frozen=True, slots=True)
class HpcAcceptanceContext:
    """Validated operator inputs and their public configuration models."""

    dataset_path: Path
    execution_path: Path
    evidence_dir: Path
    manifest: DatasetManifest
    profile: ExecutionProfile
    package_versions: dict[str, str]
    git_commit: str


def require_cli_paths(
    dataset: str | Path | None,
    execution: str | Path | None,
    evidence: str | Path | None,
) -> tuple[Path, Path, Path]:
    """Require all operator-owned inputs before any scheduler interaction."""
    values = (
        ("--cyexm-dataset", dataset),
        ("--cyexm-execution", execution),
        ("--cyexm-evidence-dir", evidence),
    )
    parsed: list[Path] = []
    for option, value in values:
        if value is None or not str(value).strip():
            raise HpcAcceptanceError(f"explicit HPC acceptance requires {option} PATH")
        parsed.append(Path(value))
    return parsed[0], parsed[1], parsed[2]


def require_explicit_hpc_selection(arguments: Sequence[str]) -> None:
    """Require the process invocation itself to select exactly ``hpc`` tests."""
    selected: list[str] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "-m":
            if index + 1 >= len(arguments):
                break
            selected.append(arguments[index + 1])
            index += 2
            continue
        if argument.startswith("-m="):
            selected.append(argument[3:])
        elif argument.startswith("-m") and argument != "-m":
            selected.append(argument[2:])
        index += 1
    if selected != ["hpc"]:
        raise HpcAcceptanceError(
            "explicit HPC acceptance requires the actual pytest invocation to "
            "contain exactly '-m hpc'"
        )


def require_hpc_opt_in(value: object) -> None:
    """Require the exact destructive-test opt-in before any fixture action."""
    if value is not True:
        raise HpcAcceptanceError(
            "explicit HPC acceptance requires the exact --run-hpc opt-in"
        )


def require_coordinator_native_threads(
    profile: ExecutionProfile,
    *,
    environ: Mapping[str, str] = os.environ,
) -> None:
    """Require native settings before pytest can load scientific runtimes."""
    expected = {
        ITK_THREAD_ENV_VAR: str(profile.native_threads.itk),
        **{name: str(profile.native_threads.blas) for name in BLAS_THREAD_ENV_VARS},
    }
    mismatches = [
        f"{name}={environ.get(name)!r}, expected {value!r}"
        for name, value in expected.items()
        if environ.get(name) != value
    ]
    if mismatches:
        raise HpcAcceptanceError(
            "coordinator native threads must be exported before HPC pytest starts: "
            + "; ".join(mismatches)
        )


def _is_disjoint(first: Path, second: Path) -> bool:
    return not first.is_relative_to(second) and not second.is_relative_to(first)


def _require_under(path: Path, root: Path, role: str) -> None:
    if not path.is_relative_to(root):
        raise HpcAcceptanceError(f"{role} must resolve below {root}; got {path}")


def _resolved_file(path: Path, role: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise HpcAcceptanceError(f"{role} must be an existing file: {path}") from error
    if not resolved.is_file():
        raise HpcAcceptanceError(f"{role} must be an existing file: {resolved}")
    return resolved


def _resolved_directory(path: Path, role: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise HpcAcceptanceError(
            f"{role} must be an existing directory: {path}"
        ) from error
    if not resolved.is_dir():
        raise HpcAcceptanceError(f"{role} must be an existing directory: {resolved}")
    return resolved


def _git_head(repository_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return completed.stdout.strip()


def _package_versions(version: Callable[[str], str]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for name in ACCEPTANCE_PACKAGES:
        try:
            value = version(name)
        except Exception as error:
            raise HpcAcceptanceError(
                f"could not inspect required package {name!r}: "
                f"{type(error).__name__}: {error}"
            ) from error
        if type(value) is not str or not value:
            raise HpcAcceptanceError(
                f"required package {name!r} returned invalid version {value!r}"
            )
        observed[name] = value
    return observed


def _canonical_timestamp(now: datetime) -> str:
    if now.tzinfo is None or now.utcoffset() is None:
        raise HpcAcceptanceError("acceptance timestamp must be timezone-aware")
    return now.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _write_json_atomic(path: Path, value: object) -> None:
    encoded = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"

    def writer(temporary: Path) -> None:
        temporary.write_text(encoded, encoding="utf-8")

    def validator(temporary: Path) -> bool:
        try:
            return json.loads(temporary.read_text(encoding="utf-8")) == value
        except (OSError, UnicodeError, json.JSONDecodeError):
            return False

    atomic_write(path, writer, validator)


def write_evidence_json(path: Path, value: object) -> None:
    """Atomically persist one machine-readable acceptance evidence document."""
    _write_json_atomic(path, value)


def write_evidence_text(path: Path, value: str) -> None:
    """Atomically persist one exact text evidence artifact."""
    if type(value) is not str or not value:
        raise HpcAcceptanceError("text evidence must be a nonempty built-in string")

    def writer(temporary: Path) -> None:
        temporary.write_text(value, encoding="utf-8")

    def validator(temporary: Path) -> bool:
        try:
            return temporary.read_text(encoding="utf-8") == value
        except (OSError, UnicodeError):
            return False

    atomic_write(path, writer, validator)


def parse_scontrol_record(output: str) -> dict[str, str]:
    """Parse one ``scontrol --oneliner`` record without losing embedded equals."""
    if type(output) is not str or not output.strip():
        raise HpcAcceptanceError("scontrol returned an empty record")
    try:
        tokens = shlex.split(output.strip())
    except ValueError as error:
        raise HpcAcceptanceError(f"could not parse scontrol record: {error}") from error
    record: dict[str, str] = {}
    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if not key or key in record:
            raise HpcAcceptanceError(
                f"scontrol record contains an invalid or duplicate field: {token!r}"
            )
        record[key] = value
    if not record:
        raise HpcAcceptanceError("scontrol record contained no key/value facts")
    return record


def parse_slurm_tres_memory_bytes(value: str) -> int:
    """Decode the allocated ``mem=`` TRES using SLURM's binary suffixes."""
    if type(value) is not str:
        raise HpcAcceptanceError(f"SLURM TRES must be a built-in string; got {value!r}")
    memory = next(
        (
            item.removeprefix("mem=")
            for item in value.split(",")
            if item.startswith("mem=")
        ),
        None,
    )
    if not memory:
        raise HpcAcceptanceError(f"SLURM TRES contains no memory fact: {value!r}")
    suffixes = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    suffix = memory[-1].upper()
    multiplier = suffixes.get(suffix, 1)
    number = memory[:-1] if suffix in suffixes else memory
    try:
        amount = int(number, 10)
    except ValueError as error:
        raise HpcAcceptanceError(
            f"SLURM TRES memory must be an integer with K/M/G/T suffix: {memory!r}"
        ) from error
    if amount < 1:
        raise HpcAcceptanceError(f"SLURM TRES memory must be positive: {memory!r}")
    return amount * multiplier


def _resource_text(
    record: Mapping[str, str],
    field: str,
    location: str,
) -> str:
    value = record.get(field)
    if type(value) is not str or not value:
        raise HpcAcceptanceError(f"{location}.{field} must be a nonempty string")
    return value


def _resource_integer(
    record: Mapping[str, str],
    field: str,
    location: str,
) -> int:
    value = _resource_text(record, field, location)
    try:
        parsed = int(value, 10)
    except ValueError as error:
        raise HpcAcceptanceError(
            f"{location}.{field} must be a positive integer; got {value!r}"
        ) from error
    if parsed < 1:
        raise HpcAcceptanceError(
            f"{location}.{field} must be a positive integer; got {value!r}"
        )
    return parsed


def _tres_facts(value: str, location: str) -> dict[str, str]:
    if type(value) is not str or not value:
        raise HpcAcceptanceError(f"{location} must be a nonempty built-in string")
    facts: dict[str, str] = {}
    for item in value.split(","):
        if "=" not in item:
            raise HpcAcceptanceError(f"{location} contains an invalid fact: {item!r}")
        key, fact = item.split("=", 1)
        if not key or not fact or key in facts:
            raise HpcAcceptanceError(f"{location} contains an invalid fact: {item!r}")
        facts[key] = fact
    return facts


def _tres_count(facts: Mapping[str, str], field: str, location: str) -> int:
    value = facts.get(field)
    try:
        parsed = int(value, 10) if type(value) is str else 0
    except ValueError as error:
        raise HpcAcceptanceError(
            f"{location} {field} must be a positive integer; got {value!r}"
        ) from error
    if parsed < 1:
        raise HpcAcceptanceError(
            f"{location} {field} must be a positive integer; got {value!r}"
        )
    return parsed


def _validate_tres_resources(
    value: str,
    *,
    location: str,
    expected_cores: int,
    expected_nodes: int | None,
    expected_memory_bytes: int | None,
) -> None:
    facts = _tres_facts(value, location)
    cores = _tres_count(facts, "cpu", location)
    if cores != expected_cores:
        raise HpcAcceptanceError(
            f"{location} cpu must equal {expected_cores}; got {cores}"
        )
    if expected_nodes is not None:
        nodes = _tres_count(facts, "node", location)
        if nodes != expected_nodes:
            raise HpcAcceptanceError(
                f"{location} node must equal {expected_nodes}; got {nodes}"
            )
    if expected_memory_bytes is not None:
        memory_bytes = parse_slurm_tres_memory_bytes(value)
        if memory_bytes < expected_memory_bytes:
            raise HpcAcceptanceError(
                f"{location} memory must cover {expected_memory_bytes} bytes; "
                f"got {memory_bytes}"
            )


def validate_slurm_job_node_resources(
    job: Mapping[str, str],
    node: Mapping[str, str],
    *,
    expected_job_id: str,
    expected_node_name: str,
    expected_partition: str,
    expected_cores: int,
    expected_memory: str,
    expected_memory_bytes: int,
) -> None:
    """Validate version-compatible SLURM job request and node allocation facts."""
    expected_memory_decoded = parse_slurm_tres_memory_bytes(f"mem={expected_memory}")
    if expected_memory_decoded != expected_memory_bytes:
        raise HpcAcceptanceError(
            "expected SLURM memory spelling and byte count disagree: "
            f"{expected_memory!r} != {expected_memory_bytes}"
        )

    exact_job_facts = {
        "JobId": expected_job_id,
        "Partition": expected_partition,
        "NodeList": expected_node_name,
    }
    for field, expected in exact_job_facts.items():
        observed = _resource_text(job, field, "job")
        if observed != expected:
            raise HpcAcceptanceError(
                f"job.{field} must equal {expected!r}; got {observed!r}"
            )
    if _resource_text(job, "JobState", "job").upper() != "RUNNING":
        raise HpcAcceptanceError("job.JobState must be RUNNING during worker smoke")
    for field, expected in (
        ("NumNodes", 1),
        ("NumCPUs", expected_cores),
        ("NumTasks", 1),
    ):
        observed = _resource_integer(job, field, "job")
        if observed != expected:
            raise HpcAcceptanceError(
                f"job.{field} must equal {expected}; got {observed}"
            )
    _validate_tres_resources(
        _resource_text(job, "TRES", "job"),
        location="job.TRES",
        expected_cores=expected_cores,
        expected_nodes=1,
        expected_memory_bytes=None,
    )

    job_memory_proven = False
    if "MinMemoryNode" in job:
        minimum = _resource_text(job, "MinMemoryNode", "job")
        if minimum != expected_memory:
            raise HpcAcceptanceError(
                f"job.MinMemoryNode must equal {expected_memory!r}; got {minimum!r}"
            )
        if parse_slurm_tres_memory_bytes(f"mem={minimum}") != expected_memory_bytes:
            raise HpcAcceptanceError(
                "job.MinMemoryNode does not equal the normalized byte request"
            )
        job_memory_proven = True
    for field in ("ReqTRES", "AllocTRES"):
        if field not in job:
            continue
        _validate_tres_resources(
            _resource_text(job, field, "job"),
            location=f"job.{field}",
            expected_cores=expected_cores,
            expected_nodes=1,
            expected_memory_bytes=expected_memory_bytes,
        )
        job_memory_proven = True

    observed_node_name = _resource_text(node, "NodeName", "node")
    if observed_node_name != expected_node_name:
        raise HpcAcceptanceError(
            f"node.NodeName must equal {expected_node_name!r}; "
            f"got {observed_node_name!r}"
        )
    node_state = _resource_text(node, "State", "node").upper()
    if "ALLOCATED" not in node_state or "MIXED" in node_state:
        raise HpcAcceptanceError(
            f"node.State must be fully ALLOCATED and not MIXED; got {node_state!r}"
        )
    for field in ("CPUTot", "CPUAlloc"):
        observed = _resource_integer(node, field, "node")
        if observed != expected_cores:
            raise HpcAcceptanceError(
                f"node.{field} must equal {expected_cores}; got {observed}"
            )

    real_memory_bytes = _resource_integer(node, "RealMemory", "node") * 1024**2
    if real_memory_bytes < expected_memory_bytes:
        raise HpcAcceptanceError(
            f"node.RealMemory must cover {expected_memory_bytes} bytes; "
            f"got {real_memory_bytes}"
        )
    if "CfgTRES" in node:
        _validate_tres_resources(
            _resource_text(node, "CfgTRES", "node"),
            location="node.CfgTRES",
            expected_cores=expected_cores,
            expected_nodes=None,
            expected_memory_bytes=expected_memory_bytes,
        )

    node_allocation_proven = False
    if "AllocTRES" in node:
        _validate_tres_resources(
            _resource_text(node, "AllocTRES", "node"),
            location="node.AllocTRES",
            expected_cores=expected_cores,
            expected_nodes=None,
            expected_memory_bytes=expected_memory_bytes,
        )
        node_allocation_proven = True
    if "AllocMem" in node:
        allocated_text = _resource_text(node, "AllocMem", "node")
        try:
            allocated_mib = int(allocated_text, 10)
        except ValueError as error:
            raise HpcAcceptanceError(
                f"node.AllocMem must be a nonnegative integer; got {allocated_text!r}"
            ) from error
        if allocated_mib < 0:
            raise HpcAcceptanceError(
                f"node.AllocMem must be a nonnegative integer; got {allocated_text!r}"
            )
        if allocated_mib > 0:
            allocated_bytes = allocated_mib * 1024**2
            if allocated_bytes < expected_memory_bytes:
                raise HpcAcceptanceError(
                    f"node.AllocMem must cover {expected_memory_bytes} bytes; "
                    f"got {allocated_bytes}"
                )
            node_allocation_proven = True
    if not job_memory_proven and not node_allocation_proven:
        raise HpcAcceptanceError(
            "SLURM evidence contains neither an authoritative memory request "
            "nor a node memory allocation fact"
        )


def _run_read_only(
    arguments: list[str],
    *,
    run: Callable[..., Any],
) -> dict[str, object]:
    executable = arguments[0]
    if executable not in {"sinfo", "scontrol", "squeue"}:
        raise HpcAcceptanceError(
            f"HPC evidence command is not read-only allowlisted: {executable!r}"
        )
    if executable == "scontrol" and not (
        arguments[1:] == ["--version"]
        or (len(arguments) >= 4 and arguments[1:3] == ["--oneliner", "show"])
    ):
        raise HpcAcceptanceError(
            f"scontrol evidence command must be read-only show/version: {arguments!r}"
        )
    try:
        completed = run(
            arguments,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as error:
        raise HpcAcceptanceError(
            f"read-only command failed to start: {arguments!r}: "
            f"{type(error).__name__}: {error}"
        ) from error
    stdout = completed.stdout
    stderr = completed.stderr
    returncode = completed.returncode
    if (
        type(stdout) is not str
        or type(stderr) is not str
        or type(returncode) is not int
    ):
        raise HpcAcceptanceError(
            f"read-only command returned malformed evidence: {arguments!r}"
        )
    result: dict[str, object] = {
        "argv": list(arguments),
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
    }
    if returncode != 0:
        raise HpcAcceptanceError(
            f"read-only command exited {returncode}: {arguments!r}: {stderr.strip()}"
        )
    return result


def _configuration_facts(output: str) -> dict[str, str]:
    facts: dict[str, str] = {}
    for line in output.splitlines():
        if " = " not in line:
            continue
        key, value = line.split(" = ", 1)
        key = key.strip()
        if key and key not in facts:
            facts[key] = value.strip()
    return facts


def _sinfo_rows(output: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    fields = (
        "partition",
        "availability",
        "state",
        "time_limit",
        "nodes",
        "cpus",
        "memory_mb",
        "oversubscribe",
        "features",
        "node_list",
    )
    for line_number, line in enumerate(output.splitlines(), start=1):
        if not line.strip():
            continue
        values = [value.strip() for value in line.split("|")]
        if len(values) != len(fields):
            raise HpcAcceptanceError(
                f"sinfo row {line_number} must contain {len(fields)} fields: {line!r}"
            )
        row: dict[str, object] = dict(zip(fields, values, strict=True))
        for numeric in ("nodes", "cpus", "memory_mb"):
            try:
                parsed = int(cast(str, row[numeric]), 10)
            except ValueError as error:
                raise HpcAcceptanceError(
                    f"sinfo {numeric} must be an integer in row {line_number}"
                ) from error
            if parsed < 1:
                raise HpcAcceptanceError(
                    f"sinfo {numeric} must be positive in row {line_number}"
                )
            row[numeric] = parsed
        rows.append(row)
    if not rows:
        raise HpcAcceptanceError("sinfo returned no rows for the selected partition")
    return rows


def collect_partition_snapshot(
    queue: str,
    *,
    _run: Callable[..., Any] = subprocess.run,
) -> dict[str, object]:
    """Collect only read-only current partition and controller facts."""
    commands = [
        _run_read_only(
            [
                "sinfo",
                "--noheader",
                "--partition",
                queue,
                "--format=%P|%a|%T|%l|%D|%c|%m|%h|%f|%N",
            ],
            run=_run,
        ),
        _run_read_only(
            ["scontrol", "--oneliner", "show", "partition", queue],
            run=_run,
        ),
        _run_read_only(
            ["scontrol", "--oneliner", "show", "config"],
            run=_run,
        ),
        _run_read_only(["sinfo", "--version"], run=_run),
    ]
    partition = parse_scontrol_record(cast(str, commands[1]["stdout"]))
    if partition.get("PartitionName") != queue:
        raise HpcAcceptanceError(
            "scontrol partition identity does not match the execution profile: "
            f"{partition.get('PartitionName')!r} != {queue!r}"
        )
    return {
        "schema_version": 1,
        "queue": queue,
        "partition": partition,
        "rows": _sinfo_rows(cast(str, commands[0]["stdout"])),
        "configuration": _configuration_facts(cast(str, commands[2]["stdout"])),
        "slurm_version": cast(str, commands[3]["stdout"]).strip(),
        "commands": commands,
    }


def collect_queue_snapshot(
    job_name: str,
    *,
    user: str,
    job_ids: Sequence[str] = (),
    _run: Callable[..., Any] = subprocess.run,
) -> dict[str, object]:
    """Read only the current user's exact harness name and recorded job IDs."""
    raw_job_ids = tuple(job_ids)
    if any(type(job_id) is not str or not job_id for job_id in raw_job_ids):
        raise HpcAcceptanceError(f"owned SLURM job IDs are invalid: {raw_job_ids!r}")
    owned_ids = tuple(sorted(set(raw_job_ids)))
    arguments = [
        "squeue",
        "--noheader",
        "--user",
        user,
        "--name",
        job_name,
    ]
    arguments.append("--format=%i|%j|%T|%N")
    command = _run_read_only(
        arguments,
        run=_run,
    )
    observed_jobs: list[dict[str, str]] = []
    for line_number, line in enumerate(cast(str, command["stdout"]).splitlines(), 1):
        if not line.strip():
            continue
        values = [value.strip() for value in line.split("|")]
        if len(values) != 4 or not all(values[:3]):
            raise HpcAcceptanceError(f"squeue row {line_number} is malformed: {line!r}")
        observed_jobs.append(
            dict(
                zip(
                    ("job_id", "job_name", "state", "nodes"),
                    values,
                    strict=True,
                )
            )
        )
    owned_id_set = set(owned_ids)
    jobs = [
        job
        for job in observed_jobs
        if job["job_name"] == job_name
        and (not owned_id_set or job["job_id"] in owned_id_set)
    ]
    unrelated_jobs = [job for job in observed_jobs if job not in jobs]
    return {
        "schema_version": 1,
        "query": {
            "job_name": job_name,
            "user": user,
            "job_ids": list(owned_ids),
        },
        "jobs": jobs,
        "unrelated_jobs": unrelated_jobs,
        "command": command,
    }


def wait_for_queue_empty(
    job_name: str,
    *,
    user: str,
    job_ids: Sequence[str] = (),
    timeout_seconds: float = 60.0,
    poll_seconds: float = 1.0,
    _run: Callable[..., Any] = subprocess.run,
    _monotonic: Callable[[], float] = time.monotonic,
    _sleep: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Return durable bounded cleanup observations, including failures."""
    if timeout_seconds <= 0.0 or poll_seconds < 0.0:
        raise HpcAcceptanceError("queue cleanup timeout must be positive and bounded")
    started = _monotonic()
    deadline = started + timeout_seconds
    attempts: list[dict[str, object]] = []
    raw_job_ids = tuple(job_ids)
    if any(type(job_id) is not str or not job_id for job_id in raw_job_ids):
        raise HpcAcceptanceError(f"owned SLURM job IDs are invalid: {raw_job_ids!r}")
    owned_ids = tuple(sorted(set(raw_job_ids)))
    query = {
        "job_name": job_name,
        "user": user,
        "job_ids": list(owned_ids),
    }
    while True:
        try:
            snapshot = collect_queue_snapshot(
                job_name,
                user=user,
                job_ids=owned_ids,
                _run=_run,
            )
        except Exception as error:
            now = _monotonic()
            attempts.append(
                {
                    "attempt": len(attempts) + 1,
                    "elapsed_seconds": float(max(0.0, now - started)),
                    "outcome": "query_error",
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                }
            )
            return {
                "schema_version": 1,
                "outcome": "query_error",
                "query": query,
                "jobs": [],
                "attempts": attempts,
                "elapsed_seconds": float(max(0.0, now - started)),
            }
        now = _monotonic()
        attempts.append(
            {
                "attempt": len(attempts) + 1,
                "elapsed_seconds": float(max(0.0, now - started)),
                "outcome": "empty" if snapshot["jobs"] == [] else "jobs_remain",
                "snapshot": snapshot,
            }
        )
        if snapshot["jobs"] == []:
            return {
                "schema_version": 1,
                "outcome": "empty",
                "query": query,
                "jobs": [],
                "attempts": attempts,
                "elapsed_seconds": float(max(0.0, now - started)),
            }
        if now >= deadline:
            return {
                "schema_version": 1,
                "outcome": "timeout",
                "query": query,
                "jobs": snapshot["jobs"],
                "attempts": attempts,
                "elapsed_seconds": float(max(0.0, now - started)),
            }
        _sleep(min(poll_seconds, max(0.0, deadline - now)))


def partition_is_exclusive(
    snapshot: dict[str, object],
    *,
    directives: Sequence[str],
) -> bool:
    """Return whether policy or an explicit approved directive owns whole nodes."""
    partition = snapshot.get("partition")
    policy = partition.get("OverSubscribe", "") if isinstance(partition, dict) else ""
    if type(policy) is str and policy.upper().startswith("EXCLUSIVE"):
        return True
    for directive in directives:
        try:
            tokens = shlex.split(directive)
        except ValueError:
            continue
        if any(
            token == "--exclusive" or token.startswith("--exclusive=")
            for token in tokens
        ):
            return True
    return False


def validate_rendered_job_script(script: str, profile: ExecutionProfile) -> int:
    """Expose the production pre-scale template check as acceptance policy."""
    try:
        return validate_slurm_job_script(script, profile)
    except (TypeError, ValueError) as error:
        raise HpcAcceptanceError(
            f"rendered job script validation failed: {error}"
        ) from error


def validate_scheduler_interface(
    scheduler_address: str,
    expected_name: str,
    facts: Mapping[str, object],
) -> None:
    """Prove the scheduler address is bound to the requested live interface."""
    if (
        facts.get("name") != expected_name
        or facts.get("exists") is not True
        or facts.get("operstate") != "up"
    ):
        raise HpcAcceptanceError(
            f"scheduler interface {expected_name!r} is missing or not up: {facts!r}"
        )
    ipv4 = facts.get("ipv4")
    if type(ipv4) is not str or not ipv4:
        raise HpcAcceptanceError(
            f"scheduler interface {expected_name!r} has no IPv4 address"
        )
    host = urlsplit(scheduler_address).hostname
    if host != ipv4:
        raise HpcAcceptanceError(
            f"scheduler address {scheduler_address!r} does not use requested "
            f"interface {expected_name!r} at {ipv4!r}"
        )


def build_smoke_tasks(
    context: HpcAcceptanceContext,
    *,
    worker_count: int,
) -> tuple[RoundTask, ...]:
    """Build bounded manifest-derived tasks for a no-registration executor probe."""
    spec = context.profile.slurm
    if spec is None:
        raise HpcAcceptanceError("smoke tasks require a SLURM execution profile")
    if type(worker_count) is not int or not 1 <= worker_count <= min(2, spec.workers):
        raise HpcAcceptanceError(
            "smoke worker_count must be a positive integer bounded by two and the "
            f"profile; got {worker_count!r}"
        )
    if len(context.manifest.rounds) < worker_count:
        raise HpcAcceptanceError(
            f"acceptance manifest needs at least {worker_count} moving rounds; "
            f"found {len(context.manifest.rounds)}"
        )
    dataset_root = context.profile.paths.dataset_root
    fixed_image = dataset_root / context.manifest.reference.registration_channel
    tasks: list[RoundTask] = []
    for round_spec in context.manifest.rounds[:worker_count]:
        no_registration_root = (
            context.evidence_dir / "no-registration" / f"round-{round_spec.round:03d}"
        )
        tasks.append(
            RoundTask(
                dataset_id=context.manifest.dataset_id,
                round=round_spec.round,
                strategy=round_spec.strategy,
                fixed_image=fixed_image,
                moving_image=dataset_root / round_spec.registration_channel,
                registration_directory=no_registration_root,
                log_path=(
                    context.evidence_dir / f"worker-round-{round_spec.round:03d}.json"
                ),
                voxel_spacing_xyz=context.manifest.voxel_spacing_um,
                registration=context.manifest.registration,
                native_threads=context.profile.native_threads,
                channels=tuple(
                    ChannelTask(
                        source=dataset_root / channel.file,
                        output=no_registration_root / "channels" / channel.file.name,
                        interpolation=channel.interpolation,
                    )
                    for channel in round_spec.channels
                ),
                chunk_size_zyx=round_spec.chunk_size_zyx,
                overlap_fraction=round_spec.overlap_fraction,
            )
        )
    return tuple(tasks)


def build_smoke_reference_task(context: HpcAcceptanceContext) -> ReferenceTask:
    """Build the no-registration reference task required by the real barrier."""
    dataset_root = context.profile.paths.dataset_root
    reference = context.manifest.reference
    no_registration_root = (
        context.evidence_dir
        / "no-registration"
        / f"round-{reference.round:03d}-reference"
    )
    channels_root = no_registration_root / "channels"
    return ReferenceTask(
        dataset_id=context.manifest.dataset_id,
        round=reference.round,
        registration_channel_source=(dataset_root / reference.registration_channel),
        registration_channel_output=(
            channels_root / reference.registration_channel.name
        ),
        registration_directory=no_registration_root / "registration",
        crop_bounds_path=None,
        log_path=(
            context.evidence_dir / f"worker-reference-round-{reference.round:03d}.json"
        ),
        voxel_spacing_xyz=context.manifest.voxel_spacing_um,
        crop=False,
        native_threads=context.profile.native_threads,
        channels=tuple(
            ReferenceChannelTask(
                source=dataset_root / channel.file,
                output=channels_root / channel.file.name,
            )
            for channel in reference.channels
        ),
    )


@contextmanager
def smoke_module_by_value() -> Iterator[None]:
    """Temporarily serialize this module by value and restore exact prior state."""
    import cloudpickle

    global _SMOKE_PICKLE_DEPTH, _SMOKE_PICKLE_WAS_REGISTERED

    module = sys.modules[__name__]
    registry = getattr(cloudpickle.cloudpickle, "_PICKLE_BY_VALUE_MODULES", None)
    if not isinstance(registry, set):
        raise HpcAcceptanceError(
            "cloudpickle does not expose registration state needed for exact cleanup"
        )
    with _SMOKE_PICKLE_LOCK:
        if _SMOKE_PICKLE_DEPTH == 0:
            _SMOKE_PICKLE_WAS_REGISTERED = module.__name__ in registry
        if module.__name__ not in registry:
            cloudpickle.register_pickle_by_value(module)
        _SMOKE_PICKLE_DEPTH += 1
    try:
        yield
    finally:
        with _SMOKE_PICKLE_LOCK:
            _SMOKE_PICKLE_DEPTH -= 1
            if _SMOKE_PICKLE_DEPTH < 0:
                _SMOKE_PICKLE_DEPTH = 0
                raise HpcAcceptanceError(
                    "cloudpickle registration nesting was corrupted"
                )
            if _SMOKE_PICKLE_DEPTH == 0:
                if _SMOKE_PICKLE_WAS_REGISTERED:
                    if module.__name__ not in registry:
                        cloudpickle.register_pickle_by_value(module)
                elif module.__name__ in registry:
                    cloudpickle.unregister_pickle_by_value(module)
                _SMOKE_PICKLE_WAS_REGISTERED = False


def collect_interface_facts(name: str) -> dict[str, object]:
    root = Path("/sys/class/net") / name
    facts: dict[str, object] = {
        "name": name,
        "exists": root.is_dir(),
        "operstate": "",
        "address": "",
        "ipv4": "",
    }
    if root.is_dir():
        for field in ("operstate", "address"):
            try:
                facts[field] = (root / field).read_text(encoding="utf-8").strip()
            except OSError:
                facts[field] = ""
        try:
            import fcntl

            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as network_socket:
                request = struct.pack("256s", name[:15].encode("utf-8"))
                response = fcntl.ioctl(network_socket.fileno(), 0x8915, request)
            facts["ipv4"] = socket.inet_ntoa(response[20:24])
        except OSError:
            facts["ipv4"] = ""
    return facts


def _decode_mountinfo_path(value: str) -> str:
    for encoded, decoded in (
        ("\\040", " "),
        ("\\011", "\t"),
        ("\\012", "\n"),
        ("\\134", "\\"),
    ):
        value = value.replace(encoded, decoded)
    return value


def collect_path_fact(path: Path) -> dict[str, object]:
    """Record canonical stat and Linux mount identity for a visible path."""
    resolved = path.resolve(strict=True)
    status = resolved.stat()
    try:
        lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise HpcAcceptanceError("could not read /proc/self/mountinfo") from error
    candidates: list[tuple[Path, dict[str, object]]] = []
    for line in lines:
        before, separator, after = line.partition(" - ")
        if not separator:
            continue
        left = before.split()
        right = after.split()
        if len(left) < 6 or len(right) < 3:
            continue
        mount_point = Path(_decode_mountinfo_path(left[4]))
        if resolved != mount_point and not resolved.is_relative_to(mount_point):
            continue
        candidates.append(
            (
                mount_point,
                {
                    "mount_id": left[0],
                    "parent_id": left[1],
                    "device": left[2],
                    "root": _decode_mountinfo_path(left[3]),
                    "mount_point": str(mount_point),
                    "options": left[5],
                    "filesystem_type": right[0],
                    "source": _decode_mountinfo_path(right[1]),
                    "super_options": right[2],
                },
            )
        )
    if not candidates:
        raise HpcAcceptanceError(f"no Linux mount fact contains {resolved}")
    _mount_point, mount = max(candidates, key=lambda item: len(item[0].parts))
    return {
        "resolved": str(resolved),
        "exists": True,
        "is_directory": resolved.is_dir(),
        "device": status.st_dev,
        "device_major_minor": f"{os.major(status.st_dev)}:{os.minor(status.st_dev)}",
        "mount": mount,
    }


def shared_path_identity(fact: Mapping[str, object]) -> tuple[str, str, str, str, str]:
    """Return the exact device/mount identity compared across coordinator/workers."""
    device = fact.get("device_major_minor")
    mount = fact.get("mount")
    if type(device) is not str or not isinstance(mount, Mapping):
        raise HpcAcceptanceError(f"path fact lacks device/mount identity: {fact!r}")
    values = (
        device,
        mount.get("filesystem_type"),
        mount.get("source"),
        mount.get("root"),
        mount.get("mount_point"),
    )
    if any(type(value) is not str or not value for value in values):
        raise HpcAcceptanceError(f"path mount identity is incomplete: {fact!r}")
    return cast(tuple[str, str, str, str, str], values)


def _scratch_probe(root: Path) -> dict[str, bool]:
    root.mkdir(parents=True, exist_ok=True)
    probe = Path(tempfile.mkdtemp(prefix="cyexm-hpc-smoke-", dir=root))
    created = probe.is_dir()
    try:
        payload = probe / "write-probe"
        expected = b"cyexm-hpc-smoke\n"
        with payload.open("wb") as stream:
            stream.write(expected)
            stream.flush()
            os.fsync(stream.fileno())
        if payload.read_bytes() != expected:
            raise HpcAcceptanceError("scratch probe content did not round-trip")
    finally:
        shutil.rmtree(probe)
    return {"created": created, "deleted": not probe.exists()}


def _planned_artifacts(task: WorkflowTask) -> tuple[str, ...]:
    if isinstance(task, ReferenceTask):
        return tuple(str(channel.output) for channel in task.channels)
    return (
        str(task.registration_directory / f"linear_transform_{task.round}.mat"),
        str(task.registration_directory / f"nonlinear_warp_{task.round}.nii.gz"),
        str(task.registration_directory / f"nonlinear_registered_{task.round}.tif"),
        *(str(channel.output) for channel in task.channels),
    )


def run_environment_smoke(
    task: WorkflowTask,
    *,
    evidence_dir: str,
    dataset_root: str,
    output_root: str,
    worker_local_directory: str,
    interface: str,
    _run: Callable[..., Any] = subprocess.run,
    _get_worker: Callable[[], Any] | None = None,
    _hostname: Callable[[], str] = socket.gethostname,
    _interface_facts: Callable[[str], dict[str, object]] = collect_interface_facts,
) -> RoundResult:
    """Record one allocated worker without importing or running registration."""
    started_at = datetime.now(UTC)
    started = time.monotonic()
    if _get_worker is None:
        from distributed import get_worker

        _get_worker = get_worker
    worker = _get_worker()
    hostname = _hostname()
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise HpcAcceptanceError("smoke worker has no SLURM_JOB_ID")
    slurm_node_name = os.environ.get("SLURMD_NODENAME") or hostname
    resources = worker.state.total_resources
    if not isinstance(resources, Mapping):
        raise HpcAcceptanceError(
            f"smoke worker resources must be a mapping; got {resources!r}"
        )
    round_slot = resources.get("round_slot")
    if round_slot != 1:
        raise HpcAcceptanceError(
            f"smoke worker must advertise exactly one round_slot; got {round_slot!r}"
        )
    memory_limit = worker.memory_manager.memory_limit
    if type(memory_limit) is not int or memory_limit < 1:
        raise HpcAcceptanceError(
            f"smoke worker memory limit must be positive; got {memory_limit!r}"
        )
    task_key = worker.get_current_task()
    if type(task_key) is not str or not task_key:
        raise HpcAcceptanceError("smoke worker must report its Dask task key")

    job_command = _run_read_only(
        ["scontrol", "--oneliner", "show", "job", job_id], run=_run
    )
    node_command = _run_read_only(
        ["scontrol", "--oneliner", "show", "node", slurm_node_name], run=_run
    )
    scratch = _scratch_probe(Path(worker_local_directory))
    if isinstance(task, ReferenceTask):
        fixed_image = task.registration_channel_source
        moving_image = task.registration_channel_source
        strategy = "reference-preparation"
    elif type(task) is RoundTask:
        fixed_image = task.fixed_image
        moving_image = task.moving_image
        strategy = task.strategy
    else:
        raise HpcAcceptanceError(
            "environment smoke accepts only exact coarse workflow tasks"
        )
    record = {
        "schema_version": 1,
        "dataset_id": task.dataset_id,
        "round": task.round,
        "hostname": hostname,
        "slurm_node_name": slurm_node_name,
        "pid": os.getpid(),
        "slurm_job_id": job_id,
        "dask_task_key": task_key,
        "worker_address": worker.address,
        "round_slot": round_slot,
        "worker_memory_limit_bytes": memory_limit,
        "interface": _interface_facts(interface),
        "paths": {
            "dataset_root": collect_path_fact(Path(dataset_root)),
            "output_root": collect_path_fact(Path(output_root)),
            "worker_local_directory": collect_path_fact(Path(worker_local_directory)),
            "evidence_dir": collect_path_fact(Path(evidence_dir)),
            "fixed_image": collect_path_fact(fixed_image),
            "moving_image": collect_path_fact(moving_image),
        },
        "scratch_probe": scratch,
        "job": parse_scontrol_record(cast(str, job_command["stdout"])),
        "node": parse_scontrol_record(cast(str, node_command["stdout"])),
        "commands": [job_command, node_command],
    }
    evidence_root = Path(evidence_dir).resolve(strict=True)
    destination = task.log_path.resolve(strict=False)
    if destination.parent != evidence_root:
        raise HpcAcceptanceError(
            f"smoke worker evidence must be a direct child of {evidence_root}"
        )
    _write_json_atomic(destination, record)
    ended_at = datetime.now(UTC)
    return RoundResult(
        dataset_id=task.dataset_id,
        round=task.round,
        status="completed",
        started_at=started_at.isoformat().replace("+00:00", "Z"),
        ended_at=ended_at.isoformat().replace("+00:00", "Z"),
        duration_seconds=float(max(0.0, time.monotonic() - started)),
        hostname=hostname,
        pid=os.getpid(),
        dask_task_key=task_key,
        slurm_job_id=job_id,
        strategy=strategy,
        native_threads=(
            ("itk", task.native_threads.itk),
            ("blas", task.native_threads.blas),
        ),
        artifact_paths=_planned_artifacts(task),
        reused_artifacts=(),
        log_path=str(task.log_path),
        failure=None,
    )


def _validate_storage(
    *,
    context_paths: tuple[Path, Path, Path],
    profile: ExecutionProfile,
    repository_root: Path,
    archive_root: Path,
    project_root: Path,
    protected_roots: Sequence[Path],
) -> None:
    dataset_path, execution_path, evidence_dir = context_paths
    for path, role in (
        (dataset_path, "dataset manifest"),
        (execution_path, "execution profile"),
        (profile.paths.dataset_root, "dataset root"),
        (profile.paths.output_root, "output root"),
        (evidence_dir, "evidence directory"),
    ):
        _require_under(path, archive_root, role)

    scratch = profile.paths.worker_local_directory
    if scratch is None:
        raise HpcAcceptanceError("SLURM acceptance requires worker scratch")
    _require_under(scratch, project_root, "worker scratch")

    protected = (repository_root, *protected_roots)
    configurations = (
        (dataset_path, "dataset manifest configuration"),
        (execution_path, "execution profile configuration"),
    )
    roots = (
        (profile.paths.dataset_root, "dataset root"),
        (profile.paths.output_root, "output root"),
        (scratch, "worker scratch"),
        (evidence_dir, "evidence directory"),
    )
    for path, role in (*configurations, *roots):
        if any(not _is_disjoint(path, protected_root) for protected_root in protected):
            raise HpcAcceptanceError(
                f"{role} must be disjoint from every protected repository and "
                f"live spheroid path: {path}"
            )
    for configuration, role in configurations:
        if any(not _is_disjoint(configuration, root) for root, _ in roots):
            raise HpcAcceptanceError(
                f"configuration {role} must be disjoint from all data, output, "
                f"scratch, and evidence roots: {configuration}"
            )
    for index, (first, first_role) in enumerate(roots):
        for second, second_role in roots[index + 1 :]:
            if not _is_disjoint(first, second):
                raise HpcAcceptanceError(
                    f"{first_role} and {second_role} must be canonically disjoint; "
                    f"got {first} and {second}"
                )


def _validate_dataset_files(
    manifest: DatasetManifest,
    profile: ExecutionProfile,
    *,
    archive_root: Path,
    protected_roots: Sequence[Path],
) -> tuple[Path, ...]:
    dataset_root = _resolved_directory(profile.paths.dataset_root, "dataset root")
    _resolved_directory(profile.paths.output_root, "output root")
    relative_paths = [channel.file for channel in manifest.reference.channels]
    for round_spec in manifest.rounds:
        relative_paths.extend(channel.file for channel in round_spec.channels)
    resolved_inputs: list[Path] = []
    for relative in dict.fromkeys(relative_paths):
        role = f"manifest input {relative}"
        resolved = _resolved_file(dataset_root / relative, role)
        if not resolved.is_relative_to(dataset_root):
            raise HpcAcceptanceError(
                f"{role} resolves outside the dataset root or into a protected "
                f"path: {resolved}"
            )
        _require_under(resolved, archive_root, role)
        if any(
            not _is_disjoint(resolved, protected_root)
            for protected_root in protected_roots
        ):
            raise HpcAcceptanceError(
                f"{role} resolves into a protected path: {resolved}"
            )
        resolved_inputs.append(resolved)
    return tuple(resolved_inputs)


def prepare_acceptance_context(
    dataset: str | Path | None,
    execution: str | Path | None,
    evidence: str | Path | None,
    *,
    repository_root: Path,
    archive_root: Path = Path("/archive"),
    project_root: Path = Path("/project"),
    protected_roots: Sequence[Path] = (),
    argv: Sequence[str] | None = None,
    _now: Callable[[], datetime] = lambda: datetime.now(UTC),
    _git_commit: Callable[[Path], str] = _git_head,
    _version: Callable[[str], str] = package_version,
) -> HpcAcceptanceContext:
    """Validate explicit operator inputs and atomically record provenance."""
    raw_dataset, raw_execution, raw_evidence = require_cli_paths(
        dataset, execution, evidence
    )
    dataset_path = _resolved_file(raw_dataset, "dataset manifest")
    execution_path = _resolved_file(raw_execution, "execution profile")
    evidence_dir = _resolved_directory(raw_evidence, "evidence directory")
    if any(evidence_dir.iterdir()):
        raise HpcAcceptanceError(
            f"evidence directory must be empty before acceptance: {evidence_dir}"
        )

    repository = repository_root.resolve(strict=True)
    archive = archive_root.resolve(strict=True)
    project = project_root.resolve(strict=True)
    protected = tuple(path.resolve(strict=False) for path in protected_roots)
    manifest = load_dataset_manifest(dataset_path)
    profile = load_execution_profile(execution_path)
    if profile.executor != "slurm" or profile.slurm is None:
        raise HpcAcceptanceError(
            "--cyexm-execution must load a strict executor: slurm profile"
        )
    _validate_storage(
        context_paths=(dataset_path, execution_path, evidence_dir),
        profile=profile,
        repository_root=repository,
        archive_root=archive,
        project_root=project,
        protected_roots=protected,
    )
    _validate_dataset_files(
        manifest,
        profile,
        archive_root=archive,
        protected_roots=(repository, *protected),
    )

    versions = _package_versions(_version)
    commit = _git_commit(repository)
    if (
        type(commit) is not str
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit.lower())
    ):
        raise HpcAcceptanceError(
            f"git HEAD must be a 40-character hash; got {commit!r}"
        )

    arguments = tuple(sys.argv if argv is None else argv)
    record = {
        "schema_version": 1,
        "timestamp_utc": _canonical_timestamp(_now()),
        "git_commit": commit,
        "package_versions": versions,
        "paths": {
            "dataset_manifest": str(dataset_path),
            "execution_profile": str(execution_path),
            "evidence_dir": str(evidence_dir),
            "dataset_root": str(profile.paths.dataset_root),
            "output_root": str(profile.paths.output_root),
            "worker_local_directory": str(profile.paths.worker_local_directory),
        },
        "argv": list(arguments),
    }
    _write_json_atomic(evidence_dir / "acceptance-context.json", record)
    return HpcAcceptanceContext(
        dataset_path=dataset_path,
        execution_path=execution_path,
        evidence_dir=evidence_dir,
        manifest=manifest,
        profile=profile,
        package_versions=versions,
        git_commit=commit,
    )


def validate_additional_acceptance_profile(
    context: HpcAcceptanceContext,
    execution: str | Path,
    *,
    repository_root: Path,
    archive_root: Path = Path("/archive"),
    project_root: Path = Path("/project"),
    protected_roots: Sequence[Path] = (),
) -> tuple[Path, ExecutionProfile]:
    """Apply the canonical storage/input safety policy to an extra profile."""
    execution_path = _resolved_file(Path(execution), "additional execution profile")
    repository = repository_root.resolve(strict=True)
    archive = archive_root.resolve(strict=True)
    project = project_root.resolve(strict=True)
    protected = tuple(path.resolve(strict=False) for path in protected_roots)
    profile = load_execution_profile(execution_path)
    if profile.executor != "slurm" or profile.slurm is None:
        raise HpcAcceptanceError(
            "additional execution profile must load a strict executor: slurm profile"
        )
    _validate_storage(
        context_paths=(context.dataset_path, execution_path, context.evidence_dir),
        profile=profile,
        repository_root=repository,
        archive_root=archive,
        project_root=project,
        protected_roots=protected,
    )
    _validate_dataset_files(
        context.manifest,
        profile,
        archive_root=archive,
        protected_roots=(repository, *protected),
    )
    return execution_path, profile


__all__ = [
    "ACCEPTANCE_PACKAGES",
    "HpcAcceptanceContext",
    "HpcAcceptanceError",
    "build_smoke_reference_task",
    "build_smoke_tasks",
    "collect_interface_facts",
    "collect_partition_snapshot",
    "collect_path_fact",
    "collect_queue_snapshot",
    "parse_scontrol_record",
    "parse_slurm_tres_memory_bytes",
    "partition_is_exclusive",
    "prepare_acceptance_context",
    "require_cli_paths",
    "require_coordinator_native_threads",
    "require_explicit_hpc_selection",
    "require_hpc_opt_in",
    "run_environment_smoke",
    "shared_path_identity",
    "smoke_module_by_value",
    "validate_additional_acceptance_profile",
    "validate_rendered_job_script",
    "validate_scheduler_interface",
    "validate_slurm_job_node_resources",
    "wait_for_queue_empty",
    "write_evidence_json",
    "write_evidence_text",
]
