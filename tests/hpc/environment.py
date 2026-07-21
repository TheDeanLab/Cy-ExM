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
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, cast

from cyexm.config import (
    DatasetManifest,
    ExecutionProfile,
    load_dataset_manifest,
    load_execution_profile,
)
from cyexm.io.atomic import atomic_write
from cyexm.workflow.models import ChannelTask, RoundResult, RoundTask

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
    _run: Callable[..., Any] = subprocess.run,
) -> dict[str, object]:
    """Read the current user's jobs with the exact harness job name."""
    command = _run_read_only(
        [
            "squeue",
            "--noheader",
            "--user",
            user,
            "--name",
            job_name,
            "--format=%i|%j|%T|%N",
        ],
        run=_run,
    )
    jobs: list[dict[str, str]] = []
    for line_number, line in enumerate(cast(str, command["stdout"]).splitlines(), 1):
        if not line.strip():
            continue
        values = [value.strip() for value in line.split("|")]
        if len(values) != 4 or not all(values[:3]):
            raise HpcAcceptanceError(f"squeue row {line_number} is malformed: {line!r}")
        jobs.append(
            dict(
                zip(
                    ("job_id", "job_name", "state", "nodes"),
                    values,
                    strict=True,
                )
            )
        )
    return {
        "schema_version": 1,
        "job_name": job_name,
        "user": user,
        "jobs": jobs,
        "command": command,
    }


def wait_for_queue_empty(
    job_name: str,
    *,
    user: str,
    timeout_seconds: float = 60.0,
    poll_seconds: float = 1.0,
    _run: Callable[..., Any] = subprocess.run,
    _monotonic: Callable[[], float] = time.monotonic,
    _sleep: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Bound cleanup observation while SLURM retires completed job records."""
    if timeout_seconds <= 0.0 or poll_seconds < 0.0:
        raise HpcAcceptanceError("queue cleanup timeout must be positive and bounded")
    started = _monotonic()
    deadline = started + timeout_seconds
    attempts: list[dict[str, object]] = []
    while True:
        snapshot = collect_queue_snapshot(job_name, user=user, _run=_run)
        attempts.append(snapshot)
        if snapshot["jobs"] == []:
            return {
                "schema_version": 1,
                "job_name": job_name,
                "user": user,
                "jobs": [],
                "attempts": attempts,
                "elapsed_seconds": float(max(0.0, _monotonic() - started)),
            }
        now = _monotonic()
        if now >= deadline:
            raise HpcAcceptanceError(
                f"owned jobs remain after {timeout_seconds:.1f}s: {snapshot['jobs']!r}"
            )
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
    fixed_image = dataset_root / context.manifest.reference.channel
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


@contextmanager
def smoke_module_by_value() -> Iterator[None]:
    """Make the test-only smoke runner self-contained for remote workers."""
    import cloudpickle

    module = sys.modules[__name__]
    cloudpickle.register_pickle_by_value(module)
    try:
        yield
    finally:
        cloudpickle.unregister_pickle_by_value(module)


def _default_interface_facts(name: str) -> dict[str, object]:
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


def _path_fact(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    status = resolved.stat()
    return {
        "resolved": str(resolved),
        "exists": True,
        "is_directory": resolved.is_dir(),
        "device": status.st_dev,
    }


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


def _planned_artifacts(task: RoundTask) -> tuple[str, ...]:
    return (
        str(task.registration_directory / f"linear_transform_{task.round}.mat"),
        str(task.registration_directory / f"nonlinear_warp_{task.round}.nii.gz"),
        str(task.registration_directory / f"nonlinear_registered_{task.round}.tif"),
        *(str(channel.output) for channel in task.channels),
    )


def run_environment_smoke(
    task: RoundTask,
    *,
    evidence_dir: str,
    dataset_root: str,
    output_root: str,
    worker_local_directory: str,
    interface: str,
    _run: Callable[..., Any] = subprocess.run,
    _get_worker: Callable[[], Any] | None = None,
    _hostname: Callable[[], str] = socket.gethostname,
    _interface_facts: Callable[[str], dict[str, object]] = _default_interface_facts,
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
            "dataset_root": _path_fact(Path(dataset_root)),
            "output_root": _path_fact(Path(output_root)),
            "worker_local_directory": _path_fact(Path(worker_local_directory)),
            "fixed_image": _path_fact(task.fixed_image),
            "moving_image": _path_fact(task.moving_image),
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
        strategy=task.strategy,
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
    live_roots: Sequence[Path],
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

    protected = (repository_root, *live_roots)
    if any(not _is_disjoint(evidence_dir, root) for root in protected):
        raise HpcAcceptanceError(
            "evidence directory must be disjoint from every protected repository "
            "and live spheroid path"
        )
    for root, role in (
        (profile.paths.dataset_root, "dataset root"),
        (profile.paths.output_root, "output root"),
        (scratch, "worker scratch"),
    ):
        if not _is_disjoint(evidence_dir, root):
            raise HpcAcceptanceError(
                f"evidence directory must be disjoint from {role}: {root}"
            )


def _validate_dataset_files(
    manifest: DatasetManifest,
    profile: ExecutionProfile,
) -> None:
    dataset_root = _resolved_directory(profile.paths.dataset_root, "dataset root")
    _resolved_directory(profile.paths.output_root, "output root")
    relative_paths = [manifest.reference.channel]
    for round_spec in manifest.rounds:
        relative_paths.extend(channel.file for channel in round_spec.channels)
    for relative in dict.fromkeys(relative_paths):
        _resolved_file(dataset_root / relative, f"manifest input {relative}")


def prepare_acceptance_context(
    dataset: str | Path | None,
    execution: str | Path | None,
    evidence: str | Path | None,
    *,
    repository_root: Path,
    archive_root: Path = Path("/archive"),
    project_root: Path = Path("/project"),
    live_roots: Sequence[Path] = (),
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
    live = tuple(path.resolve(strict=False) for path in live_roots)
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
        live_roots=live,
    )
    _validate_dataset_files(manifest, profile)

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


__all__ = [
    "ACCEPTANCE_PACKAGES",
    "HpcAcceptanceContext",
    "HpcAcceptanceError",
    "build_smoke_tasks",
    "collect_partition_snapshot",
    "collect_queue_snapshot",
    "parse_scontrol_record",
    "parse_slurm_tres_memory_bytes",
    "partition_is_exclusive",
    "prepare_acceptance_context",
    "require_cli_paths",
    "run_environment_smoke",
    "smoke_module_by_value",
    "wait_for_queue_empty",
    "write_evidence_json",
    "write_evidence_text",
]
