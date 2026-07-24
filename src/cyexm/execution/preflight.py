"""Standard-library-only evidence collection for distributed workers."""

from __future__ import annotations

import os
import shutil
import socket
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Self, cast
from urllib.parse import urlsplit

from cyexm.errors import WorkerPreflightError

_PROBE_KEYS = frozenset(
    {
        "ok",
        "worker_address",
        "hostname",
        "pid",
        "slurm_job_id",
        "package_versions",
        "native_threads",
        "dataset_visible",
        "output_writable",
        "scratch_usable",
        "memory_limit_bytes",
        "network_host",
        "errors",
    }
)
_OUTPUT_PREFIX = ".cyexm-preflight-"
_SCRATCH_PREFIX = "cyexm-preflight-"
_MISSING = "<missing>"


def _exact_string(value: object, location: str, *, empty: bool = True) -> str:
    if type(value) is not str or (not empty and not value):
        qualifier = "nonempty " if not empty else ""
        raise TypeError(f"{location} must be a {qualifier}built-in string")
    return value


def _exact_boolean(value: object, location: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{location} must be a built-in boolean")
    return value


def _exact_integer(value: object, location: str, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise TypeError(f"{location} must be a built-in integer of at least {minimum}")
    return value


def _validate_pairs(
    value: object,
    location: str,
    *,
    json_arrays: bool,
) -> tuple[tuple[str, str], ...]:
    outer_type = list if json_arrays else tuple
    pair_type = list if json_arrays else tuple
    if type(value) is not outer_type:
        shape = "JSON array" if json_arrays else "exact tuple"
        raise TypeError(f"{location} must be a {shape}")
    parsed: list[tuple[str, str]] = []
    seen: set[str] = set()
    for index, pair in enumerate(value):
        if type(pair) is not pair_type or len(pair) != 2:
            shape = "two-item JSON array" if json_arrays else "two-item exact tuple"
            raise TypeError(f"{location}[{index}] must be a {shape}")
        name = _exact_string(pair[0], f"{location}[{index}][0]", empty=False)
        item = _exact_string(pair[1], f"{location}[{index}][1]")
        if name in seen:
            raise ValueError(f"{location} contains duplicate name {name!r}")
        seen.add(name)
        parsed.append((name, item))
    parsed.sort()
    return tuple(parsed)


def _validate_errors(
    value: object,
    *,
    json_array: bool,
) -> tuple[str, ...]:
    expected_type = list if json_array else tuple
    if type(value) is not expected_type:
        shape = "JSON array" if json_array else "exact tuple"
        raise TypeError(f"errors must be a {shape}")
    return tuple(
        _exact_string(item, f"errors[{index}]", empty=False)
        for index, item in enumerate(value)
    )


@dataclass(frozen=True, slots=True)
class WorkerProbe:
    """Immutable, JSON-serializable evidence from one connected worker."""

    ok: bool
    worker_address: str
    hostname: str
    pid: int
    slurm_job_id: str | None
    package_versions: tuple[tuple[str, str], ...]
    native_threads: tuple[tuple[str, str], ...]
    dataset_visible: bool
    output_writable: bool
    scratch_usable: bool
    memory_limit_bytes: int
    network_host: str
    errors: tuple[str, ...]

    def __post_init__(self) -> None:
        _exact_boolean(self.ok, "ok")
        _exact_string(self.worker_address, "worker_address")
        _exact_string(self.hostname, "hostname")
        _exact_integer(self.pid, "pid", minimum=0)
        if self.slurm_job_id is not None:
            _exact_string(self.slurm_job_id, "slurm_job_id", empty=False)
        package_versions = _validate_pairs(
            self.package_versions,
            "package_versions",
            json_arrays=False,
        )
        native_threads = _validate_pairs(
            self.native_threads,
            "native_threads",
            json_arrays=False,
        )
        if package_versions != self.package_versions:
            raise ValueError("package_versions must use canonical sorted ordering")
        if native_threads != self.native_threads:
            raise ValueError("native_threads must use canonical sorted ordering")
        _exact_boolean(self.dataset_visible, "dataset_visible")
        _exact_boolean(self.output_writable, "output_writable")
        _exact_boolean(self.scratch_usable, "scratch_usable")
        _exact_integer(self.memory_limit_bytes, "memory_limit_bytes", minimum=0)
        _exact_string(self.network_host, "network_host")
        errors = _validate_errors(self.errors, json_array=False)
        if self.ok != (not errors):
            raise ValueError("ok must be true exactly when errors is empty")
        if self.ok and (
            not self.worker_address
            or not self.hostname
            or self.pid < 1
            or not self.network_host
            or self.memory_limit_bytes < 1
            or not self.dataset_visible
            or not self.output_writable
            or not self.scratch_usable
        ):
            raise ValueError("ok probe evidence must contain usable worker resources")

    def to_dict(self) -> dict[str, object]:
        """Return only strict JSON-compatible built-in primitives."""
        return {
            "ok": self.ok,
            "worker_address": self.worker_address,
            "hostname": self.hostname,
            "pid": self.pid,
            "slurm_job_id": self.slurm_job_id,
            "package_versions": [list(pair) for pair in self.package_versions],
            "native_threads": [list(pair) for pair in self.native_threads],
            "dataset_visible": self.dataset_visible,
            "output_writable": self.output_writable,
            "scratch_usable": self.scratch_usable,
            "memory_limit_bytes": self.memory_limit_bytes,
            "network_host": self.network_host,
            "errors": list(self.errors),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        """Decode one exact worker-probe JSON object."""
        if type(payload) is not dict:
            raise TypeError("WorkerProbe payload must be a built-in JSON object")
        if frozenset(payload) != _PROBE_KEYS:
            raise ValueError(
                "WorkerProbe keys must be exactly: " + ", ".join(sorted(_PROBE_KEYS))
            )
        slurm_job_id = payload["slurm_job_id"]
        if slurm_job_id is not None:
            slurm_job_id = _exact_string(
                slurm_job_id,
                "slurm_job_id",
                empty=False,
            )
        return cls(
            ok=_exact_boolean(payload["ok"], "ok"),
            worker_address=_exact_string(
                payload["worker_address"],
                "worker_address",
            ),
            hostname=_exact_string(payload["hostname"], "hostname"),
            pid=_exact_integer(payload["pid"], "pid", minimum=0),
            slurm_job_id=slurm_job_id,
            package_versions=_validate_pairs(
                payload["package_versions"],
                "package_versions",
                json_arrays=True,
            ),
            native_threads=_validate_pairs(
                payload["native_threads"],
                "native_threads",
                json_arrays=True,
            ),
            dataset_visible=_exact_boolean(
                payload["dataset_visible"],
                "dataset_visible",
            ),
            output_writable=_exact_boolean(
                payload["output_writable"],
                "output_writable",
            ),
            scratch_usable=_exact_boolean(
                payload["scratch_usable"],
                "scratch_usable",
            ),
            memory_limit_bytes=_exact_integer(
                payload["memory_limit_bytes"],
                "memory_limit_bytes",
                minimum=0,
            ),
            network_host=_exact_string(
                payload["network_host"],
                "network_host",
            ),
            errors=_validate_errors(payload["errors"], json_array=True),
        )


def _error(errors: list[str], message: str, error: BaseException) -> None:
    errors.append(f"{message}: {type(error).__name__}: {error}")


def _expected_items(
    values: Mapping[str, str],
    location: str,
    errors: list[str],
) -> tuple[tuple[str, str], ...]:
    if not isinstance(values, Mapping):
        errors.append(f"{location} must be a mapping of built-in strings")
        return ()
    parsed: list[tuple[str, str]] = []
    try:
        items = tuple(values.items())
    except Exception as error:
        _error(errors, f"could not inspect {location}", error)
        return ()
    for index, pair in enumerate(items):
        name, expected = pair
        if type(name) is not str or not name:
            errors.append(f"{location} key {index} must be a nonempty built-in string")
            continue
        if type(expected) is not str or not expected:
            errors.append(f"{location}[{name!r}] must be a nonempty built-in string")
            continue
        parsed.append((name, expected))
    parsed.sort()
    return tuple(parsed)


def _probe_versions(
    expected_versions: Mapping[str, str],
    version: Callable[[str], str],
    errors: list[str],
) -> tuple[tuple[str, str], ...]:
    observed: list[tuple[str, str]] = []
    for name, expected in _expected_items(
        expected_versions,
        "expected_versions",
        errors,
    ):
        try:
            actual = version(name)
        except PackageNotFoundError:
            observed.append((name, _MISSING))
            errors.append(f"package {name!r} is not installed; expected {expected!r}")
            continue
        except Exception as error:
            observed.append((name, _MISSING))
            _error(errors, f"could not inspect package {name!r}", error)
            continue
        if type(actual) is not str or not actual:
            observed.append((name, "<invalid>"))
            errors.append(
                f"package {name!r} returned an invalid built-in version string: "
                f"{actual!r}"
            )
            continue
        observed.append((name, actual))
        if actual != expected:
            errors.append(
                f"package {name!r} version mismatch: expected {expected!r}, "
                f"found {actual!r}"
            )
    return tuple(observed)


def _probe_threads(
    expected_threads: Mapping[str, str],
    environ: Mapping[str, str],
    errors: list[str],
) -> tuple[tuple[str, str], ...]:
    observed: list[tuple[str, str]] = []
    for name, expected in _expected_items(
        expected_threads,
        "expected_threads",
        errors,
    ):
        actual = environ.get(name)
        if actual is None:
            observed.append((name, _MISSING))
            errors.append(
                f"native thread variable {name} is missing; expected {expected!r}"
            )
            continue
        if type(actual) is not str:
            observed.append((name, "<invalid>"))
            errors.append(
                f"native thread variable {name} must be a built-in string; "
                f"found {actual!r}"
            )
            continue
        observed.append((name, actual))
        if actual != expected:
            errors.append(
                f"native thread variable {name} mismatch: expected {expected!r}, "
                f"found {actual!r}"
            )
    return tuple(observed)


def _path_argument(value: str, role: str, errors: list[str]) -> Path | None:
    if type(value) is not str or not value:
        errors.append(f"{role} must be a nonempty built-in path string")
        return None
    path = Path(value)
    if not path.is_absolute():
        errors.append(f"{role} must be an absolute path; got {value!r}")
        return None
    return path


def _probe_dataset(path: Path | None, errors: list[str]) -> bool:
    if path is None:
        return False
    try:
        with os.scandir(path) as entries:
            next(entries, None)
    except Exception as error:
        _error(errors, f"dataset root is not readable at {path}", error)
        return False
    return True


def _close_probe_file(handle: object, role: str, errors: list[str]) -> bool:
    try:
        handle.close()  # type: ignore[attr-defined]
    except Exception as error:
        _error(errors, f"could not close {role} probe file", error)
        return False
    return True


def _owned_probe_file(path: Path, parent: Path, prefix: str) -> bool:
    return path.parent == parent and path.name.startswith(prefix)


def _probe_output(path: Path | None, errors: list[str]) -> bool:
    if path is None:
        return False
    handle: object | None = None
    probe_path: Path | None = None
    written = False
    cleanup = True
    try:
        # Close/delete are deliberately separate so both failures can be evidence.
        handle = tempfile.NamedTemporaryFile(  # noqa: SIM115
            mode="w+b",
            prefix=_OUTPUT_PREFIX,
            dir=path,
            delete=False,
        )
        probe_path = Path(handle.name)  # type: ignore[attr-defined]
        if not _owned_probe_file(probe_path, path, _OUTPUT_PREFIX):
            errors.append(
                f"output probe returned an unsafe path outside {path}: {probe_path}"
            )
            return False
        handle.write(b"cyexm-worker-preflight\n")  # type: ignore[attr-defined]
        handle.flush()  # type: ignore[attr-defined]
        os.fsync(handle.fileno())  # type: ignore[attr-defined]
        written = _close_probe_file(handle, "output", errors)
        handle = None
    except Exception as error:
        _error(errors, f"output root is not writable at {path}", error)
    finally:
        if handle is not None:
            cleanup = _close_probe_file(handle, "output", errors) and cleanup
        if probe_path is not None:
            if _owned_probe_file(probe_path, path, _OUTPUT_PREFIX):
                try:
                    probe_path.unlink(missing_ok=True)
                except Exception as error:
                    cleanup = False
                    _error(
                        errors,
                        f"could not delete output probe file {probe_path}",
                        error,
                    )
            else:
                cleanup = False
                errors.append(
                    f"refused to delete unsafe output probe path {probe_path}"
                )
    return written and cleanup


def _owned_scratch_directory(path: Path, parent: Path) -> bool:
    return path.parent == parent and path.name.startswith(_SCRATCH_PREFIX)


def _probe_scratch(path: Path | None, errors: list[str]) -> bool:
    if path is None:
        return False
    probe_directory: Path | None = None
    handle: object | None = None
    created = False
    written = False
    cleanup = True
    try:
        path.mkdir(parents=True, exist_ok=True)
        if not path.is_dir():
            raise NotADirectoryError(path)
        probe_directory = Path(tempfile.mkdtemp(prefix=_SCRATCH_PREFIX, dir=str(path)))
        if not _owned_scratch_directory(probe_directory, path):
            errors.append(
                "scratch probe returned an unsafe directory outside "
                f"{path}: {probe_directory}"
            )
            return False
        created = True
        try:
            # Close/delete are deliberately separate so both failures are retained.
            handle = tempfile.NamedTemporaryFile(  # noqa: SIM115
                mode="w+b",
                prefix=".write-",
                dir=probe_directory,
                delete=False,
            )
            handle.write(b"cyexm-worker-scratch-preflight\n")  # type: ignore[attr-defined]
            handle.flush()  # type: ignore[attr-defined]
            os.fsync(handle.fileno())  # type: ignore[attr-defined]
            written = _close_probe_file(handle, "scratch", errors)
            handle = None
        except Exception as error:
            _error(errors, f"scratch directory is not writable at {path}", error)
    except Exception as error:
        _error(errors, f"could not create scratch probe directory at {path}", error)
    finally:
        if handle is not None:
            cleanup = _close_probe_file(handle, "scratch", errors) and cleanup
        if probe_directory is not None:
            if _owned_scratch_directory(probe_directory, path):
                try:
                    shutil.rmtree(probe_directory)
                    if probe_directory.exists():
                        raise OSError("scratch probe directory still exists")
                except Exception as error:
                    cleanup = False
                    _error(
                        errors,
                        f"could not delete scratch probe directory {probe_directory}",
                        error,
                    )
            else:
                cleanup = False
                errors.append(
                    f"refused to delete unsafe scratch probe path {probe_directory}"
                )
    return created and written and cleanup


def _worker_identity(
    worker: object | None,
    errors: list[str],
) -> tuple[str, int, str]:
    address = ""
    memory_limit = 0
    if worker is None:
        errors.append(
            "worker address is unavailable because dask_worker was not injected"
        )
        errors.append(
            "worker memory limit is unavailable because dask_worker was not injected"
        )
        return address, memory_limit, ""
    worker_value = cast(Any, worker)
    try:
        candidate = worker_value.address
    except Exception as error:
        _error(errors, "could not inspect worker address", error)
    else:
        if type(candidate) is not str or not candidate:
            errors.append(
                "worker address must be a nonempty built-in string; "
                f"found {candidate!r}"
            )
        else:
            address = candidate
    try:
        candidate_memory = worker_value.memory_manager.memory_limit
    except Exception as error:
        _error(errors, "could not inspect worker memory limit", error)
    else:
        if type(candidate_memory) is not int or candidate_memory < 1:
            errors.append(
                "worker memory limit must be a positive built-in integer; "
                f"found {candidate_memory!r}"
            )
        else:
            memory_limit = candidate_memory

    network_host = ""
    if address:
        try:
            parsed = urlsplit(address)
            candidate_host = parsed.hostname
        except ValueError as error:
            _error(errors, f"could not parse worker network address {address!r}", error)
        else:
            if type(candidate_host) is not str or not candidate_host:
                errors.append(
                    "worker address does not contain a usable network host: "
                    f"{address!r}"
                )
            else:
                network_host = candidate_host
    return address, memory_limit, network_host


def _hostname(errors: list[str]) -> str:
    try:
        hostname = socket.gethostname()
    except Exception as error:
        _error(errors, "could not inspect worker hostname", error)
        return ""
    if (
        type(hostname) is not str
        or not hostname
        or "\n" in hostname
        or "\r" in hostname
    ):
        errors.append(
            "worker hostname must be a nonempty single-line built-in string; "
            f"found {hostname!r}"
        )
        return ""
    return hostname


def _slurm_job_id(environ: Mapping[str, str], errors: list[str]) -> str | None:
    value = environ.get("SLURM_JOB_ID")
    if value is None:
        return None
    if type(value) is not str or not value or "\n" in value or "\r" in value:
        errors.append(
            "SLURM_JOB_ID must be absent or a nonempty single-line built-in string; "
            f"found {value!r}"
        )
        return None
    return value


def probe_worker(
    *,
    dataset_root: str,
    output_root: str,
    worker_local_directory: str,
    expected_versions: Mapping[str, str],
    expected_threads: Mapping[str, str],
    dask_worker: object | None = None,
    _version: Callable[[str], str] = package_version,
    _worker: object | None = None,
) -> dict[str, object]:
    """Collect all worker failures without importing scientific/scheduler runtimes.

    ``distributed.Client.run`` injects its worker through the specially named
    ``dask_worker`` keyword. ``_worker`` is a narrow unit-test override.
    """
    errors: list[str] = []
    worker = _worker if _worker is not None else dask_worker
    address, memory_limit, network_host = _worker_identity(worker, errors)
    hostname = _hostname(errors)
    try:
        pid = os.getpid()
    except Exception as error:
        _error(errors, "could not inspect worker process ID", error)
        pid = 0
    if type(pid) is not int or pid < 1:
        errors.append(
            f"worker process ID must be a positive built-in integer; found {pid!r}"
        )
        pid = 0
    slurm_job_id = _slurm_job_id(os.environ, errors)
    versions = _probe_versions(expected_versions, _version, errors)
    threads = _probe_threads(expected_threads, os.environ, errors)

    dataset = _path_argument(dataset_root, "dataset_root", errors)
    output = _path_argument(output_root, "output_root", errors)
    scratch = _path_argument(
        worker_local_directory,
        "worker_local_directory",
        errors,
    )
    dataset_visible = _probe_dataset(dataset, errors)
    output_writable = _probe_output(output, errors)
    scratch_usable = _probe_scratch(scratch, errors)

    probe = WorkerProbe(
        ok=not errors,
        worker_address=address,
        hostname=hostname,
        pid=pid,
        slurm_job_id=slurm_job_id,
        package_versions=tuple(sorted(versions)),
        native_threads=tuple(sorted(threads)),
        dataset_visible=dataset_visible,
        output_writable=output_writable,
        scratch_usable=scratch_usable,
        memory_limit_bytes=memory_limit,
        network_host=network_host,
        errors=tuple(errors),
    )
    return probe.to_dict()


def _raw_identity(scheduler_address: object, payload: object) -> str:
    worker_address: object = "<unknown>"
    hostname: object = "<unknown>"
    job_id: object = "<unknown>"
    if type(payload) is dict:
        worker_address = payload.get("worker_address", "<unknown>")
        hostname = payload.get("hostname", "<unknown>")
        job_id = payload.get("slurm_job_id", "<unknown>")
    if job_id is None:
        job_id = "<none>"
    return (
        f"scheduler_address={scheduler_address!r}, "
        f"worker_address={worker_address!r}, hostname={hostname!r}, "
        f"SLURM_JOB_ID={job_id}"
    )


def validate_worker_probes(
    probes: Mapping[str, Mapping[str, object]],
) -> tuple[WorkerProbe, ...]:
    """Strictly decode all probes or raise one identity-rich aggregate failure."""
    if not isinstance(probes, Mapping):
        raise WorkerPreflightError(
            "Worker preflight evidence must be a mapping keyed by worker address"
        )
    decoded: list[tuple[str, WorkerProbe]] = []
    failures: list[str] = []
    try:
        items = sorted(probes.items(), key=lambda item: repr(item[0]))
    except Exception as error:
        raise WorkerPreflightError(
            "Could not inspect worker preflight evidence: "
            f"{type(error).__name__}: {error}"
        ) from error
    for scheduler_address, payload in items:
        identity = _raw_identity(scheduler_address, payload)
        if type(scheduler_address) is not str or not scheduler_address:
            failures.append(
                f"{identity}: scheduler worker address must be a nonempty "
                "built-in string"
            )
            continue
        try:
            probe = WorkerProbe.from_dict(payload)
        except (TypeError, ValueError) as error:
            failures.append(
                f"{identity}: invalid probe payload: {type(error).__name__}: {error}"
            )
            continue
        decoded.append((scheduler_address, probe))
        if not probe.ok:
            for error in probe.errors:
                failures.append(f"{identity}: {error}")
    if failures:
        details = "\n".join(f"- {failure}" for failure in failures)
        raise WorkerPreflightError(f"Worker preflight failed:\n{details}")
    decoded.sort(key=lambda item: item[0])
    return tuple(probe for _address, probe in decoded)


def validate_slurm_worker_identities(
    probes: tuple[WorkerProbe, ...],
) -> None:
    """Require one exact SLURM job and one host/process identity per worker."""
    if type(probes) is not tuple or any(
        type(probe) is not WorkerProbe for probe in probes
    ):
        raise TypeError("SLURM worker probes must be an exact tuple of WorkerProbe")
    job_ids: set[str] = set()
    worker_addresses: set[str] = set()
    host_processes: set[tuple[str, int]] = set()
    failures: list[str] = []
    for probe in probes:
        identity = (
            f"worker_address={probe.worker_address!r}, hostname={probe.hostname!r}, "
            f"pid={probe.pid!r}, SLURM_JOB_ID={probe.slurm_job_id!r}"
        )
        if probe.slurm_job_id is None:
            failures.append(f"{identity}: SLURM_JOB_ID is required")
        elif probe.slurm_job_id in job_ids:
            failures.append(f"{identity}: duplicate SLURM_JOB_ID")
        else:
            job_ids.add(probe.slurm_job_id)
        if probe.worker_address in worker_addresses:
            failures.append(f"{identity}: duplicate worker address")
        worker_addresses.add(probe.worker_address)
        host_process = (probe.hostname, probe.pid)
        if host_process in host_processes:
            failures.append(f"{identity}: duplicate host/process identity")
        host_processes.add(host_process)
    if failures:
        raise WorkerPreflightError(
            "SLURM worker identity validation failed:\n- " + "\n- ".join(failures)
        )


__all__ = [
    "WorkerProbe",
    "probe_worker",
    "validate_slurm_worker_identities",
    "validate_worker_probes",
]
