"""Owned, import-light SLURMCluster execution for coarse round tasks."""

from __future__ import annotations

import copy
import logging
import shlex
import sys
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version as package_version
from typing import Any, cast

from cyexm.config.models import ExecutionProfile, SlurmSpec
from cyexm.errors import ExecutorStartupError, WorkerPreflightError
from cyexm.execution._dask_common import (
    collect_distributed_results,
    future_is_unfinished,
    utc_now,
    validate_distributed_tasks,
)
from cyexm.execution.base import RoundTaskRunner
from cyexm.execution.preflight import (
    WorkerProbe,
    probe_worker,
    validate_worker_probes,
)
from cyexm.runtime import NATIVE_THREAD_ENV_VARS
from cyexm.workflow.models import RoundResult, RoundTask, canonical_json
from cyexm.workflow.round_task import run_round_task

_LOGGER = logging.getLogger(__name__)
_COORDINATOR_PACKAGES = (
    "cy-exm",
    "dask",
    "distributed",
    "dask-jobqueue",
    "antspyx",
    "numpy",
    "scipy",
    "tifffile",
)
_CONFIG_NAME = "cyexm-slurm"
_DEATH_TIMEOUT_SECONDS = 60
_SHEBANG = "#!/usr/bin/env bash"
_WORKER_COMMAND = "distributed.cli.dask_worker"
_WORKER_EXTRA_ARGS = ["--resources", "round_slot=1"]
_EVIDENCE_KEYS = frozenset(
    {
        "executor",
        "requested",
        "rendered",
        "observed",
        "startup",
        "attempt_counts",
        "timestamps",
        "cleanup",
    }
)
_REQUESTED_KEYS = frozenset(
    {"workers", "configured_workers", "startup_timeout", "task_retries"}
)
_OBSERVED_KEYS = frozenset(
    {
        "coordinator_versions",
        "scheduler_address",
        "connected_workers",
        "worker_probes",
        "cancellation",
    }
)
_STARTUP_PHASES = (
    "not_started",
    "module_loading",
    "configuration",
    "cluster_construction",
    "template_rendering",
    "scaling",
    "client_creation",
    "worker_wait",
    "coordinator_version_collection",
    "worker_snapshot",
    "worker_probe",
    "worker_validation",
    "worker_stability_check",
    "complete",
)
_CLEANUP_RESOURCES = ("client", "cluster", "configuration")


def _evidence_object(
    value: object,
    location: str,
    *,
    exact_keys: frozenset[str] | None = None,
    allowed_keys: frozenset[str] | None = None,
) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{location} must be an exact JSON object")
    result = cast(dict[object, object], value)
    for key in result:
        if type(key) is not str:
            raise TypeError(f"{location} keys must be exact built-in strings")
    keys = frozenset(cast(dict[str, object], result))
    if exact_keys is not None and keys != exact_keys:
        raise ValueError(
            f"{location} keys must be exactly {sorted(exact_keys)!r}; got "
            f"{sorted(keys)!r}"
        )
    if allowed_keys is not None and not keys <= allowed_keys:
        raise ValueError(
            f"{location} contains unsupported keys: {sorted(keys - allowed_keys)!r}"
        )
    return cast(dict[str, object], result)


def _evidence_array(value: object, location: str) -> list[object]:
    if type(value) is not list:
        raise TypeError(f"{location} must be an exact JSON array")
    return cast(list[object], value)


def _evidence_string(value: object, location: str, *, empty: bool = False) -> str:
    if type(value) is not str or (not empty and not value):
        qualifier = "" if empty else "nonempty "
        raise TypeError(f"{location} must be a {qualifier}exact built-in string")
    return value


def _evidence_integer(value: object, location: str, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise TypeError(
            f"{location} must be an exact built-in integer of at least {minimum}"
        )
    return value


def _evidence_boolean(value: object, location: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{location} must be an exact built-in boolean")
    return value


def _evidence_error(value: object, location: str) -> None:
    error = _evidence_object(
        value,
        location,
        exact_keys=frozenset({"type", "message"}),
    )
    _evidence_string(error["type"], f"{location}.type")
    _evidence_string(error["message"], f"{location}.message", empty=True)


def _evidence_timestamp(value: object, location: str) -> datetime:
    encoded = _evidence_string(value, location)
    if not encoded.endswith("Z"):
        raise ValueError(f"{location} must use canonical UTC Z notation")
    try:
        parsed = datetime.fromisoformat(encoded[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{location} must be an ISO 8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError(f"{location} must be UTC")
    return parsed


def _validate_requested(value: object) -> tuple[int, int | None, int | None]:
    requested = _evidence_object(value, "execution_evidence.requested")
    if frozenset(requested) == frozenset({"workers"}):
        workers = _evidence_integer(
            requested["workers"],
            "execution_evidence.requested.workers",
            minimum=0,
        )
        if workers != 0:
            raise ValueError(
                "minimal requested evidence is valid only for zero workers"
            )
        return workers, None, None
    if frozenset(requested) != _REQUESTED_KEYS:
        raise ValueError(
            "execution_evidence.requested keys must match the accepted SLURM schema"
        )
    workers = _evidence_integer(
        requested["workers"],
        "execution_evidence.requested.workers",
        minimum=1,
    )
    configured = _evidence_integer(
        requested["configured_workers"],
        "execution_evidence.requested.configured_workers",
        minimum=1,
    )
    if workers > configured:
        raise ValueError("requested workers cannot exceed configured_workers")
    _evidence_string(
        requested["startup_timeout"],
        "execution_evidence.requested.startup_timeout",
    )
    retries = _evidence_integer(
        requested["task_retries"],
        "execution_evidence.requested.task_retries",
        minimum=0,
    )
    return workers, configured, retries


def _validate_rendered(value: object) -> bool:
    rendered = _evidence_object(
        value,
        "execution_evidence.rendered",
        allowed_keys=frozenset({"rendered_job_script_template"}),
    )
    if not rendered:
        return False
    _evidence_string(
        rendered["rendered_job_script_template"],
        "execution_evidence.rendered.rendered_job_script_template",
    )
    return True


def _validate_versions(value: object) -> dict[str, str]:
    versions = _evidence_object(
        value, "execution_evidence.observed.coordinator_versions"
    )
    if frozenset(versions) != frozenset(_COORDINATOR_PACKAGES):
        raise ValueError(
            "coordinator_versions must contain the exact probed package set"
        )
    decoded: dict[str, str] = {}
    for name, version in versions.items():
        _evidence_string(name, "coordinator_versions package name")
        decoded[name] = _evidence_string(
            version,
            f"coordinator_versions[{name!r}]",
        )
    return decoded


def _validate_cancellation(value: object) -> None:
    cancellation = _evidence_object(
        value,
        "execution_evidence.observed.cancellation",
        allowed_keys=frozenset(
            {"attempted", "unfinished_futures", "force", "succeeded", "error"}
        ),
    )
    required = frozenset({"attempted", "unfinished_futures", "succeeded"})
    if not required <= frozenset(cancellation):
        raise ValueError("cancellation evidence is missing required keys")
    attempted = _evidence_boolean(
        cancellation["attempted"],
        "execution_evidence.observed.cancellation.attempted",
    )
    unfinished = _evidence_integer(
        cancellation["unfinished_futures"],
        "execution_evidence.observed.cancellation.unfinished_futures",
        minimum=0,
    )
    succeeded = cancellation["succeeded"]
    if not attempted:
        if (
            unfinished != 0
            or succeeded is not None
            or frozenset(cancellation) != required
        ):
            raise ValueError(
                "unattempted cancellation must have zero unfinished futures, null "
                "success, and no force/error claims"
            )
        return
    if unfinished < 1:
        raise ValueError("attempted cancellation requires unfinished futures")
    if cancellation.get("force") is not True:
        raise ValueError("attempted cancellation must truthfully record force=true")
    success = _evidence_boolean(
        succeeded,
        "execution_evidence.observed.cancellation.succeeded",
    )
    if success and "error" in cancellation:
        raise ValueError("successful cancellation cannot contain an error")
    if not success:
        if "error" not in cancellation:
            raise ValueError("failed cancellation must contain an error")
        _evidence_error(
            cancellation["error"],
            "execution_evidence.observed.cancellation.error",
        )


def _validate_observed(
    value: object,
    *,
    worker_count: int,
    startup_phase: str,
    startup_outcome: str,
) -> bool:
    observed = _evidence_object(
        value,
        "execution_evidence.observed",
        allowed_keys=_OBSERVED_KEYS,
    )
    versions: dict[str, str] | None = None
    if "coordinator_versions" in observed:
        versions = _validate_versions(observed["coordinator_versions"])
    scheduler_present = "scheduler_address" in observed
    workers_present = "connected_workers" in observed
    if scheduler_present != workers_present:
        raise ValueError(
            "scheduler_address and connected_workers must be recorded together"
        )
    connected: tuple[str, ...] = ()
    if scheduler_present:
        _evidence_string(
            observed["scheduler_address"],
            "execution_evidence.observed.scheduler_address",
        )
        worker_values = _evidence_array(
            observed["connected_workers"],
            "execution_evidence.observed.connected_workers",
        )
        connected = tuple(
            _evidence_string(
                address,
                f"execution_evidence.observed.connected_workers[{index}]",
            )
            for index, address in enumerate(worker_values)
        )
        if connected != tuple(sorted(set(connected))):
            raise ValueError("connected_workers must be unique and canonically sorted")
        if len(connected) != worker_count:
            raise ValueError("connected_workers must match requested workers")
    if "worker_probes" in observed:
        if not connected:
            raise ValueError("worker_probes require connected worker evidence")
        probe_values = _evidence_array(
            observed["worker_probes"],
            "execution_evidence.observed.worker_probes",
        )
        probes = tuple(
            WorkerProbe.from_dict(
                _evidence_object(
                    probe,
                    f"execution_evidence.observed.worker_probes[{index}]",
                )
            )
            for index, probe in enumerate(probe_values)
        )
        addresses = tuple(probe.worker_address for probe in probes)
        if addresses != tuple(sorted(set(addresses))):
            raise ValueError("worker_probes must be unique and sorted by address")
        validated_workers = _STARTUP_PHASES.index(
            startup_phase
        ) >= _STARTUP_PHASES.index("worker_stability_check")
        if validated_workers and addresses != connected:
            raise ValueError(
                "validated worker probe addresses must exactly match connected_workers"
            )
        if validated_workers and versions is not None:
            expected_versions = tuple(sorted(versions.items()))
            if any(probe.package_versions != expected_versions for probe in probes):
                raise ValueError(
                    "validated worker package versions must match coordinator_versions"
                )
        if validated_workers and any(not probe.ok for probe in probes):
            raise ValueError(
                "worker probes must all be ok after successful worker validation"
            )
    cancellation_present = "cancellation" in observed
    if cancellation_present:
        _validate_cancellation(observed["cancellation"])
    phase_index = _STARTUP_PHASES.index(startup_phase)
    if "coordinator_versions" in observed and phase_index < _STARTUP_PHASES.index(
        "worker_snapshot"
    ):
        raise ValueError(
            "coordinator versions were claimed before collection completed"
        )
    if scheduler_present and phase_index < _STARTUP_PHASES.index("worker_probe"):
        raise ValueError("scheduler workers were claimed before the worker snapshot")
    if "worker_probes" in observed and phase_index < _STARTUP_PHASES.index(
        "worker_probe"
    ):
        raise ValueError("worker probes were claimed before the probe phase")
    if startup_outcome == "interrupted" and not cancellation_present:
        raise ValueError("interrupted startup requires cancellation evidence")
    if startup_outcome not in {"interrupted", "succeeded"} and cancellation_present:
        raise ValueError("cancellation evidence requires an interrupted execution")
    completed_startup = startup_phase == "complete" and startup_outcome == "succeeded"
    if completed_startup:
        required = {
            "coordinator_versions",
            "scheduler_address",
            "connected_workers",
            "worker_probes",
        }
        if not required <= set(observed):
            raise ValueError("completed startup is missing observed preflight evidence")
    return cancellation_present


def _validate_startup(value: object) -> tuple[str, str]:
    startup = _evidence_object(
        value,
        "execution_evidence.startup",
        allowed_keys=frozenset(
            {"phase", "outcome", "scale_attempted", "scale_succeeded", "error"}
        ),
    )
    required = frozenset({"phase", "outcome", "scale_attempted", "scale_succeeded"})
    if not required <= frozenset(startup):
        raise ValueError("startup evidence is missing required keys")
    phase = _evidence_string(startup["phase"], "execution_evidence.startup.phase")
    if phase not in _STARTUP_PHASES:
        raise ValueError(f"unsupported startup phase: {phase!r}")
    outcome = _evidence_string(
        startup["outcome"],
        "execution_evidence.startup.outcome",
    )
    if outcome not in {"not_started", "succeeded", "failed", "interrupted"}:
        raise ValueError(f"unsupported terminal startup outcome: {outcome!r}")
    attempted = _evidence_boolean(
        startup["scale_attempted"],
        "execution_evidence.startup.scale_attempted",
    )
    succeeded = startup["scale_succeeded"]
    if succeeded is not None:
        _evidence_boolean(succeeded, "execution_evidence.startup.scale_succeeded")
    error_present = "error" in startup
    if outcome == "not_started":
        if (
            phase != "not_started"
            or attempted
            or succeeded is not None
            or error_present
        ):
            raise ValueError(
                "not_started outcome contains contradictory startup claims"
            )
    elif outcome == "succeeded":
        if (
            phase != "complete"
            or not attempted
            or succeeded is not True
            or error_present
        ):
            raise ValueError(
                "successful startup contains contradictory phase/scale/error"
            )
    else:
        if phase == "complete" or not error_present:
            raise ValueError(
                "failed/interrupted startup requires a non-complete phase and error"
            )
        _evidence_error(startup["error"], "execution_evidence.startup.error")
    phase_index = _STARTUP_PHASES.index(phase)
    scaling_index = _STARTUP_PHASES.index("scaling")
    if phase_index < scaling_index and (attempted or succeeded is not None):
        raise ValueError("startup claims scaling before the scaling phase")
    if phase_index == scaling_index and (not attempted or succeeded is not False):
        raise ValueError("terminal scaling-phase evidence must record a failed scale")
    if (
        phase_index > scaling_index
        and phase != "not_started"
        and (not attempted or succeeded is not True)
    ):
        raise ValueError(
            "post-scaling startup phase requires successful scale evidence"
        )
    return phase, outcome


def _validate_attempts(
    value: object,
    *,
    expected_identities: tuple[tuple[str, int], ...],
    task_retries: int,
    startup_outcome: str,
    cancellation_present: bool,
) -> None:
    attempts = _evidence_array(value, "execution_evidence.attempt_counts")
    decoded: list[tuple[str, int]] = []
    for index, item in enumerate(attempts):
        attempt = _evidence_object(
            item,
            f"execution_evidence.attempt_counts[{index}]",
            exact_keys=frozenset({"dataset_id", "round", "attempts"}),
        )
        dataset_id = _evidence_string(
            attempt["dataset_id"],
            f"execution_evidence.attempt_counts[{index}].dataset_id",
        )
        round_number = _evidence_integer(
            attempt["round"],
            f"execution_evidence.attempt_counts[{index}].round",
            minimum=0,
        )
        count = _evidence_integer(
            attempt["attempts"],
            f"execution_evidence.attempt_counts[{index}].attempts",
            minimum=1,
        )
        identity = (dataset_id, round_number)
        if identity in decoded:
            raise ValueError(f"duplicate attempt identity: {identity!r}")
        decoded.append(identity)
        if count > task_retries + 1:
            raise ValueError(
                "attempt count exceeds the configured executor retry policy"
            )
    decoded_identities = tuple(decoded)
    if startup_outcome != "succeeded":
        if decoded_identities:
            raise ValueError("round attempts cannot precede successful startup")
        return
    if cancellation_present:
        submitted = set(decoded_identities)
        canonical_subset = tuple(
            identity for identity in expected_identities if identity in submitted
        )
        if decoded_identities != canonical_subset:
            raise ValueError(
                "cancelled attempt evidence must be a canonical subset of submitted "
                "tasks"
            )
        return
    if decoded_identities != expected_identities:
        raise ValueError(
            "settled attempt evidence must exactly match submitted tasks in "
            "canonical order"
        )


def _validate_timestamps(value: object) -> None:
    timestamps = _evidence_object(
        value,
        "execution_evidence.timestamps",
        exact_keys=frozenset({"started_at", "ended_at"}),
    )
    started = _evidence_timestamp(
        timestamps["started_at"],
        "execution_evidence.timestamps.started_at",
    )
    ended = _evidence_timestamp(
        timestamps["ended_at"],
        "execution_evidence.timestamps.ended_at",
    )
    if ended < started:
        raise ValueError("execution evidence ended_at cannot precede started_at")


def _validate_cleanup(value: object, *, completed_startup: bool) -> None:
    cleanup = _evidence_object(
        value,
        "execution_evidence.cleanup",
        allowed_keys=frozenset((*_CLEANUP_RESOURCES, "order")),
    )
    resources = [name for name in _CLEANUP_RESOURCES if name in cleanup]
    if not resources:
        if cleanup:
            raise ValueError("cleanup order cannot exist without owned resources")
        if completed_startup:
            raise ValueError(
                "completed startup requires owned-resource cleanup evidence"
            )
        return
    order = _evidence_array(
        cleanup.get("order"),
        "execution_evidence.cleanup.order",
    )
    decoded_order = [
        _evidence_string(item, f"execution_evidence.cleanup.order[{index}]")
        for index, item in enumerate(order)
    ]
    if decoded_order != resources:
        raise ValueError(
            "cleanup order must match exact client/cluster/configuration order"
        )
    for name in resources:
        status = _evidence_object(
            cleanup[name],
            f"execution_evidence.cleanup.{name}",
            allowed_keys=frozenset({"attempted", "succeeded", "error"}),
        )
        if not {"attempted", "succeeded"} <= set(status):
            raise ValueError(f"cleanup {name} is missing attempted/succeeded")
        if (
            _evidence_boolean(
                status["attempted"],
                f"execution_evidence.cleanup.{name}.attempted",
            )
            is not True
        ):
            raise ValueError(f"cleanup {name} must have been attempted")
        succeeded = _evidence_boolean(
            status["succeeded"],
            f"execution_evidence.cleanup.{name}.succeeded",
        )
        if succeeded and "error" in status:
            raise ValueError(f"successful cleanup {name} cannot contain an error")
        if not succeeded:
            if "error" not in status:
                raise ValueError(f"failed cleanup {name} must contain an error")
            _evidence_error(
                status["error"],
                f"execution_evidence.cleanup.{name}.error",
            )
    if completed_startup and resources != list(_CLEANUP_RESOURCES):
        raise ValueError(
            "completed startup requires client, cluster, and configuration cleanup"
        )


def _canonical_execution_evidence(
    value: object,
    *,
    expected_tasks: tuple[RoundTask, ...],
    task_retries: int,
) -> str:
    """Validate and detach Task 2 evidence against the submitted work context."""
    if type(expected_tasks) is not tuple:
        raise TypeError("expected_tasks must be an exact tuple")
    expected_identities: list[tuple[str, int]] = []
    for index, task in enumerate(expected_tasks):
        if type(task) is not RoundTask:
            raise TypeError(f"expected_tasks[{index}] must be an exact RoundTask")
        identity = (task.dataset_id, task.round)
        if identity in expected_identities:
            raise ValueError(f"duplicate submitted task identity: {identity!r}")
        expected_identities.append(identity)
    retries = _evidence_integer(task_retries, "task_retries", minimum=0)
    evidence = _evidence_object(
        value,
        "execution_evidence",
        exact_keys=_EVIDENCE_KEYS,
    )
    if _evidence_string(evidence["executor"], "execution_evidence.executor") != "slurm":
        raise ValueError("execution_evidence.executor must be 'slurm'")
    workers, configured, requested_retries = _validate_requested(evidence["requested"])
    if requested_retries is not None and requested_retries != retries:
        raise ValueError(
            "execution evidence task_retries does not match the execution profile"
        )
    rendered = _validate_rendered(evidence["rendered"])
    phase, outcome = _validate_startup(evidence["startup"])
    completed = phase == "complete" and outcome == "succeeded"
    template_index = _STARTUP_PHASES.index("template_rendering")
    phase_index = _STARTUP_PHASES.index(phase)
    if phase_index > template_index and not rendered:
        raise ValueError(
            "post-template startup evidence is missing its rendered template"
        )
    if phase_index <= template_index and rendered:
        raise ValueError(
            "rendered template cannot precede successful template rendering"
        )
    if workers == 0:
        if configured is not None or outcome != "not_started" or expected_identities:
            raise ValueError("zero-worker evidence must remain not_started")
        for key in ("rendered", "observed", "attempt_counts", "cleanup"):
            if evidence[key] != {} and evidence[key] != []:
                raise ValueError(f"zero-worker evidence must leave {key} empty")
    elif not expected_identities:
        raise ValueError("nonzero worker evidence requires submitted tasks")
    cancellation_present = _validate_observed(
        evidence["observed"],
        worker_count=workers,
        startup_phase=phase,
        startup_outcome=outcome,
    )
    _validate_attempts(
        evidence["attempt_counts"],
        expected_identities=tuple(expected_identities),
        task_retries=retries,
        startup_outcome=outcome,
        cancellation_present=cancellation_present,
    )
    _validate_timestamps(evidence["timestamps"])
    _validate_cleanup(evidence["cleanup"], completed_startup=completed)
    return canonical_json(copy.deepcopy(evidence))


@dataclass(frozen=True, slots=True)
class _SlurmModules:
    SLURMCluster: Any
    Client: Any
    as_completed: Any
    parse_timedelta: Any
    config_set: Any


def _slurm_modules() -> _SlurmModules:
    """Load Dask and dask-jobqueue only when SLURM execution is selected."""
    from dask import config as dask_config
    from dask.utils import parse_timedelta
    from dask_jobqueue import SLURMCluster  # pyright: ignore[reportPrivateImportUsage]
    from distributed import Client, as_completed

    return _SlurmModules(
        SLURMCluster,
        Client,
        as_completed,
        parse_timedelta,
        dask_config.set,
    )


def _error_payload(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _native_thread_values(profile: ExecutionProfile) -> dict[str, str]:
    return {
        name: str(
            profile.native_threads.itk if index == 0 else profile.native_threads.blas
        )
        for index, name in enumerate(NATIVE_THREAD_ENV_VARS)
    }


def _native_thread_prologue(profile: ExecutionProfile) -> list[str]:
    return [
        f"export {name}={shlex.quote(value)}"
        for name, value in _native_thread_values(profile).items()
    ]


def _jobqueue_config_overlay() -> dict[str, object]:
    """Isolate worker rendering from ambient ``jobqueue.slurm`` settings."""
    values: dict[str, object] = {
        "account": None,
        "project": None,
        "cores": None,
        "memory": None,
        "processes": None,
        "python": None,
        "interface": None,
        "death-timeout": _DEATH_TIMEOUT_SECONDS,
        "local-directory": None,
        "shared-temp-directory": None,
        "extra": None,
        "worker-command": _WORKER_COMMAND,
        "worker-extra-args": [],
        "shebang": _SHEBANG,
        "queue": None,
        "walltime": None,
        "env-extra": None,
        "job-script-prologue": [],
        "job-cpu": None,
        "job-mem": None,
        "job-extra": None,
        "job-extra-directives": [],
        "header-skip": None,
        "job-directives-skip": [],
        "log-directory": None,
        "scheduler-options": {},
    }
    return {f"jobqueue.{_CONFIG_NAME}.{name}": value for name, value in values.items()}


def _scheduler_snapshot(
    client: Any,
    *,
    expected_workers: int | None,
) -> tuple[str, tuple[str, ...]]:
    try:
        info = client.scheduler_info()
    except Exception as error:
        raise WorkerPreflightError(
            "Could not inspect connected SLURM workers: "
            f"{type(error).__name__}: {error}"
        ) from error
    if not isinstance(info, Mapping):
        raise WorkerPreflightError(
            "Dask scheduler_info must be a mapping with address and workers"
        )
    scheduler_address = info.get("address")
    if type(scheduler_address) is not str or not scheduler_address:
        raise WorkerPreflightError(
            "Dask scheduler_info address must be a nonempty built-in string"
        )
    workers = info.get("workers")
    if not isinstance(workers, Mapping):
        raise WorkerPreflightError(
            "Dask scheduler_info workers must be a mapping keyed by worker address"
        )
    addresses: list[str] = []
    for address in workers:
        if type(address) is not str or not address:
            raise WorkerPreflightError(
                "Dask scheduler worker addresses must be nonempty built-in strings"
            )
        addresses.append(address)
    if len(addresses) != len(set(addresses)):
        raise WorkerPreflightError("Dask scheduler reported duplicate worker addresses")
    addresses.sort()
    if expected_workers is not None and len(addresses) != expected_workers:
        raise WorkerPreflightError(
            f"Worker preflight expected {expected_workers} connected workers; "
            f"scheduler reported {len(addresses)}: {addresses!r}"
        )
    return scheduler_address, tuple(addresses)


def _returned_probe_addresses(probes: object) -> tuple[str, ...]:
    if not isinstance(probes, Mapping):
        raise WorkerPreflightError(
            "Worker preflight responses must be a mapping keyed by worker address"
        )
    addresses: list[str] = []
    for address in probes:
        if type(address) is not str or not address:
            raise WorkerPreflightError(
                "Worker preflight returned worker addresses as nonempty built-in "
                "strings"
            )
        addresses.append(address)
    addresses.sort()
    return tuple(addresses)


def _decodable_probe_evidence(
    probes: Mapping[object, object],
) -> list[dict[str, object]]:
    evidence: list[tuple[str, dict[str, object]]] = []
    for address, payload in probes.items():
        if type(address) is not str or not isinstance(payload, Mapping):
            continue
        try:
            decoded = WorkerProbe.from_dict(cast(Mapping[str, object], payload))
        except (TypeError, ValueError):
            continue
        evidence.append((address, decoded.to_dict()))
    evidence.sort(key=lambda item: item[0])
    return [payload for _address, payload in evidence]


class SlurmExecutor:
    """Run independent rounds through an owned ``SLURMCluster``."""

    def __init__(
        self,
        profile: ExecutionProfile,
        *,
        _runner: RoundTaskRunner = run_round_task,
        _modules: Callable[[], Any] = _slurm_modules,
        _version: Callable[[str], str] = package_version,
    ) -> None:
        self.profile = profile
        self._runner = _runner
        self._modules = _modules
        self._version = _version
        self._execution_evidence: dict[str, object] = {}

    @property
    def execution_evidence(self) -> Mapping[str, object]:
        """Return a detached JSON-compatible copy of observed execution evidence."""
        return copy.deepcopy(self._execution_evidence)

    def _spec(self) -> SlurmSpec:
        if self.profile.executor != "slurm":
            raise ExecutorStartupError("SlurmExecutor requires a slurm profile")
        if not isinstance(self.profile.slurm, SlurmSpec):
            raise ExecutorStartupError("SlurmExecutor requires SlurmSpec settings")
        if self.profile.paths.worker_local_directory is None:
            raise ExecutorStartupError(
                "SlurmExecutor requires a worker_local_directory"
            )
        return self.profile.slurm

    def _coordinator_versions(self) -> dict[str, str]:
        versions: dict[str, str] = {}
        for name in _COORDINATOR_PACKAGES:
            try:
                value = self._version(name)
            except Exception as error:
                raise ExecutorStartupError(
                    f"Could not inspect coordinator package {name!r}: "
                    f"{type(error).__name__}: {error}"
                ) from error
            if type(value) is not str or not value:
                raise ExecutorStartupError(
                    f"Coordinator package {name!r} returned an invalid version: "
                    f"{value!r}"
                )
            versions[name] = value
        return versions

    def _begin_evidence(
        self,
        *,
        worker_count: int,
        spec: SlurmSpec | None,
    ) -> tuple[
        dict[str, object],
        dict[str, object],
        dict[str, object],
        dict[str, object],
        dict[str, object],
    ]:
        requested: dict[str, object] = {"workers": worker_count}
        if spec is not None:
            requested.update(
                {
                    "configured_workers": spec.workers,
                    "startup_timeout": spec.startup_timeout,
                    "task_retries": spec.task_retries,
                }
            )
        rendered: dict[str, object] = {}
        observed: dict[str, object] = {}
        startup: dict[str, object] = {
            "phase": "not_started" if spec is None else "module_loading",
            "outcome": "not_started" if spec is None else "in_progress",
            "scale_attempted": False,
            "scale_succeeded": None,
        }
        cleanup: dict[str, object] = {}
        self._execution_evidence = {
            "executor": "slurm",
            "requested": requested,
            "rendered": rendered,
            "observed": observed,
            "startup": startup,
            "attempt_counts": [],
            "timestamps": {"started_at": utc_now()},
            "cleanup": cleanup,
        }
        return (
            rendered,
            observed,
            startup,
            cleanup,
            cast(dict[str, object], self._execution_evidence),
        )

    @staticmethod
    def _attempt_evidence(
        tasks: tuple[RoundTask, ...],
        attempts: list[int],
    ) -> list[dict[str, object]]:
        return [
            {
                "dataset_id": task.dataset_id,
                "round": task.round,
                "attempts": attempts[index],
            }
            for index, task in enumerate(tasks)
            if attempts[index] > 0
        ]

    @staticmethod
    def _close_owned(
        client: Any | None,
        cluster: Any | None,
        configuration: AbstractContextManager[Any] | None,
        cleanup: dict[str, object],
    ) -> None:
        exit_info = sys.exc_info()
        order: list[str] = []

        def record(name: str, close: Callable[[], object]) -> None:
            order.append(name)
            status: dict[str, object] = {"attempted": True}
            try:
                close()
            except BaseException as error:
                status["succeeded"] = False
                status["error"] = _error_payload(error)
                _LOGGER.warning("Failed to close owned SLURM %s: %s", name, error)
            else:
                status["succeeded"] = True
            cleanup[name] = status

        for name, resource in (("client", client), ("cluster", cluster)):
            if resource is None:
                continue
            record(name, resource.close)
        if configuration is not None:
            record(
                "configuration",
                lambda: configuration.__exit__(*exit_info),
            )
        if order:
            cleanup["order"] = order

    def execute(self, tasks: tuple[RoundTask, ...]) -> tuple[RoundResult, ...]:
        if not tasks:
            _rendered, _observed, _startup, _cleanup, evidence = self._begin_evidence(
                worker_count=0,
                spec=None,
            )
            cast(dict[str, str], evidence["timestamps"])["ended_at"] = utc_now()
            return ()

        spec = self._spec()
        validate_distributed_tasks(tasks, executor_name="SlurmExecutor")
        worker_count = min(len(tasks), spec.workers)
        rendered, observed, startup, cleanup, evidence = self._begin_evidence(
            worker_count=worker_count,
            spec=spec,
        )
        attempts = [0 for _task in tasks]
        cluster: Any | None = None
        client: Any | None = None
        configuration: AbstractContextManager[Any] | None = None
        active: set[Any] = set()

        try:
            try:
                startup["phase"] = "module_loading"
                modules = self._modules()
                startup["phase"] = "configuration"
                opened_configuration = modules.config_set(_jobqueue_config_overlay())
                opened_configuration.__enter__()
                configuration = opened_configuration
                startup["phase"] = "cluster_construction"
                opened_cluster = modules.SLURMCluster(
                    config_name=_CONFIG_NAME,
                    queue=spec.queue,
                    cores=spec.cores_per_worker,
                    processes=spec.processes_per_worker,
                    memory=spec.memory,
                    walltime=spec.walltime,
                    job_cpu=spec.cores_per_worker,
                    job_mem=spec.memory,
                    interface=spec.interface,
                    scheduler_options={"interface": spec.scheduler_interface},
                    local_directory=str(self.profile.paths.worker_local_directory),
                    nanny=True,
                    death_timeout=_DEATH_TIMEOUT_SECONDS,
                    worker_command=_WORKER_COMMAND,
                    python=sys.executable,
                    shebang=_SHEBANG,
                    job_name=spec.job_name,
                    job_script_prologue=_native_thread_prologue(self.profile),
                    job_extra_directives=list(spec.job_extra_directives),
                    job_directives_skip=[],
                    worker_extra_args=list(_WORKER_EXTRA_ARGS),
                )
                cluster = opened_cluster
                startup["phase"] = "template_rendering"
                script = opened_cluster.job_script()
                if type(script) is not str or not script:
                    raise TypeError(
                        "SLURMCluster.job_script() must return a nonempty built-in "
                        "string"
                    )
                rendered["rendered_job_script_template"] = script
                startup["phase"] = "scaling"
                startup["scale_attempted"] = True
                try:
                    opened_cluster.scale(jobs=worker_count)
                except BaseException:
                    startup["scale_succeeded"] = False
                    raise
                startup["scale_succeeded"] = True
                startup["phase"] = "client_creation"
                opened_client = modules.Client(
                    opened_cluster,
                    timeout=spec.startup_timeout,
                )
                client = opened_client
                startup["phase"] = "worker_wait"
                opened_client.wait_for_workers(
                    worker_count,
                    timeout=modules.parse_timedelta(spec.startup_timeout),
                )

                startup["phase"] = "coordinator_version_collection"
                versions = self._coordinator_versions()
                observed["coordinator_versions"] = dict(versions)
                startup["phase"] = "worker_snapshot"
                scheduler_address, connected = _scheduler_snapshot(
                    opened_client,
                    expected_workers=worker_count,
                )
                observed["scheduler_address"] = scheduler_address
                observed["connected_workers"] = list(connected)
                startup["phase"] = "worker_probe"
                raw_probes = opened_client.run(
                    probe_worker,
                    dataset_root=str(self.profile.paths.dataset_root),
                    output_root=str(self.profile.paths.output_root),
                    worker_local_directory=str(
                        self.profile.paths.worker_local_directory
                    ),
                    expected_versions=dict(versions),
                    expected_threads=_native_thread_values(self.profile),
                )
                if isinstance(raw_probes, Mapping):
                    observed["worker_probes"] = _decodable_probe_evidence(raw_probes)
                returned = _returned_probe_addresses(raw_probes)
                if returned != connected:
                    raise WorkerPreflightError(
                        "Worker preflight returned worker addresses that do not "
                        "match the expected connected workers: "
                        f"expected={connected!r}, "
                        f"returned={returned!r}"
                    )
                startup["phase"] = "worker_validation"
                probes = validate_worker_probes(
                    cast(Mapping[str, Mapping[str, object]], raw_probes)
                )
                mismatches = [
                    f"scheduler address {address!r} reported worker address "
                    f"{probe.worker_address!r}"
                    for address, probe in zip(connected, probes, strict=True)
                    if probe.worker_address != address
                ]
                if mismatches:
                    raise WorkerPreflightError(
                        "Worker preflight reported worker address mismatches:\n- "
                        + "\n- ".join(mismatches)
                    )
                startup["phase"] = "worker_stability_check"
                scheduler_after, connected_after = _scheduler_snapshot(
                    opened_client,
                    expected_workers=None,
                )
                if scheduler_after != scheduler_address or connected_after != connected:
                    raise WorkerPreflightError(
                        "SLURM worker set changed during preflight: "
                        f"before=({scheduler_address!r}, {connected!r}), "
                        f"after=({scheduler_after!r}, {connected_after!r})"
                    )
                startup["phase"] = "complete"
                startup["outcome"] = "succeeded"
            except ExecutorStartupError as error:
                startup["outcome"] = "failed"
                startup["error"] = _error_payload(error)
                raise
            except Exception as error:
                startup["outcome"] = "failed"
                startup["error"] = _error_payload(error)
                raise ExecutorStartupError(
                    f"Failed to start and preflight {worker_count} SLURM workers "
                    f"within {spec.startup_timeout}: {error}"
                ) from error

            execution_client = client
            assert execution_client is not None
            results, _attempts = collect_distributed_results(
                tasks,
                runner=self._runner,
                client=execution_client,
                as_completed=modules.as_completed,
                task_retries=spec.task_retries,
                active=active,
                attempts=attempts,
            )
            return results
        except BaseException as error:
            if startup["outcome"] == "in_progress":
                startup["outcome"] = (
                    "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
                )
                startup["error"] = _error_payload(error)
            if isinstance(error, KeyboardInterrupt):
                unfinished: tuple[Any, ...] = ()
                if client is not None:
                    unfinished = tuple(
                        future
                        for future in active
                        if future_is_unfinished(future, logger=_LOGGER)
                    )
                cancellation: dict[str, object] = {
                    "attempted": False,
                    "unfinished_futures": len(unfinished),
                    "succeeded": None,
                }
                if client is not None and unfinished:
                    cancellation["attempted"] = True
                    cancellation["force"] = True
                    try:
                        client.cancel(unfinished, force=True)
                    except BaseException as cancellation_error:
                        cancellation["succeeded"] = False
                        cancellation["error"] = _error_payload(cancellation_error)
                        _LOGGER.warning(
                            "Failed to cancel unfinished SLURM futures: %s",
                            cancellation_error,
                        )
                    else:
                        cancellation["succeeded"] = True
                observed["cancellation"] = cancellation
            raise
        finally:
            evidence["attempt_counts"] = self._attempt_evidence(tasks, attempts)
            self._close_owned(client, cluster, configuration, cleanup)
            cast(dict[str, str], evidence["timestamps"])["ended_at"] = utc_now()


__all__ = ["SlurmExecutor"]
