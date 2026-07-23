"""Owned, import-light SLURMCluster execution for coarse and fine tasks."""

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

from cyexm.config.execution import _validate_execution_profile
from cyexm.config.memory import SlurmMemoryQuantity, normalize_slurm_memory
from cyexm.config.models import ExecutionProfile, SlurmSpec
from cyexm.errors import (
    ConfigurationValidationError,
    ExecutorStartupError,
    RoundExecutionError,
    WorkerPreflightError,
)
from cyexm.execution._dask_common import (
    ROUND_RESOURCE,
    FineAttemptObservation,
    collect_distributed_fine_results,
    collect_distributed_results,
    dask_scheduler_config_overlay,
    future_is_unfinished,
    utc_now,
    validate_distributed_tasks,
)
from cyexm.execution._slurm_evidence import (
    FinePlacementAttempt,
    FineScaleEvent,
    FineSlurmEvidence,
    PreControllerTerminal,
    validate_fine_slurm_evidence,
)
from cyexm.execution.base import FineWorkflowTaskRunner, WorkflowTaskRunner
from cyexm.execution.dispatcher import FineGraphController
from cyexm.execution.preflight import (
    WorkerProbe,
    probe_worker,
    validate_slurm_worker_identities,
    validate_worker_probes,
)
from cyexm.runtime import NATIVE_THREAD_ENV_VARS
from cyexm.workflow.fine_models import FineTaskResult, FineWorkflowTask
from cyexm.workflow.models import (
    ReferenceTask,
    RoundResult,
    RoundTask,
    WorkflowTask,
    canonical_json,
)
from cyexm.workflow.task import run_workflow_task

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
_FINE_EVIDENCE_KEYS = _EVIDENCE_KEYS | frozenset({"schema_version", "fine_graph"})
_REQUESTED_KEYS = frozenset(
    {"workers", "configured_workers", "startup_timeout", "task_retries", "memory"}
)
_MEMORY_EVIDENCE_KEYS = frozenset(
    {"profile", "semantic_bytes", "scheduler", "scheduler_bytes", "dask"}
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
    "template_validation",
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


@dataclass(frozen=True, slots=True)
class _SerializedWorkflowTaskRunner:
    payload: bytes

    def __call__(self, task: WorkflowTask) -> RoundResult:
        import cloudpickle

        if type(task) not in {ReferenceTask, RoundTask}:
            raise TypeError("serialized coarse task must use an exact task type")
        runner = cloudpickle.loads(self.payload)
        if not callable(runner):
            raise TypeError("serialized coarse runner payload must be callable")
        result = runner(task)
        if type(result) is not RoundResult:
            raise TypeError("serialized coarse runner must return an exact RoundResult")
        return result


def _serialize_workflow_task_runner(
    runner: WorkflowTaskRunner,
) -> _SerializedWorkflowTaskRunner:
    import cloudpickle

    return _SerializedWorkflowTaskRunner(cloudpickle.dumps(runner))


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


def _memory_evidence(quantity: SlurmMemoryQuantity) -> dict[str, object]:
    return {
        "profile": quantity.profile,
        "semantic_bytes": quantity.semantic_bytes,
        "scheduler": quantity.scheduler,
        "scheduler_bytes": quantity.scheduler_bytes,
        "dask": quantity.dask,
    }


def _validate_memory_evidence(value: object) -> None:
    location = "execution_evidence.requested.memory"
    memory = _evidence_object(value, location, exact_keys=_MEMORY_EVIDENCE_KEYS)
    profile = _evidence_string(memory["profile"], f"{location}.profile")
    _evidence_integer(memory["semantic_bytes"], f"{location}.semantic_bytes", minimum=1)
    _evidence_string(memory["scheduler"], f"{location}.scheduler")
    _evidence_integer(
        memory["scheduler_bytes"],
        f"{location}.scheduler_bytes",
        minimum=1,
    )
    _evidence_string(memory["dask"], f"{location}.dask")
    try:
        expected = _memory_evidence(
            normalize_slurm_memory(profile, f"{location}.profile")
        )
    except ConfigurationValidationError as error:
        raise ValueError(str(error)) from error
    if memory != expected:
        raise ValueError(
            f"{location} must exactly match normalized profile memory; "
            f"expected {expected!r}, got {memory!r}"
        )


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
    _validate_memory_evidence(requested["memory"])
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


def _validate_cancellation(value: object) -> int | None:
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
        return None
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
    return unfinished


def _validate_observed(
    value: object,
    *,
    worker_count: int,
    startup_phase: str,
    startup_outcome: str,
    fine_mode: bool,
) -> tuple[bool, int | None]:
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
        validated_workers = _STARTUP_PHASES.index(
            startup_phase
        ) >= _STARTUP_PHASES.index("worker_stability_check")
        if validated_workers and (
            (fine_mode and not set(addresses) <= set(connected))
            or (not fine_mode and addresses != connected)
        ):
            raise ValueError(
                "worker probe addresses must be a valid subset of connected_workers"
            )
        if validated_workers and versions is not None:
            expected_versions = tuple(sorted(versions.items()))
            if any(probe.package_versions != expected_versions for probe in probes):
                raise ValueError(
                    "validated worker package versions must match coordinator_versions"
                )
        if (
            validated_workers
            and not fine_mode
            and any(not probe.ok for probe in probes)
        ):
            raise ValueError(
                "worker probes must all be ok after successful worker validation"
            )
    cancellation_present = "cancellation" in observed
    cancellation_unfinished: int | None = None
    if cancellation_present:
        cancellation_unfinished = _validate_cancellation(observed["cancellation"])
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
    return cancellation_present, cancellation_unfinished


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
    cancellation_unfinished: int | None,
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
    if cancellation_unfinished is not None:
        if not decoded_identities:
            raise ValueError("attempted cancellation requires an accepted task attempt")
        if cancellation_unfinished > len(decoded_identities):
            raise ValueError(
                "unfinished cancellation futures cannot exceed accepted task attempts"
            )
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
    expected_tasks: tuple[WorkflowTask, ...],
    task_retries: int,
    expected_results: tuple[RoundResult, ...] = (),
    fine_task_results: tuple[FineTaskResult, ...] = (),
    expected_interruption: tuple[str, str] | None = None,
) -> str:
    """Validate and detach Task 2 evidence against the submitted work context."""
    if type(expected_tasks) is not tuple:
        raise TypeError("expected_tasks must be an exact tuple")
    expected_identities: list[tuple[str, int]] = []
    for index, task in enumerate(expected_tasks):
        if type(task) not in {ReferenceTask, RoundTask}:
            raise TypeError(f"expected_tasks[{index}] must be an exact WorkflowTask")
        identity = (task.dataset_id, task.round)
        if identity in expected_identities:
            raise ValueError(f"duplicate submitted task identity: {identity!r}")
        expected_identities.append(identity)
    retries = _evidence_integer(task_retries, "task_retries", minimum=0)
    raw_evidence = _evidence_object(value, "execution_evidence")
    fine_mode = "schema_version" in raw_evidence or "fine_graph" in raw_evidence
    evidence = _evidence_object(
        value,
        "execution_evidence",
        exact_keys=_FINE_EVIDENCE_KEYS if fine_mode else _EVIDENCE_KEYS,
    )
    if (
        fine_mode
        and _evidence_integer(
            evidence["schema_version"],
            "execution_evidence.schema_version",
            minimum=2,
        )
        != 2
    ):
        raise ValueError("fine SLURM execution evidence schema_version must equal 2")
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
    cancellation_present, cancellation_unfinished = _validate_observed(
        evidence["observed"],
        worker_count=workers,
        startup_phase=phase,
        startup_outcome=outcome,
        fine_mode=fine_mode,
    )
    attempt_identities = (
        tuple(expected_identities[:1]) if fine_mode else tuple(expected_identities)
    )
    _validate_attempts(
        evidence["attempt_counts"],
        expected_identities=attempt_identities,
        task_retries=retries,
        startup_outcome=outcome,
        cancellation_present=cancellation_present,
        cancellation_unfinished=cancellation_unfinished,
    )
    _validate_timestamps(evidence["timestamps"])
    _validate_cleanup(evidence["cleanup"], completed_startup=completed)
    if fine_mode:
        if configured is None:
            raise ValueError("fine SLURM evidence requires configured_workers")
        raw_probes = _evidence_array(
            cast(Mapping[str, object], evidence["observed"]).get("worker_probes"),
            "execution_evidence.observed.worker_probes",
        )
        validate_fine_slurm_evidence(
            evidence["fine_graph"],
            configured_workers=configured,
            task_retries=retries,
            cancellation_present=cancellation_present,
            startup_outcome=outcome,
            reference_attempted=bool(
                _evidence_array(
                    evidence["attempt_counts"],
                    "execution_evidence.attempt_counts",
                )
            ),
            round_tasks=tuple(cast(RoundTask, task) for task in expected_tasks[1:]),
            observed_workers=tuple(
                cast(
                    list[str],
                    _evidence_array(
                        cast(Mapping[str, object], evidence["observed"]).get(
                            "connected_workers",
                            [],
                        ),
                        "execution_evidence.observed.connected_workers",
                    ),
                )
            ),
            observed_probes=tuple(
                WorkerProbe.from_dict(
                    _evidence_object(
                        raw,
                        f"execution_evidence.observed.worker_probes[{index}]",
                    )
                )
                for index, raw in enumerate(raw_probes)
            ),
        )
        fine_graph = cast(Mapping[str, object], evidence["fine_graph"])
        terminal = fine_graph["pre_controller_terminal"]
        interruption_cause = fine_graph["interruption"]
        if (expected_interruption is None) != (interruption_cause is None):
            raise ValueError(
                "fine SLURM cancellation must match the caught interruption"
            )
        if (
            isinstance(interruption_cause, Mapping)
            and (
                interruption_cause.get("exception_type"),
                interruption_cause.get("message"),
            )
            != expected_interruption
        ):
            raise ValueError(
                "fine SLURM caught interruption cause is not authoritative"
            )
        reference_results = tuple(
            result
            for result in expected_results
            if result.round == expected_tasks[0].round
        )
        if isinstance(terminal, Mapping):
            phase = terminal.get("phase")
            if phase == "reference":
                if len(reference_results) > 1 or (
                    reference_results
                    and terminal.get("status") != reference_results[0].status
                ):
                    raise ValueError(
                        "fine SLURM reference terminal contradicts public result"
                    )
                if reference_results and reference_results[0].failure is not None:
                    failure = reference_results[0].failure
                    if (
                        terminal.get("exception_type"),
                        terminal.get("message"),
                    ) != (failure.exception_type, failure.message):
                        raise ValueError(
                            "fine SLURM reference terminal cause is not authoritative"
                        )
            elif phase == "moving":
                has_moving = any(type(task) is RoundTask for task in expected_tasks)
                reused_terminal = terminal.get("status") == "reused" and not has_moving
                cancelled_terminal = (
                    terminal.get("status") == "cancelled" and has_moving
                )
                if (
                    reference_results == ()
                    or reference_results[0].status not in {"completed", "reused"}
                    or (not reused_terminal and not cancelled_terminal)
                ):
                    raise ValueError(
                        "fine SLURM moving terminal contradicts submitted work"
                    )
            if (
                terminal.get("status") == "cancelled"
                and (
                    terminal.get("exception_type"),
                    terminal.get("message"),
                )
                != expected_interruption
            ):
                raise ValueError("fine SLURM cancellation cause is not authoritative")
        expected_rounds = {
            task.round: task for task in expected_tasks if type(task) is RoundTask
        }
        public_moving = tuple(
            result for result in expected_results if result.round in expected_rounds
        )
        if type(fine_task_results) is not tuple or any(
            type(result) is not FineTaskResult for result in fine_task_results
        ):
            raise TypeError("fine_task_results must be exact FineTaskResult values")
        if public_moving or fine_task_results:
            from cyexm.execution.dispatcher import _summarize_fine_round
            from cyexm.workflow._execution_validation import (
                _validate_fine_task_result,
            )

            derived = tuple(
                _summarize_fine_round(
                    expected_rounds[summary.round],
                    tuple(
                        result
                        for result in fine_task_results
                        if result.round == summary.round
                    ),
                )
                for summary in public_moving
            )
            if derived != public_moving:
                raise ValueError(
                    "fine SLURM public summaries must exactly equal canonical "
                    "granular aggregation"
                )
            placements = cast(
                list[object],
                cast(Mapping[str, object], evidence["fine_graph"])["placements"],
            )
            final_placements = {
                cast(str, item["task_key"]): item
                for raw in placements
                if isinstance(raw, Mapping)
                for item in [cast(Mapping[str, object], raw)]
            }
            result_keys = tuple(result.key.encoded for result in fine_task_results)
            if result_keys != tuple(final_placements):
                raise ValueError(
                    "fine SLURM granular results must uniquely and canonically "
                    "cover final placements"
                )
            graph_dependencies = {
                cast(str, item["key"]): tuple(cast(list[str], item["dependencies"]))
                for raw in cast(
                    list[object],
                    cast(Mapping[str, object], evidence["fine_graph"])["task_graph"],
                )
                if isinstance(raw, Mapping)
                for item in [cast(Mapping[str, object], raw)]
            }
            for result in fine_task_results:
                _validate_fine_task_result(result.task, result)
                if result.task.round_task != expected_rounds.get(result.round) or tuple(
                    dependency.encoded for dependency in result.task.dependencies
                ) != graph_dependencies.get(result.key.encoded):
                    raise ValueError(
                        "fine SLURM granular result is not bound to the exact "
                        "submitted task graph"
                    )
                placement = final_placements.get(result.key.encoded)
                if placement is None:
                    raise ValueError(
                        "fine SLURM granular result lacks placement evidence"
                    )
                raw = result.to_dict()
                for field in (
                    "status",
                    "dask_task_key",
                    "started_at",
                    "ended_at",
                    "duration_seconds",
                    "hostname",
                    "pid",
                    "slurm_job_id",
                    "native_threads",
                    "artifact_paths",
                    "reused_artifacts",
                    "log_path",
                    "failure",
                ):
                    if placement[field] != raw[field]:
                        raise ValueError(
                            "fine SLURM granular result contradicts terminal "
                            f"placement field {field}"
                        )
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
    from dask_jobqueue import (  # pyright: ignore[reportMissingImports]
        SLURMCluster,  # pyright: ignore[reportPrivateImportUsage]
    )
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
    jobqueue_values: dict[str, object] = {
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
    values = dask_scheduler_config_overlay()
    values.update(
        {
            f"jobqueue.{_CONFIG_NAME}.{name}": value
            for name, value in jobqueue_values.items()
        }
    )
    return values


def _rendered_script_error(detail: str) -> ValueError:
    return ValueError(f"rendered SLURM job script is unsafe: {detail}")


def _worker_option_values(
    tokens: list[str],
) -> tuple[dict[str, list[str]], list[str]]:
    valued = {
        "--name",
        "--nthreads",
        "--memory-limit",
        "--death-timeout",
        "--local-directory",
        "--resources",
        "--interface",
    }
    flags = {"--nanny"}
    values = {name: [] for name in valued}
    observed_flags: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in flags:
            observed_flags.append(token)
            index += 1
            continue
        option, separator, attached = token.partition("=")
        if option not in valued:
            raise _rendered_script_error(f"unexpected worker option {token!r}")
        if separator:
            if not attached:
                raise _rendered_script_error(f"worker option {option!r} is empty")
            values[option].append(attached)
            index += 1
            continue
        if index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
            raise _rendered_script_error(f"worker option {option!r} has no value")
        values[option].append(tokens[index + 1])
        index += 2
    return values, observed_flags


def validate_slurm_job_script(script: str, profile: ExecutionProfile) -> int:
    """Validate the template and return its exact Dask worker memory bytes."""
    if type(script) is not str or not script.strip():
        raise _rendered_script_error("template must be a nonempty built-in string")
    spec = profile.slurm
    scratch = profile.paths.worker_local_directory
    if (
        profile.executor != "slurm"
        or not isinstance(spec, SlurmSpec)
        or scratch is None
    ):
        raise _rendered_script_error("validation requires a complete SLURM profile")
    memory = normalize_slurm_memory(spec.memory)
    lines = [line.strip() for line in script.splitlines() if line.strip()]
    if not lines or lines[0] != _SHEBANG:
        raise _rendered_script_error(f"expected shebang {_SHEBANG!r}")
    expected_headers = [
        f"#SBATCH -J {spec.job_name}",
        f"#SBATCH -p {spec.queue}",
        "#SBATCH -n 1",
        f"#SBATCH --cpus-per-task={spec.cores_per_worker}",
        f"#SBATCH --mem={memory.scheduler}",
        f"#SBATCH -t {spec.walltime}",
        *(f"#SBATCH {directive}" for directive in spec.job_extra_directives),
    ]
    observed_headers = [line for line in lines if line.startswith("#SBATCH")]
    if observed_headers != expected_headers:
        raise _rendered_script_error(
            f"headers must be exactly {expected_headers!r}; got {observed_headers!r}"
        )

    normalized = script.replace("\\\n", " ")
    worker_lines = [
        line.strip()
        for line in normalized.splitlines()
        if " -m distributed.cli.dask_worker " in f" {line.strip()} "
    ]
    if len(worker_lines) != 1:
        raise _rendered_script_error(
            f"expected exactly one Dask worker command; got {len(worker_lines)}"
        )
    try:
        command = shlex.split(worker_lines[0])
    except ValueError as error:
        raise _rendered_script_error(
            f"worker command cannot be parsed: {error}"
        ) from error
    module_indexes = [
        index
        for index, token in enumerate(command[:-1])
        if token == "-m" and command[index + 1] == _WORKER_COMMAND
    ]
    if len(module_indexes) != 1:
        raise _rendered_script_error("worker command must launch the managed module")
    module_index = module_indexes[0]
    if module_index + 2 >= len(command):
        raise _rendered_script_error("worker command has no scheduler address")
    worker_arguments = command[module_index + 3 :]
    values, flags = _worker_option_values(worker_arguments)
    expected_values: dict[str, list[str] | None] = {
        "--name": None,
        "--nthreads": [str(spec.cores_per_worker)],
        "--memory-limit": None,
        "--death-timeout": [str(_DEATH_TIMEOUT_SECONDS)],
        "--local-directory": [str(scratch)],
        "--resources": (
            [] if profile.parallelism.mode == "fine_grained" else ["round_slot=1"]
        ),
        "--interface": [spec.interface],
    }
    for option, expected in expected_values.items():
        observed = values[option]
        expected_count = 0 if expected == [] else 1
        if len(observed) != expected_count or (
            expected is not None and observed != expected
        ):
            raise _rendered_script_error(
                f"worker option {option!r} must occur once with {expected!r}; "
                f"got {observed!r}"
            )
    worker_memory = values["--memory-limit"][0]
    try:
        from dask.utils import parse_bytes

        worker_memory_bytes = parse_bytes(worker_memory)
    except (TypeError, ValueError) as error:
        raise _rendered_script_error(
            f"worker memory limit cannot be parsed: {worker_memory!r}"
        ) from error
    if (
        type(worker_memory_bytes) is not int
        or worker_memory_bytes != memory.scheduler_bytes
    ):
        raise _rendered_script_error(
            "worker memory limit must equal the scheduler-normalized request "
            f"{memory.scheduler_bytes} bytes; got {worker_memory!r} "
            f"({worker_memory_bytes!r} bytes)"
        )
    if flags != ["--nanny"]:
        raise _rendered_script_error(
            f"worker flags must be exactly ['--nanny']; got {flags!r}"
        )
    return worker_memory_bytes


def _activate_round_slot(
    *,
    worker_address: str,
    dask_scheduler: Any,
) -> None:
    """Make one preflighted worker eligible for scientific work."""
    dask_scheduler.add_resources(
        worker_address,
        resources=dict(ROUND_RESOURCE),
    )


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


def _validate_scheduler_resources(
    client: Any,
    *,
    expected_scheduler: str,
    candidates: tuple[str, ...],
    validated: tuple[str, ...],
) -> None:
    try:
        info = client.scheduler_info()
        workers = info["workers"]
    except Exception as error:
        raise WorkerPreflightError(
            "Could not inspect gated SLURM worker resources: "
            f"{type(error).__name__}: {error}"
        ) from error
    if not isinstance(workers, Mapping):
        raise WorkerPreflightError("Dask scheduler workers must be a mapping")
    if info.get("address") != expected_scheduler:
        raise WorkerPreflightError(
            "SLURM scheduler identity changed during resource activation: "
            f"expected={expected_scheduler!r}, observed={info.get('address')!r}"
        )
    expected = set(candidates) | set(validated)
    if set(workers) != expected:
        raise WorkerPreflightError(
            "SLURM worker set changed during resource activation: "
            f"expected={sorted(expected)!r}, observed={sorted(workers)!r}"
        )
    for address, expected_slot in (
        *((address, 0) for address in candidates),
        *((address, 1) for address in validated),
    ):
        details = workers[address]
        resources = details.get("resources") if isinstance(details, Mapping) else None
        if not isinstance(resources, Mapping):
            raise WorkerPreflightError(
                f"Worker {address!r} has no scheduler resource mapping"
            )
        observed = resources.get("round_slot", 0)
        if observed != expected_slot:
            raise WorkerPreflightError(
                f"Worker {address!r} must expose round_slot={expected_slot}; "
                f"got {observed!r}"
            )


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


def _decodable_probe_batch(probes: object) -> dict[str, WorkerProbe]:
    if not isinstance(probes, Mapping):
        return {}
    decoded: dict[str, WorkerProbe] = {}
    for address, payload in probes.items():
        if type(address) is not str or not isinstance(payload, Mapping):
            continue
        try:
            decoded[address] = WorkerProbe.from_dict(
                cast(Mapping[str, object], payload)
            )
        except (TypeError, ValueError):
            continue
    return decoded


def _validated_probe_batch(
    raw_probes: object,
    *,
    expected_addresses: tuple[str, ...],
    prior_probes: Mapping[str, WorkerProbe] | None = None,
) -> dict[str, WorkerProbe]:
    """Validate exact scheduler keys, embedded identities, and SLURM uniqueness."""
    returned = _returned_probe_addresses(raw_probes)
    if returned != expected_addresses:
        raise WorkerPreflightError(
            "Worker preflight returned worker addresses that do not match the "
            f"expected connected workers: expected={expected_addresses!r}, "
            f"returned={returned!r}"
        )
    probes = validate_worker_probes(
        cast(Mapping[str, Mapping[str, object]], raw_probes)
    )
    mismatches = [
        f"scheduler address {address!r} reported worker address "
        f"{probe.worker_address!r}"
        for address, probe in zip(expected_addresses, probes, strict=True)
        if probe.worker_address != address
    ]
    if mismatches:
        raise WorkerPreflightError(
            "Worker preflight reported worker address mismatches:\n- "
            + "\n- ".join(mismatches)
        )
    combined = (*(() if prior_probes is None else prior_probes.values()), *probes)
    validate_slurm_worker_identities(tuple(combined))
    return dict(zip(expected_addresses, probes, strict=True))


class SlurmExecutor:
    """Execute coarse rounds or fine stages through an owned ``SLURMCluster``.

    The normalized profile selects the existing round-level runner or the shared
    coordinator-owned fine graph. This class owns cluster scaling, worker
    preflight, resource activation, task placement, cancellation, and cleanup;
    workers execute one immutable task and never submit descendants.

    Parameters
    ----------
    profile : ExecutionProfile
        Exact normalized ``slurm`` profile. ``slurm.workers`` is the global cap,
        and fine candidates receive a scheduler task slot only after exact
        preflight and identity validation.

    Raises
    ------
    ExecutorStartupError
        If the profile, scheduler startup, scaling, or worker lifecycle is invalid.
    WorkerPreflightError
        If a candidate worker cannot satisfy the exact placement and environment
        contract.

    See Also
    --------
    SequentialExecutor :
        Execute the same normalized graph without scheduler resources.
    LocalDaskExecutor :
        Execute the same graph in a bounded local process cluster.

    Notes
    -----
    The reference barrier and all moving work share one owned scheduler lifecycle.
    Fine mode records every scale event and settled placement attempt, including
    rejected or retried attempts, before cleanup. This operational evidence never
    authorizes scientific artifact reuse.
    """

    def __init__(
        self,
        profile: ExecutionProfile,
        *,
        _runner: WorkflowTaskRunner = run_workflow_task,
        _fine_runner: FineWorkflowTaskRunner | None = None,
        _modules: Callable[[], Any] = _slurm_modules,
        _version: Callable[[str], str] = package_version,
    ) -> None:
        self.profile = profile
        self._runner = _runner
        self._fine_runner = _fine_runner
        self._modules = _modules
        self._version = _version
        self._execution_evidence: dict[str, object] = {}
        self._fine_task_results: tuple[FineTaskResult, ...] = ()
        self._fine_attempt_counts: tuple[tuple[str, int], ...] = ()
        self._fine_reference_result: RoundResult | None = None
        self._fine_round_results: tuple[RoundResult, ...] = ()

    @property
    def execution_evidence(self) -> Mapping[str, object]:
        """Return a detached JSON-compatible copy of observed execution evidence."""
        return copy.deepcopy(self._execution_evidence)

    @property
    def fine_task_results(self) -> tuple[FineTaskResult, ...]:
        """Return exact settled granular results for public-summary validation."""
        return self._fine_task_results

    @property
    def fine_reference_result(self) -> RoundResult | None:
        """Return the exact settled reference result, when one exists."""
        return self._fine_reference_result

    @property
    def fine_round_results(self) -> tuple[RoundResult, ...]:
        """Return exact terminal moving summaries retained by the controller."""
        return self._fine_round_results

    def _spec(self) -> SlurmSpec:
        if self.profile.executor != "slurm":
            raise ExecutorStartupError("SlurmExecutor requires a slurm profile")
        if not isinstance(self.profile.slurm, SlurmSpec):
            raise ExecutorStartupError("SlurmExecutor requires SlurmSpec settings")
        if self.profile.paths.worker_local_directory is None:
            raise ExecutorStartupError(
                "SlurmExecutor requires a worker_local_directory"
            )
        try:
            _validate_execution_profile(self.profile)
        except ConfigurationValidationError as error:
            raise ExecutorStartupError(
                f"SlurmExecutor execution profile is invalid: {error}"
            ) from error
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
        memory: SlurmMemoryQuantity | None,
    ) -> tuple[
        dict[str, object],
        dict[str, object],
        dict[str, object],
        dict[str, object],
        dict[str, object],
    ]:
        requested: dict[str, object] = {"workers": worker_count}
        if spec is not None and memory is not None:
            requested.update(
                {
                    "configured_workers": spec.workers,
                    "startup_timeout": spec.startup_timeout,
                    "task_retries": spec.task_retries,
                    "memory": _memory_evidence(memory),
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
        tasks: tuple[WorkflowTask, ...],
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

    def execute(
        self,
        reference_task: ReferenceTask,
        round_tasks: tuple[RoundTask, ...],
    ) -> tuple[RoundResult, ...]:
        spec = self._spec()
        memory = normalize_slurm_memory(spec.memory)
        tasks: tuple[WorkflowTask, ...] = (reference_task, *round_tasks)
        validate_distributed_tasks(tasks, executor_name="SlurmExecutor")
        submission_runner = _serialize_workflow_task_runner(self._runner)
        fine_mode = self.profile.parallelism.mode == "fine_grained"
        worker_count = 1 if fine_mode else min(max(1, len(round_tasks)), spec.workers)
        rendered, observed, startup, cleanup, evidence = self._begin_evidence(
            worker_count=worker_count,
            spec=spec,
            memory=memory,
        )
        fine_state: FineSlurmEvidence | None = None
        if fine_mode:
            fine_state = FineSlurmEvidence(spec.workers)
            evidence["schema_version"] = 2
            evidence["fine_graph"] = fine_state.to_dict()
        reference_attempts = [0]
        moving_attempts = [0 for _task in round_tasks]
        cluster: Any | None = None
        client: Any | None = None
        configuration: AbstractContextManager[Any] | None = None
        active: set[Any] = set()
        validated_workers: set[str] = set()
        validated_probes: dict[str, WorkerProbe] = {}
        observed_probes: dict[str, WorkerProbe] = {}
        requested_workers = worker_count
        controller: FineGraphController | None = None
        self._fine_task_results = ()
        self._fine_attempt_counts = ()
        self._fine_reference_result = None
        self._fine_round_results = ()

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
                    memory=memory.dask,
                    walltime=spec.walltime,
                    job_cpu=spec.cores_per_worker,
                    job_mem=memory.scheduler,
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
                    worker_extra_args=([] if fine_mode else list(_WORKER_EXTRA_ARGS)),
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
                startup["phase"] = "template_validation"
                validate_slurm_job_script(script, self.profile)
                startup["phase"] = "scaling"
                startup["scale_attempted"] = True
                initial_scale_event: FineScaleEvent | None = None
                if fine_state is not None:
                    initial_scale_event = fine_state.begin_scale(
                        worker_count, utc_now()
                    )
                try:
                    opened_cluster.scale(jobs=worker_count)
                except BaseException as error:
                    startup["scale_succeeded"] = False
                    if initial_scale_event is not None:
                        initial_scale_event.fail(error, utc_now())
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
                if initial_scale_event is not None:
                    initial_scale_event.connected.extend(connected)
                    assert fine_state is not None
                    fine_state.candidates.update(connected)
                startup["phase"] = "worker_probe"
                probe_arguments: dict[str, object] = {
                    "dataset_root": str(self.profile.paths.dataset_root),
                    "output_root": str(self.profile.paths.output_root),
                    "worker_local_directory": str(
                        self.profile.paths.worker_local_directory
                    ),
                    "expected_versions": dict(versions),
                    "expected_threads": _native_thread_values(self.profile),
                }
                if fine_mode:
                    probe_arguments["workers"] = list(connected)
                raw_probes = opened_client.run(probe_worker, **probe_arguments)
                if isinstance(raw_probes, Mapping):
                    initial_decoded = _decodable_probe_batch(raw_probes)
                    if fine_mode:
                        initial_decoded = {
                            address: probe
                            for address, probe in initial_decoded.items()
                            if probe.worker_address == address
                        }
                    observed_probes.update(initial_decoded)
                    observed["worker_probes"] = [
                        observed_probes[address].to_dict()
                        for address in sorted(observed_probes)
                    ]
                    if initial_scale_event is not None:
                        initial_scale_event.probed.extend(sorted(initial_decoded))
                startup["phase"] = "worker_validation"
                if fine_mode:
                    batch = _validated_probe_batch(
                        raw_probes,
                        expected_addresses=connected,
                    )
                    probes = tuple(batch[address] for address in connected)
                    assert initial_scale_event is not None
                else:
                    returned = _returned_probe_addresses(raw_probes)
                    if returned != connected:
                        raise WorkerPreflightError(
                            "Worker preflight returned worker addresses that do not "
                            "match the expected connected workers: "
                            f"expected={connected!r}, returned={returned!r}"
                        )
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
                if fine_mode:
                    assert initial_scale_event is not None
                    assert fine_state is not None
                    validated_probes.update(batch)
                    for address in connected:
                        opened_client.run_on_scheduler(
                            _activate_round_slot,
                            worker_address=address,
                        )
                        initial_scale_event.activated.append(address)
                        validated_workers.add(address)
                        fine_state.validated.add(address)
                    _validate_scheduler_resources(
                        opened_client,
                        expected_scheduler=scheduler_address,
                        candidates=(),
                        validated=tuple(sorted(validated_workers)),
                    )
                    assert fine_state is not None
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
                if initial_scale_event is not None:
                    initial_scale_event.ended_at = utc_now()
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
            reference_results, _attempts = collect_distributed_results(
                (reference_task,),
                runner=submission_runner,
                client=execution_client,
                as_completed=modules.as_completed,
                task_retries=spec.task_retries,
                active=active,
                attempts=reference_attempts,
            )
            if fine_mode:
                self._fine_reference_result = reference_results[0]
            if reference_results[0].status not in {"completed", "reused"}:
                if fine_state is not None:
                    reference_result = reference_results[0]
                    failure = reference_result.failure
                    fine_state.pre_controller_terminal = PreControllerTerminal(
                        phase="reference",
                        status=reference_result.status,
                        exception_type=(
                            "Cancelled" if failure is None else failure.exception_type
                        ),
                        message="" if failure is None else failure.message,
                    )
                return reference_results
            if not round_tasks:
                if fine_state is not None:
                    fine_state.pre_controller_terminal = PreControllerTerminal(
                        phase="moving",
                        status="reused",
                        exception_type="",
                        message="all submitted moving rounds were already complete",
                    )
                return reference_results
            if fine_mode:
                assert fine_state is not None

                def eligible_capacity(
                    controller: FineGraphController,
                    running_count: int,
                ) -> int:
                    nonlocal requested_workers
                    desired = max(
                        1,
                        min(
                            spec.workers,
                            running_count + controller.ready_count,
                        ),
                    )
                    if desired <= requested_workers:
                        return len(validated_workers)
                    event = fine_state.begin_scale(desired, utc_now())
                    try:
                        opened_cluster.scale(jobs=desired)
                        requested_workers = desired
                        execution_client.wait_for_workers(
                            desired,
                            timeout=modules.parse_timedelta(spec.startup_timeout),
                        )
                        next_scheduler, next_connected = _scheduler_snapshot(
                            execution_client,
                            expected_workers=desired,
                        )
                        cast(dict[str, object], evidence["requested"])["workers"] = (
                            requested_workers
                        )
                        observed["connected_workers"] = list(next_connected)
                        newcomers = tuple(
                            address
                            for address in next_connected
                            if address not in validated_workers
                        )
                        event.connected.extend(newcomers)
                        fine_state.candidates.update(newcomers)
                        if next_scheduler != scheduler_address:
                            raise WorkerPreflightError(
                                "SLURM scheduler identity changed during scale-up"
                            )
                        if not validated_workers < set(next_connected):
                            raise WorkerPreflightError(
                                "SLURM worker scale-up did not preserve every "
                                "validated worker and add a new candidate"
                            )
                        _validate_scheduler_resources(
                            execution_client,
                            expected_scheduler=scheduler_address,
                            candidates=newcomers,
                            validated=tuple(sorted(validated_workers)),
                        )
                        raw_new = execution_client.run(
                            probe_worker,
                            dataset_root=str(self.profile.paths.dataset_root),
                            output_root=str(self.profile.paths.output_root),
                            worker_local_directory=str(
                                self.profile.paths.worker_local_directory
                            ),
                            expected_versions=dict(versions),
                            expected_threads=_native_thread_values(self.profile),
                            workers=list(newcomers),
                        )
                        decoded_new = {
                            address: probe
                            for address, probe in _decodable_probe_batch(
                                raw_new
                            ).items()
                            if probe.worker_address == address
                        }
                        observed_probes.update(decoded_new)
                        event.probed.extend(sorted(decoded_new))
                        observed["worker_probes"] = [
                            observed_probes[address].to_dict()
                            for address in sorted(observed_probes)
                        ]
                        new_batch = _validated_probe_batch(
                            raw_new,
                            expected_addresses=newcomers,
                            prior_probes=validated_probes,
                        )
                        validated_probes.update(new_batch)
                        for address in newcomers:
                            execution_client.run_on_scheduler(
                                _activate_round_slot,
                                worker_address=address,
                            )
                            event.activated.append(address)
                            validated_workers.add(address)
                            fine_state.validated.add(address)
                        _validate_scheduler_resources(
                            execution_client,
                            expected_scheduler=scheduler_address,
                            candidates=(),
                            validated=tuple(sorted(validated_workers)),
                        )
                    except BaseException as error:
                        event.fail(error, utc_now())
                        raise
                    finally:
                        event.ended_at = utc_now()
                    return len(validated_workers)

                fine_runner = self._fine_runner
                if fine_runner is None:
                    from cyexm.workflow.fine_task import run_fine_task

                    fine_runner = run_fine_task
                controller = FineGraphController(round_tasks)

                def validate_placement(
                    _task: FineWorkflowTask,
                    result: FineTaskResult,
                ) -> FineTaskResult:
                    matches = tuple(
                        (address, probe)
                        for address, probe in validated_probes.items()
                        if probe.slurm_job_id == result.slurm_job_id
                    )
                    if len(matches) != 1:
                        raise RoundExecutionError(
                            "SLURM fine task result must name exactly one activated "
                            "worker job"
                        )
                    address, probe = matches[0]
                    if (
                        address not in validated_workers
                        or result.hostname != probe.hostname
                        or result.pid != probe.pid
                    ):
                        raise RoundExecutionError(
                            "SLURM fine task placement does not match the exact "
                            "activated worker address/hostname/pid/job identity"
                        )
                    return result

                def record_attempt(observation: FineAttemptObservation) -> None:
                    raw = observation.raw_result
                    matches = (
                        ()
                        if raw is None
                        else tuple(
                            address
                            for address, probe in validated_probes.items()
                            if probe.slurm_job_id == raw.slurm_job_id
                        )
                    )
                    probe = (
                        validated_probes[matches[0]]
                        if raw is not None
                        and len(matches) == 1
                        and matches[0] in validated_workers
                        else None
                    )
                    placement_accepted = (
                        raw is not None
                        and probe is not None
                        and raw.hostname == probe.hostname
                        and raw.pid == probe.pid
                        and raw.slurm_job_id == probe.slurm_job_id
                    )
                    fine_state.record_attempt(
                        FinePlacementAttempt(
                            task_key=observation.task.key.encoded,
                            attempt=observation.attempt,
                            status=observation.settled_result.status,
                            dask_task_key=observation.dask_task_key,
                            started_at=observation.settled_result.started_at,
                            ended_at=observation.settled_result.ended_at,
                            duration_seconds=(
                                observation.settled_result.duration_seconds
                            ),
                            worker_address=(matches[0] if len(matches) == 1 else None),
                            hostname=None if raw is None else raw.hostname,
                            pid=None if raw is None else raw.pid,
                            slurm_job_id=None if raw is None else raw.slurm_job_id,
                            native_threads=observation.settled_result.native_threads,
                            artifact_paths=observation.settled_result.artifact_paths,
                            reused_artifacts=(
                                observation.settled_result.reused_artifacts
                            ),
                            log_path=observation.settled_result.log_path,
                            placement_accepted=placement_accepted,
                            rejection_reason=observation.rejection_reason,
                            failure=observation.settled_result.failure,
                        )
                    )

                moving_results, task_results, attempts = (
                    collect_distributed_fine_results(
                        controller,
                        runner=fine_runner,
                        client=execution_client,
                        as_completed=modules.as_completed,
                        task_retries=spec.task_retries,
                        capacity=spec.workers,
                        active=active,
                        capacity_provider=eligible_capacity,
                        result_validator=validate_placement,
                        attempt_observer=record_attempt,
                    )
                )
                self._fine_task_results = task_results
                self._fine_attempt_counts = attempts
                self._fine_round_results = moving_results
                fine_state.finalize_graph(
                    controller,
                    attempts,
                    task_results,
                    validated_probes,
                )
                validate_fine_slurm_evidence(
                    fine_state.to_dict(),
                    configured_workers=spec.workers,
                    task_retries=spec.task_retries,
                    cancellation_present=False,
                    startup_outcome="succeeded",
                    reference_attempted=True,
                    round_tasks=round_tasks,
                    observed_workers=tuple(
                        cast(list[str], observed["connected_workers"])
                    ),
                    observed_probes=tuple(validated_probes.values()),
                )
            else:
                moving_results, _attempts = collect_distributed_results(
                    round_tasks,
                    runner=submission_runner,
                    client=execution_client,
                    as_completed=modules.as_completed,
                    task_retries=spec.task_retries,
                    active=active,
                    attempts=moving_attempts,
                )
            return (*reference_results, *moving_results)
        except BaseException as error:
            if (
                fine_state is not None
                and fine_state.events
                and fine_state.events[-1].ended_at is None
            ):
                fine_state.events[-1].fail(error, utc_now())
            if startup["outcome"] == "in_progress":
                startup["outcome"] = (
                    "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
                )
                startup["error"] = _error_payload(error)
            if (
                fine_state is not None
                and controller is None
                and fine_state.pre_controller_terminal is None
            ):
                if (
                    isinstance(error, KeyboardInterrupt)
                    and self._fine_reference_result is not None
                    and self._fine_reference_result.status in {"completed", "reused"}
                ):
                    fine_state.pre_controller_terminal = PreControllerTerminal(
                        phase="moving",
                        status="cancelled",
                        exception_type=type(error).__name__,
                        message=str(error),
                    )
                elif (
                    isinstance(error, KeyboardInterrupt)
                    and startup["outcome"] == "succeeded"
                ):
                    fine_state.pre_controller_terminal = PreControllerTerminal(
                        phase="reference",
                        status="cancelled",
                        exception_type=type(error).__name__,
                        message=str(error),
                    )
                elif startup["outcome"] in {"failed", "interrupted"}:
                    fine_state.pre_controller_terminal = PreControllerTerminal(
                        phase="startup",
                        status=(
                            "cancelled"
                            if isinstance(error, KeyboardInterrupt)
                            else "failed"
                        ),
                        exception_type=type(error).__name__,
                        message=str(error),
                    )
            if isinstance(error, KeyboardInterrupt):
                if fine_state is not None:
                    fine_state.cancelled = True
                    fine_state.interruption = (type(error).__name__, str(error))
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
            evidence["attempt_counts"] = self._attempt_evidence(
                tasks, [*reference_attempts, *moving_attempts]
            )
            if fine_state is not None:
                if controller is not None:
                    self._fine_task_results = controller.task_results
                    self._fine_round_results = controller._settled_round_results()
                    fine_state.finalize_graph(
                        controller,
                        self._fine_attempt_counts,
                        self._fine_task_results,
                        validated_probes,
                    )
                evidence["fine_graph"] = fine_state.to_dict()
            self._close_owned(client, cluster, configuration, cleanup)
            cast(dict[str, str], evidence["timestamps"])["ended_at"] = utc_now()


__all__ = ["SlurmExecutor", "validate_slurm_job_script"]
