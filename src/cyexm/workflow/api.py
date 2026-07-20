from __future__ import annotations

import json
import logging
import os
import socket
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Protocol, cast
from uuid import uuid4

from cyexm.config._validation import finite_positive_float
from cyexm.config.dataset import _validate_dataset_manifest
from cyexm.config.execution import _validate_execution_profile
from cyexm.config.models import (
    ExecutionProfile,
    RegistrationSpec,
)
from cyexm.errors import (
    ConfigurationValidationError,
    ExecutorStartupError,
    RoundExecutionError,
    WorkerPreflightError,
    WorkflowExecutionError,
    WorkflowPlanError,
)
from cyexm.execution.base import WorkflowExecutor
from cyexm.execution.sequential import (
    SequentialExecutor,
    _SequentialExecutionInterrupted,
)
from cyexm.io.atomic import atomic_write
from cyexm.registration.artifacts import FileIdentity
from cyexm.workflow._execution_validation import (
    _canonical_path,
    _validate_round_result,
    _validate_round_result_structure,
    _validate_round_task_structure,
)
from cyexm.workflow.layout import WorkflowLayout
from cyexm.workflow.models import (
    RoundResult,
    RoundTask,
    WorkflowPlan,
    WorkflowResult,
    canonical_json,
    sha256_fingerprint,
)
from cyexm.workflow.status import (
    _inspect_workflow_status,
    _read_scientific_snapshot,
    _safe_descendant,
    _safe_workflow_layout,
    _ScientificSnapshot,
    _snapshot_from_text,
)

type ExecutorFactory = Callable[[ExecutionProfile], WorkflowExecutor]

_LOGGER = logging.getLogger(__name__)


class _ExecutionEvidenceOwner(Protocol):
    @property
    def execution_evidence(self) -> object: ...


class _ExecutorEvidenceError(ExecutorStartupError):
    def __init__(self, validation_error: BaseException) -> None:
        self.validation_error = validation_error
        super().__init__(
            "invalid SLURM execution evidence: "
            f"{type(validation_error).__name__}: {validation_error}"
        )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _configure_sequential_native_threads(profile: ExecutionProfile) -> None:
    if profile.executor != "sequential":
        return
    from cyexm.runtime import NativeThreadConfig, configure_native_threads

    configure_native_threads(
        NativeThreadConfig(
            itk_threads=profile.native_threads.itk,
            blas_threads=profile.native_threads.blas,
        )
    )


def _default_executor_factory(profile: ExecutionProfile) -> WorkflowExecutor:
    if profile.executor == "sequential":
        from cyexm.workflow.round_task import run_round_task

        return SequentialExecutor(runner=run_round_task)
    if profile.executor == "local_dask":
        from cyexm.execution.dask import LocalDaskExecutor

        return LocalDaskExecutor(profile)
    if profile.executor == "slurm":
        from cyexm.execution.slurm import SlurmExecutor

        return SlurmExecutor(profile)
    raise ExecutorStartupError(
        f"{profile.executor} execution is not available in this installed milestone"
    )


def _write_text_atomically(path: Path, text: str) -> None:
    def writer(temporary: Path) -> None:
        temporary.write_text(text, encoding="utf-8")

    atomic_write(
        path,
        writer,
        lambda temporary: temporary.read_text(encoding="utf-8") == text,
    )


def _validate_plan_identity(plan: WorkflowPlan) -> str:
    try:
        scientific = json.loads(plan.scientific_json)
        normalized = canonical_json(scientific)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise WorkflowPlanError(
            f"WorkflowPlan.scientific_json must be canonical finite JSON: {error}"
        ) from error
    if normalized != plan.scientific_json:
        raise WorkflowPlanError(
            "WorkflowPlan.scientific_json must use canonical JSON encoding"
        )
    fingerprint = sha256_fingerprint(normalized)
    if fingerprint != plan.scientific_fingerprint:
        raise WorkflowPlanError(
            "WorkflowPlan scientific fingerprint does not match scientific_json"
        )
    if not isinstance(scientific, Mapping) or scientific.get("dataset_id") != (
        plan.dataset_id
    ):
        raise WorkflowPlanError(
            "WorkflowPlan scientific dataset_id does not match the plan"
        )
    return fingerprint


def _ensure_scientific_snapshot(
    plan: WorkflowPlan,
    layout: WorkflowLayout,
) -> None:
    try:
        existing = _read_scientific_snapshot(layout.plan_path, plan.dataset_id)
    except WorkflowPlanError as error:
        raise WorkflowPlanError(
            f"Existing scientific-plan fingerprint mismatch at {layout.plan_path}: "
            f"{error}; select a new output root"
        ) from error
    if existing is not None:
        if existing.fingerprint != plan.scientific_fingerprint:
            raise WorkflowPlanError(
                f"Existing scientific-plan fingerprint mismatch at {layout.plan_path}; "
                "select a new output root"
            )
        return
    text = plan.scientific_json + "\n"
    _write_text_atomically(layout.plan_path, text)
    written = _read_scientific_snapshot(layout.plan_path, plan.dataset_id)
    if written is None or written.fingerprint != plan.scientific_fingerprint:
        raise WorkflowPlanError(
            f"Scientific plan snapshot validation failed at {layout.plan_path}"
        )


def _validate_operational_paths(layout: WorkflowLayout) -> None:
    _safe_descendant(
        layout.plan_path,
        root=layout.root,
        location="scientific-plan.json",
    )
    _safe_descendant(
        layout.workflow_runs,
        root=layout.root,
        location="workflow-runs directory",
    )


def _manifest_source_path(
    relative: object,
    *,
    dataset_root: Path,
    location: str,
) -> Path:
    if not isinstance(relative, PurePosixPath):
        raise WorkflowPlanError(f"{location} must be a PurePosixPath; got {relative!r}")
    raw_parts = str(relative).split("/")
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in raw_parts):
        raise WorkflowPlanError(
            f"{location} must be a safe relative POSIX path; got {relative!r}"
        )
    try:
        resolved = dataset_root.joinpath(*relative.parts).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise WorkflowPlanError(
            f"{location} must resolve to an existing input file"
        ) from error
    if not resolved.is_file() or not resolved.is_relative_to(dataset_root):
        raise WorkflowPlanError(
            f"{location} must resolve to a file within dataset_root "
            f"{dataset_root}; got {resolved}"
        )
    return resolved


def _registration_matches_snapshot(
    registration: RegistrationSpec,
    snapshot: _ScientificSnapshot,
) -> bool:
    try:
        return (
            type(registration.crop) is bool
            and registration.crop == snapshot.registration.crop
            and registration.linear.type == snapshot.registration.linear_type
            and registration.linear.accuracy == snapshot.registration.linear_accuracy
            and registration.nonlinear.type == snapshot.registration.nonlinear_type
            and registration.nonlinear.accuracy
            == snapshot.registration.nonlinear_accuracy
        )
    except AttributeError:
        return False


def _validate_workflow_plan(
    plan: WorkflowPlan,
    layout: WorkflowLayout,
) -> _ScientificSnapshot:
    fingerprint = _validate_plan_identity(plan)
    snapshot = _snapshot_from_text(
        plan.scientific_json,
        layout=layout,
        dataset_id=plan.dataset_id,
    )
    if snapshot.fingerprint != fingerprint:
        raise WorkflowPlanError(
            "WorkflowPlan scientific fingerprint does not match its strict "
            "scientific schema"
        )
    if type(plan.schema_version) is not int or plan.schema_version != 1:
        raise WorkflowPlanError(
            "WorkflowPlan.schema_version must be the built-in integer 1; "
            f"got {plan.schema_version!r}"
        )

    execution = plan.execution
    try:
        _validate_execution_profile(execution)
    except ConfigurationValidationError as error:
        raise WorkflowPlanError(
            f"WorkflowPlan execution profile is invalid: {error}"
        ) from error
    dataset_root = _canonical_path(
        execution.paths.dataset_root,
        "WorkflowPlan.execution.paths.dataset_root",
    )
    output_root = _canonical_path(
        execution.paths.output_root,
        "WorkflowPlan.execution.paths.output_root",
    )
    if not dataset_root.is_dir():
        raise WorkflowPlanError(
            "WorkflowPlan.execution.paths.dataset_root must be an existing "
            f"directory; got {dataset_root}"
        )
    if output_root.exists() and not output_root.is_dir():
        raise WorkflowPlanError(
            "WorkflowPlan.execution.paths.output_root must resolve to a "
            f"directory; got {output_root}"
        )
    if layout.root != output_root / plan.dataset_id:
        raise WorkflowPlanError(
            "WorkflowPlan output layout does not match execution.paths.output_root"
        )

    for source in snapshot.sources:
        if not source.path.is_relative_to(dataset_root):
            raise WorkflowPlanError(
                f"Scientific source resolves outside dataset_root {dataset_root}: "
                f"{source.path}"
            )
        try:
            current = FileIdentity.from_path(source.path)
        except (OSError, RuntimeError) as error:
            raise WorkflowPlanError(
                f"Scientific source is no longer readable: {source.path}"
            ) from error
        if current != source.identity:
            raise WorkflowPlanError(
                f"Scientific source identity changed after planning: {source.path}"
            )

    manifest = plan.manifest
    try:
        _validate_dataset_manifest(manifest)
    except ConfigurationValidationError as error:
        raise WorkflowPlanError(f"WorkflowPlan manifest is invalid: {error}") from error
    if (
        type(manifest.schema_version) is not int
        or manifest.schema_version != plan.schema_version
        or manifest.dataset_id != plan.dataset_id
        or manifest.voxel_spacing_um != snapshot.voxel_spacing_xyz
        or not _registration_matches_snapshot(manifest.registration, snapshot)
    ):
        raise WorkflowPlanError(
            "WorkflowPlan manifest does not match the scientific plan"
        )
    fixed_image = _manifest_source_path(
        manifest.reference.channel,
        dataset_root=dataset_root,
        location="WorkflowPlan.manifest.reference.channel",
    )
    if (
        type(manifest.reference.round) is not int
        or manifest.reference.round != snapshot.reference_round
        or fixed_image != snapshot.fixed_source.path
        or plan.fixed_image != fixed_image
    ):
        raise WorkflowPlanError(
            "WorkflowPlan reference does not match the scientific plan"
        )

    round_contracts = {item.round: item for item in snapshot.rounds}
    manifest_rounds = {item.round: item for item in manifest.rounds}
    if len(manifest_rounds) != len(manifest.rounds) or set(manifest_rounds) != set(
        round_contracts
    ):
        raise WorkflowPlanError(
            "WorkflowPlan manifest rounds do not match the scientific plan"
        )
    if type(plan.tasks) is not tuple:
        raise WorkflowPlanError("WorkflowPlan.tasks must be an exact tuple")
    for task_index, task in enumerate(plan.tasks):
        _validate_round_task_structure(task, task_index)
    task_rounds = tuple(task.round for task in plan.tasks)
    if task_rounds != tuple(sorted(round_contracts)) or len(set(task_rounds)) != len(
        task_rounds
    ):
        raise WorkflowPlanError(
            "WorkflowPlan tasks must match the unique sorted scientific rounds"
        )

    expected_artifacts: list[Path] = []
    for task in plan.tasks:
        contract = round_contracts[task.round]
        manifest_round = manifest_rounds[task.round]
        round_layout = layout.round(task.round)
        registration_directory = _safe_descendant(
            task.registration_directory,
            root=layout.root,
            location=f"WorkflowPlan task {task.round} registration_directory",
        )
        log_path = _safe_descendant(
            task.log_path,
            root=layout.root,
            location=f"WorkflowPlan task {task.round} log_path",
        )
        if registration_directory != round_layout.registration:
            raise WorkflowPlanError(
                f"WorkflowPlan task {task.round} registration_directory does "
                "not match its output layout"
            )
        if log_path != round_layout.log:
            raise WorkflowPlanError(
                f"WorkflowPlan task {task.round} log_path does not match its "
                "output layout"
            )
        task_spacing = task.voxel_spacing_xyz
        try:
            valid_task_spacing = type(task_spacing) is tuple and len(task_spacing) == 3
            if valid_task_spacing:
                tuple(
                    finite_positive_float(
                        value,
                        f"WorkflowPlan task {task.round} voxel_spacing_xyz[{index}]",
                    )
                    for index, value in enumerate(task_spacing)
                )
        except ConfigurationValidationError:
            valid_task_spacing = False
        if not valid_task_spacing:
            raise WorkflowPlanError(
                f"WorkflowPlan task {task.round} voxel_spacing_xyz must contain "
                "three positive finite numbers"
            )
        if (
            task.dataset_id != plan.dataset_id
            or task.strategy != contract.strategy
            or task.fixed_image != fixed_image
            or task.moving_image != contract.registration_source.path
            or task.voxel_spacing_xyz != snapshot.voxel_spacing_xyz
            or not _registration_matches_snapshot(task.registration, snapshot)
            or task.native_threads != execution.native_threads
            or task.chunk_size_zyx != contract.chunk_size_zyx
            or task.overlap_fraction != contract.overlap_fraction
            or type(task.force) is not bool
            or task.force
        ):
            raise WorkflowPlanError(
                f"WorkflowPlan task {task.round} does not match its scientific "
                "or execution contract"
            )

        registration_source = _manifest_source_path(
            manifest_round.registration_channel,
            dataset_root=dataset_root,
            location=f"WorkflowPlan.manifest round {task.round} registration_channel",
        )
        if (
            manifest_round.strategy != contract.strategy
            or registration_source != contract.registration_source.path
            or manifest_round.registration_channel
            != contract.registration_manifest_channel
            or manifest_round.chunk_size_zyx != contract.chunk_size_zyx
            or manifest_round.overlap_fraction != contract.overlap_fraction
            or len(manifest_round.channels) != len(contract.channels)
            or len(task.channels) != len(contract.channels)
        ):
            raise WorkflowPlanError(
                f"WorkflowPlan manifest round {task.round} does not match the "
                "scientific plan"
            )
        for index, (channel_task, channel_spec, channel_contract) in enumerate(
            zip(task.channels, manifest_round.channels, contract.channels, strict=True)
        ):
            channel_source = _manifest_source_path(
                channel_spec.file,
                dataset_root=dataset_root,
                location=f"WorkflowPlan.manifest round {task.round} channel {index}",
            )
            _safe_descendant(
                channel_task.output,
                root=layout.root,
                location=f"WorkflowPlan task {task.round} channel {index} output",
            )
            if (
                channel_spec.file != channel_contract.manifest_file
                or channel_spec.interpolation != channel_contract.interpolation
                or channel_source != channel_contract.source.path
                or channel_task.source != channel_contract.source.path
                or channel_task.output != channel_contract.output
                or channel_task.interpolation != channel_contract.interpolation
            ):
                raise WorkflowPlanError(
                    f"WorkflowPlan task {task.round} channel {index} does not "
                    "match the scientific plan"
                )
        expected_artifacts.extend(item.path for item in contract.artifacts)

    expected_tuple = tuple(expected_artifacts)
    if plan.expected_artifacts != expected_tuple or len(set(expected_tuple)) != len(
        expected_tuple
    ):
        raise WorkflowPlanError(
            "WorkflowPlan.expected_artifacts must exactly match the scientific "
            "task outputs"
        )
    for artifact in plan.expected_artifacts:
        _safe_descendant(
            artifact,
            root=layout.root,
            location="WorkflowPlan expected_artifacts entry",
        )
    return snapshot


def _validate_force_arguments(
    plan: WorkflowPlan,
    force_rounds: frozenset[int],
    force_all: bool,
) -> None:
    if type(force_all) is not bool:
        raise WorkflowPlanError(f"force_all must be a boolean; got {force_all!r}")
    if not isinstance(force_rounds, frozenset) or any(
        type(round_number) is not int for round_number in force_rounds
    ):
        raise WorkflowPlanError(
            "force_rounds must be a frozenset of built-in round integers"
        )
    if force_all and force_rounds:
        raise WorkflowPlanError("force_all and force_rounds are mutually exclusive")
    known = frozenset(task.round for task in plan.tasks)
    unknown = tuple(sorted(force_rounds - known))
    if unknown:
        raise WorkflowPlanError(f"force_rounds contains unknown rounds: {unknown!r}")


def _execution_profile_mapping(profile: ExecutionProfile) -> dict[str, object]:
    local_dask = profile.local_dask
    slurm = profile.slurm
    return {
        "schema_version": profile.schema_version,
        "executor": profile.executor,
        "source_path": str(profile.source_path),
        "paths": {
            "dataset_root": str(profile.paths.dataset_root),
            "output_root": str(profile.paths.output_root),
            "worker_local_directory": (
                None
                if profile.paths.worker_local_directory is None
                else str(profile.paths.worker_local_directory)
            ),
        },
        "native_threads": {
            "itk": profile.native_threads.itk,
            "blas": profile.native_threads.blas,
        },
        "local_dask": (
            None
            if local_dask is None
            else {
                "workers": local_dask.workers,
                "threads_per_worker": local_dask.threads_per_worker,
                "processes": local_dask.processes,
                "memory_limit": local_dask.memory_limit,
                "startup_timeout": local_dask.startup_timeout,
                "task_retries": local_dask.task_retries,
            }
        ),
        "slurm": (
            None
            if slurm is None
            else {
                "queue": slurm.queue,
                "workers": slurm.workers,
                "cores_per_worker": slurm.cores_per_worker,
                "processes_per_worker": slurm.processes_per_worker,
                "memory": slurm.memory,
                "walltime": slurm.walltime,
                "interface": slurm.interface,
                "scheduler_interface": slurm.scheduler_interface,
                "startup_timeout": slurm.startup_timeout,
                "task_retries": slurm.task_retries,
                "job_name": slurm.job_name,
                "job_extra_directives": list(slurm.job_extra_directives),
            }
        ),
    }


def _task_artifact_paths(plan: WorkflowPlan, task: RoundTask) -> tuple[str, ...]:
    return tuple(
        str(path)
        for path in plan.expected_artifacts
        if path == task.registration_directory
        or path.is_relative_to(task.registration_directory)
        or any(path == channel.output for channel in task.channels)
    )


def _reused_result(
    plan: WorkflowPlan,
    task: RoundTask,
    artifact_paths: tuple[str, ...],
    timestamp: str,
) -> RoundResult:
    return RoundResult(
        dataset_id=task.dataset_id,
        round=task.round,
        status="reused",
        started_at=timestamp,
        ended_at=timestamp,
        duration_seconds=0.0,
        hostname=socket.gethostname(),
        pid=os.getpid(),
        dask_task_key=None,
        slurm_job_id=None,
        strategy=task.strategy,
        native_threads=(
            ("itk", task.native_threads.itk),
            ("blas", task.native_threads.blas),
        ),
        artifact_paths=artifact_paths or _task_artifact_paths(plan, task),
        reused_artifacts=artifact_paths or _task_artifact_paths(plan, task),
        log_path=str(task.log_path),
        failure=None,
    )


def _ordered_results(
    submitted: tuple[RoundTask, ...],
    results: tuple[RoundResult, ...],
) -> tuple[RoundResult, ...]:
    if type(results) is not tuple:
        raise RoundExecutionError(
            "Executor must return an exact tuple containing exact RoundResult values"
        )
    for index, result in enumerate(results):
        _validate_round_result_structure(result, f"Executor result at index {index}")
    expected = tuple(task.round for task in submitted)
    actual = tuple(result.round for result in results)
    if len(set(actual)) != len(actual) or set(actual) != set(expected):
        raise RoundExecutionError(
            "Executor must return exactly one result per submitted round; "
            f"expected={expected!r}, got={actual!r}"
        )
    by_round = {result.round: result for result in results}
    ordered = tuple(by_round[task.round] for task in submitted)
    for task, result in zip(submitted, ordered, strict=True):
        _validate_round_result(task, result)
    return ordered


def _validated_partial_results(
    submitted: tuple[RoundTask, ...],
    results: tuple[RoundResult, ...],
) -> tuple[RoundResult, ...]:
    if type(results) is not tuple:
        return ()
    tasks_by_round = {task.round: task for task in submitted}
    valid: list[RoundResult] = []
    seen: set[int] = set()
    for index, result in enumerate(results):
        try:
            _validate_round_result_structure(
                result,
                f"Cancelled partial result at index {index}",
            )
        except RoundExecutionError:
            continue
        task = tasks_by_round.get(result.round)
        if task is None or result.round in seen:
            continue
        try:
            _validate_round_result(task, result)
        except RoundExecutionError:
            continue
        seen.add(result.round)
        valid.append(result)
    return tuple(sorted(valid, key=lambda item: item.round))


def _write_run_record(
    path: Path,
    *,
    run_id: str,
    plan: WorkflowPlan,
    result: WorkflowResult,
    force_all: bool,
    force_rounds: frozenset[int],
    operational_failure: Mapping[str, object] | None = None,
    execution_evidence_error: Mapping[str, object] | None = None,
) -> None:
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "dataset_id": plan.dataset_id,
        "scientific_fingerprint": plan.scientific_fingerprint,
        "status": result.status,
        "started_at": result.started_at,
        "ended_at": result.ended_at,
        "force_all": force_all,
        "force_rounds": sorted(force_rounds),
        "execution_profile": _execution_profile_mapping(plan.execution),
        "execution": (
            None
            if execution_evidence_error is not None
            else json.loads(result.execution_evidence_json)
        ),
        "rounds": [round_result.to_dict() for round_result in result.rounds],
    }
    if operational_failure is not None:
        payload["operational_failure"] = dict(operational_failure)
    if execution_evidence_error is not None:
        payload["execution_evidence_error"] = dict(execution_evidence_error)
    _write_text_atomically(path, canonical_json(payload) + "\n")


def _executor_execution_evidence_json(
    profile: ExecutionProfile,
    executor: object | None,
    expected_tasks: tuple[RoundTask, ...],
) -> str:
    if profile.executor != "slurm" or executor is None:
        return "{}"
    try:
        from cyexm.execution.slurm import _canonical_execution_evidence

        if profile.slurm is None:
            raise TypeError("SLURM execution evidence requires a SlurmSpec")
        evidence = cast(_ExecutionEvidenceOwner, executor).execution_evidence
        return _canonical_execution_evidence(
            evidence,
            expected_tasks=expected_tasks,
            task_retries=profile.slurm.task_retries,
        )
    except KeyboardInterrupt:
        raise
    except BaseException as error:
        raise _ExecutorEvidenceError(error) from error


def _evidence_error_payload(error: BaseException) -> dict[str, str]:
    return {"exception_type": type(error).__name__, "message": str(error)}


def run_workflow(
    plan: WorkflowPlan,
    *,
    force_rounds: frozenset[int] = frozenset(),
    force_all: bool = False,
    _executor_factory: ExecutorFactory | None = None,
) -> WorkflowResult:
    """Execute pending workflow rounds and persist operational provenance."""
    layout = _safe_workflow_layout(
        plan.execution.paths.output_root,
        plan.dataset_id,
    )
    _validate_operational_paths(layout)
    _validate_workflow_plan(plan, layout)
    _validate_force_arguments(plan, force_rounds, force_all)
    _configure_sequential_native_threads(plan.execution)
    started_at = _utc_now()
    run_id = str(uuid4())
    run_record = layout.run_record(run_id)
    _safe_descendant(
        run_record,
        root=layout.root,
        location="workflow run record",
    )
    preserved_results: tuple[RoundResult, ...] = ()
    submitted: tuple[RoundTask, ...] = ()
    executor: WorkflowExecutor | None = None
    execution_evidence_json = "{}"
    try:
        _ensure_scientific_snapshot(plan, layout)
        status = _inspect_workflow_status(
            plan.execution.paths.output_root,
            plan.dataset_id,
            _tasks=plan.tasks,
        )
        if status.scientific_fingerprint != plan.scientific_fingerprint:
            raise WorkflowPlanError(
                "Persisted scientific-plan fingerprint changed before execution"
            )
        status_by_round = {item.round: item for item in status.rounds}
        plan_rounds = tuple(task.round for task in plan.tasks)
        if set(status_by_round) != set(plan_rounds):
            raise WorkflowPlanError(
                "Persisted scientific plan rounds do not match WorkflowPlan tasks; "
                f"snapshot={tuple(sorted(status_by_round))!r}, plan={plan_rounds!r}"
            )

        reused = tuple(
            _reused_result(
                plan,
                task,
                status_by_round[task.round].complete_artifacts,
                started_at,
            )
            for task in plan.tasks
            if status_by_round[task.round].status == "complete"
            and not force_all
            and task.round not in force_rounds
        )
        preserved_results = reused
        reused_rounds = frozenset(result.round for result in reused)
        submitted = tuple(
            replace(
                task,
                force=force_all or task.round in force_rounds,
            )
            for task in plan.tasks
            if task.round not in reused_rounds
        )
        if submitted:
            factory = _executor_factory or _default_executor_factory
            executor = factory(plan.execution)
            raw_results = executor.execute(submitted)
            executed = _ordered_results(submitted, raw_results)
            preserved_results = tuple(
                sorted((*reused, *executed), key=lambda result: result.round)
            )
            execution_evidence_json = _executor_execution_evidence_json(
                plan.execution,
                executor,
                submitted,
            )
        else:
            executed = ()
        rounds = tuple(sorted((*reused, *executed), key=lambda result: result.round))
        preserved_results = rounds
        unsuccessful = tuple(
            sorted(
                result.round
                for result in rounds
                if result.status not in {"completed", "reused"}
            )
        )
        workflow_status = "failed" if unsuccessful else "completed"
        result = WorkflowResult(
            dataset_id=plan.dataset_id,
            scientific_fingerprint=plan.scientific_fingerprint,
            status=workflow_status,
            started_at=started_at,
            ended_at=_utc_now(),
            rounds=rounds,
            run_record=str(run_record),
            execution_evidence_json=execution_evidence_json,
        )
        _write_run_record(
            run_record,
            run_id=run_id,
            plan=plan,
            result=result,
            force_all=force_all,
            force_rounds=force_rounds,
        )
    except ExecutorStartupError as startup_error:
        if executor is None:
            raise
        active_error = startup_error
        evidence_error: _ExecutorEvidenceError | None = None
        if isinstance(startup_error, _ExecutorEvidenceError):
            evidence_error = startup_error
        else:
            try:
                execution_evidence_json = _executor_execution_evidence_json(
                    plan.execution,
                    executor,
                    submitted,
                )
            except _ExecutorEvidenceError as contract_error:
                evidence_error = contract_error
                active_error = contract_error
        if evidence_error is not None:
            failure_phase = "execution_evidence"
        elif isinstance(startup_error, WorkerPreflightError):
            failure_phase = "preflight"
        else:
            failure_phase = "startup"
        failed = WorkflowResult(
            dataset_id=plan.dataset_id,
            scientific_fingerprint=plan.scientific_fingerprint,
            status="failed",
            started_at=started_at,
            ended_at=_utc_now(),
            rounds=preserved_results,
            run_record=str(run_record),
            execution_evidence_json=execution_evidence_json,
        )
        _write_run_record(
            run_record,
            run_id=run_id,
            plan=plan,
            result=failed,
            force_all=force_all,
            force_rounds=force_rounds,
            operational_failure={
                "phase": failure_phase,
                "exception_type": type(active_error).__name__,
                "message": str(active_error),
            },
            execution_evidence_error=(
                None
                if evidence_error is None
                else _evidence_error_payload(evidence_error.validation_error)
            ),
        )
        active_error._attach_operational_record(
            run_record=str(run_record),
            execution_evidence_json=(
                execution_evidence_json if evidence_error is None else None
            ),
        )
        if active_error is startup_error:
            raise
        raise active_error from startup_error
    except KeyboardInterrupt as interruption:
        evidence_error = None
        try:
            execution_evidence_json = _executor_execution_evidence_json(
                plan.execution,
                executor,
                submitted,
            )
        except BaseException as evidence_access_error:
            evidence_error = (
                evidence_access_error
                if isinstance(evidence_access_error, _ExecutorEvidenceError)
                else _ExecutorEvidenceError(evidence_access_error)
            )
            _LOGGER.error(
                "invalid SLURM execution evidence during cancellation: %s",
                evidence_error,
            )
        original_interruption = interruption
        if isinstance(interruption, _SequentialExecutionInterrupted):
            partial_results = _validated_partial_results(
                submitted,
                interruption.completed_results,
            )
            preserved_results = tuple(
                sorted(
                    (*preserved_results, *partial_results),
                    key=lambda result: result.round,
                )
            )
            original_interruption = interruption.original
        cancelled = WorkflowResult(
            dataset_id=plan.dataset_id,
            scientific_fingerprint=plan.scientific_fingerprint,
            status="cancelled",
            started_at=started_at,
            ended_at=_utc_now(),
            rounds=preserved_results,
            run_record=str(run_record),
            execution_evidence_json=execution_evidence_json,
        )
        _write_run_record(
            run_record,
            run_id=run_id,
            plan=plan,
            result=cancelled,
            force_all=force_all,
            force_rounds=force_rounds,
            execution_evidence_error=(
                None
                if evidence_error is None
                else _evidence_error_payload(evidence_error.validation_error)
            ),
        )
        if original_interruption is interruption:
            raise
        raise original_interruption from None
    if unsuccessful:
        raise WorkflowExecutionError(unsuccessful, str(run_record))
    return result


__all__ = ["run_workflow"]
