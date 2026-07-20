from __future__ import annotations

import json
import math
import os
import socket
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from uuid import uuid4

from cyexm.config._validation import finite_positive_float
from cyexm.config.dataset import _validate_dataset_manifest
from cyexm.config.execution import _validate_execution_profile
from cyexm.config.models import (
    ExecutionProfile,
    NativeThreadSpec,
    RegistrationSpec,
    RegistrationStageSpec,
)
from cyexm.errors import (
    ConfigurationValidationError,
    ExecutorStartupError,
    RoundExecutionError,
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
from cyexm.runtime import NATIVE_THREAD_ENV_VARS
from cyexm.workflow.layout import WorkflowLayout
from cyexm.workflow.models import (
    ChannelTask,
    RoundFailure,
    RoundResult,
    RoundTask,
    WorkflowPlan,
    WorkflowResult,
    canonical_json,
    sha256_fingerprint,
)
from cyexm.workflow.status import (
    _read_scientific_snapshot,
    _safe_descendant,
    _safe_workflow_layout,
    _ScientificSnapshot,
    _snapshot_from_text,
    inspect_workflow_status,
)

type ExecutorFactory = Callable[[ExecutionProfile], WorkflowExecutor]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _unavailable_round_runner(task: RoundTask) -> RoundResult:
    raise RoundExecutionError(
        f"Round {task.round} scientific execution is not available until Phase 2B"
    )


def _default_executor_factory(profile: ExecutionProfile) -> WorkflowExecutor:
    if profile.executor == "sequential":
        return SequentialExecutor(runner=_unavailable_round_runner)
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


def _canonical_path(path: object, location: str) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise WorkflowPlanError(
            f"{location} must be an absolute pathlib.Path; got {path!r}"
        )
    try:
        resolved = path.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise WorkflowPlanError(
            f"{location} cannot be resolved safely; got {path}"
        ) from error
    if resolved != path:
        raise WorkflowPlanError(
            f"{location} must be canonical; got {path}, resolved {resolved}"
        )
    return path


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


def _validate_round_task_structure(task: object, index: int) -> None:
    location = f"WorkflowPlan.tasks[{index}]"
    if type(task) is not RoundTask:
        raise WorkflowPlanError(f"{location} must be a RoundTask; got {task!r}")
    if type(task.dataset_id) is not str or not task.dataset_id:
        raise WorkflowPlanError(
            f"{location}.dataset_id must be a nonempty built-in string"
        )
    if type(task.round) is not int or task.round < 0:
        raise WorkflowPlanError(
            f"{location}.round must be a nonnegative built-in integer"
        )
    if type(task.strategy) is not str or task.strategy not in {
        "whole_image",
        "serial_chunked",
    }:
        raise WorkflowPlanError(
            f"{location}.strategy must be an exact supported built-in string"
        )
    for field_name, path in (
        ("fixed_image", task.fixed_image),
        ("moving_image", task.moving_image),
        ("registration_directory", task.registration_directory),
        ("log_path", task.log_path),
    ):
        _canonical_path(path, f"{location}.{field_name}")

    spacing = task.voxel_spacing_xyz
    if (
        type(spacing) is not tuple
        or len(spacing) != 3
        or any(
            type(value) is not float or not math.isfinite(value) or value <= 0.0
            for value in spacing
        )
    ):
        raise WorkflowPlanError(
            f"{location}.voxel_spacing_xyz must be an exact tuple of three "
            "positive finite built-in floats"
        )

    registration = task.registration
    if type(registration) is not RegistrationSpec:
        raise WorkflowPlanError(f"{location}.registration must be a RegistrationSpec")
    if type(registration.crop) is not bool:
        raise WorkflowPlanError(
            f"{location}.registration.crop must be a built-in boolean"
        )
    for stage_name, stage in (
        ("linear", registration.linear),
        ("nonlinear", registration.nonlinear),
    ):
        stage_location = f"{location}.registration.{stage_name}"
        if type(stage) is not RegistrationStageSpec:
            raise WorkflowPlanError(f"{stage_location} must be a RegistrationStageSpec")
        if type(stage.type) is not str or not stage.type:
            raise WorkflowPlanError(
                f"{stage_location}.type must be a nonempty built-in string"
            )
        if type(stage.accuracy) is not str or not stage.accuracy:
            raise WorkflowPlanError(
                f"{stage_location}.accuracy must be a nonempty built-in string"
            )

    native_threads = task.native_threads
    if type(native_threads) is not NativeThreadSpec:
        raise WorkflowPlanError(f"{location}.native_threads must be a NativeThreadSpec")
    for field_name, value in (
        ("itk", native_threads.itk),
        ("blas", native_threads.blas),
    ):
        if type(value) is not int or value < 1:
            raise WorkflowPlanError(
                f"{location}.native_threads.{field_name} must be a positive "
                "built-in integer"
            )

    if type(task.channels) is not tuple or not task.channels:
        raise WorkflowPlanError(f"{location}.channels must be a nonempty exact tuple")
    for channel_index, channel in enumerate(task.channels):
        channel_location = f"{location}.channels[{channel_index}]"
        if type(channel) is not ChannelTask:
            raise WorkflowPlanError(f"{channel_location} must be a ChannelTask")
        _canonical_path(channel.source, f"{channel_location}.source")
        _canonical_path(channel.output, f"{channel_location}.output")
        if type(channel.interpolation) is not str or channel.interpolation not in {
            "linear",
            "nearestNeighbor",
        }:
            raise WorkflowPlanError(
                f"{channel_location}.interpolation must be an exact supported "
                "built-in string"
            )

    chunk_size = task.chunk_size_zyx
    if chunk_size is not None:
        if type(chunk_size) is not tuple or len(chunk_size) != 3:
            raise WorkflowPlanError(
                f"{location}.chunk_size_zyx must be null or an exact tuple of "
                "three positive built-in integers"
            )
        for chunk_index, value in enumerate(chunk_size):
            if type(value) is not int or value < 1:
                raise WorkflowPlanError(
                    f"{location}.chunk_size_zyx[{chunk_index}] must be a positive "
                    "built-in integer"
                )
    overlap = task.overlap_fraction
    if overlap is not None and (
        type(overlap) is not float
        or not math.isfinite(overlap)
        or not 0.0 <= overlap < 1.0
    ):
        raise WorkflowPlanError(
            f"{location}.overlap_fraction must be null or a finite built-in "
            "float from zero up to one"
        )
    if task.strategy == "whole_image" and (
        chunk_size is not None or overlap is not None
    ):
        raise WorkflowPlanError(
            f"{location}.chunk_size_zyx and {location}.overlap_fraction must be "
            "null for whole_image"
        )
    if task.strategy == "serial_chunked" and (chunk_size is None or overlap is None):
        raise WorkflowPlanError(
            f"{location}.chunk_size_zyx and {location}.overlap_fraction are "
            "required for serial_chunked"
        )
    if type(task.force) is not bool:
        raise WorkflowPlanError(f"{location}.force must be a built-in boolean")


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
    if type(results) is not tuple or not all(
        type(result) is RoundResult for result in results
    ):
        raise RoundExecutionError(
            "Executor must return an exact tuple containing exact RoundResult values"
        )
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


def _planned_task_artifacts(task: RoundTask) -> tuple[str, ...]:
    return (
        str(task.registration_directory / f"linear_transform_{task.round}.mat"),
        str(task.registration_directory / f"nonlinear_warp_{task.round}.nii.gz"),
        str(task.registration_directory / f"nonlinear_registered_{task.round}.tif"),
        *(str(channel.output) for channel in task.channels),
    )


def _parse_utc_timestamp(value: object, location: str) -> datetime:
    if type(value) is not str or "T" not in value or not value.endswith("Z"):
        raise RoundExecutionError(
            f"{location} must be an ISO 8601 UTC timestamp ending in Z"
        )
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise RoundExecutionError(
            f"{location} must be an ISO 8601 UTC timestamp ending in Z"
        ) from error
    if parsed.utcoffset() != UTC.utcoffset(None):
        raise RoundExecutionError(
            f"{location} must be an ISO 8601 UTC timestamp ending in Z"
        )
    return parsed


def _validate_round_result_structure(result: object, location: str) -> None:
    if type(result) is not RoundResult:
        raise RoundExecutionError(f"{location} must be an exact RoundResult")
    for field_name, value in (
        ("dataset_id", result.dataset_id),
        ("status", result.status),
        ("started_at", result.started_at),
        ("ended_at", result.ended_at),
        ("hostname", result.hostname),
        ("strategy", result.strategy),
        ("log_path", result.log_path),
    ):
        if type(value) is not str or not value:
            raise RoundExecutionError(
                f"{location} {field_name} must be a nonempty built-in string"
            )
    if type(result.round) is not int or result.round < 0:
        raise RoundExecutionError(
            f"{location} round must be a nonnegative built-in integer"
        )
    if (
        type(result.duration_seconds) is not float
        or not math.isfinite(result.duration_seconds)
        or result.duration_seconds < 0.0
    ):
        raise RoundExecutionError(
            f"{location} duration_seconds must be a nonnegative finite built-in float"
        )
    if type(result.pid) is not int or result.pid < 1:
        raise RoundExecutionError(f"{location} pid must be a positive built-in integer")
    for field_name, value in (
        ("dask_task_key", result.dask_task_key),
        ("slurm_job_id", result.slurm_job_id),
    ):
        if value is not None and (type(value) is not str or not value):
            raise RoundExecutionError(
                f"{location} {field_name} must be null or a nonempty built-in string"
            )

    native_threads = result.native_threads
    if type(native_threads) is not tuple:
        raise RoundExecutionError(f"{location} native_threads must be an exact tuple")
    allowed_names = frozenset({"itk", "blas", *NATIVE_THREAD_ENV_VARS})
    names: list[str] = []
    for index, entry in enumerate(native_threads):
        entry_location = f"{location} native_threads[{index}]"
        if type(entry) is not tuple or len(entry) != 2:
            raise RoundExecutionError(
                f"{entry_location} must be an exact (name, count) tuple"
            )
        name, count = entry
        if type(name) is not str or name not in allowed_names:
            raise RoundExecutionError(
                f"{entry_location} name must be an exact expected built-in string"
            )
        if type(count) is not int or count < 1:
            raise RoundExecutionError(
                f"{entry_location} count must be a positive built-in integer"
            )
        names.append(name)
    if len(set(names)) != len(names):
        raise RoundExecutionError(f"{location} native_threads names must be unique")

    for field_name, values in (
        ("artifact_paths", result.artifact_paths),
        ("reused_artifacts", result.reused_artifacts),
    ):
        if type(values) is not tuple:
            raise RoundExecutionError(f"{location} {field_name} must be an exact tuple")
        if any(type(value) is not str for value in values):
            raise RoundExecutionError(
                f"{location} {field_name} entries must be built-in strings"
            )

    failure = result.failure
    if failure is not None:
        if type(failure) is not RoundFailure:
            raise RoundExecutionError(
                f"{location} failure must be null or an exact RoundFailure"
            )
        for field_name, value in (
            ("stage", failure.stage),
            ("exception_type", failure.exception_type),
            ("message", failure.message),
        ):
            if type(value) is not str or (field_name != "message" and not value):
                raise RoundExecutionError(
                    f"{location} failure.{field_name} must be a built-in string"
                )


def _validate_round_result(task: RoundTask, result: RoundResult) -> None:
    location = f"Executor result for round {task.round}"
    _validate_round_result_structure(result, location)
    if result.dataset_id != task.dataset_id:
        raise RoundExecutionError(f"{location} dataset_id does not match the task")
    if type(result.round) is not int or result.round != task.round:
        raise RoundExecutionError(f"{location} round does not match the task")
    if result.strategy != task.strategy:
        raise RoundExecutionError(f"{location} strategy does not match the task")
    if result.status not in {"reused", "completed", "failed", "cancelled"}:
        raise RoundExecutionError(f"{location} has unknown status {result.status!r}")
    if result.status == "failed" and result.failure is None:
        raise RoundExecutionError(f"{location} failure is required for failed status")
    if result.status != "failed" and result.failure is not None:
        raise RoundExecutionError(
            f"{location} failure must be absent unless status is failed"
        )
    compact_threads = (
        ("itk", task.native_threads.itk),
        ("blas", task.native_threads.blas),
    )
    expanded_threads = tuple(
        (
            name,
            task.native_threads.itk
            if name == NATIVE_THREAD_ENV_VARS[0]
            else task.native_threads.blas,
        )
        for name in NATIVE_THREAD_ENV_VARS
    )
    if (
        result.native_threads != compact_threads
        and result.native_threads != expanded_threads
    ):
        raise RoundExecutionError(f"{location} native_threads do not match the task")
    started_at = _parse_utc_timestamp(result.started_at, f"{location} started_at")
    ended_at = _parse_utc_timestamp(result.ended_at, f"{location} ended_at")
    if ended_at < started_at:
        raise RoundExecutionError(f"{location} ended_at must not precede started_at")
    if result.log_path != str(task.log_path):
        raise RoundExecutionError(f"{location} log_path does not match the task")

    planned = _planned_task_artifacts(task)
    if len(set(result.artifact_paths)) != len(result.artifact_paths) or any(
        path not in planned for path in result.artifact_paths
    ):
        raise RoundExecutionError(
            f"{location} artifact_paths must be unique planned task artifacts"
        )
    if result.status in {"completed", "reused"} and result.artifact_paths != planned:
        raise RoundExecutionError(
            f"{location} artifact_paths must contain every planned artifact"
        )
    if len(set(result.reused_artifacts)) != len(result.reused_artifacts) or any(
        path not in result.artifact_paths for path in result.reused_artifacts
    ):
        raise RoundExecutionError(
            f"{location} reused_artifacts must be a unique artifact_paths subset"
        )
    if result.status == "reused" and result.reused_artifacts != result.artifact_paths:
        raise RoundExecutionError(
            f"{location} reused status must identify every artifact as reused"
        )


def _validated_partial_results(
    submitted: tuple[RoundTask, ...],
    results: tuple[RoundResult, ...],
) -> tuple[RoundResult, ...]:
    tasks_by_round = {task.round: task for task in submitted}
    valid: list[RoundResult] = []
    seen: set[int] = set()
    for result in results:
        if type(result) is not RoundResult:
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
        "execution": {},
        "rounds": [round_result.to_dict() for round_result in result.rounds],
    }
    _write_text_atomically(path, canonical_json(payload) + "\n")


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
    try:
        _ensure_scientific_snapshot(plan, layout)
        status = inspect_workflow_status(
            plan.execution.paths.output_root,
            plan.dataset_id,
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
            executed = _ordered_results(submitted, executor.execute(submitted))
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
        )
        _write_run_record(
            run_record,
            run_id=run_id,
            plan=plan,
            result=result,
            force_all=force_all,
            force_rounds=force_rounds,
        )
    except KeyboardInterrupt as interruption:
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
        )
        _write_run_record(
            run_record,
            run_id=run_id,
            plan=plan,
            result=cancelled,
            force_all=force_all,
            force_rounds=force_rounds,
        )
        if original_interruption is interruption:
            raise
        raise original_interruption from None
    if unsuccessful:
        raise WorkflowExecutionError(unsuccessful, str(run_record))
    return result


__all__ = ["run_workflow"]
