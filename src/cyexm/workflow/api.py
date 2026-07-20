from __future__ import annotations

import json
import os
import socket
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from cyexm.config.models import ExecutionProfile
from cyexm.errors import (
    ExecutorStartupError,
    RoundExecutionError,
    WorkflowExecutionError,
    WorkflowPlanError,
)
from cyexm.execution.base import WorkflowExecutor
from cyexm.execution.sequential import SequentialExecutor
from cyexm.io.atomic import atomic_write
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
    _read_scientific_snapshot,
    _safe_workflow_layout,
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


def _ensure_scientific_snapshot(plan: WorkflowPlan, layout: WorkflowLayout) -> None:
    _validate_plan_identity(plan)
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
    expected = tuple(task.round for task in submitted)
    actual = tuple(result.round for result in results)
    if len(set(actual)) != len(actual) or set(actual) != set(expected):
        raise RoundExecutionError(
            "Executor must return exactly one result per submitted round; "
            f"expected={expected!r}, got={actual!r}"
        )
    return tuple(sorted(results, key=lambda result: result.round))


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
    _validate_force_arguments(plan, force_rounds, force_all)
    layout = _safe_workflow_layout(
        plan.execution.paths.output_root,
        plan.dataset_id,
    )
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

    started_at = _utc_now()
    run_id = str(uuid4())
    run_record = layout.run_record(run_id)
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
    reused_rounds = frozenset(result.round for result in reused)
    submitted = tuple(
        replace(
            task,
            force=force_all or task.round in force_rounds,
        )
        for task in plan.tasks
        if task.round not in reused_rounds
    )
    try:
        if submitted:
            factory = _executor_factory or _default_executor_factory
            executor = factory(plan.execution)
            executed = _ordered_results(submitted, executor.execute(submitted))
        else:
            executed = ()
    except KeyboardInterrupt:
        cancelled = WorkflowResult(
            dataset_id=plan.dataset_id,
            scientific_fingerprint=plan.scientific_fingerprint,
            status="cancelled",
            started_at=started_at,
            ended_at=_utc_now(),
            rounds=tuple(sorted(reused, key=lambda result: result.round)),
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
        raise

    rounds = tuple(sorted((*reused, *executed), key=lambda result: result.round))
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
    if unsuccessful:
        raise WorkflowExecutionError(unsuccessful, str(run_record))
    return result


__all__ = ["run_workflow"]
