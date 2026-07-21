from __future__ import annotations

import getpass
import json
from functools import partial
from pathlib import Path
from typing import cast

import pytest

from cyexm.errors import WorkflowExecutionError
from cyexm.registration.artifacts import ArtifactStore, FileIdentity
from cyexm.workflow.api import run_workflow
from cyexm.workflow.layout import WorkflowLayout
from cyexm.workflow.models import RoundResult, RoundTask, WorkflowPlan, WorkflowResult
from cyexm.workflow.planning import build_workflow_plan
from cyexm.workflow.status import inspect_workflow_status
from cyexm.workflow.task import run_workflow_task
from hpc.environment import (
    HpcAcceptanceContext,
    HpcAcceptanceError,
    collect_queue_snapshot,
    write_evidence_json,
)
from hpc.workflow_acceptance import (
    HpcWorkflowAcceptanceError,
    Task5OutputRoots,
    bounded_acceptance_manifest,
    bounded_slurm_profile,
    coordination_timeout_seconds,
    execute_cancellation_acceptance,
    execute_owned_slurm_workflow,
    parallel_placement_evidence,
    prepare_task5_output_roots,
    run_cancellation_task,
    run_failure_task,
    scientific_parity_evidence,
    sequential_reference_profile,
    task5_output_roots,
    validate_injected_failure_record,
    validate_task5_output_ownership,
)

pytestmark = pytest.mark.hpc


def _round_artifacts(plan: WorkflowPlan, round_number: int) -> tuple[Path, ...]:
    task = next(item for item in plan.tasks if item.round == round_number)
    return tuple(
        path
        for path in plan.expected_artifacts
        if path.is_relative_to(task.registration_directory.parent)
    )


def _artifact_identities(
    plan: WorkflowPlan,
    round_number: int,
) -> tuple[tuple[FileIdentity, FileIdentity], ...]:
    store = ArtifactStore()
    return tuple(
        (
            FileIdentity.from_path(path),
            FileIdentity.from_path(store.record_path(path)),
        )
        for path in _round_artifacts(plan, round_number)
    )


def _identity_evidence(
    identities: tuple[tuple[FileIdentity, FileIdentity], ...],
) -> list[dict[str, object]]:
    return [
        {"artifact": artifact.to_dict(), "completion_record": record.to_dict()}
        for artifact, record in identities
    ]


def _run_records(plan: WorkflowPlan) -> list[dict[str, object]]:
    layout = WorkflowLayout.for_dataset(
        plan.execution.paths.output_root,
        plan.dataset_id,
    )
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(layout.workflow_runs.glob("*.json"))
    ]


def _round_by_strategy(
    tasks: tuple[RoundTask, ...],
    strategy: str,
) -> RoundTask:
    return next(task for task in tasks if task.strategy == strategy)


def _result_by_round(
    result: WorkflowResult,
    round_number: int,
) -> RoundResult:
    return next(item for item in result.rounds if item.round == round_number)


def _assert_resume(
    result: WorkflowResult,
    *,
    reused_round: int,
    recomputed_round: int,
) -> None:
    assert result.succeeded
    reused = _result_by_round(result, reused_round)
    recomputed = _result_by_round(result, recomputed_round)
    assert reused.status == "reused"
    assert reused.reused_artifacts == reused.artifact_paths
    assert recomputed.status == "completed"
    assert recomputed.reused_artifacts != recomputed.artifact_paths
    execution = json.loads(result.execution_evidence_json)
    assert execution["attempt_counts"] == [
        {
            "dataset_id": result.dataset_id,
            "round": recomputed_round,
            "attempts": 1,
        }
    ]


def _run_task5_acceptance(
    context: HpcAcceptanceContext,
    *,
    roots: Task5OutputRoots,
    ownership: dict[str, object],
) -> dict[str, object]:
    manifest = bounded_acceptance_manifest(context.manifest)

    slurm_profile = bounded_slurm_profile(context.profile, roots.parallel)
    parallel_plan = build_workflow_plan(manifest, slurm_profile)
    sequential_plan = build_workflow_plan(
        manifest,
        sequential_reference_profile(context.profile, roots.sequential),
    )
    failure_plan = build_workflow_plan(
        manifest,
        bounded_slurm_profile(context.profile, roots.failure),
    )
    cancellation_plan = build_workflow_plan(
        manifest,
        bounded_slurm_profile(context.profile, roots.cancellation),
    )
    assert len(parallel_plan.tasks) == 2
    assert (
        parallel_plan.scientific_fingerprint == sequential_plan.scientific_fingerprint
    )
    coordination_timeout = coordination_timeout_seconds(context.profile)
    user = getpass.getuser()

    parallel = cast(
        WorkflowResult,
        execute_owned_slurm_workflow(
            parallel_plan,
            runner=run_workflow_task,
            evidence_dir=context.evidence_dir,
            operation="parallel",
            user=user,
        ),
    )
    placement = parallel_placement_evidence(parallel.rounds)
    write_evidence_json(context.evidence_dir / "parallel-placement.json", placement)

    sequential = run_workflow(sequential_plan)
    assert sequential.succeeded
    write_evidence_json(
        context.evidence_dir / "sequential-reference.json",
        sequential.to_dict(),
    )
    parity = scientific_parity_evidence(sequential_plan, parallel_plan)
    write_evidence_json(context.evidence_dir / "scientific-parity.json", parity)

    failure_success = _round_by_strategy(failure_plan.tasks, "whole_image")
    failure_target = _round_by_strategy(failure_plan.tasks, "serial_chunked")
    failure_marker = roots.failure / "committed-before-failure.json"
    failure_runner = partial(
        run_failure_task,
        successful_round=failure_success.round,
        failing_round=failure_target.round,
        committed_marker=failure_marker,
        timeout_seconds=coordination_timeout,
    )
    try:
        execute_owned_slurm_workflow(
            failure_plan,
            runner=failure_runner,
            evidence_dir=context.evidence_dir,
            operation="independent-failure",
            user=user,
        )
    except WorkflowExecutionError as error:
        failure_record = json.loads(Path(error.run_record).read_text(encoding="utf-8"))
    else:
        raise HpcWorkflowAcceptanceError(
            "injected independent round failure unexpectedly succeeded"
        )
    injected_failure = validate_injected_failure_record(
        failure_record,
        successful_round=failure_success.round,
        failing_round=failure_target.round,
        committed_marker=failure_marker,
    )
    failure_identities = _artifact_identities(failure_plan, failure_success.round)
    failure_resume = cast(
        WorkflowResult,
        execute_owned_slurm_workflow(
            failure_plan,
            runner=run_workflow_task,
            evidence_dir=context.evidence_dir,
            operation="failure-resume",
            user=user,
        ),
    )
    _assert_resume(
        failure_resume,
        reused_round=failure_success.round,
        recomputed_round=failure_target.round,
    )
    assert _artifact_identities(failure_plan, failure_success.round) == (
        failure_identities
    )
    write_evidence_json(
        context.evidence_dir / "failure-resume.json",
        {
            "failed_run": failure_record,
            "injected_failure": injected_failure,
            "coordination_timeout_seconds": coordination_timeout,
            "resume": failure_resume.to_dict(),
            "preserved_artifacts": _identity_evidence(failure_identities),
        },
    )

    cancellation_success = _round_by_strategy(cancellation_plan.tasks, "whole_image")
    cancellation_target = _round_by_strategy(
        cancellation_plan.tasks,
        "serial_chunked",
    )
    cancellation_committed = roots.cancellation / "committed-before-cancel.json"
    cancellation_ready = roots.cancellation / "cancellation-ready.json"
    cancellation_runner = partial(
        run_cancellation_task,
        successful_round=cancellation_success.round,
        blocked_round=cancellation_target.round,
        committed_marker=cancellation_committed,
        ready_marker=cancellation_ready,
        timeout_seconds=coordination_timeout,
    )
    exit_code = execute_cancellation_acceptance(
        cancellation_plan,
        runner=cancellation_runner,
        evidence_dir=context.evidence_dir,
        operation="cancellation",
        user=user,
        ready_marker=cancellation_ready,
        successful_round=cancellation_success.round,
        blocked_round=cancellation_target.round,
        timeout_seconds=coordination_timeout,
    )
    assert exit_code == 130
    cancellation_records = _run_records(cancellation_plan)
    assert len(cancellation_records) == 1
    cancelled_record = cancellation_records[0]
    assert cancelled_record["status"] == "cancelled"
    cancellation_execution = cast(dict[str, object], cancelled_record["execution"])
    observed = cast(dict[str, object], cancellation_execution["observed"])
    cancellation = cast(dict[str, object], observed["cancellation"])
    assert cancellation["attempted"] is True
    assert cast(int, cancellation["unfinished_futures"]) >= 1
    assert cancellation["force"] is True
    assert cancellation["succeeded"] is True
    cleanup = cast(dict[str, object], cancellation_execution["cleanup"])
    assert cleanup["order"] == ["client", "cluster", "configuration"]
    assert cast(dict[str, object], cleanup["client"])["succeeded"] is True
    assert cast(dict[str, object], cleanup["cluster"])["succeeded"] is True

    cancellation_status = inspect_workflow_status(
        cancellation_plan.execution.paths.output_root,
        cancellation_plan.dataset_id,
    )
    status_by_round = {item.round: item.status for item in cancellation_status.rounds}
    assert status_by_round[cancellation_success.round] == "complete"
    assert status_by_round[cancellation_target.round] != "complete"
    cancellation_identities = _artifact_identities(
        cancellation_plan,
        cancellation_success.round,
    )
    cancellation_resume = cast(
        WorkflowResult,
        execute_owned_slurm_workflow(
            cancellation_plan,
            runner=run_workflow_task,
            evidence_dir=context.evidence_dir,
            operation="cancellation-resume",
            user=user,
        ),
    )
    _assert_resume(
        cancellation_resume,
        reused_round=cancellation_success.round,
        recomputed_round=cancellation_target.round,
    )
    assert _artifact_identities(cancellation_plan, cancellation_success.round) == (
        cancellation_identities
    )
    write_evidence_json(
        context.evidence_dir / "cancellation-resume.json",
        {
            "exit_code": exit_code,
            "coordination_timeout_seconds": coordination_timeout,
            "cancelled_run": cancelled_record,
            "resume": cancellation_resume.to_dict(),
            "preserved_artifacts": _identity_evidence(cancellation_identities),
        },
    )

    validate_task5_output_ownership(roots, ownership)
    return {
        "schema_version": 1,
        "status": "completed",
        "selected_rounds": [task.round for task in parallel_plan.tasks],
        "placement": placement,
        "scientific_parity": parity,
        "coordination_timeout_seconds": coordination_timeout,
        "failure_resume": {
            "reused_round": failure_success.round,
            "recomputed_round": failure_target.round,
        },
        "cancellation_resume": {
            "exit_code": exit_code,
            "reused_round": cancellation_success.round,
            "recomputed_round": cancellation_target.round,
        },
        "output_ownership": ownership,
    }


def test_real_slurm_workflow_acceptance(
    hpc_context: HpcAcceptanceContext,
) -> None:
    context = hpc_context
    spec = context.profile.slurm
    assert spec is not None
    user = getpass.getuser()
    result: dict[str, object] | None = None
    roots: Task5OutputRoots | None = None
    ownership: dict[str, object] | None = None
    acceptance_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    ownership_error: BaseException | None = None
    try:
        roots = task5_output_roots(context.profile)
        ownership = prepare_task5_output_roots(roots)
        write_evidence_json(
            context.evidence_dir / "task5-output-ownership.json",
            ownership,
        )
        result = _run_task5_acceptance(
            context,
            roots=roots,
            ownership=ownership,
        )
    except BaseException as error:
        acceptance_error = error
    finally:
        if roots is not None and ownership is not None:
            try:
                validate_task5_output_ownership(roots, ownership)
                write_evidence_json(
                    context.evidence_dir / "task5-output-ownership-final.json",
                    {**ownership, "outcome": "confined"},
                )
            except BaseException as error:
                ownership_error = error
        try:
            final_queue = collect_queue_snapshot(spec.job_name, user=user)
            write_evidence_json(
                context.evidence_dir / "workflow-final-queue.json",
                final_queue,
            )
            if final_queue["jobs"] != []:
                raise HpcAcceptanceError(
                    "named Task 5 SLURM jobs remain after workflow acceptance"
                )
        except BaseException as error:
            cleanup_error = error
        write_evidence_json(
            context.evidence_dir / "workflow-acceptance.json",
            {
                "schema_version": 1,
                "status": (
                    "completed"
                    if acceptance_error is None
                    and cleanup_error is None
                    and ownership_error is None
                    else "failed"
                ),
                "result": result,
                "acceptance_error": (
                    None
                    if acceptance_error is None
                    else {
                        "exception_type": type(acceptance_error).__name__,
                        "message": str(acceptance_error),
                    }
                ),
                "cleanup_error": (
                    None
                    if cleanup_error is None
                    else {
                        "exception_type": type(cleanup_error).__name__,
                        "message": str(cleanup_error),
                    }
                ),
                "ownership_error": (
                    None
                    if ownership_error is None
                    else {
                        "exception_type": type(ownership_error).__name__,
                        "message": str(ownership_error),
                    }
                ),
            },
        )
    if cleanup_error is not None:
        raise cleanup_error from acceptance_error
    if ownership_error is not None:
        raise ownership_error from acceptance_error
    if acceptance_error is not None:
        raise acceptance_error
    assert result is not None
