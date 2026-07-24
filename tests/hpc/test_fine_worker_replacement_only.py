from __future__ import annotations

import getpass
import json
from dataclasses import replace
from functools import partial
from typing import cast

import pytest

from cyexm.workflow.models import WorkflowResult
from cyexm.workflow.planning import build_workflow_plan
from cyexm.workflow.task import run_workflow_task
from hpc.environment import HpcAcceptanceContext, write_evidence_json
from hpc.fine_workflow_acceptance import fine_acceptance_manifest
from hpc.test_fine_grained_workflow import _replacement_lineage
from hpc.workflow_acceptance import (
    HpcWorkflowAcceptanceError,
    execute_owned_slurm_workflow,
    run_fine_worker_loss_task,
)

pytestmark = pytest.mark.hpc


def test_fine_worker_replacement_only(hpc_context: HpcAcceptanceContext) -> None:
    spec = hpc_context.profile.slurm
    if (
        hpc_context.profile.parallelism.mode != "fine_grained"
        or spec is None
        or spec.workers < 2
        or spec.processes_per_worker != 1
        or spec.task_retries < 1
    ):
        raise HpcWorkflowAcceptanceError(
            "replacement-only acceptance requires a fine-grained profile with "
            "at least two dedicated one-process workers and one task retry"
        )
    manifest, targets = fine_acceptance_manifest(hpc_context.manifest)
    root = hpc_context.profile.paths.output_root / "cyexm-fine-replacement-only"
    if root.exists():
        raise RuntimeError("replacement-only output root must be fresh")
    root.mkdir()
    profile = replace(
        hpc_context.profile,
        paths=replace(hpc_context.profile.paths, output_root=root),
    )
    plan = build_workflow_plan(manifest, profile)
    marker = hpc_context.evidence_dir / "fine-worker-loss-marker.json"
    runner = partial(
        run_fine_worker_loss_task,
        target_key=targets["replacement_target"],
        loss_marker=marker,
    )
    result = cast(
        WorkflowResult,
        execute_owned_slurm_workflow(
            plan,
            runner=run_workflow_task,
            fine_runner=runner,
            evidence_dir=hpc_context.evidence_dir,
            operation="fine-worker-replacement",
            user=getpass.getuser(),
        ),
    )
    evidence = cast(dict[str, object], json.loads(result.execution_evidence_json))
    evidence_path = (
        hpc_context.evidence_dir / "fine-worker-replacement-slurm-execution.json"
    )
    lineage = _replacement_lineage(
        evidence,
        evidence_path=evidence_path,
        outcome_path=(
            hpc_context.evidence_dir / "fine-worker-replacement-outcome.json"
        ),
        queue_after_path=(
            hpc_context.evidence_dir / "fine-worker-replacement-queue-after.json"
        ),
        loss_marker_path=marker,
        target_key=targets["replacement_target"],
    )
    report = {
        "schema_version": 1,
        "git_commit": hpc_context.git_commit,
        "dataset_id": manifest.dataset_id,
        "status": result.status,
        "replacement_lineage": lineage,
    }
    write_evidence_json(
        hpc_context.evidence_dir / "fine-worker-replacement-only.json",
        report,
    )
    assert result.status == "completed"
