from __future__ import annotations

from cyexm.workflow.models import ReferenceTask, RoundResult, RoundTask, WorkflowTask


def run_workflow_task(task: WorkflowTask) -> RoundResult:
    if type(task) is ReferenceTask:
        from cyexm.workflow.reference_task import run_reference_task

        return run_reference_task(task)
    if type(task) is RoundTask:
        from cyexm.workflow.round_task import run_round_task

        return run_round_task(task)
    raise TypeError(
        f"workflow task must be exact ReferenceTask or RoundTask; got {task!r}"
    )


__all__ = ("run_workflow_task",)
