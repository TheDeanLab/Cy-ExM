from __future__ import annotations

from typing import cast

from cyexm.workflow.fine_models import (
    ChannelTransformTask,
    ChunkAssemblyTask,
    FineWorkflowTask,
    NonlinearChunkTask,
    RoundPreparationTask,
    RoundValidationTask,
    WholeImageRegistrationTask,
)
from cyexm.workflow.models import (
    ReferenceTask,
    RoundTask,
    WorkflowTask,
    WorkflowTaskResult,
)


def run_workflow_task(task: WorkflowTask) -> WorkflowTaskResult:
    if type(task) is ReferenceTask:
        from cyexm.workflow.reference_task import run_reference_task

        return run_reference_task(task)
    if type(task) is RoundTask:
        from cyexm.workflow.round_task import run_round_task

        return run_round_task(task)
    if type(task) in (
        RoundPreparationTask,
        NonlinearChunkTask,
        ChunkAssemblyTask,
        WholeImageRegistrationTask,
        ChannelTransformTask,
        RoundValidationTask,
    ):
        from cyexm.workflow.fine_task import run_fine_task

        return run_fine_task(cast(FineWorkflowTask, task))
    raise TypeError(
        "workflow task must be exact ReferenceTask or RoundTask or an exact "
        f"fine task; got {task!r}"
    )


__all__ = ("run_workflow_task",)
