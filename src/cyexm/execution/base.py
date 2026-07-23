from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from cyexm.workflow.fine_models import FineTaskResult, FineWorkflowTask
from cyexm.workflow.models import (
    ReferenceTask,
    RoundResult,
    RoundTask,
    WorkflowTask,
    WorkflowTaskResult,
)

type WorkflowTaskRunner = Callable[[WorkflowTask], WorkflowTaskResult]
type FineWorkflowTaskRunner = Callable[[FineWorkflowTask], FineTaskResult]


class WorkflowExecutor(Protocol):
    """Scheduler-neutral execution boundary for immutable round tasks."""

    def execute(
        self,
        reference_task: ReferenceTask,
        round_tasks: tuple[RoundTask, ...],
    ) -> tuple[RoundResult, ...]:
        raise NotImplementedError


__all__ = ["FineWorkflowTaskRunner", "WorkflowExecutor", "WorkflowTaskRunner"]
