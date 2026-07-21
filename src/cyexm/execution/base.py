from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from cyexm.workflow.models import ReferenceTask, RoundResult, RoundTask, WorkflowTask

type WorkflowTaskRunner = Callable[[WorkflowTask], RoundResult]
type RoundTaskRunner = Callable[[RoundTask], RoundResult]


class WorkflowExecutor(Protocol):
    """Scheduler-neutral execution boundary for immutable round tasks."""

    def execute(
        self,
        reference_task: ReferenceTask,
        round_tasks: tuple[RoundTask, ...],
    ) -> tuple[RoundResult, ...]:
        raise NotImplementedError


__all__ = ["RoundTaskRunner", "WorkflowExecutor", "WorkflowTaskRunner"]
