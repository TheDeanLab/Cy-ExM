from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from cyexm.workflow.models import RoundResult, RoundTask

type RoundTaskRunner = Callable[["RoundTask"], "RoundResult"]


class WorkflowExecutor(Protocol):
    """Scheduler-neutral execution boundary for immutable round tasks."""

    def execute(self, tasks: tuple[RoundTask, ...]) -> tuple[RoundResult, ...]:
        raise NotImplementedError


__all__ = ["RoundTaskRunner", "WorkflowExecutor"]
