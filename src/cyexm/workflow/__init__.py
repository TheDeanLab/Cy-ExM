from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cyexm.config import load_dataset_manifest, load_execution_profile
    from cyexm.workflow.api import run_workflow
    from cyexm.workflow.planning import build_workflow_plan
    from cyexm.workflow.status import inspect_workflow_status


def __getattr__(name: str) -> object:
    if name in {"load_dataset_manifest", "load_execution_profile"}:
        from cyexm import config

        value = getattr(config, name)
    elif name == "run_workflow":
        from cyexm.workflow.api import run_workflow

        value = run_workflow
    elif name == "build_workflow_plan":
        from cyexm.workflow.planning import build_workflow_plan

        value = build_workflow_plan
    elif name == "inspect_workflow_status":
        from cyexm.workflow.status import inspect_workflow_status

        value = inspect_workflow_status
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})


__all__ = [
    "build_workflow_plan",
    "inspect_workflow_status",
    "load_dataset_manifest",
    "load_execution_profile",
    "run_workflow",
]
