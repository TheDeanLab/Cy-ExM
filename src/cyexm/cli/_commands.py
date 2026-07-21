from __future__ import annotations

import argparse
import json
import signal
from dataclasses import dataclass
from types import FrameType
from typing import Protocol, cast


class _StructuredValue(Protocol):
    def to_dict(self) -> dict[str, object]: ...


class _Paths(Protocol):
    dataset_root: object
    output_root: object
    worker_local_directory: object | None


class _NativeThreads(Protocol):
    itk: int
    blas: int


class _LocalDask(Protocol):
    workers: int
    threads_per_worker: int
    processes: bool
    memory_limit: str
    startup_timeout: str
    task_retries: int


class _Slurm(Protocol):
    queue: str
    workers: int
    cores_per_worker: int
    processes_per_worker: int
    memory: str
    walltime: str
    interface: str
    scheduler_interface: str
    startup_timeout: str
    task_retries: int
    job_name: str
    job_extra_directives: tuple[str, ...]


class _Execution(Protocol):
    executor: str
    paths: _Paths
    native_threads: _NativeThreads
    local_dask: _LocalDask | None
    slurm: _Slurm | None


class _RoundTask(Protocol):
    round: int


class _Plan(Protocol):
    dataset_id: str
    scientific_fingerprint: str
    scientific_json: str
    tasks: tuple[_RoundTask, ...]
    execution: _Execution


class _WorkflowApi(Protocol):
    def load_dataset_manifest(self, path: str) -> object: ...

    def load_execution_profile(self, path: str) -> object: ...

    def build_workflow_plan(self, manifest: object, execution: object) -> _Plan: ...

    def run_workflow(
        self,
        plan: _Plan,
        *,
        force_rounds: frozenset[int],
        force_all: bool,
    ) -> _StructuredValue: ...

    def inspect_workflow_status(
        self,
        output_root: str,
        dataset_id: str,
    ) -> _StructuredValue: ...


@dataclass(frozen=True, slots=True)
class CommandOutput:
    payload: dict[str, object]
    human: str


def _load_workflow_api() -> _WorkflowApi:
    import cyexm.workflow as workflow

    return cast(_WorkflowApi, workflow)


def _build_plan(api: _WorkflowApi, namespace: argparse.Namespace) -> _Plan:
    manifest = api.load_dataset_manifest(namespace.dataset_yaml)
    execution = api.load_execution_profile(namespace.execution_yaml)
    return api.build_workflow_plan(manifest, execution)


def _path_mapping(paths: _Paths) -> dict[str, object]:
    worker_local_directory = paths.worker_local_directory
    return {
        "dataset_root": str(paths.dataset_root),
        "output_root": str(paths.output_root),
        "worker_local_directory": (
            None if worker_local_directory is None else str(worker_local_directory)
        ),
    }


def _local_dask_mapping(spec: _LocalDask) -> dict[str, object]:
    return {
        "workers": spec.workers,
        "threads_per_worker": spec.threads_per_worker,
        "processes": spec.processes,
        "memory_limit": spec.memory_limit,
        "startup_timeout": spec.startup_timeout,
        "task_retries": spec.task_retries,
    }


def _slurm_mapping(spec: _Slurm) -> dict[str, object]:
    return {
        "queue": spec.queue,
        "workers": spec.workers,
        "cores_per_worker": spec.cores_per_worker,
        "processes_per_worker": spec.processes_per_worker,
        "memory": spec.memory,
        "walltime": spec.walltime,
        "interface": spec.interface,
        "scheduler_interface": spec.scheduler_interface,
        "startup_timeout": spec.startup_timeout,
        "task_retries": spec.task_retries,
        "job_name": spec.job_name,
        "job_extra_directives": list(spec.job_extra_directives),
    }


def _resource_request(execution: _Execution) -> dict[str, object]:
    native_threads = execution.native_threads
    resources: dict[str, object] = {
        "executor": execution.executor,
        "native_threads": {
            "itk": native_threads.itk,
            "blas": native_threads.blas,
        },
        "paths": _path_mapping(execution.paths),
    }
    local_dask = execution.local_dask
    if local_dask is not None:
        resources["local_dask"] = _local_dask_mapping(local_dask)
    slurm = execution.slurm
    if slurm is not None:
        resources["slurm"] = _slurm_mapping(slurm)
    return resources


def _round_numbers(plan: _Plan) -> list[int]:
    return [task.round for task in plan.tasks]


def _validation_output(plan: _Plan) -> CommandOutput:
    rounds = _round_numbers(plan)
    return CommandOutput(
        payload={
            "dataset_id": plan.dataset_id,
            "rounds": rounds,
            "scientific_fingerprint": plan.scientific_fingerprint,
            "valid": True,
        },
        human=(
            f"Validated dataset {plan.dataset_id}: {len(rounds)} round(s); "
            f"scientific fingerprint {plan.scientific_fingerprint}"
        ),
    )


def _plan_output(plan: _Plan) -> CommandOutput:
    scientific_plan = json.loads(plan.scientific_json)
    rounds = _round_numbers(plan)
    executor = plan.execution.executor
    resources = _resource_request(plan.execution)
    return CommandOutput(
        payload={
            "dataset_id": plan.dataset_id,
            "resource_request": resources,
            "scientific_fingerprint": plan.scientific_fingerprint,
            "scientific_plan": scientific_plan,
        },
        human=(
            f"Planned dataset {plan.dataset_id}: {len(rounds)} round(s) using "
            f"{executor}; scientific fingerprint {plan.scientific_fingerprint}\n"
            f"Scientific plan: {plan.scientific_json}\n"
            "Resource request: "
            + json.dumps(resources, allow_nan=False, sort_keys=True)
        ),
    )


def _sigterm_as_keyboard_interrupt(
    signum: int,
    frame: FrameType | None,
) -> None:
    del signum, frame
    raise KeyboardInterrupt


def _run(api: _WorkflowApi, namespace: argparse.Namespace) -> CommandOutput:
    plan = _build_plan(api, namespace)
    previous_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _sigterm_as_keyboard_interrupt)
    try:
        result = api.run_workflow(
            plan,
            force_rounds=frozenset(namespace.force_round),
            force_all=namespace.force_all,
        )
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    payload = result.to_dict()
    return CommandOutput(
        payload=payload,
        human=(
            f"Workflow {payload['dataset_id']}: {payload['status']}; "
            f"run record {payload['run_record']}"
        ),
    )


def _status(api: _WorkflowApi, namespace: argparse.Namespace) -> CommandOutput:
    status = api.inspect_workflow_status(namespace.output_root, namespace.dataset_id)
    payload = status.to_dict()
    state = "complete" if payload["complete"] else "incomplete"
    rounds = payload["rounds"]
    round_count = len(rounds) if isinstance(rounds, list) else 0
    return CommandOutput(
        payload=payload,
        human=(f"Workflow {payload['dataset_id']}: {state}; {round_count} round(s)"),
    )


def dispatch(namespace: argparse.Namespace) -> CommandOutput:
    """Delegate one parsed workflow command to the public workflow API."""
    api = _load_workflow_api()
    if namespace.workflow_command == "validate":
        return _validation_output(_build_plan(api, namespace))
    if namespace.workflow_command == "plan":
        return _plan_output(_build_plan(api, namespace))
    if namespace.workflow_command == "run":
        return _run(api, namespace)
    if namespace.workflow_command == "status":
        return _status(api, namespace)
    raise AssertionError(f"Unknown workflow command: {namespace.workflow_command!r}")


__all__ = ["CommandOutput", "dispatch"]
