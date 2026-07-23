from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest
from dask.utils import format_bytes, parse_bytes

from cyexm.config.models import (
    ExecutionPaths,
    ExecutionProfile,
    NativeThreadSpec,
    NonlinearRegistrationStageSpec,
    ParallelismSpec,
    RegistrationSpec,
    RegistrationStageSpec,
    SlurmSpec,
)
from cyexm.errors import ExecutorStartupError, RoundExecutionError, WorkerPreflightError
from cyexm.execution import slurm as slurm_module
from cyexm.execution._slurm_evidence import (
    FineSlurmEvidence,
    validate_fine_slurm_evidence,
)
from cyexm.execution.preflight import WorkerProbe
from cyexm.execution.slurm import SlurmExecutor
from cyexm.runtime import NATIVE_THREAD_ENV_VARS
from cyexm.workflow.fine_models import FineTaskResult, FineWorkflowTask
from cyexm.workflow.models import (
    ChannelTask,
    ReferenceChannelTask,
    ReferenceTask,
    RoundFailure,
    RoundResult,
    RoundTask,
    WorkflowTask,
)

pytestmark = pytest.mark.unit

_PACKAGES = (
    "cy-exm",
    "dask",
    "distributed",
    "dask-jobqueue",
    "antspyx",
    "numpy",
    "scipy",
    "tifffile",
)


class _StringAlias(str):
    pass


def _profile(
    tmp_path: Path,
    *,
    workers: int = 4,
    retries: int = 0,
    directives: tuple[str, ...] = (),
    mode: str = "coarse",
) -> ExecutionProfile:
    return ExecutionProfile(
        schema_version=1,
        executor="slurm",
        paths=ExecutionPaths(
            dataset_root=tmp_path / "dataset",
            output_root=tmp_path / "output",
            worker_local_directory=tmp_path / "scratch",
        ),
        native_threads=NativeThreadSpec(itk=2, blas=3),
        local_dask=None,
        slurm=SlurmSpec(
            queue="512GB",
            workers=workers,
            cores_per_worker=72,
            processes_per_worker=1,
            memory="500GB",
            walltime="72:00:00",
            interface="ib0",
            scheduler_interface="ib0",
            startup_timeout="15m",
            task_retries=retries,
            job_name="cyexm-registration",
            job_extra_directives=directives,
        ),
        source_path=tmp_path / "execution.yaml",
        parallelism=ParallelismSpec(mode=mode),  # type: ignore[arg-type]
    )


def _task(
    tmp_path: Path,
    round_number: int,
    *,
    dataset_id: str = "dataset-1",
) -> RoundTask:
    round_root = tmp_path / f"round-{round_number:03d}"
    return RoundTask(
        dataset_id=dataset_id,
        round=round_number,
        strategy="whole_image",
        fixed_image=tmp_path / "fixed.tif",
        moving_image=tmp_path / f"moving-{round_number}.tif",
        registration_directory=round_root / "registration",
        log_path=round_root / "logs" / f"round-{round_number:03d}.log",
        voxel_spacing_xyz=(1.0, 1.0, 1.0),
        registration=RegistrationSpec(
            crop=False,
            linear=RegistrationStageSpec(type="TRSAA", accuracy="high"),
            nonlinear=NonlinearRegistrationStageSpec(
                type="SyNOnly", accuracy="high", metric="mattes", radius=None
            ),
        ),
        native_threads=NativeThreadSpec(itk=2, blas=3),
        channels=(
            ChannelTask(
                source=tmp_path / f"moving-{round_number}.tif",
                output=round_root / "channels" / f"moving-{round_number}.tif",
                interpolation="linear",
            ),
        ),
        chunk_size_zyx=None,
        overlap_fraction=None,
    )


def _reference_task(tmp_path: Path) -> ReferenceTask:
    round_root = tmp_path / "round-001"
    source = tmp_path / "reference-source.tif"
    output = round_root / "channels" / "reference.tif"
    return ReferenceTask(
        dataset_id="dataset-1",
        round=1,
        registration_channel_source=source,
        registration_channel_output=output,
        registration_directory=round_root / "registration",
        crop_bounds_path=None,
        log_path=round_root / "logs" / "round-001.log",
        voxel_spacing_xyz=(1.0, 1.0, 1.0),
        crop=False,
        native_threads=NativeThreadSpec(itk=2, blas=3),
        channels=(ReferenceChannelTask(source=source, output=output),),
    )


def _result(task: RoundTask, *, status: str = "completed") -> RoundResult:
    failure = (
        RoundFailure("round", "RuntimeError", "injected failed result")
        if status == "failed"
        else None
    )
    return RoundResult(
        dataset_id=task.dataset_id,
        round=task.round,
        status=cast(Any, status),
        started_at="2026-07-20T10:00:00Z",
        ended_at="2026-07-20T10:00:01Z",
        duration_seconds=1.0,
        hostname="worker-01",
        pid=123,
        dask_task_key=None,
        slurm_job_id="12345",
        strategy=task.strategy,
        native_threads=(("itk", 2), ("blas", 3)),
        artifact_paths=(
            str(task.registration_directory / f"linear_transform_{task.round}.mat"),
            str(task.registration_directory / f"nonlinear_warp_{task.round}.nii.gz"),
            str(task.registration_directory / f"nonlinear_registered_{task.round}.tif"),
            *(str(channel.output) for channel in task.channels),
        ),
        reused_artifacts=(),
        log_path=str(task.log_path),
        failure=failure,
    )


def _reference_result(task: ReferenceTask, *, status: str = "completed") -> RoundResult:
    failure = (
        RoundFailure("reference", "RuntimeError", "injected failed result")
        if status == "failed"
        else None
    )
    return RoundResult(
        dataset_id=task.dataset_id,
        round=task.round,
        status=cast(Any, status),
        started_at="2026-07-20T10:00:00Z",
        ended_at="2026-07-20T10:00:01Z",
        duration_seconds=1.0,
        hostname="worker-01",
        pid=123,
        dask_task_key=None,
        slurm_job_id="12345",
        strategy="reference-preparation",
        native_threads=(("itk", 2), ("blas", 3)),
        artifact_paths=(str(task.registration_channel_output),),
        reused_artifacts=(),
        log_path=str(task.log_path),
        failure=failure,
    )


def _workflow_result(task: WorkflowTask) -> RoundResult:
    return _reference_result(task) if isinstance(task, ReferenceTask) else _result(task)


def _fine_result(
    task: FineWorkflowTask,
    *,
    status: str = "completed",
    dask_task_key: str | None = "AUTO",
) -> FineTaskResult:
    failure = (
        RoundFailure(task.key.stage, "RuntimeError", "injected failed result")
        if status == "failed"
        else None
    )
    return FineTaskResult(
        task=task,
        status=cast(Any, status),
        started_at="2026-07-20T10:00:00Z",
        ended_at="2026-07-20T10:00:01Z",
        duration_seconds=1.0,
        hostname="node-a",
        pid=123,
        dask_task_key=dask_task_key,
        slurm_job_id="9001",
        native_threads=tuple(
            (
                name,
                task.native_threads.itk
                if name == NATIVE_THREAD_ENV_VARS[0]
                else task.native_threads.blas,
            )
            for name in NATIVE_THREAD_ENV_VARS
        ),
        artifact_paths=(
            tuple(str(path) for path in task.output_paths)
            if status in {"completed", "reused"}
            else ()
        ),
        reused_artifacts=(),
        log_path=str(task.log_path),
        failure=failure,
        prepared_descriptor=None,
    )


def _probe(address: str, *, ok: bool = True) -> dict[str, object]:
    errors = () if ok else ("injected preflight failure",)
    host = address.removeprefix("tcp://").split(":", 1)[0]
    job_id = str(9001 + max(0, ord(host[-1]) - ord("a")))
    return WorkerProbe(
        ok=ok,
        worker_address=address,
        hostname=host,
        pid=123,
        slurm_job_id=job_id,
        package_versions=tuple(sorted((name, "test-version") for name in _PACKAGES)),
        native_threads=tuple(
            sorted(
                (name, "2" if name == NATIVE_THREAD_ENV_VARS[0] else "3")
                for name in NATIVE_THREAD_ENV_VARS
            )
        ),
        dataset_visible=ok,
        output_writable=ok,
        scratch_usable=ok,
        memory_limit_bytes=500_363_689_984 if ok else 0,
        network_host=host,
        errors=errors,
    ).to_dict()


@dataclass(frozen=True)
class _SubmitCall:
    runner: Callable[..., object]
    task: WorkflowTask | FineWorkflowTask
    key: str
    pure: bool
    retries: int
    resources: dict[str, int]


class _FakeFuture:
    def __init__(
        self,
        task: WorkflowTask | FineWorkflowTask,
        key: str,
        outcome: object,
        *,
        done: bool,
    ) -> None:
        self.task = task
        self.key = key
        self.outcome = outcome
        self._done = done

    def done(self) -> bool:
        return self._done


class _FakeAsCompleted(Iterator[tuple[_FakeFuture, object]]):
    def __init__(
        self,
        modules: _FakeSlurmModules,
        futures: list[_FakeFuture],
    ) -> None:
        self._modules = modules
        order = {
            round_number: index for index, round_number in enumerate(modules.order)
        }
        self._queue = sorted(
            futures,
            key=lambda future: order.get(future.task.round, len(order)),
        )
        self._yielded = 0

    def __iter__(self) -> _FakeAsCompleted:
        return self

    def __next__(self) -> tuple[_FakeFuture, object]:
        if (
            self._modules.interrupt_after is not None
            and self._yielded == self._modules.interrupt_after
        ):
            self._modules.interrupt_after = None
            raise KeyboardInterrupt("injected collection interruption")
        if not self._queue:
            raise StopIteration
        self._yielded += 1
        future = self._queue.pop(0)
        future._done = True
        return future, future.outcome

    def add(self, future: _FakeFuture) -> None:
        self._queue.append(future)


class _FakeCluster:
    def __init__(self, modules: _FakeSlurmModules) -> None:
        self._modules = modules

    def job_script(self) -> str:
        if self._modules.job_script_error is not None:
            raise self._modules.job_script_error
        return self._modules.job_script_text

    def scale(self, *, jobs: int) -> None:
        self._modules.scale_calls.append({"jobs": jobs})
        if self._modules.scale_error is not None and (
            self._modules.scale_error_call is None
            or len(self._modules.scale_calls) == self._modules.scale_error_call
        ):
            raise self._modules.scale_error
        if self._modules.dynamic_workers:
            self._modules.requested_jobs = jobs

    def close(self) -> None:
        self._modules.events.append("cluster.close")
        if self._modules.cluster_close_error is not None:
            raise self._modules.cluster_close_error


class _FakeScheduler:
    def __init__(self, modules: _FakeSlurmModules) -> None:
        self._modules = modules

    def add_resources(
        self,
        worker_address: str,
        *,
        resources: dict[str, int],
    ) -> None:
        self._modules.scheduler_resource_calls.append((worker_address, resources))
        self._modules.scheduler_resources[worker_address] = dict(resources)


class _FakeClient:
    def __init__(self, modules: _FakeSlurmModules) -> None:
        self._modules = modules

    def wait_for_workers(self, workers: int, *, timeout: float) -> None:
        self._modules.wait_for_workers_calls.append((workers, timeout))
        if self._modules.wait_error is not None:
            raise self._modules.wait_error

    def scheduler_info(self) -> dict[str, object]:
        self._modules.scheduler_info_calls += 1
        addresses = self._modules.connected_workers
        if self._modules.dynamic_workers:
            addresses = addresses[: self._modules.requested_jobs]
        if (
            self._modules.disappear_after_probe
            and self._modules.run_calls
            and addresses
        ):
            addresses = addresses[:-1]
        return {
            "address": self._modules.scheduler_address,
            "workers": {
                address: {
                    "resources": dict(
                        self._modules.scheduler_resources.get(address, {})
                    )
                }
                for address in addresses
            },
        }

    def run(
        self,
        function: Callable[..., dict[str, object]],
        **kwargs: object,
    ) -> Mapping[str, Mapping[str, object]]:
        self._modules.events.append("client.run")
        self._modules.run_calls.append((function, kwargs))
        if self._modules.run_error is not None:
            raise self._modules.run_error
        selected = kwargs.get("workers")
        addresses = (
            tuple(cast(list[str], selected))
            if selected is not None
            else (
                self._modules.connected_workers[: self._modules.requested_jobs]
                if self._modules.dynamic_workers
                else self._modules.connected_workers
            )
        )
        if self._modules.probe_results is not None:
            return {
                address: self._modules.probe_results[address]
                for address in addresses
                if address in self._modules.probe_results
            }
        return {address: _probe(address) for address in addresses}

    def run_on_scheduler(
        self,
        function: Callable[..., object],
        *,
        worker_address: str,
    ) -> object:
        self._modules.events.append(f"activate:{worker_address}")
        if self._modules.activation_error is not None:
            raise self._modules.activation_error
        return function(
            worker_address=worker_address,
            dask_scheduler=self._modules.fake_scheduler,
        )

    def submit(
        self,
        runner: Callable[..., object],
        task: WorkflowTask | FineWorkflowTask,
        *,
        key: str,
        pure: bool,
        retries: int,
        resources: dict[str, int],
    ) -> _FakeFuture:
        self._modules.events.append(f"submit:{task.round}")
        submit_error = self._modules.submit_errors.get(
            task.round,
            self._modules.submit_error,
        )
        if submit_error is not None:
            raise submit_error
        self._modules.submit_calls.append(
            _SubmitCall(runner, task, key, pure, retries, resources)
        )
        identity = task.key.encoded if hasattr(task, "key") else task.round
        configured = self._modules.outcomes.setdefault(identity, [])
        outcome = (
            configured.pop(0)
            if configured
            else (
                _fine_result(task) if hasattr(task, "key") else _workflow_result(task)
            )
        )
        if type(outcome) is FineTaskResult and outcome.dask_task_key == "AUTO":
            outcome = replace(outcome, dask_task_key=key)
        return _FakeFuture(
            task,
            key,
            outcome,
            done=task.round not in self._modules.pending_rounds,
        )

    def cancel(self, futures: tuple[_FakeFuture, ...], *, force: bool) -> None:
        self._modules.cancel_call = (futures, force)
        if self._modules.cancel_error is not None:
            raise self._modules.cancel_error

    def close(self) -> None:
        self._modules.events.append("client.close")
        if self._modules.client_close_error is not None:
            raise self._modules.client_close_error


class _FakeSlurmModules:
    def __init__(self) -> None:
        self.cluster_kwargs: dict[str, object] | None = None
        self.scale_calls: list[dict[str, int]] = []
        self.wait_for_workers_calls: list[tuple[int, float]] = []
        self.client_timeout: str | None = None
        self.as_completed_call: tuple[bool, bool] | None = None
        self.submit_calls: list[_SubmitCall] = []
        self.run_calls: list[tuple[Callable[..., object], dict[str, object]]] = []
        self.scheduler_info_calls = 0
        self.events: list[str] = []
        self.cancel_call: tuple[tuple[_FakeFuture, ...], bool] | None = None
        self.outcomes: dict[int | str, list[object]] = {}
        self.order: tuple[int, ...] = ()
        self.interrupt_after: int | None = None
        self.pending_rounds: set[int] = set()
        self.connected_workers: tuple[str, ...] = (
            "tcp://node-a:1234",
            "tcp://node-b:1234",
            "tcp://node-c:1234",
            "tcp://node-d:1234",
        )
        self.dynamic_workers = False
        self.requested_jobs = 0
        self.scheduler_resources: dict[str, dict[str, int]] = {}
        self.scheduler_resource_calls: list[tuple[str, dict[str, int]]] = []
        self.activation_error: BaseException | None = None
        self.fake_scheduler = _FakeScheduler(self)
        self.probe_results: Mapping[str, Mapping[str, object]] | None = None
        self.scheduler_address = "tcp://scheduler:8786"
        self.job_script_text = "#!/bin/bash\n#SBATCH --partition=512GB\n"
        self.rendered_header_injections: tuple[str, ...] = ()
        self.rendered_worker_injections: tuple[str, ...] = ()
        self.rendered_worker_memory: str | None = None
        self.disappear_after_probe = False
        self.cluster_error: BaseException | None = None
        self.job_script_error: BaseException | None = None
        self.scale_error: BaseException | None = None
        self.scale_error_call: int | None = None
        self.client_error: BaseException | None = None
        self.wait_error: BaseException | None = None
        self.run_error: BaseException | None = None
        self.submit_error: BaseException | None = None
        self.submit_errors: dict[int, BaseException] = {}
        self.cancel_error: BaseException | None = None
        self.client_close_error: BaseException | None = None
        self.cluster_close_error: BaseException | None = None
        self.config_exit_error: BaseException | None = None
        self.config_overlays: list[dict[str, object]] = []
        self.config_context_active = False

    class _ConfigContext(AbstractContextManager[None]):
        def __init__(
            self,
            modules: _FakeSlurmModules,
            values: dict[str, object],
        ) -> None:
            self._modules = modules
            self._values = values

        def __enter__(self) -> None:
            self._modules.config_overlays.append(self._values)
            self._modules.config_context_active = True

        def __exit__(self, *exc_info: object) -> None:
            del exc_info
            self._modules.config_context_active = False
            if self._modules.config_exit_error is not None:
                raise self._modules.config_exit_error

    def config_set(self, values: dict[str, object]) -> AbstractContextManager[None]:
        return self._ConfigContext(self, values)

    def SLURMCluster(self, **kwargs: object) -> _FakeCluster:
        if self.cluster_error is not None:
            raise self.cluster_error
        self.cluster_kwargs = kwargs
        prologue = "\n".join(cast(list[str], kwargs.get("job_script_prologue", [])))
        directives = "\n".join(
            f"#SBATCH {directive}"
            for directive in cast(list[str], kwargs.get("job_extra_directives", []))
        )
        python = str(kwargs.get("python", "python"))
        worker_command = str(
            kwargs.get("worker_command", "distributed.cli.dask_worker")
        )
        worker_extra_args = " ".join(
            cast(list[str], kwargs.get("worker_extra_args", []))
        )
        worker_memory = format_bytes(parse_bytes(cast(str, kwargs["memory"]))).replace(
            " ", ""
        )
        if self.rendered_worker_memory is not None:
            worker_memory = self.rendered_worker_memory
        rendered_headers = "\n".join(self.rendered_header_injections)
        rendered_worker = " ".join(self.rendered_worker_injections)
        resource_fragment = (
            f"--resources {worker_extra_args.removeprefix('--resources ')} "
            if worker_extra_args
            else ""
        )
        self.job_script_text = (
            f"{kwargs.get('shebang', '#!/bin/bash')}\n"
            f"#SBATCH -J {kwargs['job_name']}\n"
            f"#SBATCH -p {kwargs['queue']}\n"
            "#SBATCH -n 1\n"
            f"#SBATCH --cpus-per-task={kwargs['job_cpu']}\n"
            f"#SBATCH --mem={kwargs['job_mem']}\n"
            f"#SBATCH -t {kwargs['walltime']}\n"
            f"{directives}\n"
            f"{rendered_headers}\n"
            f"{prologue}\n{python} -m {worker_command} tcp://scheduler:8786 "
            f"--name fake --nthreads {kwargs['cores']} "
            f"--memory-limit {worker_memory} --nanny "
            f"--death-timeout {kwargs['death_timeout']} "
            f"--local-directory {kwargs['local_directory']} "
            f"{resource_fragment}"
            f"--interface {kwargs['interface']} {rendered_worker}\n"
        )
        return _FakeCluster(self)

    def Client(self, cluster: _FakeCluster, *, timeout: str) -> _FakeClient:
        del cluster
        self.client_timeout = timeout
        if self.client_error is not None:
            raise self.client_error
        return _FakeClient(self)

    def as_completed(
        self,
        futures: list[_FakeFuture],
        *,
        with_results: bool,
        raise_errors: bool,
    ) -> _FakeAsCompleted:
        self.as_completed_call = (with_results, raise_errors)
        return _FakeAsCompleted(self, futures)

    def parse_timedelta(self, value: str) -> float:
        assert value == "15m"
        return 900.0


def _executor(
    tmp_path: Path,
    modules: _FakeSlurmModules,
    *,
    workers: int = 4,
    retries: int = 0,
    mode: str = "coarse",
) -> SlurmExecutor:
    return SlurmExecutor(
        _profile(tmp_path, workers=workers, retries=retries, mode=mode),
        _modules=lambda: modules,
        _version=lambda name: "test-version" if name in _PACKAGES else "unexpected",
    )


def test_real_module_loader_gets_timeout_parser_from_pinned_dask_utils(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dask_jobqueue = ModuleType("dask_jobqueue")
    dask_jobqueue.SLURMCluster = object  # type: ignore[attr-defined]
    distributed = ModuleType("distributed")
    distributed.Client = object  # type: ignore[attr-defined]
    distributed.as_completed = object()  # type: ignore[attr-defined]
    dask = ModuleType("dask")
    dask.__path__ = []  # type: ignore[attr-defined]
    dask_config = ModuleType("dask.config")
    config_set = object()
    dask_config.set = config_set  # type: ignore[attr-defined]
    dask.config = dask_config  # type: ignore[attr-defined]
    dask_utils = ModuleType("dask.utils")
    timeout_parser = object()
    dask_utils.parse_timedelta = timeout_parser  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "dask_jobqueue", dask_jobqueue)
    monkeypatch.setitem(sys.modules, "distributed", distributed)
    monkeypatch.setitem(sys.modules, "dask", dask)
    monkeypatch.setitem(sys.modules, "dask.utils", dask_utils)

    modules = slurm_module._slurm_modules()

    assert modules.parse_timedelta is timeout_parser
    assert modules.config_set is config_set


def test_slurm_executor_builds_one_process_jobs_with_round_resource(
    tmp_path: Path,
) -> None:
    modules = _FakeSlurmModules()
    modules.connected_workers = modules.connected_workers[:3]
    tasks = tuple(_task(tmp_path, number) for number in (2, 3, 4))

    _executor(tmp_path, modules).execute(_reference_task(tmp_path), tasks)

    assert modules.cluster_kwargs == {
        "config_name": "cyexm-slurm",
        "queue": "512GB",
        "cores": 72,
        "processes": 1,
        "memory": "466GiB",
        "walltime": "72:00:00",
        "job_cpu": 72,
        "job_mem": "466G",
        "interface": "ib0",
        "scheduler_options": {"interface": "ib0"},
        "local_directory": str(tmp_path / "scratch"),
        "nanny": True,
        "death_timeout": 60,
        "worker_command": "distributed.cli.dask_worker",
        "python": sys.executable,
        "shebang": "#!/usr/bin/env bash",
        "job_name": "cyexm-registration",
        "job_script_prologue": [
            f"export {name}={'2' if index == 0 else '3'}"
            for index, name in enumerate(NATIVE_THREAD_ENV_VARS)
        ],
        "job_extra_directives": [],
        "job_directives_skip": [],
        "worker_extra_args": ["--resources", "round_slot=1"],
    }
    assert modules.scale_calls == [{"jobs": 3}]
    assert modules.wait_for_workers_calls == [(3, 900.0)]
    assert modules.client_timeout == "15m"
    assert "#SBATCH --mem=466G" in modules.job_script_text
    assert "--memory-limit 466.00GiB" in modules.job_script_text
    assert (
        slurm_module.validate_slurm_job_script(
            modules.job_script_text,
            _profile(tmp_path),
        )
        == 500_363_689_984
    )


def test_fine_slurm_starts_candidates_without_slots_then_activates_after_preflight(
    tmp_path: Path,
) -> None:
    modules = _FakeSlurmModules()
    modules.dynamic_workers = True
    task = _task(tmp_path, 2)

    results = _executor(tmp_path, modules, workers=3, mode="fine_grained").execute(
        _reference_task(tmp_path), (task,)
    )

    assert modules.cluster_kwargs is not None
    assert modules.cluster_kwargs["worker_extra_args"] == []
    assert "--resources" not in modules.job_script_text
    assert modules.scale_calls[0] == {"jobs": 1}
    assert modules.wait_for_workers_calls[0] == (1, 900.0)
    assert modules.run_calls[0][1]["workers"] == ["tcp://node-a:1234"]
    assert modules.scheduler_resource_calls[0] == (
        "tcp://node-a:1234",
        {"round_slot": 1},
    )
    assert modules.events.index("client.run") < modules.events.index(
        "activate:tcp://node-a:1234"
    )
    assert modules.events.index("activate:tcp://node-a:1234") < modules.events.index(
        "submit:1"
    )
    assert all(call.resources == {"round_slot": 1} for call in modules.submit_calls)
    assert [call.key for call in modules.submit_calls] == [
        "cyexm-dataset-1-reference-round-001",
        "cyexm-round-002-whole-image-main",
        "cyexm-round-002-channel-0000",
        "cyexm-round-002-validation-main",
    ]
    assert tuple(result.round for result in results) == (1, 2)


def test_fine_slurm_scales_up_from_ready_width_without_scaling_down(
    tmp_path: Path,
) -> None:
    modules = _FakeSlurmModules()
    modules.dynamic_workers = True
    tasks = tuple(_task(tmp_path, number) for number in (2, 3, 4))

    _executor(tmp_path, modules, workers=2, mode="fine_grained").execute(
        _reference_task(tmp_path), tasks
    )

    assert modules.scale_calls == [{"jobs": 1}, {"jobs": 2}]
    assert modules.wait_for_workers_calls == [(1, 900.0), (2, 900.0)]
    assert [call[1]["workers"] for call in modules.run_calls] == [
        ["tcp://node-a:1234"],
        ["tcp://node-b:1234"],
    ]
    assert modules.scheduler_resource_calls == [
        ("tcp://node-a:1234", {"round_slot": 1}),
        ("tcp://node-b:1234", {"round_slot": 1}),
    ]


def test_fine_slurm_retries_exact_task_with_deterministic_attempt_identity(
    tmp_path: Path,
) -> None:
    from cyexm.workflow.fine_planning import initial_fine_tasks

    modules = _FakeSlurmModules()
    modules.dynamic_workers = True
    task = _task(tmp_path, 2)
    initial = initial_fine_tasks((task,))[0]
    modules.outcomes[initial.key.encoded] = [
        _fine_result(initial, status="failed"),
        _fine_result(initial),
    ]

    executor = _executor(
        tmp_path,
        modules,
        workers=2,
        retries=1,
        mode="fine_grained",
    )
    results = executor.execute(_reference_task(tmp_path), (task,))

    assert results[1].status == "completed"
    assert [call.key for call in modules.submit_calls[1:3]] == [
        "cyexm-round-002-whole-image-main",
        "cyexm-round-002-whole-image-main-attempt-002",
    ]
    assert executor._fine_attempt_counts[0] == (
        "round-002:whole-image:main",
        2,
    )


def test_fine_slurm_activation_failure_closes_every_owned_resource(
    tmp_path: Path,
) -> None:
    modules = _FakeSlurmModules()
    modules.dynamic_workers = True
    modules.activation_error = RuntimeError("injected activation failure")

    with pytest.raises(ExecutorStartupError, match="activation"):
        _executor(tmp_path, modules, mode="fine_grained").execute(
            _reference_task(tmp_path), (_task(tmp_path, 2),)
        )

    assert modules.events[-2:] == ["client.close", "cluster.close"]


def test_fine_slurm_preflight_failure_never_activates_or_submits(
    tmp_path: Path,
) -> None:
    modules = _FakeSlurmModules()
    modules.dynamic_workers = True
    address = modules.connected_workers[0]
    modules.probe_results = {address: _probe(address, ok=False)}

    with pytest.raises(ExecutorStartupError, match="preflight"):
        _executor(tmp_path, modules, mode="fine_grained").execute(
            _reference_task(tmp_path), (_task(tmp_path, 2),)
        )

    assert modules.scheduler_resource_calls == []
    assert modules.submit_calls == []
    assert modules.events[-2:] == ["client.close", "cluster.close"]


def test_fine_slurm_worker_instability_fails_closed_before_scientific_submit(
    tmp_path: Path,
) -> None:
    modules = _FakeSlurmModules()
    modules.dynamic_workers = True
    modules.disappear_after_probe = True

    with pytest.raises(ExecutorStartupError, match="worker set changed"):
        _executor(tmp_path, modules, mode="fine_grained").execute(
            _reference_task(tmp_path), (_task(tmp_path, 2),)
        )

    assert modules.submit_calls == []
    assert modules.events[-2:] == ["client.close", "cluster.close"]


def test_fine_slurm_cancellation_during_scale_closes_owned_lifecycle(
    tmp_path: Path,
) -> None:
    modules = _FakeSlurmModules()
    modules.dynamic_workers = True
    tasks = (_task(tmp_path, 2), _task(tmp_path, 3))
    modules.scale_error = KeyboardInterrupt("injected scale cancellation")
    modules.scale_error_call = 2

    executor = _executor(tmp_path, modules, workers=2, mode="fine_grained")
    reference = _reference_task(tmp_path)
    with pytest.raises(KeyboardInterrupt, match="scale cancellation"):
        executor.execute(reference, tasks)

    assert modules.scale_calls == [{"jobs": 1}, {"jobs": 2}]
    assert [call.task.round for call in modules.submit_calls] == [1]
    assert modules.events[-2:] == ["client.close", "cluster.close"]
    encoded = slurm_module._canonical_execution_evidence(
        executor.execution_evidence,
        expected_tasks=(reference, *tasks),
        task_retries=0,
    )
    assert (
        cast(dict[str, object], json.loads(encoded)["fine_graph"])["cancelled"] is True
    )


def test_fine_slurm_records_versioned_scale_graph_and_placement_evidence(
    tmp_path: Path,
) -> None:
    modules = _FakeSlurmModules()
    modules.dynamic_workers = True
    tasks = (_task(tmp_path, 2), _task(tmp_path, 3))
    executor = _executor(tmp_path, modules, workers=2, mode="fine_grained")

    executor.execute(_reference_task(tmp_path), tasks)

    evidence = cast(dict[str, object], executor.execution_evidence)
    assert evidence["schema_version"] == 2
    fine = cast(dict[str, object], evidence["fine_graph"])
    assert fine["mode"] == "fine_grained"
    assert fine["worker_cap"] == 2
    assert fine["candidate_workers"] == [
        "tcp://node-a:1234",
        "tcp://node-b:1234",
    ]
    assert fine["validated_workers"] == [
        "tcp://node-a:1234",
        "tcp://node-b:1234",
    ]
    scale_events = cast(list[dict[str, object]], fine["scale_events"])
    assert [event["requested"] for event in scale_events] == [1, 2]
    assert all(event["failed"] == [] for event in scale_events)
    assert cast(list[object], fine["task_graph"])
    assert cast(list[object], fine["attempt_counts"])
    assert cast(list[object], fine["placements"])
    assert fine["cancelled"] is False
    observed = cast(dict[str, object], evidence["observed"])
    validate_fine_slurm_evidence(
        fine,
        configured_workers=2,
        task_retries=0,
        cancellation_present=False,
        round_tasks=tasks,
        observed_probes=tuple(
            WorkerProbe.from_dict(item)
            for item in cast(list[dict[str, object]], observed["worker_probes"])
        ),
    )
    encoded = slurm_module._canonical_execution_evidence(
        evidence,
        expected_tasks=(_reference_task(tmp_path), *tasks),
        task_retries=0,
    )
    assert json.loads(encoded) == evidence


def test_fine_slurm_typed_evidence_snapshots_are_detached_and_canonical() -> None:
    state = FineSlurmEvidence(worker_cap=2)
    event = state.begin_scale(1, "2026-07-23T10:00:00Z")
    event.connected.append("tcp://node-a:1234")
    event.probed.append("tcp://node-a:1234")
    event.activated.append("tcp://node-a:1234")
    event.ended_at = "2026-07-23T10:00:01Z"
    state.candidates.add("tcp://node-a:1234")
    state.validated.add("tcp://node-a:1234")

    first = state.to_dict()
    cast(list[str], first["candidate_workers"]).append("mutated")

    second = state.to_dict()
    assert second["candidate_workers"] == ["tcp://node-a:1234"]
    assert second["scale_events"] == [
        {
            "requested": 1,
            "connected": ["tcp://node-a:1234"],
            "probed": ["tcp://node-a:1234"],
            "activated": ["tcp://node-a:1234"],
            "failed": [],
            "started_at": "2026-07-23T10:00:00Z",
            "ended_at": "2026-07-23T10:00:01Z",
        }
    ]


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (
            lambda fine: cast(list[dict[str, object]], fine["task_graph"])[1].update(
                dependencies=[]
            ),
            "channels must depend",
        ),
        (
            lambda fine: cast(list[dict[str, object]], fine["placements"])[0].update(
                slurm_job_id="wrong-job"
            ),
            "does not match worker probe identity",
        ),
        (
            lambda fine: cast(list[dict[str, object]], fine["scale_events"])[1].update(
                probed=[]
            ),
            "activation must be a subset",
        ),
        (
            lambda fine: cast(list[str], fine["blocked_descendants"]).append(
                "round-002:validation:main"
            ),
            "blocked descendants must be unattempted",
        ),
    ),
)
def test_fine_slurm_evidence_rejects_noncanonical_graph_scale_and_placement(
    tmp_path: Path,
    mutation: Callable[[dict[str, object]], object],
    match: str,
) -> None:
    modules = _FakeSlurmModules()
    modules.dynamic_workers = True
    tasks = (_task(tmp_path, 2), _task(tmp_path, 3))
    executor = _executor(tmp_path, modules, workers=2, mode="fine_grained")
    executor.execute(_reference_task(tmp_path), tasks)
    evidence = deepcopy(cast(dict[str, object], executor.execution_evidence))

    mutation(cast(dict[str, object], evidence["fine_graph"]))

    with pytest.raises(ValueError, match=match):
        slurm_module._canonical_execution_evidence(
            evidence,
            expected_tasks=(_reference_task(tmp_path), *tasks),
            task_retries=0,
        )


def test_slurm_executor_exports_resolved_threads_before_worker_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for index, name in enumerate(NATIVE_THREAD_ENV_VARS):
        if index % 2:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, "999")
    modules = _FakeSlurmModules()
    modules.connected_workers = modules.connected_workers[:1]
    executor = _executor(tmp_path, modules)

    executor.execute(_reference_task(tmp_path), (_task(tmp_path, 2),))

    assert modules.cluster_kwargs is not None
    prologue = cast(list[str], modules.cluster_kwargs["job_script_prologue"])
    assert prologue == [
        f"export {name}={'2' if index == 0 else '3'}"
        for index, name in enumerate(NATIVE_THREAD_ENV_VARS)
    ]
    script = cast(
        dict[str, str],
        cast(dict[str, object], executor.execution_evidence["rendered"]),
    )["rendered_job_script_template"]
    worker_command = " -m distributed.cli.dask_worker"
    for line in prologue:
        assert script.index(line) < script.index(worker_command)
    assert modules.run_calls[0][1]["expected_threads"] == {
        NATIVE_THREAD_ENV_VARS[0]: "2",
        **dict.fromkeys(NATIVE_THREAD_ENV_VARS[1:], "3"),
    }


@pytest.mark.parametrize("directives", [(), ("--qos=lab",)])
def test_slurm_executor_supplies_isolated_jobqueue_launch_configuration(
    tmp_path: Path,
    directives: tuple[str, ...],
) -> None:
    malicious = {
        "account": "evil-account",
        "project": "evil-project",
        "job-script-prologue": ["echo evil-prologue"],
        "env-extra": ["echo evil-env-extra"],
        "job-directives-skip": ["#SBATCH -p", "--cpus-per-task", "--mem"],
        "header-skip": ["#SBATCH -t"],
        "python": "/evil/python",
        "worker-command": "evil.worker",
        "job-extra": ["--constraint=evil-extra"],
        "extra": ["--evil-worker-extra"],
        "shebang": "#!/evil/shell",
        "log-directory": "/evil/logs",
    }
    modules = _FakeSlurmModules()
    modules.connected_workers = modules.connected_workers[:1]
    executor = SlurmExecutor(
        _profile(tmp_path, directives=directives),
        _modules=lambda: modules,
        _version=lambda _name: "test-version",
    )

    executor.execute(_reference_task(tmp_path), (_task(tmp_path, 2),))

    assert len(modules.config_overlays) == 1
    overlay = modules.config_overlays[0]
    assert overlay["distributed.scheduler.worker-ttl"] is None
    for name, hostile_value in malicious.items():
        key = f"jobqueue.cyexm-slurm.{name}"
        assert key in overlay
        assert overlay[key] != hostile_value

    rendered = cast(
        dict[str, str],
        cast(dict[str, object], executor.execution_evidence["rendered"]),
    )["rendered_job_script_template"]
    for forbidden in (
        "evil-account",
        "evil-project",
        "evil-prologue",
        "evil-env-extra",
        "/evil/python",
        "evil.worker",
        "evil-extra",
        "evil-worker-extra",
        "#!/evil/shell",
        "/evil/logs",
    ):
        assert forbidden not in rendered
    for managed in (
        "#SBATCH -p 512GB",
        "#SBATCH --cpus-per-task=72",
        "#SBATCH --mem=466G",
        "#SBATCH -t 72:00:00",
    ):
        assert managed in rendered
    assert ("#SBATCH --qos=lab" in rendered) is bool(directives)


@pytest.mark.parametrize(
    "directive",
    (
        "--nodes=2",
        "-N2",
        "--ntasks-per-node=2",
        "--mem-per-gpu=20GB",
        "--job-name=unauthorized",
        "-Junauthorized",
        "--chdir=/tmp",
        "-D/tmp",
        "--resources=round_slot=2",
        "--nthreads=1",
    ),
)
def test_slurm_executor_revalidates_direct_profile_before_loading_scheduler(
    tmp_path: Path,
    directive: str,
) -> None:
    modules = _FakeSlurmModules()
    executor = SlurmExecutor(
        _profile(tmp_path, directives=(directive,)),
        _modules=lambda: modules,
        _version=lambda _name: "test-version",
    )

    with pytest.raises(ExecutorStartupError, match="execution profile"):
        executor.execute(_reference_task(tmp_path), (_task(tmp_path, 2),))

    assert modules.cluster_kwargs is None
    assert modules.scale_calls == []
    assert modules.submit_calls == []


@pytest.mark.parametrize(
    ("header", "worker"),
    (
        ("#SBATCH --nodes=2", ""),
        ("#SBATCH -N2", ""),
        ("#SBATCH --ntasks-per-node=2", ""),
        ("#SBATCH --mem-per-gpu=20GB", ""),
        ("#SBATCH --partition=unauthorized", ""),
        ("#SBATCH -punauthorized", ""),
        ("#SBATCH --job-name=unauthorized", ""),
        ("#SBATCH -Junauthorized", ""),
        ("#SBATCH --output=/tmp/job.out", ""),
        ("#SBATCH --chdir=/tmp", ""),
        ("", "--resources round_slot=2"),
        ("", "--nthreads 1"),
        ("", "--nworkers 2"),
        ("", "--memory-limit 1GB"),
        ("", "--interface eth0"),
        ("", "--local-directory /tmp"),
    ),
)
def test_slurm_executor_rejects_rendered_resource_collision_before_scale(
    tmp_path: Path,
    header: str,
    worker: str,
) -> None:
    modules = _FakeSlurmModules()
    modules.connected_workers = modules.connected_workers[:1]
    modules.rendered_header_injections = (header,) if header else ()
    modules.rendered_worker_injections = (worker,) if worker else ()

    with pytest.raises(ExecutorStartupError, match="rendered SLURM job script"):
        _executor(tmp_path, modules).execute(
            _reference_task(tmp_path), (_task(tmp_path, 2),)
        )

    assert modules.scale_calls == []
    assert modules.submit_calls == []
    assert modules.events == ["cluster.close"]


@pytest.mark.parametrize("worker_memory", ("465.99GiB", "466.01GiB", "500GB"))
def test_slurm_executor_rejects_inexact_rendered_worker_memory_before_scale(
    tmp_path: Path,
    worker_memory: str,
) -> None:
    modules = _FakeSlurmModules()
    modules.connected_workers = modules.connected_workers[:1]
    modules.rendered_worker_memory = worker_memory

    with pytest.raises(ExecutorStartupError, match="worker memory limit"):
        _executor(tmp_path, modules).execute(
            _reference_task(tmp_path), (_task(tmp_path, 2),)
        )

    assert modules.scale_calls == []
    assert modules.submit_calls == []
    assert modules.events == ["cluster.close"]


def test_slurm_executor_bounds_jobs_by_configured_workers(tmp_path: Path) -> None:
    modules = _FakeSlurmModules()
    tasks = tuple(_task(tmp_path, number) for number in (2, 3, 4))
    modules.connected_workers = modules.connected_workers[:2]

    _executor(tmp_path, modules, workers=2).execute(_reference_task(tmp_path), tasks)

    assert modules.scale_calls == [{"jobs": 2}]
    assert modules.wait_for_workers_calls == [(2, 900.0)]


def test_slurm_executor_runs_reference_when_no_moving_rounds_are_pending(
    tmp_path: Path,
) -> None:
    loaded = False

    def modules() -> _FakeSlurmModules:
        nonlocal loaded
        loaded = True
        value = _FakeSlurmModules()
        value.connected_workers = value.connected_workers[:1]
        return value

    executor = SlurmExecutor(_profile(tmp_path), _modules=modules)

    results = executor.execute(_reference_task(tmp_path), ())

    assert tuple(result.round for result in results) == (1,)
    assert loaded is True
    evidence = executor.execution_evidence
    assert cast(dict[str, object], evidence["requested"])["workers"] == 1
    assert cast(dict[str, object], evidence["startup"])["outcome"] == "succeeded"
    assert evidence["attempt_counts"] == [
        {"dataset_id": "dataset-1", "round": 1, "attempts": 1}
    ]


def test_slurm_executor_captures_exact_script_versions_scheduler_and_probes(
    tmp_path: Path,
) -> None:
    modules = _FakeSlurmModules()
    modules.connected_workers = modules.connected_workers[:1]
    executor = _executor(tmp_path, modules)

    executor.execute(_reference_task(tmp_path), (_task(tmp_path, 2),))

    evidence = executor.execution_evidence
    assert evidence["executor"] == "slurm"
    assert evidence["requested"] == {
        "workers": 1,
        "configured_workers": 4,
        "startup_timeout": "15m",
        "task_retries": 0,
        "memory": {
            "profile": "500GB",
            "semantic_bytes": 500_000_000_000,
            "scheduler": "466G",
            "scheduler_bytes": 500_363_689_984,
            "dask": "466GiB",
        },
    }
    rendered = cast(dict[str, object], evidence["rendered"])
    assert rendered["rendered_job_script_template"] == modules.job_script_text
    observed = cast(dict[str, object], evidence["observed"])
    assert "submitted_job_script" not in observed
    assert observed["scheduler_address"] == modules.scheduler_address
    assert observed["connected_workers"] == ["tcp://node-a:1234"]
    assert cast(dict[str, str], observed["coordinator_versions"]) == dict.fromkeys(
        _PACKAGES,
        "test-version",
    )
    assert (
        cast(list[dict[str, object]], observed["worker_probes"])[0]["worker_address"]
        == "tcp://node-a:1234"
    )
    assert evidence["attempt_counts"] == [
        {"dataset_id": "dataset-1", "round": 1, "attempts": 1},
        {"dataset_id": "dataset-1", "round": 2, "attempts": 1},
    ]
    timestamps = cast(dict[str, str], evidence["timestamps"])
    assert timestamps["started_at"].endswith("Z")
    assert timestamps["ended_at"].endswith("Z")
    assert timestamps["ended_at"] >= timestamps["started_at"]
    assert evidence["startup"] == {
        "phase": "complete",
        "outcome": "succeeded",
        "scale_attempted": True,
        "scale_succeeded": True,
    }
    assert evidence["cleanup"] == {
        "client": {"attempted": True, "succeeded": True},
        "cluster": {"attempted": True, "succeeded": True},
        "configuration": {"attempted": True, "succeeded": True},
        "order": ["client", "cluster", "configuration"],
    }


def test_slurm_execution_evidence_is_detached_canonical_json(
    tmp_path: Path,
) -> None:
    modules = _FakeSlurmModules()
    modules.connected_workers = modules.connected_workers[:1]
    executor = _executor(tmp_path, modules)
    executor.execute(_reference_task(tmp_path), (_task(tmp_path, 2),))

    first = cast(dict[str, object], executor.execution_evidence)
    encoded = json.dumps(first, allow_nan=False, sort_keys=True, separators=(",", ":"))
    assert json.loads(encoded) == first
    cast(dict[str, object], first["requested"])["workers"] = 999
    cast(dict[str, object], first["rendered"])["rendered_job_script_template"] = (
        "mutated"
    )

    second = executor.execution_evidence
    assert cast(dict[str, object], second["requested"])["workers"] == 1
    assert (
        cast(dict[str, object], second["rendered"])["rendered_job_script_template"]
        == modules.job_script_text
    )


def _invalid_execution_evidence(
    evidence: dict[str, object],
    case: str,
) -> None:
    requested = cast(dict[str, object], evidence["requested"])
    rendered = cast(dict[str, object], evidence["rendered"])
    observed = cast(dict[str, object], evidence["observed"])
    startup = cast(dict[str, object], evidence["startup"])
    attempts = cast(list[dict[str, object]], evidence["attempt_counts"])
    timestamps = cast(dict[str, object], evidence["timestamps"])
    cleanup = cast(dict[str, object], evidence["cleanup"])
    if case == "root_missing":
        evidence.pop("executor")
    elif case == "root_extra":
        evidence["submitted_job_script"] = "stale"
    elif case == "root_key_alias":
        value = evidence.pop("executor")
        evidence[_StringAlias("executor")] = value
    elif case == "executor":
        evidence["executor"] = "local_dask"
    elif case == "requested_missing":
        requested.pop("workers")
    elif case == "requested_extra":
        requested["queue"] = "512GB"
    elif case == "requested_bool":
        requested["workers"] = True
    elif case == "requested_bound":
        requested["workers"] = 5
    elif case == "requested_retry_mismatch":
        requested["task_retries"] = 1
    elif case == "requested_memory_profile_mismatch":
        cast(dict[str, object], requested["memory"])["profile"] = "499GB"
    elif case == "requested_memory_semantic_mismatch":
        cast(dict[str, object], requested["memory"])["semantic_bytes"] = 1
    elif case == "requested_memory_scheduler_mismatch":
        cast(dict[str, object], requested["memory"])["scheduler"] = "500G"
    elif case == "requested_memory_bytes_mismatch":
        cast(dict[str, object], requested["memory"])["scheduler_bytes"] = 1
    elif case == "requested_memory_dask_mismatch":
        cast(dict[str, object], requested["memory"])["dask"] = "500GiB"
    elif case == "rendered_stale":
        rendered["submitted_job_script"] = rendered.pop("rendered_job_script_template")
    elif case == "rendered_missing":
        rendered.clear()
    elif case == "rendered_alias":
        rendered["rendered_job_script_template"] = _StringAlias("script")
    elif case == "observed_extra":
        observed["job_ids"] = ["98765"]
    elif case == "scheduler_without_workers":
        observed.pop("connected_workers")
    elif case == "workers_unsorted":
        observed["connected_workers"] = ["tcp://z:1", "tcp://a:1"]
    elif case == "workers_duplicate":
        worker = cast(list[str], observed["connected_workers"])[0]
        observed["connected_workers"] = [worker, worker]
    elif case == "version_key_alias":
        versions = cast(dict[str, str], observed["coordinator_versions"])
        value = versions.pop("cy-exm")
        versions[_StringAlias("cy-exm")] = value
    elif case == "version_missing":
        cast(dict[str, str], observed["coordinator_versions"]).pop("scipy")
    elif case == "probe_extra":
        probe = cast(list[dict[str, object]], observed["worker_probes"])[0]
        probe["unknown"] = True
    elif case == "probe_address":
        probe = cast(list[dict[str, object]], observed["worker_probes"])[0]
        probe["worker_address"] = "tcp://different:1"
    elif case == "probe_version_mismatch":
        probe = cast(list[dict[str, object]], observed["worker_probes"])[0]
        versions = cast(list[list[str]], probe["package_versions"])
        versions[0][1] = "different-version"
    elif case == "probe_not_ok_after_complete":
        probe = cast(list[dict[str, object]], observed["worker_probes"])[0]
        probe["ok"] = False
        probe["errors"] = ["injected valid probe failure"]
    elif case == "cancellation_not_attempted_success":
        observed["cancellation"] = {
            "attempted": False,
            "unfinished_futures": 0,
            "succeeded": True,
        }
    elif case == "cancellation_attempted_no_force":
        observed["cancellation"] = {
            "attempted": True,
            "unfinished_futures": 1,
            "succeeded": True,
        }
    elif case == "startup_phase":
        startup["phase"] = "unknown"
    elif case == "startup_outcome":
        startup["outcome"] = "in_progress"
    elif case == "startup_success_error":
        startup["error"] = {"type": "RuntimeError", "message": "impossible"}
    elif case == "startup_scale_attempted":
        startup["scale_attempted"] = False
    elif case == "startup_scale_succeeded":
        startup["scale_succeeded"] = None
    elif case == "early_observed_facts":
        startup.update(
            {
                "phase": "worker_wait",
                "outcome": "failed",
                "error": {"type": "RuntimeError", "message": "wait failed"},
            }
        )
        attempts.clear()
    elif case == "interrupted_without_cancellation":
        startup.update(
            {
                "phase": "worker_wait",
                "outcome": "interrupted",
                "error": {"type": "KeyboardInterrupt", "message": "stopped"},
            }
        )
        observed.clear()
        attempts.clear()
    elif case == "attempt_missing":
        attempts[0].pop("round")
    elif case == "attempt_bool":
        attempts[0]["attempts"] = True
    elif case == "attempt_duplicate":
        attempts.append(dict(attempts[0]))
    elif case == "attempt_dataset":
        attempts[0]["dataset_id"] = "unrelated-dataset"
    elif case == "attempt_round":
        attempts[0]["round"] = 999
    elif case == "attempt_retry_overflow":
        attempts[0]["attempts"] = 100
    elif case == "attempt_empty":
        attempts.clear()
    elif case == "attempt_zero_without_cancel":
        attempts[0]["attempts"] = 0
    elif case == "timestamps_missing":
        timestamps.pop("ended_at")
    elif case == "timestamps_non_utc":
        timestamps["started_at"] = "2026-07-20T10:00:00+01:00"
    elif case == "timestamps_reverse":
        timestamps["started_at"] = "9999-07-20T10:00:00Z"
    elif case == "cleanup_extra":
        cleanup["scheduler"] = {"attempted": True, "succeeded": True}
    elif case == "cleanup_order":
        cleanup["order"] = ["cluster", "client", "configuration"]
    elif case == "cleanup_attempted":
        cast(dict[str, object], cleanup["client"])["attempted"] = False
    elif case == "cleanup_nullable":
        cast(dict[str, object], cleanup["client"])["succeeded"] = None
    elif case == "cleanup_success_error":
        cast(dict[str, object], cleanup["client"])["error"] = {
            "type": "RuntimeError",
            "message": "impossible",
        }
    elif case == "cleanup_failure_without_error":
        cast(dict[str, object], cleanup["client"])["succeeded"] = False
    else:
        raise AssertionError(f"unknown invalid evidence case: {case}")


def test_slurm_evidence_schema_returns_a_detached_canonical_snapshot(
    tmp_path: Path,
) -> None:
    modules = _FakeSlurmModules()
    modules.connected_workers = modules.connected_workers[:1]
    executor = _executor(tmp_path, modules)
    task = _task(tmp_path, 2)
    executor.execute(_reference_task(tmp_path), (task,))
    evidence = cast(dict[str, object], executor.execution_evidence)

    encoded = slurm_module._canonical_execution_evidence(
        evidence,
        expected_tasks=(_reference_task(tmp_path), task),
        task_retries=0,
    )

    assert json.loads(encoded) == evidence
    cast(dict[str, object], evidence["rendered"])["rendered_job_script_template"] = (
        "mutated"
    )
    assert (
        cast(dict[str, object], json.loads(encoded)["rendered"])[
            "rendered_job_script_template"
        ]
        == modules.job_script_text
    )


def test_slurm_evidence_schema_preserves_truthful_failed_probe_mismatch(
    tmp_path: Path,
) -> None:
    modules = _FakeSlurmModules()
    modules.connected_workers = modules.connected_workers[:1]
    executor = _executor(tmp_path, modules)
    task = _task(tmp_path, 2)
    executor.execute(_reference_task(tmp_path), (task,))
    evidence = deepcopy(cast(dict[str, object], executor.execution_evidence))
    startup = cast(dict[str, object], evidence["startup"])
    startup.update(
        {
            "phase": "worker_validation",
            "outcome": "failed",
            "error": {
                "type": "WorkerPreflightError",
                "message": "package version mismatch",
            },
        }
    )
    evidence["attempt_counts"] = []
    observed = cast(dict[str, object], evidence["observed"])
    probe = cast(list[dict[str, object]], observed["worker_probes"])[0]
    probe["ok"] = False
    probe["errors"] = ["package cy-exm expected test-version; got other-version"]
    versions = cast(list[list[str]], probe["package_versions"])
    versions[1][1] = "other-version"

    encoded = slurm_module._canonical_execution_evidence(
        evidence,
        expected_tasks=(task,),
        task_retries=0,
    )

    assert json.loads(encoded) == evidence


@pytest.mark.parametrize(
    "reported_addresses",
    [
        ("tcp://node-b:1234", "tcp://node-a:1234"),
        ("tcp://node-a:1234", "tcp://node-a:1234"),
    ],
    ids=("swapped", "duplicate"),
)
def test_slurm_evidence_schema_preserves_failed_probe_identity_diagnostics(
    reported_addresses: tuple[str, str],
    tmp_path: Path,
) -> None:
    modules = _FakeSlurmModules()
    scheduler_addresses = ("tcp://node-a:1234", "tcp://node-b:1234")
    modules.connected_workers = scheduler_addresses
    modules.probe_results = {
        scheduler_addresses[1]: _probe(reported_addresses[1]),
        scheduler_addresses[0]: _probe(reported_addresses[0]),
    }
    tasks = (_task(tmp_path, 2), _task(tmp_path, 3))
    executor = _executor(tmp_path, modules, workers=2)

    with pytest.raises(WorkerPreflightError, match="reported worker address"):
        executor.execute(_reference_task(tmp_path), tasks)

    evidence = cast(dict[str, object], executor.execution_evidence)
    observed = cast(dict[str, object], evidence["observed"])
    probes = cast(list[dict[str, object]], observed["worker_probes"])
    assert tuple(probe["worker_address"] for probe in probes) == reported_addresses
    assert evidence["attempt_counts"] == []

    encoded = slurm_module._canonical_execution_evidence(
        evidence,
        expected_tasks=(_reference_task(tmp_path), *tasks),
        task_retries=0,
    )

    assert json.loads(encoded) == evidence


@pytest.mark.parametrize(
    "case",
    [
        "root_missing",
        "root_extra",
        "root_key_alias",
        "executor",
        "requested_missing",
        "requested_extra",
        "requested_bool",
        "requested_bound",
        "requested_retry_mismatch",
        "requested_memory_profile_mismatch",
        "requested_memory_semantic_mismatch",
        "requested_memory_scheduler_mismatch",
        "requested_memory_bytes_mismatch",
        "requested_memory_dask_mismatch",
        "rendered_stale",
        "rendered_missing",
        "rendered_alias",
        "observed_extra",
        "scheduler_without_workers",
        "workers_unsorted",
        "workers_duplicate",
        "version_key_alias",
        "version_missing",
        "probe_extra",
        "probe_address",
        "probe_version_mismatch",
        "probe_not_ok_after_complete",
        "cancellation_not_attempted_success",
        "cancellation_attempted_no_force",
        "startup_phase",
        "startup_outcome",
        "startup_success_error",
        "startup_scale_attempted",
        "startup_scale_succeeded",
        "early_observed_facts",
        "interrupted_without_cancellation",
        "attempt_missing",
        "attempt_bool",
        "attempt_duplicate",
        "attempt_dataset",
        "attempt_round",
        "attempt_retry_overflow",
        "attempt_empty",
        "attempt_zero_without_cancel",
        "timestamps_missing",
        "timestamps_non_utc",
        "timestamps_reverse",
        "cleanup_extra",
        "cleanup_order",
        "cleanup_attempted",
        "cleanup_nullable",
        "cleanup_success_error",
        "cleanup_failure_without_error",
    ],
)
def test_slurm_evidence_schema_fails_closed_on_malformed_or_inconsistent_values(
    case: str,
    tmp_path: Path,
) -> None:
    modules = _FakeSlurmModules()
    modules.connected_workers = modules.connected_workers[:1]
    executor = _executor(tmp_path, modules)
    task = _task(tmp_path, 2)
    executor.execute(_reference_task(tmp_path), (task,))
    evidence = deepcopy(cast(dict[str, object], executor.execution_evidence))
    _invalid_execution_evidence(evidence, case)

    with pytest.raises((TypeError, ValueError)):
        slurm_module._canonical_execution_evidence(
            evidence,
            expected_tasks=(task,),
            task_retries=0,
        )


@pytest.mark.parametrize("case", ["missing", "reordered"])
def test_slurm_evidence_schema_requires_every_submitted_task_in_canonical_order(
    case: str,
    tmp_path: Path,
) -> None:
    modules = _FakeSlurmModules()
    modules.connected_workers = modules.connected_workers[:1]
    tasks = (_task(tmp_path, 2), _task(tmp_path, 3))
    executor = _executor(tmp_path, modules, workers=1)
    executor.execute(_reference_task(tmp_path), tasks)
    evidence = deepcopy(cast(dict[str, object], executor.execution_evidence))
    attempts = cast(list[dict[str, object]], evidence["attempt_counts"])
    if case == "missing":
        attempts.pop()
    else:
        attempts.reverse()

    with pytest.raises(ValueError):
        slurm_module._canonical_execution_evidence(
            evidence,
            expected_tasks=tasks,
            task_retries=0,
        )


def test_slurm_evidence_schema_accepts_canonical_submitted_subset_on_cancel(
    tmp_path: Path,
) -> None:
    modules = _FakeSlurmModules()
    modules.connected_workers = modules.connected_workers[:1]
    tasks = (_task(tmp_path, 2), _task(tmp_path, 3))
    executor = _executor(tmp_path, modules, workers=1)
    executor.execute(_reference_task(tmp_path), tasks)
    evidence = deepcopy(cast(dict[str, object], executor.execution_evidence))
    observed = cast(dict[str, object], evidence["observed"])
    observed["cancellation"] = {
        "attempted": True,
        "unfinished_futures": 1,
        "force": True,
        "succeeded": True,
    }
    attempts = cast(list[dict[str, object]], evidence["attempt_counts"])
    attempts.pop()

    encoded = slurm_module._canonical_execution_evidence(
        evidence,
        expected_tasks=(_reference_task(tmp_path), *tasks),
        task_retries=0,
    )

    assert json.loads(encoded)["attempt_counts"] == attempts


@pytest.mark.parametrize(
    ("attempts_to_keep", "unfinished_futures", "interrupt_startup"),
    [(0, 1, False), (1, 2, False), (0, 1, True)],
    ids=(
        "no-accepted-task",
        "more-futures-than-tasks",
        "interrupted-preflight-without-accepted-task",
    ),
)
def test_slurm_evidence_schema_rejects_impossible_cancellation_cardinality(
    attempts_to_keep: int,
    unfinished_futures: int,
    interrupt_startup: bool,
    tmp_path: Path,
) -> None:
    modules = _FakeSlurmModules()
    modules.connected_workers = modules.connected_workers[:1]
    task = _task(tmp_path, 2)
    executor = _executor(tmp_path, modules, workers=1)
    executor.execute(_reference_task(tmp_path), (task,))
    evidence = deepcopy(cast(dict[str, object], executor.execution_evidence))
    observed = cast(dict[str, object], evidence["observed"])
    observed["cancellation"] = {
        "attempted": True,
        "unfinished_futures": unfinished_futures,
        "force": True,
        "succeeded": True,
    }
    attempts = cast(list[dict[str, object]], evidence["attempt_counts"])
    del attempts[attempts_to_keep:]
    if interrupt_startup:
        startup = cast(dict[str, object], evidence["startup"])
        startup.update(
            {
                "phase": "worker_probe",
                "outcome": "interrupted",
                "error": {
                    "type": "KeyboardInterrupt",
                    "message": "interrupted preflight",
                },
            }
        )

    with pytest.raises(ValueError):
        slurm_module._canonical_execution_evidence(
            evidence,
            expected_tasks=(task,),
            task_retries=0,
        )


def test_slurm_executor_preflight_uses_exact_versions_threads_and_roots(
    tmp_path: Path,
) -> None:
    modules = _FakeSlurmModules()
    modules.connected_workers = modules.connected_workers[:1]

    _executor(tmp_path, modules).execute(
        _reference_task(tmp_path), (_task(tmp_path, 2),)
    )

    assert len(modules.run_calls) == 1
    function, kwargs = modules.run_calls[0]
    assert function.__name__ == "probe_worker"
    assert kwargs == {
        "dataset_root": str(tmp_path / "dataset"),
        "output_root": str(tmp_path / "output"),
        "worker_local_directory": str(tmp_path / "scratch"),
        "expected_versions": dict.fromkeys(_PACKAGES, "test-version"),
        "expected_threads": {
            NATIVE_THREAD_ENV_VARS[0]: "2",
            **dict.fromkeys(NATIVE_THREAD_ENV_VARS[1:], "3"),
        },
    }
    assert modules.events.index("client.run") < modules.events.index("submit:2")


def test_slurm_executor_preserves_coordinator_version_failure_as_startup_cause(
    tmp_path: Path,
) -> None:
    modules = _FakeSlurmModules()
    modules.connected_workers = modules.connected_workers[:1]

    def version(name: str) -> str:
        if name == "numpy":
            raise RuntimeError("injected coordinator metadata failure")
        return "test-version"

    executor = SlurmExecutor(
        _profile(tmp_path),
        _modules=lambda: modules,
        _version=version,
    )

    with pytest.raises(
        ExecutorStartupError,
        match=r"coordinator package 'numpy'",
    ) as caught:
        executor.execute(_reference_task(tmp_path), (_task(tmp_path, 2),))

    assert isinstance(caught.value.__cause__, RuntimeError)
    assert modules.submit_calls == []
    assert modules.events[-2:] == ["client.close", "cluster.close"]


@pytest.mark.parametrize(
    "failure_point",
    ["cluster", "job_script", "scale", "client", "wait", "run"],
)
def test_slurm_executor_wraps_startup_failures_and_cleans_partial_ownership(
    tmp_path: Path,
    failure_point: str,
) -> None:
    modules = _FakeSlurmModules()
    modules.connected_workers = modules.connected_workers[:1]
    setattr(modules, f"{failure_point}_error", RuntimeError(f"{failure_point} failed"))

    with pytest.raises(ExecutorStartupError) as captured:
        _executor(tmp_path, modules).execute(
            _reference_task(tmp_path), (_task(tmp_path, 2),)
        )

    assert isinstance(captured.value.__cause__, RuntimeError)
    expected = {
        "cluster": [],
        "job_script": ["cluster.close"],
        "scale": ["cluster.close"],
        "client": ["cluster.close"],
        "wait": ["client.close", "cluster.close"],
        "run": ["client.run", "client.close", "cluster.close"],
    }
    assert modules.events == expected[failure_point]
    assert not modules.submit_calls


@pytest.mark.parametrize(
    ("failure_point", "phase", "scale_attempted", "scale_succeeded"),
    [
        ("cluster", "cluster_construction", False, None),
        ("job_script", "template_rendering", False, None),
        ("scale", "scaling", True, False),
        ("client", "client_creation", True, True),
        ("wait", "worker_wait", True, True),
        ("run", "worker_probe", True, True),
    ],
)
def test_slurm_executor_records_exact_failed_startup_phase_and_scale_outcome(
    tmp_path: Path,
    failure_point: str,
    phase: str,
    scale_attempted: bool,
    scale_succeeded: bool | None,
) -> None:
    modules = _FakeSlurmModules()
    modules.connected_workers = modules.connected_workers[:1]
    setattr(modules, f"{failure_point}_error", RuntimeError(f"{failure_point} failed"))
    executor = _executor(tmp_path, modules)

    with pytest.raises(ExecutorStartupError):
        executor.execute(_reference_task(tmp_path), (_task(tmp_path, 2),))

    startup = cast(dict[str, object], executor.execution_evidence["startup"])
    assert startup == {
        "phase": phase,
        "outcome": "failed",
        "scale_attempted": scale_attempted,
        "scale_succeeded": scale_succeeded,
        "error": {"type": "RuntimeError", "message": f"{failure_point} failed"},
    }
    assert executor.execution_evidence["attempt_counts"] == []


def test_slurm_executor_does_not_count_a_task_rejected_during_submission(
    tmp_path: Path,
) -> None:
    modules = _FakeSlurmModules()
    modules.connected_workers = modules.connected_workers[:1]
    modules.submit_error = RuntimeError("scheduler rejected submission")
    executor = _executor(tmp_path, modules)

    with pytest.raises(RuntimeError, match="scheduler rejected submission"):
        executor.execute(_reference_task(tmp_path), (_task(tmp_path, 2),))

    assert executor.execution_evidence["attempt_counts"] == []


@pytest.mark.parametrize(
    ("failed_round", "expected_attempts", "expected_cancellation"),
    [
        (
            2,
            [{"dataset_id": "dataset-1", "round": 1, "attempts": 1}],
            {
                "attempted": False,
                "unfinished_futures": 0,
                "succeeded": None,
            },
        ),
        (
            3,
            [
                {"dataset_id": "dataset-1", "round": 1, "attempts": 1},
                {"dataset_id": "dataset-1", "round": 2, "attempts": 1},
            ],
            {
                "attempted": True,
                "unfinished_futures": 1,
                "force": True,
                "succeeded": True,
            },
        ),
    ],
    ids=("before-first-accept", "after-one-accept"),
)
def test_slurm_executor_cancellation_counts_only_scheduler_accepted_tasks(
    failed_round: int,
    expected_attempts: list[dict[str, object]],
    expected_cancellation: dict[str, object],
    tmp_path: Path,
) -> None:
    modules = _FakeSlurmModules()
    modules.connected_workers = modules.connected_workers[:2]
    modules.pending_rounds = {2}
    modules.submit_errors[failed_round] = KeyboardInterrupt(
        "injected submission interruption"
    )
    tasks = (_task(tmp_path, 2), _task(tmp_path, 3))
    executor = _executor(tmp_path, modules, workers=2)

    with pytest.raises(KeyboardInterrupt, match="submission interruption"):
        executor.execute(_reference_task(tmp_path), tasks)

    evidence = cast(dict[str, object], executor.execution_evidence)
    assert evidence["attempt_counts"] == expected_attempts
    observed = cast(dict[str, object], evidence["observed"])
    assert observed["cancellation"] == expected_cancellation
    encoded = slurm_module._canonical_execution_evidence(
        evidence,
        expected_tasks=(_reference_task(tmp_path), *tasks),
        task_retries=0,
    )
    assert json.loads(encoded) == evidence


def test_slurm_executor_records_preflight_interrupt_without_fake_cancellation(
    tmp_path: Path,
) -> None:
    modules = _FakeSlurmModules()
    modules.connected_workers = modules.connected_workers[:1]
    modules.run_error = KeyboardInterrupt("injected preflight interruption")
    executor = _executor(tmp_path, modules)

    with pytest.raises(KeyboardInterrupt, match="preflight interruption"):
        executor.execute(_reference_task(tmp_path), (_task(tmp_path, 2),))

    assert executor.execution_evidence["startup"] == {
        "phase": "worker_probe",
        "outcome": "interrupted",
        "scale_attempted": True,
        "scale_succeeded": True,
        "error": {
            "type": "KeyboardInterrupt",
            "message": "injected preflight interruption",
        },
    }
    observed = cast(dict[str, object], executor.execution_evidence["observed"])
    assert observed["cancellation"] == {
        "attempted": False,
        "unfinished_futures": 0,
        "succeeded": None,
    }
    assert modules.cancel_call is None
    assert modules.submit_calls == []


def test_slurm_executor_aborts_before_submit_on_reported_probe_failure(
    tmp_path: Path,
) -> None:
    modules = _FakeSlurmModules()
    address = modules.connected_workers[0]
    modules.connected_workers = (address,)
    modules.probe_results = {address: _probe(address, ok=False)}

    executor = _executor(tmp_path, modules)
    with pytest.raises(WorkerPreflightError, match="injected preflight failure"):
        executor.execute(_reference_task(tmp_path), (_task(tmp_path, 2),))

    assert modules.submit_calls == []
    assert modules.events[-2:] == ["client.close", "cluster.close"]
    startup = cast(dict[str, object], executor.execution_evidence["startup"])
    assert startup["phase"] == "worker_validation"
    assert startup["outcome"] == "failed"
    assert startup["scale_attempted"] is True
    assert startup["scale_succeeded"] is True
    error = cast(dict[str, str], startup["error"])
    assert error["type"] == "WorkerPreflightError"
    assert "injected preflight failure" in error["message"]


@pytest.mark.parametrize(
    ("connected", "returned", "message"),
    [
        (("tcp://node-a:1",), {}, "returned worker addresses"),
        (
            ("tcp://node-a:1",),
            {"tcp://node-b:1": _probe("tcp://node-b:1")},
            "returned worker addresses",
        ),
        (("tcp://node-a:1", "tcp://node-b:1"), None, "expected 1 connected"),
    ],
)
def test_slurm_executor_requires_exact_requested_connected_and_probed_worker_sets(
    tmp_path: Path,
    connected: tuple[str, ...],
    returned: Mapping[str, Mapping[str, object]] | None,
    message: str,
) -> None:
    modules = _FakeSlurmModules()
    modules.connected_workers = connected
    modules.probe_results = returned

    with pytest.raises(WorkerPreflightError, match=message):
        _executor(tmp_path, modules, workers=1).execute(
            _reference_task(tmp_path), (_task(tmp_path, 2),)
        )

    assert modules.submit_calls == []


def test_slurm_executor_detects_worker_disappearance_after_probe(
    tmp_path: Path,
) -> None:
    modules = _FakeSlurmModules()
    modules.connected_workers = modules.connected_workers[:2]
    modules.disappear_after_probe = True

    with pytest.raises(WorkerPreflightError, match="worker set changed"):
        _executor(tmp_path, modules, workers=2).execute(
            _reference_task(tmp_path), (_task(tmp_path, 2), _task(tmp_path, 3))
        )

    assert modules.submit_calls == []


def test_slurm_executor_rejects_probe_identity_mismatch_for_scheduler_address(
    tmp_path: Path,
) -> None:
    modules = _FakeSlurmModules()
    address = modules.connected_workers[0]
    modules.connected_workers = (address,)
    modules.probe_results = {address: _probe("tcp://different-node:1234")}

    with pytest.raises(WorkerPreflightError, match="reported worker address"):
        _executor(tmp_path, modules).execute(
            _reference_task(tmp_path), (_task(tmp_path, 2),)
        )

    assert modules.submit_calls == []


def test_slurm_executor_submits_zero_retry_one_slot_unique_keys_and_returns_ordered(
    tmp_path: Path,
) -> None:
    modules = _FakeSlurmModules()
    modules.connected_workers = modules.connected_workers[:2]
    modules.order = (3, 2)

    results = _executor(tmp_path, modules).execute(
        _reference_task(tmp_path), (_task(tmp_path, 2), _task(tmp_path, 3))
    )

    assert modules.as_completed_call == (True, False)
    submitted = [
        (call.key, call.pure, call.retries, call.resources)
        for call in modules.submit_calls
    ]
    assert submitted == [
        ("cyexm-dataset-1-reference-round-001", False, 0, {"round_slot": 1}),
        ("cyexm-dataset-1-round-002", False, 0, {"round_slot": 1}),
        ("cyexm-dataset-1-round-003", False, 0, {"round_slot": 1}),
    ]
    assert tuple(result.round for result in results) == (1, 2, 3)


def test_slurm_executor_does_not_submit_moving_rounds_after_reference_failure(
    tmp_path: Path,
) -> None:
    modules = _FakeSlurmModules()
    modules.connected_workers = modules.connected_workers[:2]
    reference = _reference_task(tmp_path)
    modules.outcomes[1] = [_reference_result(reference, status="failed")]
    executor = _executor(tmp_path, modules, workers=2)

    results = executor.execute(
        reference,
        (_task(tmp_path, 2), _task(tmp_path, 3)),
    )

    assert tuple(result.round for result in results) == (1,)
    assert results[0].status == "failed"
    assert [call.task.round for call in modules.submit_calls] == [1]
    assert executor.execution_evidence["attempt_counts"] == [
        {"dataset_id": "dataset-1", "round": 1, "attempts": 1}
    ]


def test_slurm_executor_submission_deserializes_without_worker_importing_hpc(
    tmp_path: Path,
) -> None:
    import os
    import subprocess
    from functools import partial

    from dask._task_spec import Task as DaskTask
    from distributed.protocol import pickle as distributed_pickle
    from hpc.environment import run_environment_smoke, smoke_module_by_value

    modules = _FakeSlurmModules()
    modules.connected_workers = modules.connected_workers[:1]
    runner = partial(
        run_environment_smoke,
        evidence_dir=str(tmp_path / "evidence"),
        dataset_root=str(tmp_path / "dataset"),
        output_root=str(tmp_path / "output"),
        worker_local_directory=str(tmp_path / "scratch"),
        interface="ib0",
    )
    executor = SlurmExecutor(
        _profile(tmp_path),
        _runner=runner,
        _modules=lambda: modules,
        _version=lambda name: "test-version" if name in _PACKAGES else "unexpected",
    )

    with smoke_module_by_value():
        executor.execute(_reference_task(tmp_path), (_task(tmp_path, 2),))
        call = modules.submit_calls[0]
        payload = distributed_pickle.dumps(DaskTask(call.key, call.runner, call.task))
    payload_path = tmp_path / "submission.pickle"
    payload_path.write_bytes(payload)
    source_root = Path(__file__).resolve().parents[2] / "src"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(source_root)

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import pathlib, sys\n"
            "from distributed.protocol import pickle as distributed_pickle\n"
            "submission = distributed_pickle.loads(\n"
            "    pathlib.Path(sys.argv[1]).read_bytes()\n"
            ")\n"
            "assert callable(submission.func)\n"
            "try: submission.func(*submission.args, **submission.kwargs)\n"
            "except ValueError as error: assert 'No worker found' in str(error)\n"
            "else: raise AssertionError('embedded runner unexpectedly succeeded')\n"
            "assert 'hpc' not in sys.modules and 'hpc.environment' not in sys.modules",
            str(payload_path),
        ],
        cwd="/tmp",
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_slurm_executor_resubmits_with_bounded_unique_attempts(
    tmp_path: Path,
) -> None:
    modules = _FakeSlurmModules()
    modules.connected_workers = modules.connected_workers[:1]
    task = _task(tmp_path, 2)
    modules.outcomes[2] = [
        (ValueError, ValueError("first"), None),
        _result(task),
    ]
    executor = _executor(tmp_path, modules, retries=1)

    result = executor.execute(_reference_task(tmp_path), (task,))[1]

    assert result.status == "completed"
    assert [call.key for call in modules.submit_calls] == [
        "cyexm-dataset-1-reference-round-001",
        "cyexm-dataset-1-round-002",
        "cyexm-dataset-1-round-002-attempt-002",
    ]
    assert all(call.retries == 0 for call in modules.submit_calls)
    assert executor.execution_evidence["attempt_counts"] == [
        {"dataset_id": "dataset-1", "round": 1, "attempts": 1},
        {"dataset_id": "dataset-1", "round": 2, "attempts": 2},
    ]


def test_slurm_executor_preserves_independent_result_after_exhausted_failure(
    tmp_path: Path,
) -> None:
    modules = _FakeSlurmModules()
    modules.connected_workers = modules.connected_workers[:2]
    failed_task = _task(tmp_path, 2)
    modules.outcomes[2] = [(RuntimeError, RuntimeError("exhausted"), None)]

    reference, failed, completed = _executor(tmp_path, modules).execute(
        _reference_task(tmp_path), (failed_task, _task(tmp_path, 3))
    )

    assert reference.status == "completed"
    assert failed.status == "failed"
    assert failed.failure == RoundFailure("round", "RuntimeError", "exhausted")
    assert completed.status == "completed"
    assert modules.cancel_call is None


def test_slurm_executor_validates_tasks_and_results_like_local_dask(
    tmp_path: Path,
) -> None:
    loaded = False

    def modules_factory() -> _FakeSlurmModules:
        nonlocal loaded
        loaded = True
        return _FakeSlurmModules()

    malformed_task = replace(_task(tmp_path, 2), round=True)
    with pytest.raises(RoundExecutionError, match=r"tasks\[1\].*round"):
        SlurmExecutor(_profile(tmp_path), _modules=modules_factory).execute(
            _reference_task(tmp_path), (malformed_task,)
        )
    assert loaded is False

    modules = _FakeSlurmModules()
    modules.connected_workers = modules.connected_workers[:1]
    task = _task(tmp_path, 2)
    modules.outcomes[2] = [replace(_result(task), dataset_id="wrong")]
    result = _executor(tmp_path, modules).execute(_reference_task(tmp_path), (task,))[1]
    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.exception_type == "RoundExecutionError"


def test_slurm_executor_rejects_duplicate_task_identity_before_cluster_mutation(
    tmp_path: Path,
) -> None:
    loaded = False

    def modules_factory() -> _FakeSlurmModules:
        nonlocal loaded
        loaded = True
        return _FakeSlurmModules()

    task = _task(tmp_path, 2)
    with pytest.raises(RoundExecutionError, match="duplicate task identity"):
        SlurmExecutor(_profile(tmp_path), _modules=modules_factory).execute(
            _reference_task(tmp_path), (task, task)
        )
    assert loaded is False


def test_slurm_executor_cancels_only_actual_unfinished_on_keyboard_interrupt(
    tmp_path: Path,
) -> None:
    modules = _FakeSlurmModules()
    modules.connected_workers = modules.connected_workers[:3]
    modules.order = (2, 3, 4)
    modules.interrupt_after = 2
    modules.pending_rounds = {4}
    executor = _executor(tmp_path, modules)

    with pytest.raises(KeyboardInterrupt, match="collection interruption"):
        executor.execute(
            _reference_task(tmp_path),
            tuple(_task(tmp_path, number) for number in (2, 3, 4)),
        )

    assert modules.cancel_call is not None
    cancelled, force = modules.cancel_call
    assert {future.task.round for future in cancelled} == {4}
    assert force is True
    observed = cast(dict[str, object], executor.execution_evidence["observed"])
    assert observed["cancellation"] == {
        "attempted": True,
        "unfinished_futures": 1,
        "force": True,
        "succeeded": True,
    }
    assert modules.events[-2:] == ["client.close", "cluster.close"]


def test_slurm_executor_preserves_primary_baseexception_when_cleanup_fails(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    modules = _FakeSlurmModules()
    modules.connected_workers = modules.connected_workers[:1]
    modules.interrupt_after = 0
    modules.pending_rounds = {1}
    modules.cancel_error = SystemExit("cancel failed")
    modules.client_close_error = KeyboardInterrupt("client close failed")
    modules.cluster_close_error = SystemExit("cluster close failed")
    modules.config_exit_error = RuntimeError("configuration cleanup failed")
    executor = _executor(tmp_path, modules)

    with (
        caplog.at_level(logging.WARNING, logger="cyexm.execution.slurm"),
        pytest.raises(KeyboardInterrupt, match="collection interruption"),
    ):
        executor.execute(_reference_task(tmp_path), (_task(tmp_path, 2),))

    assert "cancel failed" in caplog.text
    assert "client close failed" in caplog.text
    assert "cluster close failed" in caplog.text
    assert "configuration cleanup failed" in caplog.text
    cleanup = cast(dict[str, object], executor.execution_evidence["cleanup"])
    assert cast(dict[str, object], cleanup["client"])["succeeded"] is False
    assert cast(dict[str, object], cleanup["cluster"])["succeeded"] is False
    assert cast(dict[str, object], cleanup["configuration"])["succeeded"] is False
    assert modules.events[-2:] == ["client.close", "cluster.close"]


def test_slurm_executor_does_not_invoke_sequential_fallback_on_startup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modules = _FakeSlurmModules()
    modules.cluster_error = RuntimeError("submission unavailable")
    fallback_called = False

    def fallback(*_args: object, **_kwargs: object) -> None:
        nonlocal fallback_called
        fallback_called = True

    monkeypatch.setattr(
        "cyexm.execution.sequential.SequentialExecutor.execute",
        fallback,
    )

    with pytest.raises(ExecutorStartupError):
        _executor(tmp_path, modules).execute(
            _reference_task(tmp_path), (_task(tmp_path, 2),)
        )

    assert fallback_called is False


def test_slurm_executor_exports_from_execution_namespace() -> None:
    import cyexm.execution

    assert cyexm.execution.SlurmExecutor is SlurmExecutor
    assert "SlurmExecutor" in cyexm.execution.__all__


def test_slurm_executor_rejects_non_slurm_or_missing_slurm_profiles(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    non_slurm = replace(profile, executor="sequential", slurm=None)
    missing_spec = replace(profile, slurm=None)

    with pytest.raises(ExecutorStartupError, match="requires a slurm profile"):
        SlurmExecutor(non_slurm).execute(
            _reference_task(tmp_path), (_task(tmp_path, 2),)
        )
    with pytest.raises(ExecutorStartupError, match="requires SlurmSpec"):
        SlurmExecutor(missing_spec).execute(
            _reference_task(tmp_path), (_task(tmp_path, 2),)
        )
