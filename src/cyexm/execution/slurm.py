"""Owned, import-light SLURMCluster execution for coarse round tasks."""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.metadata import version as package_version
from typing import Any, cast

from cyexm.config.models import ExecutionProfile, SlurmSpec
from cyexm.errors import ExecutorStartupError, WorkerPreflightError
from cyexm.execution._dask_common import (
    collect_distributed_results,
    future_is_unfinished,
    utc_now,
    validate_distributed_tasks,
)
from cyexm.execution.base import RoundTaskRunner
from cyexm.execution.preflight import (
    WorkerProbe,
    probe_worker,
    validate_worker_probes,
)
from cyexm.runtime import NATIVE_THREAD_ENV_VARS
from cyexm.workflow.models import RoundResult, RoundTask
from cyexm.workflow.round_task import run_round_task

_LOGGER = logging.getLogger(__name__)
_COORDINATOR_PACKAGES = (
    "cy-exm",
    "dask",
    "distributed",
    "dask-jobqueue",
    "antspyx",
    "numpy",
    "scipy",
    "tifffile",
)
_WORKER_EXTRA_ARGS = ["--resources", "round_slot=1"]


@dataclass(frozen=True, slots=True)
class _SlurmModules:
    SLURMCluster: Any
    Client: Any
    as_completed: Any
    parse_timedelta: Any


def _slurm_modules() -> _SlurmModules:
    """Load Dask and dask-jobqueue only when SLURM execution is selected."""
    from dask.utils import parse_timedelta
    from dask_jobqueue import SLURMCluster  # pyright: ignore[reportPrivateImportUsage]
    from distributed import Client, as_completed

    return _SlurmModules(SLURMCluster, Client, as_completed, parse_timedelta)


def _error_payload(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _scheduler_snapshot(
    client: Any,
    *,
    expected_workers: int | None,
) -> tuple[str, tuple[str, ...]]:
    try:
        info = client.scheduler_info()
    except Exception as error:
        raise WorkerPreflightError(
            "Could not inspect connected SLURM workers: "
            f"{type(error).__name__}: {error}"
        ) from error
    if not isinstance(info, Mapping):
        raise WorkerPreflightError(
            "Dask scheduler_info must be a mapping with address and workers"
        )
    scheduler_address = info.get("address")
    if type(scheduler_address) is not str or not scheduler_address:
        raise WorkerPreflightError(
            "Dask scheduler_info address must be a nonempty built-in string"
        )
    workers = info.get("workers")
    if not isinstance(workers, Mapping):
        raise WorkerPreflightError(
            "Dask scheduler_info workers must be a mapping keyed by worker address"
        )
    addresses: list[str] = []
    for address in workers:
        if type(address) is not str or not address:
            raise WorkerPreflightError(
                "Dask scheduler worker addresses must be nonempty built-in strings"
            )
        addresses.append(address)
    if len(addresses) != len(set(addresses)):
        raise WorkerPreflightError("Dask scheduler reported duplicate worker addresses")
    addresses.sort()
    if expected_workers is not None and len(addresses) != expected_workers:
        raise WorkerPreflightError(
            f"Worker preflight expected {expected_workers} connected workers; "
            f"scheduler reported {len(addresses)}: {addresses!r}"
        )
    return scheduler_address, tuple(addresses)


def _returned_probe_addresses(probes: object) -> tuple[str, ...]:
    if not isinstance(probes, Mapping):
        raise WorkerPreflightError(
            "Worker preflight responses must be a mapping keyed by worker address"
        )
    addresses: list[str] = []
    for address in probes:
        if type(address) is not str or not address:
            raise WorkerPreflightError(
                "Worker preflight returned worker addresses as nonempty built-in "
                "strings"
            )
        addresses.append(address)
    addresses.sort()
    return tuple(addresses)


def _decodable_probe_evidence(
    probes: Mapping[object, object],
) -> list[dict[str, object]]:
    evidence: list[tuple[str, dict[str, object]]] = []
    for address, payload in probes.items():
        if type(address) is not str or not isinstance(payload, Mapping):
            continue
        try:
            decoded = WorkerProbe.from_dict(cast(Mapping[str, object], payload))
        except (TypeError, ValueError):
            continue
        evidence.append((address, decoded.to_dict()))
    evidence.sort(key=lambda item: item[0])
    return [payload for _address, payload in evidence]


class SlurmExecutor:
    """Run independent rounds through an owned ``SLURMCluster``."""

    def __init__(
        self,
        profile: ExecutionProfile,
        *,
        _runner: RoundTaskRunner = run_round_task,
        _modules: Callable[[], Any] = _slurm_modules,
        _version: Callable[[str], str] = package_version,
    ) -> None:
        self.profile = profile
        self._runner = _runner
        self._modules = _modules
        self._version = _version
        self._execution_evidence: dict[str, object] = {}

    @property
    def execution_evidence(self) -> Mapping[str, object]:
        """Return a detached JSON-compatible copy of observed execution evidence."""
        return copy.deepcopy(self._execution_evidence)

    def _spec(self) -> SlurmSpec:
        if self.profile.executor != "slurm":
            raise ExecutorStartupError("SlurmExecutor requires a slurm profile")
        if not isinstance(self.profile.slurm, SlurmSpec):
            raise ExecutorStartupError("SlurmExecutor requires SlurmSpec settings")
        if self.profile.paths.worker_local_directory is None:
            raise ExecutorStartupError(
                "SlurmExecutor requires a worker_local_directory"
            )
        return self.profile.slurm

    def _coordinator_versions(self) -> dict[str, str]:
        versions: dict[str, str] = {}
        for name in _COORDINATOR_PACKAGES:
            try:
                value = self._version(name)
            except Exception as error:
                raise ExecutorStartupError(
                    f"Could not inspect coordinator package {name!r}: "
                    f"{type(error).__name__}: {error}"
                ) from error
            if type(value) is not str or not value:
                raise ExecutorStartupError(
                    f"Coordinator package {name!r} returned an invalid version: "
                    f"{value!r}"
                )
            versions[name] = value
        return versions

    def _begin_evidence(
        self,
        *,
        worker_count: int,
        spec: SlurmSpec | None,
    ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        requested: dict[str, object] = {"workers": worker_count}
        if spec is not None:
            requested.update(
                {
                    "configured_workers": spec.workers,
                    "startup_timeout": spec.startup_timeout,
                    "task_retries": spec.task_retries,
                }
            )
        observed: dict[str, object] = {}
        cleanup: dict[str, object] = {}
        self._execution_evidence = {
            "executor": "slurm",
            "requested": requested,
            "observed": observed,
            "attempt_counts": [],
            "timestamps": {"started_at": utc_now()},
            "cleanup": cleanup,
        }
        return observed, cleanup, cast(dict[str, object], self._execution_evidence)

    @staticmethod
    def _attempt_evidence(
        tasks: tuple[RoundTask, ...],
        attempts: list[int],
    ) -> list[dict[str, object]]:
        return [
            {
                "dataset_id": task.dataset_id,
                "round": task.round,
                "attempts": attempts[index],
            }
            for index, task in enumerate(tasks)
        ]

    @staticmethod
    def _close_owned(
        client: Any | None,
        cluster: Any | None,
        cleanup: dict[str, object],
    ) -> None:
        order: list[str] = []
        for name, resource in (("client", client), ("cluster", cluster)):
            if resource is None:
                continue
            order.append(name)
            status: dict[str, object] = {"attempted": True}
            try:
                resource.close()
            except BaseException as error:
                status["succeeded"] = False
                status["error"] = _error_payload(error)
                _LOGGER.warning("Failed to close owned SLURM %s: %s", name, error)
            else:
                status["succeeded"] = True
            cleanup[name] = status
        if order:
            cleanup["order"] = order

    def execute(self, tasks: tuple[RoundTask, ...]) -> tuple[RoundResult, ...]:
        if not tasks:
            _observed, _cleanup, evidence = self._begin_evidence(
                worker_count=0,
                spec=None,
            )
            cast(dict[str, str], evidence["timestamps"])["ended_at"] = utc_now()
            return ()

        spec = self._spec()
        validate_distributed_tasks(tasks, executor_name="SlurmExecutor")
        worker_count = min(len(tasks), spec.workers)
        observed, cleanup, evidence = self._begin_evidence(
            worker_count=worker_count,
            spec=spec,
        )
        attempts = [0 for _task in tasks]
        cluster: Any | None = None
        client: Any | None = None
        active: set[Any] = set()

        try:
            try:
                modules = self._modules()
                opened_cluster = modules.SLURMCluster(
                    queue=spec.queue,
                    cores=spec.cores_per_worker,
                    processes=spec.processes_per_worker,
                    memory=spec.memory,
                    walltime=spec.walltime,
                    interface=spec.interface,
                    scheduler_options={"interface": spec.scheduler_interface},
                    local_directory=str(self.profile.paths.worker_local_directory),
                    job_name=spec.job_name,
                    job_extra_directives=list(spec.job_extra_directives),
                    worker_extra_args=list(_WORKER_EXTRA_ARGS),
                )
                cluster = opened_cluster
                script = opened_cluster.job_script()
                if type(script) is not str or not script:
                    raise TypeError(
                        "SLURMCluster.job_script() must return a nonempty built-in "
                        "string"
                    )
                observed["submitted_job_script"] = script
                opened_cluster.scale(jobs=worker_count)
                opened_client = modules.Client(
                    opened_cluster,
                    timeout=spec.startup_timeout,
                )
                client = opened_client
                opened_client.wait_for_workers(
                    worker_count,
                    timeout=modules.parse_timedelta(spec.startup_timeout),
                )

                versions = self._coordinator_versions()
                observed["coordinator_versions"] = dict(versions)
                scheduler_address, connected = _scheduler_snapshot(
                    opened_client,
                    expected_workers=worker_count,
                )
                observed["scheduler_address"] = scheduler_address
                observed["connected_workers"] = list(connected)
                raw_probes = opened_client.run(
                    probe_worker,
                    dataset_root=str(self.profile.paths.dataset_root),
                    output_root=str(self.profile.paths.output_root),
                    worker_local_directory=str(
                        self.profile.paths.worker_local_directory
                    ),
                    expected_versions=dict(versions),
                    expected_threads={
                        name: str(
                            self.profile.native_threads.itk
                            if index == 0
                            else self.profile.native_threads.blas
                        )
                        for index, name in enumerate(NATIVE_THREAD_ENV_VARS)
                    },
                )
                if isinstance(raw_probes, Mapping):
                    observed["worker_probes"] = _decodable_probe_evidence(raw_probes)
                returned = _returned_probe_addresses(raw_probes)
                if returned != connected:
                    raise WorkerPreflightError(
                        "Worker preflight returned worker addresses that do not "
                        "match the expected connected workers: "
                        f"expected={connected!r}, "
                        f"returned={returned!r}"
                    )
                probes = validate_worker_probes(
                    cast(Mapping[str, Mapping[str, object]], raw_probes)
                )
                mismatches = [
                    f"scheduler address {address!r} reported worker address "
                    f"{probe.worker_address!r}"
                    for address, probe in zip(connected, probes, strict=True)
                    if probe.worker_address != address
                ]
                if mismatches:
                    raise WorkerPreflightError(
                        "Worker preflight reported worker address mismatches:\n- "
                        + "\n- ".join(mismatches)
                    )
                scheduler_after, connected_after = _scheduler_snapshot(
                    opened_client,
                    expected_workers=None,
                )
                if scheduler_after != scheduler_address or connected_after != connected:
                    raise WorkerPreflightError(
                        "SLURM worker set changed during preflight: "
                        f"before=({scheduler_address!r}, {connected!r}), "
                        f"after=({scheduler_after!r}, {connected_after!r})"
                    )
            except ExecutorStartupError:
                raise
            except Exception as error:
                raise ExecutorStartupError(
                    f"Failed to start and preflight {worker_count} SLURM workers "
                    f"within {spec.startup_timeout}: {error}"
                ) from error

            execution_client = client
            assert execution_client is not None
            results, _attempts = collect_distributed_results(
                tasks,
                runner=self._runner,
                client=execution_client,
                as_completed=modules.as_completed,
                task_retries=spec.task_retries,
                active=active,
                attempts=attempts,
            )
            return results
        except KeyboardInterrupt:
            if client is not None:
                unfinished = tuple(
                    future
                    for future in active
                    if future_is_unfinished(future, logger=_LOGGER)
                )
                cancellation: dict[str, object] = {
                    "unfinished_futures": len(unfinished),
                    "force": True,
                }
                try:
                    if unfinished:
                        client.cancel(unfinished, force=True)
                except BaseException as error:
                    cancellation["succeeded"] = False
                    cancellation["error"] = _error_payload(error)
                    _LOGGER.warning(
                        "Failed to cancel unfinished SLURM futures: %s", error
                    )
                else:
                    cancellation["succeeded"] = True
                observed["cancellation"] = cancellation
            raise
        finally:
            evidence["attempt_counts"] = self._attempt_evidence(tasks, attempts)
            self._close_owned(client, cluster, cleanup)
            cast(dict[str, str], evidence["timestamps"])["ended_at"] = utc_now()


__all__ = ["SlurmExecutor"]
