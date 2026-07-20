from __future__ import annotations

import json
import logging
import os
import re
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest
from tiff_bytes import write_cyexm_uint16_tiff

import cyexm.workflow.api as workflow_api
from cyexm.config.models import (
    ChannelSpec,
    DatasetManifest,
    ExecutionPaths,
    ExecutionProfile,
    LocalDaskSpec,
    NativeThreadSpec,
    ReferenceSpec,
    RegistrationSpec,
    RegistrationStageSpec,
    RoundSpec,
    SlurmSpec,
)
from cyexm.errors import (
    ExecutorStartupError,
    RoundExecutionError,
    WorkerPreflightError,
    WorkflowExecutionError,
    WorkflowPlanError,
)
from cyexm.execution.base import WorkflowExecutor
from cyexm.execution.sequential import SequentialExecutor
from cyexm.registration._request_policy import (
    common_parameters,
    fixed_reference_parameters,
)
from cyexm.registration.artifacts import (
    ArtifactStore,
    CompletionRecord,
    FileIdentity,
    StageRequest,
)
from cyexm.runtime import NATIVE_THREAD_ENV_VARS
from cyexm.spatial import ImageGeometry
from cyexm.workflow import run_workflow
from cyexm.workflow.layout import WorkflowLayout
from cyexm.workflow.models import (
    ChannelTask,
    RoundFailure,
    RoundResult,
    RoundTask,
    WorkflowPlan,
    canonical_json,
    sha256_fingerprint,
)

pytestmark = pytest.mark.unit


class _StringAlias(str):
    pass


class _IntegerAlias(int):
    pass


class _TupleAlias(tuple[object, ...]):
    pass


class _RoundFailureAlias(RoundFailure):
    pass


def _registration() -> RegistrationSpec:
    return RegistrationSpec(
        crop=False,
        linear=RegistrationStageSpec(type="TRSAA", accuracy="high"),
        nonlinear=RegistrationStageSpec(type="SyNOnly", accuracy="high"),
    )


def _task(output_root: Path, round_number: int) -> RoundTask:
    round_layout = WorkflowLayout.for_dataset(output_root, "dataset-1").round(
        round_number
    )
    return RoundTask(
        dataset_id="dataset-1",
        round=round_number,
        strategy="whole_image",
        fixed_image=output_root.parent / "dataset" / "fixed.tif",
        moving_image=output_root.parent / "dataset" / f"moving-{round_number}.tif",
        registration_directory=round_layout.registration,
        log_path=round_layout.log,
        voxel_spacing_xyz=(1.0, 1.0, 1.0),
        registration=_registration(),
        native_threads=NativeThreadSpec(itk=1, blas=1),
        channels=(
            ChannelTask(
                source=output_root.parent / "dataset" / f"moving-{round_number}.tif",
                output=round_layout.channels / f"moving-{round_number}.tif",
                interpolation="linear",
            ),
        ),
        chunk_size_zyx=None,
        overlap_fraction=None,
    )


def _artifacts(task: RoundTask) -> tuple[Path, ...]:
    return (
        task.registration_directory / f"linear_transform_{task.round}.mat",
        task.registration_directory / f"nonlinear_warp_{task.round}.nii.gz",
        task.registration_directory / f"nonlinear_registered_{task.round}.tif",
        *(channel.output for channel in task.channels),
    )


def _plan(tmp_path: Path, rounds: tuple[int, ...] = (2, 3)) -> WorkflowPlan:
    output_root = tmp_path / "output"
    tasks = tuple(_task(output_root, round_number) for round_number in rounds)
    sources = (tasks[0].fixed_image, *(task.moving_image for task in tasks))
    for source in sources:
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"source:{source.name}".encode())
    manifest = DatasetManifest(
        schema_version=1,
        dataset_id="dataset-1",
        voxel_spacing_um=(1.0, 1.0, 1.0),
        reference=ReferenceSpec(round=1, channel=PurePosixPath("fixed.tif")),
        registration=_registration(),
        rounds=tuple(
            RoundSpec(
                round=task.round,
                strategy=task.strategy,
                registration_channel=PurePosixPath(f"moving-{task.round}.tif"),
                channels=(
                    ChannelSpec(PurePosixPath(f"moving-{task.round}.tif"), "linear"),
                ),
            )
            for task in tasks
        ),
    )
    execution = ExecutionProfile(
        schema_version=1,
        executor="sequential",
        paths=ExecutionPaths(
            dataset_root=tmp_path / "dataset",
            output_root=output_root,
            worker_local_directory=None,
        ),
        native_threads=NativeThreadSpec(itk=1, blas=1),
        local_dask=None,
        slurm=None,
        source_path=tmp_path / "execution.yaml",
    )
    layout = WorkflowLayout.for_dataset(output_root, "dataset-1")

    def source_mapping(path: Path) -> dict[str, object]:
        stat = path.stat()
        return {
            "path": str(path.resolve()),
            "shape_zyx": [1, 1, 1],
            "dtype": "uint16",
            "axes": "ZYX",
            "geometry": None,
            "st_size": stat.st_size,
            "st_mtime_ns": stat.st_mtime_ns,
        }

    scientific_json = canonical_json(
        {
            "schema_version": 1,
            "dataset_id": "dataset-1",
            "voxel_spacing_xyz_um": [1.0, 1.0, 1.0],
            "reference": {
                "round": 1,
                "channel": str(tasks[0].fixed_image.resolve()),
            },
            "registration": {
                "crop": False,
                "linear": {"type": "TRSAA", "accuracy": "high"},
                "nonlinear": {"type": "SyNOnly", "accuracy": "high"},
            },
            "sources": [source_mapping(path) for path in sorted(sources)],
            "rounds": [
                {
                    "round": task.round,
                    "strategy": task.strategy,
                    "registration_channel": str(task.moving_image.resolve()),
                    "registration_manifest_channel": f"moving-{task.round}.tif",
                    "chunk_size_zyx": None,
                    "overlap_fraction": None,
                    "channels": [
                        {
                            "file": str(channel.source.resolve()),
                            "manifest_file": channel.source.name,
                            "output": channel.output.relative_to(
                                layout.root
                            ).as_posix(),
                            "interpolation": channel.interpolation,
                        }
                        for channel in task.channels
                    ],
                }
                for task in tasks
            ],
            "policies": {
                "axis_convention": "external-zyx-ants-xyz-v1",
                "output_format": "ome-compatible-tiff-uint16-v1",
                "postprocessing": "clip-uint16-no-normalization-v1",
            },
        }
    )
    return WorkflowPlan(
        schema_version=1,
        dataset_id="dataset-1",
        manifest=manifest,
        execution=execution,
        fixed_image=tasks[0].fixed_image,
        tasks=tasks,
        expected_artifacts=tuple(
            artifact for task in tasks for artifact in _artifacts(task)
        ),
        scientific_json=scientific_json,
        scientific_fingerprint=sha256_fingerprint(scientific_json),
    )


def _serial_plan(tmp_path: Path) -> WorkflowPlan:
    plan = _plan(tmp_path, rounds=(2,))
    task = replace(
        plan.tasks[0],
        strategy="serial_chunked",
        chunk_size_zyx=(1, 2, 2),
        overlap_fraction=0.0,
    )
    manifest_round = replace(
        plan.manifest.rounds[0],
        strategy="serial_chunked",
        chunk_size_zyx=(1, 2, 2),
        overlap_fraction=0.0,
    )
    manifest = replace(plan.manifest, rounds=(manifest_round,))
    scientific = json.loads(plan.scientific_json)
    scientific_round = scientific["rounds"][0]
    scientific_round["strategy"] = "serial_chunked"
    scientific_round["chunk_size_zyx"] = [1, 2, 2]
    scientific_round["overlap_fraction"] = 0.0
    scientific_json = canonical_json(scientific)
    return replace(
        plan,
        manifest=manifest,
        tasks=(task,),
        scientific_json=scientific_json,
        scientific_fingerprint=sha256_fingerprint(scientific_json),
    )


def _invalid_task(
    tmp_path: Path,
    case: str,
) -> tuple[WorkflowPlan, RoundTask]:
    plan = (
        _serial_plan(tmp_path)
        if case.startswith("chunk_") or case.startswith("overlap_")
        else _plan(tmp_path)
    )
    task = plan.tasks[0]
    channel = task.channels[0]
    if case == "dataset_id":
        return plan, replace(task, dataset_id=_StringAlias(task.dataset_id))
    if case == "round_bool":
        return plan, replace(task, round=True)  # type: ignore[arg-type]
    if case == "round_alias":
        return plan, replace(task, round=_IntegerAlias(task.round))
    if case == "strategy":
        return plan, replace(task, strategy=_StringAlias(task.strategy))
    if case == "fixed_image":
        return plan, replace(task, fixed_image=str(task.fixed_image))  # type: ignore[arg-type]
    if case == "moving_image":
        return plan, replace(task, moving_image=str(task.moving_image))  # type: ignore[arg-type]
    if case == "registration_directory":
        return plan, replace(  # type: ignore[arg-type]
            task,
            registration_directory=str(task.registration_directory),
        )
    if case == "log_path":
        return plan, replace(task, log_path=str(task.log_path))  # type: ignore[arg-type]
    if case == "spacing_container":
        return plan, replace(  # type: ignore[arg-type]
            task,
            voxel_spacing_xyz=list(task.voxel_spacing_xyz),
        )
    if case == "spacing_scalar":
        return plan, replace(task, voxel_spacing_xyz=(1, 1.0, 1.0))
    if case == "registration_model":
        registration = task.registration
        return plan, replace(
            task,
            registration=SimpleNamespace(
                crop=registration.crop,
                linear=registration.linear,
                nonlinear=registration.nonlinear,
            ),  # type: ignore[arg-type]
        )
    if case == "registration_scalar":
        registration = task.registration
        return plan, replace(
            task,
            registration=replace(
                registration,
                linear=replace(
                    registration.linear,
                    type=_StringAlias(registration.linear.type),
                ),
            ),
        )
    if case == "native_model":
        return plan, replace(  # type: ignore[arg-type]
            task,
            native_threads=SimpleNamespace(itk=1, blas=1),
        )
    if case == "native_itk":
        return plan, replace(
            task,
            native_threads=NativeThreadSpec(itk=True, blas=1),  # type: ignore[arg-type]
        )
    if case == "native_blas":
        return plan, replace(
            task,
            native_threads=NativeThreadSpec(itk=1, blas=True),  # type: ignore[arg-type]
        )
    if case == "channels_container":
        return plan, replace(task, channels=list(task.channels))  # type: ignore[arg-type]
    if case == "channel_model":
        return plan, replace(
            task,
            channels=(
                SimpleNamespace(
                    source=channel.source,
                    output=channel.output,
                    interpolation=channel.interpolation,
                ),
            ),  # type: ignore[arg-type]
        )
    if case == "channel_source":
        return plan, replace(
            task,
            channels=(replace(channel, source=str(channel.source)),),  # type: ignore[arg-type]
        )
    if case == "channel_output":
        return plan, replace(
            task,
            channels=(replace(channel, output=str(channel.output)),),  # type: ignore[arg-type]
        )
    if case == "channel_interpolation":
        return plan, replace(
            task,
            channels=(
                replace(
                    channel,
                    interpolation=_StringAlias(channel.interpolation),
                ),
            ),
        )
    if case == "chunk_container":
        return plan, replace(task, chunk_size_zyx=[1, 2, 2])  # type: ignore[arg-type]
    if case == "chunk_scalar":
        return plan, replace(task, chunk_size_zyx=(True, 2, 2))  # type: ignore[arg-type]
    if case == "overlap_int":
        return plan, replace(task, overlap_fraction=0)  # type: ignore[arg-type]
    if case == "overlap_bool":
        return plan, replace(task, overlap_fraction=False)  # type: ignore[arg-type]
    if case == "force":
        return plan, replace(task, force=0)  # type: ignore[arg-type]
    raise AssertionError(f"unknown invalid task case: {case}")


def _local_execution(plan: WorkflowPlan, tmp_path: Path) -> ExecutionProfile:
    return replace(
        plan.execution,
        executor="local_dask",
        paths=replace(
            plan.execution.paths,
            worker_local_directory=tmp_path / "scratch",
        ),
        local_dask=LocalDaskSpec(
            workers=2,
            threads_per_worker=1,
            processes=True,
            memory_limit="2GB",
            startup_timeout="30s",
            task_retries=0,
        ),
    )


def _slurm_execution(plan: WorkflowPlan, tmp_path: Path) -> ExecutionProfile:
    return replace(
        plan.execution,
        executor="slurm",
        paths=replace(
            plan.execution.paths,
            worker_local_directory=tmp_path / "scratch",
        ),
        slurm=SlurmSpec(
            queue="512GB",
            workers=2,
            cores_per_worker=72,
            processes_per_worker=1,
            memory="500GB",
            walltime="72:00:00",
            interface="ib0",
            scheduler_interface="ib0",
            startup_timeout="15m",
            task_retries=0,
            job_name="cyexm-registration",
            job_extra_directives=(),
        ),
    )


def _slurm_evidence(*, outcome: str = "succeeded") -> dict[str, object]:
    return {
        "executor": "slurm",
        "requested": {
            "workers": 1,
            "configured_workers": 2,
            "startup_timeout": "15m",
            "task_retries": 0,
        },
        "rendered": {
            "rendered_job_script_template": (
                "#!/usr/bin/env bash\n#SBATCH --partition=512GB\n"
            )
        },
        "observed": {
            "coordinator_versions": {
                "cy-exm": "0.1.0",
                "dask": "2025.1.0",
                "distributed": "2025.1.0",
                "dask-jobqueue": "0.9.0",
                "antspyx": "0.5.4",
                "numpy": "2.0.2",
                "scipy": "1.16.0",
                "tifffile": "2026.7.17",
            },
            "scheduler_address": "tcp://scheduler:8786",
            "connected_workers": ["tcp://worker:1234"],
            "worker_probes": [
                {
                    "ok": True,
                    "worker_address": "tcp://worker:1234",
                    "hostname": "worker",
                    "pid": 1234,
                    "slurm_job_id": "98765",
                    "package_versions": [
                        ["antspyx", "0.5.4"],
                        ["cy-exm", "0.1.0"],
                        ["dask", "2025.1.0"],
                        ["dask-jobqueue", "0.9.0"],
                        ["distributed", "2025.1.0"],
                        ["numpy", "2.0.2"],
                        ["scipy", "1.16.0"],
                        ["tifffile", "2026.7.17"],
                    ],
                    "native_threads": [
                        ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1"],
                        ["OMP_NUM_THREADS", "1"],
                    ],
                    "dataset_visible": True,
                    "output_writable": True,
                    "scratch_usable": True,
                    "memory_limit_bytes": 500_000_000_000,
                    "network_host": "worker",
                    "errors": [],
                }
            ],
        },
        "startup": {
            "phase": "complete",
            "outcome": outcome,
            "scale_attempted": True,
            "scale_succeeded": True,
        },
        "attempt_counts": [{"dataset_id": "dataset-1", "round": 2, "attempts": 1}],
        "timestamps": {
            "started_at": "2026-07-20T10:00:00Z",
            "ended_at": "2026-07-20T10:00:01Z",
        },
        "cleanup": {
            "client": {"attempted": True, "succeeded": True},
            "cluster": {"attempted": True, "succeeded": True},
            "configuration": {"attempted": True, "succeeded": True},
            "order": ["client", "cluster", "configuration"],
        },
    }


def _invalid_execution(
    plan: WorkflowPlan,
    tmp_path: Path,
    case: str,
) -> ExecutionProfile:
    if case == "schema":
        return replace(plan.execution, schema_version=True)  # type: ignore[arg-type]
    if case == "executor":
        return replace(plan.execution, executor="threads")  # type: ignore[arg-type]
    if case == "source_relative":
        return replace(plan.execution, source_path=Path("execution.yaml"))
    if case == "source_noncanonical":
        return replace(
            plan.execution,
            source_path=tmp_path / "profiles" / ".." / "execution.yaml",
        )
    if case == "path_overlap":
        return replace(
            plan.execution,
            paths=replace(
                plan.execution.paths,
                output_root=plan.execution.paths.dataset_root / "output",
            ),
        )
    if case == "native_itk":
        return replace(
            plan.execution,
            native_threads=NativeThreadSpec(itk=True, blas=1),  # type: ignore[arg-type]
        )
    if case == "native_blas":
        return replace(
            plan.execution,
            native_threads=NativeThreadSpec(itk=1, blas=0),
        )
    if case == "worker_missing":
        profile = _local_execution(plan, tmp_path)
        return replace(
            profile,
            paths=replace(profile.paths, worker_local_directory=None),
        )
    if case == "scratch_archive":
        profile = _local_execution(plan, tmp_path)
        return replace(
            profile,
            paths=replace(
                profile.paths,
                worker_local_directory=Path("/archive/cyexm/scratch"),
            ),
        )
    if case.startswith("local_"):
        profile = _local_execution(plan, tmp_path)
        assert profile.local_dask is not None
        field, value = {
            "local_workers": ("workers", True),
            "local_threads": ("threads_per_worker", 0),
            "local_processes": ("processes", 1),
            "local_processes_false": ("processes", False),
            "local_memory": ("memory_limit", ""),
            "local_timeout": ("startup_timeout", "bad\ntimeout"),
            "local_retries": ("task_retries", True),
            "local_negative_retries": ("task_retries", -1),
        }[case]
        return replace(
            profile,
            local_dask=replace(profile.local_dask, **{field: value}),
        )
    profile = _slurm_execution(plan, tmp_path)
    assert profile.slurm is not None
    if case == "section_exclusivity":
        return replace(
            profile,
            local_dask=_local_execution(plan, tmp_path).local_dask,
        )
    if case == "slurm_directive":
        return replace(
            profile,
            slurm=replace(
                profile.slurm,
                job_extra_directives=("--partition=debug",),
            ),
        )
    if case == "slurm_directives_list":
        return replace(
            profile,
            slurm=replace(
                profile.slurm,
                job_extra_directives=[],  # type: ignore[arg-type]
            ),
        )
    if case == "slurm_directive_newline":
        return replace(
            profile,
            slurm=replace(
                profile.slurm,
                job_extra_directives=("--account=lab\n--exclusive",),
            ),
        )
    field, value = {
        "slurm_workers": ("workers", True),
        "slurm_cores": ("cores_per_worker", 0),
        "slurm_processes": ("processes_per_worker", 2),
        "slurm_queue": ("queue", ""),
        "slurm_memory": ("memory", "bad\nvalue"),
        "slurm_walltime": ("walltime", ""),
        "slurm_interface": ("interface", ""),
        "slurm_scheduler": ("scheduler_interface", ""),
        "slurm_timeout": ("startup_timeout", ""),
        "slurm_retries": ("task_retries", True),
        "slurm_negative_retries": ("task_retries", -1),
        "slurm_job_name": ("job_name", ""),
    }[case]
    return replace(profile, slurm=replace(profile.slurm, **{field: value}))


def _result(task: RoundTask, *, failed: bool = False) -> RoundResult:
    return RoundResult(
        dataset_id=task.dataset_id,
        round=task.round,
        status="failed" if failed else "completed",
        started_at="2026-07-20T10:00:00Z",
        ended_at="2026-07-20T10:00:01Z",
        duration_seconds=1.0,
        hostname="worker-01",
        pid=123,
        dask_task_key=None,
        slurm_job_id=None,
        strategy=task.strategy,
        native_threads=(("itk", 1), ("blas", 1)),
        artifact_paths=tuple(str(path) for path in _artifacts(task)),
        reused_artifacts=(),
        log_path=str(task.log_path),
        failure=(RoundFailure("round", "RuntimeError", "injected") if failed else None),
    )


def _invalid_result(task: RoundTask, case: str) -> RoundResult:
    result = _result(task, failed=case.startswith("failure_"))
    if case == "dataset_id":
        return replace(result, dataset_id=_StringAlias(result.dataset_id))
    if case == "round":
        return replace(result, round=_IntegerAlias(result.round))
    if case == "status":
        return replace(result, status=_StringAlias(result.status))
    if case == "started_at":
        return replace(result, started_at=_StringAlias(result.started_at))
    if case == "ended_at":
        return replace(result, ended_at=_StringAlias(result.ended_at))
    if case == "duration":
        return replace(result, duration_seconds=1)  # type: ignore[arg-type]
    if case == "hostname":
        return replace(result, hostname=_StringAlias(result.hostname))
    if case == "pid":
        return replace(result, pid=_IntegerAlias(result.pid))
    if case == "dask_task_key":
        return replace(result, dask_task_key=_StringAlias("task-key"))
    if case == "slurm_job_id":
        return replace(result, slurm_job_id=_StringAlias("123"))
    if case == "strategy":
        return replace(result, strategy=_StringAlias(result.strategy))
    if case == "native_container":
        return replace(  # type: ignore[arg-type]
            result,
            native_threads=_TupleAlias(result.native_threads),
        )
    if case == "native_entry":
        return replace(
            result,
            native_threads=(
                _TupleAlias(result.native_threads[0]),  # type: ignore[arg-type]
                result.native_threads[1],
            ),
        )
    if case == "native_name":
        return replace(
            result,
            native_threads=(
                (_StringAlias("itk"), 1),
                ("blas", 1),
            ),
        )
    if case == "native_value":
        return replace(
            result,
            native_threads=(("itk", True), ("blas", 1)),  # type: ignore[arg-type]
        )
    if case == "native_zero":
        return replace(result, native_threads=(("itk", 0), ("blas", 1)))
    if case == "native_name_unknown":
        return replace(result, native_threads=(("ITK", 1), ("blas", 1)))
    if case == "artifact_container":
        return replace(  # type: ignore[arg-type]
            result,
            artifact_paths=_TupleAlias(result.artifact_paths),
        )
    if case == "artifact_element":
        return replace(
            result,
            artifact_paths=(
                _StringAlias(result.artifact_paths[0]),
                *result.artifact_paths[1:],
            ),
        )
    if case == "reused_container":
        return replace(  # type: ignore[arg-type]
            result,
            reused_artifacts=_TupleAlias(result.reused_artifacts),
        )
    if case == "reused_element":
        return replace(
            result,
            reused_artifacts=(_StringAlias(result.artifact_paths[0]),),
        )
    if case == "log_path":
        return replace(result, log_path=_StringAlias(result.log_path))
    assert result.failure is not None
    if case == "failure_model":
        return replace(
            result,
            failure=_RoundFailureAlias(
                result.failure.stage,
                result.failure.exception_type,
                result.failure.message,
            ),
        )
    if case == "failure_stage":
        return replace(
            result,
            failure=replace(result.failure, stage=_StringAlias(result.failure.stage)),
        )
    if case == "failure_exception_type":
        return replace(
            result,
            failure=replace(
                result.failure,
                exception_type=_StringAlias(result.failure.exception_type),
            ),
        )
    if case == "failure_message":
        return replace(
            result,
            failure=replace(
                result.failure,
                message=_StringAlias(result.failure.message),
            ),
        )
    raise AssertionError(f"unknown invalid result case: {case}")


class _FakeExecutor:
    def __init__(
        self,
        callback: Callable[[tuple[RoundTask, ...]], tuple[RoundResult, ...]],
        *,
        execution_evidence: object | None = None,
    ) -> None:
        self.callback = callback
        self.calls: list[tuple[RoundTask, ...]] = []
        if execution_evidence is not None:
            self.execution_evidence = execution_evidence

    def execute(self, tasks: tuple[RoundTask, ...]) -> tuple[RoundResult, ...]:
        self.calls.append(tasks)
        return self.callback(tasks)


def _factory(
    executor: WorkflowExecutor,
) -> Callable[[ExecutionProfile], WorkflowExecutor]:
    return lambda _profile: executor


def _commit_artifact(path: Path, plan: WorkflowPlan) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"artifact")
    geometry = ImageGeometry(
        shape=(1, 1, 1),
        origin=(0.0, 0.0, 0.0),
        spacing=(1.0, 1.0, 1.0),
        direction=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    task = next(task for task in plan.tasks if path in _artifacts(task))
    affine, warp, registered, *_ = _artifacts(task)
    inputs = (
        FileIdentity.from_path(task.fixed_image),
        FileIdentity.from_path(task.moving_image),
    )
    common = {
        "axis_convention_version": "external-zyx-ants-xyz-v1",
        "crop": False,
        "imaging_round": task.round,
        "pixel_size_xyz": [1.0, 1.0, 1.0],
    }
    threads = {name: 1 for name in NATIVE_THREAD_ENV_VARS}
    if path == affine:
        fixed_reference = task.registration_directory / "fixed_reference.tif"
        fixed_reference.parent.mkdir(parents=True, exist_ok=True)
        write_cyexm_uint16_tiff(fixed_reference, geometry)
        fixed_request = StageRequest.for_identities(
            stage="fixed-reference",
            inputs=inputs,
            geometry=geometry,
            method="uint16-tiff",
            accuracy="exact",
            native_threads=threads,
            upstream=None,
            parameters={
                **common_parameters(
                    crop=False,
                    imaging_round=task.round,
                    pixel_size_xyz=task.voxel_spacing_xyz,
                ),
                **fixed_reference_parameters(0.0, 1.0),
            },
        )
        fixed_record = CompletionRecord(
            schema_version=1,
            request=fixed_request,
            artifact=FileIdentity.from_path(fixed_reference),
        )
        ArtifactStore().record_path(fixed_reference).write_text(
            fixed_record.to_json(), encoding="utf-8"
        )
        request = StageRequest.for_identities(
            stage="affine",
            inputs=inputs,
            geometry=geometry,
            method="TRSAA",
            accuracy="high",
            native_threads=threads,
            upstream=None,
            parameters={
                **common,
                "image_interpolation": "linear",
                "mask_interpolation": "nearestNeighbor",
                "moving_geometry": geometry.to_dict(),
                "registration_parameters": {
                    "aff_metric": "mattes",
                    "aff_random_sampling_rate": 0.5,
                    "aff_sampling": 32,
                    "mask_all_stages": True,
                    "verbose": True,
                    "reg_iterations": [1000, 500, 250, 100],
                },
            },
        )
    elif path == warp:
        request = StageRequest.for_identities(
            stage="nonlinear",
            inputs=inputs,
            geometry=geometry,
            method="SyNOnly",
            accuracy="high",
            native_threads=threads,
            upstream=FileIdentity.from_path(affine),
            parameters={
                **common,
                "image_interpolation": "linear",
                "mask_interpolation": "nearestNeighbor",
                "moving_geometry": geometry.to_dict(),
                "registration_mode": "whole-image",
                "registration_parameters": {
                    "aff_random_sampling_rate": 1.0,
                    "flow_sigma": 3.0,
                    "initial_transform": "Identity",
                    "mask_all_stages": True,
                    "reg_iterations": [100, 70, 50],
                    "singleprecision": False,
                    "syn_metric": "mattes",
                    "syn_sampling": 32,
                    "total_sigma": 0.5,
                    "verbose": True,
                },
                "scalar_postprocessing_policy": "whole-global-histogram-match-v1",
            },
        )
    elif path == registered:
        request = StageRequest.for_identities(
            stage="registered-image",
            inputs=inputs,
            geometry=geometry,
            method="SyNOnly",
            accuracy="high",
            native_threads=threads,
            upstream=FileIdentity.from_path(warp),
            parameters={
                **common,
                "format": "uint16-tiff",
                "minimum": 0.0,
                "maximum": 1.0,
                "scalar_postprocessing_policy": "whole-global-histogram-match-v1",
            },
        )
    else:
        channel = next(channel for channel in task.channels if channel.output == path)
        request = StageRequest.for_identities(
            stage="channel-transform-v1",
            inputs=(
                FileIdentity.from_path(task.fixed_image),
                FileIdentity.from_path(channel.source),
                FileIdentity.from_path(affine),
                FileIdentity.from_path(warp),
            ),
            geometry=geometry,
            method="warp-affine",
            accuracy="not-applicable",
            native_threads=threads,
            upstream=FileIdentity.from_path(warp),
            parameters={
                "pixel_size_xyz_um": [1.0, 1.0, 1.0],
                "interpolation": channel.interpolation,
                "transform_order": ["warp", "affine"],
                "whichtoinvert": [False, False],
                "axis_convention": "external-zyx-ants-xyz-v1",
                "output_format": "ome-compatible-tiff-uint16-v1",
                "postprocessing": "clip-uint16-no-normalization-v1",
            },
        )
    record = CompletionRecord(
        schema_version=1,
        request=request,
        artifact=FileIdentity.from_path(path),
    )
    ArtifactStore().record_path(path).write_text(record.to_json(), encoding="utf-8")


def _corrupt_tiff_while_preserving_record_identity(path: Path) -> None:
    record = CompletionRecord.read(ArtifactStore().record_path(path))
    contents = bytearray(path.read_bytes())
    assert len(contents) >= 8
    contents[:4] = b"NOPE"
    path.write_bytes(contents)
    accessed = path.stat().st_atime_ns
    os.utime(path, ns=(accessed, record.artifact.st_mtime_ns))
    assert FileIdentity.from_path(path) == record.artifact


def test_run_workflow_writes_scientific_snapshot_before_execution(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    layout = WorkflowLayout.for_dataset(
        plan.execution.paths.output_root, plan.dataset_id
    )

    def execute(tasks: tuple[RoundTask, ...]) -> tuple[RoundResult, ...]:
        assert layout.plan_path.read_text(encoding="utf-8").strip() == (
            plan.scientific_json
        )
        return tuple(_result(task) for task in tasks)

    result = run_workflow(
        plan,
        _executor_factory=_factory(_FakeExecutor(execute)),
    )

    assert result.succeeded
    assert Path(result.run_record).is_file()
    assert (
        json.loads(Path(result.run_record).read_text(encoding="utf-8"))["status"]
        == "completed"
    )
    assert not tuple(layout.root.rglob(".*.tmp*"))


def test_run_workflow_lazily_uses_the_production_round_worker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from cyexm.workflow import round_task as round_task_module

    plan = _plan(tmp_path, rounds=(2,))
    calls: list[int] = []

    def run_round(task: RoundTask) -> RoundResult:
        calls.append(task.round)
        return _result(task)

    monkeypatch.setattr(round_task_module, "run_round_task", run_round)

    result = run_workflow(plan)

    assert calls == [2]
    assert tuple(item.status for item in result.rounds) == ("completed",)


def test_run_workflow_selects_local_dask_without_manufacturing_observations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from cyexm.execution import dask as dask_execution

    sequential_plan = _plan(tmp_path)
    plan = replace(
        sequential_plan,
        execution=_local_execution(sequential_plan, tmp_path),
    )
    selected: list[tuple[str, tuple[int, ...]]] = []

    def execute(
        executor: dask_execution.LocalDaskExecutor,
        tasks: tuple[RoundTask, ...],
    ) -> tuple[RoundResult, ...]:
        selected.append(
            (executor.profile.executor, tuple(task.round for task in tasks))
        )
        return tuple(_result(task) for task in tasks)

    monkeypatch.setattr(dask_execution.LocalDaskExecutor, "execute", execute)

    result = run_workflow(plan)

    assert selected == [("local_dask", (2, 3))]
    assert result.succeeded
    assert result.execution_evidence_json == "{}"
    run_record = json.loads(Path(result.run_record).read_text(encoding="utf-8"))
    assert run_record["execution"] == {}
    assert run_record["execution_profile"]["executor"] == "local_dask"
    assert run_record["execution_profile"]["local_dask"] == {
        "memory_limit": "2GB",
        "processes": True,
        "startup_timeout": "30s",
        "task_retries": 0,
        "threads_per_worker": 1,
        "workers": 2,
    }
    assert run_record["scientific_fingerprint"] == plan.scientific_fingerprint
    assert plan.scientific_json == sequential_plan.scientific_json
    assert plan.scientific_fingerprint == sequential_plan.scientific_fingerprint
    assert "local_dask" not in plan.scientific_json


def test_local_dask_failed_round_does_not_manufacture_cluster_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from cyexm.execution import dask as dask_execution

    sequential_plan = _plan(tmp_path, rounds=(2,))
    plan = replace(
        sequential_plan,
        execution=_local_execution(sequential_plan, tmp_path),
    )
    monkeypatch.setattr(
        dask_execution.LocalDaskExecutor,
        "execute",
        lambda _self, tasks: tuple(_result(task, failed=True) for task in tasks),
    )

    with pytest.raises(WorkflowExecutionError) as captured:
        run_workflow(plan)

    run_record = json.loads(Path(captured.value.run_record).read_text(encoding="utf-8"))
    assert run_record["status"] == "failed"
    assert run_record["execution"] == {}
    assert run_record["execution_profile"]["executor"] == "local_dask"


def test_local_dask_cancellation_does_not_manufacture_cluster_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from cyexm.execution import dask as dask_execution

    sequential_plan = _plan(tmp_path, rounds=(2,))
    plan = replace(
        sequential_plan,
        execution=_local_execution(sequential_plan, tmp_path),
    )

    def interrupt(
        _executor: dask_execution.LocalDaskExecutor,
        _tasks: tuple[RoundTask, ...],
    ) -> tuple[RoundResult, ...]:
        raise KeyboardInterrupt

    monkeypatch.setattr(dask_execution.LocalDaskExecutor, "execute", interrupt)

    with pytest.raises(KeyboardInterrupt):
        run_workflow(plan)

    records = tuple(
        WorkflowLayout.for_dataset(
            plan.execution.paths.output_root,
            plan.dataset_id,
        ).workflow_runs.glob("*.json")
    )
    assert len(records) == 1
    run_record = json.loads(records[0].read_text(encoding="utf-8"))
    assert run_record["status"] == "cancelled"
    assert run_record["execution"] == {}
    assert run_record["execution_profile"]["executor"] == "local_dask"


def test_all_reused_local_workflow_does_not_open_an_executor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sequential_plan = _plan(tmp_path, rounds=(2,))
    plan = replace(
        sequential_plan,
        execution=_local_execution(sequential_plan, tmp_path),
    )
    for artifact in _artifacts(plan.tasks[0]):
        _commit_artifact(artifact, plan)
    factory_calls: list[str] = []
    executor_calls: list[tuple[int, ...]] = []
    scheduler_modules_before = {
        name
        for name in sys.modules
        if name == "distributed" or name.startswith("distributed.")
    }

    class UnexpectedExecutor:
        def execute(self, tasks: tuple[RoundTask, ...]) -> tuple[RoundResult, ...]:
            executor_calls.append(tuple(task.round for task in tasks))
            return tuple(_result(task) for task in tasks)

    def tracking_factory(profile: ExecutionProfile) -> WorkflowExecutor:
        factory_calls.append(profile.executor)
        return UnexpectedExecutor()

    monkeypatch.setattr(workflow_api, "_default_executor_factory", tracking_factory)

    result = run_workflow(plan)

    assert factory_calls == []
    assert executor_calls == []
    assert {
        name
        for name in sys.modules
        if name == "distributed" or name.startswith("distributed.")
    } == scheduler_modules_before
    assert tuple(round_result.status for round_result in result.rounds) == ("reused",)
    assert result.execution_evidence_json == "{}"
    run_record = json.loads(Path(result.run_record).read_text(encoding="utf-8"))
    assert run_record["execution"] == {}
    assert run_record["execution_profile"]["executor"] == "local_dask"


def test_local_dask_constructor_failure_never_falls_back_to_sequential(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from cyexm.execution import dask as dask_execution

    sequential_plan = _plan(tmp_path, rounds=(2,))
    plan = replace(
        sequential_plan,
        execution=_local_execution(sequential_plan, tmp_path),
    )
    sequential_calls: list[tuple[int, ...]] = []

    class FailingLocalDaskExecutor:
        def __init__(self, profile: ExecutionProfile) -> None:
            del profile
            raise ExecutorStartupError("injected local Dask constructor failure")

    monkeypatch.setattr(
        dask_execution,
        "LocalDaskExecutor",
        FailingLocalDaskExecutor,
    )
    monkeypatch.setattr(
        SequentialExecutor,
        "execute",
        lambda _self, tasks: (
            sequential_calls.append(tuple(task.round for task in tasks))
            or tuple(_result(task) for task in tasks)
        ),
    )

    with pytest.raises(
        ExecutorStartupError,
        match="injected local Dask constructor failure",
    ):
        run_workflow(plan)

    layout = WorkflowLayout.for_dataset(
        plan.execution.paths.output_root,
        plan.dataset_id,
    )
    assert sequential_calls == []
    assert layout.plan_path.read_text(encoding="utf-8") == plan.scientific_json + "\n"
    assert not layout.workflow_runs.exists()


def test_run_workflow_lazily_selects_slurm_without_falling_back_to_sequential(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from cyexm.execution import slurm as slurm_execution

    sequential_plan = _plan(tmp_path, rounds=(2,))
    plan = replace(
        sequential_plan,
        execution=_slurm_execution(sequential_plan, tmp_path),
    )
    evidence = _slurm_evidence()
    slurm_calls: list[tuple[str, tuple[int, ...]]] = []
    sequential_calls: list[tuple[int, ...]] = []

    class FakeSlurmExecutor:
        def __init__(self, profile: ExecutionProfile) -> None:
            self.profile = profile

        @property
        def execution_evidence(self) -> object:
            return evidence

        def execute(self, tasks: tuple[RoundTask, ...]) -> tuple[RoundResult, ...]:
            slurm_calls.append(
                (self.profile.executor, tuple(task.round for task in tasks))
            )
            return tuple(_result(task) for task in tasks)

    monkeypatch.setattr(slurm_execution, "SlurmExecutor", FakeSlurmExecutor)
    monkeypatch.setattr(
        SequentialExecutor,
        "execute",
        lambda _self, tasks: (
            sequential_calls.append(tuple(task.round for task in tasks))
            or tuple(_result(task) for task in tasks)
        ),
    )

    result = run_workflow(plan)

    assert slurm_calls == [("slurm", (2,))]
    assert sequential_calls == []
    assert json.loads(result.execution_evidence_json) == evidence


def test_slurm_execution_evidence_is_operational_detached_and_canonical(
    tmp_path: Path,
) -> None:
    sequential_plan = _plan(tmp_path, rounds=(2,))
    plan = replace(
        sequential_plan,
        execution=_slurm_execution(sequential_plan, tmp_path),
    )
    evidence = _slurm_evidence()
    fake = _FakeExecutor(
        lambda tasks: tuple(_result(task) for task in tasks),
        execution_evidence=evidence,
    )

    result = run_workflow(plan, _executor_factory=_factory(fake))

    expected = canonical_json(evidence)
    assert result.execution_evidence_json == expected
    run_record = json.loads(Path(result.run_record).read_text(encoding="utf-8"))
    assert run_record["execution"] == evidence
    assert run_record["scientific_fingerprint"] == plan.scientific_fingerprint
    assert "rendered_job_script_template" not in plan.scientific_json
    assert "scheduler_address" not in plan.scientific_json

    rendered = evidence["rendered"]
    assert isinstance(rendered, dict)
    rendered["rendered_job_script_template"] = "mutated"
    assert result.execution_evidence_json == expected
    assert (
        json.loads(Path(result.run_record).read_text(encoding="utf-8"))["execution"]
        != evidence
    )


def test_slurm_failed_round_persists_post_preflight_execution_evidence(
    tmp_path: Path,
) -> None:
    sequential_plan = _plan(tmp_path, rounds=(2,))
    plan = replace(
        sequential_plan,
        execution=_slurm_execution(sequential_plan, tmp_path),
    )
    evidence = _slurm_evidence()
    fake = _FakeExecutor(
        lambda tasks: tuple(_result(task, failed=True) for task in tasks),
        execution_evidence=evidence,
    )

    with pytest.raises(WorkflowExecutionError) as captured:
        run_workflow(plan, _executor_factory=_factory(fake))

    run_record = json.loads(Path(captured.value.run_record).read_text(encoding="utf-8"))
    assert run_record["status"] == "failed"
    assert run_record["execution"] == evidence
    assert run_record["execution"]["startup"]["outcome"] == "succeeded"


@pytest.mark.parametrize(
    ("error_type", "evidence_phase", "failure_phase"),
    [
        (ExecutorStartupError, "worker_wait", "startup"),
        (WorkerPreflightError, "worker_validation", "preflight"),
    ],
    ids=("startup", "preflight"),
)
def test_slurm_startup_failure_writes_operational_summary_and_re_raises_same_error(
    error_type: type[ExecutorStartupError],
    evidence_phase: str,
    failure_phase: str,
    tmp_path: Path,
) -> None:
    sequential_plan = _plan(tmp_path, rounds=(2, 3))
    plan = replace(
        sequential_plan,
        execution=_slurm_execution(sequential_plan, tmp_path),
    )
    for artifact in _artifacts(plan.tasks[0]):
        _commit_artifact(artifact, plan)
    evidence = _slurm_evidence(outcome="failed")
    startup = evidence["startup"]
    assert isinstance(startup, dict)
    startup.update(
        {
            "phase": evidence_phase,
            "error": {"type": error_type.__name__, "message": "injected failure"},
        }
    )
    if evidence_phase == "worker_wait":
        evidence["observed"] = {}
    evidence["attempt_counts"] = []
    cause = RuntimeError("preserved cause")
    raised = error_type("injected failure")
    raised.__cause__ = cause

    def fail(_tasks: tuple[RoundTask, ...]) -> tuple[RoundResult, ...]:
        raise raised

    fake = _FakeExecutor(fail, execution_evidence=evidence)

    with pytest.raises(error_type, match="injected failure") as captured:
        run_workflow(plan, _executor_factory=_factory(fake))

    assert captured.value is raised
    assert captured.value.__cause__ is cause
    assert captured.value.run_record is not None
    assert captured.value.execution_evidence_json == canonical_json(evidence)
    run_record = json.loads(Path(captured.value.run_record).read_text(encoding="utf-8"))
    assert run_record["status"] == "failed"
    assert run_record["execution"] == evidence
    assert run_record["operational_failure"] == {
        "phase": failure_phase,
        "exception_type": error_type.__name__,
        "message": "injected failure",
    }
    assert [item["round"] for item in run_record["rounds"]] == [2]
    assert run_record["rounds"][0]["status"] == "reused"
    assert run_record["execution"]["attempt_counts"] == []


@pytest.mark.parametrize(
    "message",
    [
        "returned worker addresses differ from the scheduler snapshot",
        "probe reported worker address different from its scheduler key",
    ],
    ids=("returned-set", "probe-identity"),
)
def test_slurm_preflight_address_discrepancy_remains_truthful_operational_evidence(
    message: str,
    tmp_path: Path,
) -> None:
    sequential_plan = _plan(tmp_path, rounds=(2,))
    plan = replace(
        sequential_plan,
        execution=_slurm_execution(sequential_plan, tmp_path),
    )
    evidence = _slurm_evidence(outcome="failed")
    startup = evidence["startup"]
    assert isinstance(startup, dict)
    startup.update(
        {
            "phase": "worker_validation",
            "error": {"type": "WorkerPreflightError", "message": message},
        }
    )
    evidence["attempt_counts"] = []
    observed = evidence["observed"]
    assert isinstance(observed, dict)
    probes = observed["worker_probes"]
    assert isinstance(probes, list)
    probe = probes[0]
    assert isinstance(probe, dict)
    probe["worker_address"] = "tcp://unexpected-worker:1234"
    probe["hostname"] = "unexpected-worker"
    probe["network_host"] = "unexpected-worker"
    raised = WorkerPreflightError(message)

    def fail(_tasks: tuple[RoundTask, ...]) -> tuple[RoundResult, ...]:
        raise raised

    fake = _FakeExecutor(fail, execution_evidence=evidence)

    with pytest.raises(WorkerPreflightError, match=message) as captured:
        run_workflow(plan, _executor_factory=_factory(fake))

    assert captured.value is raised
    assert captured.value.run_record is not None
    assert captured.value.execution_evidence_json == canonical_json(evidence)
    run_record = json.loads(Path(captured.value.run_record).read_text(encoding="utf-8"))
    assert run_record["execution"] == evidence
    assert run_record["operational_failure"]["phase"] == "preflight"


def test_slurm_interruption_persists_cancelled_execution_evidence(
    tmp_path: Path,
) -> None:
    sequential_plan = _plan(tmp_path, rounds=(2,))
    plan = replace(
        sequential_plan,
        execution=_slurm_execution(sequential_plan, tmp_path),
    )
    evidence = _slurm_evidence()
    observed = evidence["observed"]
    assert isinstance(observed, dict)
    observed["cancellation"] = {
        "attempted": True,
        "unfinished_futures": 1,
        "force": True,
        "succeeded": True,
    }

    def interrupt(_tasks: tuple[RoundTask, ...]) -> tuple[RoundResult, ...]:
        raise KeyboardInterrupt("injected SLURM interruption")

    fake = _FakeExecutor(interrupt, execution_evidence=evidence)

    with pytest.raises(KeyboardInterrupt, match="SLURM interruption"):
        run_workflow(plan, _executor_factory=_factory(fake))

    records = tuple(
        WorkflowLayout.for_dataset(
            plan.execution.paths.output_root,
            plan.dataset_id,
        ).workflow_runs.glob("*.json")
    )
    assert len(records) == 1
    run_record = json.loads(records[0].read_text(encoding="utf-8"))
    assert run_record["status"] == "cancelled"
    assert run_record["execution"] == evidence


@pytest.mark.parametrize(
    "round_failed", [False, True], ids=("success", "round-failure")
)
def test_slurm_invalid_evidence_fails_closed_with_a_durable_operational_summary(
    round_failed: bool,
    tmp_path: Path,
) -> None:
    sequential_plan = _plan(tmp_path, rounds=(2,))
    plan = replace(
        sequential_plan,
        execution=_slurm_execution(sequential_plan, tmp_path),
    )
    invalid = _slurm_evidence()
    rendered = invalid["rendered"]
    assert isinstance(rendered, dict)
    rendered["submitted_job_script"] = rendered.pop("rendered_job_script_template")
    fake = _FakeExecutor(
        lambda tasks: tuple(_result(task, failed=round_failed) for task in tasks),
        execution_evidence=invalid,
    )

    with pytest.raises(
        ExecutorStartupError,
        match="invalid SLURM execution evidence",
    ) as captured:
        run_workflow(plan, _executor_factory=_factory(fake))

    assert captured.value.run_record is not None
    assert captured.value.execution_evidence_json is None
    run_record = json.loads(Path(captured.value.run_record).read_text(encoding="utf-8"))
    assert run_record["status"] == "failed"
    assert run_record["execution"] is None
    assert run_record["execution_evidence_error"]["exception_type"] in {
        "TypeError",
        "ValueError",
    }
    assert run_record["operational_failure"]["phase"] == "execution_evidence"
    assert [item["status"] for item in run_record["rounds"]] == [
        "failed" if round_failed else "completed"
    ]


@pytest.mark.parametrize(
    "case",
    ["dataset", "round", "retry-overflow", "profile-retry-mismatch"],
)
def test_slurm_attempt_evidence_is_validated_against_submitted_work_and_profile(
    case: str,
    tmp_path: Path,
) -> None:
    sequential_plan = _plan(tmp_path, rounds=(2,))
    plan = replace(
        sequential_plan,
        execution=_slurm_execution(sequential_plan, tmp_path),
    )
    evidence = _slurm_evidence()
    attempts = evidence["attempt_counts"]
    requested = evidence["requested"]
    assert isinstance(attempts, list)
    assert isinstance(attempts[0], dict)
    assert isinstance(requested, dict)
    if case == "dataset":
        attempts[0]["dataset_id"] = "unrelated-dataset"
    elif case == "round":
        attempts[0]["round"] = 999
    elif case == "retry-overflow":
        attempts[0]["attempts"] = 100
    else:
        requested["task_retries"] = 1
    fake = _FakeExecutor(
        lambda tasks: tuple(_result(task) for task in tasks),
        execution_evidence=evidence,
    )

    with pytest.raises(ExecutorStartupError, match="invalid SLURM execution evidence"):
        run_workflow(plan, _executor_factory=_factory(fake))


def test_slurm_invalid_evidence_during_interruption_preserves_the_primary_cancel(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    sequential_plan = _plan(tmp_path, rounds=(2,))
    plan = replace(
        sequential_plan,
        execution=_slurm_execution(sequential_plan, tmp_path),
    )
    invalid = _slurm_evidence()
    invalid["submitted_job_script"] = "stale"
    interruption = KeyboardInterrupt("primary cancellation")

    def interrupt(_tasks: tuple[RoundTask, ...]) -> tuple[RoundResult, ...]:
        raise interruption

    fake = _FakeExecutor(interrupt, execution_evidence=invalid)

    with (
        caplog.at_level(logging.ERROR, logger="cyexm.workflow.api"),
        pytest.raises(KeyboardInterrupt, match="primary cancellation") as captured,
    ):
        run_workflow(plan, _executor_factory=_factory(fake))

    assert captured.value is interruption
    records = tuple(
        WorkflowLayout.for_dataset(
            plan.execution.paths.output_root,
            plan.dataset_id,
        ).workflow_runs.glob("*.json")
    )
    assert len(records) == 1
    run_record = json.loads(records[0].read_text(encoding="utf-8"))
    assert run_record["status"] == "cancelled"
    assert run_record["execution"] is None
    assert run_record["execution_evidence_error"]["exception_type"] in {
        "TypeError",
        "ValueError",
    }
    assert "invalid SLURM execution evidence during cancellation" in caplog.text


def test_slurm_evidence_property_baseexception_cannot_replace_primary_cancel(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    sequential_plan = _plan(tmp_path, rounds=(2,))
    plan = replace(
        sequential_plan,
        execution=_slurm_execution(sequential_plan, tmp_path),
    )
    interruption = KeyboardInterrupt("primary cancellation")

    class BrokenEvidenceExecutor:
        @property
        def execution_evidence(self) -> object:
            raise SystemExit("injected evidence property defect")

        def execute(self, _tasks: tuple[RoundTask, ...]) -> tuple[RoundResult, ...]:
            raise interruption

    with (
        caplog.at_level(logging.ERROR, logger="cyexm.workflow.api"),
        pytest.raises(KeyboardInterrupt, match="primary cancellation") as captured,
    ):
        run_workflow(
            plan,
            _executor_factory=_factory(BrokenEvidenceExecutor()),
        )

    assert captured.value is interruption
    records = tuple(
        WorkflowLayout.for_dataset(
            plan.execution.paths.output_root,
            plan.dataset_id,
        ).workflow_runs.glob("*.json")
    )
    assert len(records) == 1
    run_record = json.loads(records[0].read_text(encoding="utf-8"))
    assert run_record["status"] == "cancelled"
    assert run_record["execution"] is None
    assert run_record["execution_evidence_error"] == {
        "exception_type": "SystemExit",
        "message": "injected evidence property defect",
    }


def test_slurm_evidence_property_system_exit_is_a_mapped_contract_failure(
    tmp_path: Path,
) -> None:
    sequential_plan = _plan(tmp_path, rounds=(2,))
    plan = replace(
        sequential_plan,
        execution=_slurm_execution(sequential_plan, tmp_path),
    )

    class BrokenEvidenceExecutor:
        @property
        def execution_evidence(self) -> object:
            raise SystemExit("injected evidence property defect")

        def execute(self, tasks: tuple[RoundTask, ...]) -> tuple[RoundResult, ...]:
            return tuple(_result(task) for task in tasks)

    with pytest.raises(
        ExecutorStartupError,
        match="invalid SLURM execution evidence",
    ) as captured:
        run_workflow(
            plan,
            _executor_factory=_factory(BrokenEvidenceExecutor()),
        )

    assert captured.value.run_record is not None
    assert captured.value.execution_evidence_json is None
    run_record = json.loads(Path(captured.value.run_record).read_text(encoding="utf-8"))
    assert run_record["execution"] is None
    assert run_record["execution_evidence_error"] == {
        "exception_type": "SystemExit",
        "message": "injected evidence property defect",
    }


def test_slurm_invalid_evidence_replaces_startup_error_with_contract_error(
    tmp_path: Path,
) -> None:
    sequential_plan = _plan(tmp_path, rounds=(2,))
    plan = replace(
        sequential_plan,
        execution=_slurm_execution(sequential_plan, tmp_path),
    )
    invalid = _slurm_evidence(outcome="failed")
    invalid["unknown"] = True
    startup_error = WorkerPreflightError("primary preflight failure")

    def fail(_tasks: tuple[RoundTask, ...]) -> tuple[RoundResult, ...]:
        raise startup_error

    fake = _FakeExecutor(fail, execution_evidence=invalid)

    with pytest.raises(
        ExecutorStartupError,
        match="invalid SLURM execution evidence",
    ) as captured:
        run_workflow(plan, _executor_factory=_factory(fake))

    assert captured.value is not startup_error
    assert captured.value.__cause__ is startup_error
    assert captured.value.run_record is not None
    assert captured.value.execution_evidence_json is None


def test_local_dask_does_not_claim_injected_slurm_evidence(tmp_path: Path) -> None:
    sequential_plan = _plan(tmp_path, rounds=(2,))
    plan = replace(
        sequential_plan,
        execution=_local_execution(sequential_plan, tmp_path),
    )
    fake = _FakeExecutor(
        lambda tasks: tuple(_result(task) for task in tasks),
        execution_evidence=_slurm_evidence(),
    )

    result = run_workflow(plan, _executor_factory=_factory(fake))

    assert result.execution_evidence_json == "{}"
    run_record = json.loads(Path(result.run_record).read_text(encoding="utf-8"))
    assert run_record["execution"] == {}


def test_all_reused_slurm_workflow_does_not_construct_an_executor_or_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sequential_plan = _plan(tmp_path, rounds=(2,))
    plan = replace(
        sequential_plan,
        execution=_slurm_execution(sequential_plan, tmp_path),
    )
    for artifact in _artifacts(plan.tasks[0]):
        _commit_artifact(artifact, plan)

    def unexpected_factory(_profile: ExecutionProfile) -> WorkflowExecutor:
        raise AssertionError("all-reused SLURM workflow constructed an executor")

    monkeypatch.setattr(workflow_api, "_default_executor_factory", unexpected_factory)

    result = run_workflow(plan)

    assert tuple(round_result.status for round_result in result.rounds) == ("reused",)
    assert result.execution_evidence_json == "{}"
    run_record = json.loads(Path(result.run_record).read_text(encoding="utf-8"))
    assert run_record["execution"] == {}


def test_run_workflow_skips_a_complete_round_without_opening_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from cyexm.workflow import round_task as round_task_module

    plan = _plan(tmp_path, rounds=(2,))
    for artifact in _artifacts(plan.tasks[0]):
        _commit_artifact(artifact, plan)
    calls: list[int] = []
    monkeypatch.setattr(
        round_task_module,
        "run_round_task",
        lambda task: calls.append(task.round) or _result(task),
    )

    result = run_workflow(plan)

    assert calls == []
    assert tuple(item.status for item in result.rounds) == ("reused",)


def test_run_workflow_executes_when_a_channel_record_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from cyexm.workflow import round_task as round_task_module

    plan = _plan(tmp_path, rounds=(2,))
    for artifact in _artifacts(plan.tasks[0])[:-1]:
        _commit_artifact(artifact, plan)
    calls: list[int] = []
    monkeypatch.setattr(
        round_task_module,
        "run_round_task",
        lambda task: calls.append(task.round) or _result(task),
    )

    result = run_workflow(plan)

    assert calls == [2]
    assert tuple(item.status for item in result.rounds) == ("completed",)


def test_run_workflow_executes_when_registration_evidence_is_stale(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from cyexm.workflow import round_task as round_task_module

    plan = _plan(tmp_path, rounds=(2,))
    artifacts = _artifacts(plan.tasks[0])
    for artifact in artifacts:
        _commit_artifact(artifact, plan)
    artifacts[0].write_bytes(b"stale-affine")
    calls: list[int] = []
    monkeypatch.setattr(
        round_task_module,
        "run_round_task",
        lambda task: calls.append(task.round) or _result(task),
    )

    result = run_workflow(plan)

    assert calls == [2]
    assert tuple(item.status for item in result.rounds) == ("completed",)


def test_run_workflow_executes_when_fixed_reference_tiff_is_corrupt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from cyexm.workflow import round_task as round_task_module

    plan = _plan(tmp_path, rounds=(2,))
    for artifact in _artifacts(plan.tasks[0]):
        _commit_artifact(artifact, plan)
    fixed_reference = plan.tasks[0].registration_directory / "fixed_reference.tif"
    _corrupt_tiff_while_preserving_record_identity(fixed_reference)
    calls: list[int] = []
    monkeypatch.setattr(
        round_task_module,
        "run_round_task",
        lambda task: calls.append(task.round) or _result(task),
    )

    result = run_workflow(plan)

    assert calls == [2]
    assert result.rounds[0].status == "completed"


def test_run_workflow_forces_only_selected_rounds(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    fake = _FakeExecutor(lambda tasks: tuple(_result(task) for task in tasks))

    run_workflow(
        plan,
        force_rounds=frozenset({3}),
        _executor_factory=_factory(fake),
    )

    submitted = fake.calls[0]
    assert tuple(task.round for task in submitted) == (2, 3)
    assert tuple(task.force for task in submitted) == (False, True)
    assert tuple(task.force for task in plan.tasks) == (False, False)


def test_run_workflow_force_all_replaces_every_immutable_task(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    fake = _FakeExecutor(lambda tasks: tuple(_result(task) for task in tasks))

    run_workflow(plan, force_all=True, _executor_factory=_factory(fake))

    assert all(task.force for task in fake.calls[0])
    assert not any(task.force for task in plan.tasks)


def test_run_workflow_rejects_both_force_modes_before_writing(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    fake = _FakeExecutor(lambda tasks: tuple(_result(task) for task in tasks))

    with pytest.raises(WorkflowPlanError, match="force_all"):
        run_workflow(
            plan,
            force_all=True,
            force_rounds=frozenset({2}),
            _executor_factory=_factory(fake),
        )

    assert not plan.execution.paths.output_root.exists()
    assert fake.calls == []


def test_run_workflow_rejects_unknown_forced_round_before_writing(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    fake = _FakeExecutor(lambda tasks: tuple(_result(task) for task in tasks))

    with pytest.raises(WorkflowPlanError, match=r"unknown.*99"):
        run_workflow(
            plan,
            force_rounds=frozenset({99}),
            _executor_factory=_factory(fake),
        )

    assert not plan.execution.paths.output_root.exists()
    assert fake.calls == []


def test_run_workflow_rejects_unsafe_dataset_id_before_reading_or_writing(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    scientific = json.loads(plan.scientific_json)
    scientific["dataset_id"] = "../outside"
    scientific_json = canonical_json(scientific)
    unsafe = replace(
        plan,
        dataset_id="../outside",
        scientific_json=scientific_json,
        scientific_fingerprint=sha256_fingerprint(scientific_json),
    )
    fake = _FakeExecutor(lambda tasks: tuple(_result(task) for task in tasks))

    with pytest.raises(WorkflowPlanError, match="dataset_id"):
        run_workflow(unsafe, _executor_factory=_factory(fake))

    assert not (tmp_path / "outside").exists()
    assert fake.calls == []


def test_run_workflow_rejects_workflow_runs_symlink_escape_before_mutation(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    layout = WorkflowLayout.for_dataset(
        plan.execution.paths.output_root, plan.dataset_id
    )
    layout.root.mkdir(parents=True)
    outside = tmp_path / "outside-runs"
    outside.mkdir()
    layout.workflow_runs.symlink_to(outside, target_is_directory=True)
    fake = _FakeExecutor(lambda tasks: tuple(_result(task) for task in tasks))

    with pytest.raises(WorkflowPlanError, match="outside workflow output"):
        run_workflow(plan, _executor_factory=_factory(fake))

    assert not layout.plan_path.exists()
    assert tuple(outside.iterdir()) == ()
    assert fake.calls == []


def test_run_workflow_rejects_run_record_symlink_escape_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    layout = WorkflowLayout.for_dataset(
        plan.execution.paths.output_root, plan.dataset_id
    )
    layout.workflow_runs.mkdir(parents=True)
    outside = tmp_path / "outside-run.json"
    outside.write_text("do not replace", encoding="utf-8")
    layout.run_record("fixed-run-id").symlink_to(outside)
    monkeypatch.setattr(workflow_api, "uuid4", lambda: "fixed-run-id")
    fake = _FakeExecutor(lambda tasks: tuple(_result(task) for task in tasks))

    with pytest.raises(WorkflowPlanError, match="outside workflow output"):
        run_workflow(plan, _executor_factory=_factory(fake))

    assert not layout.plan_path.exists()
    assert outside.read_text(encoding="utf-8") == "do not replace"
    assert fake.calls == []


def test_run_workflow_rejects_malformed_scientific_schema_before_mutation(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    scientific = json.loads(plan.scientific_json)
    del scientific["policies"]
    scientific_json = canonical_json(scientific)
    malformed = replace(
        plan,
        scientific_json=scientific_json,
        scientific_fingerprint=sha256_fingerprint(scientific_json),
    )
    fake = _FakeExecutor(lambda tasks: tuple(_result(task) for task in tasks))

    with pytest.raises(WorkflowPlanError, match=r"scientific plan.*keys"):
        run_workflow(malformed, _executor_factory=_factory(fake))

    assert not plan.execution.paths.output_root.exists()
    assert fake.calls == []


def test_run_workflow_rejects_task_output_mismatch_before_mutation(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    forged_task = replace(
        plan.tasks[0],
        registration_directory=tmp_path / "outside" / "registration",
    )
    forged = replace(plan, tasks=(forged_task, *plan.tasks[1:]))
    fake = _FakeExecutor(lambda tasks: tuple(_result(task) for task in tasks))

    with pytest.raises(WorkflowPlanError, match="registration_directory"):
        run_workflow(forged, _executor_factory=_factory(fake))

    assert not plan.execution.paths.output_root.exists()
    assert not (tmp_path / "outside").exists()
    assert fake.calls == []


def test_run_workflow_rejects_expected_artifact_mismatch_before_mutation(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    forged = replace(plan, expected_artifacts=plan.expected_artifacts[:-1])
    fake = _FakeExecutor(lambda tasks: tuple(_result(task) for task in tasks))

    with pytest.raises(WorkflowPlanError, match="expected_artifacts"):
        run_workflow(forged, _executor_factory=_factory(fake))

    assert not plan.execution.paths.output_root.exists()
    assert fake.calls == []


def test_run_workflow_rejects_boolean_manifest_spacing_before_mutation(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    forged_manifest = replace(
        plan.manifest,
        voxel_spacing_um=(True, 1.0, 1.0),  # type: ignore[arg-type]
    )
    forged = replace(plan, manifest=forged_manifest)
    fake = _FakeExecutor(lambda tasks: tuple(_result(task) for task in tasks))

    with pytest.raises(WorkflowPlanError, match="voxel_spacing"):
        run_workflow(forged, _executor_factory=_factory(fake))

    assert not plan.execution.paths.output_root.exists()
    assert fake.calls == []


def test_run_workflow_rejects_boolean_task_spacing_before_mutation(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    forged_task = replace(
        plan.tasks[0],
        voxel_spacing_xyz=(True, 1.0, 1.0),  # type: ignore[arg-type]
    )
    forged = replace(plan, tasks=(forged_task, *plan.tasks[1:]))
    fake = _FakeExecutor(lambda tasks: tuple(_result(task) for task in tasks))

    with pytest.raises(WorkflowPlanError, match="voxel_spacing"):
        run_workflow(forged, _executor_factory=_factory(fake))

    assert not plan.execution.paths.output_root.exists()
    assert fake.calls == []


def test_run_workflow_wraps_overflowing_task_spacing_before_mutation(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    forged_task = replace(
        plan.tasks[0],
        voxel_spacing_xyz=(10**1000, 1.0, 1.0),
    )
    forged = replace(plan, tasks=(forged_task, *plan.tasks[1:]))
    fake = _FakeExecutor(lambda tasks: tuple(_result(task) for task in tasks))

    with pytest.raises(WorkflowPlanError, match="voxel_spacing"):
        run_workflow(forged, _executor_factory=_factory(fake))

    assert not plan.execution.paths.output_root.exists()
    assert fake.calls == []


@pytest.mark.parametrize(
    ("case", "field"),
    [
        ("dataset_id", "dataset_id"),
        ("round_bool", "round"),
        ("round_alias", "round"),
        ("strategy", "strategy"),
        ("fixed_image", "fixed_image"),
        ("moving_image", "moving_image"),
        ("registration_directory", "registration_directory"),
        ("log_path", "log_path"),
        ("spacing_container", "voxel_spacing_xyz"),
        ("spacing_scalar", "voxel_spacing_xyz"),
        ("registration_model", "registration"),
        ("registration_scalar", "registration.linear.type"),
        ("native_model", "native_threads"),
        ("native_itk", "native_threads.itk"),
        ("native_blas", "native_threads.blas"),
        ("channels_container", "channels"),
        ("channel_model", "channels[0]"),
        ("channel_source", "channels[0].source"),
        ("channel_output", "channels[0].output"),
        ("channel_interpolation", "channels[0].interpolation"),
        ("chunk_container", "chunk_size_zyx"),
        ("chunk_scalar", "chunk_size_zyx[0]"),
        ("overlap_int", "overlap_fraction"),
        ("overlap_bool", "overlap_fraction"),
        ("force", "force"),
    ],
)
def test_run_workflow_rejects_nonexact_round_task_fields_before_mutation(
    case: str,
    field: str,
    tmp_path: Path,
) -> None:
    plan, invalid_task = _invalid_task(tmp_path, case)
    forged = replace(plan, tasks=(invalid_task, *plan.tasks[1:]))
    fake = _FakeExecutor(lambda tasks: tuple(_result(task) for task in tasks))

    with pytest.raises(
        WorkflowPlanError,
        match=rf"tasks\[0\].*{re.escape(field)}",
    ):
        run_workflow(forged, _executor_factory=_factory(fake))

    assert not plan.execution.paths.output_root.exists()
    assert fake.calls == []


@pytest.mark.parametrize(
    "case",
    [
        "schema",
        "executor",
        "source_relative",
        "source_noncanonical",
        "path_overlap",
        "native_itk",
        "native_blas",
        "worker_missing",
        "scratch_archive",
        "local_workers",
        "local_threads",
        "local_processes",
        "local_processes_false",
        "local_memory",
        "local_timeout",
        "local_retries",
        "local_negative_retries",
        "section_exclusivity",
        "slurm_directive",
        "slurm_directives_list",
        "slurm_directive_newline",
        "slurm_workers",
        "slurm_cores",
        "slurm_processes",
        "slurm_queue",
        "slurm_memory",
        "slurm_walltime",
        "slurm_interface",
        "slurm_scheduler",
        "slurm_timeout",
        "slurm_retries",
        "slurm_negative_retries",
        "slurm_job_name",
    ],
)
def test_run_workflow_revalidates_direct_execution_profile_before_mutation(
    case: str,
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    forged = replace(plan, execution=_invalid_execution(plan, tmp_path, case))
    fake = _FakeExecutor(lambda tasks: tuple(_result(task) for task in tasks))

    with pytest.raises(WorkflowPlanError, match="execution profile is invalid"):
        run_workflow(forged, _executor_factory=_factory(fake))

    assert not plan.execution.paths.output_root.exists()
    assert fake.calls == []


def test_run_workflow_rejects_incompatible_snapshot_without_changing_it(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    layout = WorkflowLayout.for_dataset(
        plan.execution.paths.output_root, plan.dataset_id
    )
    layout.plan_path.parent.mkdir(parents=True)
    before = b'{"dataset_id":"different","rounds":[],"schema_version":1}\n'
    layout.plan_path.write_bytes(before)
    fake = _FakeExecutor(lambda tasks: tuple(_result(task) for task in tasks))

    with pytest.raises(WorkflowPlanError, match="scientific-plan fingerprint"):
        run_workflow(plan, _executor_factory=_factory(fake))

    assert layout.plan_path.read_bytes() == before
    assert not layout.workflow_runs.exists()
    assert fake.calls == []


def test_run_workflow_excludes_complete_rounds_and_reports_them_reused(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    for artifact in _artifacts(plan.tasks[0]):
        _commit_artifact(artifact, plan)
    fake = _FakeExecutor(lambda tasks: tuple(_result(task) for task in tasks))

    result = run_workflow(plan, _executor_factory=_factory(fake))

    assert tuple(task.round for task in fake.calls[0]) == (3,)
    assert tuple(item.round for item in result.rounds) == (2, 3)
    assert tuple(item.status for item in result.rounds) == ("reused", "completed")
    assert result.rounds[0].reused_artifacts == tuple(
        str(path) for path in _artifacts(plan.tasks[0])
    )


def test_run_workflow_force_submits_an_otherwise_complete_round(tmp_path: Path) -> None:
    plan = _plan(tmp_path, rounds=(2,))
    for artifact in _artifacts(plan.tasks[0]):
        _commit_artifact(artifact, plan)
    fake = _FakeExecutor(lambda tasks: tuple(_result(task) for task in tasks))

    result = run_workflow(
        plan,
        force_rounds=frozenset({2}),
        _executor_factory=_factory(fake),
    )

    assert tuple(task.round for task in fake.calls[0]) == (2,)
    assert fake.calls[0][0].force
    assert result.rounds[0].status == "completed"


def test_run_workflow_does_not_reuse_artifacts_from_other_native_threads(
    tmp_path: Path,
) -> None:
    original = _plan(tmp_path, rounds=(2,))
    for artifact in _artifacts(original.tasks[0]):
        _commit_artifact(artifact, original)
    requested_threads = NativeThreadSpec(itk=2, blas=3)
    task = replace(original.tasks[0], native_threads=requested_threads)
    execution = replace(original.execution, native_threads=requested_threads)
    plan = replace(original, tasks=(task,), execution=execution)
    fake = _FakeExecutor(
        lambda tasks: tuple(
            replace(
                _result(item),
                native_threads=(
                    ("itk", requested_threads.itk),
                    ("blas", requested_threads.blas),
                ),
            )
            for item in tasks
        )
    )

    result = run_workflow(plan, _executor_factory=_factory(fake))

    assert tuple(item.round for item in fake.calls[0]) == (2,)
    assert result.rounds[0].status == "completed"


def test_run_workflow_writes_failed_summary_then_raises_sorted_rounds(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    fake = _FakeExecutor(
        lambda tasks: (_result(tasks[1]), _result(tasks[0], failed=True))
    )

    with pytest.raises(WorkflowExecutionError) as captured:
        run_workflow(plan, _executor_factory=_factory(fake))

    assert captured.value.failed_rounds == (2,)
    record = Path(captured.value.run_record)
    assert record.is_file()
    payload = json.loads(record.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert [item["round"] for item in payload["rounds"]] == [2, 3]
    assert not tuple(record.parent.glob(".*.tmp*"))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda result, _task: replace(result, dataset_id="other"), "dataset_id"),
        (
            lambda result, task: replace(
                result,
                artifact_paths=(
                    *result.artifact_paths,
                    str(task.log_path.parent / "rogue.tif"),
                ),
            ),
            "artifact_paths",
        ),
        (
            lambda result, _task: replace(result, status="failed", failure=None),
            "failure",
        ),
        (
            lambda result, _task: replace(result, started_at="not-a-timestamp"),
            "started_at",
        ),
        (
            lambda result, _task: replace(
                result,
                ended_at="2026-07-20T10:00:01+00:00",
            ),
            "ended_at",
        ),
        (
            lambda result, _task: replace(
                result,
                started_at="2026-07-20T10:00:02Z",
            ),
            "ended_at must not precede started_at",
        ),
    ],
)
def test_run_workflow_rejects_inconsistent_executor_results_before_persistence(
    mutate: Callable[[RoundResult, RoundTask], RoundResult],
    message: str,
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)

    def execute(tasks: tuple[RoundTask, ...]) -> tuple[RoundResult, ...]:
        return (mutate(_result(tasks[0]), tasks[0]), _result(tasks[1]))

    with pytest.raises(RoundExecutionError, match=message):
        run_workflow(
            plan,
            _executor_factory=_factory(_FakeExecutor(execute)),
        )

    records = tuple(
        WorkflowLayout.for_dataset(
            plan.execution.paths.output_root, plan.dataset_id
        ).workflow_runs.glob("*.json")
    )
    assert records == ()


@pytest.mark.parametrize(
    ("case", "field"),
    [
        ("dataset_id", "dataset_id"),
        ("round", "round"),
        ("status", "status"),
        ("started_at", "started_at"),
        ("ended_at", "ended_at"),
        ("duration", "duration_seconds"),
        ("hostname", "hostname"),
        ("pid", "pid"),
        ("dask_task_key", "dask_task_key"),
        ("slurm_job_id", "slurm_job_id"),
        ("strategy", "strategy"),
        ("native_container", "native_threads"),
        ("native_entry", "native_threads"),
        ("native_name", "native_threads"),
        ("native_value", "native_threads"),
        ("native_zero", "native_threads"),
        ("native_name_unknown", "native_threads"),
        ("artifact_container", "artifact_paths"),
        ("artifact_element", "artifact_paths"),
        ("reused_container", "reused_artifacts"),
        ("reused_element", "reused_artifacts"),
        ("log_path", "log_path"),
        ("failure_model", "failure"),
        ("failure_stage", "failure.stage"),
        ("failure_exception_type", "failure.exception_type"),
        ("failure_message", "failure.message"),
    ],
)
def test_run_workflow_rejects_nonexact_round_result_fields_before_persistence(
    case: str,
    field: str,
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)

    def execute(tasks: tuple[RoundTask, ...]) -> tuple[RoundResult, ...]:
        return (_invalid_result(tasks[0], case), _result(tasks[1]))

    with pytest.raises(RoundExecutionError, match=re.escape(field)):
        run_workflow(
            plan,
            _executor_factory=_factory(_FakeExecutor(execute)),
        )

    records = tuple(
        WorkflowLayout.for_dataset(
            plan.execution.paths.output_root, plan.dataset_id
        ).workflow_runs.glob("*.json")
    )
    assert records == ()


@pytest.mark.parametrize("round_value", [[], "2"])
def test_run_workflow_validates_result_round_before_indexing_results(
    round_value: object,
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)

    def execute(tasks: tuple[RoundTask, ...]) -> tuple[RoundResult, ...]:
        invalid = replace(_result(tasks[0]), round=round_value)  # type: ignore[arg-type]
        return (invalid, _result(tasks[1]))

    with pytest.raises(RoundExecutionError, match=r"round.*built-in integer"):
        run_workflow(
            plan,
            _executor_factory=_factory(_FakeExecutor(execute)),
        )

    records = tuple(
        WorkflowLayout.for_dataset(
            plan.execution.paths.output_root, plan.dataset_id
        ).workflow_runs.glob("*.json")
    )
    assert records == ()


def test_run_workflow_writes_cancelled_summary_and_reraises(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)

    def cancel(_tasks: tuple[RoundTask, ...]) -> tuple[RoundResult, ...]:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_workflow(
            plan,
            _executor_factory=_factory(_FakeExecutor(cancel)),
        )

    records = tuple(
        WorkflowLayout.for_dataset(
            plan.execution.paths.output_root, plan.dataset_id
        ).workflow_runs.glob("*.json")
    )
    assert len(records) == 1
    payload = json.loads(records[0].read_text(encoding="utf-8"))
    assert payload["status"] == "cancelled"
    assert payload["rounds"] == []


def test_run_workflow_skips_unhashable_partial_result_during_cancellation(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)

    def interrupt(tasks: tuple[RoundTask, ...]) -> tuple[RoundResult, ...]:
        invalid = replace(_result(tasks[0]), round=[])  # type: ignore[arg-type]
        raise workflow_api._SequentialExecutionInterrupted(
            KeyboardInterrupt("injected"),
            (invalid,),
        )

    with pytest.raises(KeyboardInterrupt, match="injected"):
        run_workflow(
            plan,
            _executor_factory=_factory(_FakeExecutor(interrupt)),
        )

    records = tuple(
        WorkflowLayout.for_dataset(
            plan.execution.paths.output_root, plan.dataset_id
        ).workflow_runs.glob("*.json")
    )
    assert len(records) == 1
    payload = json.loads(records[0].read_text(encoding="utf-8"))
    assert payload["status"] == "cancelled"
    assert payload["rounds"] == []


def test_run_workflow_cancelled_summary_preserves_completed_sequential_rounds(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)

    def runner(task: RoundTask) -> RoundResult:
        if task.round == 3:
            raise KeyboardInterrupt("injected")
        return _result(task)

    with pytest.raises(KeyboardInterrupt, match="injected"):
        run_workflow(
            plan,
            _executor_factory=_factory(SequentialExecutor(runner=runner)),
        )

    records = tuple(
        WorkflowLayout.for_dataset(
            plan.execution.paths.output_root, plan.dataset_id
        ).workflow_runs.glob("*.json")
    )
    assert len(records) == 1
    payload = json.loads(records[0].read_text(encoding="utf-8"))
    assert payload["status"] == "cancelled"
    assert [item["round"] for item in payload["rounds"]] == [2]


def test_run_workflow_records_cancellation_during_snapshot_creation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)

    def cancel_snapshot(_plan: WorkflowPlan, _layout: WorkflowLayout) -> None:
        raise KeyboardInterrupt("snapshot interruption")

    monkeypatch.setattr(workflow_api, "_ensure_scientific_snapshot", cancel_snapshot)

    with pytest.raises(KeyboardInterrupt, match="snapshot interruption"):
        run_workflow(
            plan,
            _executor_factory=_factory(_FakeExecutor(lambda tasks: ())),
        )

    records = tuple(
        WorkflowLayout.for_dataset(
            plan.execution.paths.output_root, plan.dataset_id
        ).workflow_runs.glob("*.json")
    )
    assert len(records) == 1
    payload = json.loads(records[0].read_text(encoding="utf-8"))
    assert payload["status"] == "cancelled"
    assert payload["rounds"] == []


def test_run_workflow_recovers_a_cancelled_final_record_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    original_write = workflow_api._write_run_record
    calls = 0

    def interrupt_once(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise KeyboardInterrupt("record interruption")
        original_write(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(workflow_api, "_write_run_record", interrupt_once)

    with pytest.raises(KeyboardInterrupt, match="record interruption"):
        run_workflow(
            plan,
            _executor_factory=_factory(
                _FakeExecutor(lambda tasks: tuple(_result(task) for task in tasks))
            ),
        )

    records = tuple(
        WorkflowLayout.for_dataset(
            plan.execution.paths.output_root, plan.dataset_id
        ).workflow_runs.glob("*.json")
    )
    assert len(records) == 1
    payload = json.loads(records[0].read_text(encoding="utf-8"))
    assert payload["status"] == "cancelled"
    assert [item["round"] for item in payload["rounds"]] == [2, 3]
