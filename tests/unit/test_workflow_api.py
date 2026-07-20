from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path, PurePosixPath

import pytest

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
from cyexm.errors import RoundExecutionError, WorkflowExecutionError, WorkflowPlanError
from cyexm.execution.base import WorkflowExecutor
from cyexm.execution.sequential import SequentialExecutor
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


class _FakeExecutor:
    def __init__(
        self,
        callback: Callable[[tuple[RoundTask, ...]], tuple[RoundResult, ...]],
    ) -> None:
        self.callback = callback
        self.calls: list[tuple[RoundTask, ...]] = []

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
