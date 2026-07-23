from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from cyexm.chunking import compute_chunk_grid
from cyexm.config.models import (
    NativeThreadSpec,
    NonlinearRegistrationStageSpec,
    RegistrationSpec,
    RegistrationStageSpec,
)
from cyexm.execution.base import WorkflowTaskRunner
from cyexm.runtime import NATIVE_THREAD_ENV_VARS, ResolvedNativeThreads
from cyexm.workflow.fine_models import (
    ChannelTransformTask,
    ChunkAssemblyTask,
    FineTaskKey,
    FineTaskResult,
    NonlinearChunkTask,
    RoundPreparationTask,
    RoundValidationTask,
    WholeImageRegistrationTask,
)
from cyexm.workflow.fine_task import run_fine_task
from cyexm.workflow.models import ChannelTask, RoundResult, RoundTask
from cyexm.workflow.task import run_workflow_task

pytestmark = pytest.mark.unit


def _round_task(tmp_path: Path, *, strategy: str = "serial_chunked") -> RoundTask:
    root = tmp_path / "round-002"
    return RoundTask(
        dataset_id="dataset",
        round=2,
        strategy=strategy,  # type: ignore[arg-type]
        fixed_image=tmp_path / "fixed.tif",
        moving_image=tmp_path / "moving.tif",
        registration_directory=root / "registration",
        log_path=root / "logs" / "round-002.log",
        voxel_spacing_xyz=(1.0, 1.0, 1.0),
        registration=RegistrationSpec(
            crop=False,
            linear=RegistrationStageSpec("TRSAA", "high"),
            nonlinear=NonlinearRegistrationStageSpec("SyNOnly", "high", "CC", 4),
        ),
        native_threads=NativeThreadSpec(2, 3),
        channels=(
            ChannelTask(
                source=tmp_path / "488.tif",
                output=root / "channels" / "488.tif",
                interpolation="linear",
            ),
            ChannelTask(
                source=tmp_path / "647.tif",
                output=root / "channels" / "647.tif",
                interpolation="nearestNeighbor",
            ),
        ),
        chunk_size_zyx=(4, 8, 8) if strategy == "serial_chunked" else None,
        overlap_fraction=0.25 if strategy == "serial_chunked" else None,
    )


def _descriptor() -> dict[str, object]:
    identity = {"path": "/tmp/artifact", "st_size": 1, "st_mtime_ns": 2}
    return {
        "format_version": 1,
        "fixed_scalar": identity,
        "moving_linear_scalar": identity,
        "fixed_mask": identity,
        "moving_linear_mask": identity,
        "native_fixed": identity,
        "native_moving": identity,
        "moving_crop": None,
        "affine": identity,
        "geometry": {
            "shape": [9, 17, 19],
            "origin": [0.0, 0.0, 0.0],
            "spacing": [1.0, 1.0, 1.0],
            "direction": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        },
        "scalar_policy": "float32-preserve-registration-values-v1",
        "mask_policy": (
            "greater-than-zero-affine-nearest-neighbor-threshold-0.5-float32-v1"
        ),
        "native_threads": {"ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": 2},
    }


def _tasks(tmp_path: Path) -> tuple[object, ...]:
    chunked = _round_task(tmp_path)
    preparation = RoundPreparationTask.create(chunked)
    chunk = NonlinearChunkTask.create(
        chunked,
        compute_chunk_grid((9, 17, 19), (4, 8, 8), 0.25)[0],
        prepared_descriptor=_descriptor(),
        dependencies=(preparation.key,),
    )
    assembly = ChunkAssemblyTask.create(
        chunked,
        prepared_descriptor=_descriptor(),
        dependencies=tuple(
            FineTaskKey(2, "chunk", f"{item.chunk_id:04d}")
            for item in compute_chunk_grid((9, 17, 19), (4, 8, 8), 0.25)
        ),
    )
    whole_round = _round_task(tmp_path, strategy="whole_image")
    whole = WholeImageRegistrationTask.create(whole_round)
    channel = ChannelTransformTask.create(whole_round, 0, dependencies=(whole.key,))
    validation = RoundValidationTask.create(
        whole_round,
        dependencies=tuple(
            FineTaskKey(2, "channel", f"{index:04d}")
            for index in range(len(whole_round.channels))
        ),
    )
    return preparation, chunk, assembly, whole, channel, validation


def _resolved(task: object) -> ResolvedNativeThreads:
    round_task = task.round_task  # type: ignore[attr-defined]
    itk_name, *blas_names = NATIVE_THREAD_ENV_VARS
    return ResolvedNativeThreads(
        values={
            itk_name: round_task.native_threads.itk,
            **{name: round_task.native_threads.blas for name in blas_names},
        },
        sources={name: "explicit" for name in NATIVE_THREAD_ENV_VARS},
    )


def _progress_request(
    task: object,
    tmp_path: Path,
) -> object:
    from cyexm.registration.artifacts import FileIdentity, StageRequest
    from cyexm.spatial import ImageGeometry

    source = tmp_path / "progress-source.tif"
    upstream = tmp_path / "progress-upstream.mat"
    source.write_bytes(b"source")
    upstream.write_bytes(b"upstream")
    is_channel = type(task) is ChannelTransformTask
    return StageRequest.for_identities(
        stage="channel-transform-v1" if is_channel else "nonlinear",
        inputs=(FileIdentity.from_path(source),),
        geometry=ImageGeometry(
            shape=(3, 4, 5),
            origin=(0.0, 0.0, 0.0),
            spacing=(1.0, 1.0, 1.0),
            direction=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ),
        method="warp-affine" if is_channel else "SyNOnly",
        accuracy="not-applicable" if is_channel else "high",
        native_threads=_resolved(task).values,
        upstream=FileIdentity.from_path(upstream),
        parameters={
            "nonlinear_metric": "CC",
            "nonlinear_radius": 4,
            "interpolation": "linear",
        },
    )


def _commit_test_artifact(path: Path, request: object) -> None:
    from cyexm.registration.artifacts import ArtifactStore

    path.parent.mkdir(parents=True, exist_ok=True)
    ArtifactStore().commit(
        path,
        request,  # type: ignore[arg-type]
        lambda candidate: candidate.write_bytes(b"artifact"),
        lambda candidate: candidate.read_bytes() == b"artifact",
    )


def test_fine_task_import_is_scientific_and_scheduler_runtime_free() -> None:
    code = """
import sys
import cyexm.workflow.fine_task
for name in ("ants", "numpy", "scipy", "dask", "distributed", "dask_jobqueue"):
    assert name not in sys.modules, name
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=False,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path.cwd() / "src")},
    )

    assert result.returncode == 0, result.stderr


def test_workflow_runner_type_accepts_fine_results() -> None:
    assert "WorkflowTaskResult" in str(WorkflowTaskRunner.__value__)


def test_native_threads_are_configured_before_stage_dispatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import cyexm.workflow.fine_task as worker

    task = WholeImageRegistrationTask.create(
        _round_task(tmp_path, strategy="whole_image")
    )
    configured = False

    def configure(_config: object) -> ResolvedNativeThreads:
        nonlocal configured
        configured = True
        return _resolved(task)

    def execute(_task: object, _resolved: object) -> object:
        assert configured
        return worker._StageOutcome(task.output_paths, (), None)

    monkeypatch.setattr(worker, "configure_native_threads", configure)
    monkeypatch.setattr(worker, "_execute_fine_task", execute)

    result = run_fine_task(task)

    assert result.status == "completed"
    assert result.key == task.key


@pytest.mark.parametrize(
    ("task_index", "stage"),
    (
        (0, "preparation"),
        (1, "chunk"),
        (2, "assembly"),
        (3, "whole-image"),
        (4, "channel"),
        (5, "validation"),
    ),
)
def test_each_stage_failure_returns_one_exact_isolated_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    task_index: int,
    stage: str,
) -> None:
    import cyexm.workflow.fine_task as worker

    task = _tasks(tmp_path)[task_index]
    monkeypatch.setattr(
        worker,
        "configure_native_threads",
        lambda _config: _resolved(task),
    )
    monkeypatch.setattr(
        worker,
        "_execute_fine_task",
        lambda _task, _resolved: (_ for _ in ()).throw(
            RuntimeError(f"injected {stage}")
        ),
    )

    result = run_fine_task(task)

    assert type(result) is FineTaskResult
    assert result.status == "failed"
    assert result.key == task.key
    assert result.failure is not None
    assert result.failure.stage == stage
    assert result.failure.message == f"injected {stage}"
    assert result.log_path.endswith(task.key.log_name)
    assert not result.log_path.endswith("round-002.log")
    assert Path(result.log_path).is_file()


def test_stage_failure_preserves_a_real_newly_committed_partial_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import cyexm.workflow.fine_task as worker
    from cyexm.registration.artifacts import ArtifactStore

    task = _tasks(tmp_path)[3]
    path = task.output_paths[1]
    request = _progress_request(task, tmp_path)
    monkeypatch.setattr(
        worker, "configure_native_threads", lambda _config: _resolved(task)
    )

    def execute(*_args: object) -> object:
        path.parent.mkdir(parents=True, exist_ok=True)
        ArtifactStore().commit(
            path,
            request,  # type: ignore[arg-type]
            lambda candidate: candidate.write_bytes(b"partial"),
            lambda candidate: candidate.read_bytes() == b"partial",
        )
        raise RuntimeError("after commit")

    monkeypatch.setattr(worker, "_execute_fine_task", execute)

    result = run_fine_task(task)

    assert result.status == "failed"
    assert result.artifact_paths == (str(path),)
    assert result.reused_artifacts == ()


def test_stage_failure_preserves_real_accepted_unchanged_reuse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import cyexm.workflow.fine_task as worker
    from cyexm.registration.artifacts import ArtifactStore

    task = _tasks(tmp_path)[3]
    path = task.output_paths[1]
    request = _progress_request(task, tmp_path)
    _commit_test_artifact(path, request)
    monkeypatch.setattr(
        worker, "configure_native_threads", lambda _config: _resolved(task)
    )

    def execute(*_args: object) -> object:
        assert ArtifactStore().reuse(path, request, lambda candidate: candidate) == path  # type: ignore[arg-type]
        raise RuntimeError("after reuse")

    monkeypatch.setattr(worker, "_execute_fine_task", execute)

    result = run_fine_task(task)

    assert result.artifact_paths == (str(path),)
    assert result.reused_artifacts == (str(path),)


def test_forced_stage_failure_never_reports_reuse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import cyexm.workflow.fine_task as worker
    from cyexm.registration.artifacts import ArtifactStore

    round_task = replace(_round_task(tmp_path, strategy="whole_image"), force=True)
    task = WholeImageRegistrationTask.create(round_task)
    path = task.output_paths[1]
    request = _progress_request(task, tmp_path)
    _commit_test_artifact(path, request)
    monkeypatch.setattr(
        worker, "configure_native_threads", lambda _config: _resolved(task)
    )

    def execute(*_args: object) -> object:
        ArtifactStore().reuse(path, request, lambda candidate: candidate)  # type: ignore[arg-type]
        raise RuntimeError("after forced reuse")

    monkeypatch.setattr(worker, "_execute_fine_task", execute)

    result = run_fine_task(task)

    assert result.artifact_paths == (str(path),)
    assert result.reused_artifacts == ()


@pytest.mark.parametrize(
    "mismatch",
    (
        "metric",
        "radius",
        "interpolation",
        "stage",
        "method",
        "accuracy",
        "geometry",
        "source",
        "upstream",
    ),
)
def test_failure_rejects_current_but_request_mismatched_completion_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mismatch: str,
) -> None:
    import cyexm.workflow.fine_task as worker
    from cyexm.registration.artifacts import ArtifactStore, FileIdentity, StageRequest
    from cyexm.spatial import ImageGeometry

    task = _tasks(tmp_path)[4 if mismatch == "interpolation" else 3]
    path = task.output_paths[0 if mismatch == "interpolation" else 1]
    expected = _progress_request(task, tmp_path)
    assert type(expected) is StageRequest
    parameters = dict(expected.parameters)
    alternate = tmp_path / f"alternate-{mismatch}"
    alternate.write_bytes(b"alternate")
    replacements: dict[str, object] = {}
    if mismatch == "metric":
        parameters["nonlinear_metric"] = "Mattes"
        replacements["parameters"] = parameters
    elif mismatch == "radius":
        parameters["nonlinear_radius"] = 3
        replacements["parameters"] = parameters
    elif mismatch == "interpolation":
        parameters["interpolation"] = "nearestNeighbor"
        replacements["parameters"] = parameters
    elif mismatch == "stage":
        replacements["stage"] = "affine"
    elif mismatch == "method":
        replacements["method"] = "SyN"
    elif mismatch == "accuracy":
        replacements["accuracy"] = "low"
    elif mismatch == "geometry":
        replacements["geometry"] = ImageGeometry(
            shape=(3, 4, 6),
            origin=(0.0, 0.0, 0.0),
            spacing=(1.0, 1.0, 1.0),
            direction=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        )
    elif mismatch == "source":
        replacements["inputs"] = (FileIdentity.from_path(alternate),)
    else:
        replacements["upstream"] = FileIdentity.from_path(alternate)
    stale = replace(expected, **replacements)  # type: ignore[arg-type]
    _commit_test_artifact(path, stale)
    monkeypatch.setattr(
        worker, "configure_native_threads", lambda _config: _resolved(task)
    )

    def execute(*_args: object) -> object:
        ArtifactStore().reuse(path, expected, lambda candidate: candidate)
        raise AssertionError("request-mismatched reuse must not be accepted")

    monkeypatch.setattr(worker, "_execute_fine_task", execute)

    result = run_fine_task(task)

    assert result.status == "failed"
    assert result.artifact_paths == ()
    assert result.reused_artifacts == ()


def test_failure_revalidates_the_exact_accepted_record_after_progress(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import cyexm.workflow.fine_task as worker
    from cyexm.registration.artifacts import (
        ArtifactStore,
        CompletionRecord,
        FileIdentity,
        StageRequest,
    )

    task = _tasks(tmp_path)[3]
    path = task.output_paths[1]
    request = _progress_request(task, tmp_path)
    assert type(request) is StageRequest
    store = ArtifactStore()
    monkeypatch.setattr(
        worker, "configure_native_threads", lambda _config: _resolved(task)
    )

    def execute(*_args: object) -> object:
        _commit_test_artifact(path, request)
        changed = replace(
            request,
            parameters={**request.parameters, "nonlinear_radius": 99},
        )
        store.record_path(path).write_text(
            CompletionRecord(1, changed, FileIdentity.from_path(path)).to_json(),
            encoding="utf-8",
        )
        raise RuntimeError("record changed after accepted progress")

    monkeypatch.setattr(worker, "_execute_fine_task", execute)

    result = run_fine_task(task)

    assert result.status == "failed"
    assert result.artifact_paths == ()
    assert result.reused_artifacts == ()


def test_worker_rejects_unplanned_stage_outcome(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import cyexm.workflow.fine_task as worker

    task = _tasks(tmp_path)[3]
    monkeypatch.setattr(
        worker, "configure_native_threads", lambda _config: _resolved(task)
    )
    monkeypatch.setattr(
        worker,
        "_execute_fine_task",
        lambda *_args: worker._StageOutcome((tmp_path / "wrong.tif",), (), None),
    )

    result = run_fine_task(task)

    assert result.status == "failed"
    assert result.artifact_paths == ()
    assert result.failure is not None
    assert "exact planned outputs" in result.failure.message


def test_exact_dispatch_preserves_coarse_round_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import cyexm.workflow.round_task as coarse

    task = _round_task(tmp_path, strategy="whole_image")
    expected = RoundResult(
        dataset_id=task.dataset_id,
        round=task.round,
        status="completed",
        started_at="start",
        ended_at="end",
        duration_seconds=1.0,
        hostname="host",
        pid=1,
        dask_task_key=None,
        slurm_job_id=None,
        strategy="whole_image",
        native_threads=(),
        artifact_paths=(),
        reused_artifacts=(),
        log_path=str(task.log_path),
        failure=None,
    )
    monkeypatch.setattr(coarse, "run_round_task", lambda _task: expected)

    result = run_workflow_task(task)

    assert type(result) is RoundResult
    assert result is expected


def test_exact_dispatch_routes_fine_tasks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import cyexm.workflow.fine_task as fine

    task = WholeImageRegistrationTask.create(
        _round_task(tmp_path, strategy="whole_image")
    )
    expected = SimpleNamespace(marker="fine")
    monkeypatch.setattr(fine, "run_fine_task", lambda _task: expected)

    assert run_workflow_task(task) is expected


def test_preparation_adapter_delegates_and_returns_exact_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import cyexm.registration.prepared as prepared
    import cyexm.workflow.fine_task as worker

    task = _tasks(tmp_path)[0]
    registration = object()
    manifest = SimpleNamespace(to_dict=lambda: _descriptor())

    def commit(actual: object) -> object:
        assert actual is registration
        for path in task.output_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")
        return manifest

    monkeypatch.setattr(worker, "_new_image_registration", lambda *_args: registration)
    monkeypatch.setattr(prepared, "commit_prepared_round", commit)

    outcome = worker._run_preparation(task)

    assert outcome.artifact_paths == task.output_paths
    assert outcome.prepared_descriptor == _descriptor()


def test_chunk_adapter_delegates_and_returns_exact_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import cyexm.registration.chunk_stages as stages
    import cyexm.workflow.fine_task as worker

    task = _tasks(tmp_path)[1]
    context = object()
    region = object()
    artifact = SimpleNamespace(
        warp_path=task.output_paths[0],
        scalar_core_path=task.output_paths[1],
        reused=False,
    )
    monkeypatch.setattr(
        worker,
        "_chunk_context",
        lambda *_args, **_kwargs: (context, region, (task.chunk,)),
    )

    def run(actual_context: object, actual_region: object, chunk: object) -> object:
        assert (actual_context, actual_region, chunk) == (
            context,
            region,
            task.chunk,
        )
        return artifact

    monkeypatch.setattr(stages, "run_nonlinear_chunk", run)

    assert worker._run_chunk(task, _resolved(task).values).artifact_paths == (
        task.output_paths
    )


def test_chunk_context_reconstructs_only_the_prepared_region(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import numpy as np

    import cyexm.registration.artifacts as artifacts
    import cyexm.registration.backend as backend_module
    import cyexm.registration.prepared as prepared
    import cyexm.workflow.fine_task as worker
    from cyexm.registration.artifacts import FileIdentity
    from cyexm.registration.prepared import PreparedRegion
    from cyexm.spatial import ImageGeometry

    task = _tasks(tmp_path)[1]
    geometry = ImageGeometry(
        shape=task.chunk.extended_shape,
        origin=(7.0, 8.0, 9.0),
        spacing=(1.0, 2.0, 3.0),
        direction=((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    )
    whole_geometry = ImageGeometry(
        shape=(9, 17, 19),
        origin=(0.0, 0.0, 0.0),
        spacing=(1.0, 1.0, 1.0),
        direction=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    identity = FileIdentity("/tmp/source", 1, 2)
    manifest = SimpleNamespace(
        geometry=whole_geometry,
        native_fixed=identity,
        native_moving=identity,
        affine=identity,
    )
    array = np.zeros(task.chunk.extended_shape, dtype=np.float32)
    region = PreparedRegion(array, array, array, array, geometry)
    calls: list[ImageGeometry] = []

    class Backend:
        def __init__(self, _threads: object) -> None:
            pass

        def load_image(self, *_args: object) -> object:
            raise AssertionError("chunk workers must not load a full TIFF")

        def from_geometry(self, _array: object, *, geometry: ImageGeometry) -> object:
            calls.append(geometry)
            return SimpleNamespace(geometry=geometry)

    monkeypatch.setattr(worker, "_validated_prepared_manifest", lambda *_args: manifest)
    monkeypatch.setattr(backend_module, "AntsRegistrationBackend", Backend)
    monkeypatch.setattr(prepared, "read_prepared_region", lambda *_args: region)
    monkeypatch.setattr(artifacts, "ArtifactStore", lambda: SimpleNamespace())

    context, actual_region, _chunks = worker._chunk_context(
        task, assembly=False, actual_threads=_resolved(task).values
    )

    assert actual_region is not None
    assert context.fixed is actual_region.fixed
    assert calls == [geometry, geometry, geometry, geometry]


def test_prepared_bundle_is_bound_to_exact_round_and_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import cyexm.workflow.fine_task as worker
    import cyexm.workflow.round_task as round_worker
    from cyexm.registration._request_policy import (
        PREPARED_MASK_POLICY,
        PREPARED_SCALAR_POLICY,
        common_parameters,
    )
    from cyexm.registration.artifacts import (
        ArtifactStore,
        CompletionRecord,
        FileIdentity,
        StageRequest,
    )
    from cyexm.registration.prepared import PreparedManifest, PreparedRoundPaths
    from cyexm.spatial import ImageGeometry
    from cyexm.workflow._round_provenance import RegistrationArtifactPaths

    task = _round_task(tmp_path)
    for path in (task.fixed_image, task.moving_image):
        path.write_bytes(path.name.encode())
    registration = RegistrationArtifactPaths.for_round(
        task.registration_directory, task.round
    )
    registration.affine.parent.mkdir(parents=True, exist_ok=True)
    registration.affine.write_bytes(b"affine")
    prepared = PreparedRoundPaths.for_round(task.registration_directory, task.round)
    prepared.directory.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(prepared.array_paths):
        path.write_bytes(bytes((index,)))
    geometry = ImageGeometry(
        shape=(9, 17, 19),
        origin=(0.0, 0.0, 0.0),
        spacing=(1.0, 1.0, 1.0),
        direction=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    actual_threads = _resolved(RoundPreparationTask.create(task)).values
    fixed = FileIdentity.from_path(task.fixed_image)
    moving = FileIdentity.from_path(task.moving_image)
    affine = FileIdentity.from_path(registration.affine)
    array_identities = tuple(
        FileIdentity.from_path(path) for path in prepared.array_paths
    )
    manifest = PreparedManifest(
        format_version=1,
        fixed_scalar=array_identities[0],
        moving_linear_scalar=array_identities[1],
        fixed_mask=array_identities[2],
        moving_linear_mask=array_identities[3],
        native_fixed=fixed,
        native_moving=moving,
        moving_crop=None,
        affine=affine,
        geometry=geometry,
        scalar_policy=PREPARED_SCALAR_POLICY,
        mask_policy=PREPARED_MASK_POLICY,
        native_threads=dict(sorted(actual_threads.items())),
    )
    prepared.manifest.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    store = ArtifactStore()
    common = common_parameters(
        crop=task.registration.crop,
        imaging_round=task.round,
        pixel_size_xyz=task.voxel_spacing_xyz,
    )
    for role, identity, path in zip(
        (
            "fixed_scalar",
            "moving_linear_scalar",
            "fixed_mask",
            "moving_linear_mask",
        ),
        array_identities,
        prepared.array_paths,
        strict=True,
    ):
        request = StageRequest.for_identities(
            stage=f"prepared:{role}",
            inputs=(fixed, moving),
            geometry=geometry,
            method=task.registration.linear.type,
            accuracy=task.registration.linear.accuracy,
            native_threads=actual_threads,
            upstream=affine,
            parameters={
                **common,
                "format_version": 1,
                "prepared_role": role,
                "scalar_policy": PREPARED_SCALAR_POLICY,
                "mask_policy": PREPARED_MASK_POLICY,
            },
        )
        store.record_path(path).write_text(
            CompletionRecord(1, request, identity).to_json(), encoding="utf-8"
        )
    manifest_request = StageRequest.for_identities(
        stage="prepared:manifest",
        inputs=(fixed, moving, *array_identities),
        geometry=geometry,
        method=task.registration.linear.type,
        accuracy=task.registration.linear.accuracy,
        native_threads=actual_threads,
        upstream=affine,
        parameters={
            "format_version": 1,
            "scalar_policy": PREPARED_SCALAR_POLICY,
            "mask_policy": PREPARED_MASK_POLICY,
        },
    )
    store.record_path(prepared.manifest).write_text(
        CompletionRecord(
            1, manifest_request, FileIdentity.from_path(prepared.manifest)
        ).to_json(),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        round_worker, "_inspect_image_geometry", lambda *_args: geometry
    )

    assert (
        worker._validated_prepared_manifest(task, manifest.to_dict(), actual_threads)
        == manifest
    )
    task.moving_image.write_bytes(b"substituted")
    with pytest.raises(ValueError, match="sources"):
        worker._validated_prepared_manifest(task, manifest.to_dict(), actual_threads)


def test_assembly_adapter_delegates_and_returns_exact_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import cyexm.registration.chunk_stages as stages
    import cyexm.workflow.fine_task as worker

    task = _tasks(tmp_path)[2]
    chunks = compute_chunk_grid((9, 17, 19), (4, 8, 8), 0.25)
    chunk_root = task.round_task.registration_directory / "round_2_chunks"
    for chunk in chunks:
        for suffix in ("warp.nii.gz", "registered_core.npy"):
            path = chunk_root / f"chunk_{chunk.chunk_id:04d}_{suffix}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")
    context = SimpleNamespace(
        warp_path=lambda chunk_id: chunk_root / f"chunk_{chunk_id:04d}_warp.nii.gz",
        scalar_core_path=lambda chunk_id: (
            chunk_root / f"chunk_{chunk_id:04d}_registered_core.npy"
        ),
    )
    assembled = SimpleNamespace(
        warp_path=task.output_paths[0],
        registered_path=task.output_paths[1],
        nonlinear_metrics_path=task.output_paths[2],
        chunk_manifest_path=task.output_paths[3],
    )
    monkeypatch.setattr(
        worker,
        "_chunk_context",
        lambda *_args, **_kwargs: (context, None, chunks),
    )
    monkeypatch.setattr(stages, "assemble_nonlinear_chunks", lambda *_args: assembled)

    assert worker._run_assembly(task, _resolved(task).values).artifact_paths == (
        task.output_paths
    )


def test_whole_image_adapter_delegates_and_returns_exact_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import cyexm.workflow.fine_task as worker

    task = _tasks(tmp_path)[3]

    class Registration:
        def register(self) -> None:
            for path in task.output_paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")

    monkeypatch.setattr(
        worker, "_new_image_registration", lambda *_args: Registration()
    )
    monkeypatch.setattr(worker, "_accepted_registration", lambda _task: object())

    assert worker._run_whole_image(task).artifact_paths == task.output_paths


def test_channel_adapter_delegates_and_returns_exact_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import cyexm.registration.channels as channels
    import cyexm.workflow._round_provenance as provenance
    import cyexm.workflow.fine_task as worker

    task = _tasks(tmp_path)[4]
    monkeypatch.setattr(
        worker,
        "_accepted_registration",
        lambda _task: SimpleNamespace(affine=object(), warp=object()),
    )
    monkeypatch.setattr(
        worker, "_source_geometries", lambda _task: SimpleNamespace(fixed=object())
    )
    monkeypatch.setattr(
        channels, "transform_channel", lambda **_kwargs: task.output_paths[0]
    )
    monkeypatch.setattr(provenance, "channel_artifact_valid", lambda *_args: True)

    assert worker._run_channel(task).artifact_paths == task.output_paths


def test_validation_adapter_requires_and_returns_exact_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import cyexm.workflow.fine_task as worker

    task = _tasks(tmp_path)[5]
    monkeypatch.setattr(
        worker,
        "_accepted_registration",
        lambda _task: SimpleNamespace(
            artifacts={path: object() for path in task.output_paths}
        ),
    )

    assert worker._run_validation(task).artifact_paths == task.output_paths


def test_exact_dispatch_rejects_fine_task_subclasses(tmp_path: Path) -> None:
    class Pretender(WholeImageRegistrationTask):
        pass

    task = WholeImageRegistrationTask.create(
        _round_task(tmp_path, strategy="whole_image")
    )
    pretender = Pretender(
        task.round_task,
        task.key,
        task.dependencies,
        task.output_paths,
        task.log_path,
    )

    with pytest.raises(TypeError, match="exact"):
        run_workflow_task(pretender)  # type: ignore[arg-type]
