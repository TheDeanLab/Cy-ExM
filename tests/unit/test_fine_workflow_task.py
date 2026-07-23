from __future__ import annotations

import os
import subprocess
import sys
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
        dependencies=(chunk.key,),
    )
    whole_round = _round_task(tmp_path, strategy="whole_image")
    whole = WholeImageRegistrationTask.create(whole_round)
    channel = ChannelTransformTask.create(whole_round, 0, dependencies=(whole.key,))
    validation = RoundValidationTask.create(whole_round, dependencies=(channel.key,))
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

    def execute(_task: object) -> object:
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
        lambda _task: (_ for _ in ()).throw(RuntimeError(f"injected {stage}")),
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
