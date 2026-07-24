from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from cyexm.chunking import compute_chunk_grid
from cyexm.config.models import (
    NativeThreadSpec,
    NonlinearRegistrationStageSpec,
    RegistrationSpec,
    RegistrationStageSpec,
)
from cyexm.workflow.fine_models import (
    ChannelTransformTask,
    ChunkAssemblyTask,
    FineTaskResult,
    NonlinearChunkTask,
    RoundPreparationTask,
    RoundValidationTask,
    WholeImageRegistrationTask,
)
from cyexm.workflow.fine_planning import (
    channel_tasks,
    chunk_tasks,
    initial_fine_tasks,
)
from cyexm.workflow.models import ChannelTask, RoundFailure, RoundTask

pytestmark = pytest.mark.unit


def _round_task(tmp_path: Path, round_number: int, strategy: str) -> RoundTask:
    root = tmp_path / f"round-{round_number:03d}"
    return RoundTask(
        dataset_id="dataset",
        round=round_number,
        strategy=strategy,  # type: ignore[arg-type]
        fixed_image=tmp_path / "fixed.tif",
        moving_image=tmp_path / f"moving-{round_number}.tif",
        registration_directory=root / "registration",
        log_path=root / "logs" / f"round-{round_number:03d}.log",
        voxel_spacing_xyz=(1.0, 1.0, 1.0),
        registration=RegistrationSpec(
            crop=False,
            linear=RegistrationStageSpec("TRSAA", "high"),
            nonlinear=NonlinearRegistrationStageSpec("SyNOnly", "high", "CC", 4),
        ),
        native_threads=NativeThreadSpec(1, 1),
        channels=(
            ChannelTask(
                source=tmp_path / f"channel-{round_number}-0.tif",
                output=root / "channels" / "0.tif",
                interpolation="linear",
            ),
            ChannelTask(
                source=tmp_path / f"channel-{round_number}-1.tif",
                output=root / "channels" / "1.tif",
                interpolation="nearestNeighbor",
            ),
        ),
        chunk_size_zyx=(4, 8, 8) if strategy == "serial_chunked" else None,
        overlap_fraction=0.25 if strategy == "serial_chunked" else None,
    )


def _prepared_result(task: RoundPreparationTask) -> FineTaskResult:
    identity = {"path": "/tmp/artifact", "st_size": 1, "st_mtime_ns": 2}
    descriptor = {
        "format_version": 2,
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
        "native_threads": {"ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": 1},
        "pixeltypes": {
            "fixed_mask": "float",
            "fixed_scalar": "float",
            "moving_linear_mask": "float",
            "moving_linear_scalar": "float",
        },
    }
    return FineTaskResult(
        task=task,
        status="completed",
        started_at="2026-07-23T00:00:00Z",
        ended_at="2026-07-23T00:00:01Z",
        duration_seconds=1.0,
        hostname="host",
        pid=1,
        dask_task_key=None,
        slurm_job_id=None,
        native_threads=(("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", 1),),
        artifact_paths=tuple(str(path) for path in task.output_paths),
        reused_artifacts=(),
        log_path=str(task.log_path),
        failure=None,
        prepared_descriptor=descriptor,
    )


def test_initial_fine_tasks_are_pure_and_strategy_specific(tmp_path: Path) -> None:
    rounds = (
        _round_task(tmp_path, 3, "whole_image"),
        _round_task(tmp_path, 2, "serial_chunked"),
    )

    first = initial_fine_tasks(rounds)
    second = initial_fine_tasks(rounds)

    assert first == second
    assert [task.round for task in first] == [2, 3]
    assert type(first[0]) is RoundPreparationTask
    assert type(first[1]) is WholeImageRegistrationTask
    assert not tmp_path.joinpath("round-002").exists()


def test_successful_preparation_expands_the_exact_canonical_grid(
    tmp_path: Path,
) -> None:
    round_task = _round_task(tmp_path, 2, "serial_chunked")
    preparation = RoundPreparationTask.create(round_task)
    prepared = _prepared_result(preparation)

    descendants = chunk_tasks(prepared)
    chunks = tuple(task for task in descendants if type(task) is NonlinearChunkTask)
    assembly = descendants[-1]
    expected = compute_chunk_grid((9, 17, 19), (4, 8, 8), 0.25)

    assert tuple(task.chunk for task in chunks) == expected
    assert tuple(task.key.item for task in chunks) == tuple(
        f"{chunk.chunk_id:04d}" for chunk in expected
    )
    assert type(assembly) is ChunkAssemblyTask
    assert assembly.dependencies == tuple(task.key for task in chunks)


@pytest.mark.parametrize("status", ("failed", "cancelled"))
def test_chunk_expansion_requires_successful_preparation(
    tmp_path: Path, status: str
) -> None:
    preparation = RoundPreparationTask.create(
        _round_task(tmp_path, 2, "serial_chunked")
    )
    result = _prepared_result(preparation)
    result = replace(
        result,
        status=status,
        failure=(
            RoundFailure("preparation", "RuntimeError", "injected")
            if status == "failed"
            else None
        ),
        prepared_descriptor=None,
    )

    with pytest.raises(ValueError, match="completed preparation"):
        chunk_tasks(result)


def test_whole_image_round_cannot_expand_chunks(tmp_path: Path) -> None:
    whole = WholeImageRegistrationTask.create(_round_task(tmp_path, 2, "whole_image"))
    result = FineTaskResult(
        task=whole,
        status="completed",
        started_at="2026-07-23T00:00:00Z",
        ended_at="2026-07-23T00:00:01Z",
        duration_seconds=1.0,
        hostname="host",
        pid=1,
        dask_task_key=None,
        slurm_job_id=None,
        native_threads=(("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", 1),),
        artifact_paths=tuple(str(path) for path in whole.output_paths),
        reused_artifacts=(),
        log_path=str(whole.log_path),
        failure=None,
    )

    with pytest.raises(ValueError, match="preparation"):
        chunk_tasks(result)


def test_accepted_transforms_expand_every_channel_then_validation(
    tmp_path: Path,
) -> None:
    whole = WholeImageRegistrationTask.create(_round_task(tmp_path, 2, "whole_image"))
    result = FineTaskResult(
        task=whole,
        status="completed",
        started_at="2026-07-23T00:00:00Z",
        ended_at="2026-07-23T00:00:01Z",
        duration_seconds=1.0,
        hostname="host",
        pid=1,
        dask_task_key=None,
        slurm_job_id=None,
        native_threads=(("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", 1),),
        artifact_paths=tuple(str(path) for path in whole.output_paths),
        reused_artifacts=(),
        log_path=str(whole.log_path),
        failure=None,
    )

    descendants = channel_tasks(result)
    channels = descendants[:-1]
    validation = descendants[-1]

    assert all(type(task) is ChannelTransformTask for task in channels)
    assert tuple(task.channel_index for task in channels) == (0, 1)
    assert all(task.dependencies == (whole.key,) for task in channels)
    assert type(validation) is RoundValidationTask
    assert validation.dependencies == tuple(task.key for task in channels)
