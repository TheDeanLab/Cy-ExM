from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
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
    FineTaskKey,
    FineTaskResult,
    NonlinearChunkTask,
    RoundPreparationTask,
    RoundValidationTask,
    WholeImageRegistrationTask,
)
from cyexm.workflow.models import ChannelTask, RoundTask

pytestmark = pytest.mark.unit


def _round_task(tmp_path: Path, *, strategy: str = "serial_chunked") -> RoundTask:
    round_root = tmp_path / "round-002"
    channels = (
        ChannelTask(
            source=tmp_path / "round_2_488.tif",
            output=round_root / "channels" / "round_2_488.tif",
            interpolation="linear",
        ),
        ChannelTask(
            source=tmp_path / "round_2_647.tif",
            output=round_root / "channels" / "round_2_647.tif",
            interpolation="linear",
        ),
    )
    return RoundTask(
        dataset_id="dataset-1",
        round=2,
        strategy=strategy,  # type: ignore[arg-type]
        fixed_image=tmp_path / "round_1_488.tif",
        moving_image=tmp_path / "round_2_488.tif",
        registration_directory=round_root / "registration",
        log_path=round_root / "logs" / "round-002.log",
        voxel_spacing_xyz=(0.324, 0.324, 0.9),
        registration=RegistrationSpec(
            crop=True,
            linear=RegistrationStageSpec(type="TRSAA", accuracy="high"),
            nonlinear=NonlinearRegistrationStageSpec(
                type="SyNOnly", accuracy="high", metric="CC", radius=4
            ),
        ),
        native_threads=NativeThreadSpec(itk=2, blas=3),
        channels=channels,
        chunk_size_zyx=(4, 8, 8) if strategy == "serial_chunked" else None,
        overlap_fraction=0.25 if strategy == "serial_chunked" else None,
    )


def _descriptor() -> dict[str, object]:
    identity = {"path": "/tmp/artifact", "st_size": 1, "st_mtime_ns": 2}
    return {
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
            "spacing": [0.324, 0.324, 0.9],
            "direction": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        },
        "scalar_policy": "float32-preserve-registration-values-v1",
        "mask_policy": (
            "greater-than-zero-affine-nearest-neighbor-threshold-0.5-float32-v1"
        ),
        "native_threads": {"ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": 2},
        "pixeltypes": {
            "fixed_mask": "float",
            "fixed_scalar": "float",
            "moving_linear_mask": "float",
            "moving_linear_scalar": "float",
        },
    }


def test_fine_task_key_has_stable_encoding() -> None:
    key = FineTaskKey(round=2, stage="chunk", item="0007")

    assert key.encoded == "round-002:chunk:0007"
    assert key.log_name == "round-002-chunk-0007.log"
    assert FineTaskKey(2, "chunk", "10000").item == "10000"
    with pytest.raises(ValueError):
        FineTaskKey(2, "chunk", "00007")


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("round", True, TypeError),
        ("round", -1, ValueError),
        ("stage", "chunks", ValueError),
        ("item", "../0007", ValueError),
    ),
)
def test_fine_task_key_rejects_malformed_values(
    field: str, value: object, error: type[Exception]
) -> None:
    arguments: dict[str, object] = {"round": 2, "stage": "chunk", "item": "0007"}
    arguments[field] = value

    with pytest.raises(error):
        FineTaskKey(**arguments)  # type: ignore[arg-type]


def test_fine_task_models_are_frozen_and_use_unique_logs(tmp_path: Path) -> None:
    round_task = _round_task(tmp_path)
    preparation = RoundPreparationTask.create(round_task)
    chunk = NonlinearChunkTask.create(
        round_task,
        compute_chunk_grid((9, 17, 19), (4, 8, 8), 0.25)[7],
        prepared_descriptor=_descriptor(),
        dependencies=(preparation.key,),
    )

    assert preparation.log_path.name == "round-002-preparation-main.log"
    assert chunk.log_path.name == "round-002-chunk-0007.log"
    assert preparation.log_path != round_task.log_path
    assert chunk.log_path != round_task.log_path
    with pytest.raises(FrozenInstanceError):
        chunk.log_path = round_task.log_path  # type: ignore[misc]


def test_fine_task_models_reject_invalid_round_and_outputs(tmp_path: Path) -> None:
    round_task = _round_task(tmp_path)
    task = RoundPreparationTask.create(round_task)

    with pytest.raises(ValueError, match="embedded round"):
        replace(task, key=FineTaskKey(round=3, stage="preparation", item="main"))
    with pytest.raises(ValueError, match="unique"):
        replace(task, output_paths=(Path("same"), Path("same")))
    with pytest.raises(ValueError, match="canonical task log"):
        replace(task, log_path=tmp_path / ".." / "unsafe.log")


def test_stage_specific_models_reject_invalid_inputs(tmp_path: Path) -> None:
    chunked = _round_task(tmp_path)
    whole = _round_task(tmp_path, strategy="whole_image")
    chunk = compute_chunk_grid((9, 17, 19), (4, 8, 8), 0.25)[0]

    with pytest.raises(ValueError, match="serial-chunked"):
        NonlinearChunkTask.create(
            whole,
            chunk,
            prepared_descriptor=_descriptor(),
            dependencies=(),
        )
    with pytest.raises(ValueError, match="channel_index"):
        ChannelTransformTask.create(chunked, True, dependencies=())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="channel_index"):
        ChannelTransformTask.create(chunked, 9, dependencies=())

    preparation = RoundPreparationTask.create(chunked)
    with pytest.raises(ValueError, match="stage"):
        replace(
            preparation,
            key=FineTaskKey(round=2, stage="assembly", item="main"),
        )


def test_direct_chunk_requires_its_exact_canonical_grid_member(
    tmp_path: Path,
) -> None:
    round_task = _round_task(tmp_path)
    preparation = RoundPreparationTask.create(round_task)
    canonical = compute_chunk_grid((9, 17, 19), (4, 8, 8), 0.25)[0]
    task = NonlinearChunkTask.create(
        round_task,
        canonical,
        prepared_descriptor=_descriptor(),
        dependencies=(preparation.key,),
    )

    with pytest.raises(ValueError, match="canonical prepared-grid member"):
        replace(task, chunk=replace(canonical, x_end_ext=canonical.x_end_ext - 1))
    with pytest.raises(ValueError):
        replace(task, chunk=replace(canonical, chunk_id=1))


def test_direct_chunk_supports_canonical_ids_above_9999(tmp_path: Path) -> None:
    round_task = replace(
        _round_task(tmp_path),
        chunk_size_zyx=(1, 1, 1),
        overlap_fraction=0.0,
    )
    descriptor = json.loads(json.dumps(_descriptor()))
    descriptor["geometry"]["shape"] = [1, 1, 10001]
    chunks = compute_chunk_grid((1, 1, 10001), (1, 1, 1), 0.0)
    task = NonlinearChunkTask.create(
        round_task,
        chunks[10000],
        prepared_descriptor=descriptor,
        dependencies=(RoundPreparationTask.create(round_task).key,),
    )

    assert task.key.item == "10000"
    assert task.chunk.chunk_id == 10000


def test_all_exact_task_types_embed_the_authoritative_round(tmp_path: Path) -> None:
    chunked = _round_task(tmp_path)
    whole = _round_task(tmp_path, strategy="whole_image")
    preparation = RoundPreparationTask.create(chunked)
    descriptor = _descriptor()
    chunk = NonlinearChunkTask.create(
        chunked,
        compute_chunk_grid((9, 17, 19), (4, 8, 8), 0.25)[0],
        prepared_descriptor=descriptor,
        dependencies=(preparation.key,),
    )
    assembly = ChunkAssemblyTask.create(
        chunked,
        prepared_descriptor=descriptor,
        dependencies=tuple(
            FineTaskKey(2, "chunk", f"{item.chunk_id:04d}")
            for item in compute_chunk_grid((9, 17, 19), (4, 8, 8), 0.25)
        ),
    )
    whole_task = WholeImageRegistrationTask.create(whole)
    channel = ChannelTransformTask.create(
        whole,
        0,
        dependencies=(whole_task.key,),
    )
    validation = RoundValidationTask.create(
        whole,
        dependencies=tuple(
            FineTaskKey(2, "channel", f"{index:04d}")
            for index in range(len(whole.channels))
        ),
    )

    for task in (preparation, chunk, assembly):
        assert task.round_task is chunked
    for task in (whole_task, channel, validation):
        assert task.round_task is whole


def test_fine_result_is_strict_and_json_compatible(tmp_path: Path) -> None:
    round_task = _round_task(tmp_path)
    task = RoundPreparationTask.create(round_task)
    result = FineTaskResult(
        task=task,
        status="completed",
        started_at="2026-07-23T00:00:00Z",
        ended_at="2026-07-23T00:00:01Z",
        duration_seconds=1.0,
        hostname="node-1",
        pid=123,
        dask_task_key="worker-key",
        slurm_job_id="42",
        native_threads=(("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", 2),),
        artifact_paths=tuple(str(path) for path in task.output_paths),
        reused_artifacts=(),
        log_path=str(task.log_path),
        failure=None,
        prepared_descriptor=_descriptor(),
    )

    payload = result.to_dict()
    assert payload["key"] == "round-002:preparation:main"
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
    with pytest.raises(TypeError):
        result.prepared_descriptor["geometry"] = {}  # type: ignore[index]
    geometry = result.prepared_descriptor["geometry"]  # type: ignore[index]
    with pytest.raises(TypeError):
        geometry["shape"][0] = 99  # type: ignore[index]
    with pytest.raises(TypeError):
        geometry["direction"][0][0] = 99  # type: ignore[index]
    copied = result.to_dict()["prepared_descriptor"]
    assert isinstance(copied, dict)
    copied["geometry"] = {}  # type: ignore[index]
    assert result.to_dict()["prepared_descriptor"] != copied
    with pytest.raises(ValueError, match="log_path"):
        replace(result, log_path="/tmp/shared.log")
    with pytest.raises(ValueError, match="prepared descriptor"):
        replace(result, prepared_descriptor={"shape": float("nan")})
