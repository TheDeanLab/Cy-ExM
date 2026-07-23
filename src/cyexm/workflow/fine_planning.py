"""Pure incremental expansion for fine-grained workflow tasks."""

from __future__ import annotations

from collections.abc import Mapping

from cyexm.chunking import compute_chunk_grid
from cyexm.registration.prepared import PreparedManifest
from cyexm.workflow.fine_models import (
    ChannelTransformTask,
    ChunkAssemblyTask,
    FineTaskResult,
    NonlinearChunkTask,
    RoundPreparationTask,
    RoundValidationTask,
    WholeImageRegistrationTask,
)
from cyexm.workflow.models import RoundTask


def initial_fine_tasks(
    round_tasks: tuple[RoundTask, ...],
) -> tuple[RoundPreparationTask | WholeImageRegistrationTask, ...]:
    """Create only fine tasks knowable before moving-round execution."""
    ordered = sorted(round_tasks, key=lambda task: task.round)
    if any(type(task) is not RoundTask for task in ordered):
        raise TypeError("round_tasks must contain exact RoundTask values")
    if len({task.round for task in ordered}) != len(ordered):
        raise ValueError("round_tasks must use unique round numbers")
    return tuple(
        (
            RoundPreparationTask.create(task)
            if task.strategy == "serial_chunked"
            else WholeImageRegistrationTask.create(task)
        )
        for task in ordered
    )


def chunk_tasks(
    prepared_result: FineTaskResult,
) -> tuple[NonlinearChunkTask | ChunkAssemblyTask, ...]:
    """Expand canonical chunks and assembly from one successful preparation."""
    if type(prepared_result) is not FineTaskResult or (
        type(prepared_result.task) is not RoundPreparationTask
    ):
        raise ValueError("chunk expansion requires a completed preparation result")
    if prepared_result.status not in {"completed", "reused"}:
        raise ValueError("chunk expansion requires a completed preparation result")
    descriptor = prepared_result.to_dict()["prepared_descriptor"]
    if descriptor is None:
        raise ValueError("chunk expansion requires a prepared descriptor")
    if not isinstance(descriptor, Mapping):
        raise TypeError("prepared descriptor must be a JSON object")
    manifest = PreparedManifest.from_dict(descriptor)
    round_task = prepared_result.task.round_task
    if round_task.chunk_size_zyx is None or round_task.overlap_fraction is None:
        raise ValueError("chunked round requires chunk geometry")
    chunks = compute_chunk_grid(
        manifest.geometry.shape,
        round_task.chunk_size_zyx,
        round_task.overlap_fraction,
    )
    chunk_values = tuple(
        NonlinearChunkTask.create(
            round_task,
            chunk,
            prepared_descriptor=descriptor,
            dependencies=(prepared_result.key,),
        )
        for chunk in chunks
    )
    assembly = ChunkAssemblyTask.create(
        round_task,
        prepared_descriptor=descriptor,
        dependencies=tuple(task.key for task in chunk_values),
    )
    return (*chunk_values, assembly)


def channel_tasks(
    registration_result: FineTaskResult,
) -> tuple[ChannelTransformTask | RoundValidationTask, ...]:
    """Expand channels and validation from one accepted registration result."""
    if type(registration_result) is not FineTaskResult or type(
        registration_result.task
    ) not in {ChunkAssemblyTask, WholeImageRegistrationTask}:
        raise ValueError(
            "channel expansion requires a completed assembly or whole-image result"
        )
    if registration_result.status not in {"completed", "reused"}:
        raise ValueError(
            "channel expansion requires a completed assembly or whole-image result"
        )
    round_task = registration_result.task.round_task
    channels = tuple(
        ChannelTransformTask.create(
            round_task,
            index,
            dependencies=(registration_result.key,),
        )
        for index in range(len(round_task.channels))
    )
    validation = RoundValidationTask.create(
        round_task,
        dependencies=tuple(task.key for task in channels),
    )
    return (*channels, validation)


__all__ = ("channel_tasks", "chunk_tasks", "initial_fine_tasks")
