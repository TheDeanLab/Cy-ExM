from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from cyexm.registration._request_policy import (
    reference_channel_parameters,
    reference_crop_parameters,
)
from cyexm.registration.artifacts import CompletionRecord, FileIdentity, StageRequest
from cyexm.runtime import NATIVE_THREAD_ENV_VARS
from cyexm.spatial import ImageGeometry, geometry_compatible
from cyexm.workflow.models import ReferenceTask

type ArtifactSnapshot = tuple[CompletionRecord, FileIdentity]
type SnapshotReader = Callable[[Path], ArtifactSnapshot | None]
type GeometryReader = Callable[[Path, tuple[float, float, float]], ImageGeometry]


def _request_matches(actual: StageRequest, expected: StageRequest) -> bool:
    return geometry_compatible(
        actual.geometry, expected.geometry
    ) and actual == StageRequest(
        stage=expected.stage,
        inputs=expected.inputs,
        geometry=actual.geometry,
        method=expected.method,
        accuracy=expected.accuracy,
        native_threads=expected.native_threads,
        upstream=expected.upstream,
        parameters=expected.parameters,
    )


@dataclass(frozen=True, slots=True)
class ReferenceCompletionEvidence:
    artifacts: tuple[Path, ...]
    crop: ArtifactSnapshot | None
    channels: tuple[ArtifactSnapshot | None, ...]


def reference_completion_evidence(
    task: ReferenceTask,
    source_geometries: tuple[ImageGeometry, ...],
    *,
    snapshot_reader: SnapshotReader,
    geometry_reader: GeometryReader,
) -> ReferenceCompletionEvidence:
    if len(source_geometries) != len(task.channels):
        raise ValueError("reference source geometry count must match channel count")
    identities = tuple(
        FileIdentity.from_path(channel.source) for channel in task.channels
    )
    registration_index = next(
        index
        for index, channel in enumerate(task.channels)
        if channel.source == task.registration_channel_source
    )
    native_threads = {
        name: task.native_threads.itk if index == 0 else task.native_threads.blas
        for index, name in enumerate(NATIVE_THREAD_ENV_VARS)
    }
    crop = None
    if task.crop_bounds_path is not None:
        candidate = snapshot_reader(task.crop_bounds_path)
        expected_crop_request = StageRequest.for_identities(
            stage="reference-crop",
            inputs=(identities[registration_index],),
            geometry=source_geometries[registration_index],
            method="robust-bounding-box",
            accuracy="legacy-robust",
            native_threads=native_threads,
            upstream=None,
            parameters=reference_crop_parameters(),
        )
        if candidate is not None and _request_matches(
            candidate[0].request, expected_crop_request
        ):
            crop = candidate
    channel_snapshots: list[ArtifactSnapshot | None] = []
    artifacts: list[Path] = []
    if crop is not None and task.crop_bounds_path is not None:
        artifacts.append(task.crop_bounds_path)
    for channel, source_geometry in zip(task.channels, source_geometries, strict=True):
        snapshot = snapshot_reader(channel.output)
        if snapshot is not None:
            try:
                output_geometry = geometry_reader(
                    channel.output, task.voxel_spacing_xyz
                )
            except Exception:
                snapshot = None
            else:
                index = len(channel_snapshots)
                expected_request = StageRequest.for_identities(
                    stage="reference-channel",
                    inputs=(identities[index],),
                    geometry=output_geometry,
                    method="uint16-tiff",
                    accuracy="exact",
                    native_threads=native_threads,
                    upstream=None if crop is None else crop[1],
                    parameters=reference_channel_parameters(
                        index=index, cropped=task.crop
                    ),
                )
                if (
                    (task.crop and crop is None)
                    or not _request_matches(snapshot[0].request, expected_request)
                    or (
                        not task.crop
                        and not geometry_compatible(output_geometry, source_geometry)
                    )
                ):
                    snapshot = None
        channel_snapshots.append(snapshot)
        if snapshot is not None:
            artifacts.append(channel.output)
    return ReferenceCompletionEvidence(
        artifacts=tuple(artifacts),
        crop=crop,
        channels=tuple(channel_snapshots),
    )


__all__ = ("ReferenceCompletionEvidence", "reference_completion_evidence")
