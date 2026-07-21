from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from cyexm.registration.artifacts import CompletionRecord, FileIdentity
from cyexm.spatial import ImageGeometry, geometry_compatible
from cyexm.workflow.models import ReferenceTask

type ArtifactSnapshot = tuple[CompletionRecord, FileIdentity]
type SnapshotReader = Callable[[Path], ArtifactSnapshot | None]
type GeometryReader = Callable[[Path, tuple[float, float, float]], ImageGeometry]


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
    crop = (
        None
        if task.crop_bounds_path is None
        else snapshot_reader(task.crop_bounds_path)
    )
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
                if not task.crop and not geometry_compatible(
                    output_geometry, source_geometry
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
