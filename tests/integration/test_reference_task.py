from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from cyexm.config.models import NativeThreadSpec
from cyexm.registration._request_policy import crop_geometry
from cyexm.registration.artifacts import ArtifactStore, CompletionRecord, FileIdentity
from cyexm.registration.backend import AntsRegistrationBackend
from cyexm.runtime import NativeThreadConfig
from cyexm.spatial import geometry_compatible
from cyexm.workflow.models import ReferenceChannelTask, ReferenceTask
from cyexm.workflow.reference_task import run_reference_task

pytestmark = pytest.mark.integration

_SPACING = (0.4, 0.5, 0.9)
_SHAPE = (12, 24, 28)


def _write_sources(
    tmp_path: Path,
    backend: AntsRegistrationBackend,
) -> tuple[tuple[Path, ...], tuple[object, ...]]:
    import ants
    import numpy as np

    direction = np.asarray(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    arrays: list[object] = []
    paths: list[Path] = []
    for index, scale in enumerate((1.0, 2.0, 3.0)):
        array = np.zeros(_SHAPE, dtype=np.float32)
        array[2:10, 4:20, 5:24] = np.arange(8 * 16 * 19, dtype=np.float32).reshape(
            (8, 16, 19)
        ) * np.float32(scale) + np.float32(100 + index)
        image = ants.from_numpy(
            np.transpose(array, (2, 1, 0)),
            origin=(11.0, -7.0, 3.5),
            spacing=_SPACING,
            direction=direction,
        )
        path = tmp_path / f"reference-{index}.tif"
        backend.write_clipped_tiff(image, path)
        arrays.append(array)
        paths.append(path)
    return tuple(paths), tuple(arrays)


def _task(tmp_path: Path, sources: tuple[Path, ...], *, crop: bool) -> ReferenceTask:
    round_root = tmp_path / "results" / "rounds" / "round-001"
    channels = tuple(
        ReferenceChannelTask(source, round_root / "channels" / source.name)
        for source in sources
    )
    return ReferenceTask(
        dataset_id="reference-integration",
        round=1,
        registration_channel_source=sources[1],
        registration_channel_output=channels[1].output,
        registration_directory=round_root / "registration",
        crop_bounds_path=(
            round_root / "registration" / "reference_crop_indices.json"
            if crop
            else None
        ),
        log_path=round_root / "logs" / "round-001.log",
        voxel_spacing_xyz=_SPACING,
        crop=crop,
        native_threads=NativeThreadSpec(itk=1, blas=1),
        channels=channels,
    )


@pytest.mark.parametrize("crop", [False, True])
def test_reference_task_materializes_exact_shared_grid_and_reuses(
    crop: bool,
    tmp_path: Path,
    native_threads: NativeThreadConfig,
) -> None:
    import numpy as np
    import tifffile

    backend = AntsRegistrationBackend(native_threads)
    sources, arrays = _write_sources(tmp_path, backend)
    task = _task(tmp_path, sources, crop=crop)

    first = run_reference_task(task)

    assert first.status == "completed", first.failure
    store = ArtifactStore()
    source_geometry = backend.geometry(backend.load_image(sources[0], _SPACING))
    bounds = (0, _SHAPE[0], 0, _SHAPE[1], 0, _SHAPE[2])
    if task.crop_bounds_path is not None:
        import json

        bounds = tuple(json.loads(task.crop_bounds_path.read_text(encoding="utf-8")))
        crop_record = CompletionRecord.read(store.record_path(task.crop_bounds_path))
        assert crop_record.request.stage == "reference-crop"
        assert crop_record.artifact == FileIdentity.from_path(task.crop_bounds_path)
    expected_geometry = crop_geometry(source_geometry, bounds)
    z0, z1, y0, y1, x0, x1 = bounds
    for index, (channel, source_array) in enumerate(
        zip(task.channels, arrays, strict=True)
    ):
        written = tifffile.imread(channel.output)
        expected = np.clip(source_array[z0:z1, y0:y1, x0:x1], 0, 65_535).astype(
            np.uint16
        )
        np.testing.assert_array_equal(written, expected)
        assert written.dtype == np.dtype(np.uint16)
        with tifffile.TiffFile(channel.output) as opened:
            assert opened.series[0].axes == "ZYX"
        output_geometry = backend.geometry(backend.load_image(channel.output, _SPACING))
        assert geometry_compatible(output_geometry, expected_geometry)
        record = CompletionRecord.read(store.record_path(channel.output))
        assert record.request.stage == "reference-channel"
        assert record.request.parameters["channel_index"] == index
        assert record.request.parameters["cropped"] is crop
        assert record.artifact == FileIdentity.from_path(channel.output)

    managed = tuple(Path(path) for path in first.artifact_paths)
    identities = tuple(
        (FileIdentity.from_path(path), FileIdentity.from_path(store.record_path(path)))
        for path in managed
    )
    second = run_reference_task(task)

    assert second.status == "completed", second.failure
    assert second.reused_artifacts == first.artifact_paths
    assert identities == tuple(
        (FileIdentity.from_path(path), FileIdentity.from_path(store.record_path(path)))
        for path in managed
    )


def test_reference_task_rejects_geometry_mismatch_before_writes(
    tmp_path: Path,
    native_threads: NativeThreadConfig,
) -> None:
    import ants
    import numpy as np

    backend = AntsRegistrationBackend(native_threads)
    sources, _arrays = _write_sources(tmp_path, backend)
    mismatched = tmp_path / "mismatched.tif"
    image = ants.from_numpy(
        np.zeros((27, 24, 12), dtype=np.float32),
        origin=(11.0, -7.0, 3.5),
        spacing=_SPACING,
    )
    backend.write_clipped_tiff(image, mismatched)
    task = _task(tmp_path, sources, crop=False)
    task = replace(
        task,
        channels=(*task.channels[:-1], replace(task.channels[-1], source=mismatched)),
    )

    result = run_reference_task(task)

    assert result.status == "failed"
    assert result.failure is not None
    assert "geometries do not match" in result.failure.message
    assert not any(channel.output.exists() for channel in task.channels)
