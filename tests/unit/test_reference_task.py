from pathlib import Path

import pytest
from fakes import FakeBackend, fake_image

from cyexm.config.models import NativeThreadSpec
from cyexm.registration.artifacts import ArtifactStore
from cyexm.workflow.models import ReferenceChannelTask, ReferenceTask
from cyexm.workflow.reference_task import run_reference_task

pytestmark = pytest.mark.unit


def _task(tmp_path: Path, *, crop: bool = True, force: bool = False) -> ReferenceTask:
    dataset = tmp_path / "dataset"
    output = tmp_path / "output/rounds/round-001"
    dataset.mkdir()
    sources = (dataset / "fixed.tif", dataset / "signal.tif")
    for source in sources:
        source.write_bytes(b"source")
    return ReferenceTask(
        dataset_id="dataset-1",
        round=1,
        registration_channel_source=sources[0],
        registration_channel_output=output / "channels/fixed.tif",
        registration_directory=output / "registration",
        crop_bounds_path=(
            output / "registration/reference_crop_indices.json" if crop else None
        ),
        log_path=output / "logs/round-001.log",
        voxel_spacing_xyz=(1.0, 1.0, 1.0),
        crop=crop,
        native_threads=NativeThreadSpec(itk=1, blas=1),
        channels=tuple(
            ReferenceChannelTask(source, output / "channels" / source.name)
            for source in sources
        ),
        force=force,
    )


def test_reference_task_crops_every_declared_channel_with_one_bounds_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = _task(tmp_path)
    backend = FakeBackend()
    calls = 0

    def estimate(*args: object, **kwargs: object) -> tuple[int, ...]:
        nonlocal calls
        del args, kwargs
        calls += 1
        return (1, 5, 1, 5, 1, 5)

    monkeypatch.setattr("cyexm.registration.crop.robust_bounding_box", estimate)

    result = run_reference_task(task, _backend=backend)

    assert result.status == "completed"
    assert result.strategy == "reference-preparation"
    assert calls == 1
    assert backend.crop_bounds == [(1, 5, 1, 5, 1, 5)] * 2
    assert tuple(Path(path) for path in result.artifact_paths) == (
        task.crop_bounds_path,
        *(channel.output for channel in task.channels),
    )
    store = ArtifactStore()
    assert store.record_path(task.crop_bounds_path).is_file()  # type: ignore[arg-type]
    assert all(store.record_path(channel.output).is_file() for channel in task.channels)


def test_reference_task_reuses_exact_completed_outputs_without_writing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = _task(tmp_path, crop=False)
    first_backend = FakeBackend()
    first = run_reference_task(task, _backend=first_backend)
    assert first.status == "completed"
    backend = FakeBackend()
    monkeypatch.setattr(
        backend,
        "write_clipped_tiff",
        lambda image, path: pytest.fail(f"unexpected write: {path}"),
    )

    second = run_reference_task(task, _backend=backend)

    assert second.status == "completed"
    assert second.reused_artifacts == tuple(
        str(channel.output) for channel in task.channels
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("origin", (0.1, 0.0, 0.0)),
        ("spacing", (1.0, 1.0, 1.1)),
        ("direction", ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))),
    ],
)
def test_reference_task_rejects_physical_grid_mismatch(
    field: str,
    value: object,
    tmp_path: Path,
) -> None:
    task = _task(tmp_path, crop=False)
    backend = FakeBackend()
    if field == "spacing":
        original_load = backend.load_image

        def load(path: Path, pixel_size_xyz: tuple[float, float, float]) -> object:
            image = original_load(path, pixel_size_xyz)
            if path.name == "signal.tif":
                image.spacing = value  # type: ignore[attr-defined,assignment]
            return image

        backend.load_image = load  # type: ignore[method-assign]
    else:
        setattr(backend.moving_image, field, value)

    result = run_reference_task(task, _backend=backend)

    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.stage == "source-geometry"
    assert "geometries do not match" in result.failure.message
    assert not any(channel.output.exists() for channel in task.channels)


def test_reference_task_rejects_shape_mismatch_before_writing(tmp_path: Path) -> None:
    task = _task(tmp_path, crop=False)
    backend = FakeBackend(moving_image=fake_image(data=None))
    backend.moving_image.data = backend.moving_image.data[:7]

    result = run_reference_task(task, _backend=backend)

    assert result.status == "failed"
    assert not any(channel.output.exists() for channel in task.channels)


def test_reference_task_preserves_independently_committed_channel_on_later_failure(
    tmp_path: Path,
) -> None:
    task = _task(tmp_path, crop=False)
    backend = FakeBackend(fail_tiff_role="signal")

    result = run_reference_task(task, _backend=backend)

    assert result.status == "failed"
    assert result.artifact_paths == (str(task.channels[0].output),)
    assert task.channels[0].output.is_file()
    assert not task.channels[1].output.exists()


def test_reference_task_rejects_output_source_alias(tmp_path: Path) -> None:
    task = _task(tmp_path, crop=False)
    forged = ReferenceTask(
        dataset_id=task.dataset_id,
        round=task.round,
        registration_channel_source=task.registration_channel_source,
        registration_channel_output=task.channels[0].source,
        registration_directory=task.registration_directory,
        crop_bounds_path=None,
        log_path=task.log_path,
        voxel_spacing_xyz=task.voxel_spacing_xyz,
        crop=False,
        native_threads=task.native_threads,
        channels=(
            ReferenceChannelTask(task.channels[0].source, task.channels[0].source),
            task.channels[1],
        ),
    )

    result = run_reference_task(forged, _backend=FakeBackend())

    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.stage == "validation"
