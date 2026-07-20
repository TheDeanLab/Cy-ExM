from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Literal

import pytest

from cyexm.config.models import (
    NativeThreadSpec,
    RegistrationSpec,
    RegistrationStageSpec,
)
from cyexm.registration.artifacts import ArtifactStore, CompletionRecord, FileIdentity
from cyexm.registration.backend import AntsRegistrationBackend
from cyexm.registration.transforms import apply_transform_chain
from cyexm.registration.whole_image import ImageRegistration, RegistrationPaths
from cyexm.runtime import NativeThreadConfig
from cyexm.workflow import round_task as round_task_module
from cyexm.workflow.models import ChannelTask, RoundTask
from cyexm.workflow.round_task import run_round_task

pytestmark = pytest.mark.integration

type _Strategy = Literal["whole_image", "serial_chunked"]

_PIXEL_SIZE = (0.2, 0.2, 0.5)
_SHAPE = (16, 32, 32)


class _FailOnceNonlinearBackend(AntsRegistrationBackend):
    def __init__(self, native_threads: NativeThreadConfig) -> None:
        super().__init__(native_threads)
        self.linear_calls = 0
        self.nonlinear_calls = 0
        self.fail_nonlinear = True

    def linear(self, **kwargs: object) -> tuple[object, object | None, object]:
        self.linear_calls += 1
        return super().linear(**kwargs)

    def nonlinear(self, **kwargs: object) -> tuple[object, Path]:
        self.nonlinear_calls += 1
        if self.fail_nonlinear:
            self.fail_nonlinear = False
            raise RuntimeError("injected nonlinear failure after affine persistence")
        return super().nonlinear(**kwargs)


@pytest.fixture
def round_inputs(
    synthetic_tiffs: tuple[Path, Path],
    native_threads: NativeThreadConfig,
) -> tuple[Path, Path, tuple[Path, ...]]:
    del native_threads
    import numpy as np
    import tifffile

    fixed_path, moving_path = synthetic_tiffs
    moving = np.asarray(tifffile.imread(moving_path), dtype=np.float32)
    assert moving.shape == _SHAPE
    signal_path = moving_path.with_name("round-signal.tif")
    signal = np.flip(moving, axis=2) * np.float32(0.5) + np.float32(127.0)
    tifffile.imwrite(signal_path, signal, metadata={"axes": "ZYX"})
    # The registration channel is deliberately second to prove manifest order survives.
    return fixed_path, moving_path, (signal_path, moving_path)


def _task(
    tmp_path: Path,
    *,
    fixed_path: Path,
    moving_path: Path,
    channel_sources: tuple[Path, ...],
    strategy: _Strategy,
    round_number: int,
) -> RoundTask:
    round_root = tmp_path / "results" / f"round-{round_number:03d}"
    channels = tuple(
        ChannelTask(
            source=source,
            output=round_root / "channels" / source.name,
            interpolation="linear",
        )
        for source in channel_sources
    )
    return RoundTask(
        dataset_id="synthetic-round-task",
        round=round_number,
        strategy=strategy,
        fixed_image=fixed_path,
        moving_image=moving_path,
        registration_directory=round_root / "registration",
        log_path=round_root / "logs" / f"round-{round_number:03d}.log",
        voxel_spacing_xyz=_PIXEL_SIZE,
        registration=RegistrationSpec(
            crop=False,
            linear=RegistrationStageSpec(type="TRSAA", accuracy="dry run"),
            nonlinear=RegistrationStageSpec(type="SyNOnly", accuracy="dry run"),
        ),
        native_threads=NativeThreadSpec(itk=1, blas=1),
        channels=channels,
        chunk_size_zyx=_SHAPE if strategy == "serial_chunked" else None,
        overlap_fraction=0.25 if strategy == "serial_chunked" else None,
    )


def _assert_valid_completion(
    store: ArtifactStore,
    artifact: Path,
    *,
    stage: str,
) -> CompletionRecord:
    assert artifact.is_file()
    record_path = store.record_path(artifact)
    assert record_path.is_file()
    record = CompletionRecord.read(record_path)
    assert record.request.stage == stage
    assert record.artifact == FileIdentity.from_path(artifact)
    return record


def _managed_identities(
    store: ArtifactStore, artifacts: tuple[Path, ...]
) -> tuple[tuple[FileIdentity, FileIdentity], ...]:
    return tuple(
        (
            FileIdentity.from_path(artifact),
            FileIdentity.from_path(store.record_path(artifact)),
        )
        for artifact in artifacts
    )


@pytest.mark.parametrize(
    ("strategy", "round_number"),
    (("whole_image", 2), ("serial_chunked", 3)),
)
def test_round_task_runs_real_strategy_transforms_channels_and_reuses(
    strategy: _Strategy,
    round_number: int,
    round_inputs: tuple[Path, Path, tuple[Path, ...]],
    tmp_path: Path,
    native_threads: NativeThreadConfig,
) -> None:
    import numpy as np
    import tifffile

    fixed_path, moving_path, channel_sources = round_inputs
    task = _task(
        tmp_path,
        fixed_path=fixed_path,
        moving_path=moving_path,
        channel_sources=channel_sources,
        strategy=strategy,
        round_number=round_number,
    )

    first = run_round_task(task)

    assert first.status == "completed", first.failure
    assert first.failure is None
    assert first.strategy == strategy
    paths = RegistrationPaths.for_round(task.registration_directory, task.round)
    expected_artifacts = (
        paths.affine,
        paths.warp,
        paths.registered,
        *(channel.output for channel in task.channels),
    )
    assert first.artifact_paths == tuple(str(path) for path in expected_artifacts)
    assert first.artifact_paths[-len(task.channels) :] == tuple(
        str(channel.output) for channel in task.channels
    )

    store = ArtifactStore()
    _assert_valid_completion(store, paths.affine, stage="affine")
    _assert_valid_completion(store, paths.warp, stage="nonlinear")
    _assert_valid_completion(store, paths.registered, stage="registered-image")
    assert tifffile.imread(paths.registered).shape == _SHAPE

    backend = AntsRegistrationBackend(native_threads)
    fixed = backend.load_image(task.fixed_image, _PIXEL_SIZE)
    for channel in task.channels:
        _assert_valid_completion(
            store,
            channel.output,
            stage="channel-transform-v1",
        )
        written = tifffile.imread(channel.output)
        assert written.shape == _SHAPE
        assert written.dtype == np.dtype(np.uint16)
        direct = apply_transform_chain(
            fixed=fixed,
            moving=backend.load_image(channel.source, _PIXEL_SIZE),
            affine_path=paths.affine,
            warp_path=paths.warp,
            interpolation=channel.interpolation,
            native_threads=native_threads,
        )
        expected = np.clip(np.asarray(backend.to_array(direct)), 0, 65_535).astype(
            np.uint16
        )
        # This established tolerance covers ANTSpyX floating-point variability when
        # independently applying the estimated affine-plus-warp transform chain.
        np.testing.assert_allclose(written, expected, rtol=1e-4, atol=1e-5)

    first_identities = _managed_identities(store, expected_artifacts)
    second = run_round_task(task)

    assert second.status == "completed", second.failure
    assert second.failure is None
    assert second.artifact_paths == first.artifact_paths
    assert second.reused_artifacts == tuple(str(path) for path in expected_artifacts)
    assert _managed_identities(store, expected_artifacts) == first_identities


def test_round_task_reuses_real_affine_after_injected_nonlinear_failure(
    monkeypatch: pytest.MonkeyPatch,
    round_inputs: tuple[Path, Path, tuple[Path, ...]],
    tmp_path: Path,
    native_threads: NativeThreadConfig,
) -> None:
    fixed_path, moving_path, channel_sources = round_inputs
    task = _task(
        tmp_path,
        fixed_path=fixed_path,
        moving_path=moving_path,
        channel_sources=channel_sources,
        strategy="whole_image",
        round_number=4,
    )
    backend = _FailOnceNonlinearBackend(native_threads)
    _whole, chunked, runtime_paths, channel_transform = round_task_module._runtime()
    injected_whole = partial(ImageRegistration, _backend=backend)
    monkeypatch.setattr(
        round_task_module,
        "_runtime",
        lambda: (injected_whole, chunked, runtime_paths, channel_transform),
    )

    failed = run_round_task(task)

    assert failed.status == "failed"
    assert failed.failure is not None
    assert failed.failure.stage == "registration"
    assert failed.failure.message == "nonlinear registration failed for round 4"
    assert "injected nonlinear failure" in task.log_path.read_text(encoding="utf-8")
    paths = RegistrationPaths.for_round(task.registration_directory, task.round)
    store = ArtifactStore()
    _assert_valid_completion(store, paths.affine, stage="affine")
    assert not store.record_path(paths.warp).exists()
    affine_identities = _managed_identities(store, (paths.affine,))
    assert backend.linear_calls == 1
    assert backend.nonlinear_calls == 1

    resumed = run_round_task(task)

    assert resumed.status == "completed", resumed.failure
    assert resumed.failure is None
    assert backend.linear_calls == 1
    assert backend.nonlinear_calls == 2
    assert _managed_identities(store, (paths.affine,)) == affine_identities
    assert str(paths.affine) in resumed.reused_artifacts
    _assert_valid_completion(store, paths.warp, stage="nonlinear")
    for channel in task.channels:
        _assert_valid_completion(
            store,
            channel.output,
            stage="channel-transform-v1",
        )
