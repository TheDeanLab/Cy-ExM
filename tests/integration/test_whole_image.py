from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cyexm.errors import RegistrationStageError
from cyexm.registration.artifacts import ArtifactStore, CompletionRecord
from cyexm.registration.backend import AntsRegistrationBackend
from cyexm.registration.whole_image import ImageRegistration, RegistrationPaths
from cyexm.runtime import NativeThreadConfig

pytestmark = pytest.mark.integration

_PIXEL_SIZE = (0.2, 0.2, 0.5)


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
            raise RuntimeError("injected failure after affine persistence")
        return super().nonlinear(**kwargs)


def _registration(
    fixed_path: Path,
    moving_path: Path,
    output: Path,
    native_threads: NativeThreadConfig,
    *,
    backend: AntsRegistrationBackend | None = None,
) -> ImageRegistration:
    return ImageRegistration(
        fixed_path,
        moving_path,
        output,
        imaging_round=1,
        pixel_size=_PIXEL_SIZE,
        enable_logging=False,
        native_threads=native_threads,
        linear_accuracy="dry run",
        nonlinear_accuracy="dry run",
        _backend=backend,
    )


def test_whole_image_writes_spatially_consistent_output_and_completion_records(
    synthetic_tiffs: tuple[Path, Path],
    tmp_path: Path,
    native_threads: NativeThreadConfig,
) -> None:
    import ants
    import numpy as np
    import tifffile

    fixed_path, moving_path = synthetic_tiffs
    output = tmp_path / "whole"
    root_handlers = tuple(logging.getLogger().handlers)

    _registration(fixed_path, moving_path, output, native_threads).register()

    assert tuple(logging.getLogger().handlers) == root_handlers
    paths = RegistrationPaths.for_round(output, 1)
    stages = {
        paths.fixed_reference: "fixed-reference",
        paths.pre_metrics: "metrics:pre-registration",
        paths.affine: "affine",
        paths.linear_metrics: "metrics:linearly-registered",
        paths.warp: "nonlinear",
        paths.nonlinear_metrics: "metrics:nonlinear-registered",
        paths.registered: "registered-image",
    }
    store = ArtifactStore()
    for artifact, stage in stages.items():
        assert artifact.is_file()
        record_path = store.record_path(artifact)
        assert record_path.is_file()
        record = CompletionRecord.read(record_path)
        assert record.request.stage == stage
        assert record.artifact.path == str(artifact.resolve())

    with tifffile.TiffFile(paths.registered) as registered_tiff:
        series = registered_tiff.series[0]
        assert series.axes == "ZYX"
        assert series.shape == (16, 32, 32)
        assert series.dtype == np.dtype(np.uint16)
    registered_array = tifffile.imread(paths.registered)
    assert registered_array.shape == (16, 32, 32)
    assert registered_array.dtype == np.dtype(np.uint16)

    backend = AntsRegistrationBackend(native_threads)
    registered = backend.load_image(paths.registered, _PIXEL_SIZE)
    assert registered.shape == (16, 32, 32)
    assert registered.origin == (0.0, 0.0, 0.0)
    assert registered.spacing == _PIXEL_SIZE
    np.testing.assert_array_equal(registered.direction, np.eye(3))

    warp = ants.image_read(str(paths.warp))
    assert warp.shape == registered.shape
    assert warp.origin == registered.origin
    # NIfTI serializes spacing as float32; 1e-7 is serialization-only tolerance.
    np.testing.assert_allclose(warp.spacing, _PIXEL_SIZE, rtol=0.0, atol=1e-7)
    np.testing.assert_array_equal(warp.direction, registered.direction)
    assert warp.has_components
    assert warp.numpy().dtype == np.dtype(np.float32)
    assert warp.numpy().shape == (*registered.shape, 3)


def test_resume_reuses_affine_after_injected_nonlinear_failure(
    synthetic_tiffs: tuple[Path, Path],
    tmp_path: Path,
    native_threads: NativeThreadConfig,
) -> None:
    fixed_path, moving_path = synthetic_tiffs
    output = tmp_path / "resume"
    backend = _FailOnceNonlinearBackend(native_threads)
    registration = _registration(
        fixed_path,
        moving_path,
        output,
        native_threads,
        backend=backend,
    )
    paths = RegistrationPaths.for_round(output, 1)
    store = ArtifactStore()

    with pytest.raises(RegistrationStageError, match="nonlinear registration failed"):
        registration.register()

    assert backend.linear_calls == 1
    assert backend.nonlinear_calls == 1
    assert paths.affine.is_file()
    assert store.record_path(paths.affine).is_file()
    assert not store.record_path(paths.warp).exists()

    registration.register()

    assert backend.linear_calls == 1
    assert backend.nonlinear_calls == 2
    assert paths.registered.is_file()
    assert store.record_path(paths.registered).is_file()


def test_registration_import_does_not_configure_root_logging(
    native_threads: NativeThreadConfig,
) -> None:
    del native_threads
    script = """
import logging
before = tuple(logging.getLogger().handlers)
import cyexm.registration
after = tuple(logging.getLogger().handlers)
if after != before:
    raise SystemExit(f"root handlers changed: {before!r} -> {after!r}")
"""
    environment: dict[str, str] = dict(os.environ)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
