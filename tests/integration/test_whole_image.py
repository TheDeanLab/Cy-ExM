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
    crop: bool = False,
    force_override: bool = False,
) -> ImageRegistration:
    return ImageRegistration(
        fixed_path,
        moving_path,
        output,
        imaging_round=1,
        pixel_size=_PIXEL_SIZE,
        crop=crop,
        enable_logging=False,
        force_override=force_override,
        native_threads=native_threads,
        linear_accuracy="dry run",
        nonlinear_accuracy="dry run",
        _backend=backend,
    )


def test_preparation_boundary_forced_rerun_preserves_public_artifacts_and_requests(
    synthetic_tiffs: tuple[Path, Path],
    tmp_path: Path,
    native_threads: NativeThreadConfig,
) -> None:
    import ants

    fixed_path, moving_path = synthetic_tiffs
    output = tmp_path / "preparation-boundary"
    paths = RegistrationPaths.for_round(output, 1)
    artifacts = (
        paths.fixed_reference,
        paths.pre_metrics,
        paths.affine,
        paths.linear_metrics,
        paths.warp,
        paths.nonlinear_metrics,
        paths.registered,
    )
    store = ArtifactStore()

    def snapshot() -> tuple[
        set[str],
        dict[str, object],
        dict[str, str],
        dict[str, str | None],
    ]:
        records = {
            artifact.name: CompletionRecord.read(store.record_path(artifact))
            for artifact in artifacts
        }
        expected_upstream = {
            paths.fixed_reference.name: None,
            paths.pre_metrics.name: None,
            paths.affine.name: None,
            paths.linear_metrics.name: paths.affine.name,
            paths.warp.name: paths.affine.name,
            paths.nonlinear_metrics.name: paths.warp.name,
            paths.registered.name: paths.warp.name,
        }
        for artifact_name, upstream_name in expected_upstream.items():
            request_upstream = records[artifact_name].request.upstream
            if upstream_name is None:
                assert request_upstream is None
            else:
                assert request_upstream == records[upstream_name].artifact

        backend = AntsRegistrationBackend(native_threads)
        geometries = {
            paths.fixed_reference.name: backend.geometry(
                backend.load_image(paths.fixed_reference, _PIXEL_SIZE)
            ),
            paths.warp.name: backend.geometry(ants.image_read(str(paths.warp))),
            paths.registered.name: backend.geometry(
                backend.load_image(paths.registered, _PIXEL_SIZE)
            ),
        }
        return (
            {path.name for path in output.iterdir()},
            geometries,
            {
                artifact_name: record.request.stage
                for artifact_name, record in records.items()
            },
            expected_upstream,
        )

    _registration(fixed_path, moving_path, output, native_threads).register()
    before = snapshot()

    _registration(
        fixed_path,
        moving_path,
        output,
        native_threads,
        force_override=True,
    ).register()

    assert snapshot() == before


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
        metadata = registered_tiff.shaped_metadata
        assert metadata
        assert metadata[0]["cyexm"]["axis_convention_version"] == (
            "external-zyx-ants-xyz-v1"
        )
    registered_array = tifffile.imread(paths.registered)
    assert registered_array.shape == (16, 32, 32)
    assert registered_array.dtype == np.dtype(np.uint16)

    backend = AntsRegistrationBackend(native_threads)
    registered = backend.load_image(paths.registered, _PIXEL_SIZE)
    assert registered.shape == (32, 32, 16)
    assert backend.geometry(registered).shape == (16, 32, 32)
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


def test_crop_workflow_reloads_fixed_registered_and_warp_with_fixed_geometry(
    synthetic_tiffs: tuple[Path, Path],
    tmp_path: Path,
    native_threads: NativeThreadConfig,
) -> None:
    import json

    import ants
    import numpy as np

    fixed_path, moving_path = synthetic_tiffs
    output = tmp_path / "cropped-whole"

    _registration(
        fixed_path,
        moving_path,
        output,
        native_threads,
        crop=True,
    ).register()

    paths = RegistrationPaths.for_round(output, 1)
    bounds = json.loads(paths.fixed_crop.read_text(encoding="utf-8"))
    z0, z1, y0, y1, x0, x1 = bounds
    expected_shape_zyx = (z1 - z0, y1 - y0, x1 - x0)
    expected_origin_xyz = (
        x0 * _PIXEL_SIZE[0],
        y0 * _PIXEL_SIZE[1],
        z0 * _PIXEL_SIZE[2],
    )
    backend = AntsRegistrationBackend(native_threads)
    fixed_reference = backend.load_image(paths.fixed_reference, _PIXEL_SIZE)
    registered = backend.load_image(paths.registered, _PIXEL_SIZE)
    warp = ants.image_read(str(paths.warp))

    fixed_geometry = backend.geometry(fixed_reference)
    registered_geometry = backend.geometry(registered)
    warp_geometry = backend.geometry(warp)
    assert fixed_geometry.shape == expected_shape_zyx
    assert registered_geometry.shape == expected_shape_zyx
    assert warp_geometry.shape == expected_shape_zyx
    for geometry in (fixed_geometry, registered_geometry, warp_geometry):
        np.testing.assert_allclose(
            geometry.origin,
            expected_origin_xyz,
            rtol=0.0,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            geometry.spacing,
            _PIXEL_SIZE,
            rtol=0.0,
            atol=1e-7,
        )
        np.testing.assert_array_equal(geometry.direction, np.eye(3))


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
