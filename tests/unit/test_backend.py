from __future__ import annotations

import inspect
import logging
import os
from pathlib import Path

import numpy as np
import pytest
from fakes import FakeBackend, fake_image, install_fake_ants

import cyexm.registration.backend as backend_module
from cyexm.registration._native_paths import discard_path
from cyexm.registration.backend import AntsRegistrationBackend, RegistrationBackend
from cyexm.runtime import NativeThreadConfig
from cyexm.spatial import ImageGeometry

pytestmark = pytest.mark.unit


def test_registration_backend_declares_every_required_operation() -> None:
    expected = {
        "load_image": ("self", "path", "pixel_size_xyz"),
        "zero_mask": ("self", "image"),
        "threshold_mask": ("self", "image"),
        "geometry": ("self", "image"),
        "to_array": ("self", "image"),
        "from_array": ("self", "array", "reference", "has_components"),
        "from_geometry": (
            "self",
            "array",
            "geometry",
            "has_components",
            "pixeltype",
        ),
        "crop_image": ("self", "image", "bounds_zyx"),
        "create_xy_rotation_seed": (
            "self",
            "geometry",
            "angle_deg",
            "translation_yx_px",
        ),
        "compose_affines": ("self", "transforms"),
        "linear": ("self", "kwargs"),
        "nonlinear": ("self", "kwargs"),
        "apply_affine": (
            "self",
            "fixed",
            "moving",
            "transform",
            "interpolation",
        ),
        "apply_warp": (
            "self",
            "fixed",
            "moving",
            "warp_path",
            "interpolation",
        ),
        "apply_transform_chain": (
            "self",
            "fixed",
            "moving",
            "affine_path",
            "warp_path",
            "interpolation",
        ),
        "histogram_match": ("self", "source", "reference"),
        "write_affine": ("self", "transform", "path"),
        "read_affine": ("self", "path"),
        "write_warp": ("self", "warp", "path"),
        "read_warp": ("self", "path"),
        "write_identity_warp": ("self", "reference", "path"),
        "write_tiff": ("self", "image", "path", "min_value", "max_value"),
        "write_clipped_tiff": ("self", "image", "path"),
        "calculate_metrics": (
            "self",
            "fixed",
            "moving",
            "fixed_mask",
            "moving_mask",
            "sampling",
            "sampling_pct",
        ),
    }

    assert {
        name: tuple(inspect.signature(getattr(RegistrationBackend, name)).parameters)
        for name in expected
    } == expected


def test_ants_backend_satisfies_the_runtime_protocol() -> None:
    assert isinstance(AntsRegistrationBackend(), RegistrationBackend)


def test_load_image_configures_threads_before_delegating_and_logs_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    events: list[str] = []
    requested = NativeThreadConfig(itk_threads=2, blas_threads=3)
    expected = fake_image()
    path = tmp_path / "image.tif"

    class Resolved:
        def describe(self) -> str:
            return "resolved backend threads"

    def configure(config: NativeThreadConfig | None) -> Resolved:
        assert config is requested
        events.append("configure")
        return Resolved()

    def load_image(source: Path, pixel_size_xyz: object) -> object:
        events.append("load")
        assert source == path
        assert pixel_size_xyz == (0.1, 0.2, 0.3)
        return expected

    monkeypatch.setattr(backend_module, "configure_native_threads", configure)
    monkeypatch.setattr(backend_module, "load_ants_image", load_image)
    backend = AntsRegistrationBackend(requested)

    with caplog.at_level(logging.INFO, logger="cyexm.registration.backend"):
        loaded = backend.load_image(path, (0.1, 0.2, 0.3))

    assert events == ["configure", "load"]
    assert loaded is expected
    assert "resolved backend threads" in caplog.text


def test_zero_mask_delegates_after_thread_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = fake_image()
    expected = fake_image(data=np.ones((8, 8, 8), dtype=np.float32))
    events: list[str] = []

    def configure(config: NativeThreadConfig | None) -> object:
        del config
        events.append("configure")
        return _Resolved()

    def make_mask(image: object) -> object:
        events.append("zero-mask")
        assert image is source
        return expected

    monkeypatch.setattr(backend_module, "configure_native_threads", configure)
    monkeypatch.setattr(backend_module, "zero_mask", make_mask)

    result = AntsRegistrationBackend().zero_mask(source)

    assert events == ["configure", "zero-mask"]
    assert result is expected


def test_threshold_mask_uses_the_characterized_half_value_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = install_fake_ants(monkeypatch, "cyexm.registration.backend")
    monkeypatch.setattr(
        backend_module,
        "configure_native_threads",
        lambda config: _Resolved(),
    )
    source = fake_image()

    result = AntsRegistrationBackend().threshold_mask(source)

    threshold_source = calls.threshold.pop("image")
    assert threshold_source is not source
    assert threshold_source.data.dtype == np.float32
    assert np.array_equal(threshold_source.data, source.data)
    assert calls.threshold == {
        "low_thresh": 0.5,
        "high_thresh": 1.0,
        "inval": 1,
        "outval": 0,
    }
    assert result is calls.thresholded


def test_histogram_match_configures_threads_before_runtime_and_delegates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = install_fake_ants(monkeypatch, "cyexm.registration.backend")
    requested = NativeThreadConfig(itk_threads=2, blas_threads=3)

    def configure(config: NativeThreadConfig | None) -> object:
        assert config is requested
        calls.events.append("configure")
        return _Resolved()

    monkeypatch.setattr(backend_module, "configure_native_threads", configure)
    source = fake_image(data=np.full((8, 8, 8), 2.0, dtype=np.float32))
    reference = fake_image(data=np.full((8, 8, 8), 3.0, dtype=np.float32))

    result = AntsRegistrationBackend(requested).histogram_match(
        source=source,
        reference=reference,
    )

    assert calls.events[:2] == ["configure", "runtime"]
    assert calls.histogram_match == {
        "source_image": source,
        "reference_image": reference,
    }
    assert result is calls.histogram_matched


def test_geometry_returns_a_dependency_neutral_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = fake_image()
    expected = ImageGeometry(
        shape=(8, 8, 8),
        origin=(0.0, 0.0, 0.0),
        spacing=(1.0, 1.0, 1.0),
        direction=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    monkeypatch.setattr(backend_module, "ants_geometry", lambda image: expected)

    assert AntsRegistrationBackend().geometry(source) == expected


def test_to_array_returns_an_independent_external_zyx_copy() -> None:
    native_xyz = np.arange(4 * 3 * 2, dtype=np.float32).reshape(4, 3, 2)
    source = fake_image(data=native_xyz)

    array = AntsRegistrationBackend().to_array(source)
    np.testing.assert_array_equal(array, np.transpose(native_xyz, (2, 1, 0)))
    assert array.shape == (2, 3, 4)
    array[0, 0, 0] = -1

    assert source.data[0, 0, 0] != -1


def test_from_array_preserves_reference_geometry_and_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = install_fake_ants(monkeypatch, "cyexm.registration.backend")
    reference = fake_image(
        data=np.ones((5, 3, 2), dtype=np.float32),
        origin=(1.0, 2.0, 3.0),
        spacing=(0.2, 0.3, 0.4),
        direction=((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    )
    array = np.zeros((4, 5, 6, 3), dtype=np.float32)
    array[..., 0] = 11.0
    array[..., 1] = 22.0
    array[..., 2] = 33.0

    result = AntsRegistrationBackend().from_array(
        array,
        reference=reference,
        has_components=True,
    )

    call = calls.from_numpy_calls[-1]
    expected_native = np.transpose(array, (2, 1, 0, 3))
    np.testing.assert_array_equal(call["data"], expected_native)
    np.testing.assert_array_equal(call["data"][..., 0], 11.0)
    np.testing.assert_array_equal(call["data"][..., 1], 22.0)
    np.testing.assert_array_equal(call["data"][..., 2], 33.0)
    assert call["origin"] == reference.origin
    assert call["spacing"] == reference.spacing
    assert call["direction"] == reference.direction
    assert call["has_components"] is True
    assert result.has_components is True


def test_from_geometry_uses_exact_supplied_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = ImageGeometry(
        shape=(2, 3, 4),
        origin=(1.0, 2.0, 3.0),
        spacing=(0.4, 0.5, 0.6),
        direction=((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    )
    source = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

    result = AntsRegistrationBackend().from_geometry(source, geometry=geometry)

    assert AntsRegistrationBackend().geometry(result) == geometry
    assert np.array_equal(AntsRegistrationBackend().to_array(result), source)


def test_crop_image_maps_zyx_start_to_ants_xyz_physical_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = install_fake_ants(monkeypatch, "cyexm.registration.backend")
    external_zyx = np.arange(6 * 7 * 8, dtype=np.float32).reshape(6, 7, 8)
    native_xyz = np.transpose(external_zyx, (2, 1, 0))
    image = fake_image(
        data=native_xyz,
        origin=(10.0, 20.0, 30.0),
        spacing=(0.5, 1.0, 2.0),
        direction=((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    )

    cropped = AntsRegistrationBackend().crop_image(image, (1, 5, 2, 6, 3, 6))

    np.testing.assert_array_equal(
        cropped.numpy(),
        np.transpose(external_zyx[1:5, 2:6, 3:6], (2, 1, 0)),
    )
    assert calls.index_to_physical_calls == []
    assert cropped.origin == (8.0, 21.5, 32.0)
    assert cropped.spacing == image.spacing
    assert cropped.direction == image.direction


def test_crop_image_preserves_vector_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = install_fake_ants(monkeypatch, "cyexm.registration.backend")
    external_zyx = np.zeros((6, 7, 8, 3), dtype=np.float32)
    external_zyx[..., 0] = 11.0
    external_zyx[..., 1] = 22.0
    external_zyx[..., 2] = 33.0
    native_xyz = np.transpose(external_zyx, (2, 1, 0, 3))
    image = fake_image(data=native_xyz, has_components=True)

    cropped = AntsRegistrationBackend().crop_image(image, (1, 5, 2, 6, 3, 6))

    np.testing.assert_array_equal(
        cropped.numpy(),
        np.transpose(external_zyx[1:5, 2:6, 3:6, :], (2, 1, 0, 3)),
    )
    assert calls.from_numpy_calls[-1]["has_components"] is True
    assert cropped.components == 3


def test_linear_delegates_all_arguments_and_native_thread_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested = NativeThreadConfig(itk_threads=1)
    expected = (fake_image(), None, object())
    received: dict[str, object] = {}

    def register_image(**kwargs: object) -> tuple[object, object | None, object]:
        received.update(kwargs)
        return expected

    monkeypatch.setattr(backend_module.linear_kernel, "register_image", register_image)

    result = AntsRegistrationBackend(requested).linear(
        moving_image="moving",
        fixed_image="fixed",
        accuracy="low",
    )

    assert received == {
        "moving_image": "moving",
        "fixed_image": "fixed",
        "accuracy": "low",
        "native_threads": requested,
    }
    assert result is expected


def test_nonlinear_delegates_all_arguments_and_native_thread_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    requested = NativeThreadConfig(blas_threads=1)
    expected = (fake_image(), tmp_path / "warp.nii.gz")
    received: dict[str, object] = {}

    def register_image(**kwargs: object) -> tuple[object, Path]:
        received.update(kwargs)
        return expected

    monkeypatch.setattr(
        backend_module.nonlinear_kernel, "register_image", register_image
    )

    result = AntsRegistrationBackend(requested).nonlinear(
        moving_image="moving",
        fixed_image="fixed",
        accuracy="high",
    )

    assert received == {
        "moving_image": "moving",
        "fixed_image": "fixed",
        "accuracy": "high",
        "native_threads": requested,
    }
    assert result is expected


@pytest.mark.parametrize("operation", ["linear", "nonlinear"])
def test_backend_registration_delegates_configure_once_in_underlying_kernel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    operation: str,
) -> None:
    kernel = (
        backend_module.linear_kernel
        if operation == "linear"
        else backend_module.nonlinear_kernel
    )
    calls = install_fake_ants(monkeypatch, kernel.__name__, tmp_path)
    requested = NativeThreadConfig(itk_threads=1)

    class Resolved:
        def describe(self) -> str:
            calls.events.append("describe")
            return f"resolved {operation} threads"

    def backend_configure(config: NativeThreadConfig | None) -> Resolved:
        assert config is requested
        calls.events.append("backend-configure")
        return Resolved()

    def kernel_configure(config: NativeThreadConfig | None) -> Resolved:
        assert config is requested
        calls.events.append("configure")
        return Resolved()

    monkeypatch.setattr(backend_module, "configure_native_threads", backend_configure)
    monkeypatch.setattr(kernel, "configure_native_threads", kernel_configure)
    backend = AntsRegistrationBackend(requested)

    candidate: Path | None = None
    try:
        with caplog.at_level(logging.INFO):
            if operation == "linear":
                backend.linear(moving_image=fake_image(), fixed_image=fake_image())
            else:
                _, candidate = backend.nonlinear(
                    moving_image=fake_image(), fixed_image=fake_image()
                )

        assert calls.events == ["configure", "describe", "runtime"]
        assert (
            sum(
                record.name == kernel.__name__
                and record.getMessage().startswith("Native thread settings:")
                for record in caplog.records
            )
            == 1
        )
        assert not any(
            record.name == "cyexm.registration.backend" for record in caplog.records
        )
    finally:
        discard_path(candidate)


@pytest.mark.parametrize(
    ("operation", "registration_type", "accuracy", "error"),
    [
        pytest.param(
            "linear", "SyNOnly", "high", "registration type", id="linear-method"
        ),
        pytest.param("linear", "TRSAA", "unknown", "accuracy", id="linear-accuracy"),
        pytest.param(
            "nonlinear", "Rigid", "high", "registration type", id="nonlinear-method"
        ),
        pytest.param(
            "nonlinear", "SyNOnly", "unknown", "accuracy", id="nonlinear-accuracy"
        ),
    ],
)
def test_backend_rejects_invalid_registration_policy_before_configuration(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    registration_type: str,
    accuracy: str,
    error: str,
) -> None:
    events: list[str] = []
    mutation_canary = "CYEXM_TEST_BACKEND_POLICY_CONFIGURED"
    monkeypatch.delenv(mutation_canary, raising=False)
    kernel = (
        backend_module.linear_kernel
        if operation == "linear"
        else backend_module.nonlinear_kernel
    )

    def configure(config: NativeThreadConfig | None) -> _Resolved:
        del config
        events.append("configure")
        monkeypatch.setenv(mutation_canary, "mutated")
        return _Resolved()

    def runtime() -> tuple[object, object]:
        events.append("runtime")
        monkeypatch.setenv(mutation_canary, "mutated")
        raise AssertionError("scientific runtime was accessed")

    monkeypatch.setattr(backend_module, "configure_native_threads", configure)
    monkeypatch.setattr(kernel, "configure_native_threads", configure)
    monkeypatch.setattr(kernel, "_scientific_runtime", runtime)

    with pytest.raises(ValueError, match=error):
        getattr(AntsRegistrationBackend(), operation)(
            moving_image=fake_image(),
            fixed_image=fake_image(),
            registration_type=registration_type,
            accuracy=accuracy,
        )

    assert events == []
    assert mutation_canary not in os.environ


@pytest.mark.parametrize(
    "interpolation",
    [pytest.param("linear", id="scalar"), pytest.param("nearestNeighbor", id="label")],
)
def test_apply_affine_delegates_interpolation(
    monkeypatch: pytest.MonkeyPatch,
    interpolation: str,
) -> None:
    expected = fake_image()
    received: dict[str, object] = {}

    def apply_affine(**kwargs: object) -> object:
        received.update(kwargs)
        return expected

    monkeypatch.setattr(backend_module.transform_helpers, "apply_affine", apply_affine)
    backend = AntsRegistrationBackend()

    result = backend.apply_affine(
        fixed="fixed",
        moving="moving",
        transform="transform",
        interpolation=interpolation,
    )

    assert received["interpolation"] == interpolation
    assert result is expected


def test_apply_warp_delegates_path_and_interpolation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected = fake_image()
    received: dict[str, object] = {}
    warp_path = tmp_path / "warp.nii.gz"

    def apply_warp(**kwargs: object) -> object:
        received.update(kwargs)
        return expected

    monkeypatch.setattr(backend_module.transform_helpers, "apply_warp", apply_warp)

    result = AntsRegistrationBackend().apply_warp(
        fixed="fixed",
        moving="moving",
        warp_path=warp_path,
        interpolation="nearestNeighbor",
    )

    assert received["warp_path"] == warp_path
    assert received["interpolation"] == "nearestNeighbor"
    assert result is expected


def test_apply_transform_chain_prepares_then_delegates_the_established_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    requested = NativeThreadConfig(itk_threads=2, blas_threads=3)
    affine_path = tmp_path / "affine.mat"
    warp_path = tmp_path / "warp.nii.gz"
    expected = fake_image()
    events: list[str] = []
    received: dict[str, object] = {}

    def prepare(config: NativeThreadConfig | None) -> _Resolved:
        assert config is requested
        events.append("prepare")
        return _Resolved()

    def apply_transform_chain(**kwargs: object) -> object:
        events.append("transform")
        received.update(kwargs)
        return expected

    monkeypatch.setattr(backend_module, "configure_native_threads", prepare)
    monkeypatch.setattr(
        backend_module.transform_helpers,
        "apply_transform_chain",
        apply_transform_chain,
    )

    result = AntsRegistrationBackend(requested).apply_transform_chain(
        fixed="fixed",
        moving="moving",
        affine_path=affine_path,
        warp_path=warp_path,
        interpolation="nearestNeighbor",
    )

    assert events == ["prepare", "transform"]
    assert received == {
        "fixed": "fixed",
        "moving": "moving",
        "affine_path": affine_path,
        "warp_path": warp_path,
        "interpolation": "nearestNeighbor",
        "native_threads": requested,
    }
    assert result is expected


def test_fake_backend_records_transform_order_inversion_and_interpolation(
    tmp_path: Path,
) -> None:
    affine_path = tmp_path / "affine.mat"
    affine_path.write_bytes(b"affine")
    warp_path = tmp_path / "warp.nii.gz"
    warp_path.write_bytes(b"warp")
    fixed = fake_image(origin=(10.0, 20.0, 30.0))
    moving = fake_image(data=np.full((8, 8, 8), 12.9, dtype=np.float32))
    backend = FakeBackend()

    result = backend.apply_transform_chain(
        fixed=fixed,
        moving=moving,
        affine_path=affine_path,
        warp_path=warp_path,
        interpolation="linear",
    )

    assert backend.transform_lists == [(warp_path.resolve(), affine_path.resolve())]
    assert backend.transform_inversions == [(False, False)]
    assert backend.transform_interpolations == ["linear"]
    assert result.origin == fixed.origin  # type: ignore[union-attr]
    np.testing.assert_array_equal(result.data, moving.data)  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "operation", ["apply_affine", "apply_warp", "write_affine", "read_affine"]
)
def test_backend_transform_delegates_configure_once_in_underlying_helper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    operation: str,
) -> None:
    calls = install_fake_ants(monkeypatch, "cyexm.registration.transforms", tmp_path)
    requested = NativeThreadConfig(blas_threads=1)

    class Resolved:
        def describe(self) -> str:
            calls.events.append("describe")
            return f"resolved {operation} threads"

    def backend_configure(config: NativeThreadConfig | None) -> Resolved:
        assert config is requested
        calls.events.append("backend-configure")
        return Resolved()

    def helper_configure(config: NativeThreadConfig | None) -> Resolved:
        assert config is requested
        calls.events.append("configure")
        return Resolved()

    monkeypatch.setattr(backend_module, "configure_native_threads", backend_configure)
    monkeypatch.setattr(
        backend_module.transform_helpers,
        "configure_native_threads",
        helper_configure,
    )
    backend = AntsRegistrationBackend(requested)
    path = tmp_path / "delegated.mat"

    with caplog.at_level(logging.INFO):
        if operation == "apply_affine":
            backend.apply_affine(
                fixed=fake_image(),
                moving=fake_image(),
                transform=calls.transform,
                interpolation="linear",
            )
        elif operation == "apply_warp":
            backend.apply_warp(
                fixed=fake_image(),
                moving=fake_image(),
                warp_path=calls.warp_path,
                interpolation="nearestNeighbor",
            )
        elif operation == "write_affine":
            backend.write_affine(calls.transform, path)
        else:
            backend.read_affine(path)

    assert calls.events == ["configure", "describe", "runtime"]
    assert (
        sum(record.name == "cyexm.registration.transforms" for record in caplog.records)
        == 1
    )
    assert not any(
        record.name == "cyexm.registration.backend" for record in caplog.records
    )


def test_affine_readers_and_writers_use_the_supplied_artifact_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    written: list[tuple[object, Path]] = []
    expected = object()
    path = tmp_path / ".linear.mat.atomic.tmp.mat"
    monkeypatch.setattr(
        backend_module.transform_helpers,
        "export_affine_transform",
        lambda transform, destination, **kwargs: written.append(
            (transform, destination)
        ),
    )
    monkeypatch.setattr(
        backend_module.transform_helpers,
        "import_affine_transform",
        lambda source, **kwargs: expected if source == path else None,
    )
    transform = object()
    backend = AntsRegistrationBackend()

    backend.write_affine(transform, path)
    loaded = backend.read_affine(path)

    assert written == [(transform, path)]
    assert loaded is expected


def test_warp_readers_and_writers_use_the_supplied_artifact_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = install_fake_ants(monkeypatch, "cyexm.registration.backend")
    path = tmp_path / ".warp.atomic.tmp.nii.gz"
    warp = fake_image(has_components=True)
    backend = AntsRegistrationBackend()

    backend.write_warp(warp, path)
    loaded = backend.read_warp(path)

    assert calls.image_write_image is warp
    assert calls.image_write_path == str(path)
    assert calls.image_read_path == str(path)
    assert loaded is calls.read_warp


def test_identity_warp_preserves_reference_geometry_and_vector_components(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = install_fake_ants(monkeypatch, "cyexm.registration.backend")
    reference = fake_image(
        data=np.ones((5, 3, 2), dtype=np.float32),
        origin=(1.0, 2.0, 3.0),
        spacing=(0.2, 0.3, 0.4),
        direction=((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    )
    path = tmp_path / "identity.nii.gz"

    AntsRegistrationBackend().write_identity_warp(reference, path)

    creation = calls.from_numpy_calls[-1]
    assert creation["data"].shape == (5, 3, 2, 3)  # type: ignore[union-attr]
    assert np.count_nonzero(creation["data"]) == 0
    assert creation["origin"] == reference.origin
    assert creation["spacing"] == reference.spacing
    assert creation["direction"] == reference.direction
    assert creation["has_components"] is True
    assert calls.image_write_path == str(path)


def test_write_tiff_delegates_range_and_supplied_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    received: dict[str, object] = {}
    destination = tmp_path / ".registered.atomic.tmp.tif"
    image = fake_image()

    def write_tiff(
        value: object,
        path: Path,
        min_value: object,
        max_value: object,
    ) -> None:
        received.update(
            image=value,
            path=path,
            min_value=min_value,
            max_value=max_value,
        )

    monkeypatch.setattr(backend_module, "write_uint16_tiff", write_tiff)

    AntsRegistrationBackend().write_tiff(
        image,
        destination,
        min_value=100.0,
        max_value=1100.0,
    )

    assert received == {
        "image": image,
        "path": destination,
        "min_value": 100.0,
        "max_value": 1100.0,
    }


def test_write_clipped_tiff_prepares_before_delegating_the_supplied_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    requested = NativeThreadConfig(itk_threads=2, blas_threads=3)
    destination = tmp_path / ".channel.atomic.tmp.tif"
    image = fake_image()
    events: list[str] = []
    received: dict[str, object] = {}

    def prepare(config: NativeThreadConfig | None) -> _Resolved:
        assert config is requested
        events.append("prepare")
        return _Resolved()

    def write_clipped(value: object, path: Path) -> None:
        events.append("write")
        received.update(image=value, path=path)

    monkeypatch.setattr(backend_module, "configure_native_threads", prepare)
    monkeypatch.setattr(backend_module, "write_clipped_uint16_tiff", write_clipped)

    AntsRegistrationBackend(requested).write_clipped_tiff(image, destination)

    assert events == ["prepare", "write"]
    assert received == {"image": image, "path": destination}


def test_fake_backend_records_clipped_output_values(tmp_path: Path) -> None:
    image = fake_image(data=np.asarray([[[12.9, 70000.0]]], dtype=np.float32))
    destination = tmp_path / "channel.tif"
    backend = FakeBackend()

    backend.write_clipped_tiff(image, destination)

    assert backend.clipped_tiff_writes == [destination]
    assert len(backend.clipped_tiff_images) == 1
    np.testing.assert_array_equal(backend.clipped_tiff_images[0].data, image.data)
    np.testing.assert_array_equal(
        backend.clipped_tiff_values[0],
        np.asarray([[[12, 65535]]], dtype=np.uint16),
    )
    assert destination.is_file()


def test_metric_calculation_configures_threads_before_delegating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    requested = NativeThreadConfig(itk_threads=2, blas_threads=3)
    expected = {"Correlation": 0.75, "MattesMutualInformation": 0.25}
    received: dict[str, object] = {}

    def configure(config: NativeThreadConfig | None) -> _Resolved:
        assert config is requested
        events.append("configure")
        return _Resolved()

    def calculate(*args: object, **kwargs: object) -> dict[str, float]:
        events.append("calculate")
        received["args"] = args
        received["kwargs"] = kwargs
        return expected

    monkeypatch.setattr(backend_module, "configure_native_threads", configure)
    monkeypatch.setattr(backend_module, "calculate_image_metrics", calculate)

    result = AntsRegistrationBackend(requested).calculate_metrics(
        "fixed",
        "moving",
        fixed_mask="fixed-mask",
        moving_mask="moving-mask",
        sampling="random",
        sampling_pct=0.4,
    )

    assert events == ["configure", "calculate"]
    assert received == {
        "args": ("fixed", "moving"),
        "kwargs": {
            "fixed_mask": "fixed-mask",
            "moving_mask": "moving-mask",
            "sampling": "random",
            "sampling_pct": 0.4,
        },
    }
    assert result is expected


class _Resolved:
    def describe(self) -> str:
        return "test threads"
