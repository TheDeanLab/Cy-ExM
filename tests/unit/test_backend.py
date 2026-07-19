from __future__ import annotations

import inspect
import logging
from pathlib import Path

import numpy as np
import pytest
from fakes import fake_image, install_fake_ants

import cyexm.registration.backend as backend_module
from cyexm.registration.backend import AntsRegistrationBackend, RegistrationBackend
from cyexm.runtime import NativeThreadConfig
from cyexm.spatial import ImageGeometry


def test_registration_backend_declares_every_required_operation() -> None:
    expected = {
        "load_image": ("self", "path", "pixel_size_xyz"),
        "zero_mask": ("self", "image"),
        "geometry": ("self", "image"),
        "to_array": ("self", "image"),
        "from_array": ("self", "array", "reference", "has_components"),
        "crop_image": ("self", "image", "bounds_zyx"),
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
        "write_affine": ("self", "transform", "path"),
        "read_affine": ("self", "path"),
        "write_warp": ("self", "warp", "path"),
        "read_warp": ("self", "path"),
        "write_identity_warp": ("self", "reference", "path"),
        "write_tiff": ("self", "image", "path", "min_value", "max_value"),
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


def test_to_array_returns_an_independent_numpy_copy() -> None:
    source = fake_image()

    array = AntsRegistrationBackend().to_array(source)
    array[0, 0, 0] = -1

    assert source.data[0, 0, 0] != -1


def test_from_array_preserves_reference_geometry_and_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = install_fake_ants(monkeypatch, "cyexm.registration.backend")
    reference = fake_image(
        origin=(1.0, 2.0, 3.0),
        spacing=(0.2, 0.3, 0.4),
        direction=((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    )
    array = np.ones((4, 5, 6, 3), dtype=np.float32)

    result = AntsRegistrationBackend().from_array(
        array,
        reference=reference,
        has_components=True,
    )

    call = calls.from_numpy_calls[-1]
    np.testing.assert_array_equal(call["data"], array)
    assert call["origin"] == reference.origin
    assert call["spacing"] == reference.spacing
    assert call["direction"] == reference.direction
    assert call["has_components"] is True
    assert result.has_components is True


def test_crop_image_applies_zyx_bounds_and_updates_xyz_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_ants(monkeypatch, "cyexm.registration.backend")
    data = np.arange(6 * 7 * 8, dtype=np.float32).reshape(6, 7, 8)
    image = fake_image(
        data=data,
        origin=(10.0, 20.0, 30.0),
        spacing=(0.5, 1.0, 2.0),
        direction=((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    )

    cropped = AntsRegistrationBackend().crop_image(image, (1, 5, 2, 6, 3, 7))

    np.testing.assert_array_equal(cropped.numpy(), data[1:5, 2:6, 3:7])
    assert cropped.origin == (8.0, 21.5, 32.0)
    assert cropped.spacing == image.spacing
    assert cropped.direction == image.direction


def test_crop_image_preserves_vector_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = install_fake_ants(monkeypatch, "cyexm.registration.backend")
    data = np.arange(6 * 7 * 8 * 3, dtype=np.float32).reshape(6, 7, 8, 3)
    image = fake_image(data=data, has_components=True)

    cropped = AntsRegistrationBackend().crop_image(image, (1, 5, 2, 6, 3, 7))

    np.testing.assert_array_equal(cropped.numpy(), data[1:5, 2:6, 3:7, :])
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
        origin=(1.0, 2.0, 3.0),
        spacing=(0.2, 0.3, 0.4),
        direction=((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    )
    path = tmp_path / "identity.nii.gz"

    AntsRegistrationBackend().write_identity_warp(reference, path)

    creation = calls.from_numpy_calls[-1]
    assert creation["data"].shape == (8, 8, 8, 3)  # type: ignore[union-attr]
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


class _Resolved:
    def describe(self) -> str:
        return "test threads"
