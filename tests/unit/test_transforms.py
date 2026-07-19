from __future__ import annotations

import logging
from pathlib import Path

import pytest
from fakes import fake_image, install_fake_ants

import cyexm.registration.transforms as transforms
from cyexm.registration.transforms import (
    apply_affine,
    apply_transform_chain,
    apply_warp,
    export_affine_transform,
    import_affine_transform,
    set_origin_and_spacing,
    transform_image,
)
from cyexm.runtime import NativeThreadConfig


def test_transform_chain_uses_warp_then_affine(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = install_fake_ants(monkeypatch, "cyexm.registration.transforms")

    result = apply_transform_chain(
        fixed=fake_image(),
        moving=fake_image(),
        affine_path=tmp_path / "linear.mat",
        warp_path=tmp_path / "warp.nii.gz",
        interpolation="linear",
    )

    assert calls.transformlist == [
        str(tmp_path / "warp.nii.gz"),
        str(tmp_path / "linear.mat"),
    ]
    assert calls.whichtoinvert == [False, False]
    assert calls.interpolator == "linear"
    assert result is calls.applied


@pytest.mark.parametrize(
    "interpolation",
    [pytest.param("linear", id="scalar"), pytest.param("nearestNeighbor", id="label")],
)
def test_transform_chain_preserves_scalar_and_label_interpolation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    interpolation: str,
) -> None:
    calls = install_fake_ants(monkeypatch, "cyexm.registration.transforms")

    transform_image(
        fixed=fake_image(),
        moving=fake_image(),
        affine_path=tmp_path / "linear.mat",
        warp_path=tmp_path / "warp.nii.gz",
        interpolation=interpolation,
    )

    assert calls.interpolator == interpolation


@pytest.mark.parametrize(
    "interpolation",
    [pytest.param("linear", id="scalar"), pytest.param("nearestNeighbor", id="label")],
)
def test_apply_affine_preserves_requested_interpolation(
    monkeypatch: pytest.MonkeyPatch,
    interpolation: str,
) -> None:
    calls = install_fake_ants(monkeypatch, "cyexm.registration.transforms")

    result = apply_affine(
        fixed=fake_image(),
        moving=fake_image(),
        transform=calls.transform,
        interpolation=interpolation,
    )

    assert calls.transform.applied_interpolations == [interpolation]
    assert result is not calls.transform


@pytest.mark.parametrize(
    "interpolation",
    [pytest.param("linear", id="scalar"), pytest.param("nearestNeighbor", id="label")],
)
def test_apply_warp_uses_forward_warp_without_inversion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    interpolation: str,
) -> None:
    calls = install_fake_ants(monkeypatch, "cyexm.registration.transforms")
    warp_path = tmp_path / "warp.nii.gz"

    result = apply_warp(
        fixed=fake_image(),
        moving=fake_image(),
        warp_path=warp_path,
        interpolation=interpolation,
    )

    assert calls.transformlist == [str(warp_path)]
    assert calls.whichtoinvert == [False]
    assert calls.interpolator == interpolation
    assert result is calls.applied


def test_export_affine_transform_writes_the_supplied_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = install_fake_ants(monkeypatch, "cyexm.registration.transforms")
    destination = tmp_path / ".linear.mat.task.tmp.mat"

    export_affine_transform(calls.transform, destination)

    assert calls.write_transform_value is calls.transform
    assert calls.write_transform_path == str(destination)
    assert destination.read_bytes() == b"transform"


def test_import_affine_transform_reads_the_supplied_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = install_fake_ants(monkeypatch, "cyexm.registration.transforms")
    source = tmp_path / "linear.mat"

    transform = import_affine_transform(source)

    assert calls.read_transform_path == str(source)
    assert transform is calls.transform


def test_set_origin_and_spacing_preserves_unspecified_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_ants(monkeypatch, "cyexm.registration.transforms")
    image = fake_image(origin=(1.0, 2.0, 3.0), spacing=(0.1, 0.2, 0.3))

    result = set_origin_and_spacing(image, origin=(4.0, 5.0, 6.0))

    assert result is image
    assert image.origin == (4.0, 5.0, 6.0)
    assert image.spacing == (0.1, 0.2, 0.3)


def test_transform_chain_configures_threads_before_runtime_and_logs_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls = install_fake_ants(monkeypatch, "cyexm.registration.transforms")
    requested = NativeThreadConfig(itk_threads=2, blas_threads=3)

    class Resolved:
        def describe(self) -> str:
            return "resolved transform threads"

    def configure(config: NativeThreadConfig | None) -> Resolved:
        assert config is requested
        calls.events.append("configure")
        return Resolved()

    monkeypatch.setattr(transforms, "configure_native_threads", configure)

    with caplog.at_level(logging.INFO, logger="cyexm.registration.transforms"):
        apply_transform_chain(
            fixed=fake_image(),
            moving=fake_image(),
            affine_path=tmp_path / "linear.mat",
            warp_path=tmp_path / "warp.nii.gz",
            interpolation="linear",
            native_threads=requested,
        )

    assert calls.events[:2] == ["configure", "runtime"]
    assert "resolved transform threads" in caplog.text
