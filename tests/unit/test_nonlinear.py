from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import tifffile
from fakes import fake_image, install_fake_ants

import cyexm.registration.nonlinear as nonlinear
from cyexm.registration._native_paths import discard_path
from cyexm.registration.nonlinear import (
    inspect_warp_transform,
    register_image,
    transform_image,
)
from cyexm.runtime import NativeThreadConfig


@pytest.mark.parametrize(
    ("accuracy", "iterations"),
    [
        pytest.param("high", (100, 70, 50), id="high"),
        pytest.param("medium", (50, 35, 25), id="medium"),
        pytest.param("low", (5, 5, 5), id="low"),
        pytest.param("dry run", (1, 1, 1), id="dry-run"),
    ],
)
def test_nonlinear_synonly_uses_established_accuracy_schedules(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    accuracy: str,
    iterations: tuple[int, int, int],
) -> None:
    calls = install_fake_ants(monkeypatch, "cyexm.registration.nonlinear", tmp_path)

    warp_path: Path | None = None
    try:
        transformed, warp_path = register_image(
            moving_image=fake_image(),
            fixed_image=fake_image(),
            registration_type="SyNOnly",
            accuracy=accuracy,
        )

        assert calls.registration["reg_iterations"] == iterations
        assert "shrink_factors" not in calls.registration
        assert "smoothing_sigmas" not in calls.registration
        assert transformed is calls.histogram_matched
        assert warp_path != calls.warp_path
        assert warp_path.read_bytes() == calls.warp_path.read_bytes()
    finally:
        discard_path(warp_path)


def test_nonlinear_high_synonly_preserves_established_registration_parameters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = install_fake_ants(monkeypatch, "cyexm.registration.nonlinear", tmp_path)
    moving = fake_image()
    fixed = fake_image()
    moving_mask = fake_image()
    fixed_mask = fake_image()

    warp_path: Path | None = None
    try:
        _, warp_path = register_image(
            moving_image=moving,
            fixed_image=fixed,
            moving_mask=moving_mask,
            fixed_mask=fixed_mask,
            registration_type="SyNOnly",
            accuracy="high",
            verbose=True,
        )
    finally:
        discard_path(warp_path)

    registration = dict(calls.registration)
    outprefix = registration.pop("outprefix")
    assert isinstance(outprefix, str)
    assert not Path(outprefix).parent.exists()
    assert registration == {
        "fixed": fixed,
        "moving": moving,
        "mask": fixed_mask,
        "moving_mask": moving_mask,
        "type_of_transform": "SyNOnly",
        "flow_sigma": 3.0,
        "total_sigma": 0.5,
        "syn_metric": "mattes",
        "syn_sampling": 32,
        "singleprecision": False,
        "aff_random_sampling_rate": 1.0,
        "verbose": True,
        "initial_transform": "Identity",
        "mask_all_stages": True,
        "reg_iterations": (100, 70, 50),
    }
    assert calls.registration["mask"] is fixed_mask
    assert calls.registration["moving_mask"] is moving_mask
    assert "fixed_mask" not in calls.registration
    assert "number_of_bins" not in calls.registration
    assert "metric_weight" not in calls.registration
    assert "shrink_factors" not in calls.registration
    assert "smoothing_sigmas" not in calls.registration
    assert calls.resample == {
        "image": calls.registered_warped,
        "target": fixed,
        "interp_type": "linear",
    }
    assert calls.histogram_match == {
        "source_image": calls.resampled,
        "reference_image": moving,
    }


@pytest.mark.parametrize("accuracy", ["unknown", "", "HIGHER", "High"])
def test_nonlinear_rejects_unknown_accuracy_before_loading_ants(
    monkeypatch: pytest.MonkeyPatch,
    accuracy: str,
) -> None:
    def unexpected_runtime() -> tuple[object, object]:
        raise AssertionError("ANTs runtime was loaded")

    monkeypatch.setattr(nonlinear, "_scientific_runtime", unexpected_runtime)

    with pytest.raises(ValueError, match="accuracy"):
        register_image(
            moving_image=fake_image(),
            fixed_image=fake_image(),
            registration_type="SyNOnly",
            accuracy=accuracy,
        )


@pytest.mark.parametrize("registration_type", ["Rigid", "synonly", "", "BSpline"])
def test_nonlinear_rejects_unknown_method_before_loading_ants(
    monkeypatch: pytest.MonkeyPatch,
    registration_type: str,
) -> None:
    def unexpected_runtime() -> tuple[object, object]:
        raise AssertionError("ANTs runtime was loaded")

    monkeypatch.setattr(nonlinear, "_scientific_runtime", unexpected_runtime)

    with pytest.raises(ValueError, match="registration type"):
        register_image(
            moving_image=fake_image(),
            fixed_image=fake_image(),
            registration_type=registration_type,
            accuracy="high",
        )


@pytest.mark.parametrize("registration_type", ["Elastic", "SyNOnly", "SyN"])
def test_nonlinear_accepts_each_established_method(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    registration_type: str,
) -> None:
    calls = install_fake_ants(monkeypatch, "cyexm.registration.nonlinear", tmp_path)

    warp_path: Path | None = None
    try:
        _, warp_path = register_image(
            moving_image=fake_image(),
            fixed_image=fake_image(),
            registration_type=registration_type,
            accuracy="low",
        )

        assert calls.registration["type_of_transform"] == registration_type
    finally:
        discard_path(warp_path)


def test_nonlinear_rejects_a_missing_forward_warp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = install_fake_ants(monkeypatch, "cyexm.registration.nonlinear", tmp_path)
    calls.warp_path.unlink()

    with pytest.raises(FileNotFoundError, match="forward warp"):
        register_image(
            moving_image=fake_image(),
            fixed_image=fake_image(),
        )


@pytest.mark.parametrize("registration_type", ["SyN", "Elastic"])
def test_nonlinear_composes_the_complete_forward_transform_list(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    registration_type: str,
) -> None:
    calls = install_fake_ants(monkeypatch, "cyexm.registration.nonlinear", tmp_path)
    affine_path = tmp_path / "forward-affine.mat"
    affine_path.write_bytes(b"affine")
    calls.forward_transforms = [calls.warp_path, affine_path]

    warp_path: Path | None = None
    try:
        transformed, warp_path = register_image(
            moving_image=fake_image(),
            fixed_image=fake_image(),
            registration_type=registration_type,
        )

        assert calls.transformlist == [str(calls.warp_path), str(affine_path)]
        assert calls.whichtoinvert == [False, False]
        assert calls.interpolator == "linear"
        assert calls.compose is not None
        assert calls.composed_path is not None
        assert warp_path != calls.composed_path
        assert warp_path != calls.warp_path
        assert warp_path.is_file()
        assert not calls.composed_path.exists()
        assert transformed is calls.histogram_matched
    finally:
        discard_path(warp_path)


def test_nonlinear_preserves_only_the_returned_forward_warp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = install_fake_ants(monkeypatch, "cyexm.registration.nonlinear", tmp_path)
    affine_path = tmp_path / "forward-affine.mat"
    inverse_path = tmp_path / "inverse-warp.nii.gz"
    affine_path.write_bytes(b"affine")
    inverse_path.write_bytes(b"inverse")
    calls.forward_transforms = [calls.warp_path, affine_path]
    calls.inverse_transforms = [inverse_path, affine_path]

    returned_warp: Path | None = None
    try:
        _, returned_warp = register_image(
            moving_image=fake_image(),
            fixed_image=fake_image(),
            registration_type="SyN",
        )

        assert returned_warp != calls.composed_path
        assert returned_warp.is_file()
        assert calls.composed_path is not None
        assert not calls.composed_path.exists()
        assert calls.warp_path.is_file()
        assert affine_path.is_file()
        assert inverse_path.is_file()
    finally:
        discard_path(returned_warp)


def test_nonlinear_removes_every_transform_when_postprocessing_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = install_fake_ants(monkeypatch, "cyexm.registration.nonlinear", tmp_path)
    inverse_path = tmp_path / "inverse-warp.nii.gz"
    inverse_path.write_bytes(b"inverse")
    calls.inverse_transforms = [inverse_path]
    fake_ants, np_runtime = nonlinear._scientific_runtime()

    def fail_histogram_match(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("injected nonlinear postprocessing failure")

    monkeypatch.setattr(fake_ants, "histogram_match_image", fail_histogram_match)
    monkeypatch.setattr(
        nonlinear,
        "_scientific_runtime",
        lambda: (fake_ants, np_runtime),
    )

    with pytest.raises(RuntimeError, match="injected nonlinear postprocessing failure"):
        register_image(
            moving_image=fake_image(),
            fixed_image=fake_image(),
        )

    assert calls.warp_path.is_file()
    assert inverse_path.is_file()
    assert not Path(str(calls.registration["outprefix"])).parent.exists()


def test_nonlinear_native_failure_removes_owned_workspace_and_preserves_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_fake_ants(monkeypatch, "cyexm.registration.nonlinear", tmp_path)
    fake_ants, _ = nonlinear._scientific_runtime()
    workspaces: list[Path] = []
    partials: list[Path] = []
    received_outprefix: list[object] = []

    def fail_after_partial_outputs(**kwargs: object) -> object:
        outprefix = kwargs.get("outprefix")
        received_outprefix.append(outprefix)
        workspace = (
            Path(str(outprefix)).parent
            if outprefix is not None
            else tmp_path / "unowned-nonlinear-native"
        )
        workspace.mkdir(parents=True, exist_ok=True)
        prefix = Path(str(outprefix)) if outprefix is not None else workspace / "ants-"
        forward = Path(f"{prefix}1Warp.nii.gz")
        inverse = Path(f"{prefix}1InverseWarp.nii.gz")
        forward.write_bytes(b"partial-forward")
        inverse.write_bytes(b"partial-inverse")
        workspaces.append(workspace)
        partials.extend((forward, inverse))
        raise RuntimeError("injected native nonlinear failure")

    monkeypatch.setattr(fake_ants, "registration", fail_after_partial_outputs)

    with pytest.raises(RuntimeError, match="injected native nonlinear failure"):
        register_image(moving_image=fake_image(), fixed_image=fake_image())

    assert received_outprefix[0] is not None
    assert all(not workspace.exists() for workspace in workspaces)
    assert all(not partial.exists() for partial in partials)


def test_nonlinear_composition_failure_removes_workspace_residue(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = install_fake_ants(monkeypatch, "cyexm.registration.nonlinear", tmp_path)
    affine_path = tmp_path / "forward-affine.mat"
    affine_path.write_bytes(b"affine")
    calls.forward_transforms = [calls.warp_path, affine_path]
    fake_ants, _ = nonlinear._scientific_runtime()
    workspaces: list[Path] = []
    residues: list[Path] = []

    def fail_composition(**kwargs: object) -> object:
        compose = kwargs.get("compose")
        assert isinstance(compose, str)
        residue = Path(f"{compose}comptx.nii.gz")
        residue.write_bytes(b"partial-composite")
        workspaces.append(residue.parent)
        residues.append(residue)
        raise RuntimeError("injected composition failure")

    monkeypatch.setattr(fake_ants, "apply_transforms", fail_composition)

    with pytest.raises(RuntimeError, match="injected composition failure"):
        register_image(
            moving_image=fake_image(),
            fixed_image=fake_image(),
            registration_type="SyN",
        )

    assert all(not workspace.exists() for workspace in workspaces)
    assert all(not residue.exists() for residue in residues)


def test_nonlinear_returns_a_unique_validated_copy_not_the_native_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = install_fake_ants(monkeypatch, "cyexm.registration.nonlinear", tmp_path)
    first: Path | None = None
    second: Path | None = None
    try:
        _, first = register_image(moving_image=fake_image(), fixed_image=fake_image())
        _, second = register_image(moving_image=fake_image(), fixed_image=fake_image())

        assert first != second
        assert first != calls.warp_path
        assert second != calls.warp_path
        assert first.is_file()
        assert second.is_file()
        assert calls.warp_path.is_file()
        assert calls.image_read_path == str(second)
    finally:
        discard_path(first)
        discard_path(second)


def test_nonlinear_transform_uses_forward_transform_and_linear_interpolation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = install_fake_ants(monkeypatch, "cyexm.registration.nonlinear", tmp_path)
    moving = fake_image()
    fixed = fake_image()

    result = transform_image(
        moving_image=moving,
        fixed_image=fixed,
        transformed_image={"fwdtransforms": [str(calls.warp_path)]},
    )

    assert calls.transformlist == [str(calls.warp_path)]
    assert calls.interpolator == "linear"
    assert calls.histogram_match == {
        "source_image": calls.applied,
        "reference_image": moving,
    }
    assert result is calls.histogram_matched


def test_nonlinear_transform_keeps_forward_warp_then_affine_uninverted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = install_fake_ants(monkeypatch, "cyexm.registration.nonlinear", tmp_path)
    affine_path = tmp_path / "affine.mat"

    transform_image(
        moving_image=fake_image(),
        fixed_image=fake_image(),
        transformed_image={"fwdtransforms": [str(calls.warp_path), str(affine_path)]},
    )

    assert calls.transformlist == [str(calls.warp_path), str(affine_path)]
    assert calls.whichtoinvert == [False, False]


def test_inspect_warp_transform_summarizes_vector_amplitudes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = install_fake_ants(monkeypatch, "cyexm.registration.nonlinear", tmp_path)
    displacement = np.zeros((8, 8, 8, 3), dtype=np.float32)
    displacement[..., :] = (1.0, 2.0, 2.0)
    calls.read_warp = fake_image(data=displacement, has_components=True)
    image_path = tmp_path / "registered.tif"
    tifffile.imwrite(image_path, np.ones((8, 8, 8), dtype=np.uint16))

    stats = inspect_warp_transform(image_path, calls.warp_path)

    assert stats == {
        "mean": 3.0,
        "min": 3.0,
        "std": 0.0,
        "max": 3.0,
        "n_vox": 512,
    }


def test_nonlinear_configures_threads_before_runtime_and_logs_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls = install_fake_ants(monkeypatch, "cyexm.registration.nonlinear", tmp_path)
    requested = NativeThreadConfig(itk_threads=2, blas_threads=3)

    class Resolved:
        def describe(self) -> str:
            return "resolved nonlinear threads"

    def configure(config: NativeThreadConfig | None) -> Resolved:
        assert config is requested
        calls.events.append("configure")
        return Resolved()

    monkeypatch.setattr(nonlinear, "configure_native_threads", configure)

    warp_path: Path | None = None
    with caplog.at_level(logging.INFO, logger="cyexm.registration.nonlinear"):
        try:
            _, warp_path = register_image(
                moving_image=fake_image(),
                fixed_image=fake_image(),
                native_threads=requested,
            )
        finally:
            discard_path(warp_path)

    assert calls.events[:2] == ["configure", "runtime"]
    assert "resolved nonlinear threads" in caplog.text
