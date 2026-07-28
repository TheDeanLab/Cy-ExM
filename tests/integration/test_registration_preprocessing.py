from __future__ import annotations

from pathlib import Path

import pytest

from cyexm.config import FeatureMaskSpec
from cyexm.registration.artifacts import ArtifactStore, CompletionRecord
from cyexm.registration.backend import AntsRegistrationBackend
from cyexm.registration.whole_image import ImageRegistration, RegistrationPaths
from cyexm.runtime import NativeThreadConfig
from cyexm.spatial import geometry_compatible

pytestmark = pytest.mark.integration

_PIXEL_SIZE = (0.2, 0.2, 0.5)
_SIGMA_ZYX = (2.0, 4.0, 4.0)


class _CountingBackend(AntsRegistrationBackend):
    def __init__(self, native_threads: NativeThreadConfig) -> None:
        super().__init__(native_threads)
        self.linear_calls = 0
        self.nonlinear_calls = 0
        self.registration_mask_counts: list[tuple[int, int, int, int]] = []

    def _record_masks(self, kwargs: dict[str, object]) -> None:
        import numpy as np

        fixed = np.asarray(self.to_array(kwargs["fixed_mask"]))
        moving = np.asarray(self.to_array(kwargs["moving_mask"]))
        self.registration_mask_counts.append(
            (
                int(np.count_nonzero(fixed)),
                int(fixed.size),
                int(np.count_nonzero(moving)),
                int(moving.size),
            )
        )

    def linear(self, **kwargs: object) -> tuple[object, object | None, object]:
        self.linear_calls += 1
        self._record_masks(kwargs)
        return super().linear(**kwargs)

    def nonlinear(self, **kwargs: object) -> tuple[object, Path]:
        self.nonlinear_calls += 1
        self._record_masks(kwargs)
        return super().nonlinear(**kwargs)


def _registration(
    fixed: Path,
    moving: Path,
    output: Path,
    native_threads: NativeThreadConfig,
    backend: AntsRegistrationBackend,
    feature_mask: FeatureMaskSpec | None = None,
) -> ImageRegistration:
    return ImageRegistration(
        fixed,
        moving,
        output,
        imaging_round=1,
        pixel_size=_PIXEL_SIZE,
        enable_logging=False,
        native_threads=native_threads,
        linear_accuracy="dry run",
        nonlinear_accuracy="dry run",
        nonlinear_metric="CC",
        nonlinear_radius=4,
        preprocess_filter_sigma_zyx=_SIGMA_ZYX,
        preprocess_feature_mask=feature_mask,
        _backend=backend,
    )


def test_real_preprocessing_estimates_features_and_persists_raw_fixed_grid(
    synthetic_tiffs: tuple[Path, Path],
    tmp_path: Path,
    native_threads: NativeThreadConfig,
) -> None:
    import numpy as np
    import tifffile

    fixed, moving = synthetic_tiffs
    output = tmp_path / "registration-feature"
    first_backend = _CountingBackend(native_threads)

    _registration(
        fixed,
        moving,
        output,
        native_threads,
        first_backend,
    ).register()

    paths = RegistrationPaths.for_round(output, 1)
    store = ArtifactStore()
    pre_request = CompletionRecord.read(store.record_path(paths.pre_metrics)).request
    affine_request = CompletionRecord.read(store.record_path(paths.affine)).request
    warp_request = CompletionRecord.read(store.record_path(paths.warp)).request
    registered_request = CompletionRecord.read(
        store.record_path(paths.registered)
    ).request

    static = pre_request.parameters["preprocessing"]
    assert static["filter_sigma_zyx"] == list(_SIGMA_ZYX)
    assert "fixed_robust_scale" not in static
    derived = affine_request.parameters["preprocessing"]
    assert derived["fixed_robust_scale"] > 0.0
    assert derived["moving_robust_scale"] > 0.0
    assert warp_request.parameters["preprocessing"] == derived
    assert registered_request.parameters["preprocessing"] == derived
    assert (
        "preprocessing"
        not in CompletionRecord.read(
            store.record_path(paths.fixed_reference)
        ).request.parameters
    )

    registered = tifffile.imread(paths.registered)
    raw_moving = tifffile.imread(moving)
    assert registered.dtype == np.dtype(np.uint16)
    assert registered.shape == raw_moving.shape
    assert int(registered.max()) > 100
    assert int(registered.max()) <= 65_535

    loaded_fixed = first_backend.load_image(fixed, _PIXEL_SIZE)
    loaded_registered = first_backend.load_image(paths.registered, _PIXEL_SIZE)
    assert geometry_compatible(
        first_backend.geometry(loaded_registered),
        first_backend.geometry(loaded_fixed),
    )
    assert first_backend.linear_calls == 1
    assert first_backend.nonlinear_calls == 1

    resumed_backend = _CountingBackend(native_threads)
    _registration(
        fixed,
        moving,
        output,
        native_threads,
        resumed_backend,
    ).register()
    assert resumed_backend.linear_calls == 0
    assert resumed_backend.nonlinear_calls == 0


def test_real_feature_masks_drive_ants_while_output_remains_raw(
    synthetic_tiffs: tuple[Path, Path],
    tmp_path: Path,
    native_threads: NativeThreadConfig,
) -> None:
    import numpy as np
    import tifffile

    fixed, moving = synthetic_tiffs
    output = tmp_path / "feature-mask"
    backend = _CountingBackend(native_threads)
    policy = FeatureMaskSpec(
        min_component_volume_um3=0.02,
        dilation_radius_um=0.2,
    )

    _registration(
        fixed,
        moving,
        output,
        native_threads,
        backend,
        feature_mask=policy,
    ).register()

    assert len(backend.registration_mask_counts) == 2
    for (
        fixed_count,
        fixed_total,
        moving_count,
        moving_total,
    ) in backend.registration_mask_counts:
        assert 0 < fixed_count < fixed_total
        assert 0 < moving_count < moving_total

    paths = RegistrationPaths.for_round(output, 1)
    affine = CompletionRecord.read(ArtifactStore().record_path(paths.affine)).request
    evidence = affine.parameters["preprocessing"]["feature_mask"]
    assert evidence["fixed_foreground_voxels"] > 0
    assert evidence["moving_foreground_voxels"] > 0

    registered = tifffile.imread(paths.registered)
    raw_moving = tifffile.imread(moving)
    assert registered.dtype == np.dtype(np.uint16)
    assert registered.shape == raw_moving.shape
    assert int(registered.max()) > 100
